"""
WebSocket server for real-time trading data streaming.
Broadcasts trading signals, positions, and market data.
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Set, Dict, Any, Optional
import websockets
from websockets.server import WebSocketServerProtocol
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.utils.secure_logging import SecureLogger

# Setup logging  
logger = SecureLogger.setup_logger(__name__, log_file='logs/trading_websocket.log')


class TradingWebSocketServer:
    """WebSocket server for trading data streaming."""
    
    def __init__(self, host: str = "localhost", port: int = 8765):
        """
        Initialize WebSocket server.
        
        Args:
            host: Server host
            port: Server port
        """
        self.host = host
        self.port = port
        self.clients: Set[WebSocketServerProtocol] = set()
        self.trading_bot = None
        self.running = False
        
        # Data buffers
        self.latest_signals: Dict[str, Any] = {}
        self.market_data: Dict[str, Any] = {}
        self.positions: Dict[str, Any] = {}
        self.performance_metrics: Dict[str, Any] = {}
        
        logger.info(f"WebSocket server initialized on {host}:{port}")
    
    def set_trading_bot(self, bot):
        """Set reference to trading bot for data access."""
        self.trading_bot = bot
        logger.info("Trading bot connected to WebSocket server")
    
    async def register_client(self, websocket: WebSocketServerProtocol):
        """Register a new client connection."""
        self.clients.add(websocket)
        logger.info(f"Client connected. Total clients: {len(self.clients)}")
        
        # Send initial data to new client
        await self.send_initial_data(websocket)
    
    async def unregister_client(self, websocket: WebSocketServerProtocol):
        """Unregister a client connection."""
        self.clients.discard(websocket)
        logger.info(f"Client disconnected. Total clients: {len(self.clients)}")
    
    async def send_initial_data(self, websocket: WebSocketServerProtocol):
        """Send initial data to newly connected client."""
        try:
            initial_data = {
                "type": "initial",
                "timestamp": datetime.now().isoformat(),
                "data": {
                    "signals": self.latest_signals,
                    "positions": self.positions,
                    "market_data": self.market_data,
                    "performance": self.performance_metrics
                }
            }
            
            await websocket.send(json.dumps(initial_data))
            logger.debug("Sent initial data to new client")
            
        except Exception as e:
            logger.error(f"Error sending initial data: {e}")
    
    async def broadcast(self, message: Dict[str, Any]):
        """Broadcast message to all connected clients."""
        if not self.clients:
            return
        
        message_str = json.dumps(message)
        
        # Send to all clients concurrently
        disconnected_clients = set()
        
        for client in self.clients:
            try:
                await client.send(message_str)
            except websockets.exceptions.ConnectionClosed:
                disconnected_clients.add(client)
            except Exception as e:
                logger.error(f"Error broadcasting to client: {e}")
                disconnected_clients.add(client)
        
        # Remove disconnected clients
        for client in disconnected_clients:
            await self.unregister_client(client)
    
    async def broadcast_signal(self, signal: Dict[str, Any]):
        """Broadcast trading signal to all clients."""
        message = {
            "type": "signal",
            "timestamp": datetime.now().isoformat(),
            "data": signal
        }
        
        # Store latest signal
        self.latest_signals[signal.get('symbol', 'unknown')] = signal
        
        await self.broadcast(message)
        logger.info(f"Broadcasted signal for {signal.get('symbol')}: {signal.get('action')}")
    
    async def broadcast_position_update(self, position: Dict[str, Any]):
        """Broadcast position update to all clients."""
        message = {
            "type": "position_update",
            "timestamp": datetime.now().isoformat(),
            "data": position
        }
        
        # Update positions
        self.positions[position.get('symbol', 'unknown')] = position
        
        await self.broadcast(message)
        logger.info(f"Broadcasted position update for {position.get('symbol')}")
    
    async def broadcast_market_data(self, market_data: Dict[str, Any]):
        """Broadcast market data update to all clients."""
        message = {
            "type": "market_data",
            "timestamp": datetime.now().isoformat(),
            "data": market_data
        }
        
        # Update market data
        for symbol, data in market_data.items():
            self.market_data[symbol] = data
        
        await self.broadcast(message)
    
    async def broadcast_performance(self, metrics: Dict[str, Any]):
        """Broadcast performance metrics to all clients."""
        message = {
            "type": "performance",
            "timestamp": datetime.now().isoformat(),
            "data": metrics
        }
        
        # Update performance metrics
        self.performance_metrics = metrics
        
        await self.broadcast(message)
        logger.debug(f"Broadcasted performance update: P&L=${metrics.get('total_pnl', 0):.2f}")
    
    async def handle_client(self, websocket: WebSocketServerProtocol, path: str):
        """Handle client connection and messages."""
        await self.register_client(websocket)
        
        try:
            async for message in websocket:
                try:
                    data = json.loads(message)
                    await self.process_client_message(websocket, data)
                except json.JSONDecodeError:
                    logger.error(f"Invalid JSON received: {message}")
                except Exception as e:
                    logger.error(f"Error processing message: {e}")
                    
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            await self.unregister_client(websocket)
    
    async def process_client_message(self, websocket: WebSocketServerProtocol, data: Dict[str, Any]):
        """Process message from client."""
        msg_type = data.get('type')
        
        if msg_type == 'ping':
            # Respond to ping
            await websocket.send(json.dumps({
                "type": "pong",
                "timestamp": datetime.now().isoformat()
            }))
            
        elif msg_type == 'subscribe':
            # Handle subscription requests
            channels = data.get('channels', [])
            logger.info(f"Client subscribed to channels: {channels}")
            
        elif msg_type == 'command':
            # Handle trading commands (start, stop, etc.)
            command = data.get('command')
            await self.handle_trading_command(command, data.get('params', {}))
            
        elif msg_type == 'get_status':
            # Send current bot status
            if self.trading_bot:
                status = self.trading_bot.get_status()
                await websocket.send(json.dumps({
                    "type": "status",
                    "timestamp": datetime.now().isoformat(),
                    "data": status
                }))
        
        else:
            logger.warning(f"Unknown message type: {msg_type}")
    
    async def handle_trading_command(self, command: str, params: Dict[str, Any]):
        """Handle trading commands from clients."""
        if not self.trading_bot:
            logger.warning("No trading bot connected")
            return
        
        logger.info(f"Received command: {command} with params: {params}")
        
        if command == "start":
            # Start trading bot
            asyncio.create_task(self.trading_bot.start())
            
        elif command == "stop":
            # Stop trading bot
            await self.trading_bot.stop()
            
        elif command == "close_position":
            # Close specific position
            symbol = params.get('symbol')
            if symbol:
                await self.trading_bot._close_position(symbol, "manual")
                
        elif command == "update_config":
            # Update trading configuration
            config_updates = params.get('config', {})
            # This would update bot configuration
            logger.info(f"Config update requested: {config_updates}")
            
        else:
            logger.warning(f"Unknown command: {command}")
    
    async def data_update_loop(self):
        """Periodic data update loop."""
        while self.running:
            try:
                if self.trading_bot and self.trading_bot.is_running:
                    # Get latest data from bot
                    status = self.trading_bot.get_status()
                    
                    # Broadcast market data
                    if status.get('market_data'):
                        await self.broadcast_market_data(status['market_data'])
                    
                    # Broadcast performance
                    performance = {
                        'balance': status.get('balance', 0),
                        'total_pnl': status.get('total_pnl', 0),
                        'total_trades': status.get('total_trades', 0),
                        'win_rate': status.get('win_rate', 0),
                        'open_positions': status.get('open_positions', 0)
                    }
                    await self.broadcast_performance(performance)
                    
                    # Check for new signals
                    if hasattr(self.trading_bot, 'signal_queue'):
                        while not self.trading_bot.signal_queue.empty():
                            signal = self.trading_bot.signal_queue.get()
                            await self.broadcast_signal(signal)
                
                # Sleep before next update
                await asyncio.sleep(1)  # Update every second
                
            except Exception as e:
                logger.error(f"Error in data update loop: {e}")
                await asyncio.sleep(5)
    
    async def start(self):
        """Start the WebSocket server."""
        self.running = True
        
        # Start data update loop
        update_task = asyncio.create_task(self.data_update_loop())
        
        # Start WebSocket server
        logger.info(f"Starting WebSocket server on {self.host}:{self.port}")
        
        async with websockets.serve(self.handle_client, self.host, self.port):
            logger.info(f"WebSocket server running on ws://{self.host}:{self.port}")
            
            try:
                await asyncio.Future()  # Run forever
            except KeyboardInterrupt:
                logger.info("WebSocket server stopped by user")
            finally:
                self.running = False
                update_task.cancel()


class TradingWebSocketClient:
    """WebSocket client for testing and dashboard integration."""
    
    def __init__(self, url: str = "ws://localhost:8765"):
        """
        Initialize WebSocket client.
        
        Args:
            url: WebSocket server URL
        """
        self.url = url
        self.websocket = None
        self.running = False
        
    async def connect(self):
        """Connect to WebSocket server."""
        try:
            self.websocket = await websockets.connect(self.url)
            self.running = True
            logger.info(f"Connected to WebSocket server at {self.url}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect: {e}")
            return False
    
    async def disconnect(self):
        """Disconnect from WebSocket server."""
        self.running = False
        if self.websocket:
            await self.websocket.close()
            logger.info("Disconnected from WebSocket server")
    
    async def send_message(self, message: Dict[str, Any]):
        """Send message to server."""
        if self.websocket:
            await self.websocket.send(json.dumps(message))
    
    async def receive_messages(self):
        """Receive messages from server."""
        if not self.websocket:
            return
        
        try:
            async for message in self.websocket:
                data = json.loads(message)
                yield data
        except websockets.exceptions.ConnectionClosed:
            logger.warning("Connection closed by server")
            self.running = False
        except Exception as e:
            logger.error(f"Error receiving message: {e}")
    
    async def subscribe(self, channels: list):
        """Subscribe to specific data channels."""
        await self.send_message({
            "type": "subscribe",
            "channels": channels
        })
    
    async def send_command(self, command: str, params: Dict[str, Any] = None):
        """Send trading command to server."""
        await self.send_message({
            "type": "command",
            "command": command,
            "params": params or {}
        })
    
    async def get_status(self):
        """Request current bot status."""
        await self.send_message({
            "type": "get_status"
        })


async def main():
    """Main function to run WebSocket server."""
    server = TradingWebSocketServer(host="0.0.0.0", port=8765)
    
    # Optionally connect to trading bot
    # from src.trading.live_trader import LiveTradingBot
    # bot = LiveTradingBot(...)
    # server.set_trading_bot(bot)
    
    await server.start()


if __name__ == "__main__":
    asyncio.run(main())