"""
Market data WebSocket endpoints for real-time streaming.
Handles live price feeds, order book, and trade data.
"""

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException
from typing import Dict, List, Optional, Set
import asyncio
import json
from datetime import datetime
import pandas as pd
import numpy as np
import structlog
from collections import defaultdict
import ccxt.async_support as ccxt

log = structlog.get_logger()

router = APIRouter()

class MarketDataManager:
    """Manages market data subscriptions and broadcasting."""
    
    def __init__(self):
        self.clients: Dict[str, Set[WebSocket]] = defaultdict(set)
        self.exchange = None
        self.market_data_task = None
        self.order_book_cache = {}
        self.price_cache = {}
        self.running = False
        
    async def initialize_exchange(self):
        """Initialize exchange connection."""
        try:
            self.exchange = ccxt.binance({
                'enableRateLimit': True,
                'rateLimit': 1200,
            })
            await self.exchange.load_markets()
            log.info("Exchange initialized successfully")
        except Exception as e:
            log.error(f"Failed to initialize exchange: {e}")
            self.exchange = None
    
    async def subscribe(self, channel: str, websocket: WebSocket):
        """Subscribe a client to a channel."""
        self.clients[channel].add(websocket)
        log.info(f"Client subscribed to {channel}")
        
        # Send cached data if available
        if channel == "prices" and self.price_cache:
            await websocket.send_json({
                "channel": "prices",
                "data": self.price_cache,
                "timestamp": datetime.now().isoformat()
            })
        elif channel == "orderbook" and self.order_book_cache:
            await websocket.send_json({
                "channel": "orderbook",
                "data": self.order_book_cache,
                "timestamp": datetime.now().isoformat()
            })
    
    async def unsubscribe(self, channel: str, websocket: WebSocket):
        """Unsubscribe a client from a channel."""
        if websocket in self.clients[channel]:
            self.clients[channel].remove(websocket)
            log.info(f"Client unsubscribed from {channel}")
    
    async def unsubscribe_all(self, websocket: WebSocket):
        """Unsubscribe a client from all channels."""
        for channel in self.clients:
            if websocket in self.clients[channel]:
                self.clients[channel].remove(websocket)
    
    async def broadcast(self, channel: str, data: Dict):
        """Broadcast data to all clients in a channel."""
        if channel not in self.clients:
            return
        
        disconnected = set()
        
        for websocket in self.clients[channel]:
            try:
                await websocket.send_json({
                    "channel": channel,
                    "data": data,
                    "timestamp": datetime.now().isoformat()
                })
            except Exception as e:
                log.warning(f"Failed to send to client: {e}")
                disconnected.add(websocket)
        
        # Remove disconnected clients
        for ws in disconnected:
            self.clients[channel].discard(ws)
    
    async def start_market_data_stream(self):
        """Start streaming market data."""
        if self.running:
            return
        
        self.running = True
        
        if not self.exchange:
            await self.initialize_exchange()
        
        while self.running:
            try:
                # Fetch and broadcast price data
                await self.fetch_and_broadcast_prices()
                
                # Fetch and broadcast order book
                await self.fetch_and_broadcast_orderbook()
                
                # Fetch and broadcast trades
                await self.fetch_and_broadcast_trades()
                
                await asyncio.sleep(1)  # Update every second
                
            except Exception as e:
                log.error(f"Market data stream error: {e}")
                await asyncio.sleep(5)  # Wait before retry
    
    async def fetch_and_broadcast_prices(self):
        """Fetch and broadcast price data."""
        if not self.exchange:
            # Use mock data if exchange not available
            mock_price = 50000 + np.random.randn() * 100
            self.price_cache = {
                "symbol": "BTC/USDT",
                "price": mock_price,
                "change_24h": np.random.uniform(-5, 5),
                "volume_24h": np.random.uniform(1e9, 2e9),
                "high_24h": mock_price + 500,
                "low_24h": mock_price - 500
            }
        else:
            try:
                ticker = await self.exchange.fetch_ticker('BTC/USDT')
                self.price_cache = {
                    "symbol": ticker['symbol'],
                    "price": ticker['last'],
                    "change_24h": ticker['percentage'],
                    "volume_24h": ticker['quoteVolume'],
                    "high_24h": ticker['high'],
                    "low_24h": ticker['low']
                }
            except Exception as e:
                log.error(f"Failed to fetch prices: {e}")
        
        await self.broadcast("prices", self.price_cache)
    
    async def fetch_and_broadcast_orderbook(self):
        """Fetch and broadcast order book data."""
        if not self.exchange:
            # Use mock data
            base_price = 50000
            self.order_book_cache = {
                "bids": [[base_price - i*10, np.random.uniform(0.1, 1)] for i in range(1, 21)],
                "asks": [[base_price + i*10, np.random.uniform(0.1, 1)] for i in range(1, 21)]
            }
        else:
            try:
                orderbook = await self.exchange.fetch_order_book('BTC/USDT', limit=20)
                self.order_book_cache = {
                    "bids": orderbook['bids'][:20],
                    "asks": orderbook['asks'][:20]
                }
            except Exception as e:
                log.error(f"Failed to fetch orderbook: {e}")
        
        await self.broadcast("orderbook", self.order_book_cache)
    
    async def fetch_and_broadcast_trades(self):
        """Fetch and broadcast recent trades."""
        trades_data = []
        
        if not self.exchange:
            # Use mock data
            for _ in range(10):
                trades_data.append({
                    "timestamp": datetime.now().isoformat(),
                    "price": 50000 + np.random.randn() * 50,
                    "amount": np.random.uniform(0.01, 1),
                    "side": np.random.choice(["buy", "sell"])
                })
        else:
            try:
                trades = await self.exchange.fetch_trades('BTC/USDT', limit=10)
                trades_data = [
                    {
                        "timestamp": trade['datetime'],
                        "price": trade['price'],
                        "amount": trade['amount'],
                        "side": trade['side']
                    }
                    for trade in trades
                ]
            except Exception as e:
                log.error(f"Failed to fetch trades: {e}")
        
        await self.broadcast("trades", trades_data)
    
    async def stop_market_data_stream(self):
        """Stop streaming market data."""
        self.running = False
        if self.exchange:
            await self.exchange.close()


# Global market data manager
market_manager = MarketDataManager()


@router.websocket("/ws/market")
async def websocket_market_data(websocket: WebSocket):
    """WebSocket endpoint for market data streaming."""
    await websocket.accept()
    client_id = f"market_{id(websocket)}"
    
    log.info(f"Market data WebSocket connected: {client_id}")
    
    # Start market data stream if not running
    if not market_manager.running:
        asyncio.create_task(market_manager.start_market_data_stream())
    
    try:
        while True:
            # Receive subscription requests
            data = await websocket.receive_json()
            
            action = data.get("action")
            channel = data.get("channel")
            
            if action == "subscribe":
                if channel in ["prices", "orderbook", "trades", "all"]:
                    if channel == "all":
                        # Subscribe to all channels
                        for ch in ["prices", "orderbook", "trades"]:
                            await market_manager.subscribe(ch, websocket)
                    else:
                        await market_manager.subscribe(channel, websocket)
                    
                    await websocket.send_json({
                        "status": "subscribed",
                        "channel": channel,
                        "timestamp": datetime.now().isoformat()
                    })
                else:
                    await websocket.send_json({
                        "error": f"Invalid channel: {channel}",
                        "timestamp": datetime.now().isoformat()
                    })
            
            elif action == "unsubscribe":
                if channel == "all":
                    await market_manager.unsubscribe_all(websocket)
                else:
                    await market_manager.unsubscribe(channel, websocket)
                
                await websocket.send_json({
                    "status": "unsubscribed",
                    "channel": channel,
                    "timestamp": datetime.now().isoformat()
                })
            
            elif action == "ping":
                await websocket.send_json({
                    "action": "pong",
                    "timestamp": datetime.now().isoformat()
                })
    
    except WebSocketDisconnect:
        log.info(f"Market data WebSocket disconnected: {client_id}")
        await market_manager.unsubscribe_all(websocket)
    except Exception as e:
        log.error(f"Market data WebSocket error: {e}")
        await market_manager.unsubscribe_all(websocket)
    finally:
        try:
            await websocket.close()
        except:
            pass


@router.websocket("/ws/positions")
async def websocket_positions(websocket: WebSocket):
    """WebSocket endpoint for position updates."""
    await websocket.accept()
    client_id = f"positions_{id(websocket)}"
    
    log.info(f"Positions WebSocket connected: {client_id}")
    
    try:
        # Validate authentication
        data = await websocket.receive_json()
        token = data.get("token")
        
        # Simple token validation (implement proper auth in production)
        if not token:
            await websocket.send_json({"error": "Authentication required"})
            await websocket.close()
            return
        
        # Send initial positions
        positions = generate_mock_positions()
        await websocket.send_json({
            "channel": "positions",
            "data": positions,
            "timestamp": datetime.now().isoformat()
        })
        
        # Stream position updates
        while True:
            await asyncio.sleep(5)  # Update every 5 seconds
            
            # Update positions with random changes
            for position in positions:
                position['current_price'] *= np.random.uniform(0.99, 1.01)
                position['pnl'] = (position['current_price'] - position['entry_price']) * position['quantity']
                position['pnl_pct'] = ((position['current_price'] / position['entry_price']) - 1) * 100
            
            await websocket.send_json({
                "channel": "positions",
                "data": positions,
                "timestamp": datetime.now().isoformat()
            })
    
    except WebSocketDisconnect:
        log.info(f"Positions WebSocket disconnected: {client_id}")
    except Exception as e:
        log.error(f"Positions WebSocket error: {e}")
    finally:
        try:
            await websocket.close()
        except:
            pass


def generate_mock_positions() -> List[Dict]:
    """Generate mock position data."""
    symbols = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT']
    positions = []
    
    for symbol in symbols:
        if np.random.random() > 0.5:  # 50% chance of having position
            entry_price = {
                'BTC/USDT': 50000,
                'ETH/USDT': 3000,
                'BNB/USDT': 300
            }[symbol] * np.random.uniform(0.95, 1.05)
            
            current_price = entry_price * np.random.uniform(0.95, 1.05)
            quantity = np.random.uniform(0.01, 1)
            
            positions.append({
                'symbol': symbol,
                'side': np.random.choice(['LONG', 'SHORT']),
                'entry_price': entry_price,
                'current_price': current_price,
                'quantity': quantity,
                'value': current_price * quantity,
                'pnl': (current_price - entry_price) * quantity,
                'pnl_pct': ((current_price / entry_price) - 1) * 100,
                'entry_time': datetime.now().isoformat()
            })
    
    return positions


@router.websocket("/ws/alerts")
async def websocket_alerts(websocket: WebSocket):
    """WebSocket endpoint for alert notifications."""
    await websocket.accept()
    client_id = f"alerts_{id(websocket)}"
    
    log.info(f"Alerts WebSocket connected: {client_id}")
    
    try:
        # Generate and send alerts periodically
        alert_types = ['signal', 'risk', 'trade', 'system']
        severities = ['info', 'warning', 'error', 'success']
        
        while True:
            await asyncio.sleep(np.random.uniform(10, 30))  # Random interval
            
            alert = {
                'type': np.random.choice(alert_types),
                'severity': np.random.choice(severities),
                'message': generate_alert_message(),
                'timestamp': datetime.now().isoformat()
            }
            
            await websocket.send_json({
                "channel": "alerts",
                "data": alert,
                "timestamp": datetime.now().isoformat()
            })
    
    except WebSocketDisconnect:
        log.info(f"Alerts WebSocket disconnected: {client_id}")
    except Exception as e:
        log.error(f"Alerts WebSocket error: {e}")
    finally:
        try:
            await websocket.close()
        except:
            pass


def generate_alert_message() -> str:
    """Generate a random alert message."""
    messages = [
        "Strong buy signal detected on BTC/USDT",
        "Position stop-loss triggered",
        "Model confidence exceeds 90%",
        "Unusual volume spike detected",
        "Risk limit approaching",
        "New high probability setup identified",
        "Market regime change detected",
        "API rate limit warning",
        "Model retraining completed",
        "Drawdown alert: -5% in last hour"
    ]
    return np.random.choice(messages)