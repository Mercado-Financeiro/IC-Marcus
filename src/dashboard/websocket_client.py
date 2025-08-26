"""
WebSocket client for real-time dashboard updates.
Handles live data streaming for the ML Trading Dashboard.
"""

import asyncio
import json
import websocket
import threading
from typing import Dict, Callable, Optional, List
from datetime import datetime
import streamlit as st
import pandas as pd
import numpy as np
from queue import Queue
import logging

logger = logging.getLogger(__name__)

class DashboardWebSocketClient:
    """WebSocket client for real-time dashboard updates."""
    
    def __init__(self, url: str = "ws://localhost:8000/ws"):
        """
        Initialize WebSocket client.
        
        Args:
            url: WebSocket server URL
        """
        self.url = url
        self.ws = None
        self.running = False
        self.callbacks = {}
        self.message_queue = Queue()
        self.reconnect_interval = 5
        self.max_reconnect_attempts = 10
        self.current_reconnect_attempt = 0
        
    def connect(self):
        """Establish WebSocket connection."""
        try:
            self.ws = websocket.WebSocketApp(
                self.url,
                on_open=self.on_open,
                on_message=self.on_message,
                on_error=self.on_error,
                on_close=self.on_close
            )
            
            # Run in separate thread
            self.wst = threading.Thread(target=self.ws.run_forever)
            self.wst.daemon = True
            self.wst.start()
            self.running = True
            logger.info(f"WebSocket connected to {self.url}")
            
        except Exception as e:
            logger.error(f"WebSocket connection failed: {e}")
            self.handle_reconnect()
    
    def on_open(self, ws):
        """Handle connection open."""
        logger.info("WebSocket connection opened")
        self.current_reconnect_attempt = 0
        
        # Subscribe to channels
        self.subscribe_to_channels()
    
    def on_message(self, ws, message):
        """Handle incoming message."""
        try:
            data = json.loads(message)
            channel = data.get('channel', 'default')
            
            # Put message in queue for processing
            self.message_queue.put(data)
            
            # Execute callbacks
            if channel in self.callbacks:
                for callback in self.callbacks[channel]:
                    callback(data)
                    
        except json.JSONDecodeError:
            logger.error(f"Failed to decode message: {message}")
        except Exception as e:
            logger.error(f"Error processing message: {e}")
    
    def on_error(self, ws, error):
        """Handle WebSocket error."""
        logger.error(f"WebSocket error: {error}")
    
    def on_close(self, ws, close_status_code, close_msg):
        """Handle connection close."""
        logger.info(f"WebSocket closed: {close_status_code} - {close_msg}")
        self.running = False
        self.handle_reconnect()
    
    def handle_reconnect(self):
        """Handle reconnection logic."""
        if self.current_reconnect_attempt < self.max_reconnect_attempts:
            self.current_reconnect_attempt += 1
            logger.info(f"Reconnecting... Attempt {self.current_reconnect_attempt}")
            threading.Timer(self.reconnect_interval, self.connect).start()
        else:
            logger.error("Max reconnection attempts reached")
    
    def subscribe_to_channels(self):
        """Subscribe to data channels."""
        channels = ['predictions', 'market_data', 'positions', 'alerts']
        
        for channel in channels:
            self.send_message({
                'action': 'subscribe',
                'channel': channel
            })
    
    def send_message(self, message: Dict):
        """Send message to WebSocket server."""
        if self.ws and self.running:
            try:
                self.ws.send(json.dumps(message))
            except Exception as e:
                logger.error(f"Failed to send message: {e}")
    
    def register_callback(self, channel: str, callback: Callable):
        """
        Register callback for channel.
        
        Args:
            channel: Channel name
            callback: Callback function
        """
        if channel not in self.callbacks:
            self.callbacks[channel] = []
        self.callbacks[channel].append(callback)
    
    def unregister_callback(self, channel: str, callback: Callable):
        """Unregister callback from channel."""
        if channel in self.callbacks:
            self.callbacks[channel].remove(callback)
    
    def close(self):
        """Close WebSocket connection."""
        self.running = False
        if self.ws:
            self.ws.close()


class StreamlitWebSocketManager:
    """Manager for WebSocket integration with Streamlit."""
    
    def __init__(self):
        """Initialize WebSocket manager for Streamlit."""
        if 'ws_client' not in st.session_state:
            st.session_state.ws_client = None
        if 'ws_data' not in st.session_state:
            st.session_state.ws_data = {
                'predictions': [],
                'market_data': {},
                'positions': [],
                'alerts': []
            }
    
    def initialize_connection(self, url: str = "ws://localhost:8000/ws"):
        """Initialize WebSocket connection."""
        if st.session_state.ws_client is None:
            client = DashboardWebSocketClient(url)
            
            # Register callbacks
            client.register_callback('predictions', self.handle_prediction)
            client.register_callback('market_data', self.handle_market_data)
            client.register_callback('positions', self.handle_positions)
            client.register_callback('alerts', self.handle_alerts)
            
            # Connect
            client.connect()
            st.session_state.ws_client = client
            
            return client
        
        return st.session_state.ws_client
    
    def handle_prediction(self, data: Dict):
        """Handle prediction updates."""
        st.session_state.ws_data['predictions'].append({
            'timestamp': datetime.now(),
            'prediction': data.get('prediction'),
            'confidence': data.get('confidence'),
            'signal': data.get('signal')
        })
        
        # Keep only last 100 predictions
        if len(st.session_state.ws_data['predictions']) > 100:
            st.session_state.ws_data['predictions'] = \
                st.session_state.ws_data['predictions'][-100:]
    
    def handle_market_data(self, data: Dict):
        """Handle market data updates."""
        st.session_state.ws_data['market_data'].update({
            'price': data.get('price'),
            'volume': data.get('volume'),
            'bid': data.get('bid'),
            'ask': data.get('ask'),
            'timestamp': datetime.now()
        })
    
    def handle_positions(self, data: Dict):
        """Handle position updates."""
        st.session_state.ws_data['positions'] = data.get('positions', [])
    
    def handle_alerts(self, data: Dict):
        """Handle alert notifications."""
        alert = {
            'timestamp': datetime.now(),
            'type': data.get('type'),
            'message': data.get('message'),
            'severity': data.get('severity', 'info')
        }
        
        st.session_state.ws_data['alerts'].append(alert)
        
        # Show toast notification
        if alert['severity'] == 'error':
            st.error(alert['message'])
        elif alert['severity'] == 'warning':
            st.warning(alert['message'])
        else:
            st.info(alert['message'])
    
    def get_latest_prediction(self) -> Optional[Dict]:
        """Get latest prediction from WebSocket."""
        predictions = st.session_state.ws_data.get('predictions', [])
        return predictions[-1] if predictions else None
    
    def get_market_data(self) -> Dict:
        """Get current market data."""
        return st.session_state.ws_data.get('market_data', {})
    
    def get_positions(self) -> List[Dict]:
        """Get current positions."""
        return st.session_state.ws_data.get('positions', [])
    
    def get_alerts(self, limit: int = 10) -> List[Dict]:
        """Get recent alerts."""
        alerts = st.session_state.ws_data.get('alerts', [])
        return alerts[-limit:] if alerts else []
    
    def render_connection_status(self):
        """Render WebSocket connection status."""
        if st.session_state.ws_client and st.session_state.ws_client.running:
            st.success("🟢 Connected to live data")
        else:
            st.error("🔴 Disconnected from live data")
            if st.button("Reconnect"):
                self.initialize_connection()


class MockDataGenerator:
    """Generate mock data for testing without real WebSocket."""
    
    @staticmethod
    def generate_market_data() -> Dict:
        """Generate mock market data."""
        base_price = 50000
        return {
            'price': base_price + np.random.randn() * 100,
            'volume': np.random.uniform(1000, 5000),
            'bid': base_price - np.random.uniform(10, 50),
            'ask': base_price + np.random.uniform(10, 50),
            'change_24h': np.random.uniform(-5, 5),
            'high_24h': base_price + np.random.uniform(500, 1000),
            'low_24h': base_price - np.random.uniform(500, 1000),
            'timestamp': datetime.now()
        }
    
    @staticmethod
    def generate_prediction() -> Dict:
        """Generate mock prediction."""
        prediction = np.random.random()
        signal = 'LONG' if prediction > 0.65 else 'SHORT' if prediction < 0.35 else 'NEUTRAL'
        
        return {
            'prediction': prediction,
            'confidence': np.random.uniform(0.5, 0.95),
            'signal': signal,
            'timestamp': datetime.now()
        }
    
    @staticmethod
    def generate_positions() -> List[Dict]:
        """Generate mock positions."""
        symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT']
        positions = []
        
        for symbol in symbols[:np.random.randint(1, 4)]:
            entry_price = np.random.uniform(40000, 60000) if symbol == 'BTCUSDT' else np.random.uniform(100, 5000)
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
                'timestamp': datetime.now()
            })
        
        return positions