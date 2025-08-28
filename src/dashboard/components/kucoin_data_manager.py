"""
KuCoin Data Manager for direct integration with Streamlit.
Eliminates need for separate WebSocket server.
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
from typing import Dict, List, Optional, Any
import asyncio

# Try to import CCXT
try:
    import ccxt
    HAS_CCXT = True
except ImportError:
    HAS_CCXT = False

class KuCoinDataManager:
    """Direct KuCoin data management for Streamlit."""
    
    def __init__(self):
        """Initialize data manager."""
        self.exchange = None
        if HAS_CCXT:
            try:
                self.exchange = ccxt.kucoin({
                    'enableRateLimit': True,
                    'timeout': 30000,
                    'options': {'fetchOHLCVLimit': 1000}
                })
                # Try to load markets
                markets = self.exchange.load_markets()
                st.success("✅ Connected to KuCoin exchange")
            except Exception as e:
                st.warning(f"⚠️ KuCoin connection failed: {e}. Using mock data.")
                self.exchange = None
        else:
            st.info("ℹ️ CCXT not available. Using mock data for demonstration.")
    
    @st.cache_data(ttl=10)  # Cache for 10 seconds
    def get_ticker_data(_self, symbol: str) -> Dict[str, Any]:
        """Get ticker data with caching."""
        if _self.exchange:
            try:
                ticker = _self.exchange.fetch_ticker(symbol)
                return {
                    'symbol': ticker['symbol'],
                    'last': ticker['last'],
                    'high': ticker['high'],
                    'low': ticker['low'],
                    'volume': ticker['quoteVolume'] or ticker['baseVolume'],
                    'change': ticker['change'],
                    'percentage': ticker['percentage'],
                    'timestamp': ticker['timestamp']
                }
            except Exception as e:
                st.error(f"Error fetching ticker: {e}")
        
        # Mock data fallback
        return _self._generate_mock_ticker(symbol)
    
    @st.cache_data(ttl=5)  # Cache for 5 seconds
    def get_orderbook_data(_self, symbol: str, limit: int = 10) -> Dict[str, Any]:
        """Get order book data with caching."""
        if _self.exchange:
            try:
                orderbook = _self.exchange.fetch_order_book(symbol, limit=limit*2)
                return {
                    'symbol': symbol,
                    'bids': orderbook['bids'][:limit],
                    'asks': orderbook['asks'][:limit],
                    'timestamp': orderbook['timestamp']
                }
            except Exception as e:
                st.error(f"Error fetching order book: {e}")
        
        # Mock data fallback
        return _self._generate_mock_orderbook(symbol, limit)
    
    @st.cache_data(ttl=3)  # Cache for 3 seconds
    def get_trades_data(_self, symbol: str, limit: int = 20) -> List[Dict[str, Any]]:
        """Get recent trades data with caching."""
        if _self.exchange:
            try:
                trades = _self.exchange.fetch_trades(symbol, limit=limit)
                return [
                    {
                        'id': trade['id'],
                        'timestamp': trade['timestamp'],
                        'price': trade['price'],
                        'amount': trade['amount'],
                        'side': trade['side'],
                        'symbol': symbol
                    }
                    for trade in trades
                ]
            except Exception as e:
                st.error(f"Error fetching trades: {e}")
        
        # Mock data fallback
        return _self._generate_mock_trades(symbol, limit)
    
    @st.cache_data(ttl=60)  # Cache for 1 minute
    def get_chart_data(_self, symbol: str, timeframe: str = '1h', limit: int = 100) -> List[Dict[str, Any]]:
        """Get OHLCV chart data with caching."""
        if _self.exchange:
            try:
                ohlcv = _self.exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
                return [
                    {
                        'timestamp': item[0],
                        'open': item[1],
                        'high': item[2],
                        'low': item[3],
                        'close': item[4],
                        'volume': item[5]
                    }
                    for item in ohlcv
                ]
            except Exception as e:
                st.error(f"Error fetching chart data: {e}")
        
        # Mock data fallback
        return _self._generate_mock_chart_data(symbol, limit)
    
    def _generate_mock_ticker(self, symbol: str) -> Dict[str, Any]:
        """Generate mock ticker data."""
        import random
        base_price = 50000 if 'BTC' in symbol else 3000 if 'ETH' in symbol else 1.0
        
        # Add some randomness but keep it realistic
        price_variation = random.uniform(-0.05, 0.05)
        current_price = base_price * (1 + price_variation)
        
        return {
            'symbol': symbol,
            'last': current_price,
            'high': current_price * 1.02,
            'low': current_price * 0.98,
            'volume': random.uniform(1000000, 5000000),
            'change': current_price - base_price,
            'percentage': price_variation * 100,
            'timestamp': int(datetime.now().timestamp() * 1000)
        }
    
    def _generate_mock_orderbook(self, symbol: str, limit: int) -> Dict[str, Any]:
        """Generate mock order book data."""
        import random
        base_price = 50000 if 'BTC' in symbol else 3000 if 'ETH' in symbol else 1.0
        
        bids = []
        asks = []
        
        for i in range(limit):
            bid_price = base_price * (1 - (i + 1) * 0.0001)
            ask_price = base_price * (1 + (i + 1) * 0.0001)
            
            bids.append([bid_price, random.uniform(0.1, 5.0)])
            asks.append([ask_price, random.uniform(0.1, 5.0)])
        
        return {
            'symbol': symbol,
            'bids': bids,
            'asks': asks,
            'timestamp': int(datetime.now().timestamp() * 1000)
        }
    
    def _generate_mock_trades(self, symbol: str, limit: int) -> List[Dict[str, Any]]:
        """Generate mock recent trades."""
        import random
        base_price = 50000 if 'BTC' in symbol else 3000 if 'ETH' in symbol else 1.0
        
        trades = []
        for i in range(limit):
            trades.append({
                'id': f"mock_{i}_{int(time.time())}",
                'timestamp': int(datetime.now().timestamp() * 1000) - i * 1000,
                'price': base_price * (1 + random.uniform(-0.001, 0.001)),
                'amount': random.uniform(0.01, 1.0),
                'side': random.choice(['buy', 'sell']),
                'symbol': symbol
            })
        return trades
    
    def _generate_mock_chart_data(self, symbol: str, limit: int) -> List[Dict[str, Any]]:
        """Generate mock OHLCV data."""
        import random
        base_price = 50000 if 'BTC' in symbol else 3000 if 'ETH' in symbol else 1.0
        
        data = []
        current_price = base_price
        
        for i in range(limit):
            timestamp = int(datetime.now().timestamp() * 1000) - (limit - i) * 3600000  # Hour intervals
            
            # Random walk
            price_change = random.uniform(-0.02, 0.02)
            current_price *= (1 + price_change)
            
            open_price = current_price
            close_price = current_price * (1 + random.uniform(-0.01, 0.01))
            high_price = max(open_price, close_price) * (1 + random.uniform(0, 0.005))
            low_price = min(open_price, close_price) * (1 - random.uniform(0, 0.005))
            
            data.append({
                'timestamp': timestamp,
                'open': open_price,
                'high': high_price,
                'low': low_price,
                'close': close_price,
                'volume': random.uniform(100, 1000)
            })
            
            current_price = close_price
        
        return data
    
    def is_connected(self) -> bool:
        """Check if connected to real exchange."""
        return self.exchange is not None
    
    def get_available_symbols(self) -> List[str]:
        """Get available trading symbols."""
        if self.exchange:
            try:
                markets = self.exchange.markets
                symbols = [symbol for symbol in markets.keys() if '/USDT' in symbol]
                return sorted(symbols[:50])  # Limit to 50 symbols
            except:
                pass
        
        # Default symbols for mock data
        return [
            "BTC/USDT", "ETH/USDT", "BNB/USDT", "ADA/USDT", "XRP/USDT",
            "SOL/USDT", "DOT/USDT", "AVAX/USDT", "MATIC/USDT", "LINK/USDT",
            "UNI/USDT", "LTC/USDT", "BCH/USDT", "ETC/USDT", "ATOM/USDT"
        ]


# Singleton instance
@st.cache_resource
def get_kucoin_data_manager() -> KuCoinDataManager:
    """Get or create KuCoin data manager singleton."""
    return KuCoinDataManager()