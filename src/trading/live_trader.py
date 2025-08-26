#!/usr/bin/env python3
"""
Live Trading Bot with ML Model Integration
Real-time cryptocurrency trading using pre-trained models.
"""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
import pickle
import yaml
import json
import asyncio
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import threading
import queue
import logging
import ccxt.async_support as ccxt
import websockets
from dataclasses import dataclass, asdict

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.utils.secure_logging import SecureLogger
from src.utils.secure_loader import safe_load_model
from src.features.engineering import FeatureEngineer
from src.trading.paper_trader import Position

# Setup logging
logger = SecureLogger.setup_logger(__name__, log_file='logs/live_trader.log')


@dataclass
class TradingSignal:
    """Trading signal data structure."""
    timestamp: datetime
    symbol: str
    action: str  # 'buy', 'sell', 'hold'
    confidence: float
    predicted_return: float
    features: Dict[str, float]
    model_type: str
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'symbol': self.symbol,
            'action': self.action,
            'confidence': self.confidence,
            'predicted_return': self.predicted_return,
            'features': self.features,
            'model_type': self.model_type
        }


@dataclass
class MarketData:
    """Market data structure."""
    timestamp: datetime
    symbol: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    bid: float
    ask: float
    spread: float
    
    @classmethod
    def from_ticker(cls, ticker: dict, symbol: str):
        """Create from exchange ticker data."""
        return cls(
            timestamp=datetime.now(),
            symbol=symbol,
            open=ticker.get('open', 0),
            high=ticker.get('high', 0),
            low=ticker.get('low', 0),
            close=ticker.get('last', 0),
            volume=ticker.get('baseVolume', 0),
            bid=ticker.get('bid', 0),
            ask=ticker.get('ask', 0),
            spread=ticker.get('ask', 0) - ticker.get('bid', 0)
        )


class LiveTradingBot:
    """Live trading bot with ML model integration."""
    
    def __init__(
        self,
        model_path: str,
        model_type: str = 'xgboost',
        exchange_name: str = 'binance',
        config_path: str = 'configs/trading.yaml',
        mode: str = 'simulation'  # 'simulation' or 'live'
    ):
        """
        Initialize live trading bot.
        
        Args:
            model_path: Path to pre-trained model
            model_type: Type of model ('xgboost' or 'lstm')
            exchange_name: Exchange to trade on
            config_path: Path to trading configuration
            mode: Trading mode (simulation or live)
        """
        self.model_path = Path(model_path)
        self.model_type = model_type.lower()
        self.exchange_name = exchange_name
        self.mode = mode
        
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Initialize components
        self.model = None
        self.feature_engineer = FeatureEngineer()
        self.exchange = None
        self.positions: Dict[str, Position] = {}
        self.signal_queue = queue.Queue()
        self.market_data: Dict[str, MarketData] = {}
        
        # Trading parameters
        self.initial_capital = self.config['trading']['initial_capital']
        self.position_size_pct = self.config['trading']['position_size_pct']
        self.max_positions = self.config['trading']['max_positions']
        self.confidence_threshold = self.config['trading']['confidence_threshold']
        self.stop_loss_pct = self.config['risk']['stop_loss_pct']
        self.take_profit_pct = self.config['risk']['take_profit_pct']
        
        # Performance tracking
        self.balance = self.initial_capital
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.total_pnl = 0.0
        
        # Control flags
        self.is_running = False
        self.shutdown_event = threading.Event()
        
        logger.info(f"LiveTradingBot initialized in {mode} mode")
    
    def _load_config(self, config_path: str) -> dict:
        """Load trading configuration."""
        config_file = Path(config_path)
        if not config_file.exists():
            # Return default configuration
            return self._get_default_config()
        
        with open(config_file, 'r') as f:
            return yaml.safe_load(f)
    
    def _get_default_config(self) -> dict:
        """Get default trading configuration."""
        return {
            'trading': {
                'initial_capital': 10000,
                'position_size_pct': 0.1,
                'max_positions': 3,
                'confidence_threshold': 0.65,
                'symbols': ['BTC/USDT', 'ETH/USDT'],
                'timeframe': '1h'
            },
            'risk': {
                'stop_loss_pct': 0.02,
                'take_profit_pct': 0.05,
                'max_drawdown_pct': 0.10,
                'trailing_stop': True,
                'trailing_stop_pct': 0.015
            },
            'model': {
                'update_interval': 3600,  # seconds
                'min_data_points': 100,
                'feature_lookback': 24
            },
            'exchange': {
                'rate_limit': True,
                'timeout': 30000,
                'enable_rate_limit': True
            }
        }
    
    async def initialize(self):
        """Initialize bot components."""
        try:
            # Load model
            logger.info(f"Loading model from {self.model_path}")
            self.model = safe_load_model(self.model_path, model_type='auto')
            logger.info(f"Model loaded successfully: {self.model_type}")
            
            # Initialize exchange
            await self._initialize_exchange()
            
            # Load initial market data
            await self._load_initial_data()
            
            logger.info("Bot initialization complete")
            
        except Exception as e:
            logger.error(f"Initialization failed: {e}")
            raise
    
    async def _initialize_exchange(self):
        """Initialize exchange connection."""
        try:
            if self.mode == 'simulation':
                logger.info("Running in simulation mode - no real trades")
                # Create mock exchange for simulation
                self.exchange = None  # Will use mock data
            else:
                # Initialize real exchange
                exchange_class = getattr(ccxt, self.exchange_name)
                
                # Load API credentials from environment
                api_key = os.getenv(f"{self.exchange_name.upper()}_API_KEY")
                api_secret = os.getenv(f"{self.exchange_name.upper()}_SECRET")
                
                self.exchange = exchange_class({
                    'apiKey': api_key,
                    'secret': api_secret,
                    'enableRateLimit': self.config['exchange']['enable_rate_limit'],
                    'timeout': self.config['exchange']['timeout']
                })
                
                # Test connection
                await self.exchange.load_markets()
                logger.info(f"Connected to {self.exchange_name}")
                
        except Exception as e:
            logger.error(f"Exchange initialization failed: {e}")
            raise
    
    async def _load_initial_data(self):
        """Load initial market data for all symbols."""
        for symbol in self.config['trading']['symbols']:
            try:
                if self.exchange:
                    # Get real data
                    ticker = await self.exchange.fetch_ticker(symbol)
                    self.market_data[symbol] = MarketData.from_ticker(ticker, symbol)
                else:
                    # Use mock data for simulation
                    self.market_data[symbol] = self._generate_mock_market_data(symbol)
                
                logger.info(f"Loaded initial data for {symbol}")
                
            except Exception as e:
                logger.error(f"Failed to load data for {symbol}: {e}")
    
    def _generate_mock_market_data(self, symbol: str) -> MarketData:
        """Generate mock market data for simulation."""
        base_price = 50000 if 'BTC' in symbol else 3000
        return MarketData(
            timestamp=datetime.now(),
            symbol=symbol,
            open=base_price * (1 + np.random.randn() * 0.001),
            high=base_price * (1 + abs(np.random.randn()) * 0.002),
            low=base_price * (1 - abs(np.random.randn()) * 0.002),
            close=base_price * (1 + np.random.randn() * 0.001),
            volume=np.random.uniform(100, 1000),
            bid=base_price * 0.999,
            ask=base_price * 1.001,
            spread=base_price * 0.002
        )
    
    async def generate_signal(self, symbol: str) -> Optional[TradingSignal]:
        """Generate trading signal for a symbol."""
        try:
            # Get historical data
            df = await self._get_historical_data(symbol)
            
            if df is None or len(df) < self.config['model']['min_data_points']:
                logger.warning(f"Insufficient data for {symbol}")
                return None
            
            # Generate features
            features_df = self.feature_engineer.create_features(df)
            
            # Get latest features
            latest_features = features_df.iloc[-1:].copy()
            
            # Make prediction
            if self.model_type == 'xgboost':
                prediction = self.model.predict_proba(latest_features)[0, 1]
                predicted_return = self.model.predict(latest_features)[0]
            else:
                # LSTM model handling
                # This would require proper tensor conversion
                prediction = 0.5  # Placeholder
                predicted_return = 0.0
            
            # Determine action
            if prediction > self.confidence_threshold + 0.1:
                action = 'buy'
            elif prediction < self.confidence_threshold - 0.1:
                action = 'sell'
            else:
                action = 'hold'
            
            signal = TradingSignal(
                timestamp=datetime.now(),
                symbol=symbol,
                action=action,
                confidence=float(prediction),
                predicted_return=float(predicted_return),
                features=latest_features.to_dict('records')[0],
                model_type=self.model_type
            )
            
            logger.info(f"Generated signal for {symbol}: {action} (confidence: {prediction:.3f})")
            return signal
            
        except Exception as e:
            logger.error(f"Signal generation failed for {symbol}: {e}")
            return None
    
    async def _get_historical_data(self, symbol: str, limit: int = 200) -> Optional[pd.DataFrame]:
        """Get historical OHLCV data."""
        try:
            if self.exchange:
                # Get real data
                timeframe = self.config['trading']['timeframe']
                ohlcv = await self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
                
                df = pd.DataFrame(
                    ohlcv,
                    columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
                )
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df.set_index('timestamp', inplace=True)
                
            else:
                # Generate mock historical data
                df = self._generate_mock_historical_data(symbol, limit)
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to get historical data for {symbol}: {e}")
            return None
    
    def _generate_mock_historical_data(self, symbol: str, limit: int) -> pd.DataFrame:
        """Generate mock historical data for simulation."""
        base_price = 50000 if 'BTC' in symbol else 3000
        
        # Generate random walk
        returns = np.random.randn(limit) * 0.01
        prices = base_price * np.exp(np.cumsum(returns))
        
        timestamps = pd.date_range(
            end=datetime.now(),
            periods=limit,
            freq='1H'
        )
        
        df = pd.DataFrame({
            'open': prices * (1 + np.random.randn(limit) * 0.001),
            'high': prices * (1 + abs(np.random.randn(limit)) * 0.002),
            'low': prices * (1 - abs(np.random.randn(limit)) * 0.002),
            'close': prices,
            'volume': np.random.uniform(100, 1000, limit)
        }, index=timestamps)
        
        return df
    
    async def execute_trade(self, signal: TradingSignal) -> bool:
        """Execute trade based on signal."""
        try:
            symbol = signal.symbol
            
            # Check position limits
            if len(self.positions) >= self.max_positions:
                logger.warning("Max positions reached, skipping trade")
                return False
            
            # Check if we already have a position in this symbol
            if symbol in self.positions and self.positions[symbol].status == 'open':
                logger.warning(f"Already have open position in {symbol}")
                return False
            
            # Calculate position size
            position_size = self.balance * self.position_size_pct
            current_price = self.market_data[symbol].close
            
            # Apply slippage
            if signal.action == 'buy':
                entry_price = current_price * 1.001  # 0.1% slippage
                size = position_size / entry_price
                
                # Create position
                position = Position(symbol, 'long', size, entry_price)
                
            elif signal.action == 'sell' and symbol in self.positions:
                # Close existing position
                position = self.positions[symbol]
                exit_price = current_price * 0.999  # 0.1% slippage
                pnl = position.close(exit_price)
                
                # Update performance
                self.total_pnl += pnl
                self.balance += pnl
                self.total_trades += 1
                
                if pnl > 0:
                    self.winning_trades += 1
                else:
                    self.losing_trades += 1
                
                logger.info(f"Closed position in {symbol}: PnL = ${pnl:.2f}")
                return True
                
            else:
                return False
            
            # Store position
            self.positions[symbol] = position
            self.total_trades += 1
            
            # Apply fees
            fee = position_size * 0.001
            self.balance -= fee
            
            logger.info(f"Opened {position.side} position in {symbol}: "
                       f"size={size:.4f}, entry=${entry_price:.2f}")
            
            return True
            
        except Exception as e:
            logger.error(f"Trade execution failed: {e}")
            return False
    
    async def manage_positions(self):
        """Manage open positions (stop-loss, take-profit)."""
        for symbol, position in list(self.positions.items()):
            if position.status == 'closed':
                continue
            
            try:
                current_price = self.market_data[symbol].close
                entry_price = position.entry_price
                
                # Calculate current P&L percentage
                if position.side == 'long':
                    pnl_pct = (current_price - entry_price) / entry_price
                else:
                    pnl_pct = (entry_price - current_price) / entry_price
                
                # Check stop-loss
                if pnl_pct <= -self.stop_loss_pct:
                    logger.warning(f"Stop-loss triggered for {symbol}")
                    await self._close_position(symbol, "stop-loss")
                
                # Check take-profit
                elif pnl_pct >= self.take_profit_pct:
                    logger.info(f"Take-profit triggered for {symbol}")
                    await self._close_position(symbol, "take-profit")
                
                # Trailing stop (if enabled)
                elif self.config['risk']['trailing_stop'] and pnl_pct > 0:
                    trailing_stop_price = current_price * (1 - self.config['risk']['trailing_stop_pct'])
                    if position.side == 'long' and current_price <= trailing_stop_price:
                        logger.info(f"Trailing stop triggered for {symbol}")
                        await self._close_position(symbol, "trailing-stop")
                        
            except Exception as e:
                logger.error(f"Position management error for {symbol}: {e}")
    
    async def _close_position(self, symbol: str, reason: str):
        """Close a position."""
        if symbol not in self.positions:
            return
        
        position = self.positions[symbol]
        current_price = self.market_data[symbol].close
        
        # Apply slippage
        if position.side == 'long':
            exit_price = current_price * 0.999
        else:
            exit_price = current_price * 1.001
        
        pnl = position.close(exit_price)
        
        # Update performance
        self.total_pnl += pnl
        self.balance += pnl
        
        if pnl > 0:
            self.winning_trades += 1
        else:
            self.losing_trades += 1
        
        logger.info(f"Closed {symbol} ({reason}): PnL = ${pnl:.2f}")
    
    async def update_market_data(self):
        """Update market data for all symbols."""
        for symbol in self.config['trading']['symbols']:
            try:
                if self.exchange:
                    ticker = await self.exchange.fetch_ticker(symbol)
                    self.market_data[symbol] = MarketData.from_ticker(ticker, symbol)
                else:
                    # Update mock data with random walk
                    old_data = self.market_data[symbol]
                    new_price = old_data.close * (1 + np.random.randn() * 0.001)
                    self.market_data[symbol] = MarketData(
                        timestamp=datetime.now(),
                        symbol=symbol,
                        open=old_data.close,
                        high=max(old_data.close, new_price),
                        low=min(old_data.close, new_price),
                        close=new_price,
                        volume=np.random.uniform(100, 1000),
                        bid=new_price * 0.999,
                        ask=new_price * 1.001,
                        spread=new_price * 0.002
                    )
                    
            except Exception as e:
                logger.error(f"Failed to update market data for {symbol}: {e}")
    
    async def trading_loop(self):
        """Main trading loop."""
        logger.info("Starting trading loop")
        
        while not self.shutdown_event.is_set():
            try:
                # Update market data
                await self.update_market_data()
                
                # Generate signals for all symbols
                for symbol in self.config['trading']['symbols']:
                    signal = await self.generate_signal(symbol)
                    
                    if signal and signal.action != 'hold':
                        # Execute trade
                        success = await self.execute_trade(signal)
                        
                        # Store signal for dashboard
                        self.signal_queue.put(signal.to_dict())
                
                # Manage existing positions
                await self.manage_positions()
                
                # Log performance
                self._log_performance()
                
                # Sleep before next iteration
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Trading loop error: {e}")
                await asyncio.sleep(10)
    
    def _log_performance(self):
        """Log current performance metrics."""
        win_rate = self.winning_trades / max(self.total_trades, 1)
        current_value = self.balance + sum(
            p.unrealized_pnl(self.market_data[p.symbol].close)
            for p in self.positions.values()
            if p.status == 'open'
        )
        
        logger.info(f"Performance - Balance: ${self.balance:.2f}, "
                   f"Total P&L: ${self.total_pnl:.2f}, "
                   f"Win Rate: {win_rate:.1%}, "
                   f"Trades: {self.total_trades}, "
                   f"Portfolio Value: ${current_value:.2f}")
    
    async def start(self):
        """Start the trading bot."""
        try:
            # Initialize components
            await self.initialize()
            
            # Set running flag
            self.is_running = True
            
            # Start trading loop
            await self.trading_loop()
            
        except KeyboardInterrupt:
            logger.info("Received shutdown signal")
        except Exception as e:
            logger.error(f"Bot crashed: {e}")
        finally:
            await self.stop()
    
    async def stop(self):
        """Stop the trading bot."""
        logger.info("Stopping trading bot...")
        
        # Set shutdown flag
        self.shutdown_event.set()
        self.is_running = False
        
        # Close all positions
        for symbol in list(self.positions.keys()):
            if self.positions[symbol].status == 'open':
                await self._close_position(symbol, "shutdown")
        
        # Close exchange connection
        if self.exchange:
            await self.exchange.close()
        
        # Final performance report
        self._generate_final_report()
        
        logger.info("Trading bot stopped")
    
    def _generate_final_report(self):
        """Generate final performance report."""
        win_rate = self.winning_trades / max(self.total_trades, 1)
        avg_win = self.total_pnl / max(self.winning_trades, 1) if self.winning_trades > 0 else 0
        avg_loss = self.total_pnl / max(self.losing_trades, 1) if self.losing_trades > 0 else 0
        
        report = {
            'final_balance': self.balance,
            'initial_capital': self.initial_capital,
            'total_pnl': self.total_pnl,
            'roi': (self.balance - self.initial_capital) / self.initial_capital,
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'win_rate': win_rate,
            'average_win': avg_win,
            'average_loss': avg_loss,
            'profit_factor': abs(avg_win / avg_loss) if avg_loss != 0 else 0
        }
        
        # Save report
        report_file = Path('reports') / f'trading_report_{datetime.now():%Y%m%d_%H%M%S}.json'
        report_file.parent.mkdir(exist_ok=True)
        
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Final Report:\n{json.dumps(report, indent=2)}")
    
    def get_status(self) -> dict:
        """Get current bot status for dashboard."""
        return {
            'is_running': self.is_running,
            'mode': self.mode,
            'balance': self.balance,
            'total_pnl': self.total_pnl,
            'total_trades': self.total_trades,
            'open_positions': len([p for p in self.positions.values() if p.status == 'open']),
            'win_rate': self.winning_trades / max(self.total_trades, 1),
            'positions': [p.to_dict() for p in self.positions.values()],
            'market_data': {s: asdict(d) for s, d in self.market_data.items()}
        }


async def main():
    """Main function to run the trading bot."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Live Trading Bot')
    parser.add_argument('--model', type=str, default='artifacts/models/xgboost_optuna_20250825_092645.pkl',
                       help='Path to model file')
    parser.add_argument('--model-type', type=str, default='xgboost',
                       choices=['xgboost', 'lstm'], help='Type of model')
    parser.add_argument('--exchange', type=str, default='binance',
                       help='Exchange to trade on')
    parser.add_argument('--mode', type=str, default='simulation',
                       choices=['simulation', 'live'], help='Trading mode')
    parser.add_argument('--config', type=str, default='configs/trading.yaml',
                       help='Path to configuration file')
    
    args = parser.parse_args()
    
    # Create bot
    bot = LiveTradingBot(
        model_path=args.model,
        model_type=args.model_type,
        exchange_name=args.exchange,
        config_path=args.config,
        mode=args.mode
    )
    
    # Start bot
    await bot.start()


if __name__ == "__main__":
    asyncio.run(main())