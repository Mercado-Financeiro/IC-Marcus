#!/usr/bin/env python3
"""
Paper trading bot for cryptocurrency ML models.
Simulates real trading without actual money.
"""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
import pickle
import yaml
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import threading
import queue
import logging

# Add parent to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.data.binance_loader import CryptoDataLoader
from src.features.engineering import FeatureEngineer
from src.inference.predict import CryptoPredictor

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class Position:
    """Represents a trading position."""
    
    def __init__(self, symbol: str, side: str, size: float, entry_price: float):
        self.symbol = symbol
        self.side = side  # 'long' or 'short'
        self.size = size
        self.entry_price = entry_price
        self.entry_time = datetime.now()
        self.exit_price = None
        self.exit_time = None
        self.pnl = 0.0
        self.status = 'open'  # open, closed
        
    def close(self, exit_price: float):
        """Close the position."""
        self.exit_price = exit_price
        self.exit_time = datetime.now()
        
        if self.side == 'long':
            self.pnl = (exit_price - self.entry_price) * self.size
        else:  # short
            self.pnl = (self.entry_price - exit_price) * self.size
        
        self.status = 'closed'
        return self.pnl
    
    def unrealized_pnl(self, current_price: float) -> float:
        """Calculate unrealized P&L."""
        if self.status == 'closed':
            return self.pnl
        
        if self.side == 'long':
            return (current_price - self.entry_price) * self.size
        else:
            return (self.entry_price - current_price) * self.size
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            'symbol': self.symbol,
            'side': self.side,
            'size': self.size,
            'entry_price': self.entry_price,
            'entry_time': self.entry_time.isoformat() if self.entry_time else None,
            'exit_price': self.exit_price,
            'exit_time': self.exit_time.isoformat() if self.exit_time else None,
            'pnl': self.pnl,
            'status': self.status
        }


class PaperTradingBot:
    """Paper trading bot for testing ML strategies."""
    
    def __init__(
        self,
        model_path: str,
        initial_capital: float = 10000,
        position_size_pct: float = 0.1,
        max_positions: int = 1,
        fee_pct: float = 0.001,
        slippage_pct: float = 0.0005
    ):
        """Initialize paper trading bot.
        
        Args:
            model_path: Path to trained model
            initial_capital: Starting capital
            position_size_pct: Position size as % of capital
            max_positions: Maximum concurrent positions
            fee_pct: Trading fee percentage
            slippage_pct: Slippage percentage
        """
        self.predictor = CryptoPredictor(model_path)
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.position_size_pct = position_size_pct
        self.max_positions = max_positions
        self.fee_pct = fee_pct
        self.slippage_pct = slippage_pct
        
        # Trading state
        self.positions = []
        self.closed_positions = []
        self.trades = []
        self.equity_curve = [initial_capital]
        self.timestamps = [datetime.now()]
        
        # Performance metrics
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.total_pnl = 0.0
        self.max_drawdown = 0.0
        
        # Control
        self.is_running = False
        self.trading_thread = None
        
        logger.info(f"Paper trading bot initialized with ${initial_capital}")
    
    def calculate_position_size(self, capital: float, price: float) -> float:
        """Calculate position size based on capital."""
        position_value = capital * self.position_size_pct
        position_size = position_value / price
        return position_size
    
    def apply_slippage(self, price: float, side: str) -> float:
        """Apply slippage to execution price."""
        if side in ['long', 'buy']:
            return price * (1 + self.slippage_pct)
        else:
            return price * (1 - self.slippage_pct)
    
    def calculate_fees(self, value: float) -> float:
        """Calculate trading fees."""
        return value * self.fee_pct
    
    def open_position(self, symbol: str, side: str, price: float) -> Position:
        """Open a new position."""
        # Check if we can open more positions
        open_positions = [p for p in self.positions if p.status == 'open']
        if len(open_positions) >= self.max_positions:
            logger.warning(f"Max positions reached ({self.max_positions})")
            return None
        
        # Calculate position size
        size = self.calculate_position_size(self.capital, price)
        
        # Apply slippage
        execution_price = self.apply_slippage(price, side)
        
        # Calculate fees
        position_value = size * execution_price
        fees = self.calculate_fees(position_value)
        
        # Check if we have enough capital
        total_cost = position_value + fees
        if total_cost > self.capital:
            logger.warning(f"Insufficient capital: need ${total_cost:.2f}, have ${self.capital:.2f}")
            return None
        
        # Create position
        position = Position(symbol, side, size, execution_price)
        self.positions.append(position)
        
        # Update capital
        self.capital -= total_cost
        
        # Record trade
        trade = {
            'timestamp': datetime.now(),
            'symbol': symbol,
            'side': side,
            'size': size,
            'price': execution_price,
            'fees': fees,
            'type': 'open'
        }
        self.trades.append(trade)
        self.total_trades += 1
        
        logger.info(f"Opened {side} position: {size:.4f} {symbol} @ ${execution_price:.2f}")
        
        return position
    
    def close_position(self, position: Position, price: float) -> float:
        """Close an existing position."""
        if position.status == 'closed':
            logger.warning("Position already closed")
            return 0
        
        # Apply slippage (opposite direction)
        close_side = 'sell' if position.side == 'long' else 'buy'
        execution_price = self.apply_slippage(price, close_side)
        
        # Close position
        pnl = position.close(execution_price)
        
        # Calculate fees
        position_value = position.size * execution_price
        fees = self.calculate_fees(position_value)
        
        # Update capital
        self.capital += position_value - fees
        self.total_pnl += pnl - fees
        
        # Update win/loss stats
        if pnl > fees:
            self.winning_trades += 1
        else:
            self.losing_trades += 1
        
        # Move to closed positions
        self.closed_positions.append(position)
        
        # Record trade
        trade = {
            'timestamp': datetime.now(),
            'symbol': position.symbol,
            'side': close_side,
            'size': position.size,
            'price': execution_price,
            'fees': fees,
            'pnl': pnl - fees,
            'type': 'close'
        }
        self.trades.append(trade)
        
        logger.info(f"Closed position: {position.size:.4f} {position.symbol} @ ${execution_price:.2f}, PnL: ${pnl-fees:.2f}")
        
        return pnl - fees
    
    def update_equity(self, current_prices: Dict[str, float]):
        """Update equity curve with current prices."""
        # Calculate total equity
        total_equity = self.capital
        
        # Add unrealized P&L from open positions
        for position in self.positions:
            if position.status == 'open' and position.symbol in current_prices:
                unrealized = position.unrealized_pnl(current_prices[position.symbol])
                total_equity += position.size * current_prices[position.symbol]
        
        # Update equity curve
        self.equity_curve.append(total_equity)
        self.timestamps.append(datetime.now())
        
        # Update max drawdown
        peak = max(self.equity_curve)
        drawdown = (total_equity - peak) / peak if peak > 0 else 0
        self.max_drawdown = min(self.max_drawdown, drawdown)
        
        return total_equity
    
    def generate_signal(self, symbol: str, timeframe: str = '15m') -> Tuple[int, float]:
        """Generate trading signal using ML model.
        
        Returns:
            Tuple of (signal, probability)
        """
        try:
            result = self.predictor.predict_next(symbol, timeframe)
            
            if result:
                return result['signal'], result['probability']
            else:
                return 0, 0.5
        except Exception as e:
            logger.error(f"Error generating signal: {e}")
            return 0, 0.5
    
    def trading_loop(self, symbols: List[str], interval_seconds: int = 60):
        """Main trading loop.
        
        Args:
            symbols: List of symbols to trade
            interval_seconds: Check interval in seconds
        """
        logger.info(f"Starting trading loop for {symbols}")
        
        while self.is_running:
            try:
                current_prices = {}
                
                for symbol in symbols:
                    # Get current price (simplified - use last close)
                    loader = CryptoDataLoader()
                    df = loader.load_data(
                        symbol=symbol,
                        timeframe='1m',
                        start_date=(datetime.now() - timedelta(hours=1)).strftime('%Y-%m-%d %H:%M'),
                        end_date=datetime.now().strftime('%Y-%m-%d %H:%M')
                    )
                    
                    if not df.empty:
                        current_price = df['close'].iloc[-1]
                        current_prices[symbol] = current_price
                        
                        # Generate signal
                        signal, probability = self.generate_signal(symbol)
                        
                        # Check existing position
                        existing_position = None
                        for pos in self.positions:
                            if pos.symbol == symbol and pos.status == 'open':
                                existing_position = pos
                                break
                        
                        # Trading logic
                        if existing_position:
                            # Check if we should close
                            if (existing_position.side == 'long' and signal <= 0) or \
                               (existing_position.side == 'short' and signal >= 0):
                                self.close_position(existing_position, current_price)
                        else:
                            # Check if we should open
                            if signal == 1:  # Long signal
                                self.open_position(symbol, 'long', current_price)
                            elif signal == -1:  # Short signal
                                self.open_position(symbol, 'short', current_price)
                
                # Update equity
                if current_prices:
                    equity = self.update_equity(current_prices)
                    logger.info(f"Current equity: ${equity:.2f} (PnL: ${equity - self.initial_capital:.2f})")
                
                # Sleep
                time.sleep(interval_seconds)
                
            except Exception as e:
                logger.error(f"Error in trading loop: {e}")
                time.sleep(interval_seconds)
    
    def start(self, symbols: List[str], interval_seconds: int = 60):
        """Start paper trading."""
        if self.is_running:
            logger.warning("Bot already running")
            return
        
        self.is_running = True
        self.trading_thread = threading.Thread(
            target=self.trading_loop,
            args=(symbols, interval_seconds)
        )
        self.trading_thread.start()
        logger.info("Paper trading started")
    
    def stop(self):
        """Stop paper trading."""
        if not self.is_running:
            logger.warning("Bot not running")
            return
        
        self.is_running = False
        if self.trading_thread:
            self.trading_thread.join()
        
        # Close all open positions at current prices
        logger.info("Closing all open positions...")
        for position in self.positions:
            if position.status == 'open':
                # Get last known price (simplified)
                last_price = position.entry_price  # Use entry as fallback
                self.close_position(position, last_price)
        
        logger.info("Paper trading stopped")
    
    def get_performance_stats(self) -> Dict:
        """Get performance statistics."""
        total_trades = len(self.closed_positions)
        
        if total_trades == 0:
            win_rate = 0
            avg_win = 0
            avg_loss = 0
            profit_factor = 0
        else:
            wins = [p.pnl for p in self.closed_positions if p.pnl > 0]
            losses = [abs(p.pnl) for p in self.closed_positions if p.pnl < 0]
            
            win_rate = len(wins) / total_trades if total_trades > 0 else 0
            avg_win = np.mean(wins) if wins else 0
            avg_loss = np.mean(losses) if losses else 0
            
            total_wins = sum(wins) if wins else 0
            total_losses = sum(losses) if losses else 0
            profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')
        
        current_equity = self.equity_curve[-1] if self.equity_curve else self.initial_capital
        
        # Calculate Sharpe (simplified)
        if len(self.equity_curve) > 1:
            returns = pd.Series(self.equity_curve).pct_change().dropna()
            sharpe = returns.mean() / returns.std() * np.sqrt(365) if returns.std() > 0 else 0
        else:
            sharpe = 0
        
        return {
            'total_trades': total_trades,
            'open_positions': len([p for p in self.positions if p.status == 'open']),
            'win_rate': win_rate,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'total_pnl': self.total_pnl,
            'current_equity': current_equity,
            'return_pct': (current_equity / self.initial_capital - 1) * 100,
            'max_drawdown': self.max_drawdown * 100,
            'sharpe_ratio': sharpe
        }
    
    def save_results(self, filepath: str):
        """Save trading results to file."""
        results = {
            'config': {
                'initial_capital': self.initial_capital,
                'position_size_pct': self.position_size_pct,
                'max_positions': self.max_positions,
                'fee_pct': self.fee_pct,
                'slippage_pct': self.slippage_pct
            },
            'performance': self.get_performance_stats(),
            'trades': [
                {
                    'timestamp': t['timestamp'].isoformat(),
                    'symbol': t['symbol'],
                    'side': t['side'],
                    'size': t['size'],
                    'price': t['price'],
                    'fees': t['fees'],
                    'pnl': t.get('pnl', 0),
                    'type': t['type']
                }
                for t in self.trades
            ],
            'positions': [p.to_dict() for p in self.closed_positions],
            'equity_curve': self.equity_curve,
            'timestamps': [t.isoformat() for t in self.timestamps]
        }
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Results saved to {filepath}")


def main():
    """Main function for CLI usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Paper Trading Bot')
    parser.add_argument('--model', required=True, help='Path to trained model')
    parser.add_argument('--symbols', nargs='+', default=['BTCUSDT'], help='Symbols to trade')
    parser.add_argument('--capital', type=float, default=10000, help='Initial capital')
    parser.add_argument('--size', type=float, default=0.1, help='Position size as % of capital')
    parser.add_argument('--interval', type=int, default=60, help='Check interval in seconds')
    parser.add_argument('--duration', type=int, default=3600, help='Duration in seconds')
    parser.add_argument('--output', default='paper_trading_results.json', help='Output file')
    
    args = parser.parse_args()
    
    # Create bot
    bot = PaperTradingBot(
        model_path=args.model,
        initial_capital=args.capital,
        position_size_pct=args.size
    )
    
    try:
        # Start trading
        bot.start(args.symbols, args.interval)
        
        # Run for specified duration
        logger.info(f"Running for {args.duration} seconds...")
        time.sleep(args.duration)
        
        # Stop trading
        bot.stop()
        
        # Print results
        stats = bot.get_performance_stats()
        print("\n" + "="*60)
        print("📊 PAPER TRADING RESULTS")
        print("="*60)
        for key, value in stats.items():
            if isinstance(value, float):
                print(f"{key}: {value:.2f}")
            else:
                print(f"{key}: {value}")
        
        # Save results
        bot.save_results(args.output)
        
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        bot.stop()


if __name__ == "__main__":
    main()