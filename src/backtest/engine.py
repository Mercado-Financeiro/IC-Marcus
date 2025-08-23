"""Backtest engine with realistic costs and t+1 execution."""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Union, Tuple
from dataclasses import dataclass
import warnings
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from src.utils.logging_config import get_logger, safe_divide, validate_dataframe

warnings.filterwarnings('ignore')

# Initialize module logger
log = get_logger(__name__)


@dataclass
class BacktestConfig:
    """Configuration for backtest engine."""
    initial_capital: float = 100000
    fee_bps: float = 5.0  # basis points
    slippage_bps: float = 5.0
    funding_apr: float = 0.0  # Annual funding rate
    borrow_apr: float = 0.0  # Borrow cost for shorts
    max_leverage: float = 1.0
    allow_short: bool = True
    position_sizing: str = 'fixed'  # fixed, kelly, volatility_target
    kelly_fraction: float = 0.25
    vol_target: float = 0.15  # 15% annualized
    trading_hours: Optional[Dict] = None
    min_trade_size: float = 100  # Minimum trade size in currency


class BacktestEngine:
    """Engine for backtesting trading strategies with realistic constraints."""
    
    def __init__(
        self,
        initial_capital: float = 100000,
        fee_bps: float = 5.0,
        slippage_bps: float = 5.0,
        funding_apr: float = 0.0,
        borrow_apr: float = 0.0,
        max_leverage: float = 1.0,
        allow_short: bool = True,
        trading_hours: Optional[Dict] = None
    ):
        """Initialize backtest engine.
        
        Args:
            initial_capital: Starting capital
            fee_bps: Transaction fee in basis points
            slippage_bps: Slippage in basis points
            funding_apr: Annual funding rate for leveraged positions
            borrow_apr: Annual borrow rate for short positions
            max_leverage: Maximum leverage allowed
            allow_short: Whether to allow short positions
            trading_hours: Dict with 'start' and 'end' times (HH:MM format)
        """
        self.initial_capital = initial_capital
        self.fee_bps = fee_bps
        self.slippage_bps = slippage_bps
        self.funding_apr = funding_apr
        self.borrow_apr = borrow_apr
        self.max_leverage = max_leverage
        self.allow_short = allow_short
        self.trading_hours = trading_hours
        
        # State variables
        self.capital = initial_capital
        self.position = 0  # Direction: -1, 0, 1
        self.position_size = 0  # Value in currency
        self.position_units = 0  # Number of units (e.g., BTC)
        self.entry_price = 0
        
    def calculate_costs(self, trade_value: float, holding_period: int = 1) -> float:
        """Calculate total costs for a trade.
        
        Args:
            trade_value: Absolute value of the trade
            holding_period: Holding period in bars (for funding/borrow)
            
        Returns:
            Total cost in currency units
        """
        # Transaction costs (fee + slippage)
        transaction_cost = trade_value * (self.fee_bps + self.slippage_bps) / 10000
        
        # Funding cost (pro-rated)
        # Assuming 15-minute bars, 96 bars per day
        bars_per_year = 365 * 24 * 4  # 15-min bars
        funding_cost = trade_value * self.funding_apr * safe_divide(holding_period, bars_per_year, default=0)
        
        # Borrow cost for shorts
        borrow_cost = trade_value * self.borrow_apr * safe_divide(holding_period, bars_per_year, default=0)
        
        return transaction_cost + funding_cost + borrow_cost
    
    def calculate_position_size(
        self,
        method: str = 'fixed',
        capital: float = None,
        signal_strength: float = 1.0,
        volatility: float = 0.02,
        vol_target: float = 0.15,
        kelly_fraction: float = 0.25
    ) -> float:
        """Calculate position size based on method.
        
        Args:
            method: Sizing method (fixed, kelly, volatility_target)
            capital: Current capital
            signal_strength: Signal strength or win probability
            volatility: Current volatility estimate
            vol_target: Target volatility
            kelly_fraction: Fraction of Kelly criterion to use
            
        Returns:
            Position size in currency units
        """
        if capital is None:
            capital = self.capital
        
        if method == 'fixed':
            # Fixed fraction of capital
            size = capital * self.max_leverage * abs(signal_strength)
            
        elif method == 'kelly':
            # Kelly criterion
            # f = (p * b - q) / b
            # where p = win prob, q = loss prob, b = win/loss ratio
            p = signal_strength  # Win probability
            q = 1 - p
            b = 1  # Assume 1:1 win/loss ratio (simplified)
            
            kelly_f = (p * b - q) / b if b > 0 else 0
            kelly_f = max(0, min(kelly_f, 1))  # Bound between 0 and 1
            
            size = capital * kelly_f * kelly_fraction
            
        elif method == 'volatility_target':
            # Size to target specific volatility
            if volatility > 0:
                size = capital * safe_divide(vol_target, volatility, default=0) * abs(signal_strength)
            else:
                size = capital * self.max_leverage * abs(signal_strength)
        else:
            size = capital * self.max_leverage * abs(signal_strength)
        
        # Apply leverage constraint
        size = min(size, capital * self.max_leverage)
        
        return size
    
    def _check_trading_hours(self, timestamp: pd.Timestamp) -> bool:
        """Check if timestamp is within trading hours.
        
        Args:
            timestamp: Timestamp to check
            
        Returns:
            True if within trading hours
        """
        if self.trading_hours is None:
            return True
        
        hour_min = timestamp.strftime('%H:%M')
        start = self.trading_hours.get('start', '00:00')
        end = self.trading_hours.get('end', '23:59')
        
        return start <= hour_min < end
    
    def generate_signals_with_thresholds(
        self,
        probas: np.ndarray,
        threshold_long: float = 0.65,
        threshold_short: float = 0.35,
        mode: str = 'double'
    ) -> np.ndarray:
        """Generate trading signals with configurable thresholds.
        
        Args:
            probas: Predicted probabilities (0 to 1)
            threshold_long: Threshold for long positions
            threshold_short: Threshold for short positions
            mode: 'single' or 'double' threshold mode
            
        Returns:
            Array of signals (-1, 0, 1)
        """
        signals = np.zeros_like(probas)
        
        if mode == 'double':
            # Double threshold with neutral zone
            signals[probas > threshold_long] = 1  # Long
            signals[probas < threshold_short] = -1  # Short
            # Between thresholds = 0 (neutral/flat)
        else:
            # Single threshold (binary)
            signals[probas > threshold_long] = 1
            signals[probas <= threshold_long] = -1
            
        return signals.astype(int)
    
    def run_backtest(self, data: pd.DataFrame, signals: pd.Series) -> pd.DataFrame:
        """Run backtest with t+1 execution.
        
        Args:
            data: OHLCV data
            signals: Trading signals (-1, 0, 1)
            
        Returns:
            DataFrame with backtest results
        """
        # Initialize results
        results = pd.DataFrame(index=data.index)
        results['open'] = data['open']
        results['close'] = data['close']
        results['signal'] = signals
        results['positions'] = 0.0
        results['position_units'] = 0.0  # Track actual units
        results['position_size'] = 0.0  # Track position value in currency
        results['entry_prices'] = np.nan
        results['pnl'] = 0.0
        results['unrealized_pnl'] = 0.0
        results['realized_pnl'] = 0.0
        results['equity'] = float(self.initial_capital)
        results['trades'] = 0
        results['costs'] = 0.0
        results['position_value'] = 0.0
        results['turnover'] = 0.0
        results['total_costs'] = 0.0
        
        # State tracking
        equity = self.initial_capital
        position = 0  # Direction: -1, 0, 1
        position_units = 0  # Actual units held
        position_size = 0  # Position value in currency
        entry_price = 0
        
        for i in range(len(data)):
            current_price = data['close'].iloc[i]
            
            # Calculate unrealized P&L for existing position
            if position_units != 0 and entry_price > 0:
                if position > 0:  # Long position
                    unrealized_pnl = position_units * (current_price - entry_price)
                else:  # Short position
                    unrealized_pnl = -position_units * (current_price - entry_price)
                results.loc[results.index[i], 'unrealized_pnl'] = unrealized_pnl
                results.loc[results.index[i], 'pnl'] = unrealized_pnl
            
            # Execute previous bar's signal at current bar (t+1 execution)
            if i > 0:
                prev_signal = signals.iloc[i-1]
                
                # Check if signal changed and needs execution
                if prev_signal != position:
                    
                    # Check trading hours
                    if not self._check_trading_hours(data.index[i]):
                        continue
                    
                    execution_price = data['open'].iloc[i]
                    
                    # Apply slippage
                    if prev_signal > position:  # Buying
                        execution_price *= (1 + safe_divide(self.slippage_bps, 10000, default=0))
                    elif prev_signal < position:  # Selling
                        execution_price *= (1 - safe_divide(self.slippage_bps, 10000, default=0))
                    
                    # Close existing position
                    if position != 0 and position_units != 0:
                        close_value = abs(position_units * execution_price)
                        close_cost = self.calculate_costs(close_value)
                        equity -= close_cost
                        results.loc[results.index[i], 'costs'] = close_cost
                        results.loc[results.index[i], 'total_costs'] += close_cost
                        
                        # Realize P&L
                        if position > 0:
                            realized_pnl = position_units * (execution_price - entry_price)
                        else:
                            realized_pnl = -position_units * (execution_price - entry_price)
                        
                        # Add realized P&L to equity (THIS IS THE KEY FIX!)
                        equity += realized_pnl
                        results.loc[results.index[i], 'realized_pnl'] = realized_pnl
                        
                        # Reset position size when closing
                        position_size = 0
                    
                    # Open new position
                    if prev_signal != 0:
                        # Calculate position size
                        position_size = self.calculate_position_size(
                            method='fixed',
                            capital=equity,
                            signal_strength=abs(prev_signal)
                        )
                        
                        # Check if short is allowed
                        if prev_signal < 0 and not self.allow_short:
                            position = 0
                            position_units = 0
                            position_size = 0
                        else:
                            # Store position direction
                            position = prev_signal
                            entry_price = execution_price
                            
                            # Calculate position in units (e.g., BTC)
                            position_units = safe_divide(position_size, execution_price, default=0)
                            
                            # Opening costs
                            open_cost = self.calculate_costs(position_size)
                            equity -= open_cost
                            results.loc[results.index[i], 'costs'] += open_cost
                            results.loc[results.index[i], 'total_costs'] += open_cost
                            results.loc[results.index[i], 'entry_prices'] = entry_price
                    else:
                        position = 0
                        position_units = 0
                        position_size = 0
                        entry_price = 0
                    
                    # Record trade
                    results.loc[results.index[i], 'trades'] = 1
                    
                    # Calculate turnover
                    if i > 0:
                        prev_position = results['positions'].iloc[i-1]
                        turnover = abs(position - prev_position)
                        results.loc[results.index[i], 'turnover'] = turnover
            
            # Update position and equity
            results.loc[results.index[i], 'positions'] = position
            results.loc[results.index[i], 'position_units'] = position_units
            results.loc[results.index[i], 'position_size'] = position_size
            results.loc[results.index[i], 'equity'] = equity
            results.loc[results.index[i], 'position_value'] = abs(position_units) * current_price if position_units != 0 else 0
            
            # Assert equity consistency
            assert np.isfinite(equity), f"Non-finite equity at bar {i}: {equity}"
            assert equity >= 0 or self.allow_short, f"Negative equity without shorting at bar {i}: {equity}"
        
        return results
    
    def calculate_metrics(self, results: pd.DataFrame) -> Dict:
        """Calculate performance metrics.
        
        Args:
            results: Backtest results DataFrame
            
        Returns:
            Dictionary of metrics
        """
        # Returns
        returns = results['equity'].pct_change().dropna()
        
        # Basic metrics
        total_return = safe_divide(results['equity'].iloc[-1], self.initial_capital, default=1) - 1
        
        # Annualized metrics (assuming 15-min bars)
        bars_per_year = 365 * 24 * 4
        n_bars = len(results)
        years = n_bars / bars_per_year
        
        annualized_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
        volatility = returns.std() * np.sqrt(bars_per_year)
        
        # Sharpe ratio
        sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
        
        # Sortino ratio
        downside_returns = returns[returns < 0]
        downside_vol = downside_returns.std() * np.sqrt(bars_per_year)
        sortino_ratio = annualized_return / downside_vol if downside_vol > 0 else 0
        
        # Maximum drawdown
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Drawdown duration
        drawdown_start = None
        max_duration = 0
        current_duration = 0
        
        for i, dd in enumerate(drawdown):
            if dd < 0:
                if drawdown_start is None:
                    drawdown_start = i
                current_duration = i - drawdown_start
            else:
                if current_duration > max_duration:
                    max_duration = current_duration
                drawdown_start = None
                current_duration = 0
        
        max_drawdown_duration = max(max_duration, current_duration)
        
        # Calmar ratio
        calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # Win rate
        profitable_trades = (results['pnl'] > 0).sum()
        total_trades = results['trades'].sum()
        win_rate = profitable_trades / total_trades if total_trades > 0 else 0
        
        # Profit factor
        gross_profits = results[results['pnl'] > 0]['pnl'].sum()
        gross_losses = abs(results[results['pnl'] < 0]['pnl'].sum())
        profit_factor = safe_divide(gross_profits, gross_losses, default=0)
        
        # Deflated Sharpe Ratio
        dsr = self.calculate_dsr(sharpe_ratio, n_trials=100, T=n_bars)
        
        # Turnover
        avg_turnover = results['turnover'].mean()
        
        # Total costs
        total_costs = results['total_costs'].sum()
        
        metrics = {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio,
            'max_drawdown': max_drawdown,
            'max_drawdown_duration': max_drawdown_duration,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'dsr': dsr,
            'total_trades': total_trades,
            'turnover': avg_turnover,
            'total_costs': total_costs
        }
        
        return metrics
    
    def optimize_thresholds_for_ev(
        self,
        data: pd.DataFrame,
        probas: np.ndarray,
        threshold_range: Tuple[float, float] = (0.2, 0.8),
        step: float = 0.05,
        min_gap: float = 0.2
    ) -> Dict:
        """Optimize thresholds to maximize expected value.
        
        Args:
            data: OHLCV data
            probas: Predicted probabilities
            threshold_range: Range for threshold search
            step: Step size for grid search
            min_gap: Minimum gap between short and long thresholds
            
        Returns:
            Dict with optimal thresholds and metrics
        """
        best_ev = -np.inf
        best_thresholds = {'long': 0.65, 'short': 0.35}
        best_metrics = {}
        
        # Grid search for optimal thresholds
        short_range = np.arange(threshold_range[0], threshold_range[1] - min_gap, step)
        
        for th_short in short_range:
            long_range = np.arange(th_short + min_gap, threshold_range[1] + step, step)
            
            for th_long in long_range:
                # Generate signals with current thresholds
                signals = self.generate_signals_with_thresholds(
                    probas, th_long, th_short, mode='double'
                )
                
                # Run backtest
                results = self.run_backtest(data, pd.Series(signals, index=data.index))
                metrics = self.calculate_metrics(results)
                
                # Calculate expected value (net return adjusted for risk)
                ev = metrics['annualized_return'] - 0.5 * metrics['volatility']**2
                
                # Penalize for excessive trading
                ev -= 0.01 * metrics['turnover']
                
                # Bonus for good Sharpe
                if metrics['sharpe_ratio'] > 1.0:
                    ev *= 1.1
                
                if ev > best_ev:
                    best_ev = ev
                    best_thresholds = {'long': th_long, 'short': th_short}
                    best_metrics = metrics
                    best_metrics['expected_value'] = ev
                    best_metrics['abstention_rate'] = (signals == 0).mean()
        
        return {
            'thresholds': best_thresholds,
            'metrics': best_metrics,
            'expected_value': best_ev
        }
    
    def calculate_dsr(self, sharpe: float, n_trials: int, T: int) -> float:
        """Calculate Deflated Sharpe Ratio.
        
        Args:
            sharpe: Observed Sharpe ratio
            n_trials: Number of strategies tested
            T: Number of observations
            
        Returns:
            Deflated Sharpe Ratio
        """
        # PSR = Prob[SR > 0] using the formula from Bailey & López de Prado
        # Simplified version
        if T <= 1 or n_trials <= 0:
            return sharpe
        
        # Adjustment factor
        adjustment = np.sqrt((T - 1) / T) * (1 - np.euler_gamma * np.log(n_trials) / T)
        adjustment = max(0.1, adjustment)  # Prevent negative or zero
        
        dsr = sharpe * adjustment
        
        return dsr