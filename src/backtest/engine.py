"""Backtest engine with realistic costs and t+1 execution."""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Union, Tuple
from dataclasses import dataclass
import warnings
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

warnings.filterwarnings('ignore')


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
        self.position = 0
        self.position_size = 0
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
        funding_cost = trade_value * self.funding_apr * holding_period / bars_per_year
        
        # Borrow cost for shorts
        borrow_cost = trade_value * self.borrow_apr * holding_period / bars_per_year
        
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
                size = capital * (vol_target / volatility) * abs(signal_strength)
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
        results['entry_prices'] = np.nan
        results['pnl'] = 0.0
        results['equity'] = self.initial_capital
        results['trades'] = 0
        results['costs'] = 0.0
        results['position_value'] = 0.0
        results['turnover'] = 0.0
        results['total_costs'] = 0.0
        
        # State tracking
        equity = self.initial_capital
        position = 0
        entry_price = 0
        
        for i in range(len(data)):
            current_price = data['close'].iloc[i]
            
            # Calculate P&L for existing position
            if position != 0 and entry_price > 0:
                if position > 0:
                    pnl = position * (current_price - entry_price)
                else:  # Short position
                    pnl = -position * (current_price - entry_price)
                results.loc[results.index[i], 'pnl'] = pnl
            
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
                        execution_price *= (1 + self.slippage_bps / 10000)
                    elif prev_signal < position:  # Selling
                        execution_price *= (1 - self.slippage_bps / 10000)
                    
                    # Close existing position
                    if position != 0:
                        close_value = abs(position * execution_price)
                        close_cost = self.calculate_costs(close_value)
                        equity -= close_cost
                        results.loc[results.index[i], 'costs'] = close_cost
                        results.loc[results.index[i], 'total_costs'] += close_cost
                        
                        # Realize P&L
                        if position > 0:
                            realized_pnl = position * (execution_price - entry_price)
                        else:
                            realized_pnl = -position * (execution_price - entry_price)
                        equity += realized_pnl
                    
                    # Open new position
                    if prev_signal != 0:
                        # Calculate position size
                        position_value = self.calculate_position_size(
                            method='fixed',
                            capital=equity,
                            signal_strength=abs(prev_signal)
                        )
                        
                        # Check if short is allowed
                        if prev_signal < 0 and not self.allow_short:
                            position = 0
                        else:
                            # Store position as signal direction for simplicity
                            # Actual size is tracked separately
                            position = prev_signal
                            entry_price = execution_price
                            
                            # Opening costs
                            open_cost = self.calculate_costs(position_value)
                            equity -= open_cost
                            results.loc[results.index[i], 'costs'] += open_cost
                            results.loc[results.index[i], 'total_costs'] += open_cost
                            results.loc[results.index[i], 'entry_prices'] = entry_price
                    else:
                        position = 0
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
            results.loc[results.index[i], 'equity'] = equity
            results.loc[results.index[i], 'position_value'] = abs(position) * current_price if position != 0 else 0
        
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
        total_return = (results['equity'].iloc[-1] / self.initial_capital) - 1
        
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
        profit_factor = gross_profits / gross_losses if gross_losses > 0 else 0
        
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