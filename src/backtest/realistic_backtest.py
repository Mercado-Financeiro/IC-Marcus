"""
Realistic Backtest with Market Impact and Execution Costs

Implements a production-grade backtester with:
- Almgren-Chriss market impact model (simplified)
- Variable slippage based on volatility
- Funding rates for perpetual futures
- Comprehensive risk metrics including DSR

This is where strategy meets reality.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass, field
import warnings
from datetime import datetime, timedelta

from src.metrics.dsr import (
    calculate_all_sharpe_metrics, 
    sharpe_ratio,
    calculate_max_drawdown
)
from src.models.threshold_optimizer import TradingCosts
from src.utils.logging import log as logger


@dataclass
class ExecutionConfig:
    """Configuration for realistic execution modeling."""
    
    # Basic costs
    trading_costs: TradingCosts = field(default_factory=TradingCosts)
    
    # Market impact parameters (Almgren-Chriss simplified)
    temporary_impact_factor: float = 0.1  # λ in basis points per sqrt(volume/ADV)
    permanent_impact_factor: float = 0.05  # α for permanent price impact
    
    # Slippage model
    base_slippage_bps: float = 5.0
    volatility_slippage_multiplier: float = 2.0  # Slippage increases with volatility
    
    # Funding rates (for perpetuals)
    funding_interval_hours: int = 8
    avg_funding_rate: float = 0.0001  # 0.01% per 8 hours typical
    
    # Execution constraints
    max_position_size: float = 1.0  # Maximum position as fraction of portfolio
    min_trade_size: float = 0.001  # Minimum trade size
    max_daily_turnover: float = 2.0  # Maximum daily turnover
    
    # Risk limits
    max_drawdown_limit: float = 0.20  # 20% max drawdown before stopping
    position_limit_per_signal: float = 0.1  # Max 10% per signal
    
    # Latency
    execution_delay_bars: int = 1  # Delay between signal and execution


@dataclass
class BacktestResults:
    """Container for backtest results."""
    
    # Returns
    returns: pd.Series
    equity_curve: pd.Series
    positions: pd.Series
    signals: pd.Series
    
    # Metrics
    total_return: float
    annualized_return: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    max_drawdown: float
    
    # DSR metrics
    dsr: float
    psr: float
    
    # Trading statistics
    n_trades: int
    win_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    turnover: float
    
    # Cost breakdown
    total_fees: float
    total_slippage: float
    total_impact: float
    total_funding: float
    
    # Metadata
    metadata: Dict


class RealisticBacktest:
    """
    Production-grade backtester with realistic execution modeling.
    
    This backtester accounts for all the frictions that eat returns:
    market impact, variable slippage, funding rates, and execution delays.
    """
    
    def __init__(
        self,
        config: Optional[ExecutionConfig] = None,
        n_trials: int = 1  # For DSR calculation
    ):
        """
        Initialize backtester.
        
        Args:
            config: Execution configuration
            n_trials: Number of strategies tested (for DSR)
        """
        self.config = config or ExecutionConfig()
        self.n_trials = n_trials
        self.last_results = None
        
    def calculate_market_impact(
        self,
        trade_size: float,
        volume: float,
        adv: float,
        volatility: float
    ) -> Tuple[float, float]:
        """
        Calculate market impact using simplified Almgren-Chriss model.
        
        Impact = temporary + permanent components
        
        Args:
            trade_size: Size of trade (fraction of ADV)
            volume: Current volume
            adv: Average daily volume
            volatility: Current volatility
            
        Returns:
            (temporary_impact, permanent_impact) in percentage
        """
        # Participation rate
        participation = abs(trade_size) / adv if adv > 0 else 0.1
        
        # Temporary impact (square-root model)
        temp_impact = self.config.temporary_impact_factor * np.sqrt(participation) * volatility
        
        # Permanent impact (linear model)
        perm_impact = self.config.permanent_impact_factor * participation
        
        # Convert from bps to percentage
        temp_impact_pct = temp_impact / 10000
        perm_impact_pct = perm_impact / 10000
        
        return temp_impact_pct, perm_impact_pct
    
    def calculate_slippage(
        self,
        volatility: float,
        volume_ratio: float = 1.0
    ) -> float:
        """
        Calculate variable slippage based on market conditions.
        
        Args:
            volatility: Current volatility (annualized)
            volume_ratio: Current volume / average volume
            
        Returns:
            Slippage in percentage
        """
        # Base slippage
        base = self.config.base_slippage_bps / 10000
        
        # Adjust for volatility (higher vol = higher slippage)
        vol_adjustment = self.config.volatility_slippage_multiplier * volatility / 0.20  # Normalized to 20% annual vol
        
        # Adjust for volume (lower volume = higher slippage)
        volume_adjustment = 1.0 / np.sqrt(max(volume_ratio, 0.1))
        
        total_slippage = base * vol_adjustment * volume_adjustment
        
        return total_slippage
    
    def calculate_funding_cost(
        self,
        position: float,
        funding_rate: float,
        hours_held: float
    ) -> float:
        """
        Calculate funding cost for perpetual futures.
        
        Args:
            position: Position size
            funding_rate: Current funding rate
            hours_held: Hours position was held
            
        Returns:
            Funding cost as percentage of position
        """
        if position == 0:
            return 0.0
        
        # Number of funding periods
        n_periods = hours_held / self.config.funding_interval_hours
        
        # Total funding cost
        funding_cost = abs(position) * funding_rate * n_periods
        
        return funding_cost
    
    def run(
        self,
        signals: pd.Series,
        prices: pd.Series,
        volumes: Optional[pd.Series] = None,
        volatility: Optional[pd.Series] = None,
        funding_rates: Optional[pd.Series] = None,
        initial_capital: float = 1.0
    ) -> BacktestResults:
        """
        Run realistic backtest with all frictions.
        
        Args:
            signals: Trading signals (0 or 1, or -1/0/1 for long/short)
            prices: Price series
            volumes: Volume series (optional)
            volatility: Volatility series (optional)
            funding_rates: Funding rate series (optional)
            initial_capital: Starting capital
            
        Returns:
            BacktestResults with comprehensive metrics
        """
        logger.info("Starting realistic backtest", n_bars=len(signals))
        
        # Initialize arrays
        n = len(signals)
        positions = np.zeros(n)
        equity = np.zeros(n)
        returns = np.zeros(n)
        costs = np.zeros(n)
        
        # Cost breakdown
        fees = np.zeros(n)
        slippage = np.zeros(n)
        impact = np.zeros(n)
        funding = np.zeros(n)
        
        # Calculate returns
        price_returns = prices.pct_change().fillna(0)
        
        # Default volatility if not provided (20-day rolling std)
        if volatility is None:
            volatility = price_returns.rolling(20).std().fillna(0.01) * np.sqrt(252)
        
        # Default volume if not provided
        if volumes is None:
            volumes = pd.Series(1.0, index=signals.index)
        adv = volumes.rolling(20).mean().fillna(volumes.mean())
        
        # Default funding if not provided
        if funding_rates is None:
            funding_rates = pd.Series(self.config.avg_funding_rate, index=signals.index)
        
        # Tracking variables
        current_position = 0
        current_equity = initial_capital
        n_trades = 0
        last_trade_bar = -self.config.execution_delay_bars
        
        for i in range(n):
            # Apply execution delay
            signal_bar = i - self.config.execution_delay_bars
            if signal_bar < 0:
                signal = 0
            else:
                signal = signals.iloc[signal_bar]
            
            # Check if we should trade
            target_position = signal * min(
                self.config.position_limit_per_signal,
                self.config.max_position_size
            )
            
            trade_size = target_position - current_position
            
            # Apply minimum trade size filter
            if abs(trade_size) < self.config.min_trade_size:
                trade_size = 0
            
            # Calculate costs if trading
            if trade_size != 0 and i > last_trade_bar + 1:
                n_trades += 1
                last_trade_bar = i
                
                # Market impact
                temp_impact, perm_impact = self.calculate_market_impact(
                    trade_size,
                    volumes.iloc[i] if i < len(volumes) else 1.0,
                    adv.iloc[i] if i < len(adv) else 1.0,
                    volatility.iloc[i] if i < len(volatility) else 0.01
                )
                impact[i] = (temp_impact + perm_impact) * abs(trade_size)
                
                # Slippage
                slippage[i] = self.calculate_slippage(
                    volatility.iloc[i] if i < len(volatility) else 0.01,
                    volumes.iloc[i] / adv.iloc[i] if i < len(adv) and adv.iloc[i] > 0 else 1.0
                ) * abs(trade_size)
                
                # Trading fees
                fees[i] = self.config.trading_costs.total_pct * abs(trade_size)
                
                # Total cost
                costs[i] = fees[i] + slippage[i] + impact[i]
                
                # Update position
                current_position = target_position
            
            # Calculate funding if holding position
            if current_position != 0 and i > 0:
                hours_held = 24 / len(signals) * 365  # Approximate hours per bar
                funding[i] = self.calculate_funding_cost(
                    current_position,
                    funding_rates.iloc[i] if i < len(funding_rates) else self.config.avg_funding_rate,
                    hours_held
                )
                costs[i] += funding[i]
            
            # Calculate returns
            if current_position != 0:
                position_return = current_position * price_returns.iloc[i] if i < len(price_returns) else 0
                returns[i] = position_return - costs[i]
            else:
                returns[i] = -costs[i]
            
            # Update equity
            current_equity = current_equity * (1 + returns[i])
            equity[i] = current_equity
            positions[i] = current_position
            
            # Risk management: stop if max drawdown exceeded
            if i > 20:
                recent_peak = np.max(equity[max(0, i-252):i+1])
                current_dd = (current_equity - recent_peak) / recent_peak
                if current_dd < -self.config.max_drawdown_limit:
                    logger.warning(f"Max drawdown limit hit at bar {i}: {current_dd:.2%}")
                    current_position = 0  # Close position
        
        # Convert to pandas
        returns_series = pd.Series(returns, index=signals.index)
        equity_series = pd.Series(equity, index=signals.index)
        positions_series = pd.Series(positions, index=signals.index)
        
        # Calculate metrics
        total_return = (equity[-1] / initial_capital - 1) if len(equity) > 0 else 0
        
        # Sharpe metrics (with DSR)
        sharpe_metrics = calculate_all_sharpe_metrics(
            returns_series[returns_series != 0],  # Remove zero returns
            n_trials=self.n_trials,
            periods_per_year=252
        )
        
        # Sortino ratio (downside deviation)
        downside_returns = returns_series[returns_series < 0]
        sortino = (
            np.mean(returns_series) / np.std(downside_returns) * np.sqrt(252)
            if len(downside_returns) > 0 and np.std(downside_returns) > 0
            else 0
        )
        
        # Calmar ratio
        max_dd = calculate_max_drawdown(returns_series)
        calmar = (
            np.mean(returns_series) * 252 / abs(max_dd)
            if max_dd < 0
            else 0
        )
        
        # Trading statistics
        winning_trades = returns_series[returns_series > 0]
        losing_trades = returns_series[returns_series < 0]
        
        win_rate = len(winning_trades) / n_trades if n_trades > 0 else 0
        avg_win = np.mean(winning_trades) if len(winning_trades) > 0 else 0
        avg_loss = np.mean(losing_trades) if len(losing_trades) > 0 else 0
        
        profit_factor = (
            abs(np.sum(winning_trades) / np.sum(losing_trades))
            if len(losing_trades) > 0 and np.sum(losing_trades) != 0
            else np.inf if len(winning_trades) > 0 else 0
        )
        
        # Turnover
        position_changes = np.abs(np.diff(positions))
        turnover = np.sum(position_changes) / len(positions) * 252  # Annualized
        
        # Create results
        results = BacktestResults(
            returns=returns_series,
            equity_curve=equity_series,
            positions=positions_series,
            signals=signals,
            total_return=total_return,
            annualized_return=total_return * 252 / len(returns_series),
            sharpe_ratio=sharpe_metrics.sharpe_ratio,
            sortino_ratio=sortino,
            calmar_ratio=calmar,
            max_drawdown=max_dd,
            dsr=sharpe_metrics.dsr,
            psr=sharpe_metrics.psr,
            n_trades=n_trades,
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            profit_factor=profit_factor,
            turnover=turnover,
            total_fees=np.sum(fees),
            total_slippage=np.sum(slippage),
            total_impact=np.sum(impact),
            total_funding=np.sum(funding),
            metadata={
                'n_bars': len(signals),
                'total_costs': np.sum(costs),
                'costs_per_trade': np.sum(costs) / n_trades if n_trades > 0 else 0,
                'execution_config': self.config
            }
        )
        
        self.last_results = results
        
        logger.info(
            "Backtest complete",
            total_return=f"{total_return:.2%}",
            sharpe=f"{sharpe_metrics.sharpe_ratio:.2f}",
            dsr=f"{sharpe_metrics.dsr:.2f}",
            max_dd=f"{max_dd:.2%}",
            n_trades=n_trades,
            total_costs=f"{np.sum(costs):.2%}"
        )
        
        return results
    
    def compare_with_buy_hold(
        self,
        prices: pd.Series,
        results: Optional[BacktestResults] = None
    ) -> Dict:
        """
        Compare strategy with buy & hold.
        
        Args:
            prices: Price series
            results: Backtest results to compare
            
        Returns:
            Comparison metrics
        """
        if results is None:
            results = self.last_results
            if results is None:
                raise ValueError("No results to compare")
        
        # Buy & hold returns
        bh_returns = prices.pct_change().fillna(0)
        bh_total_return = (prices.iloc[-1] / prices.iloc[0] - 1)
        
        # Buy & hold with costs (entry and exit only)
        bh_costs = self.config.trading_costs.total_pct
        bh_net_return = bh_total_return - bh_costs
        
        # Buy & hold Sharpe
        bh_sharpe = sharpe_ratio(bh_returns)
        
        # Buy & hold max drawdown
        bh_cumulative = (1 + bh_returns).cumprod()
        bh_running_max = np.maximum.accumulate(bh_cumulative)
        bh_drawdown = (bh_cumulative - bh_running_max) / bh_running_max
        bh_max_dd = np.min(bh_drawdown)
        
        comparison = {
            'strategy': {
                'total_return': results.total_return,
                'sharpe': results.sharpe_ratio,
                'dsr': results.dsr,
                'max_dd': results.max_drawdown,
                'n_trades': results.n_trades
            },
            'buy_hold': {
                'total_return': bh_net_return,
                'sharpe': bh_sharpe,
                'max_dd': bh_max_dd,
                'n_trades': 2  # Entry and exit
            },
            'outperformance': {
                'return': results.total_return - bh_net_return,
                'sharpe': results.sharpe_ratio - bh_sharpe,
                'max_dd': results.max_drawdown - bh_max_dd  # Less negative is better
            }
        }
        
        logger.info(
            "Strategy vs Buy & Hold",
            strategy_return=f"{results.total_return:.2%}",
            bh_return=f"{bh_net_return:.2%}",
            outperformance=f"{comparison['outperformance']['return']:.2%}"
        )
        
        return comparison
    
    def generate_report(
        self,
        results: Optional[BacktestResults] = None,
        save_path: Optional[str] = None
    ) -> str:
        """
        Generate comprehensive backtest report.
        
        Args:
            results: Results to report
            save_path: Path to save report
            
        Returns:
            Report as string
        """
        if results is None:
            results = self.last_results
            if results is None:
                raise ValueError("No results to report")
        
        report = []
        report.append("="*60)
        report.append("REALISTIC BACKTEST REPORT")
        report.append("="*60)
        
        # Performance metrics
        report.append("\n## PERFORMANCE METRICS")
        report.append(f"Total Return: {results.total_return:.2%}")
        report.append(f"Annualized Return: {results.annualized_return:.2%}")
        report.append(f"Sharpe Ratio: {results.sharpe_ratio:.3f}")
        report.append(f"DSR (Deflated): {results.dsr:.3f}")
        report.append(f"PSR (Probabilistic): {results.psr:.3f}")
        report.append(f"Sortino Ratio: {results.sortino_ratio:.3f}")
        report.append(f"Calmar Ratio: {results.calmar_ratio:.3f}")
        report.append(f"Max Drawdown: {results.max_drawdown:.2%}")
        
        # Trading statistics
        report.append("\n## TRADING STATISTICS")
        report.append(f"Number of Trades: {results.n_trades}")
        report.append(f"Win Rate: {results.win_rate:.2%}")
        report.append(f"Average Win: {results.avg_win:.4%}")
        report.append(f"Average Loss: {results.avg_loss:.4%}")
        report.append(f"Profit Factor: {results.profit_factor:.2f}")
        report.append(f"Annual Turnover: {results.turnover:.1f}x")
        
        # Cost breakdown
        report.append("\n## COST BREAKDOWN")
        report.append(f"Total Fees: {results.total_fees:.4%}")
        report.append(f"Total Slippage: {results.total_slippage:.4%}")
        report.append(f"Total Market Impact: {results.total_impact:.4%}")
        report.append(f"Total Funding: {results.total_funding:.4%}")
        report.append(f"Total All Costs: {results.metadata['total_costs']:.4%}")
        report.append(f"Cost per Trade: {results.metadata['costs_per_trade']:.4%}")
        
        report.append("\n" + "="*60)
        
        report_str = "\n".join(report)
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report_str)
            logger.info(f"Report saved to {save_path}")
        
        return report_str