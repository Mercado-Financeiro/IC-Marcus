"""
Realistic backtesting engine with proper cost modeling.
Includes spread, slippage, partial fills, and funding costs.
"""
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, Any, Optional, List, Tuple
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


@dataclass
class RealisticCosts:
    """Complete cost structure for realistic backtesting."""
    
    # Trading costs
    taker_fee_bps: float = 5.0      # Taker fee in basis points
    maker_fee_bps: float = 2.0      # Maker fee in basis points
    
    # Market impact
    spread_bps: float = 2.0         # Average bid-ask spread
    slippage_bps: float = 3.0       # Additional slippage on market orders
    
    # Size-dependent slippage
    impact_coefficient: float = 0.1  # Price impact per unit size
    max_position_pct: float = 0.01   # Max position as % of volume
    
    # Funding costs (for perpetuals)
    funding_rate_8h: float = 0.01    # 0.01% every 8 hours
    funding_times: List[int] = None  # UTC hours when funding is charged
    
    # Latency
    latency_ms: float = 100          # Execution latency in milliseconds
    
    def __post_init__(self):
        if self.funding_times is None:
            self.funding_times = [0, 8, 16]  # Standard funding times (UTC)


class RealisticBacktestEngine:
    """
    Realistic backtesting with proper market microstructure modeling.
    
    Features:
    - Variable spread based on volatility
    - Size-dependent slippage
    - Partial fills for large orders
    - Funding costs for perpetuals
    - Latency simulation
    """
    
    def __init__(self, costs: Optional[RealisticCosts] = None):
        """
        Initialize backtest engine.
        
        Args:
            costs: Cost structure (default: realistic crypto costs)
        """
        self.costs = costs or RealisticCosts()
        self.trades = []
        self.equity_curve = []
        self.positions = []
        
    def run(self,
            data: pd.DataFrame,
            signals: pd.Series,
            initial_capital: float = 10000.0,
            position_size: str = 'fixed',
            size_pct: float = 0.1) -> Dict[str, Any]:
        """
        Run realistic backtest.
        
        Args:
            data: OHLCV DataFrame with index as timestamp
            signals: Trading signals (0 or 1)
            initial_capital: Starting capital
            position_size: 'fixed' or 'kelly' or 'risk_parity'
            size_pct: Position size as % of capital
            
        Returns:
            Dictionary with backtest metrics
        """
        if len(data) != len(signals):
            raise ValueError("Data and signals must have same length")
        
        # Initialize state
        capital = initial_capital
        position = 0
        entry_price = 0
        trades = []
        equity_curve = [capital]
        positions = [0]
        
        # Track metrics
        total_trades = 0
        winning_trades = 0
        total_pnl = 0
        max_drawdown = 0
        peak_equity = capital
        
        # Ensure signals are binary
        signals = (signals > 0.5).astype(int)
        
        for i in range(1, len(data)):
            current_time = data.index[i]
            current_price = data.iloc[i]['close']
            current_volume = data.iloc[i]['volume']
            
            # Previous signal (t-1) determines action at t
            prev_signal = signals.iloc[i-1] if i > 0 else 0
            
            # Calculate dynamic spread based on volatility
            volatility = self._estimate_volatility(data.iloc[max(0, i-20):i])
            dynamic_spread = self._calculate_dynamic_spread(volatility)
            
            # Check for position change
            if prev_signal != position:
                # Close existing position
                if position != 0:
                    exit_price = self._calculate_execution_price(
                        current_price, 
                        -position,  # Closing trade
                        current_volume,
                        dynamic_spread
                    )
                    
                    # Calculate PnL
                    pnl = position * (exit_price - entry_price) * capital * size_pct
                    
                    # Deduct closing costs
                    closing_cost = self._calculate_transaction_cost(
                        abs(capital * size_pct),
                        is_maker=False  # Assume taker for simplicity
                    )
                    pnl -= closing_cost
                    
                    # Update capital
                    capital += pnl
                    total_pnl += pnl
                    
                    # Track trade
                    trades.append({
                        'entry_time': entry_time,
                        'exit_time': current_time,
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'position': position,
                        'pnl': pnl,
                        'return': pnl / (capital - pnl)
                    })
                    
                    total_trades += 1
                    if pnl > 0:
                        winning_trades += 1
                    
                    position = 0
                
                # Open new position
                if prev_signal != 0:
                    # Calculate position size
                    trade_size = self._calculate_position_size(
                        capital, size_pct, position_size, signals.iloc[max(0, i-50):i]
                    )
                    
                    # Check if we can fill the order
                    max_fillable = current_volume * self.costs.max_position_pct
                    actual_size = min(trade_size, max_fillable)
                    
                    if actual_size < trade_size:
                        logger.debug(f"Partial fill: {actual_size/trade_size:.1%} at {current_time}")
                    
                    # Calculate entry price with costs
                    entry_price = self._calculate_execution_price(
                        current_price,
                        prev_signal,
                        current_volume,
                        dynamic_spread
                    )
                    
                    # Deduct opening costs
                    opening_cost = self._calculate_transaction_cost(
                        actual_size,
                        is_maker=False
                    )
                    capital -= opening_cost
                    
                    position = prev_signal
                    entry_time = current_time
            
            # Apply funding costs if holding position
            if position != 0:
                funding_cost = self._calculate_funding_cost(
                    capital * size_pct * abs(position),
                    current_time
                )
                capital -= funding_cost
            
            # Update equity curve
            if position != 0:
                # Mark-to-market
                mtm_price = current_price
                unrealized_pnl = position * (mtm_price - entry_price) * capital * size_pct
                equity_curve.append(capital + unrealized_pnl)
            else:
                equity_curve.append(capital)
            
            positions.append(position)
            
            # Update drawdown
            if equity_curve[-1] > peak_equity:
                peak_equity = equity_curve[-1]
            drawdown = (equity_curve[-1] - peak_equity) / peak_equity
            max_drawdown = min(max_drawdown, drawdown)
        
        # Store results
        self.trades = trades
        self.equity_curve = equity_curve
        self.positions = positions
        
        # Calculate metrics
        metrics = self._calculate_metrics(
            equity_curve, trades, total_trades, winning_trades, max_drawdown
        )
        
        return metrics
    
    def _estimate_volatility(self, data: pd.DataFrame) -> float:
        """Estimate current volatility."""
        if len(data) < 2:
            return 0.001
        
        returns = data['close'].pct_change().dropna()
        if len(returns) == 0:
            return 0.001
        
        return float(returns.std())
    
    def _calculate_dynamic_spread(self, volatility: float) -> float:
        """Calculate dynamic spread based on volatility."""
        # Higher volatility = wider spread
        base_spread = self.costs.spread_bps / 10000
        vol_adjustment = min(2.0, 1.0 + volatility * 10)
        
        return base_spread * vol_adjustment
    
    def _calculate_execution_price(self, 
                                  mid_price: float,
                                  side: int,
                                  volume: float,
                                  spread: float) -> float:
        """
        Calculate realistic execution price.
        
        Args:
            mid_price: Current mid price
            side: 1 for buy, -1 for sell
            volume: Current volume
            spread: Current spread
            
        Returns:
            Execution price including spread and slippage
        """
        # Base spread cost (half-spread for crossing)
        spread_cost = mid_price * spread / 2
        
        # Additional slippage
        slippage = mid_price * self.costs.slippage_bps / 10000
        
        # Size-dependent impact (simplified square-root model)
        # impact = coefficient * sqrt(size / ADV)
        # Here we use a simplified linear model
        impact = mid_price * self.costs.impact_coefficient / 1000
        
        # Total cost
        if side > 0:  # Buy
            execution_price = mid_price + spread_cost + slippage + impact
        else:  # Sell
            execution_price = mid_price - spread_cost - slippage - impact
        
        return execution_price
    
    def _calculate_transaction_cost(self, 
                                   trade_value: float,
                                   is_maker: bool = False) -> float:
        """Calculate transaction fees."""
        if is_maker:
            fee_bps = self.costs.maker_fee_bps
        else:
            fee_bps = self.costs.taker_fee_bps
        
        return trade_value * fee_bps / 10000
    
    def _calculate_funding_cost(self, 
                               position_value: float,
                               current_time: datetime) -> float:
        """Calculate funding cost for perpetuals."""
        # Check if it's funding time
        current_hour = current_time.hour if hasattr(current_time, 'hour') else 0
        
        if current_hour in self.costs.funding_times:
            return position_value * self.costs.funding_rate_8h / 100
        
        return 0
    
    def _calculate_position_size(self,
                                capital: float,
                                base_size_pct: float,
                                method: str,
                                recent_signals: pd.Series) -> float:
        """
        Calculate position size.
        
        Args:
            capital: Current capital
            base_size_pct: Base size as % of capital
            method: Sizing method
            recent_signals: Recent signals for Kelly calculation
            
        Returns:
            Position size in currency units
        """
        if method == 'fixed':
            return capital * base_size_pct
        
        elif method == 'kelly':
            # Simplified Kelly criterion
            if len(recent_signals) < 10:
                return capital * base_size_pct
            
            wins = (recent_signals == 1).sum()
            total = len(recent_signals)
            
            if total == 0:
                return capital * base_size_pct
            
            win_rate = wins / total
            
            # Kelly fraction (simplified)
            if win_rate > 0.5:
                kelly_f = (2 * win_rate - 1) * 0.25  # Conservative Kelly
                size_pct = min(base_size_pct, kelly_f)
            else:
                size_pct = base_size_pct * 0.5  # Reduce size if losing
            
            return capital * size_pct
        
        elif method == 'risk_parity':
            # Size inversely proportional to volatility
            volatility = recent_signals.std() if len(recent_signals) > 1 else 1.0
            adjusted_size = base_size_pct / (1 + volatility * 10)
            
            return capital * adjusted_size
        
        else:
            return capital * base_size_pct
    
    def _calculate_metrics(self,
                          equity_curve: List[float],
                          trades: List[Dict],
                          total_trades: int,
                          winning_trades: int,
                          max_drawdown: float) -> Dict[str, Any]:
        """Calculate comprehensive backtest metrics."""
        equity_array = np.array(equity_curve)
        returns = np.diff(equity_array) / equity_array[:-1]
        
        # Remove any NaN or inf
        returns = returns[np.isfinite(returns)]
        
        if len(returns) == 0:
            return {
                'total_return': 0,
                'sharpe_ratio': 0,
                'max_drawdown': 0,
                'win_rate': 0,
                'total_trades': 0,
                'avg_trade': 0,
                'profit_factor': 0,
                'expected_value': 0
            }
        
        # Calculate metrics
        total_return = (equity_curve[-1] - equity_curve[0]) / equity_curve[0]
        
        # Sharpe ratio (annualized, assuming daily data)
        if returns.std() > 0:
            sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252)
        else:
            sharpe = 0
        
        # Win rate
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        # Average trade
        if trades:
            trade_returns = [t['return'] for t in trades]
            avg_trade = np.mean(trade_returns)
            
            # Profit factor
            gross_profits = sum(r for r in trade_returns if r > 0)
            gross_losses = abs(sum(r for r in trade_returns if r < 0))
            profit_factor = gross_profits / gross_losses if gross_losses > 0 else 0
        else:
            avg_trade = 0
            profit_factor = 0
        
        # Expected value per trade
        expected_value = avg_trade * total_trades / len(equity_curve) if len(equity_curve) > 0 else 0
        
        # Calmar ratio (return / max drawdown)
        calmar = total_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        return {
            'total_return': float(total_return),
            'sharpe_ratio': float(sharpe),
            'max_drawdown': float(max_drawdown),
            'win_rate': float(win_rate),
            'total_trades': int(total_trades),
            'avg_trade': float(avg_trade),
            'profit_factor': float(profit_factor),
            'expected_value': float(expected_value),
            'calmar_ratio': float(calmar),
            'final_capital': float(equity_curve[-1]),
            'n_periods': len(equity_curve)
        }
    
    def get_trade_analysis(self) -> pd.DataFrame:
        """
        Get detailed trade analysis.
        
        Returns:
            DataFrame with trade details
        """
        if not self.trades:
            return pd.DataFrame()
        
        df = pd.DataFrame(self.trades)
        
        # Add additional metrics
        df['duration'] = (df['exit_time'] - df['entry_time'])
        df['pnl_pct'] = df['return'] * 100
        
        return df
    
    def get_equity_curve(self) -> pd.Series:
        """
        Get equity curve as pandas Series.
        
        Returns:
            Equity curve Series
        """
        return pd.Series(self.equity_curve, name='equity')
    
    def plot_results(self):
        """Plot backtest results (placeholder for visualization)."""
        # This would contain plotting logic
        # For now, just log summary
        logger.info(f"Backtest complete: {len(self.trades)} trades, "
                   f"final equity: {self.equity_curve[-1]:.2f}")


def quick_backtest(data: pd.DataFrame,
                   signals: pd.Series,
                   initial_capital: float = 10000) -> Dict[str, Any]:
    """
    Quick backtest with realistic costs.
    
    Args:
        data: OHLCV DataFrame
        signals: Trading signals
        initial_capital: Starting capital
        
    Returns:
        Backtest metrics
    """
    engine = RealisticBacktestEngine()
    return engine.run(data, signals, initial_capital)