"""
Vectorized backtesting with vectorbt.
Aligned with PRD section 11 - Backtesting requirements.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import vectorbt as vbt
from vectorbt.portfolio import Portfolio
from vectorbt.records import Orders
import structlog
from pathlib import Path

log = structlog.get_logger()


class VectorbtBacktester:
    """
    Vectorized backtester using vectorbt for fast and accurate backtesting.
    Implements PRD requirements:
    - Realistic costs (fees + slippage)
    - Walk-forward analysis
    - Comprehensive metrics
    """
    
    def __init__(
        self,
        fee_bps: float = 10.0,  # 0.1% Binance spot fee
        slippage_bps: float = 10.0,  # 0.1% estimated slippage
        init_cash: float = 10000.0,
        size_type: str = 'percent',  # 'amount' or 'percent'
        size: float = 0.95,  # 95% of available cash per trade
        direction: str = 'longonly',  # 'longonly', 'shortonly', 'both'
        freq: str = '15T'  # 15 minutes
    ):
        """Initialize backtester with configuration."""
        
        self.fee_bps = fee_bps
        self.slippage_bps = slippage_bps
        self.total_cost_bps = fee_bps + slippage_bps
        self.init_cash = init_cash
        self.size_type = size_type
        self.size = size
        self.direction = direction
        self.freq = freq
        
        # Configure vectorbt settings
        vbt.settings.portfolio['init_cash'] = init_cash
        vbt.settings.portfolio['fees'] = self.total_cost_bps / 10000  # Convert bps to decimal
        vbt.settings.portfolio['slippage'] = 0  # Already included in fees
        vbt.settings.portfolio['freq'] = freq
        
    def prepare_signals(
        self,
        predictions: Union[pd.Series, np.ndarray],
        threshold: float = 0.5,
        prices: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Convert predictions to trading signals.
        
        Args:
            predictions: Model predictions (probabilities)
            threshold: Decision threshold
            prices: Price data (for signal alignment)
            
        Returns:
            DataFrame with entry and exit signals
        """
        
        if isinstance(predictions, np.ndarray):
            predictions = pd.Series(predictions)
            
        # Generate binary signals
        signals = (predictions > threshold).astype(int)
        
        # Generate entry/exit signals
        entries = signals.diff() > 0  # Buy when signal turns positive
        exits = signals.diff() < 0    # Sell when signal turns negative
        
        # Handle first signal
        entries.iloc[0] = signals.iloc[0] == 1
        exits.iloc[0] = False
        
        # Align with prices if provided
        if prices is not None:
            entries = entries.reindex(prices.index, fill_value=False)
            exits = exits.reindex(prices.index, fill_value=False)
            
        return pd.DataFrame({
            'entries': entries,
            'exits': exits,
            'signal': signals
        })
    
    def run_backtest(
        self,
        prices: pd.DataFrame,
        signals: pd.DataFrame,
        column: str = 'close'
    ) -> Portfolio:
        """
        Run vectorized backtest.
        
        Args:
            prices: OHLCV price data
            signals: DataFrame with 'entries' and 'exits' columns
            column: Price column to use for execution
            
        Returns:
            Portfolio object with results
        """
        
        # Ensure proper alignment
        prices = prices.copy()
        signals = signals.copy()
        
        # Align indices
        common_index = prices.index.intersection(signals.index)
        prices = prices.loc[common_index]
        signals = signals.loc[common_index]
        
        # Get price series
        price_series = prices[column]
        
        # Create portfolio
        portfolio = vbt.Portfolio.from_signals(
            close=price_series,
            entries=signals['entries'],
            exits=signals['exits'],
            init_cash=self.init_cash,
            fees=self.total_cost_bps / 10000,
            freq=self.freq,
            direction=self.direction
        )
        
        return portfolio
    
    def calculate_metrics(self, portfolio: Portfolio) -> Dict:
        """
        Calculate comprehensive metrics as per PRD.
        
        Returns dict with:
        - Sharpe ratio (annualized)
        - Maximum drawdown
        - Total return
        - Expected value per trade
        - Win rate
        - Number of trades
        - Turnover
        """
        
        try:
            stats = portfolio.stats()
            orders = portfolio.orders.records_readable
            
            # Core metrics
            metrics = {
                # Returns
                'total_return': float(portfolio.total_return()),
                'annual_return': float(portfolio.annualized_return()),
                
                # Risk metrics
                'sharpe_ratio': float(portfolio.sharpe_ratio()),
                'sortino_ratio': float(portfolio.sortino_ratio()),
                'calmar_ratio': float(portfolio.calmar_ratio()),
                'max_drawdown': float(portfolio.max_drawdown()),
                
                # Trade metrics
                'n_trades': len(orders),
                'win_rate': float(portfolio.win_rate()) if len(orders) > 0 else 0,
                'avg_win': float(portfolio.avg_win()) if len(orders) > 0 else 0,
                'avg_loss': float(portfolio.avg_loss()) if len(orders) > 0 else 0,
                'profit_factor': float(portfolio.profit_factor()) if len(orders) > 0 else 0,
                
                # Expected value per trade (PRD critical metric)
                'ev_per_trade': float(portfolio.total_profit() / max(1, len(orders))),
                'ev_after_costs': float((portfolio.total_profit() - portfolio.total_fees()) / max(1, len(orders))),
                
                # Costs
                'total_fees': float(portfolio.total_fees()),
                'fee_per_trade': float(portfolio.total_fees() / max(1, len(orders))),
                
                # Additional
                'final_value': float(portfolio.final_value()),
                'total_profit': float(portfolio.total_profit()),
            }
            
            # Add turnover (trades per period)
            if len(portfolio.wrapper.index) > 0:
                days = (portfolio.wrapper.index[-1] - portfolio.wrapper.index[0]).days
                if days > 0:
                    metrics['daily_turnover'] = len(orders) / days
                else:
                    metrics['daily_turnover'] = 0
            else:
                metrics['daily_turnover'] = 0
                
        except Exception as e:
            log.error(f"Error calculating metrics: {e}")
            metrics = self._get_default_metrics()
            
        return metrics
    
    def _get_default_metrics(self) -> Dict:
        """Return default metrics when calculation fails."""
        return {
            'total_return': 0.0,
            'annual_return': 0.0,
            'sharpe_ratio': 0.0,
            'sortino_ratio': 0.0,
            'calmar_ratio': 0.0,
            'max_drawdown': 0.0,
            'n_trades': 0,
            'win_rate': 0.0,
            'avg_win': 0.0,
            'avg_loss': 0.0,
            'profit_factor': 0.0,
            'ev_per_trade': 0.0,
            'ev_after_costs': 0.0,
            'total_fees': 0.0,
            'fee_per_trade': 0.0,
            'final_value': self.init_cash,
            'total_profit': 0.0,
            'daily_turnover': 0.0
        }
    
    def walk_forward_backtest(
        self,
        prices: pd.DataFrame,
        predictions: pd.Series,
        window_size: int = 252,  # ~1 year for daily, ~15 days for 15min
        test_size: int = 63,     # ~3 months for daily, ~4 days for 15min
        threshold: float = 0.5
    ) -> Tuple[List[Portfolio], pd.DataFrame]:
        """
        Walk-forward backtest as per PRD section 11.
        
        Returns:
            Tuple of (list of portfolios, combined metrics)
        """
        
        portfolios = []
        all_metrics = []
        
        # Generate walk-forward windows
        total_len = len(prices)
        
        for start_idx in range(0, total_len - window_size - test_size, test_size):
            # Define window
            train_end = start_idx + window_size
            test_end = train_end + test_size
            
            # Get test data
            test_prices = prices.iloc[train_end:test_end]
            test_predictions = predictions.iloc[train_end:test_end]
            
            # Generate signals for test period
            test_signals = self.prepare_signals(
                test_predictions,
                threshold,
                test_prices
            )
            
            # Run backtest on test period
            portfolio = self.run_backtest(test_prices, test_signals)
            portfolios.append(portfolio)
            
            # Calculate metrics
            metrics = self.calculate_metrics(portfolio)
            metrics['window'] = len(portfolios)
            metrics['start_date'] = test_prices.index[0]
            metrics['end_date'] = test_prices.index[-1]
            all_metrics.append(metrics)
            
            log.info(
                f"Window {len(portfolios)}: "
                f"Sharpe={metrics['sharpe_ratio']:.2f}, "
                f"Return={metrics['total_return']:.2%}, "
                f"Trades={metrics['n_trades']}"
            )
            
        # Combine metrics
        metrics_df = pd.DataFrame(all_metrics)
        
        # Calculate aggregate statistics
        summary = {
            'avg_sharpe': metrics_df['sharpe_ratio'].mean(),
            'std_sharpe': metrics_df['sharpe_ratio'].std(),
            'avg_return': metrics_df['total_return'].mean(),
            'avg_max_dd': metrics_df['max_drawdown'].mean(),
            'total_trades': metrics_df['n_trades'].sum(),
            'avg_win_rate': metrics_df['win_rate'].mean(),
            'avg_ev_after_costs': metrics_df['ev_after_costs'].mean(),
        }
        
        log.info(f"Walk-forward summary: {summary}")
        
        return portfolios, metrics_df
    
    def plot_results(
        self,
        portfolio: Portfolio,
        save_path: Optional[str] = None
    ) -> None:
        """Generate comprehensive plots."""
        
        try:
            # Create subplots
            fig = portfolio.plot(subplots=[
                'cum_returns',
                'orders',
                'trade_pnl',
                'drawdowns',
                'underwater'
            ])
            
            if save_path:
                fig.write_html(save_path)
                log.info(f"Saved plot to {save_path}")
            else:
                fig.show()
                
        except Exception as e:
            log.error(f"Error plotting results: {e}")


def backtest_strategy(
    prices: pd.DataFrame,
    predictions: pd.Series,
    threshold: float = 0.5,
    fee_bps: float = 10.0,
    slippage_bps: float = 10.0,
    plot: bool = False
) -> Dict:
    """
    Quick backtest function for notebooks.
    
    Args:
        prices: OHLCV data
        predictions: Model predictions (probabilities)
        threshold: Decision threshold
        fee_bps: Trading fee in basis points
        slippage_bps: Slippage in basis points
        plot: Whether to plot results
        
    Returns:
        Dictionary with backtest metrics
    """
    
    # Initialize backtester
    backtester = VectorbtBacktester(
        fee_bps=fee_bps,
        slippage_bps=slippage_bps
    )
    
    # Prepare signals
    signals = backtester.prepare_signals(predictions, threshold, prices)
    
    # Run backtest
    portfolio = backtester.run_backtest(prices, signals)
    
    # Calculate metrics
    metrics = backtester.calculate_metrics(portfolio)
    
    # Plot if requested
    if plot:
        backtester.plot_results(portfolio)
        
    # Log summary
    log.info(
        f"Backtest complete: "
        f"Return={metrics['total_return']:.2%}, "
        f"Sharpe={metrics['sharpe_ratio']:.2f}, "
        f"MaxDD={metrics['max_drawdown']:.2%}, "
        f"Trades={metrics['n_trades']}, "
        f"EV/trade={metrics['ev_after_costs']:.4f}"
    )
    
    return metrics


if __name__ == "__main__":
    # Example usage
    import sys
    
    # Generate sample data
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=1000, freq='15T')
    
    # Simulate prices
    returns = np.random.randn(len(dates)) * 0.01
    prices = pd.DataFrame({
        'close': 100 * np.exp(np.cumsum(returns)),
        'open': 100 * np.exp(np.cumsum(returns + np.random.randn(len(dates)) * 0.005)),
        'high': 100 * np.exp(np.cumsum(returns + np.abs(np.random.randn(len(dates)) * 0.005))),
        'low': 100 * np.exp(np.cumsum(returns - np.abs(np.random.randn(len(dates)) * 0.005))),
        'volume': np.random.uniform(1000, 10000, len(dates))
    }, index=dates)
    
    # Simulate predictions (with some signal)
    predictions = pd.Series(
        np.random.beta(2, 2, len(dates)),  # Beta distribution for probabilities
        index=dates
    )
    
    # Run backtest
    metrics = backtest_strategy(
        prices,
        predictions,
        threshold=0.5,
        fee_bps=10,
        slippage_bps=10,
        plot=False
    )
    
    # Print results
    print("\nBacktest Results:")
    print("-" * 50)
    for key, value in metrics.items():
        if isinstance(value, float):
            if 'return' in key or 'rate' in key or 'ratio' in key:
                print(f"{key:20s}: {value:>10.2%}")
            else:
                print(f"{key:20s}: {value:>10.4f}")
        else:
            print(f"{key:20s}: {value:>10}")