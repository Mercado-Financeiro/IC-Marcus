"""
Advanced trading metrics including PSR, DSR, and strategy capacity.

Implements sophisticated performance metrics for trading strategy evaluation.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
from scipy import stats
from scipy.optimize import minimize_scalar
import warnings
import structlog

warnings.filterwarnings('ignore')
log = structlog.get_logger()


class TradingMetrics:
    """Calculate advanced trading performance metrics."""
    
    def __init__(self, risk_free_rate: float = 0.0, trading_days: int = 365):
        """
        Initialize trading metrics calculator.
        
        Args:
            risk_free_rate: Annual risk-free rate
            trading_days: Number of trading days per year (365 for crypto)
        """
        self.risk_free_rate = risk_free_rate
        self.trading_days = trading_days
    
    def sharpe_ratio(
        self,
        returns: Union[pd.Series, np.ndarray],
        periods_per_year: Optional[int] = None
    ) -> float:
        """
        Calculate Sharpe ratio.
        
        Args:
            returns: Return series
            periods_per_year: Periods per year for annualization
            
        Returns:
            Sharpe ratio
        """
        if periods_per_year is None:
            periods_per_year = self.trading_days
        
        excess_returns = returns - self.risk_free_rate / periods_per_year
        
        if len(returns) < 2:
            return 0.0
        
        return np.sqrt(periods_per_year) * excess_returns.mean() / excess_returns.std()
    
    def probabilistic_sharpe_ratio(
        self,
        returns: Union[pd.Series, np.ndarray],
        benchmark_sr: float = 0.0,
        periods_per_year: Optional[int] = None
    ) -> Dict[str, float]:
        """
        Calculate Probabilistic Sharpe Ratio (PSR).
        
        Accounts for uncertainty in Sharpe ratio estimation and 
        adjusts for higher moments (skewness and kurtosis).
        
        Reference: Bailey & López de Prado (2012)
        
        Args:
            returns: Return series
            benchmark_sr: Benchmark Sharpe ratio to test against
            periods_per_year: Periods per year
            
        Returns:
            Dictionary with PSR metrics
        """
        if periods_per_year is None:
            periods_per_year = self.trading_days
        
        n = len(returns)
        if n < 4:
            return {'psr': 0.0, 'sr': 0.0, 'sr_std': np.inf}
        
        # Calculate Sharpe ratio
        sr = self.sharpe_ratio(returns, periods_per_year)
        
        # Calculate higher moments
        skew = stats.skew(returns)
        kurt = stats.kurtosis(returns, fisher=True)  # Excess kurtosis
        
        # Standard error of Sharpe ratio (adjusted for higher moments)
        sr_std = np.sqrt((1 + 0.5 * sr**2 - skew * sr + (kurt / 4) * sr**2) / (n - 1))
        
        # Probabilistic Sharpe Ratio
        psr = stats.norm.cdf((sr - benchmark_sr) / sr_std)
        
        # Confidence intervals
        conf_level = 0.95
        z_score = stats.norm.ppf((1 + conf_level) / 2)
        sr_lower = sr - z_score * sr_std
        sr_upper = sr + z_score * sr_std
        
        return {
            'sharpe_ratio': sr,
            'psr': psr,
            'sr_std': sr_std,
            'sr_lower_95': sr_lower,
            'sr_upper_95': sr_upper,
            'skewness': skew,
            'kurtosis': kurt,
            'n_observations': n
        }
    
    def deflated_sharpe_ratio(
        self,
        returns: Union[pd.Series, np.ndarray],
        n_trials: int = 1,
        n_independent: Optional[int] = None,
        periods_per_year: Optional[int] = None
    ) -> Dict[str, float]:
        """
        Calculate Deflated Sharpe Ratio (DSR).
        
        Adjusts for multiple testing and selection bias.
        
        Reference: Bailey & López de Prado (2014)
        
        Args:
            returns: Return series
            n_trials: Number of strategies tested
            n_independent: Number of independent strategies
            periods_per_year: Periods per year
            
        Returns:
            Dictionary with DSR metrics
        """
        if periods_per_year is None:
            periods_per_year = self.trading_days
        
        # Get PSR metrics
        psr_metrics = self.probabilistic_sharpe_ratio(returns, 0.0, periods_per_year)
        sr = psr_metrics['sharpe_ratio']
        sr_std = psr_metrics['sr_std']
        skew = psr_metrics['skewness']
        kurt = psr_metrics['kurtosis']
        n = len(returns)
        
        # Estimate number of independent trials if not provided
        if n_independent is None:
            # Use Bonferroni correction approximation
            n_independent = min(n_trials, np.exp(1) * np.log(n_trials) if n_trials > 1 else 1)
        
        # Expected maximum Sharpe ratio under null hypothesis
        euler_mascheroni = 0.5772156649
        expected_max_sr = np.sqrt(2 * np.log(n_independent)) - \
                         (euler_mascheroni + np.log(np.sqrt(2 * np.log(n_independent)))) / \
                         (2 * np.sqrt(2 * np.log(n_independent)))
        
        # Variance of maximum Sharpe ratio
        var_max_sr = (np.pi**2 / 6) / (4 * np.log(n_independent))
        
        # Standard deviation adjustment for higher moments
        adj_factor = 1 + 0.5 * sr**2 - skew * sr + (kurt / 4) * sr**2
        
        # Deflated Sharpe Ratio
        dsr = stats.norm.cdf((sr - expected_max_sr * np.sqrt(adj_factor / (n - 1))) / 
                            np.sqrt(var_max_sr * adj_factor / (n - 1)))
        
        return {
            'sharpe_ratio': sr,
            'deflated_sharpe_ratio': dsr,
            'expected_max_sr': expected_max_sr,
            'n_trials': n_trials,
            'n_independent': n_independent,
            'passes_test': dsr > 0.95  # 95% confidence
        }
    
    def sortino_ratio(
        self,
        returns: Union[pd.Series, np.ndarray],
        target_return: float = 0.0,
        periods_per_year: Optional[int] = None
    ) -> float:
        """
        Calculate Sortino ratio.
        
        Args:
            returns: Return series
            target_return: Target return (MAR)
            periods_per_year: Periods per year
            
        Returns:
            Sortino ratio
        """
        if periods_per_year is None:
            periods_per_year = self.trading_days
        
        excess_returns = returns - target_return
        downside_returns = excess_returns[excess_returns < 0]
        
        if len(downside_returns) == 0:
            return np.inf
        
        downside_std = np.sqrt(np.mean(downside_returns**2))
        
        if downside_std == 0:
            return np.inf
        
        return np.sqrt(periods_per_year) * excess_returns.mean() / downside_std
    
    def calmar_ratio(
        self,
        returns: Union[pd.Series, np.ndarray],
        periods_per_year: Optional[int] = None
    ) -> float:
        """
        Calculate Calmar ratio.
        
        Args:
            returns: Return series
            periods_per_year: Periods per year
            
        Returns:
            Calmar ratio
        """
        if periods_per_year is None:
            periods_per_year = self.trading_days
        
        # Calculate cumulative returns
        cum_returns = (1 + returns).cumprod()
        
        # Calculate max drawdown
        running_max = cum_returns.expanding().max()
        drawdown = (cum_returns - running_max) / running_max
        max_dd = drawdown.min()
        
        if max_dd == 0:
            return np.inf
        
        # Annualized return
        total_return = cum_returns.iloc[-1] - 1
        n_periods = len(returns)
        annual_return = (1 + total_return) ** (periods_per_year / n_periods) - 1
        
        return -annual_return / max_dd
    
    def strategy_capacity(
        self,
        returns: Union[pd.Series, np.ndarray],
        volumes: Union[pd.Series, np.ndarray],
        impact_model: str = 'linear',
        participation_rate: float = 0.1
    ) -> Dict[str, float]:
        """
        Estimate strategy capacity.
        
        Estimates the maximum capital that can be deployed before
        market impact degrades performance significantly.
        
        Args:
            returns: Return series
            volumes: Volume series (in base currency)
            impact_model: Price impact model ('linear', 'sqrt', 'power')
            participation_rate: Maximum participation rate
            
        Returns:
            Dictionary with capacity metrics
        """
        # Calculate average daily volume
        avg_volume = volumes.mean()
        median_volume = volumes.median()
        
        # Calculate strategy statistics
        sr = self.sharpe_ratio(returns)
        avg_return = returns.mean()
        volatility = returns.std()
        
        # Estimate turnover
        positions = np.sign(returns)  # Simplified position estimation
        turnover = np.abs(np.diff(positions)).mean() if len(positions) > 1 else 1.0
        
        # Maximum daily capital based on participation
        max_daily_capital = avg_volume * participation_rate
        
        # Estimate capacity based on impact model
        if impact_model == 'linear':
            # Linear impact: impact = lambda * volume
            # Capacity where impact = half of expected return
            lambda_param = 0.0001  # Typical value, should be calibrated
            capacity = (avg_return / 2) * avg_volume / (lambda_param * turnover)
            
        elif impact_model == 'sqrt':
            # Square-root impact: impact = lambda * sqrt(volume)
            lambda_param = 0.001
            capacity = ((avg_return / 2) / lambda_param) ** 2 * avg_volume / turnover
            
        elif impact_model == 'power':
            # Power law impact: impact = lambda * volume^alpha
            lambda_param = 0.0001
            alpha = 0.6  # Typical value between 0.5 and 1
            capacity = ((avg_return / 2) / lambda_param) ** (1/alpha) * avg_volume / turnover
            
        else:
            raise ValueError(f"Unknown impact model: {impact_model}")
        
        # Apply Kelly criterion for optimal sizing
        kelly_fraction = avg_return / (volatility ** 2)
        kelly_capacity = capacity * min(kelly_fraction, 0.25)  # Cap at 25% Kelly
        
        return {
            'estimated_capacity': capacity,
            'kelly_adjusted_capacity': kelly_capacity,
            'max_daily_capital': max_daily_capital,
            'avg_daily_volume': avg_volume,
            'median_daily_volume': median_volume,
            'turnover': turnover,
            'participation_rate': participation_rate,
            'impact_model': impact_model
        }
    
    def rolling_metrics(
        self,
        returns: pd.Series,
        window: int = 252,
        min_periods: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Calculate rolling performance metrics.
        
        Args:
            returns: Return series with datetime index
            window: Rolling window size
            min_periods: Minimum periods required
            
        Returns:
            DataFrame with rolling metrics
        """
        if min_periods is None:
            min_periods = window // 2
        
        rolling = returns.rolling(window=window, min_periods=min_periods)
        
        metrics = pd.DataFrame(index=returns.index)
        
        # Rolling Sharpe
        metrics['rolling_sharpe'] = rolling.apply(
            lambda x: self.sharpe_ratio(x) if len(x) >= min_periods else np.nan
        )
        
        # Rolling Sortino
        metrics['rolling_sortino'] = rolling.apply(
            lambda x: self.sortino_ratio(x) if len(x) >= min_periods else np.nan
        )
        
        # Rolling volatility
        metrics['rolling_volatility'] = rolling.std() * np.sqrt(self.trading_days)
        
        # Rolling return
        metrics['rolling_return'] = rolling.mean() * self.trading_days
        
        # Rolling max drawdown
        def calc_max_dd(x):
            cum_ret = (1 + x).cumprod()
            running_max = cum_ret.expanding().max()
            dd = (cum_ret - running_max) / running_max
            return dd.min()
        
        metrics['rolling_max_drawdown'] = rolling.apply(calc_max_dd)
        
        # Rolling skewness and kurtosis
        metrics['rolling_skewness'] = rolling.skew()
        metrics['rolling_kurtosis'] = rolling.kurt()
        
        return metrics
    
    def underwater_curve(
        self,
        returns: Union[pd.Series, np.ndarray]
    ) -> Union[pd.Series, np.ndarray]:
        """
        Calculate underwater curve (drawdown over time).
        
        Args:
            returns: Return series
            
        Returns:
            Underwater curve
        """
        cum_returns = (1 + returns).cumprod()
        running_max = cum_returns.expanding().max()
        underwater = (cum_returns - running_max) / running_max
        
        return underwater
    
    def tail_metrics(
        self,
        returns: Union[pd.Series, np.ndarray],
        confidence_levels: List[float] = [0.95, 0.99]
    ) -> Dict[str, float]:
        """
        Calculate tail risk metrics (VaR, CVaR, etc.).
        
        Args:
            returns: Return series
            confidence_levels: Confidence levels for VaR/CVaR
            
        Returns:
            Dictionary with tail metrics
        """
        metrics = {}
        
        for conf in confidence_levels:
            # Value at Risk (VaR)
            var = np.percentile(returns, (1 - conf) * 100)
            metrics[f'var_{int(conf*100)}'] = var
            
            # Conditional Value at Risk (CVaR) / Expected Shortfall
            cvar = returns[returns <= var].mean()
            metrics[f'cvar_{int(conf*100)}'] = cvar
        
        # Tail ratio (upside vs downside)
        q95 = np.percentile(returns, 95)
        q05 = np.percentile(returns, 5)
        metrics['tail_ratio'] = abs(q95 / q05) if q05 != 0 else np.inf
        
        # Maximum drawdown duration
        underwater = self.underwater_curve(returns)
        drawdown_periods = (underwater < 0).astype(int)
        
        # Find consecutive drawdown periods
        changes = np.diff(np.concatenate([[0], drawdown_periods, [0]]))
        starts = np.where(changes == 1)[0]
        ends = np.where(changes == -1)[0]
        
        if len(starts) > 0 and len(ends) > 0:
            durations = ends - starts
            metrics['max_drawdown_duration'] = durations.max() if len(durations) > 0 else 0
            metrics['avg_drawdown_duration'] = durations.mean() if len(durations) > 0 else 0
        else:
            metrics['max_drawdown_duration'] = 0
            metrics['avg_drawdown_duration'] = 0
        
        return metrics
    
    def information_ratio(
        self,
        returns: Union[pd.Series, np.ndarray],
        benchmark_returns: Union[pd.Series, np.ndarray],
        periods_per_year: Optional[int] = None
    ) -> float:
        """
        Calculate Information Ratio.
        
        Args:
            returns: Strategy returns
            benchmark_returns: Benchmark returns
            periods_per_year: Periods per year
            
        Returns:
            Information ratio
        """
        if periods_per_year is None:
            periods_per_year = self.trading_days
        
        active_returns = returns - benchmark_returns
        
        if len(active_returns) < 2:
            return 0.0
        
        tracking_error = active_returns.std()
        
        if tracking_error == 0:
            return np.inf if active_returns.mean() > 0 else -np.inf
        
        return np.sqrt(periods_per_year) * active_returns.mean() / tracking_error
    
    def calculate_all_metrics(
        self,
        returns: pd.Series,
        volumes: Optional[pd.Series] = None,
        benchmark_returns: Optional[pd.Series] = None,
        n_trials: int = 1
    ) -> Dict[str, Any]:
        """
        Calculate all trading metrics.
        
        Args:
            returns: Return series
            volumes: Volume series (optional)
            benchmark_returns: Benchmark returns (optional)
            n_trials: Number of strategies tested (for DSR)
            
        Returns:
            Dictionary with all metrics
        """
        metrics = {}
        
        # Basic metrics
        metrics['total_return'] = (1 + returns).prod() - 1
        metrics['annual_return'] = (1 + metrics['total_return']) ** (self.trading_days / len(returns)) - 1
        metrics['volatility'] = returns.std() * np.sqrt(self.trading_days)
        metrics['sharpe_ratio'] = self.sharpe_ratio(returns)
        metrics['sortino_ratio'] = self.sortino_ratio(returns)
        metrics['calmar_ratio'] = self.calmar_ratio(returns)
        
        # Advanced metrics
        psr_metrics = self.probabilistic_sharpe_ratio(returns)
        metrics.update({f'psr_{k}': v for k, v in psr_metrics.items()})
        
        dsr_metrics = self.deflated_sharpe_ratio(returns, n_trials)
        metrics.update({f'dsr_{k}': v for k, v in dsr_metrics.items()})
        
        # Drawdown metrics
        underwater = self.underwater_curve(returns)
        metrics['max_drawdown'] = underwater.min()
        metrics['avg_drawdown'] = underwater[underwater < 0].mean() if any(underwater < 0) else 0
        
        # Tail metrics
        tail_metrics = self.tail_metrics(returns)
        metrics.update(tail_metrics)
        
        # Win rate and profit factor
        winning_returns = returns[returns > 0]
        losing_returns = returns[returns < 0]
        
        metrics['win_rate'] = len(winning_returns) / len(returns) if len(returns) > 0 else 0
        metrics['avg_win'] = winning_returns.mean() if len(winning_returns) > 0 else 0
        metrics['avg_loss'] = losing_returns.mean() if len(losing_returns) > 0 else 0
        
        if metrics['avg_loss'] != 0:
            metrics['profit_factor'] = -winning_returns.sum() / losing_returns.sum()
        else:
            metrics['profit_factor'] = np.inf if winning_returns.sum() > 0 else 0
        
        # Capacity metrics (if volumes provided)
        if volumes is not None:
            capacity_metrics = self.strategy_capacity(returns, volumes)
            metrics.update({f'capacity_{k}': v for k, v in capacity_metrics.items()})
        
        # Information ratio (if benchmark provided)
        if benchmark_returns is not None:
            metrics['information_ratio'] = self.information_ratio(returns, benchmark_returns)
        
        return metrics


def calculate_trading_metrics(
    returns: pd.Series,
    volumes: Optional[pd.Series] = None,
    benchmark_returns: Optional[pd.Series] = None,
    risk_free_rate: float = 0.0,
    trading_days: int = 365,
    n_trials: int = 1
) -> Dict[str, Any]:
    """
    Convenience function to calculate all trading metrics.
    
    Args:
        returns: Return series
        volumes: Volume series (optional)
        benchmark_returns: Benchmark returns (optional)
        risk_free_rate: Annual risk-free rate
        trading_days: Trading days per year
        n_trials: Number of strategies tested
        
    Returns:
        Dictionary with all metrics
    """
    calculator = TradingMetrics(risk_free_rate, trading_days)
    return calculator.calculate_all_metrics(returns, volumes, benchmark_returns, n_trials)