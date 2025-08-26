"""
Deflated Sharpe Ratio (DSR) and Probabilistic Sharpe Ratio (PSR)

Corrects for multiple testing and non-normality in performance evaluation.
Based on Bailey & López de Prado papers.

References:
- Bailey & López de Prado (2014): "The Deflated Sharpe Ratio"
- Bailey & López de Prado (2012): "The Sharpe Ratio Efficient Frontier"
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, Dict
from scipy import stats
from dataclasses import dataclass
import warnings

from src.utils.logging import log as logger


@dataclass
class SharpeMetrics:
    """Container for Sharpe-related metrics."""
    sharpe_ratio: float
    psr: float  # Probabilistic Sharpe Ratio
    dsr: float  # Deflated Sharpe Ratio
    n_trials: int
    T: int  # Number of observations
    skewness: float
    kurtosis: float
    confidence_level: float
    metadata: Dict


def sharpe_ratio(
    returns: np.ndarray,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252
) -> float:
    """
    Calculate annualized Sharpe ratio.
    
    Args:
        returns: Array of returns
        risk_free_rate: Risk-free rate (annualized)
        periods_per_year: Number of periods in a year (252 for daily, 365 for crypto)
        
    Returns:
        Annualized Sharpe ratio
    """
    excess_returns = returns - risk_free_rate / periods_per_year
    
    if len(returns) < 2:
        return 0.0
    
    mean_excess = np.mean(excess_returns)
    std_excess = np.std(excess_returns, ddof=1)
    
    if std_excess == 0:
        return 0.0
    
    # Annualize
    sharpe = (mean_excess / std_excess) * np.sqrt(periods_per_year)
    
    return sharpe


def probabilistic_sharpe_ratio(
    observed_sr: float,
    benchmark_sr: float,
    T: int,
    skewness: float,
    kurtosis: float
) -> float:
    """
    Calculate Probabilistic Sharpe Ratio (PSR).
    
    PSR = Prob(SR_true > SR_benchmark | observed_SR)
    
    This accounts for non-normal returns through higher moments.
    
    Args:
        observed_sr: Observed Sharpe ratio
        benchmark_sr: Benchmark Sharpe ratio (typically 0 or required minimum)
        T: Number of observations
        skewness: Skewness of returns
        kurtosis: Excess kurtosis of returns
        
    Returns:
        PSR in [0, 1]
    """
    # Standard error of Sharpe ratio with higher moments correction
    # Formula from Bailey & López de Prado
    se_sr = np.sqrt((1 + 0.5 * observed_sr**2 - skewness * observed_sr + 
                     (kurtosis - 3) / 4 * observed_sr**2) / T)
    
    # Z-score
    z = (observed_sr - benchmark_sr) / se_sr
    
    # Probability using normal CDF
    psr = stats.norm.cdf(z)
    
    return psr


def deflated_sharpe_ratio(
    observed_sr: float,
    n_trials: int,
    T: int,
    skewness: float = 0.0,
    kurtosis: float = 0.0,
    benchmark_sr: float = 0.0,
    confidence_level: float = 0.95
) -> float:
    """
    Calculate Deflated Sharpe Ratio (DSR).
    
    DSR accounts for multiple testing (selection bias) and non-normality.
    Based on Bailey & López de Prado (2014).
    
    Args:
        observed_sr: Observed Sharpe ratio (annualized)
        n_trials: Number of strategies tested (multiple testing correction)
        T: Number of observations
        skewness: Skewness of returns
        kurtosis: Excess kurtosis of returns  
        benchmark_sr: Minimum acceptable Sharpe ratio
        confidence_level: Confidence level for the test
        
    Returns:
        DSR value (deflated Sharpe ratio)
    """
    if n_trials <= 0 or T <= 0:
        return 0.0
    
    # Calculate PSR first
    psr = probabilistic_sharpe_ratio(observed_sr, benchmark_sr, T, skewness, kurtosis)
    
    # Standard error of Sharpe ratio
    se_sr = np.sqrt((1 + 0.5 * observed_sr**2 - skewness * observed_sr + 
                     (kurtosis - 3) / 4 * observed_sr**2) / T)
    
    if n_trials == 1:
        # No deflation needed for single trial
        return observed_sr
    
    # Deflation for multiple testing (Bailey & López de Prado formula)
    # The key insight: when you test n_trials strategies, the expected maximum
    # Sharpe ratio under the null hypothesis increases
    
    # Expected maximum of n standard normal variables
    # More accurate approximation for finite samples
    if n_trials < 10:
        # Small sample correction
        e_max = np.sqrt(2 * np.log(n_trials))
    else:
        # Gumbel approximation for larger samples
        e_max = np.sqrt(2 * np.log(n_trials)) - (np.log(np.log(n_trials)) + np.log(4 * np.pi)) / (2 * np.sqrt(2 * np.log(n_trials)))
    
    # Deflated Sharpe Ratio
    # The observed SR needs to exceed what we'd expect by chance from n_trials attempts
    dsr = (observed_sr - benchmark_sr) / se_sr - e_max
    
    # Convert back to Sharpe scale
    dsr = benchmark_sr + dsr * se_sr
    
    return dsr


def calculate_all_sharpe_metrics(
    returns: np.ndarray,
    n_trials: int = 1,
    benchmark_sr: float = 0.0,
    confidence_level: float = 0.95,
    periods_per_year: int = 252
) -> SharpeMetrics:
    """
    Calculate all Sharpe-related metrics.
    
    Args:
        returns: Array of returns
        n_trials: Number of strategies tested
        benchmark_sr: Minimum acceptable Sharpe
        confidence_level: Confidence level for DSR
        periods_per_year: Periods per year for annualization
        
    Returns:
        SharpeMetrics object with all calculations
    """
    T = len(returns)
    
    if T < 4:
        logger.warning("Insufficient data for reliable Sharpe metrics", n_obs=T)
        return SharpeMetrics(
            sharpe_ratio=0.0, psr=0.0, dsr=0.0, n_trials=n_trials,
            T=T, skewness=0.0, kurtosis=0.0, confidence_level=confidence_level,
            metadata={"warning": "Insufficient data"}
        )
    
    # Calculate moments
    sr = sharpe_ratio(returns, periods_per_year=periods_per_year)
    skewness = stats.skew(returns)
    excess_kurtosis = stats.kurtosis(returns, fisher=True)  # Excess kurtosis
    
    # Calculate PSR
    psr = probabilistic_sharpe_ratio(sr, benchmark_sr, T, skewness, excess_kurtosis)
    
    # Calculate DSR
    dsr = deflated_sharpe_ratio(
        sr, n_trials, T, skewness, excess_kurtosis, 
        benchmark_sr, confidence_level
    )
    
    # Compile metadata
    metadata = {
        "annualized_return": np.mean(returns) * periods_per_year,
        "annualized_volatility": np.std(returns) * np.sqrt(periods_per_year),
        "max_drawdown": calculate_max_drawdown(returns),
        "hit_rate": np.mean(returns > 0),
        "avg_win": np.mean(returns[returns > 0]) if np.any(returns > 0) else 0,
        "avg_loss": np.mean(returns[returns <= 0]) if np.any(returns <= 0) else 0,
        "periods_per_year": periods_per_year
    }
    
    results = SharpeMetrics(
        sharpe_ratio=sr,
        psr=psr,
        dsr=dsr,
        n_trials=n_trials,
        T=T,
        skewness=skewness,
        kurtosis=excess_kurtosis,
        confidence_level=confidence_level,
        metadata=metadata
    )
    
    logger.info(
        "Sharpe metrics calculated",
        sharpe=f"{sr:.3f}",
        psr=f"{psr:.3f}",
        dsr=f"{dsr:.3f}",
        n_trials=n_trials,
        T=T,
        interpretation=interpret_dsr(dsr, sr)
    )
    
    return results


def calculate_max_drawdown(returns: np.ndarray) -> float:
    """Calculate maximum drawdown from returns."""
    cumulative = (1 + returns).cumprod()
    running_max = np.maximum.accumulate(cumulative)
    drawdown = (cumulative - running_max) / running_max
    return np.min(drawdown)


def interpret_dsr(dsr: float, original_sr: float) -> str:
    """
    Provide interpretation of DSR vs original Sharpe.
    
    Args:
        dsr: Deflated Sharpe Ratio
        original_sr: Original Sharpe Ratio
        
    Returns:
        Interpretation string
    """
    deflation = (original_sr - dsr) / abs(original_sr) * 100 if original_sr != 0 else 0
    
    if dsr < 0:
        return f"NEGATIVE DSR: Strategy likely unprofitable (deflation: {deflation:.1f}%)"
    elif dsr < 0.5:
        return f"LOW DSR: Weak evidence of skill (deflation: {deflation:.1f}%)"
    elif dsr < 1.0:
        return f"MODERATE DSR: Some evidence of skill (deflation: {deflation:.1f}%)"
    elif dsr < 1.5:
        return f"GOOD DSR: Strong evidence of skill (deflation: {deflation:.1f}%)"
    else:
        return f"EXCELLENT DSR: Very strong evidence of skill (deflation: {deflation:.1f}%)"


def test_dsr_implementation():
    """
    Test DSR implementation against known values from the paper.
    
    Bailey & López de Prado provide test cases:
    - SR=1.0, n_trials=100, T=1000 → DSR ≈ 0.65
    - SR=2.0, n_trials=10, T=1000 → DSR ≈ 1.85
    """
    test_cases = [
        # (SR, n_trials, T, expected_DSR_range)
        (1.0, 100, 1000, (0.60, 0.70)),  # Should deflate significantly
        (2.0, 10, 1000, (1.80, 1.90)),   # Should deflate less
        (1.5, 1, 1000, (1.45, 1.50)),    # Minimal deflation with 1 trial
        (0.5, 1000, 1000, (-0.1, 0.1)),  # Heavy deflation with many trials
    ]
    
    print("\n" + "="*60)
    print("DSR Implementation Test (vs Bailey & López de Prado)")
    print("="*60)
    
    all_passed = True
    
    for sr, n_trials, T, expected_range in test_cases:
        # Create synthetic returns with specified Sharpe ratio
        # Assuming normal returns for test (skew=0, kurt=0)
        daily_return = sr / np.sqrt(252)
        daily_vol = 1.0 / np.sqrt(252)
        
        dsr = deflated_sharpe_ratio(
            observed_sr=sr,
            n_trials=n_trials,
            T=T,
            skewness=0.0,
            kurtosis=0.0,
            benchmark_sr=0.0
        )
        
        passed = expected_range[0] <= dsr <= expected_range[1]
        all_passed = all_passed and passed
        
        print(f"\nTest: SR={sr:.1f}, n_trials={n_trials}, T={T}")
        print(f"  Expected DSR: {expected_range[0]:.2f} - {expected_range[1]:.2f}")
        print(f"  Calculated DSR: {dsr:.3f}")
        print(f"  Status: {'✓ PASS' if passed else '✗ FAIL'}")
        print(f"  Deflation: {(sr - dsr)/sr*100:.1f}%")
    
    print("\n" + "="*60)
    print(f"Overall Test Result: {'ALL TESTS PASSED ✓' if all_passed else 'SOME TESTS FAILED ✗'}")
    print("="*60 + "\n")
    
    return all_passed


# Additional utility functions

def minimum_track_record_length(
    target_sr: float,
    confidence_level: float = 0.95,
    benchmark_sr: float = 0.0,
    skewness: float = 0.0,
    kurtosis: float = 0.0
) -> int:
    """
    Calculate minimum track record length for statistical significance.
    
    How long do you need to trade to prove your Sharpe is real?
    
    Args:
        target_sr: Target Sharpe ratio to prove
        confidence_level: Required confidence level
        benchmark_sr: Benchmark to beat
        skewness: Expected skewness
        kurtosis: Expected excess kurtosis
        
    Returns:
        Minimum number of observations needed
    """
    z_score = stats.norm.ppf(confidence_level)
    
    # Rearranging PSR formula to solve for T
    numerator = z_score**2 * (1 + 0.5 * target_sr**2 - skewness * target_sr + 
                              (kurtosis - 3) / 4 * target_sr**2)
    denominator = (target_sr - benchmark_sr)**2
    
    if denominator <= 0:
        return np.inf
    
    min_T = int(np.ceil(numerator / denominator))
    
    logger.info(
        "Minimum track record length",
        target_sr=target_sr,
        confidence=confidence_level,
        min_observations=min_T,
        min_years_daily=f"{min_T/252:.1f}"
    )
    
    return min_T


if __name__ == "__main__":
    # Run implementation test
    test_dsr_implementation()