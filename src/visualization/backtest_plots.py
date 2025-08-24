"""
Backtest and trading performance visualization plots.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')

# Set publication-quality defaults
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300


def plot_equity_curve_with_bands(
    returns: np.ndarray,
    initial_capital: float = 10000,
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    block_size: Optional[int] = None,
    title: str = "Equity Curve with Bootstrap Confidence Bands",
    save_path: Optional[str] = None,
    figsize: Tuple[float, float] = (14, 8)
) -> Dict[str, float]:
    """
    Plot equity curve with bootstrap confidence bands.
    
    Args:
        returns: Array of returns (not cumulative)
        initial_capital: Starting capital
        n_bootstrap: Number of bootstrap samples
        confidence_level: Confidence level for bands
        block_size: Block size for stationary bootstrap (None for iid bootstrap)
        title: Plot title
        save_path: Path to save figure
        figsize: Figure size
        
    Returns:
        Dictionary with performance metrics
    """
    # Calculate cumulative returns
    cumulative_returns = (1 + returns).cumprod()
    equity_curve = initial_capital * cumulative_returns
    
    n_periods = len(returns)
    
    # Bootstrap for confidence bands
    if block_size is None:
        # Optimal block size (Politis & White, 2004)
        block_size = int(np.sqrt(n_periods))
    
    bootstrap_curves = []
    np.random.seed(42)
    
    for _ in range(n_bootstrap):
        if block_size > 1:
            # Stationary bootstrap
            bootstrap_returns = []
            i = 0
            while len(bootstrap_returns) < n_periods:
                # Geometric distribution for block length
                block_length = min(np.random.geometric(1/block_size), 
                                  n_periods - i)
                block_start = np.random.randint(0, n_periods)
                
                for j in range(block_length):
                    bootstrap_returns.append(returns[(block_start + j) % n_periods])
                    if len(bootstrap_returns) >= n_periods:
                        break
            
            bootstrap_returns = np.array(bootstrap_returns[:n_periods])
        else:
            # IID bootstrap
            indices = np.random.choice(n_periods, n_periods, replace=True)
            bootstrap_returns = returns[indices]
        
        bootstrap_cumret = (1 + bootstrap_returns).cumprod()
        bootstrap_curves.append(initial_capital * bootstrap_cumret)
    
    bootstrap_curves = np.array(bootstrap_curves)
    
    # Calculate confidence bands
    alpha = 1 - confidence_level
    lower_band = np.percentile(bootstrap_curves, alpha/2 * 100, axis=0)
    upper_band = np.percentile(bootstrap_curves, (1 - alpha/2) * 100, axis=0)
    median_curve = np.median(bootstrap_curves, axis=0)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, 
                                   gridspec_kw={'height_ratios': [3, 1]})
    
    # Equity curve with bands
    periods = range(len(equity_curve))
    ax1.plot(periods, equity_curve, 'b-', linewidth=2, label='Actual')
    ax1.plot(periods, median_curve, 'g--', linewidth=1, 
            label='Bootstrap Median', alpha=0.7)
    ax1.fill_between(periods, lower_band, upper_band, 
                     alpha=0.3, color='gray',
                     label=f'{confidence_level*100:.0f}% CI')
    
    # Add buy-and-hold benchmark
    buy_hold = initial_capital * (1 + returns.mean()) ** np.arange(n_periods)
    ax1.plot(periods, buy_hold, 'r--', linewidth=1, 
            label='Buy & Hold', alpha=0.7)
    
    # Mark max drawdown period
    peak = np.maximum.accumulate(equity_curve)
    drawdown = (equity_curve - peak) / peak
    max_dd_end = np.argmin(drawdown)
    max_dd_start = np.argmax(equity_curve[:max_dd_end + 1])
    
    ax1.plot([max_dd_start, max_dd_end], 
            [equity_curve[max_dd_start], equity_curve[max_dd_end]],
            'r-', linewidth=3, alpha=0.7)
    ax1.annotate(f'Max DD: {drawdown[max_dd_end]*100:.1f}%',
                xy=(max_dd_end, equity_curve[max_dd_end]),
                xytext=(max_dd_end + n_periods*0.05, equity_curve[max_dd_end]),
                arrowprops=dict(arrowstyle='->', color='red'),
                fontsize=10, color='red')
    
    ax1.set_ylabel('Equity ($)', fontsize=12)
    ax1.set_title(title, fontsize=14)
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # Underwater plot (drawdown over time)
    ax2.fill_between(periods, 0, drawdown * 100, 
                     where=(drawdown < 0), color='red', alpha=0.5)
    ax2.set_xlabel('Period', fontsize=12)
    ax2.set_ylabel('Drawdown (%)', fontsize=12)
    ax2.set_title('Underwater Plot', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([min(drawdown * 100) * 1.1, 1])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.savefig(save_path.replace('.png', '.pdf'), bbox_inches='tight')
    
    plt.show()
    
    # Calculate metrics
    total_return = (equity_curve[-1] / initial_capital - 1) * 100
    annualized_return = ((equity_curve[-1] / initial_capital) ** (252/n_periods) - 1) * 100
    sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252)
    max_drawdown = drawdown.min() * 100
    
    return {
        'total_return': total_return,
        'annualized_return': annualized_return,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'final_equity': equity_curve[-1],
        'ci_lower': lower_band[-1],
        'ci_upper': upper_band[-1]
    }


def plot_drawdown_curve(
    equity_curve: np.ndarray,
    title: str = "Drawdown Analysis",
    save_path: Optional[str] = None,
    figsize: Tuple[float, float] = (14, 8)
) -> Dict[str, float]:
    """
    Plot detailed drawdown analysis.
    
    Args:
        equity_curve: Equity curve values
        title: Plot title
        save_path: Path to save figure
        figsize: Figure size
        
    Returns:
        Dictionary with drawdown statistics
    """
    # Calculate drawdown
    peak = np.maximum.accumulate(equity_curve)
    drawdown = (equity_curve - peak) / peak * 100
    
    # Find drawdown periods
    dd_start_indices = []
    dd_end_indices = []
    dd_values = []
    
    in_drawdown = False
    for i in range(1, len(drawdown)):
        if not in_drawdown and drawdown[i] < 0:
            dd_start_indices.append(i)
            in_drawdown = True
        elif in_drawdown and drawdown[i] >= 0:
            dd_end_indices.append(i)
            dd_values.append(drawdown[dd_start_indices[-1]:i].min())
            in_drawdown = False
    
    # Handle ongoing drawdown
    if in_drawdown:
        dd_end_indices.append(len(drawdown) - 1)
        dd_values.append(drawdown[dd_start_indices[-1]:].min())
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # 1. Drawdown curve
    ax1 = axes[0, 0]
    periods = range(len(drawdown))
    ax1.fill_between(periods, 0, drawdown, where=(drawdown < 0),
                     color='red', alpha=0.5, label='Drawdown')
    ax1.plot(periods, drawdown, 'r-', linewidth=1)
    
    # Mark top 5 drawdowns
    if dd_values:
        top_dd_indices = np.argsort(dd_values)[:5]
        for idx in top_dd_indices:
            if idx < len(dd_start_indices) and idx < len(dd_end_indices):
                start = dd_start_indices[idx]
                end = dd_end_indices[idx]
                min_point = start + np.argmin(drawdown[start:end+1])
                ax1.scatter(min_point, drawdown[min_point], 
                          color='darkred', s=50, zorder=5)
    
    ax1.set_xlabel('Period')
    ax1.set_ylabel('Drawdown (%)')
    ax1.set_title('Drawdown Over Time')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([min(drawdown) * 1.1, 1])
    
    # 2. Drawdown distribution
    ax2 = axes[0, 1]
    if dd_values:
        ax2.hist(dd_values, bins=20, alpha=0.7, color='red', edgecolor='black')
        ax2.axvline(x=np.mean(dd_values), color='blue', linestyle='--',
                   label=f'Mean: {np.mean(dd_values):.2f}%')
        ax2.axvline(x=np.median(dd_values), color='green', linestyle='--',
                   label=f'Median: {np.median(dd_values):.2f}%')
    ax2.set_xlabel('Drawdown Depth (%)')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Drawdown Distribution')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 3. Drawdown duration
    ax3 = axes[1, 0]
    if dd_start_indices and dd_end_indices:
        durations = [end - start for start, end in 
                    zip(dd_start_indices, dd_end_indices)]
        ax3.hist(durations, bins=15, alpha=0.7, color='blue', edgecolor='black')
        ax3.axvline(x=np.mean(durations), color='red', linestyle='--',
                   label=f'Mean: {np.mean(durations):.1f} periods')
    ax3.set_xlabel('Duration (periods)')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Drawdown Duration Distribution')
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 4. Recovery time analysis
    ax4 = axes[1, 1]
    recovery_times = []
    for i, (start, end) in enumerate(zip(dd_start_indices, dd_end_indices)):
        if end < len(equity_curve) - 1:
            # Find recovery point (back to previous peak)
            peak_value = peak[start - 1] if start > 0 else equity_curve[0]
            recovery_point = None
            for j in range(end, len(equity_curve)):
                if equity_curve[j] >= peak_value:
                    recovery_point = j
                    break
            if recovery_point:
                recovery_times.append(recovery_point - end)
    
    if recovery_times:
        ax4.hist(recovery_times, bins=15, alpha=0.7, 
                color='green', edgecolor='black')
        ax4.axvline(x=np.mean(recovery_times), color='red', linestyle='--',
                   label=f'Mean: {np.mean(recovery_times):.1f} periods')
        ax4.set_xlabel('Recovery Time (periods)')
        ax4.set_ylabel('Frequency')
        ax4.set_title('Time to Recover from Drawdown')
        ax4.legend()
    else:
        ax4.text(0.5, 0.5, 'No complete recoveries', 
                ha='center', va='center', transform=ax4.transAxes)
        ax4.set_title('Time to Recover from Drawdown')
    
    ax4.grid(True, alpha=0.3, axis='y')
    
    fig.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.savefig(save_path.replace('.png', '.pdf'), bbox_inches='tight')
    
    plt.show()
    
    return {
        'max_drawdown': min(drawdown) if len(drawdown) > 0 else 0,
        'avg_drawdown': np.mean(dd_values) if dd_values else 0,
        'n_drawdowns': len(dd_values),
        'avg_duration': np.mean(durations) if dd_start_indices else 0,
        'max_duration': max(durations) if dd_start_indices else 0,
        'avg_recovery_time': np.mean(recovery_times) if recovery_times else None
    }


def plot_returns_distribution(
    returns: np.ndarray,
    benchmark_returns: Optional[np.ndarray] = None,
    title: str = "Returns Distribution Analysis",
    save_path: Optional[str] = None,
    figsize: Tuple[float, float] = (14, 8)
) -> Dict[str, float]:
    """
    Plot returns distribution with detailed statistics.
    
    Args:
        returns: Array of returns
        benchmark_returns: Benchmark returns for comparison (optional)
        title: Plot title
        save_path: Path to save figure
        figsize: Figure size
        
    Returns:
        Dictionary with distribution statistics
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # 1. Returns histogram with normal overlay
    ax1 = axes[0, 0]
    n, bins, patches = ax1.hist(returns, bins=50, alpha=0.7, 
                                color='blue', edgecolor='black', density=True)
    
    # Fit normal distribution
    mu, sigma = returns.mean(), returns.std()
    x = np.linspace(returns.min(), returns.max(), 100)
    ax1.plot(x, stats.norm.pdf(x, mu, sigma), 'r-', linewidth=2,
            label=f'Normal(μ={mu:.4f}, σ={sigma:.4f})')
    
    # Add VaR and CVaR
    var_95 = np.percentile(returns, 5)
    cvar_95 = returns[returns <= var_95].mean()
    ax1.axvline(x=var_95, color='orange', linestyle='--',
               label=f'VaR(95%): {var_95:.4f}')
    ax1.axvline(x=cvar_95, color='red', linestyle='--',
               label=f'CVaR(95%): {cvar_95:.4f}')
    
    ax1.set_xlabel('Returns')
    ax1.set_ylabel('Density')
    ax1.set_title('Returns Distribution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Box plot comparison
    ax2 = axes[0, 1]
    data_to_plot = [returns]
    labels = ['Strategy']
    
    if benchmark_returns is not None:
        data_to_plot.append(benchmark_returns)
        labels.append('Benchmark')
    
    bp = ax2.boxplot(data_to_plot, labels=labels, patch_artist=True)
    colors = ['lightblue', 'lightgreen']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    
    ax2.set_ylabel('Returns')
    ax2.set_title('Returns Comparison')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 3. Cumulative returns
    ax3 = axes[1, 0]
    cumret = (1 + returns).cumprod() - 1
    ax3.plot(cumret, 'b-', linewidth=2, label='Strategy')
    
    if benchmark_returns is not None:
        bench_cumret = (1 + benchmark_returns).cumprod() - 1
        ax3.plot(bench_cumret, 'g-', linewidth=2, label='Benchmark')
    
    ax3.set_xlabel('Period')
    ax3.set_ylabel('Cumulative Return')
    ax3.set_title('Cumulative Returns')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Rolling volatility
    ax4 = axes[1, 1]
    window = min(60, len(returns) // 4)
    rolling_vol = pd.Series(returns).rolling(window).std() * np.sqrt(252)
    ax4.plot(rolling_vol, 'b-', linewidth=2, label='Strategy')
    ax4.axhline(y=sigma * np.sqrt(252), color='r', linestyle='--',
               label=f'Full Period: {sigma * np.sqrt(252):.3f}')
    
    if benchmark_returns is not None:
        bench_rolling_vol = pd.Series(benchmark_returns).rolling(window).std() * np.sqrt(252)
        ax4.plot(bench_rolling_vol, 'g-', linewidth=1, 
                label='Benchmark', alpha=0.7)
    
    ax4.set_xlabel('Period')
    ax4.set_ylabel('Annualized Volatility')
    ax4.set_title(f'Rolling Volatility ({window}-period)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    fig.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.savefig(save_path.replace('.png', '.pdf'), bbox_inches='tight')
    
    plt.show()
    
    # Calculate statistics
    skewness = stats.skew(returns)
    kurtosis = stats.kurtosis(returns)
    jarque_bera = stats.jarque_bera(returns)
    
    return {
        'mean': float(mu),
        'std': float(sigma),
        'skewness': float(skewness),
        'kurtosis': float(kurtosis),
        'var_95': float(var_95),
        'cvar_95': float(cvar_95),
        'sharpe': float(mu / sigma * np.sqrt(252)),
        'jarque_bera_stat': float(jarque_bera[0]),
        'jarque_bera_pval': float(jarque_bera[1]),
        'is_normal': jarque_bera[1] > 0.05
    }


def plot_qq_plot(
    returns: np.ndarray,
    dist: str = 'norm',
    title: str = "Q-Q Plot",
    save_path: Optional[str] = None,
    figsize: Tuple[float, float] = (8, 8)
) -> Dict[str, float]:
    """
    Plot Q-Q plot to assess distribution fit.
    
    Args:
        returns: Array of returns
        dist: Distribution to compare against ('norm', 't', 'laplace')
        title: Plot title
        save_path: Path to save figure
        figsize: Figure size
        
    Returns:
        Dictionary with goodness-of-fit statistics
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Standardize returns
    standardized_returns = (returns - returns.mean()) / returns.std()
    
    if dist == 'norm':
        stats.probplot(standardized_returns, dist="norm", plot=ax)
        ax.set_title(f'{title} - Normal Distribution')
    elif dist == 't':
        # Fit t-distribution
        params = stats.t.fit(standardized_returns)
        stats.probplot(standardized_returns, dist=stats.t, 
                      sparams=params, plot=ax)
        ax.set_title(f'{title} - t-Distribution (df={params[0]:.1f})')
    elif dist == 'laplace':
        stats.probplot(standardized_returns, dist="laplace", plot=ax)
        ax.set_title(f'{title} - Laplace Distribution')
    
    # Add 45-degree reference line
    ax.get_lines()[1].set_color('red')
    ax.get_lines()[1].set_linewidth(2)
    
    # Highlight tail deviations
    theoretical_quantiles = ax.get_lines()[0].get_xdata()
    sample_quantiles = ax.get_lines()[0].get_ydata()
    
    # Mark extreme quantiles
    n_points = len(theoretical_quantiles)
    tail_size = int(n_points * 0.05)
    
    ax.scatter(theoretical_quantiles[:tail_size], 
              sample_quantiles[:tail_size],
              color='red', s=50, alpha=0.5, label='Lower tail (5%)')
    ax.scatter(theoretical_quantiles[-tail_size:],
              sample_quantiles[-tail_size:],
              color='orange', s=50, alpha=0.5, label='Upper tail (5%)')
    
    ax.set_xlabel('Theoretical Quantiles')
    ax.set_ylabel('Sample Quantiles')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.savefig(save_path.replace('.png', '.pdf'), bbox_inches='tight')
    
    plt.show()
    
    # Calculate goodness-of-fit
    if dist == 'norm':
        ks_stat, ks_pval = stats.kstest(standardized_returns, 'norm')
        ad_stat, ad_crit, ad_sig = stats.anderson(standardized_returns, 'norm')
    else:
        ks_stat, ks_pval = None, None
        ad_stat, ad_crit, ad_sig = None, None, None
    
    # R-squared of Q-Q plot
    slope, intercept = np.polyfit(theoretical_quantiles, sample_quantiles, 1)
    predicted = slope * theoretical_quantiles + intercept
    r_squared = 1 - (np.sum((sample_quantiles - predicted) ** 2) / 
                    np.sum((sample_quantiles - sample_quantiles.mean()) ** 2))
    
    return {
        'r_squared': float(r_squared),
        'ks_statistic': float(ks_stat) if ks_stat else None,
        'ks_pvalue': float(ks_pval) if ks_pval else None,
        'anderson_statistic': float(ad_stat) if ad_stat else None,
        'slope': float(slope),
        'intercept': float(intercept)
    }


def plot_sharpe_comparison(
    returns: np.ndarray,
    n_strategies_tested: int = 1,
    title: str = "Sharpe Ratio Analysis",
    save_path: Optional[str] = None,
    figsize: Tuple[float, float] = (12, 6)
) -> Dict[str, float]:
    """
    Plot Sharpe ratio with Deflated Sharpe Ratio comparison.
    
    Args:
        returns: Array of returns
        n_strategies_tested: Number of strategies tested (for DSR)
        title: Plot title
        save_path: Path to save figure
        figsize: Figure size
        
    Returns:
        Dictionary with Sharpe ratios
    """
    # Calculate metrics
    mean_return = returns.mean()
    std_return = returns.std()
    n_periods = len(returns)
    
    # Standard Sharpe ratio
    sharpe = mean_return / std_return * np.sqrt(252)
    
    # Standard error of Sharpe
    skewness = stats.skew(returns)
    kurtosis = stats.kurtosis(returns)
    se_sharpe = np.sqrt((1 + 0.5 * sharpe**2 - skewness * sharpe + 
                        (kurtosis - 3) / 4 * sharpe**2) / n_periods) * np.sqrt(252)
    
    # Deflated Sharpe Ratio (Bailey & López de Prado, 2014)
    # Accounts for multiple testing and non-normality
    expected_max_sharpe = (1 - np.euler_gamma) * stats.norm.ppf(1 - 1/n_strategies_tested) + \
                         np.euler_gamma * stats.norm.ppf(1 - 1/(n_strategies_tested * np.e))
    
    deflated_sharpe = stats.norm.cdf((sharpe - expected_max_sharpe) / se_sharpe)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # 1. Sharpe ratio comparison
    ax1.bar(['Standard\nSharpe', 'Expected\nMax Sharpe'], 
           [sharpe, expected_max_sharpe],
           color=['green' if sharpe > expected_max_sharpe else 'red', 'gray'],
           alpha=0.7, edgecolor='black', linewidth=2)
    
    # Add confidence interval
    ax1.errorbar(0, sharpe, yerr=1.96 * se_sharpe, 
                fmt='none', color='black', capsize=10, capthick=2)
    
    # Add value labels
    ax1.text(0, sharpe + 0.1, f'{sharpe:.3f}', ha='center', fontsize=12)
    ax1.text(1, expected_max_sharpe + 0.1, f'{expected_max_sharpe:.3f}', 
            ha='center', fontsize=12)
    
    ax1.set_ylabel('Sharpe Ratio')
    ax1.set_title(f'Sharpe Ratio Comparison\n(Tested {n_strategies_tested} strategies)')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add significance line
    ax1.axhline(y=0, color='black', linestyle='-', linewidth=1)
    
    # 2. Probability interpretation
    ax2.barh(['Deflated\nSharpe\nRatio'], [deflated_sharpe],
            color='blue' if deflated_sharpe > 0.95 else 'orange' if deflated_sharpe > 0.5 else 'red',
            alpha=0.7, edgecolor='black', linewidth=2)
    
    # Add significance thresholds
    ax2.axvline(x=0.5, color='orange', linestyle='--', label='50% (Random)')
    ax2.axvline(x=0.95, color='green', linestyle='--', label='95% (Significant)')
    
    ax2.text(deflated_sharpe + 0.02, 0, f'{deflated_sharpe:.3f}', 
            va='center', fontsize=12)
    
    ax2.set_xlabel('Probability of Skill (not luck)')
    ax2.set_title('Deflated Sharpe Ratio\n(Corrected for selection bias)')
    ax2.set_xlim([0, 1])
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='x')
    
    # Add interpretation text
    if deflated_sharpe > 0.95:
        interpretation = "✓ Strong evidence of skill"
        color = 'green'
    elif deflated_sharpe > 0.5:
        interpretation = "⚠ Weak evidence of skill"
        color = 'orange'
    else:
        interpretation = "✗ Likely due to chance"
        color = 'red'
    
    fig.text(0.5, 0.02, interpretation, ha='center', fontsize=14,
            color=color, weight='bold')
    
    fig.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.savefig(save_path.replace('.png', '.pdf'), bbox_inches='tight')
    
    plt.show()
    
    # Additional metrics
    sortino = mean_return / returns[returns < 0].std() * np.sqrt(252) if len(returns[returns < 0]) > 0 else 0
    calmar = mean_return * 252 / abs(returns.cumsum().min()) if returns.cumsum().min() < 0 else 0
    
    return {
        'sharpe_ratio': float(sharpe),
        'sharpe_se': float(se_sharpe),
        'deflated_sharpe': float(deflated_sharpe),
        'expected_max_sharpe': float(expected_max_sharpe),
        'sortino_ratio': float(sortino),
        'calmar_ratio': float(calmar),
        'prob_skill': float(deflated_sharpe),
        'is_significant': deflated_sharpe > 0.95
    }