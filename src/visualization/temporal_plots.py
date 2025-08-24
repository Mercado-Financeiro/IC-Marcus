"""
Temporal validation and walk-forward analysis plots.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Set publication-quality defaults
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300


def plot_split_timeline(
    dates: pd.DatetimeIndex,
    splits: List[Dict[str, Any]],
    purge_gap: int = 0,
    embargo_pct: float = 0.0,
    title: str = "Walk-Forward Split Timeline with Purge/Embargo",
    save_path: Optional[str] = None,
    figsize: Tuple[float, float] = (14, 8)
) -> Dict[str, Any]:
    """
    Plot timeline visualization of train/val/test splits with purge and embargo.
    
    Args:
        dates: DatetimeIndex of the data
        splits: List of dicts with 'train', 'val', 'test' indices
        purge_gap: Number of periods to purge between train and val
        embargo_pct: Percentage of test data to embargo
        title: Plot title
        save_path: Path to save figure
        figsize: Figure size
        
    Returns:
        Dictionary with split statistics
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    n_splits = len(splits)
    colors = {'train': 'blue', 'val': 'orange', 'test': 'green',
              'purge': 'red', 'embargo': 'purple'}
    
    # Calculate date range
    date_min = dates.min()
    date_max = dates.max()
    date_range = (date_max - date_min).days
    
    for i, split in enumerate(splits):
        y_pos = n_splits - i - 1
        
        # Train period
        if 'train' in split and len(split['train']) > 0:
            train_start = dates[split['train'][0]]
            train_end = dates[split['train'][-1]]
            train_rect = patches.Rectangle(
                (train_start, y_pos - 0.4),
                train_end - train_start,
                0.8,
                linewidth=1,
                edgecolor='black',
                facecolor=colors['train'],
                alpha=0.7,
                label='Train' if i == 0 else ""
            )
            ax.add_patch(train_rect)
        
        # Purge gap
        if purge_gap > 0 and 'val' in split:
            purge_start = train_end
            purge_end = purge_start + timedelta(days=purge_gap)
            purge_rect = patches.Rectangle(
                (purge_start, y_pos - 0.4),
                purge_end - purge_start,
                0.8,
                linewidth=1,
                edgecolor='black',
                facecolor=colors['purge'],
                alpha=0.3,
                hatch='///',
                label='Purge' if i == 0 else ""
            )
            ax.add_patch(purge_rect)
        
        # Validation period
        if 'val' in split and len(split['val']) > 0:
            val_start = dates[split['val'][0]]
            val_end = dates[split['val'][-1]]
            val_rect = patches.Rectangle(
                (val_start, y_pos - 0.4),
                val_end - val_start,
                0.8,
                linewidth=1,
                edgecolor='black',
                facecolor=colors['val'],
                alpha=0.7,
                label='Validation' if i == 0 else ""
            )
            ax.add_patch(val_rect)
        
        # Test period
        if 'test' in split and len(split['test']) > 0:
            test_start = dates[split['test'][0]]
            test_end = dates[split['test'][-1]]
            
            # Calculate embargo
            if embargo_pct > 0:
                embargo_size = int(len(split['test']) * embargo_pct)
                actual_test_end = dates[split['test'][-embargo_size]]
                embargo_start = actual_test_end
            else:
                actual_test_end = test_end
                embargo_start = test_end
            
            test_rect = patches.Rectangle(
                (test_start, y_pos - 0.4),
                actual_test_end - test_start,
                0.8,
                linewidth=1,
                edgecolor='black',
                facecolor=colors['test'],
                alpha=0.7,
                label='Test' if i == 0 else ""
            )
            ax.add_patch(test_rect)
            
            # Embargo period
            if embargo_pct > 0:
                embargo_rect = patches.Rectangle(
                    (embargo_start, y_pos - 0.4),
                    test_end - embargo_start,
                    0.8,
                    linewidth=1,
                    edgecolor='black',
                    facecolor=colors['embargo'],
                    alpha=0.3,
                    hatch='\\\\\\',
                    label='Embargo' if i == 0 else ""
                )
                ax.add_patch(embargo_rect)
        
        # Add split label
        ax.text(date_min - timedelta(days=date_range * 0.02), y_pos,
               f'Split {i+1}', ha='right', va='center', fontsize=10)
    
    ax.set_xlim(date_min - timedelta(days=date_range * 0.05),
                date_max + timedelta(days=date_range * 0.05))
    ax.set_ylim(-0.5, n_splits - 0.5)
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Split', fontsize=12)
    ax.set_title(title, fontsize=14, pad=20)
    ax.legend(loc='upper right', frameon=True, fancybox=True, shadow=True)
    ax.grid(True, alpha=0.3, axis='x')
    
    # Remove y-axis ticks
    ax.set_yticks([])
    
    # Add annotations
    ax.text(0.02, 0.98, f'Purge Gap: {purge_gap} periods\nEmbargo: {embargo_pct*100:.1f}%',
           transform=ax.transAxes, fontsize=11, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.savefig(save_path.replace('.png', '.pdf'), bbox_inches='tight')
    
    plt.show()
    
    # Calculate statistics
    train_pct = np.mean([len(s.get('train', [])) for s in splits]) / len(dates) * 100
    val_pct = np.mean([len(s.get('val', [])) for s in splits]) / len(dates) * 100
    test_pct = np.mean([len(s.get('test', [])) for s in splits]) / len(dates) * 100
    
    return {
        'n_splits': n_splits,
        'avg_train_pct': train_pct,
        'avg_val_pct': val_pct,
        'avg_test_pct': test_pct,
        'purge_gap': purge_gap,
        'embargo_pct': embargo_pct
    }


def plot_walkforward_metrics(
    metrics_df: pd.DataFrame,
    metrics: List[str] = ['pr_auc', 'mcc', 'expected_value'],
    title: str = "Walk-Forward Metrics Heatmap",
    save_path: Optional[str] = None,
    figsize: Tuple[float, float] = (14, 8)
) -> Dict[str, Any]:
    """
    Plot walk-forward metrics as heatmap and line plots.
    
    Args:
        metrics_df: DataFrame with columns for each metric and rows for each window
        metrics: List of metric names to plot
        title: Plot title
        save_path: Path to save figure
        figsize: Figure size
        
    Returns:
        Dictionary with metric statistics
    """
    n_windows = len(metrics_df)
    n_metrics = len(metrics)
    
    fig = plt.figure(figsize=figsize)
    
    # Create grid
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.5], width_ratios=[3, 1])
    
    # Heatmap
    ax1 = fig.add_subplot(gs[0, 0])
    
    # Prepare data for heatmap
    heatmap_data = metrics_df[metrics].T
    
    # Normalize each metric to [0, 1] for better visualization
    normalized_data = heatmap_data.copy()
    for metric in metrics:
        if metric in normalized_data.index:
            row = normalized_data.loc[metric]
            min_val = row.min()
            max_val = row.max()
            if max_val > min_val:
                normalized_data.loc[metric] = (row - min_val) / (max_val - min_val)
    
    sns.heatmap(normalized_data, annot=heatmap_data, fmt='.3f',
                cmap='RdYlGn', center=0.5, 
                xticklabels=[f'W{i+1}' for i in range(n_windows)],
                yticklabels=metrics,
                cbar_kws={'label': 'Normalized Score'},
                ax=ax1, linewidths=1, linecolor='gray')
    
    ax1.set_title('Metrics Heatmap (values shown, colors normalized)', fontsize=12)
    ax1.set_xlabel('Window')
    ax1.set_ylabel('Metric')
    
    # Line plots
    ax2 = fig.add_subplot(gs[1, :])
    
    colors = ['blue', 'green', 'red', 'orange', 'purple']
    for i, metric in enumerate(metrics):
        if metric in metrics_df.columns:
            values = metrics_df[metric].values
            windows = range(1, len(values) + 1)
            
            ax2.plot(windows, values, '-o', color=colors[i % len(colors)],
                    linewidth=2, markersize=6, label=metric, alpha=0.8)
            
            # Add trend line
            z = np.polyfit(windows, values, 1)
            p = np.poly1d(z)
            ax2.plot(windows, p(windows), '--', color=colors[i % len(colors)],
                    alpha=0.5, linewidth=1)
            
            # Mark best and worst
            best_idx = np.argmax(values)
            worst_idx = np.argmin(values)
            ax2.scatter(best_idx + 1, values[best_idx], s=100, 
                       color=colors[i % len(colors)], zorder=5, 
                       edgecolors='black', linewidth=2)
            ax2.scatter(worst_idx + 1, values[worst_idx], s=100,
                       color=colors[i % len(colors)], zorder=5,
                       edgecolors='red', linewidth=2, marker='x')
    
    ax2.set_xlabel('Window', fontsize=12)
    ax2.set_ylabel('Metric Value', fontsize=12)
    ax2.set_title('Metrics Evolution Over Time', fontsize=12)
    ax2.legend(loc='best', frameon=True, fancybox=True, shadow=True)
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(range(1, n_windows + 1))
    ax2.set_xticklabels([f'W{i+1}' for i in range(n_windows)])
    
    # Statistics table
    ax3 = fig.add_subplot(gs[0, 1])
    ax3.axis('tight')
    ax3.axis('off')
    
    stats_data = []
    for metric in metrics:
        if metric in metrics_df.columns:
            values = metrics_df[metric].values
            stats_data.append([
                metric[:10],  # Truncate long names
                f'{np.mean(values):.3f}',
                f'{np.std(values):.3f}',
                f'{np.std(values)/np.mean(values):.3f}' if np.mean(values) != 0 else 'N/A'
            ])
    
    table = ax3.table(cellText=stats_data,
                     colLabels=['Metric', 'Mean', 'Std', 'CV'],
                     cellLoc='center',
                     loc='center',
                     colWidths=[0.3, 0.2, 0.2, 0.2])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    
    # Style the table
    for i in range(len(stats_data) + 1):
        for j in range(4):
            cell = table[(i, j)]
            if i == 0:
                cell.set_facecolor('#40466e')
                cell.set_text_props(weight='bold', color='white')
            else:
                cell.set_facecolor('#f0f0f0' if i % 2 == 0 else 'white')
    
    fig.suptitle(title, fontsize=14, y=0.98)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.savefig(save_path.replace('.png', '.pdf'), bbox_inches='tight')
    
    plt.show()
    
    # Calculate statistics
    results = {}
    for metric in metrics:
        if metric in metrics_df.columns:
            values = metrics_df[metric].values
            results[f'{metric}_mean'] = float(np.mean(values))
            results[f'{metric}_std'] = float(np.std(values))
            results[f'{metric}_cv'] = float(np.std(values) / np.mean(values)) if np.mean(values) != 0 else 0
            results[f'{metric}_trend'] = float(np.polyfit(range(len(values)), values, 1)[0])
    
    return results


def plot_temporal_stability(
    predictions_over_time: Dict[str, np.ndarray],
    dates: pd.DatetimeIndex,
    title: str = "Temporal Stability Analysis",
    save_path: Optional[str] = None,
    figsize: Tuple[float, float] = (14, 10)
) -> Dict[str, float]:
    """
    Analyze prediction stability over time.
    
    Args:
        predictions_over_time: Dict of window_name -> predictions
        dates: DatetimeIndex for x-axis
        title: Plot title
        save_path: Path to save figure
        figsize: Figure size
        
    Returns:
        Dictionary with stability metrics
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    window_names = list(predictions_over_time.keys())
    n_windows = len(window_names)
    
    # 1. Prediction distribution shift
    ax1 = axes[0, 0]
    for i, (window, preds) in enumerate(predictions_over_time.items()):
        ax1.hist(preds, bins=30, alpha=0.5, 
                label=f'{window}', density=True)
    
    ax1.set_xlabel('Predicted Probability')
    ax1.set_ylabel('Density')
    ax1.set_title('Prediction Distribution Shift')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Mean and std over time
    ax2 = axes[0, 1]
    means = [np.mean(preds) for preds in predictions_over_time.values()]
    stds = [np.std(preds) for preds in predictions_over_time.values()]
    
    x = range(n_windows)
    ax2.errorbar(x, means, yerr=stds, fmt='o-', linewidth=2, 
                capsize=5, capthick=2, markersize=8)
    ax2.set_xlabel('Window')
    ax2.set_ylabel('Mean Prediction ± Std')
    ax2.set_title('Prediction Statistics Over Time')
    ax2.set_xticks(x)
    ax2.set_xticklabels(window_names, rotation=45, ha='right')
    ax2.grid(True, alpha=0.3)
    
    # 3. Quantile evolution
    ax3 = axes[1, 0]
    quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]
    quantile_data = {q: [] for q in quantiles}
    
    for preds in predictions_over_time.values():
        for q in quantiles:
            quantile_data[q].append(np.quantile(preds, q))
    
    for q in quantiles:
        ax3.plot(x, quantile_data[q], '-o', label=f'Q{int(q*100)}',
                linewidth=2, markersize=6)
    
    ax3.set_xlabel('Window')
    ax3.set_ylabel('Prediction Quantile')
    ax3.set_title('Quantile Evolution')
    ax3.set_xticks(x)
    ax3.set_xticklabels(window_names, rotation=45, ha='right')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. KL divergence from first window
    ax4 = axes[1, 1]
    first_preds = list(predictions_over_time.values())[0]
    
    # Create histogram bins
    bins = np.linspace(0, 1, 51)
    first_hist, _ = np.histogram(first_preds, bins=bins, density=True)
    first_hist = first_hist + 1e-10  # Avoid log(0)
    
    kl_divergences = []
    for window, preds in predictions_over_time.items():
        hist, _ = np.histogram(preds, bins=bins, density=True)
        hist = hist + 1e-10  # Avoid log(0)
        
        # KL divergence
        kl_div = np.sum(hist * np.log(hist / first_hist)) * (bins[1] - bins[0])
        kl_divergences.append(kl_div)
    
    ax4.bar(x, kl_divergences, alpha=0.7, color='coral')
    ax4.set_xlabel('Window')
    ax4.set_ylabel('KL Divergence from Window 1')
    ax4.set_title('Distribution Drift (KL Divergence)')
    ax4.set_xticks(x)
    ax4.set_xticklabels(window_names, rotation=45, ha='right')
    ax4.grid(True, alpha=0.3, axis='y')
    
    # Add warning if significant drift detected
    if max(kl_divergences[1:]) > 0.1:
        ax4.annotate('⚠ Significant drift detected!', 
                    xy=(0.5, 0.95), xycoords='axes fraction',
                    fontsize=12, color='red', weight='bold',
                    ha='center')
    
    fig.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.savefig(save_path.replace('.png', '.pdf'), bbox_inches='tight')
    
    plt.show()
    
    return {
        'mean_drift': float(np.std(means)),
        'std_drift': float(np.std(stds)),
        'max_kl_divergence': float(max(kl_divergences[1:])) if len(kl_divergences) > 1 else 0,
        'mean_kl_divergence': float(np.mean(kl_divergences[1:])) if len(kl_divergences) > 1 else 0,
        'quantile_stability': float(1 - np.mean([np.std(quantile_data[q]) for q in quantiles]))
    }