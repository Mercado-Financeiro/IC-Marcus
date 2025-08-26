"""
Optuna hyperparameter optimization visualization plots.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any
import optuna
from optuna.visualization import (
    plot_optimization_history as optuna_plot_history,
    plot_param_importances as optuna_plot_importances,
    plot_parallel_coordinate as optuna_plot_parallel,
    plot_contour as optuna_plot_contour
)
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Set publication-quality defaults
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300


def plot_optimization_history(
    study: optuna.Study,
    title: str = "Optimization History",
    save_path: Optional[str] = None,
    figsize: Tuple[float, float] = (14, 6)
) -> Dict[str, Any]:
    """
    Plot Optuna optimization history with convergence analysis.
    
    Args:
        study: Optuna study object
        title: Plot title
        save_path: Path to save figure
        figsize: Figure size
        
    Returns:
        Dictionary with optimization statistics
    """
    trials = study.trials
    n_trials = len(trials)
    
    # Extract values
    values = [t.value for t in trials if t.value is not None]
    best_values = []
    current_best = float('inf') if study.direction == optuna.study.StudyDirection.MINIMIZE else float('-inf')
    
    for v in values:
        if study.direction == optuna.study.StudyDirection.MINIMIZE:
            current_best = min(current_best, v)
        else:
            current_best = max(current_best, v)
        best_values.append(current_best)
    
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # 1. Optimization history
    ax1 = axes[0]
    trial_numbers = range(1, len(values) + 1)
    ax1.scatter(trial_numbers, values, alpha=0.5, s=30, label='Trial values')
    ax1.plot(trial_numbers, best_values, 'r-', linewidth=2, label='Best value')
    
    # Mark best trial
    best_trial_idx = study.best_trial.number
    ax1.scatter(best_trial_idx + 1, study.best_value, color='green', 
               s=200, zorder=5, marker='*', 
               label=f'Best: {study.best_value:.4f}')
    
    ax1.set_xlabel('Trial')
    ax1.set_ylabel('Objective Value')
    ax1.set_title('Optimization Progress')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Convergence rate
    ax2 = axes[1]
    if len(best_values) > 10:
        improvements = []
        window = 10
        for i in range(window, len(best_values)):
            prev = best_values[i - window]
            curr = best_values[i]
            improvement = abs(curr - prev) / (abs(prev) + 1e-10)
            improvements.append(improvement)
        
        ax2.plot(range(window + 1, len(best_values) + 1), improvements, 
                'b-', linewidth=2)
        ax2.set_xlabel('Trial')
        ax2.set_ylabel('Relative Improvement (10-trial window)')
        ax2.set_title('Convergence Rate')
        ax2.grid(True, alpha=0.3)
        
        # Add convergence indicator
        if len(improvements) > 0 and improvements[-1] < 0.001:
            ax2.annotate('✓ Converged', xy=(0.7, 0.9), xycoords='axes fraction',
                        fontsize=12, color='green', weight='bold')
    
    # 3. Trial duration
    ax3 = axes[2]
    durations = [(t.datetime_complete - t.datetime_start).total_seconds() 
                 for t in trials if t.datetime_complete is not None]
    
    if durations:
        ax3.hist(durations, bins=20, alpha=0.7, color='purple', edgecolor='black')
        ax3.axvline(x=np.mean(durations), color='r', linestyle='--',
                   label=f'Mean: {np.mean(durations):.1f}s')
        ax3.set_xlabel('Duration (seconds)')
        ax3.set_ylabel('Count')
        ax3.set_title('Trial Duration Distribution')
        ax3.legend()
        ax3.grid(True, alpha=0.3, axis='y')
    
    fig.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.savefig(save_path.replace('.png', '.pdf'), bbox_inches='tight')
    
    plt.show()
    
    # Also save interactive plotly version
    if save_path:
        fig_plotly = optuna_plot_history(study)
        fig_plotly.write_html(save_path.replace('.png', '_interactive.html'))
    
    return {
        'n_trials': n_trials,
        'best_value': float(study.best_value),
        'best_trial': best_trial_idx,
        'convergence_rate': float(improvements[-1]) if improvements else None,
        'avg_trial_duration': float(np.mean(durations)) if durations else None,
        'total_duration': float(sum(durations)) if durations else None
    }


def plot_param_importances(
    study: optuna.Study,
    title: str = "Hyperparameter Importances",
    save_path: Optional[str] = None,
    figsize: Tuple[float, float] = (10, 6)
) -> Dict[str, float]:
    """
    Plot hyperparameter importances using fANOVA.
    
    Args:
        study: Optuna study object
        title: Plot title
        save_path: Path to save figure
        figsize: Figure size
        
    Returns:
        Dictionary with importance scores
    """
    try:
        # Calculate importances
        from optuna.importance import get_param_importances
        importances = get_param_importances(study)
        
        if not importances:
            print("No parameter importances could be calculated")
            return {}
        
        # Sort by importance
        sorted_params = sorted(importances.items(), key=lambda x: x[1], reverse=True)
        params = [p[0] for p in sorted_params]
        scores = [p[1] for p in sorted_params]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Bar plot
        y_pos = np.arange(len(params))
        bars = ax1.barh(y_pos, scores, alpha=0.8)
        
        # Color bars by importance
        colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(params)))[::-1]
        for bar, color in zip(bars, colors):
            bar.set_color(color)
        
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels(params)
        ax1.set_xlabel('Importance')
        ax1.set_title('Parameter Importances (fANOVA)')
        ax1.grid(True, alpha=0.3, axis='x')
        
        # Add value labels
        for i, (param, score) in enumerate(sorted_params):
            ax1.text(score, i, f' {score:.3f}', va='center')
        
        # Cumulative importance
        cumsum = np.cumsum(scores) / np.sum(scores)
        ax2.plot(range(1, len(scores) + 1), cumsum, 'b-o', linewidth=2, markersize=8)
        ax2.axhline(y=0.8, color='r', linestyle='--', 
                   label='80% threshold')
        ax2.fill_between(range(1, len(scores) + 1), 0, cumsum, alpha=0.3)
        
        # Find how many params needed for 80%
        n_important = np.argmax(cumsum >= 0.8) + 1
        ax2.axvline(x=n_important, color='g', linestyle='--',
                   label=f'{n_important} params for 80%')
        
        ax2.set_xlabel('Number of Parameters')
        ax2.set_ylabel('Cumulative Importance')
        ax2.set_title('Cumulative Parameter Importance')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_xticks(range(1, len(scores) + 1))
        
        fig.suptitle(title, fontsize=14, y=1.02)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
            plt.savefig(save_path.replace('.png', '.pdf'), bbox_inches='tight')
        
        plt.show()
        
        # Also save interactive plotly version
        if save_path:
            fig_plotly = optuna_plot_importances(study)
            fig_plotly.write_html(save_path.replace('.png', '_interactive.html'))
        
        return dict(sorted_params)
        
    except Exception as e:
        print(f"Could not calculate parameter importances: {e}")
        return {}


def plot_parallel_coordinate(
    study: optuna.Study,
    params: Optional[List[str]] = None,
    title: str = "Parallel Coordinate Plot",
    save_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Plot parallel coordinate plot for parameter relationships.
    
    Args:
        study: Optuna study object
        params: List of parameter names to include (None for all)
        title: Plot title
        save_path: Path to save figure
        
    Returns:
        Dictionary with plot info
    """
    try:
        fig = optuna_plot_parallel(study, params=params)
        fig.update_layout(
            title=title,
            height=600,
            font=dict(size=12)
        )
        
        if save_path:
            fig.write_html(save_path.replace('.png', '_parallel.html'))
            fig.write_image(save_path)
        
        fig.show()
        
        return {
            'n_params': len(params) if params else len(study.best_params),
            'n_trials': len(study.trials)
        }
        
    except Exception as e:
        print(f"Could not create parallel coordinate plot: {e}")
        return {}


def plot_contour(
    study: optuna.Study,
    params: Optional[List[str]] = None,
    title: str = "Parameter Contour Plot",
    save_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Plot contour plot for parameter interactions.
    
    Args:
        study: Optuna study object
        params: List of parameter names (max 2-3 for visualization)
        title: Plot title
        save_path: Path to save figure
        
    Returns:
        Dictionary with plot info
    """
    try:
        # Select top 2 params if not specified
        if params is None:
            try:
                from optuna.importance import get_param_importances
                importances = get_param_importances(study)
                if importances:
                    params = list(importances.keys())[:2]
                else:
                    params = list(study.best_params.keys())[:2]
            except:
                params = list(study.best_params.keys())[:2]
        
        if len(params) < 2:
            print("Need at least 2 parameters for contour plot")
            return {}
        
        fig = optuna_plot_contour(study, params=params[:2])
        fig.update_layout(
            title=title,
            height=600,
            font=dict(size=12)
        )
        
        if save_path:
            fig.write_html(save_path.replace('.png', '_contour.html'))
            fig.write_image(save_path)
        
        fig.show()
        
        return {
            'params_plotted': params[:2],
            'n_trials': len(study.trials)
        }
        
    except Exception as e:
        print(f"Could not create contour plot: {e}")
        return {}


def generate_pruning_report(
    study: optuna.Study,
    title: str = "Pruning Analysis Report",
    save_path: Optional[str] = None,
    figsize: Tuple[float, float] = (12, 8)
) -> Dict[str, Any]:
    """
    Generate comprehensive pruning analysis report.
    
    Args:
        study: Optuna study object
        title: Plot title
        save_path: Path to save figure
        figsize: Figure size
        
    Returns:
        Dictionary with pruning statistics
    """
    trials = study.trials
    
    # Categorize trials
    completed_trials = [t for t in trials if t.state == optuna.trial.TrialState.COMPLETE]
    pruned_trials = [t for t in trials if t.state == optuna.trial.TrialState.PRUNED]
    failed_trials = [t for t in trials if t.state == optuna.trial.TrialState.FAIL]
    
    n_completed = len(completed_trials)
    n_pruned = len(pruned_trials)
    n_failed = len(failed_trials)
    n_total = len(trials)
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # 1. Trial state distribution
    ax1 = axes[0, 0]
    states = ['Completed', 'Pruned', 'Failed']
    counts = [n_completed, n_pruned, n_failed]
    colors = ['green', 'orange', 'red']
    
    wedges, texts, autotexts = ax1.pie(counts, labels=states, colors=colors,
                                        autopct='%1.1f%%', startangle=90)
    ax1.set_title(f'Trial State Distribution\n(Total: {n_total} trials)')
    
    # 2. Pruning efficiency
    ax2 = axes[0, 1]
    if n_pruned > 0:
        # Calculate time saved by pruning
        completed_durations = [(t.datetime_complete - t.datetime_start).total_seconds()
                              for t in completed_trials if t.datetime_complete]
        pruned_durations = [(t.datetime_complete - t.datetime_start).total_seconds()
                           for t in pruned_trials if t.datetime_complete]
        
        if completed_durations and pruned_durations:
            avg_completed = np.mean(completed_durations)
            avg_pruned = np.mean(pruned_durations)
            time_saved_per_trial = avg_completed - avg_pruned
            total_time_saved = time_saved_per_trial * n_pruned
            
            categories = ['Avg Completed', 'Avg Pruned', 'Time Saved']
            values = [avg_completed, avg_pruned, time_saved_per_trial]
            bars = ax2.bar(categories, values, color=['blue', 'orange', 'green'])
            
            for bar, val in zip(bars, values):
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                        f'{val:.1f}s', ha='center', va='bottom')
            
            ax2.set_ylabel('Time (seconds)')
            ax2.set_title(f'Pruning Efficiency\nTotal saved: {total_time_saved:.1f}s')
        else:
            ax2.text(0.5, 0.5, 'No timing data available', 
                    ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title('Pruning Efficiency')
    else:
        ax2.text(0.5, 0.5, 'No pruned trials', 
                ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title('Pruning Efficiency')
    
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 3. Pruning timeline
    ax3 = axes[1, 0]
    trial_numbers = []
    trial_states = []
    
    for t in trials:
        trial_numbers.append(t.number)
        if t.state == optuna.trial.TrialState.COMPLETE:
            trial_states.append(1)
        elif t.state == optuna.trial.TrialState.PRUNED:
            trial_states.append(0.5)
        else:
            trial_states.append(0)
    
    ax3.scatter(trial_numbers, trial_states, c=trial_states, 
               cmap='RdYlGn', s=30, alpha=0.7)
    ax3.set_xlabel('Trial Number')
    ax3.set_ylabel('State')
    ax3.set_yticks([0, 0.5, 1])
    ax3.set_yticklabels(['Failed', 'Pruned', 'Completed'])
    ax3.set_title('Trial States Over Time')
    ax3.grid(True, alpha=0.3)
    
    # 4. Pruning statistics text
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    stats_text = f"""
    Pruning Statistics:
    ─────────────────
    Total Trials: {n_total}
    Completed: {n_completed} ({n_completed/n_total*100:.1f}%)
    Pruned: {n_pruned} ({n_pruned/n_total*100:.1f}%)
    Failed: {n_failed} ({n_failed/n_total*100:.1f}%)
    
    Pruning Rate: {n_pruned/(n_completed + n_pruned)*100:.1f}%
    """
    
    if n_pruned > 0 and completed_durations and pruned_durations:
        stats_text += f"""
    Time Saved: {total_time_saved:.1f}s
    Efficiency: {time_saved_per_trial/avg_completed*100:.1f}%
    """
    
    # Check observation_key usage
    if hasattr(study, 'user_attrs') and 'observation_key' in study.user_attrs:
        obs_key = study.user_attrs['observation_key']
        stats_text += f"""
    
    ✓ Observation Key: {obs_key}
    """
    else:
        stats_text += """
    
    ⚠ No observation_key found
    (Pruning may not be working correctly)
    """
    
    ax4.text(0.1, 0.9, stats_text, transform=ax4.transAxes,
            fontsize=11, verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    fig.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.savefig(save_path.replace('.png', '.pdf'), bbox_inches='tight')
    
    plt.show()
    
    return {
        'n_total': n_total,
        'n_completed': n_completed,
        'n_pruned': n_pruned,
        'n_failed': n_failed,
        'pruning_rate': n_pruned / (n_completed + n_pruned) if (n_completed + n_pruned) > 0 else 0,
        'time_saved': total_time_saved if n_pruned > 0 and completed_durations and pruned_durations else 0,
        'efficiency': time_saved_per_trial / avg_completed if n_pruned > 0 and completed_durations and pruned_durations else 0
    }