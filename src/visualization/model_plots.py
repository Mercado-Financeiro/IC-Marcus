"""
Model discrimination and calibration plots.
Professional-grade visualizations for model evaluation.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    precision_recall_curve, roc_curve, auc, 
    confusion_matrix, brier_score_loss
)
from sklearn.calibration import calibration_curve
from scipy import stats
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')

# Set publication-quality defaults
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 16


def plot_pr_curve_with_baseline(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    title: str = "Precision-Recall Curve",
    save_path: Optional[str] = None,
    figsize: Tuple[float, float] = (10, 8)
) -> Dict[str, float]:
    """
    Plot Precision-Recall curve with prevalence baseline.
    
    Args:
        y_true: True binary labels
        y_proba: Predicted probabilities
        title: Plot title
        save_path: Path to save figure
        figsize: Figure size
        
    Returns:
        Dictionary with PR-AUC and prevalence
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    pr_auc = auc(recall, precision)
    prevalence = y_true.mean()
    
    # Calculate lift
    lift = pr_auc / prevalence if prevalence > 0 else 0
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Main PR curve
    ax1.plot(recall, precision, 'b-', linewidth=2, 
             label=f'Model (AUC = {pr_auc:.3f})')
    ax1.axhline(y=prevalence, color='r', linestyle='--', linewidth=1.5,
                label=f'No-Skill Baseline ({prevalence:.3f})')
    ax1.fill_between(recall, precision, prevalence, 
                     where=(precision >= prevalence),
                     color='green', alpha=0.2, label='Above baseline')
    ax1.fill_between(recall, precision, prevalence,
                     where=(precision < prevalence),
                     color='red', alpha=0.2, label='Below baseline')
    
    ax1.set_xlabel('Recall')
    ax1.set_ylabel('Precision')
    ax1.set_title(f'{title}\nLift = {lift:.2f}x')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([0, 1])
    ax1.set_ylim([0, 1])
    
    # F1 score by threshold
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
    f1_scores = np.nan_to_num(f1_scores)
    
    # Skip last threshold (always 1.0)
    if len(thresholds) > 0:
        ax2.plot(thresholds, f1_scores[:-1], 'g-', linewidth=2)
        best_idx = np.argmax(f1_scores[:-1])
        ax2.axvline(x=thresholds[best_idx], color='r', linestyle='--',
                   label=f'Best F1 @ {thresholds[best_idx]:.3f}')
        ax2.scatter(thresholds[best_idx], f1_scores[best_idx], 
                   color='red', s=100, zorder=5)
    
    ax2.set_xlabel('Threshold')
    ax2.set_ylabel('F1 Score')
    ax2.set_title(f'F1 Score vs Threshold\nMax F1 = {np.max(f1_scores):.3f}')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.savefig(save_path.replace('.png', '.pdf'), bbox_inches='tight')
    
    plt.show()
    
    return {
        'pr_auc': pr_auc,
        'prevalence': prevalence,
        'lift': lift,
        'max_f1': np.max(f1_scores),
        'best_threshold': thresholds[best_idx] if len(thresholds) > 0 else 0.5
    }


def plot_pr_auc_distribution(
    pr_auc_scores: List[float],
    baseline: float,
    title: str = "PR-AUC Distribution Across Splits",
    save_path: Optional[str] = None,
    figsize: Tuple[float, float] = (12, 6)
) -> Dict[str, float]:
    """
    Plot PR-AUC distribution using boxplot and violin plot.
    
    Args:
        pr_auc_scores: List of PR-AUC scores from different splits
        baseline: Baseline (prevalence) for comparison
        title: Plot title
        save_path: Path to save figure
        figsize: Figure size
        
    Returns:
        Dictionary with statistics
    """
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=figsize)
    
    # Boxplot
    bp = ax1.boxplot(pr_auc_scores, vert=True, patch_artist=True)
    bp['boxes'][0].set_facecolor('lightblue')
    ax1.axhline(y=baseline, color='r', linestyle='--', 
                label=f'Baseline ({baseline:.3f})')
    ax1.set_ylabel('PR-AUC')
    ax1.set_title('Boxplot')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Violin plot
    parts = ax2.violinplot(pr_auc_scores, vert=True, showmeans=True, 
                           showmedians=True, showextrema=True)
    for pc in parts['bodies']:
        pc.set_facecolor('lightgreen')
        pc.set_alpha(0.7)
    ax2.axhline(y=baseline, color='r', linestyle='--',
                label=f'Baseline ({baseline:.3f})')
    ax2.set_ylabel('PR-AUC')
    ax2.set_title('Violin Plot')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Distribution over time/splits
    ax3.plot(range(1, len(pr_auc_scores) + 1), pr_auc_scores, 
             'b-o', linewidth=2, markersize=8)
    ax3.axhline(y=baseline, color='r', linestyle='--',
                label=f'Baseline ({baseline:.3f})')
    ax3.fill_between(range(1, len(pr_auc_scores) + 1),
                     baseline, pr_auc_scores,
                     where=np.array(pr_auc_scores) > baseline,
                     color='green', alpha=0.3, label='Above baseline')
    ax3.fill_between(range(1, len(pr_auc_scores) + 1),
                     baseline, pr_auc_scores,
                     where=np.array(pr_auc_scores) <= baseline,
                     color='red', alpha=0.3, label='Below baseline')
    ax3.set_xlabel('Split/Window')
    ax3.set_ylabel('PR-AUC')
    ax3.set_title('Temporal Stability')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    fig.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.savefig(save_path.replace('.png', '.pdf'), bbox_inches='tight')
    
    plt.show()
    
    return {
        'mean': np.mean(pr_auc_scores),
        'std': np.std(pr_auc_scores),
        'median': np.median(pr_auc_scores),
        'q25': np.percentile(pr_auc_scores, 25),
        'q75': np.percentile(pr_auc_scores, 75),
        'min': np.min(pr_auc_scores),
        'max': np.max(pr_auc_scores),
        'cv': np.std(pr_auc_scores) / np.mean(pr_auc_scores)
    }


def plot_roc_curve(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    title: str = "ROC Curve",
    save_path: Optional[str] = None,
    figsize: Tuple[float, float] = (8, 8)
) -> Dict[str, float]:
    """
    Plot ROC curve (as supporting evidence, PR is primary for imbalanced data).
    
    Args:
        y_true: True binary labels
        y_proba: Predicted probabilities
        title: Plot title
        save_path: Path to save figure
        figsize: Figure size
        
    Returns:
        Dictionary with ROC-AUC
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.plot(fpr, tpr, 'b-', linewidth=2,
            label=f'Model (AUC = {roc_auc:.3f})')
    ax.plot([0, 1], [0, 1], 'r--', linewidth=1.5,
            label='Random (AUC = 0.500)')
    ax.fill_between(fpr, 0, tpr, alpha=0.2, color='blue')
    
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(f'{title}\n(Note: PR curve is primary for imbalanced data)')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_aspect('equal')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.savefig(save_path.replace('.png', '.pdf'), bbox_inches='tight')
    
    plt.show()
    
    return {'roc_auc': roc_auc}


def plot_reliability_diagram(
    y_true: np.ndarray,
    y_proba_uncalibrated: np.ndarray,
    y_proba_calibrated: Optional[np.ndarray] = None,
    n_bins: int = 10,
    title: str = "Reliability Diagram",
    save_path: Optional[str] = None,
    figsize: Tuple[float, float] = (12, 6)
) -> Dict[str, float]:
    """
    Plot reliability diagram with ECE and MCE metrics.
    
    Args:
        y_true: True binary labels
        y_proba_uncalibrated: Uncalibrated probabilities
        y_proba_calibrated: Calibrated probabilities (optional)
        n_bins: Number of bins
        title: Plot title
        save_path: Path to save figure
        figsize: Figure size
        
    Returns:
        Dictionary with ECE and MCE values
    """
    def calculate_ece_mce(y_true, y_proba, n_bins):
        fraction_pos, mean_pred = calibration_curve(
            y_true, y_proba, n_bins=n_bins, strategy='uniform'
        )
        
        bin_edges = np.linspace(0, 1, n_bins + 1)
        bin_sizes = np.histogram(y_proba, bins=bin_edges)[0]
        
        ece = 0.0
        mce = 0.0
        total_samples = len(y_proba)
        
        gaps = []
        for i in range(len(fraction_pos)):
            if bin_sizes[i] > 0:
                weight = bin_sizes[i] / total_samples
                gap = abs(fraction_pos[i] - mean_pred[i])
                ece += weight * gap
                mce = max(mce, gap)
                gaps.append(gap)
        
        return fraction_pos, mean_pred, ece, mce, gaps
    
    fig, axes = plt.subplots(1, 2 if y_proba_calibrated is not None else 1,
                             figsize=figsize)
    
    if y_proba_calibrated is None:
        axes = [axes]
    
    # Uncalibrated
    frac_pos_uncal, mean_pred_uncal, ece_uncal, mce_uncal, gaps_uncal = \
        calculate_ece_mce(y_true, y_proba_uncalibrated, n_bins)
    
    axes[0].plot([0, 1], [0, 1], 'r--', linewidth=2, label='Perfect calibration')
    axes[0].plot(mean_pred_uncal, frac_pos_uncal, 'b-o', linewidth=2,
                markersize=8, label='Uncalibrated')
    
    # Add error bars
    for i, (x, y, gap) in enumerate(zip(mean_pred_uncal, frac_pos_uncal, gaps_uncal)):
        axes[0].plot([x, x], [x, y], 'gray', alpha=0.5, linewidth=1)
    
    axes[0].set_xlabel('Mean Predicted Probability')
    axes[0].set_ylabel('Fraction of Positives')
    axes[0].set_title(f'Uncalibrated\nECE = {ece_uncal:.4f}, MCE = {mce_uncal:.4f}')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlim([0, 1])
    axes[0].set_ylim([0, 1])
    axes[0].set_aspect('equal')
    
    results = {
        'ece_uncalibrated': ece_uncal,
        'mce_uncalibrated': mce_uncal
    }
    
    # Calibrated
    if y_proba_calibrated is not None:
        frac_pos_cal, mean_pred_cal, ece_cal, mce_cal, gaps_cal = \
            calculate_ece_mce(y_true, y_proba_calibrated, n_bins)
        
        axes[1].plot([0, 1], [0, 1], 'r--', linewidth=2, label='Perfect calibration')
        axes[1].plot(mean_pred_cal, frac_pos_cal, 'g-o', linewidth=2,
                    markersize=8, label='Calibrated')
        
        # Add error bars
        for i, (x, y, gap) in enumerate(zip(mean_pred_cal, frac_pos_cal, gaps_cal)):
            axes[1].plot([x, x], [x, y], 'gray', alpha=0.5, linewidth=1)
        
        # Add improvement arrow
        improvement = ((ece_uncal - ece_cal) / ece_uncal) * 100
        axes[1].annotate(f'↓ {improvement:.1f}% improvement',
                        xy=(0.7, 0.2), fontsize=12, color='green',
                        weight='bold')
        
        axes[1].set_xlabel('Mean Predicted Probability')
        axes[1].set_ylabel('Fraction of Positives')
        axes[1].set_title(f'Calibrated\nECE = {ece_cal:.4f}, MCE = {mce_cal:.4f}')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        axes[1].set_xlim([0, 1])
        axes[1].set_ylim([0, 1])
        axes[1].set_aspect('equal')
        
        results['ece_calibrated'] = ece_cal
        results['mce_calibrated'] = mce_cal
        results['ece_improvement'] = improvement
    
    fig.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.savefig(save_path.replace('.png', '.pdf'), bbox_inches='tight')
    
    plt.show()
    
    return results


def plot_brier_score_comparison(
    y_true: np.ndarray,
    predictions: Dict[str, np.ndarray],
    title: str = "Brier Score Comparison",
    save_path: Optional[str] = None,
    figsize: Tuple[float, float] = (10, 6)
) -> Dict[str, float]:
    """
    Plot Brier score comparison with baseline.
    
    Args:
        y_true: True binary labels
        predictions: Dictionary of model_name -> probabilities
        title: Plot title
        save_path: Path to save figure
        figsize: Figure size
        
    Returns:
        Dictionary with Brier scores and skill scores
    """
    prevalence = y_true.mean()
    brier_baseline = prevalence * (1 - prevalence)
    
    results = {}
    models = []
    brier_scores = []
    skill_scores = []
    
    for model_name, y_proba in predictions.items():
        brier = brier_score_loss(y_true, y_proba)
        skill = 1 - (brier / brier_baseline)
        
        models.append(model_name)
        brier_scores.append(brier)
        skill_scores.append(skill)
        
        results[f'brier_{model_name}'] = brier
        results[f'skill_{model_name}'] = skill
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Brier scores
    x_pos = np.arange(len(models))
    bars1 = ax1.bar(x_pos, brier_scores, alpha=0.8, color='steelblue')
    ax1.axhline(y=brier_baseline, color='r', linestyle='--',
                label=f'Baseline ({brier_baseline:.4f})')
    
    # Add value labels on bars
    for bar, score in zip(bars1, brier_scores):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{score:.4f}', ha='center', va='bottom')
    
    ax1.set_xlabel('Model')
    ax1.set_ylabel('Brier Score (lower is better)')
    ax1.set_title('Brier Score Comparison')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(models, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Skill scores
    bars2 = ax2.bar(x_pos, skill_scores, alpha=0.8, color='green')
    ax2.axhline(y=0, color='r', linestyle='--', label='No skill')
    
    # Add value labels on bars
    for bar, score in zip(bars2, skill_scores):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{score:.3f}', ha='center', 
                va='bottom' if score > 0 else 'top')
    
    ax2.set_xlabel('Model')
    ax2.set_ylabel('Brier Skill Score (higher is better)')
    ax2.set_title('Brier Skill Score')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(models, rotation=45, ha='right')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    fig.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.savefig(save_path.replace('.png', '.pdf'), bbox_inches='tight')
    
    plt.show()
    
    results['brier_baseline'] = brier_baseline
    return results


def plot_calibrator_comparison(
    y_true: np.ndarray,
    calibrated_predictions: Dict[str, np.ndarray],
    title: str = "Calibration Method Comparison",
    save_path: Optional[str] = None,
    figsize: Tuple[float, float] = (12, 6)
) -> Dict[str, float]:
    """
    Compare different calibration methods (Platt, Isotonic, Beta).
    
    Args:
        y_true: True binary labels
        calibrated_predictions: Dict of method_name -> calibrated probabilities
        title: Plot title
        save_path: Path to save figure
        figsize: Figure size
        
    Returns:
        Dictionary with comparison metrics
    """
    methods = list(calibrated_predictions.keys())
    n_methods = len(methods)
    
    # Calculate metrics for each method
    ece_scores = []
    brier_scores = []
    
    for method, y_proba in calibrated_predictions.items():
        # ECE calculation
        fraction_pos, mean_pred = calibration_curve(
            y_true, y_proba, n_bins=10, strategy='uniform'
        )
        bin_edges = np.linspace(0, 1, 11)
        bin_sizes = np.histogram(y_proba, bins=bin_edges)[0]
        
        ece = 0.0
        total_samples = len(y_proba)
        for i in range(len(fraction_pos)):
            if bin_sizes[i] > 0:
                weight = bin_sizes[i] / total_samples
                gap = abs(fraction_pos[i] - mean_pred[i])
                ece += weight * gap
        
        ece_scores.append(ece)
        brier_scores.append(brier_score_loss(y_true, y_proba))
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # ECE comparison
    x_pos = np.arange(len(methods))
    bars1 = ax1.bar(x_pos, ece_scores, alpha=0.8, 
                    color=['red', 'blue', 'green', 'orange'][:n_methods])
    ax1.axhline(y=0.05, color='r', linestyle='--',
                label='ECE threshold (0.05)')
    
    for bar, score in zip(bars1, ece_scores):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{score:.4f}', ha='center', va='bottom')
    
    ax1.set_xlabel('Calibration Method')
    ax1.set_ylabel('ECE (lower is better)')
    ax1.set_title('Expected Calibration Error')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(methods)
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Brier score comparison
    bars2 = ax2.bar(x_pos, brier_scores, alpha=0.8,
                    color=['red', 'blue', 'green', 'orange'][:n_methods])
    
    for bar, score in zip(bars2, brier_scores):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{score:.4f}', ha='center', va='bottom')
    
    ax2.set_xlabel('Calibration Method')
    ax2.set_ylabel('Brier Score (lower is better)')
    ax2.set_title('Brier Score')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(methods)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Highlight best method
    best_method_idx = np.argmin(ece_scores)
    ax1.patches[best_method_idx].set_edgecolor('black')
    ax1.patches[best_method_idx].set_linewidth(3)
    ax2.patches[best_method_idx].set_edgecolor('black')
    ax2.patches[best_method_idx].set_linewidth(3)
    
    fig.suptitle(f'{title}\nBest: {methods[best_method_idx]} (Beta typically includes identity)',
                fontsize=14, y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.savefig(save_path.replace('.png', '.pdf'), bbox_inches='tight')
    
    plt.show()
    
    return {
        f'ece_{method}': ece for method, ece in zip(methods, ece_scores)
    } | {
        f'brier_{method}': brier for method, brier in zip(methods, brier_scores)
    } | {
        'best_method': methods[best_method_idx]
    }


def plot_ev_curve(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    costs: Dict[str, float] = None,
    title: str = "Expected Value Curve",
    save_path: Optional[str] = None,
    figsize: Tuple[float, float] = (10, 6)
) -> Dict[str, Any]:
    """
    Plot Expected Value curve with costs.
    
    Args:
        y_true: True binary labels
        y_proba: Predicted probabilities
        costs: Dictionary with 'fee', 'slippage', 'reward' keys
        title: Plot title
        save_path: Path to save figure
        figsize: Figure size
        
    Returns:
        Dictionary with optimal threshold and EV
    """
    if costs is None:
        costs = {
            'fee': 0.001,      # 0.1% trading fee
            'slippage': 0.0005, # 0.05% slippage
            'reward': 0.01      # 1% average profit on successful trades
        }
    
    thresholds = np.linspace(0, 1, 101)
    expected_values = []
    precisions = []
    recalls = []
    
    for threshold in thresholds:
        y_pred = (y_proba >= threshold).astype(int)
        
        tp = np.sum((y_pred == 1) & (y_true == 1))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        fn = np.sum((y_pred == 0) & (y_true == 1))
        tn = np.sum((y_pred == 0) & (y_true == 0))
        
        n_trades = tp + fp
        if n_trades > 0:
            # Expected value per trade
            success_rate = tp / n_trades
            ev = (success_rate * costs['reward']) - \
                 costs['fee'] - costs['slippage']
            
            precision = tp / (tp + fp)
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        else:
            ev = 0
            precision = 0
            recall = 0
        
        expected_values.append(ev)
        precisions.append(precision)
        recalls.append(recall)
    
    expected_values = np.array(expected_values)
    best_idx = np.argmax(expected_values)
    best_threshold = thresholds[best_idx]
    best_ev = expected_values[best_idx]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # EV curve
    ax1.plot(thresholds, expected_values, 'b-', linewidth=2)
    ax1.axhline(y=0, color='r', linestyle='--', label='Break-even')
    ax1.axvline(x=best_threshold, color='g', linestyle='--',
                label=f'Optimal τ = {best_threshold:.3f}')
    ax1.scatter(best_threshold, best_ev, color='red', s=100, zorder=5)
    ax1.fill_between(thresholds, 0, expected_values,
                     where=(expected_values > 0),
                     color='green', alpha=0.3, label='Profitable')
    ax1.fill_between(thresholds, 0, expected_values,
                     where=(expected_values <= 0),
                     color='red', alpha=0.3, label='Loss')
    
    ax1.set_xlabel('Threshold (τ)')
    ax1.set_ylabel('Expected Value per Trade')
    ax1.set_title(f'EV Curve\nMax EV = {best_ev:.4f} @ τ = {best_threshold:.3f}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Precision-Recall at different thresholds
    ax2.plot(thresholds, precisions, 'g-', label='Precision', linewidth=2)
    ax2.plot(thresholds, recalls, 'b-', label='Recall', linewidth=2)
    ax2.axvline(x=best_threshold, color='r', linestyle='--',
                label=f'Optimal τ = {best_threshold:.3f}')
    ax2.set_xlabel('Threshold (τ)')
    ax2.set_ylabel('Score')
    ax2.set_title(f'Precision/Recall vs Threshold\n' +
                  f'P={precisions[best_idx]:.3f}, R={recalls[best_idx]:.3f}')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    fig.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.savefig(save_path.replace('.png', '.pdf'), bbox_inches='tight')
    
    plt.show()
    
    return {
        'optimal_threshold': best_threshold,
        'max_ev': best_ev,
        'precision_at_optimal': precisions[best_idx],
        'recall_at_optimal': recalls[best_idx],
        'profitable_range': (thresholds[expected_values > 0].min(),
                           thresholds[expected_values > 0].max()) 
                           if any(expected_values > 0) else (None, None)
    }


def plot_confusion_matrix_heatmap(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    threshold: Optional[float] = None,
    y_proba: Optional[np.ndarray] = None,
    title: str = "Confusion Matrix",
    save_path: Optional[str] = None,
    figsize: Tuple[float, float] = (8, 6)
) -> Dict[str, float]:
    """
    Plot confusion matrix as heatmap.
    
    Args:
        y_true: True binary labels
        y_pred: Predicted labels or None if using y_proba
        threshold: Threshold to use with y_proba
        y_proba: Predicted probabilities (optional)
        title: Plot title
        save_path: Path to save figure
        figsize: Figure size
        
    Returns:
        Dictionary with classification metrics
    """
    if y_pred is None and y_proba is not None and threshold is not None:
        y_pred = (y_proba >= threshold).astype(int)
    
    cm = confusion_matrix(y_true, y_pred)
    
    # Calculate metrics
    tn, fp, fn, tp = cm.ravel()
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # Matthews Correlation Coefficient
    mcc_num = (tp * tn) - (fp * fn)
    mcc_den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = mcc_num / mcc_den if mcc_den > 0 else 0
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create heatmap
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Predicted 0', 'Predicted 1'],
                yticklabels=['Actual 0', 'Actual 1'],
                cbar_kws={'label': 'Count'},
                ax=ax, square=True, linewidths=1, linecolor='black')
    
    # Add percentages
    for i in range(2):
        for j in range(2):
            percentage = cm[i, j] / cm.sum() * 100
            text = ax.texts[i * 2 + j]
            text.set_text(f'{cm[i, j]}\n({percentage:.1f}%)')
    
    if threshold is not None:
        ax.set_title(f'{title} @ τ = {threshold:.3f}\n' +
                    f'Acc={accuracy:.3f}, P={precision:.3f}, R={recall:.3f}, ' +
                    f'F1={f1:.3f}, MCC={mcc:.3f}')
    else:
        ax.set_title(f'{title}\n' +
                    f'Acc={accuracy:.3f}, P={precision:.3f}, R={recall:.3f}, ' +
                    f'F1={f1:.3f}, MCC={mcc:.3f}')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.savefig(save_path.replace('.png', '.pdf'), bbox_inches='tight')
    
    plt.show()
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'specificity': specificity,
        'f1_score': f1,
        'mcc': mcc,
        'tp': int(tp),
        'fp': int(fp),
        'tn': int(tn),
        'fn': int(fn)
    }


def plot_learning_curves(
    train_losses: List[float],
    val_losses: List[float],
    train_metrics: Optional[List[float]] = None,
    val_metrics: Optional[List[float]] = None,
    metric_name: str = "PR-AUC",
    title: str = "Learning Curves",
    save_path: Optional[str] = None,
    figsize: Tuple[float, float] = (12, 5)
) -> Dict[str, Any]:
    """
    Plot LSTM learning curves.
    
    Args:
        train_losses: Training losses per epoch
        val_losses: Validation losses per epoch
        train_metrics: Training metrics per epoch (optional)
        val_metrics: Validation metrics per epoch (optional)
        metric_name: Name of the metric
        title: Plot title
        save_path: Path to save figure
        figsize: Figure size
        
    Returns:
        Dictionary with convergence info
    """
    epochs = range(1, len(train_losses) + 1)
    
    fig, axes = plt.subplots(1, 2 if train_metrics is not None else 1,
                             figsize=figsize)
    
    if train_metrics is None:
        axes = [axes]
    
    # Loss curves
    ax1 = axes[0]
    ax1.plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=2)
    ax1.plot(epochs, val_losses, 'r-', label='Val Loss', linewidth=2)
    
    # Mark best validation loss
    best_val_epoch = np.argmin(val_losses) + 1
    ax1.scatter(best_val_epoch, val_losses[best_val_epoch - 1],
               color='green', s=100, zorder=5,
               label=f'Best Val @ {best_val_epoch}')
    
    # Check for overfitting
    if len(val_losses) > 10:
        recent_val_trend = np.polyfit(range(len(val_losses) - 10, len(val_losses)),
                                      val_losses[-10:], 1)[0]
        if recent_val_trend > 0:
            ax1.annotate('⚠ Overfitting detected', xy=(len(epochs) * 0.7, max(val_losses) * 0.9),
                        color='red', fontsize=12, weight='bold')
    
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Loss Curves')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    results = {
        'best_epoch': best_val_epoch,
        'best_val_loss': val_losses[best_val_epoch - 1],
        'final_train_loss': train_losses[-1],
        'final_val_loss': val_losses[-1]
    }
    
    # Metric curves
    if train_metrics is not None and val_metrics is not None:
        ax2 = axes[1]
        ax2.plot(epochs, train_metrics, 'b-', label=f'Train {metric_name}', linewidth=2)
        ax2.plot(epochs, val_metrics, 'r-', label=f'Val {metric_name}', linewidth=2)
        
        # Mark best validation metric
        best_metric_epoch = np.argmax(val_metrics) + 1
        ax2.scatter(best_metric_epoch, val_metrics[best_metric_epoch - 1],
                   color='green', s=100, zorder=5,
                   label=f'Best Val @ {best_metric_epoch}')
        
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel(metric_name)
        ax2.set_title(f'{metric_name} Curves')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        results['best_metric_epoch'] = best_metric_epoch
        results[f'best_val_{metric_name.lower()}'] = val_metrics[best_metric_epoch - 1]
    
    fig.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.savefig(save_path.replace('.png', '.pdf'), bbox_inches='tight')
    
    plt.show()
    
    return results


def plot_mc_dropout_uncertainty(
    predictions: np.ndarray,
    title: str = "MC Dropout Uncertainty Analysis",
    save_path: Optional[str] = None,
    figsize: Tuple[float, float] = (12, 6)
) -> Dict[str, float]:
    """
    Plot MC Dropout uncertainty analysis.
    
    Args:
        predictions: Array of shape (n_samples, n_forward_passes)
        title: Plot title
        save_path: Path to save figure
        figsize: Figure size
        
    Returns:
        Dictionary with uncertainty statistics
    """
    mean_pred = predictions.mean(axis=1)
    std_pred = predictions.std(axis=1)
    cv_pred = std_pred / (mean_pred + 1e-10)
    
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # Histogram of uncertainty (std)
    axes[0].hist(std_pred, bins=30, alpha=0.7, color='blue', edgecolor='black')
    axes[0].axvline(x=np.median(std_pred), color='r', linestyle='--',
                   label=f'Median = {np.median(std_pred):.4f}')
    axes[0].set_xlabel('Prediction Std Dev')
    axes[0].set_ylabel('Count')
    axes[0].set_title('Uncertainty Distribution')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Mean vs Uncertainty scatter
    scatter = axes[1].scatter(mean_pred, std_pred, c=cv_pred, 
                             cmap='RdYlBu_r', alpha=0.6, s=20)
    axes[1].set_xlabel('Mean Prediction')
    axes[1].set_ylabel('Std Dev')
    axes[1].set_title('Prediction vs Uncertainty')
    axes[1].grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=axes[1], label='CV')
    
    # High uncertainty samples
    high_uncertainty_threshold = np.percentile(std_pred, 90)
    high_uncertainty_mask = std_pred > high_uncertainty_threshold
    
    axes[2].boxplot([std_pred[~high_uncertainty_mask],
                    std_pred[high_uncertainty_mask]],
                   labels=['Normal', 'High Uncertainty'])
    axes[2].set_ylabel('Prediction Std Dev')
    axes[2].set_title(f'Uncertainty Groups\n' +
                     f'{high_uncertainty_mask.sum()} samples > P90')
    axes[2].grid(True, alpha=0.3, axis='y')
    
    fig.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.savefig(save_path.replace('.png', '.pdf'), bbox_inches='tight')
    
    plt.show()
    
    return {
        'mean_uncertainty': float(std_pred.mean()),
        'median_uncertainty': float(np.median(std_pred)),
        'p90_uncertainty': float(high_uncertainty_threshold),
        'high_uncertainty_samples': int(high_uncertainty_mask.sum()),
        'total_samples': len(std_pred),
        'mean_cv': float(cv_pred.mean())
    }