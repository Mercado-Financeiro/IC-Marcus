"""
Calibration metrics for model evaluation.

Includes ECE (Expected Calibration Error), MCE (Maximum Calibration Error),
and other calibration diagnostics.

References:
- "On Calibration of Modern Neural Networks" (Guo et al., 2017)
- "Verified Uncertainty Calibration" (Kumar et al., 2019)
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional, Dict, List
from sklearn.metrics import brier_score_loss
import warnings


def expected_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 15,
    strategy: str = 'uniform'
) -> Tuple[float, Dict]:
    """
    Calculate Expected Calibration Error (ECE).
    
    ECE measures the expected difference between confidence and accuracy.
    Lower ECE indicates better calibration.
    
    Args:
        y_true: True binary labels
        y_prob: Predicted probabilities
        n_bins: Number of bins for discretization
        strategy: Binning strategy ('uniform' or 'quantile')
        
    Returns:
        Tuple of (ECE value, detailed statistics per bin)
    """
    if strategy == 'uniform':
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
    elif strategy == 'quantile':
        # Use quantiles for adaptive binning
        quantiles = np.linspace(0, 100, n_bins + 1)
        bin_boundaries = np.percentile(y_prob, quantiles)
        bin_boundaries[0] = 0
        bin_boundaries[-1] = 1
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    
    ece = 0.0
    bin_stats = []
    total_samples = len(y_prob)
    
    for i in range(n_bins):
        # Get samples in this bin
        if i == n_bins - 1:
            # Include upper boundary in last bin
            bin_mask = (y_prob >= bin_boundaries[i]) & (y_prob <= bin_boundaries[i + 1])
        else:
            bin_mask = (y_prob >= bin_boundaries[i]) & (y_prob < bin_boundaries[i + 1])
        
        n_bin = bin_mask.sum()
        
        if n_bin > 0:
            # Average confidence in bin
            bin_confidence = y_prob[bin_mask].mean()
            # Accuracy in bin
            bin_accuracy = y_true[bin_mask].mean()
            # Contribution to ECE
            bin_weight = n_bin / total_samples
            bin_error = abs(bin_accuracy - bin_confidence)
            ece += bin_weight * bin_error
            
            bin_stats.append({
                'bin_id': i,
                'lower': bin_boundaries[i],
                'upper': bin_boundaries[i + 1],
                'confidence': bin_confidence,
                'accuracy': bin_accuracy,
                'error': bin_error,
                'count': n_bin,
                'weight': bin_weight
            })
    
    return ece, bin_stats


def maximum_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 15,
    strategy: str = 'uniform'
) -> Tuple[float, int]:
    """
    Calculate Maximum Calibration Error (MCE).
    
    MCE is the maximum difference between confidence and accuracy across all bins.
    
    Args:
        y_true: True binary labels
        y_prob: Predicted probabilities
        n_bins: Number of bins
        strategy: Binning strategy
        
    Returns:
        Tuple of (MCE value, bin index with maximum error)
    """
    _, bin_stats = expected_calibration_error(y_true, y_prob, n_bins, strategy)
    
    if not bin_stats:
        return 0.0, -1
    
    max_error = 0.0
    max_bin = -1
    
    for stat in bin_stats:
        if stat['error'] > max_error:
            max_error = stat['error']
            max_bin = stat['bin_id']
    
    return max_error, max_bin


def adaptive_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.05
) -> float:
    """
    Adaptive Calibration Error (ACE) with variable bin sizes.
    
    Uses adaptive binning to ensure each bin has sufficient samples.
    
    Args:
        y_true: True labels
        y_prob: Predicted probabilities
        threshold: Minimum fraction of samples per bin
        
    Returns:
        ACE value
    """
    n_samples = len(y_prob)
    min_samples = max(10, int(threshold * n_samples))
    
    # Sort probabilities
    sorted_indices = np.argsort(y_prob)
    sorted_probs = y_prob[sorted_indices]
    sorted_labels = y_true[sorted_indices]
    
    ace = 0.0
    i = 0
    
    while i < n_samples:
        # Determine bin size
        bin_size = min_samples
        while i + bin_size < n_samples and sorted_probs[i + bin_size] == sorted_probs[i + bin_size - 1]:
            bin_size += 1  # Extend bin to include all equal probabilities
        
        if i + bin_size > n_samples:
            bin_size = n_samples - i
        
        # Calculate error for this bin
        bin_probs = sorted_probs[i:i + bin_size]
        bin_labels = sorted_labels[i:i + bin_size]
        
        bin_confidence = bin_probs.mean()
        bin_accuracy = bin_labels.mean()
        bin_weight = bin_size / n_samples
        
        ace += bin_weight * abs(bin_accuracy - bin_confidence)
        
        i += bin_size
    
    return ace


def classwise_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 15
) -> Dict[str, float]:
    """
    Calculate calibration error for each class separately.
    
    Args:
        y_true: True labels
        y_prob: Predicted probabilities
        n_bins: Number of bins
        
    Returns:
        Dictionary with ECE for positive and negative classes
    """
    # Positive class calibration
    pos_mask = y_true == 1
    if pos_mask.sum() > 0:
        pos_ece, _ = expected_calibration_error(
            y_true[pos_mask],
            y_prob[pos_mask],
            n_bins=min(n_bins, pos_mask.sum() // 10)
        )
    else:
        pos_ece = np.nan
    
    # Negative class calibration (using 1 - prob)
    neg_mask = y_true == 0
    if neg_mask.sum() > 0:
        neg_ece, _ = expected_calibration_error(
            1 - y_true[neg_mask],
            1 - y_prob[neg_mask],
            n_bins=min(n_bins, neg_mask.sum() // 10)
        )
    else:
        neg_ece = np.nan
    
    return {
        'positive_ece': pos_ece,
        'negative_ece': neg_ece,
        'mean_ece': np.nanmean([pos_ece, neg_ece])
    }


def overconfidence_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 15
) -> float:
    """
    Calculate overconfidence error.
    
    Only considers bins where confidence > accuracy.
    
    Args:
        y_true: True labels
        y_prob: Predicted probabilities
        n_bins: Number of bins
        
    Returns:
        Overconfidence error
    """
    _, bin_stats = expected_calibration_error(y_true, y_prob, n_bins)
    
    oce = 0.0
    for stat in bin_stats:
        if stat['confidence'] > stat['accuracy']:
            oce += stat['weight'] * (stat['confidence'] - stat['accuracy'])
    
    return oce


def underconfidence_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 15
) -> float:
    """
    Calculate underconfidence error.
    
    Only considers bins where accuracy > confidence.
    
    Args:
        y_true: True labels
        y_prob: Predicted probabilities
        n_bins: Number of bins
        
    Returns:
        Underconfidence error
    """
    _, bin_stats = expected_calibration_error(y_true, y_prob, n_bins)
    
    uce = 0.0
    for stat in bin_stats:
        if stat['accuracy'] > stat['confidence']:
            uce += stat['weight'] * (stat['accuracy'] - stat['confidence'])
    
    return uce


def static_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 15
) -> float:
    """
    Static Calibration Error (SCE).
    
    Uses equal-mass bins (same number of samples per bin).
    
    Args:
        y_true: True labels
        y_prob: Predicted probabilities
        n_bins: Number of bins
        
    Returns:
        SCE value
    """
    n_samples = len(y_prob)
    samples_per_bin = n_samples // n_bins
    
    # Sort by probability
    sorted_indices = np.argsort(y_prob)
    sorted_probs = y_prob[sorted_indices]
    sorted_labels = y_true[sorted_indices]
    
    sce = 0.0
    
    for i in range(n_bins):
        start = i * samples_per_bin
        if i == n_bins - 1:
            end = n_samples  # Include remainder in last bin
        else:
            end = start + samples_per_bin
        
        bin_probs = sorted_probs[start:end]
        bin_labels = sorted_labels[start:end]
        
        bin_confidence = bin_probs.mean()
        bin_accuracy = bin_labels.mean()
        bin_weight = (end - start) / n_samples
        
        sce += bin_weight * abs(bin_accuracy - bin_confidence)
    
    return sce


def reliability_diagram_stats(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10
) -> Dict:
    """
    Compute statistics for reliability diagram.
    
    Args:
        y_true: True labels
        y_prob: Predicted probabilities
        n_bins: Number of bins
        
    Returns:
        Dictionary with bin statistics for plotting
    """
    _, bin_stats = expected_calibration_error(y_true, y_prob, n_bins, strategy='quantile')
    
    if not bin_stats:
        return {
            'bin_centers': np.array([]),
            'bin_accuracies': np.array([]),
            'bin_confidences': np.array([]),
            'bin_counts': np.array([]),
            'bin_edges': np.array([])
        }
    
    bin_centers = []
    bin_accuracies = []
    bin_confidences = []
    bin_counts = []
    bin_edges = [0]
    
    for stat in bin_stats:
        bin_centers.append((stat['lower'] + stat['upper']) / 2)
        bin_accuracies.append(stat['accuracy'])
        bin_confidences.append(stat['confidence'])
        bin_counts.append(stat['count'])
        bin_edges.append(stat['upper'])
    
    return {
        'bin_centers': np.array(bin_centers),
        'bin_accuracies': np.array(bin_accuracies),
        'bin_confidences': np.array(bin_confidences),
        'bin_counts': np.array(bin_counts),
        'bin_edges': np.array(bin_edges)
    }


def plot_reliability_diagram(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
    title: str = 'Reliability Diagram',
    ax: Optional[plt.Axes] = None
) -> plt.Axes:
    """
    Plot reliability diagram (calibration plot).
    
    Args:
        y_true: True labels
        y_prob: Predicted probabilities
        n_bins: Number of bins
        title: Plot title
        ax: Matplotlib axes (creates new if None)
        
    Returns:
        Matplotlib axes with plot
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    
    stats = reliability_diagram_stats(y_true, y_prob, n_bins)
    
    # Perfect calibration line
    ax.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
    
    # Actual calibration
    if len(stats['bin_centers']) > 0:
        ax.plot(stats['bin_confidences'], stats['bin_accuracies'], 
                'o-', label='Model calibration', markersize=8)
        
        # Add error bars based on bin counts
        errors = np.sqrt(stats['bin_accuracies'] * (1 - stats['bin_accuracies']) / 
                         np.maximum(stats['bin_counts'], 1))
        ax.errorbar(stats['bin_confidences'], stats['bin_accuracies'], 
                    yerr=errors, fmt='none', alpha=0.5, color='gray')
    
    # Histogram at bottom
    ax2 = ax.twinx()
    ax2.hist(y_prob, bins=stats['bin_edges'], alpha=0.3, edgecolor='black')
    ax2.set_ylabel('Count')
    
    ax.set_xlabel('Mean Predicted Probability')
    ax.set_ylabel('Fraction of Positives')
    ax.set_title(title)
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    
    # Add ECE to plot
    ece, _ = expected_calibration_error(y_true, y_prob, n_bins)
    ax.text(0.05, 0.95, f'ECE: {ece:.3f}', transform=ax.transAxes,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat'))
    
    return ax


def comprehensive_calibration_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 15
) -> Dict[str, float]:
    """
    Calculate comprehensive set of calibration metrics.
    
    Args:
        y_true: True labels
        y_prob: Predicted probabilities
        n_bins: Number of bins
        
    Returns:
        Dictionary with all calibration metrics
    """
    metrics = {}
    
    # Basic metrics
    metrics['brier_score'] = brier_score_loss(y_true, y_prob)
    
    # ECE variants
    metrics['ece_uniform'], _ = expected_calibration_error(y_true, y_prob, n_bins, 'uniform')
    metrics['ece_quantile'], _ = expected_calibration_error(y_true, y_prob, n_bins, 'quantile')
    metrics['mce'], _ = maximum_calibration_error(y_true, y_prob, n_bins)
    
    # Adaptive metrics
    metrics['ace'] = adaptive_calibration_error(y_true, y_prob)
    metrics['sce'] = static_calibration_error(y_true, y_prob, n_bins)
    
    # Over/under confidence
    metrics['overconfidence'] = overconfidence_error(y_true, y_prob, n_bins)
    metrics['underconfidence'] = underconfidence_error(y_true, y_prob, n_bins)
    
    # Class-wise
    classwise = classwise_calibration_error(y_true, y_prob, n_bins)
    metrics.update({f'classwise_{k}': v for k, v in classwise.items()})
    
    # Summary statistics
    metrics['mean_confidence'] = y_prob.mean()
    metrics['mean_accuracy'] = y_true.mean()
    metrics['confidence_accuracy_diff'] = abs(metrics['mean_confidence'] - metrics['mean_accuracy'])
    
    return metrics


def confidence_histogram(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 20,
    ax: Optional[plt.Axes] = None
) -> plt.Axes:
    """
    Plot confidence histogram colored by correctness.
    
    Args:
        y_true: True labels
        y_prob: Predicted probabilities
        n_bins: Number of bins
        ax: Matplotlib axes
        
    Returns:
        Matplotlib axes with plot
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    # Predictions
    y_pred = (y_prob >= 0.5).astype(int)
    correct = (y_pred == y_true)
    
    # Separate correct and incorrect
    correct_probs = y_prob[correct]
    incorrect_probs = y_prob[~correct]
    
    bins = np.linspace(0, 1, n_bins + 1)
    
    # Plot histograms
    ax.hist(correct_probs, bins=bins, alpha=0.7, label='Correct', color='green', edgecolor='black')
    ax.hist(incorrect_probs, bins=bins, alpha=0.7, label='Incorrect', color='red', edgecolor='black')
    
    ax.set_xlabel('Confidence')
    ax.set_ylabel('Count')
    ax.set_title('Confidence Distribution by Correctness')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add accuracy text
    accuracy = correct.mean()
    ax.text(0.05, 0.95, f'Accuracy: {accuracy:.3f}', transform=ax.transAxes,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat'))
    
    return ax