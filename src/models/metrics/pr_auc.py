"""
PR-AUC (Precision-Recall Area Under Curve) metrics.
Specialized for imbalanced classification in financial markets.
"""

import numpy as np
from typing import Tuple, Dict, Optional, List
from sklearn.metrics import precision_recall_curve, auc
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


def calculate_pr_auc(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    """
    Calculate PR-AUC (Area Under Precision-Recall Curve).
    
    Args:
        y_true: True binary labels
        y_proba: Predicted probabilities
        
    Returns:
        PR-AUC value
    """
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    return auc(recall, precision)


def calculate_pr_auc_normalized(
    y_true: np.ndarray,
    y_proba: np.ndarray
) -> Tuple[float, float, float]:
    """
    Calculate normalized PR-AUC.
    
    Normalization accounts for class imbalance by comparing to baseline (prevalence).
    A normalized PR-AUC of 0 means performance equal to random classifier,
    while 1 means perfect classifier.
    
    Args:
        y_true: True binary labels
        y_proba: Predicted probabilities
        
    Returns:
        Tuple of (pr_auc, pr_auc_normalized, baseline)
    """
    # Calculate PR-AUC
    pr_auc = calculate_pr_auc(y_true, y_proba)
    
    # Calculate baseline (prevalence)
    prevalence = y_true.mean()
    baseline = prevalence
    
    # Normalize
    if prevalence < 1:
        pr_auc_norm = (pr_auc - baseline) / (1 - baseline)
    else:
        pr_auc_norm = 0.0
    
    return pr_auc, pr_auc_norm, baseline


def calculate_pr_auc_with_confidence(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    random_state: int = 42
) -> Dict[str, float]:
    """
    Calculate PR-AUC with bootstrap confidence intervals.
    
    Args:
        y_true: True binary labels
        y_proba: Predicted probabilities
        n_bootstrap: Number of bootstrap samples
        confidence_level: Confidence level for intervals
        random_state: Random seed for reproducibility
        
    Returns:
        Dictionary with PR-AUC and confidence intervals
    """
    np.random.seed(random_state)
    n_samples = len(y_true)
    
    # Original PR-AUC
    pr_auc_orig = calculate_pr_auc(y_true, y_proba)
    
    # Bootstrap
    pr_auc_scores = []
    for _ in range(n_bootstrap):
        # Sample with replacement
        indices = np.random.choice(n_samples, n_samples, replace=True)
        y_true_boot = y_true[indices]
        y_proba_boot = y_proba[indices]
        
        # Skip if only one class
        if len(np.unique(y_true_boot)) < 2:
            continue
        
        pr_auc_boot = calculate_pr_auc(y_true_boot, y_proba_boot)
        pr_auc_scores.append(pr_auc_boot)
    
    pr_auc_scores = np.array(pr_auc_scores)
    
    # Calculate confidence intervals
    alpha = 1 - confidence_level
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100
    
    ci_lower = np.percentile(pr_auc_scores, lower_percentile)
    ci_upper = np.percentile(pr_auc_scores, upper_percentile)
    
    # Standard error
    se = np.std(pr_auc_scores)
    
    return {
        'pr_auc': pr_auc_orig,
        'pr_auc_mean': np.mean(pr_auc_scores),
        'pr_auc_std': se,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'confidence_level': confidence_level,
        'n_bootstrap': len(pr_auc_scores)
    }


def optimize_threshold_for_pr(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    beta: float = 1.0
) -> Tuple[float, float]:
    """
    Find optimal threshold that maximizes F-beta score.
    
    Args:
        y_true: True binary labels
        y_proba: Predicted probabilities
        beta: Beta parameter for F-beta score (1.0 for F1)
        
    Returns:
        Tuple of (optimal_threshold, max_fbeta_score)
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    
    # Calculate F-beta score for each threshold
    beta_squared = beta ** 2
    fbeta_scores = ((1 + beta_squared) * precision * recall) / \
                   (beta_squared * precision + recall + 1e-10)
    
    # Handle edge cases
    fbeta_scores = np.nan_to_num(fbeta_scores)
    
    # Find best threshold (excluding last point which is threshold=1)
    if len(fbeta_scores) > 1:
        best_idx = np.argmax(fbeta_scores[:-1])
        best_threshold = float(thresholds[best_idx])
        best_score = float(fbeta_scores[best_idx])
    else:
        best_threshold = 0.5
        best_score = 0.0
    
    return best_threshold, best_score


def calculate_pr_metrics_at_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    threshold: float
) -> Dict[str, float]:
    """
    Calculate precision and recall at a specific threshold.
    
    Args:
        y_true: True binary labels
        y_proba: Predicted probabilities
        threshold: Decision threshold
        
    Returns:
        Dictionary with metrics at threshold
    """
    y_pred = (y_proba >= threshold).astype(int)
    
    # Calculate confusion matrix elements
    tp = np.sum((y_pred == 1) & (y_true == 1))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    tn = np.sum((y_pred == 0) & (y_true == 0))
    
    # Calculate metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    # F1 score
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'threshold': threshold,
        'precision': precision,
        'recall': recall,
        'specificity': specificity,
        'f1_score': f1,
        'true_positives': int(tp),
        'false_positives': int(fp),
        'true_negatives': int(tn),
        'false_negatives': int(fn),
        'support_positive': int(tp + fn),
        'support_negative': int(tn + fp)
    }


def compare_pr_auc_models(
    y_true: np.ndarray,
    y_proba_1: np.ndarray,
    y_proba_2: np.ndarray,
    n_bootstrap: int = 1000,
    random_state: int = 42
) -> Dict[str, float]:
    """
    Statistical comparison of two models' PR-AUC scores.
    
    Uses bootstrap to test if difference is significant.
    
    Args:
        y_true: True binary labels
        y_proba_1: Predicted probabilities from model 1
        y_proba_2: Predicted probabilities from model 2
        n_bootstrap: Number of bootstrap samples
        random_state: Random seed
        
    Returns:
        Dictionary with comparison results
    """
    np.random.seed(random_state)
    n_samples = len(y_true)
    
    # Original PR-AUCs
    pr_auc_1 = calculate_pr_auc(y_true, y_proba_1)
    pr_auc_2 = calculate_pr_auc(y_true, y_proba_2)
    diff_orig = pr_auc_1 - pr_auc_2
    
    # Bootstrap differences
    differences = []
    for _ in range(n_bootstrap):
        indices = np.random.choice(n_samples, n_samples, replace=True)
        y_true_boot = y_true[indices]
        y_proba_1_boot = y_proba_1[indices]
        y_proba_2_boot = y_proba_2[indices]
        
        # Skip if only one class
        if len(np.unique(y_true_boot)) < 2:
            continue
        
        pr_auc_1_boot = calculate_pr_auc(y_true_boot, y_proba_1_boot)
        pr_auc_2_boot = calculate_pr_auc(y_true_boot, y_proba_2_boot)
        differences.append(pr_auc_1_boot - pr_auc_2_boot)
    
    differences = np.array(differences)
    
    # Calculate p-value (two-tailed test)
    p_value = 2 * min(
        np.mean(differences <= 0),
        np.mean(differences >= 0)
    )
    
    # Confidence interval for difference
    ci_lower = np.percentile(differences, 2.5)
    ci_upper = np.percentile(differences, 97.5)
    
    return {
        'pr_auc_model_1': pr_auc_1,
        'pr_auc_model_2': pr_auc_2,
        'difference': diff_orig,
        'difference_ci_lower': ci_lower,
        'difference_ci_upper': ci_upper,
        'p_value': p_value,
        'significant': p_value < 0.05,
        'better_model': 1 if diff_orig > 0 else 2
    }


def calculate_weighted_pr_auc(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    sample_weights: Optional[np.ndarray] = None
) -> float:
    """
    Calculate weighted PR-AUC for samples with different importance.
    
    Args:
        y_true: True binary labels
        y_proba: Predicted probabilities
        sample_weights: Weight for each sample
        
    Returns:
        Weighted PR-AUC
    """
    if sample_weights is None:
        return calculate_pr_auc(y_true, y_proba)
    
    # Normalize weights
    sample_weights = sample_weights / sample_weights.sum() * len(sample_weights)
    
    # Sort by predicted probability
    sort_idx = np.argsort(y_proba)[::-1]
    y_true_sorted = y_true[sort_idx]
    y_proba_sorted = y_proba[sort_idx]
    weights_sorted = sample_weights[sort_idx]
    
    # Calculate weighted precision and recall at each threshold
    precisions = []
    recalls = []
    
    for i in range(len(y_true_sorted)):
        threshold = y_proba_sorted[i]
        y_pred = (y_proba >= threshold).astype(int)
        
        # Weighted TP, FP, FN
        tp = np.sum(sample_weights[(y_pred == 1) & (y_true == 1)])
        fp = np.sum(sample_weights[(y_pred == 1) & (y_true == 0)])
        fn = np.sum(sample_weights[(y_pred == 0) & (y_true == 1)])
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        precisions.append(precision)
        recalls.append(recall)
    
    # Add endpoints
    precisions = [0] + precisions + [1]
    recalls = [1] + recalls + [0]
    
    # Calculate area using trapezoidal rule
    pr_auc = 0
    for i in range(len(recalls) - 1):
        pr_auc += (recalls[i] - recalls[i + 1]) * \
                  (precisions[i] + precisions[i + 1]) / 2
    
    return pr_auc


def analyze_pr_curve(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    n_thresholds: int = 100
) -> Dict[str, np.ndarray]:
    """
    Detailed analysis of precision-recall curve.
    
    Args:
        y_true: True binary labels
        y_proba: Predicted probabilities
        n_thresholds: Number of thresholds to evaluate
        
    Returns:
        Dictionary with curve analysis
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    
    # Calculate F1 at each point
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
    f1_scores = np.nan_to_num(f1_scores)
    
    # Find key points
    best_f1_idx = np.argmax(f1_scores[:-1])
    
    # High precision point (precision >= 0.9)
    high_precision_idx = np.where(precision >= 0.9)[0]
    if len(high_precision_idx) > 0:
        high_precision_point = high_precision_idx[0]
    else:
        high_precision_point = len(precision) - 1
    
    # High recall point (recall >= 0.9)
    high_recall_idx = np.where(recall >= 0.9)[0]
    if len(high_recall_idx) > 0:
        high_recall_point = high_recall_idx[-1]
    else:
        high_recall_point = 0
    
    return {
        'precision': precision,
        'recall': recall,
        'thresholds': thresholds,
        'f1_scores': f1_scores,
        'pr_auc': auc(recall, precision),
        'best_f1_score': f1_scores[best_f1_idx],
        'best_f1_threshold': thresholds[best_f1_idx] if best_f1_idx < len(thresholds) else 1.0,
        'best_f1_precision': precision[best_f1_idx],
        'best_f1_recall': recall[best_f1_idx],
        'high_precision_threshold': thresholds[high_precision_point] if high_precision_point < len(thresholds) else 1.0,
        'high_precision_recall': recall[high_precision_point],
        'high_recall_threshold': thresholds[high_recall_point] if high_recall_point < len(thresholds) else 0.0,
        'high_recall_precision': precision[high_recall_point]
    }