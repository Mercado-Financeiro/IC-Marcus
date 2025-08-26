"""
Core metrics for model evaluation.
Focus on metrics that matter: MCC, Brier Score, AUC-PR, and Expected Value with costs.
"""
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple
from sklearn.metrics import (
    confusion_matrix, 
    matthews_corrcoef, 
    brier_score_loss,
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve
)
import logging

logger = logging.getLogger(__name__)


def calculate_confusion_matrix_metrics(y_true: np.ndarray, 
                                     y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculate metrics from confusion matrix.
    
    Args:
        y_true: True binary labels
        y_pred: Predicted binary labels
        
    Returns:
        Dictionary of metrics
    """
    cm = confusion_matrix(y_true, y_pred)
    
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    else:
        # Handle edge case where only one class is present
        if len(np.unique(y_true)) == 1:
            if y_true[0] == 0:
                tn = len(y_true) - np.sum(y_pred)
                fp = np.sum(y_pred)
                fn = tp = 0
            else:
                tp = np.sum(y_pred)
                fn = len(y_true) - np.sum(y_pred)
                tn = fp = 0
        else:
            raise ValueError("Unexpected confusion matrix shape")
    
    # Basic metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
    
    return {
        'tp': float(tp),
        'fp': float(fp),
        'tn': float(tn),
        'fn': float(fn),
        'precision': float(precision),
        'recall': float(recall),
        'specificity': float(specificity),
        'f1': float(f1),
        'accuracy': float(accuracy)
    }


def matthews_correlation_coefficient(y_true: np.ndarray, 
                                   y_pred: np.ndarray) -> float:
    """
    Calculate Matthews Correlation Coefficient (MCC).
    MCC is a balanced measure that works well for imbalanced datasets.
    
    Args:
        y_true: True binary labels
        y_pred: Predicted binary labels
        
    Returns:
        MCC score between -1 and 1 (1 = perfect, 0 = random, -1 = perfectly wrong)
    """
    try:
        return float(matthews_corrcoef(y_true, y_pred))
    except ValueError:
        # Handle edge cases (e.g., all same class)
        return 0.0


def brier_score(y_true: np.ndarray, 
                y_proba: np.ndarray) -> float:
    """
    Calculate Brier Score (calibration metric).
    Lower is better. Perfect calibration = 0.
    
    Args:
        y_true: True binary labels
        y_proba: Predicted probabilities for positive class
        
    Returns:
        Brier score
    """
    return float(brier_score_loss(y_true, y_proba))


def auc_pr(y_true: np.ndarray, 
           y_proba: np.ndarray) -> float:
    """
    Calculate Area Under Precision-Recall Curve.
    Better than ROC-AUC for imbalanced datasets.
    
    Args:
        y_true: True binary labels
        y_proba: Predicted probabilities for positive class
        
    Returns:
        AUC-PR score
    """
    try:
        return float(average_precision_score(y_true, y_proba))
    except ValueError:
        # Handle edge cases
        return 0.0


def expected_value_with_costs(y_true: np.ndarray,
                             y_pred: np.ndarray,
                             costs: Dict[str, float],
                             normalize: bool = True) -> float:
    """
    Calculate Expected Value (EV) considering transaction costs.
    
    Args:
        y_true: True binary labels
        y_pred: Predicted binary labels
        costs: Dictionary with cost structure:
               - 'tp': True positive reward (profit per correct buy signal)
               - 'fp': False positive cost (loss per incorrect buy signal)  
               - 'tn': True negative reward (usually 0)
               - 'fn': False negative cost (missed opportunity cost)
               - 'fee_bps': Transaction fee in basis points
               - 'slippage_bps': Slippage cost in basis points
        normalize: If True, normalize by number of samples
        
    Returns:
        Expected value
    """
    # Get confusion matrix metrics
    metrics = calculate_confusion_matrix_metrics(y_true, y_pred)
    
    tp, fp, tn, fn = metrics['tp'], metrics['fp'], metrics['tn'], metrics['fn']
    
    # Base costs/rewards
    tp_reward = costs.get('tp', 1.0)
    fp_cost = costs.get('fp', -1.0)
    tn_reward = costs.get('tn', 0.0)
    fn_cost = costs.get('fn', 0.0)
    
    # Transaction costs
    fee_bps = costs.get('fee_bps', 0.0)
    slippage_bps = costs.get('slippage_bps', 0.0)
    transaction_cost = (fee_bps + slippage_bps) / 10000.0  # Convert bps to decimal
    
    # Calculate number of transactions (position changes)
    n_transactions = np.sum(np.abs(np.diff(y_pred, prepend=0)))
    
    # Total EV
    base_ev = (tp * tp_reward + 
               fp * fp_cost + 
               tn * tn_reward + 
               fn * fn_cost)
    
    transaction_costs = n_transactions * transaction_cost
    total_ev = base_ev - transaction_costs
    
    if normalize and len(y_true) > 0:
        total_ev = total_ev / len(y_true)
    
    logger.debug(f"EV calculation: base={base_ev:.4f}, txn_costs={transaction_costs:.4f}, "
                f"total={total_ev:.4f}, n_txn={n_transactions}")
    
    return float(total_ev)


def sharpe_ratio(returns: np.ndarray, 
                 risk_free_rate: float = 0.0) -> float:
    """
    Calculate Sharpe ratio from returns.
    
    Args:
        returns: Array of returns
        risk_free_rate: Risk-free rate (annualized)
        
    Returns:
        Sharpe ratio
    """
    if len(returns) == 0:
        return 0.0
    
    excess_returns = returns - risk_free_rate / 252  # Daily risk-free rate
    
    if np.std(returns) == 0:
        return 0.0
    
    sharpe = np.mean(excess_returns) / np.std(returns) * np.sqrt(252)  # Annualized
    return float(sharpe)


def calculate_comprehensive_metrics(y_true: np.ndarray,
                                   y_pred: np.ndarray,
                                   y_proba: np.ndarray,
                                   costs: Optional[Dict[str, float]] = None,
                                   returns: Optional[np.ndarray] = None) -> Dict[str, float]:
    """
    Calculate comprehensive set of evaluation metrics.
    
    Args:
        y_true: True binary labels
        y_pred: Predicted binary labels
        y_proba: Predicted probabilities for positive class
        costs: Cost structure for EV calculation
        returns: Actual returns for Sharpe calculation
        
    Returns:
        Dictionary of all metrics
    """
    metrics = {}
    
    # Basic confusion matrix metrics
    cm_metrics = calculate_confusion_matrix_metrics(y_true, y_pred)
    metrics.update(cm_metrics)
    
    # MCC (balanced metric)
    metrics['mcc'] = matthews_correlation_coefficient(y_true, y_pred)
    
    # Brier Score (calibration)
    metrics['brier_score'] = brier_score(y_true, y_proba)
    
    # AUC-PR (better for imbalanced)
    metrics['auc_pr'] = auc_pr(y_true, y_proba)
    
    # ROC-AUC (for completeness)
    try:
        metrics['auc_roc'] = float(roc_auc_score(y_true, y_proba))
    except ValueError:
        metrics['auc_roc'] = 0.0
    
    # Expected Value with costs
    if costs is not None:
        metrics['ev'] = expected_value_with_costs(y_true, y_pred, costs)
        
        # EV without transaction costs (for comparison)
        costs_no_txn = costs.copy()
        costs_no_txn['fee_bps'] = 0.0
        costs_no_txn['slippage_bps'] = 0.0
        metrics['ev_no_txn'] = expected_value_with_costs(y_true, y_pred, costs_no_txn)
    
    # Sharpe ratio if returns provided
    if returns is not None:
        # Create strategy returns based on predictions
        strategy_returns = y_pred * returns  # Long only strategy
        metrics['sharpe'] = sharpe_ratio(strategy_returns)
        
        # Buy and hold Sharpe for comparison
        metrics['sharpe_buy_hold'] = sharpe_ratio(returns)
    
    return metrics


def find_optimal_threshold_by_metric(y_true: np.ndarray,
                                    y_proba: np.ndarray,
                                    metric: str = 'ev',
                                    costs: Optional[Dict[str, float]] = None,
                                    n_thresholds: int = 100) -> Tuple[float, float]:
    """
    Find optimal threshold by maximizing a specific metric.
    
    Args:
        y_true: True binary labels
        y_proba: Predicted probabilities
        metric: Metric to optimize ('ev', 'f1', 'mcc', 'precision', 'recall')
        costs: Cost structure for EV optimization
        n_thresholds: Number of thresholds to test
        
    Returns:
        Tuple of (optimal_threshold, best_metric_value)
    """
    thresholds = np.linspace(0.01, 0.99, n_thresholds)
    best_metric = -np.inf
    best_threshold = 0.5
    
    for threshold in thresholds:
        y_pred = (y_proba >= threshold).astype(int)
        
        if metric == 'ev':
            if costs is None:
                raise ValueError("Costs must be provided for EV optimization")
            metric_value = expected_value_with_costs(y_true, y_pred, costs)
        elif metric == 'mcc':
            metric_value = matthews_correlation_coefficient(y_true, y_pred)
        elif metric == 'f1':
            cm_metrics = calculate_confusion_matrix_metrics(y_true, y_pred)
            metric_value = cm_metrics['f1']
        elif metric == 'precision':
            cm_metrics = calculate_confusion_matrix_metrics(y_true, y_pred)
            metric_value = cm_metrics['precision']
        elif metric == 'recall':
            cm_metrics = calculate_confusion_matrix_metrics(y_true, y_pred)
            metric_value = cm_metrics['recall']
        else:
            raise ValueError(f"Unknown metric: {metric}")
        
        if metric_value > best_metric:
            best_metric = metric_value
            best_threshold = threshold
    
    logger.info(f"Optimal threshold for {metric}: {best_threshold:.4f} "
               f"(metric value: {best_metric:.4f})")
    
    return best_threshold, best_metric


def calibration_curve(y_true: np.ndarray, 
                     y_proba: np.ndarray, 
                     n_bins: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate calibration curve for probability predictions.
    
    Args:
        y_true: True binary labels
        y_proba: Predicted probabilities
        n_bins: Number of bins for calibration curve
        
    Returns:
        Tuple of (bin_boundaries, bin_accuracies)
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    bin_accuracies = []
    
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Find predictions in this bin
        in_bin = (y_proba > bin_lower) & (y_proba <= bin_upper)
        prop_in_bin = in_bin.mean()
        
        if prop_in_bin > 0:
            accuracy_in_bin = y_true[in_bin].mean()
            bin_accuracies.append(accuracy_in_bin)
        else:
            bin_accuracies.append(0.0)
    
    return bin_boundaries[:-1], np.array(bin_accuracies)