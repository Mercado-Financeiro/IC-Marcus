"""
Metrics module for LSTM models with imbalanced classification support.
Reuses the comprehensive metrics from src.eval.metrics.
"""

from typing import Dict, Tuple, Optional, Union
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    matthews_corrcoef, cohen_kappa_score,
    roc_auc_score, average_precision_score,
    brier_score_loss, log_loss,
    confusion_matrix
)

# Import comprehensive metrics if available
try:
    from src.eval.metrics import MetricsCalculator
    EVAL_METRICS_AVAILABLE = True
except ImportError:
    EVAL_METRICS_AVAILABLE = False


class LSTMMetrics:
    """Metrics calculator for LSTM models."""
    
    def __init__(self):
        """Initialize metrics calculator."""
        if EVAL_METRICS_AVAILABLE:
            self.calculator = MetricsCalculator()
        else:
            self.calculator = None
    
    def calculate_metrics(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        threshold: float = 0.5
    ) -> Dict[str, float]:
        """
        Calculate comprehensive metrics for LSTM predictions.
        
        Args:
            y_true: True labels (0/1)
            y_pred_proba: Predicted probabilities
            threshold: Decision threshold
            
        Returns:
            Dictionary with metrics
        """
        if self.calculator is not None:
            return self.calculator.calculate_all_metrics(y_true, y_pred_proba, threshold)
        
        # Fallback to basic metrics
        y_pred = (y_pred_proba >= threshold).astype(int)
        
        # Calculate prevalence
        prevalence = y_true.mean()
        
        # Confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0),
            'mcc': matthews_corrcoef(y_true, y_pred),
            'prevalence': prevalence,
            'baseline_accuracy': max(prevalence, 1 - prevalence)
        }
        
        # Add probabilistic metrics if possible
        try:
            metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba)
            metrics['pr_auc'] = average_precision_score(y_true, y_pred_proba)
            metrics['brier_score'] = brier_score_loss(y_true, y_pred_proba)
            metrics['log_loss'] = log_loss(y_true, y_pred_proba)
            
            # Normalized PR-AUC
            if prevalence > 0 and prevalence < 1:
                metrics['pr_auc_normalized'] = (metrics['pr_auc'] - prevalence) / (1 - prevalence)
            else:
                metrics['pr_auc_normalized'] = 0
                
        except Exception:
            pass
        
        return metrics
    
    def find_optimal_threshold(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        metric: str = 'f1'
    ) -> Tuple[float, float]:
        """
        Find optimal threshold for given metric.
        
        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            metric: Metric to optimize ('f1', 'mcc', 'balanced_acc')
            
        Returns:
            Tuple of (optimal_threshold, best_score)
        """
        if self.calculator is not None:
            return self.calculator.find_optimal_threshold(y_true, y_pred_proba, metric)
        
        # Simple fallback
        best_threshold = 0.5
        best_score = -np.inf
        
        for th in np.linspace(0.1, 0.9, 17):
            y_pred = (y_pred_proba >= th).astype(int)
            
            if metric == 'f1':
                score = f1_score(y_true, y_pred, zero_division=0)
            elif metric == 'mcc':
                score = matthews_corrcoef(y_true, y_pred)
            else:
                # Balanced accuracy
                tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
                tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
                tnr = tn / (tn + fp) if (tn + fp) > 0 else 0
                score = (tpr + tnr) / 2
            
            if score > best_score:
                best_score = score
                best_threshold = th
        
        return best_threshold, best_score
    
    def print_report(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        threshold: float = 0.5,
        model_name: str = "LSTM"
    ) -> None:
        """
        Print comprehensive metrics report.
        
        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            threshold: Decision threshold
            model_name: Name of the model
        """
        metrics = self.calculate_metrics(y_true, y_pred_proba, threshold)
        
        print(f"\n{'='*60}")
        print(f"{model_name} METRICS REPORT")
        print(f"{'='*60}")
        
        print(f"\nData Statistics:")
        print(f"  Samples: {len(y_true)}")
        print(f"  Prevalence: {metrics['prevalence']:.2%}")
        print(f"  Baseline Acc: {metrics['baseline_accuracy']:.4f}")
        
        print(f"\nClassification Metrics (threshold={threshold:.2f}):")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall: {metrics['recall']:.4f}")
        print(f"  F1 Score: {metrics['f1']:.4f}")
        print(f"  MCC: {metrics['mcc']:.4f}")
        
        if 'roc_auc' in metrics:
            print(f"\nProbabilistic Metrics:")
            print(f"  ROC-AUC: {metrics['roc_auc']:.4f}")
            print(f"  PR-AUC: {metrics['pr_auc']:.4f}")
            print(f"  PR-AUC Norm: {metrics.get('pr_auc_normalized', 0):.4f}")
            print(f"  Brier Score: {metrics['brier_score']:.4f}")
            print(f"  Log Loss: {metrics.get('log_loss', 0):.4f}")
        
        # Verdict
        if 'pr_auc' in metrics:
            baseline_pr = metrics['prevalence']
            if metrics['pr_auc'] <= baseline_pr * 1.1:
                print(f"\n⚠️ Model does not significantly beat baseline PR-AUC ({baseline_pr:.4f})")
            else:
                improvement = (metrics['pr_auc'] / baseline_pr - 1) * 100
                print(f"\n✓ Model beats baseline by {improvement:.1f}%")
        
        if metrics['mcc'] < 0.1:
            print("⚠️ MCC < 0.1 indicates weak correlation")
        else:
            print(f"✓ MCC = {metrics['mcc']:.3f} indicates significant correlation")
        
        print(f"{'='*60}\n")
    
    @staticmethod
    def calculate_pos_weight(y_train: np.ndarray) -> float:
        """
        Calculate positive class weight for BCEWithLogitsLoss.
        
        Args:
            y_train: Training labels
            
        Returns:
            pos_weight value
        """
        pos_count = y_train.sum()
        neg_count = len(y_train) - pos_count
        
        if pos_count == 0:
            return 1.0
        
        return neg_count / pos_count