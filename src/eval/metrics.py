"""
Comprehensive evaluation metrics for ML models in financial context.
Includes proper metrics for imbalanced classification and calibration.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, Union, List
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    matthews_corrcoef, cohen_kappa_score,
    roc_auc_score, average_precision_score,
    brier_score_loss, log_loss,
    confusion_matrix, classification_report,
    precision_recall_curve, roc_curve,
    calibration_curve
)
from sklearn.calibration import CalibratedClassifierCV
import warnings
warnings.filterwarnings('ignore')


class MetricsCalculator:
    """Calculate comprehensive metrics for binary classification."""
    
    @staticmethod
    def calculate_all_metrics(
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        threshold: float = 0.5,
        pos_label: int = 1
    ) -> Dict[str, float]:
        """
        Calculate all relevant metrics for imbalanced classification.
        
        Args:
            y_true: True labels (0/1)
            y_pred_proba: Predicted probabilities for positive class
            threshold: Decision threshold
            pos_label: Label for positive class
            
        Returns:
            Dictionary with all metrics
        """
        # Ensure arrays
        y_true = np.asarray(y_true)
        y_pred_proba = np.asarray(y_pred_proba)
        
        # Binary predictions
        y_pred = (y_pred_proba >= threshold).astype(int)
        
        # Confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        # Calculate prevalence
        prevalence = y_true.mean()
        
        # Basic metrics
        metrics = {
            # Counts
            'support_total': len(y_true),
            'support_pos': int(y_true.sum()),
            'support_neg': int(len(y_true) - y_true.sum()),
            'prevalence': prevalence,
            
            # Confusion matrix
            'true_positives': tp,
            'false_positives': fp,
            'true_negatives': tn,
            'false_negatives': fn,
            
            # Basic metrics
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
            'f1': f1_score(y_true, y_pred, zero_division=0),
            
            # Balanced metrics
            'balanced_accuracy': ((tp/(tp+fn) if (tp+fn)>0 else 0) + 
                                 (tn/(tn+fp) if (tn+fp)>0 else 0)) / 2,
            'mcc': matthews_corrcoef(y_true, y_pred),
            'cohen_kappa': cohen_kappa_score(y_true, y_pred),
            
            # Probabilistic metrics
            'roc_auc': roc_auc_score(y_true, y_pred_proba) if len(np.unique(y_true)) > 1 else 0,
            'pr_auc': average_precision_score(y_true, y_pred_proba) if len(np.unique(y_true)) > 1 else 0,
            'brier_score': brier_score_loss(y_true, y_pred_proba),
            'log_loss': log_loss(y_true, y_pred_proba),
            
            # Baseline comparisons
            'baseline_accuracy': max(prevalence, 1 - prevalence),
            'baseline_pr_auc': prevalence,
            
            # Threshold used
            'threshold': threshold
        }
        
        # Normalized metrics
        if prevalence > 0 and prevalence < 1:
            metrics['pr_auc_normalized'] = (metrics['pr_auc'] - prevalence) / (1 - prevalence)
        else:
            metrics['pr_auc_normalized'] = 0
            
        # Improvement over baseline
        metrics['accuracy_vs_baseline'] = metrics['accuracy'] - metrics['baseline_accuracy']
        metrics['pr_auc_vs_baseline'] = metrics['pr_auc'] - metrics['baseline_pr_auc']
        
        return metrics
    
    @staticmethod
    def calculate_calibration_metrics(
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        n_bins: int = 10
    ) -> Dict[str, Union[float, np.ndarray]]:
        """
        Calculate calibration metrics.
        
        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            n_bins: Number of bins for calibration curve
            
        Returns:
            Dictionary with calibration metrics
        """
        # Expected Calibration Error (ECE)
        fraction_pos, mean_pred = calibration_curve(y_true, y_pred_proba, n_bins=n_bins)
        ece = np.mean(np.abs(fraction_pos - mean_pred))
        
        # Maximum Calibration Error (MCE)
        mce = np.max(np.abs(fraction_pos - mean_pred))
        
        # Brier Score decomposition
        # BS = Reliability - Resolution + Uncertainty
        brier = brier_score_loss(y_true, y_pred_proba)
        
        return {
            'brier_score': brier,
            'ece': ece,
            'mce': mce,
            'calibration_fraction_pos': fraction_pos,
            'calibration_mean_pred': mean_pred,
            'n_bins': n_bins
        }
    
    @staticmethod
    def find_optimal_threshold(
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        metric: str = 'f1',
        n_thresholds: int = 100
    ) -> Tuple[float, float]:
        """
        Find optimal threshold for a given metric.
        
        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            metric: Metric to optimize ('f1', 'mcc', 'balanced_acc', 'g_mean')
            n_thresholds: Number of thresholds to test
            
        Returns:
            Tuple of (optimal_threshold, best_metric_value)
        """
        thresholds = np.linspace(0.05, 0.95, n_thresholds)
        best_threshold = 0.5
        best_score = -np.inf
        
        for th in thresholds:
            y_pred = (y_pred_proba >= th).astype(int)
            
            if metric == 'f1':
                score = f1_score(y_true, y_pred, zero_division=0)
            elif metric == 'mcc':
                score = matthews_corrcoef(y_true, y_pred)
            elif metric == 'balanced_acc':
                tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
                tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
                tnr = tn / (tn + fp) if (tn + fp) > 0 else 0
                score = (tpr + tnr) / 2
            elif metric == 'g_mean':
                tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
                tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
                tnr = tn / (tn + fp) if (tn + fp) > 0 else 0
                score = np.sqrt(tpr * tnr)
            else:
                raise ValueError(f"Unknown metric: {metric}")
            
            if score > best_score:
                best_score = score
                best_threshold = th
        
        return best_threshold, best_score
    
    @staticmethod
    def find_threshold_by_expected_value(
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        reward_tp: float = 1.0,
        cost_fp: float = -1.0,
        cost_fn: float = -0.5,
        reward_tn: float = 0.0,
        n_thresholds: int = 100
    ) -> Tuple[float, float]:
        """
        Find threshold that maximizes expected value.
        
        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            reward_tp: Reward for true positive
            cost_fp: Cost for false positive
            cost_fn: Cost for false negative
            reward_tn: Reward for true negative
            n_thresholds: Number of thresholds to test
            
        Returns:
            Tuple of (optimal_threshold, expected_value)
        """
        thresholds = np.linspace(0.05, 0.95, n_thresholds)
        best_threshold = 0.5
        best_ev = -np.inf
        
        for th in thresholds:
            y_pred = (y_pred_proba >= th).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            
            # Calculate expected value per prediction
            n_total = len(y_true)
            ev = (tp * reward_tp + fp * cost_fp + fn * cost_fn + tn * reward_tn) / n_total
            
            if ev > best_ev:
                best_ev = ev
                best_threshold = th
        
        return best_threshold, best_ev
    
    @staticmethod
    def calculate_baseline_metrics(
        y_true: np.ndarray,
        strategy: str = 'majority'
    ) -> Dict[str, float]:
        """
        Calculate metrics for baseline strategies.
        
        Args:
            y_true: True labels
            strategy: 'majority', 'minority', 'random', 'stratified'
            
        Returns:
            Dictionary with baseline metrics
        """
        prevalence = y_true.mean()
        n = len(y_true)
        
        if strategy == 'majority':
            # Always predict the majority class
            y_pred = np.zeros(n) if prevalence < 0.5 else np.ones(n)
            y_pred_proba = y_pred
            
        elif strategy == 'minority':
            # Always predict the minority class
            y_pred = np.ones(n) if prevalence < 0.5 else np.zeros(n)
            y_pred_proba = y_pred
            
        elif strategy == 'random':
            # Random predictions with 50% probability
            np.random.seed(42)
            y_pred_proba = np.random.random(n)
            y_pred = (y_pred_proba >= 0.5).astype(int)
            
        elif strategy == 'stratified':
            # Random predictions with class distribution
            np.random.seed(42)
            y_pred = np.random.binomial(1, prevalence, n)
            y_pred_proba = np.full(n, prevalence)
            
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        # Calculate metrics
        metrics = {
            'strategy': strategy,
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0),
            'mcc': matthews_corrcoef(y_true, y_pred)
        }
        
        # Add probabilistic metrics where applicable
        if strategy in ['random', 'stratified']:
            try:
                metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba)
                metrics['pr_auc'] = average_precision_score(y_true, y_pred_proba)
                metrics['brier'] = brier_score_loss(y_true, y_pred_proba)
            except:
                pass
        
        return metrics
    
    @staticmethod
    def compare_with_baselines(
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        threshold: float = 0.5
    ) -> pd.DataFrame:
        """
        Compare model performance with multiple baselines.
        
        Args:
            y_true: True labels
            y_pred_proba: Model predicted probabilities
            threshold: Decision threshold for model
            
        Returns:
            DataFrame with comparison
        """
        results = []
        
        # Model metrics
        model_metrics = MetricsCalculator.calculate_all_metrics(y_true, y_pred_proba, threshold)
        results.append({
            'Method': 'Model',
            'Accuracy': model_metrics['accuracy'],
            'Precision': model_metrics['precision'],
            'Recall': model_metrics['recall'],
            'F1': model_metrics['f1'],
            'MCC': model_metrics['mcc'],
            'ROC-AUC': model_metrics.get('roc_auc', np.nan),
            'PR-AUC': model_metrics.get('pr_auc', np.nan),
            'Brier': model_metrics.get('brier_score', np.nan)
        })
        
        # Baseline strategies
        for strategy in ['majority', 'minority', 'random', 'stratified']:
            baseline = MetricsCalculator.calculate_baseline_metrics(y_true, strategy)
            results.append({
                'Method': f'Baseline_{strategy}',
                'Accuracy': baseline['accuracy'],
                'Precision': baseline['precision'],
                'Recall': baseline['recall'],
                'F1': baseline['f1'],
                'MCC': baseline['mcc'],
                'ROC-AUC': baseline.get('roc_auc', np.nan),
                'PR-AUC': baseline.get('pr_auc', np.nan),
                'Brier': baseline.get('brier', np.nan)
            })
        
        df = pd.DataFrame(results)
        df = df.round(4)
        
        # Add improvement column
        baseline_acc = df.loc[df['Method'] == 'Baseline_majority', 'Accuracy'].values[0]
        df['Improvement'] = df['Accuracy'] - baseline_acc
        
        return df
    
    @staticmethod
    def print_classification_report(
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        threshold: float = 0.5,
        detailed: bool = True
    ) -> None:
        """
        Print comprehensive classification report.
        
        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            threshold: Decision threshold
            detailed: Whether to print detailed metrics
        """
        metrics = MetricsCalculator.calculate_all_metrics(y_true, y_pred_proba, threshold)
        
        print("\n" + "="*60)
        print("CLASSIFICATION REPORT")
        print("="*60)
        
        print(f"\nData Statistics:")
        print(f"  Total samples: {metrics['support_total']}")
        print(f"  Positive samples: {metrics['support_pos']} ({metrics['prevalence']:.2%})")
        print(f"  Negative samples: {metrics['support_neg']} ({1-metrics['prevalence']:.2%})")
        print(f"  Baseline accuracy: {metrics['baseline_accuracy']:.4f}")
        
        print(f"\nConfusion Matrix:")
        print(f"  True Positives:  {metrics['true_positives']:5d}")
        print(f"  False Positives: {metrics['false_positives']:5d}")
        print(f"  True Negatives:  {metrics['true_negatives']:5d}")
        print(f"  False Negatives: {metrics['false_negatives']:5d}")
        
        print(f"\nPerformance Metrics:")
        print(f"  Accuracy:     {metrics['accuracy']:.4f} (vs baseline: {metrics['accuracy_vs_baseline']:+.4f})")
        print(f"  Precision:    {metrics['precision']:.4f}")
        print(f"  Recall:       {metrics['recall']:.4f}")
        print(f"  Specificity:  {metrics['specificity']:.4f}")
        print(f"  F1 Score:     {metrics['f1']:.4f}")
        
        print(f"\nBalanced Metrics:")
        print(f"  Balanced Acc: {metrics['balanced_accuracy']:.4f}")
        print(f"  MCC:          {metrics['mcc']:.4f}")
        print(f"  Cohen Kappa:  {metrics['cohen_kappa']:.4f}")
        
        print(f"\nProbabilistic Metrics:")
        print(f"  ROC-AUC:      {metrics['roc_auc']:.4f}")
        print(f"  PR-AUC:       {metrics['pr_auc']:.4f} (baseline: {metrics['baseline_pr_auc']:.4f})")
        print(f"  PR-AUC Norm:  {metrics['pr_auc_normalized']:.4f}")
        print(f"  Brier Score:  {metrics['brier_score']:.4f}")
        print(f"  Log Loss:     {metrics['log_loss']:.4f}")
        
        if detailed:
            # Calculate calibration metrics
            cal_metrics = MetricsCalculator.calculate_calibration_metrics(y_true, y_pred_proba)
            print(f"\nCalibration Metrics:")
            print(f"  ECE:          {cal_metrics['ece']:.4f}")
            print(f"  MCE:          {cal_metrics['mce']:.4f}")
            
            # Find optimal thresholds
            th_f1, score_f1 = MetricsCalculator.find_optimal_threshold(y_true, y_pred_proba, 'f1')
            th_mcc, score_mcc = MetricsCalculator.find_optimal_threshold(y_true, y_pred_proba, 'mcc')
            
            print(f"\nOptimal Thresholds:")
            print(f"  For F1:  {th_f1:.3f} (score: {score_f1:.4f})")
            print(f"  For MCC: {th_mcc:.3f} (score: {score_mcc:.4f})")
            print(f"  Current: {threshold:.3f}")
        
        print("="*60)


# Example usage and testing
if __name__ == "__main__":
    # Generate sample data
    np.random.seed(42)
    n_samples = 1000
    prevalence = 0.2
    
    # True labels
    y_true = np.random.binomial(1, prevalence, n_samples)
    
    # Simulated predictions (slightly better than random)
    y_pred_proba = np.random.beta(
        2 + y_true * 2,
        2 + (1 - y_true) * 2,
        n_samples
    )
    
    # Calculate metrics
    calc = MetricsCalculator()
    
    # Print report
    calc.print_classification_report(y_true, y_pred_proba, detailed=True)
    
    # Compare with baselines
    comparison = calc.compare_with_baselines(y_true, y_pred_proba)
    print("\nBaseline Comparison:")
    print(comparison.to_string())
    
    # Find optimal threshold for expected value
    th_ev, ev = calc.find_threshold_by_expected_value(
        y_true, y_pred_proba,
        reward_tp=1.0, cost_fp=-0.5, cost_fn=-0.3, reward_tn=0.1
    )
    print(f"\nOptimal threshold for EV: {th_ev:.3f} (EV: {ev:.4f})")