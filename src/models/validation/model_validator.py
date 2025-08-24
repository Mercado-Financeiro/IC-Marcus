"""
Model validator with quality gates for production release.
Ensures models meet minimum performance requirements.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False
from typing import Dict, Tuple, Optional, Any, List
from pathlib import Path
import json
from datetime import datetime
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    confusion_matrix, classification_report,
    precision_recall_curve, roc_curve
)
import warnings
warnings.filterwarnings('ignore')

# Import our custom modules
try:
    from ..metrics.quality_gates import QualityGates, calculate_comprehensive_metrics
    from ..metrics.pr_auc import analyze_pr_curve, calculate_pr_auc_with_confidence
    from ..calibration.beta import compare_calibration_methods
except ImportError:
    import sys
    base_path = Path(__file__).parent.parent
    sys.path.insert(0, str(base_path))
    from metrics.quality_gates import QualityGates, calculate_comprehensive_metrics
    from metrics.pr_auc import analyze_pr_curve, calculate_pr_auc_with_confidence
    from calibration.beta import compare_calibration_methods


class ModelValidator:
    """
    Comprehensive model validation with quality gates.
    
    Validates models against production requirements:
    - PR-AUC ≥ 1.2× prevalence
    - Brier ≤ 0.9× baseline
    - ECE ≤ 0.05
    - MCC > 0
    """
    
    def __init__(
        self,
        gates: Optional[QualityGates] = None,
        save_plots: bool = True,
        plot_dir: str = "artifacts/validation",
        verbose: bool = True
    ):
        """
        Initialize model validator.
        
        Args:
            gates: QualityGates instance (creates default if None)
            save_plots: Whether to save validation plots
            plot_dir: Directory to save plots
            verbose: Whether to print results
        """
        self.gates = gates or QualityGates()
        self.save_plots = save_plots
        self.plot_dir = Path(plot_dir)
        self.verbose = verbose
        
        if save_plots:
            self.plot_dir.mkdir(parents=True, exist_ok=True)
    
    def validate_model(
        self,
        model: Any,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        model_name: str = "model",
        threshold: Optional[float] = None
    ) -> Tuple[Dict, Dict]:
        """
        Complete model validation with quality gates.
        
        Args:
            model: Trained model with predict_proba method
            X_test: Test features
            y_test: Test labels
            model_name: Name for saving results
            threshold: Decision threshold (auto-optimized if None)
            
        Returns:
            Tuple of (gate_results, metrics)
        """
        # Get predictions
        y_proba = model.predict_proba(X_test)
        if y_proba.ndim > 1:
            y_proba = y_proba[:, 1]
        
        # Auto-optimize threshold if not provided
        if threshold is None:
            from ..metrics.pr_auc import optimize_threshold_for_pr
            threshold, _ = optimize_threshold_for_pr(y_test.values, y_proba)
            if self.verbose:
                print(f"Auto-optimized threshold: {threshold:.3f}")
        
        # Calculate comprehensive metrics
        metrics = calculate_comprehensive_metrics(y_test.values, y_proba, threshold)
        
        # Check quality gates
        gate_results = self.gates.check_all_gates(y_test.values, y_proba, threshold)
        
        # Print report
        if self.verbose:
            self.gates.print_report(gate_results)
            self._print_metrics_summary(metrics)
        
        # Generate plots
        if self.save_plots:
            self._generate_all_plots(
                y_test.values, y_proba, threshold,
                gate_results, metrics, model_name
            )
        
        # Save results
        self._save_results(gate_results, metrics, model_name)
        
        return gate_results, metrics
    
    def validate_cross_validation(
        self,
        cv_predictions: List[Dict[str, np.ndarray]],
        model_name: str = "model"
    ) -> Dict:
        """
        Validate model stability across CV folds.
        
        Args:
            cv_predictions: List of dicts with 'y_true', 'y_proba' for each fold
            model_name: Model name for reporting
            
        Returns:
            Stability analysis results
        """
        cv_metrics = []
        
        for fold_idx, fold_data in enumerate(cv_predictions):
            y_true = fold_data['y_true']
            y_proba = fold_data['y_proba']
            
            # Calculate metrics for fold
            fold_metrics = calculate_comprehensive_metrics(y_true, y_proba)
            fold_metrics['fold'] = fold_idx
            cv_metrics.append(fold_metrics)
        
        # Check stability
        stability_results = self.gates.validate_cross_validation_stability(cv_metrics)
        
        if self.verbose:
            self._print_stability_report(stability_results)
        
        return stability_results
    
    def compare_models(
        self,
        models: Dict[str, Any],
        X_test: pd.DataFrame,
        y_test: pd.Series,
        save_comparison: bool = True
    ) -> pd.DataFrame:
        """
        Compare multiple models on same test set.
        
        Args:
            models: Dictionary of {name: model}
            X_test: Test features
            y_test: Test labels
            save_comparison: Whether to save comparison table
            
        Returns:
            DataFrame with comparison results
        """
        results = []
        
        for name, model in models.items():
            if self.verbose:
                print(f"\nValidating {name}...")
            
            gates, metrics = self.validate_model(
                model, X_test, y_test, name, verbose=False
            )
            
            result = {
                'model': name,
                'pr_auc': metrics['pr_auc'],
                'pr_auc_normalized': metrics['pr_auc_normalized'],
                'brier_score': metrics['brier_score'],
                'ece': metrics['ece'],
                'mcc': metrics['mcc'],
                'pr_gate_passed': gates['pr_auc']['passed'],
                'brier_gate_passed': gates['calibration']['passed'],
                'ece_mcc_gate_passed': gates['ece_mcc']['passed'],
                'all_gates_passed': gates['summary']['all_passed'],
                'mode': gates['summary']['mode']
            }
            results.append(result)
        
        comparison_df = pd.DataFrame(results)
        comparison_df = comparison_df.sort_values('pr_auc_normalized', ascending=False)
        
        if self.verbose:
            print("\n" + "="*80)
            print("MODEL COMPARISON")
            print("="*80)
            print(comparison_df.to_string())
        
        if save_comparison:
            comparison_path = self.plot_dir / f"model_comparison_{datetime.now():%Y%m%d_%H%M%S}.csv"
            comparison_df.to_csv(comparison_path, index=False)
            if self.verbose:
                print(f"\nComparison saved to: {comparison_path}")
        
        return comparison_df
    
    def _generate_all_plots(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        threshold: float,
        gate_results: Dict,
        metrics: Dict,
        model_name: str
    ):
        """Generate all validation plots."""
        # Create figure with subplots
        fig = plt.figure(figsize=(20, 12))
        
        # 1. Calibration plot (REQUIRED)
        ax1 = plt.subplot(2, 3, 1)
        self._plot_calibration_diagram(y_true, y_proba, ax1)
        
        # 2. PR curve
        ax2 = plt.subplot(2, 3, 2)
        self._plot_pr_curve(y_true, y_proba, threshold, metrics, ax2)
        
        # 3. ROC curve
        ax3 = plt.subplot(2, 3, 3)
        self._plot_roc_curve(y_true, y_proba, ax3)
        
        # 4. Confusion matrix
        ax4 = plt.subplot(2, 3, 4)
        self._plot_confusion_matrix(y_true, y_proba, threshold, ax4)
        
        # 5. Probability distribution
        ax5 = plt.subplot(2, 3, 5)
        self._plot_probability_distribution(y_true, y_proba, threshold, ax5)
        
        # 6. Quality gates summary
        ax6 = plt.subplot(2, 3, 6)
        self._plot_gates_summary(gate_results, ax6)
        
        plt.suptitle(f"{model_name} - Validation Results", fontsize=16)
        plt.tight_layout()
        
        # Save figure
        plot_path = self.plot_dir / f"{model_name}_validation_{datetime.now():%Y%m%d_%H%M%S}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        if self.verbose:
            print(f"Plots saved to: {plot_path}")
    
    def _plot_calibration_diagram(self, y_true: np.ndarray, y_proba: np.ndarray, ax):
        """Plot calibration diagram (REQUIRED)."""
        fraction_pos, mean_pred = calibration_curve(y_true, y_proba, n_bins=10)
        
        # Calculate ECE
        ece = self.gates.calculate_ece(y_true, y_proba)
        
        # Calculate Brier score
        from sklearn.metrics import brier_score_loss
        brier = brier_score_loss(y_true, y_proba)
        
        # Plot
        ax.plot(mean_pred, fraction_pos, 's-', label='Model', markersize=8)
        ax.plot([0, 1], [0, 1], 'k--', label='Perfect calibration', alpha=0.5)
        
        # Fill area between curves
        ax.fill_between(mean_pred, mean_pred, fraction_pos, alpha=0.2)
        
        # Add metrics
        ax.set_xlabel('Mean Predicted Probability')
        ax.set_ylabel('Fraction of Positives')
        ax.set_title(f'Calibration Plot\nBrier: {brier:.3f}, ECE: {ece:.3f}')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
    
    def _plot_pr_curve(self, y_true: np.ndarray, y_proba: np.ndarray, 
                      threshold: float, metrics: Dict, ax):
        """Plot precision-recall curve."""
        precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
        pr_auc = metrics['pr_auc']
        baseline = metrics['pr_auc_baseline']
        
        # Plot curve
        ax.plot(recall, precision, label=f'PR curve (AUC = {pr_auc:.3f})')
        ax.axhline(y=baseline, color='r', linestyle='--', 
                   label=f'Baseline = {baseline:.3f}')
        
        # Mark operating point
        idx = np.argmin(np.abs(thresholds - threshold))
        ax.plot(recall[idx], precision[idx], 'ro', markersize=10,
                label=f'Operating point (τ={threshold:.2f})')
        
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title(f'Precision-Recall Curve\nPR-AUC: {pr_auc:.3f} ({pr_auc/baseline:.2f}×baseline)')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
    
    def _plot_roc_curve(self, y_true: np.ndarray, y_proba: np.ndarray, ax):
        """Plot ROC curve."""
        from sklearn.metrics import roc_auc_score
        
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        roc_auc = roc_auc_score(y_true, y_proba)
        
        ax.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.3f})')
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.5)
        
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(f'ROC Curve\nAUC: {roc_auc:.3f}')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
    
    def _plot_confusion_matrix(self, y_true: np.ndarray, y_proba: np.ndarray,
                               threshold: float, ax):
        """Plot confusion matrix."""
        y_pred = (y_proba >= threshold).astype(int)
        cm = confusion_matrix(y_true, y_pred)
        
        # Normalize
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # Plot
        if HAS_SEABORN:
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                       xticklabels=['Negative', 'Positive'],
                       yticklabels=['Negative', 'Positive'])
        else:
            # Fallback without seaborn
            im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
            ax.figure.colorbar(im, ax=ax)
            ax.set_xticks([0, 1])
            ax.set_yticks([0, 1])
            ax.set_xticklabels(['Negative', 'Positive'])
            ax.set_yticklabels(['Negative', 'Positive'])
        
        # Add percentages
        if HAS_SEABORN:
            for i in range(2):
                for j in range(2):
                    text = ax.texts[i * 2 + j]
                    text.set_text(f'{cm[i, j]}\n({cm_norm[i, j]:.1%})')
        else:
            # Add text annotations manually
            for i in range(2):
                for j in range(2):
                    ax.text(j, i, f'{cm[i, j]}\n({cm_norm[i, j]:.1%})',
                           ha='center', va='center', color='white' if cm[i, j] > cm.max()/2 else 'black')
        
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_title(f'Confusion Matrix (τ={threshold:.2f})')
    
    def _plot_probability_distribution(self, y_true: np.ndarray, y_proba: np.ndarray,
                                      threshold: float, ax):
        """Plot probability distribution by class."""
        # Separate probabilities by class
        proba_neg = y_proba[y_true == 0]
        proba_pos = y_proba[y_true == 1]
        
        # Plot histograms
        bins = np.linspace(0, 1, 50)
        ax.hist(proba_neg, bins=bins, alpha=0.5, label='Negative', color='blue', density=True)
        ax.hist(proba_pos, bins=bins, alpha=0.5, label='Positive', color='red', density=True)
        
        # Add threshold line
        ax.axvline(x=threshold, color='green', linestyle='--', linewidth=2,
                  label=f'Threshold = {threshold:.2f}')
        
        ax.set_xlabel('Predicted Probability')
        ax.set_ylabel('Density')
        ax.set_title('Probability Distribution by Class')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
    
    def _plot_gates_summary(self, gate_results: Dict, ax):
        """Plot quality gates summary."""
        ax.axis('off')
        
        # Create summary text
        summary_text = "QUALITY GATES SUMMARY\n" + "="*30 + "\n\n"
        
        # PR-AUC Gate
        pr_gate = gate_results['pr_auc']
        symbol = "✓" if pr_gate['passed'] else "✗"
        summary_text += f"{symbol} PR-AUC: {pr_gate['pr_auc']:.3f}\n"
        summary_text += f"   Required: ≥{pr_gate['required']:.3f}\n"
        summary_text += f"   Ratio: {pr_gate['ratio']:.2f}×baseline\n\n"
        
        # Calibration Gate
        cal_gate = gate_results['calibration']
        symbol = "✓" if cal_gate['passed'] else "✗"
        summary_text += f"{symbol} Brier Score: {cal_gate['brier_score']:.3f}\n"
        summary_text += f"   Required: ≤{cal_gate['required']:.3f}\n"
        summary_text += f"   Improvement: {cal_gate['improvement']*100:.1f}%\n\n"
        
        # ECE/MCC Gate
        ece_gate = gate_results['ece_mcc']
        symbol = "✓" if ece_gate['passed'] else "✗"
        summary_text += f"{symbol} ECE: {ece_gate['ece']:.3f} (≤{ece_gate['ece_threshold']})\n"
        summary_text += f"{symbol} MCC: {ece_gate['mcc']:.3f} (>{ece_gate['mcc_threshold']})\n\n"
        
        # Overall Decision
        summary = gate_results['summary']
        summary_text += "="*30 + "\n"
        summary_text += f"DECISION: {summary['mode']}\n"
        
        # Color based on decision
        if summary['all_passed']:
            color = 'green'
            summary_text += "\n✅ PRODUCTION READY"
        elif summary['mode'] == 'MONITOR_ONLY':
            color = 'orange'
            summary_text += "\n⚠️ MONITOR ONLY MODE"
        else:
            color = 'red'
            summary_text += "\n❌ NOT READY"
        
        ax.text(0.5, 0.5, summary_text, fontsize=10, ha='center', va='center',
               transform=ax.transAxes, family='monospace',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    def _print_metrics_summary(self, metrics: Dict):
        """Print detailed metrics summary."""
        print("\n" + "="*60)
        print("DETAILED METRICS")
        print("="*60)
        
        print(f"\nPR-AUC Metrics:")
        print(f"  PR-AUC: {metrics['pr_auc']:.4f}")
        print(f"  PR-AUC Normalized: {metrics['pr_auc_normalized']:.4f}")
        print(f"  Baseline (prevalence): {metrics['pr_auc_baseline']:.4f}")
        
        print(f"\nCalibration Metrics:")
        print(f"  Brier Score: {metrics['brier_score']:.4f}")
        print(f"  Brier Baseline: {metrics['brier_baseline']:.4f}")
        print(f"  ECE: {metrics['ece']:.4f}")
        
        print(f"\nClassification Metrics:")
        print(f"  MCC: {metrics['mcc']:.4f}")
        print(f"  Threshold: {metrics['threshold']:.3f}")
    
    def _print_stability_report(self, stability_results: Dict):
        """Print CV stability report."""
        print("\n" + "="*60)
        print("CROSS-VALIDATION STABILITY")
        print("="*60)
        
        for metric, results in stability_results.items():
            if metric == 'all_stable':
                continue
            
            print(f"\n{metric.upper()}:")
            print(f"  Mean: {results['mean']:.4f}")
            print(f"  Std: {results['std']:.4f}")
            print(f"  CV: {results['cv']*100:.1f}%")
            print(f"  Stable: {'Yes' if results['stable'] else 'No'}")
            
            if metric == 'mcc':
                all_positive = all(v > 0 for v in results['values'])
                print(f"  All positive: {'Yes' if all_positive else 'No'}")
        
        print(f"\nOverall Stability: {'✅ STABLE' if stability_results['all_stable'] else '❌ UNSTABLE'}")
    
    def _save_results(self, gate_results: Dict, metrics: Dict, model_name: str):
        """Save validation results to JSON."""
        results = {
            'model_name': model_name,
            'timestamp': datetime.now().isoformat(),
            'gates': {
                'pr_auc_passed': gate_results['pr_auc']['passed'],
                'calibration_passed': gate_results['calibration']['passed'],
                'ece_mcc_passed': gate_results['ece_mcc']['passed'],
                'all_passed': gate_results['summary']['all_passed'],
                'mode': gate_results['summary']['mode']
            },
            'metrics': metrics,
            'gate_details': gate_results
        }
        
        results_path = self.plot_dir / f"{model_name}_validation_{datetime.now():%Y%m%d_%H%M%S}.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        if self.verbose:
            print(f"Results saved to: {results_path}")