"""
Quality gates for model validation.
Ensures models meet minimum performance requirements before production release.

Based on:
- PR-AUC ≥ 1.2× prevalence
- Brier ≤ 0.9 × Brier-baseline
- ECE ≤ 0.05
- MCC > 0
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, List, Union
from sklearn.metrics import (
    precision_recall_curve, auc, brier_score_loss,
    matthews_corrcoef, confusion_matrix
)
from sklearn.calibration import calibration_curve
import warnings
warnings.filterwarnings('ignore')


class QualityGates:
    """
    Model quality validation against minimum production requirements.
    
    Gates:
    1. PR-AUC must be at least 1.2x better than baseline (prevalence)
    2. Brier score must be at least 10% better than baseline
    3. ECE (Expected Calibration Error) must be ≤ 0.05
    4. MCC (Matthews Correlation Coefficient) must be positive
    """
    
    def __init__(
        self,
        pr_auc_threshold: float = 1.2,
        brier_improvement: float = 0.9,
        ece_threshold: float = 0.05,
        mcc_threshold: float = 0.0
    ):
        """
        Initialize quality gates with thresholds.
        
        Args:
            pr_auc_threshold: Minimum PR-AUC ratio vs baseline (default 1.2)
            brier_improvement: Maximum Brier ratio vs baseline (default 0.9)
            ece_threshold: Maximum ECE allowed (default 0.05)
            mcc_threshold: Minimum MCC required (default 0.0)
        """
        self.pr_auc_threshold = pr_auc_threshold
        self.brier_improvement = brier_improvement
        self.ece_threshold = ece_threshold
        self.mcc_threshold = mcc_threshold
    
    @staticmethod
    def calculate_ece(y_true: np.ndarray, y_proba: np.ndarray, n_bins: int = 10) -> float:
        """
        Calculate Expected Calibration Error (ECE).
        
        ECE measures the difference between predicted probabilities and actual outcomes.
        Lower is better, with 0 being perfectly calibrated.
        
        Args:
            y_true: True binary labels
            y_proba: Predicted probabilities
            n_bins: Number of bins for calibration
            
        Returns:
            ECE value
        """
        # Get calibration data
        fraction_pos, mean_pred = calibration_curve(
            y_true, y_proba, n_bins=n_bins, strategy='uniform'
        )
        
        # Calculate bin sizes
        bin_edges = np.linspace(0, 1, n_bins + 1)
        bin_sizes = np.histogram(y_proba, bins=bin_edges)[0]
        
        # Calculate ECE
        ece = 0.0
        total_samples = len(y_proba)
        
        for i in range(len(fraction_pos)):
            if bin_sizes[i] > 0:
                weight = bin_sizes[i] / total_samples
                gap = abs(fraction_pos[i] - mean_pred[i])
                ece += weight * gap
        
        return ece
    
    def check_pr_auc_gate(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        return_details: bool = True
    ) -> Dict[str, Union[bool, float, str]]:
        """
        Check if PR-AUC meets minimum threshold.
        
        Args:
            y_true: True binary labels
            y_proba: Predicted probabilities
            return_details: Whether to return detailed results
            
        Returns:
            Dictionary with gate results
        """
        # Calculate PR-AUC
        precision, recall, _ = precision_recall_curve(y_true, y_proba)
        pr_auc = auc(recall, precision)
        
        # Calculate baseline (prevalence)
        prevalence = y_true.mean()
        baseline = prevalence
        
        # Required minimum
        required = baseline * self.pr_auc_threshold
        
        # Check if passed
        passed = pr_auc >= required
        
        # Calculate normalized PR-AUC
        pr_auc_norm = (pr_auc - baseline) / (1 - baseline) if baseline < 1 else 0
        
        result = {
            'passed': passed,
            'pr_auc': pr_auc,
            'baseline': baseline,
            'required': required,
            'ratio': pr_auc / baseline if baseline > 0 else 0,
            'pr_auc_normalized': pr_auc_norm,
            'message': f"PR-AUC: {pr_auc:.3f} {'≥' if passed else '<'} {required:.3f} ({self.pr_auc_threshold}×prevalence)"
        }
        
        if not return_details:
            return {'passed': passed}
        
        return result
    
    def check_calibration_gate(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        return_details: bool = True
    ) -> Dict[str, Union[bool, float, str]]:
        """
        Check if Brier score meets calibration requirements.
        
        Args:
            y_true: True binary labels
            y_proba: Predicted probabilities
            return_details: Whether to return detailed results
            
        Returns:
            Dictionary with gate results
        """
        # Calculate Brier score
        brier_score = brier_score_loss(y_true, y_proba)
        
        # Calculate baseline Brier score
        prevalence = y_true.mean()
        brier_baseline = prevalence * (1 - prevalence)
        
        # Required maximum
        required = brier_baseline * self.brier_improvement
        
        # Check if passed
        passed = brier_score <= required
        
        # Calculate improvement
        improvement = (brier_baseline - brier_score) / brier_baseline if brier_baseline > 0 else 0
        
        result = {
            'passed': passed,
            'brier_score': brier_score,
            'baseline': brier_baseline,
            'required': required,
            'improvement': improvement,
            'ratio': brier_score / brier_baseline if brier_baseline > 0 else float('inf'),
            'message': f"Brier: {brier_score:.3f} {'≤' if passed else '>'} {required:.3f} ({self.brier_improvement}×baseline)"
        }
        
        if not return_details:
            return {'passed': passed}
        
        return result
    
    def check_ece_mcc_gate(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        threshold: float = 0.5,
        return_details: bool = True
    ) -> Dict[str, Union[bool, float, str]]:
        """
        Check ECE and MCC requirements.
        
        Args:
            y_true: True binary labels
            y_proba: Predicted probabilities
            threshold: Decision threshold for binary predictions
            return_details: Whether to return detailed results
            
        Returns:
            Dictionary with gate results
        """
        # Calculate ECE
        ece = self.calculate_ece(y_true, y_proba)
        
        # Calculate MCC
        y_pred = (y_proba >= threshold).astype(int)
        mcc = matthews_corrcoef(y_true, y_pred)
        
        # Check gates
        ece_passed = ece <= self.ece_threshold
        mcc_passed = mcc > self.mcc_threshold
        passed = ece_passed and mcc_passed
        
        result = {
            'passed': passed,
            'ece': ece,
            'ece_passed': ece_passed,
            'ece_threshold': self.ece_threshold,
            'mcc': mcc,
            'mcc_passed': mcc_passed,
            'mcc_threshold': self.mcc_threshold,
            'message': f"ECE: {ece:.3f} {'✓' if ece_passed else '✗'} (≤{self.ece_threshold}), "
                      f"MCC: {mcc:.3f} {'✓' if mcc_passed else '✗'} (>{self.mcc_threshold})"
        }
        
        if not return_details:
            return {'passed': passed}
        
        return result
    
    def check_all_gates(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        threshold: float = 0.5
    ) -> Dict[str, Dict]:
        """
        Check all quality gates.
        
        Args:
            y_true: True binary labels
            y_proba: Predicted probabilities
            threshold: Decision threshold
            
        Returns:
            Dictionary with all gate results
        """
        results = {
            'pr_auc': self.check_pr_auc_gate(y_true, y_proba),
            'calibration': self.check_calibration_gate(y_true, y_proba),
            'ece_mcc': self.check_ece_mcc_gate(y_true, y_proba, threshold)
        }
        
        # Overall decision
        all_passed = all(gate['passed'] for gate in results.values())
        
        # Determine model mode
        if not results['pr_auc']['passed']:
            mode = 'MONITOR_ONLY'
            reason = 'PR-AUC below threshold - neutralizing decisions'
        elif not results['calibration']['passed']:
            mode = 'NEEDS_RECALIBRATION'
            reason = 'Calibration below threshold - requires recalibration'
        elif not results['ece_mcc']['passed']:
            mode = 'FAILED_QUALITY'
            reason = 'ECE or MCC below threshold - model quality insufficient'
        else:
            mode = 'PRODUCTION_READY'
            reason = 'All quality gates passed'
        
        results['summary'] = {
            'all_passed': all_passed,
            'mode': mode,
            'reason': reason,
            'prevalence': y_true.mean()
        }
        
        return results
    
    def validate_cross_validation_stability(
        self,
        cv_results: List[Dict[str, float]],
        max_std_ratio: float = 0.1
    ) -> Dict[str, Union[bool, float]]:
        """
        Check if metrics are stable across CV folds.
        
        Args:
            cv_results: List of metric dictionaries from each fold
            max_std_ratio: Maximum allowed std/mean ratio
            
        Returns:
            Stability check results
        """
        metrics_df = pd.DataFrame(cv_results)
        
        stability_results = {}
        for metric in ['pr_auc', 'mcc', 'brier_score', 'ece']:
            if metric in metrics_df.columns:
                values = metrics_df[metric].values
                mean_val = values.mean()
                std_val = values.std()
                cv_ratio = std_val / abs(mean_val) if mean_val != 0 else float('inf')
                
                is_stable = cv_ratio <= max_std_ratio
                
                # Special check for MCC - all should be positive
                if metric == 'mcc':
                    all_positive = all(v > 0 for v in values)
                    is_stable = is_stable and all_positive
                
                stability_results[metric] = {
                    'mean': mean_val,
                    'std': std_val,
                    'cv': cv_ratio,
                    'stable': is_stable,
                    'values': values.tolist()
                }
        
        all_stable = all(v['stable'] for v in stability_results.values())
        stability_results['all_stable'] = all_stable
        
        return stability_results
    
    def print_report(self, results: Dict[str, Dict]) -> None:
        """
        Print formatted quality gate report.
        
        Args:
            results: Results from check_all_gates()
        """
        print("\n" + "="*60)
        print("MODEL QUALITY GATES REPORT")
        print("="*60)
        
        # PR-AUC Gate
        pr_gate = results['pr_auc']
        print(f"\n1. PR-AUC Gate: {'✅ PASSED' if pr_gate['passed'] else '❌ FAILED'}")
        print(f"   - PR-AUC: {pr_gate['pr_auc']:.3f}")
        print(f"   - Required: {pr_gate['required']:.3f} (1.2×{pr_gate['baseline']:.3f})")
        print(f"   - Ratio: {pr_gate['ratio']:.2f}x baseline")
        
        # Calibration Gate
        cal_gate = results['calibration']
        print(f"\n2. Calibration Gate: {'✅ PASSED' if cal_gate['passed'] else '❌ FAILED'}")
        print(f"   - Brier Score: {cal_gate['brier_score']:.3f}")
        print(f"   - Required: {cal_gate['required']:.3f} (0.9×{cal_gate['baseline']:.3f})")
        print(f"   - Improvement: {cal_gate['improvement']*100:.1f}%")
        
        # ECE/MCC Gate
        ece_gate = results['ece_mcc']
        print(f"\n3. ECE/MCC Gate: {'✅ PASSED' if ece_gate['passed'] else '❌ FAILED'}")
        print(f"   - ECE: {ece_gate['ece']:.3f} (max {ece_gate['ece_threshold']})")
        print(f"   - MCC: {ece_gate['mcc']:.3f} (min {ece_gate['mcc_threshold']})")
        
        # Summary
        summary = results['summary']
        print(f"\n{'='*60}")
        print(f"DECISION: {summary['mode']}")
        print(f"REASON: {summary['reason']}")
        print(f"{'='*60}\n")
        
        # Emoji summary
        if summary['all_passed']:
            print("🎉 Model is PRODUCTION READY! 🚀")
        elif summary['mode'] == 'MONITOR_ONLY':
            print("⚠️  Model in MONITOR ONLY mode - decisions neutralized")
        elif summary['mode'] == 'NEEDS_RECALIBRATION':
            print("🔧 Model needs recalibration before production")
        else:
            print("❌ Model failed quality gates - not suitable for production")


def calculate_comprehensive_metrics(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    threshold: float = 0.5
) -> Dict[str, float]:
    """
    Calculate all metrics needed for quality gates.
    
    Args:
        y_true: True labels
        y_proba: Predicted probabilities
        threshold: Decision threshold
        
    Returns:
        Dictionary with all metrics
    """
    gates = QualityGates()
    
    # Binary predictions
    y_pred = (y_proba >= threshold).astype(int)
    
    # Calculate metrics
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    pr_auc = auc(recall, precision)
    
    prevalence = y_true.mean()
    pr_auc_norm = (pr_auc - prevalence) / (1 - prevalence) if prevalence < 1 else 0
    
    metrics = {
        'pr_auc': pr_auc,
        'pr_auc_normalized': pr_auc_norm,
        'pr_auc_baseline': prevalence,
        'brier_score': brier_score_loss(y_true, y_proba),
        'brier_baseline': prevalence * (1 - prevalence),
        'ece': gates.calculate_ece(y_true, y_proba),
        'mcc': matthews_corrcoef(y_true, y_pred),
        'prevalence': prevalence,
        'threshold': threshold
    }
    
    return metrics