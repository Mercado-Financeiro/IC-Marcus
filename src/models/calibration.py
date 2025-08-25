"""
Probability calibration and EV-based threshold optimization.
Aligned with PRD sections 7 (Calibration) and 10 (Decision Policy).
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss
import matplotlib.pyplot as plt
import structlog

log = structlog.get_logger()


class ProbabilityCalibrator:
    """
    Calibrate model probabilities using isotonic or sigmoid methods.
    PRD Section 7: Calibration is mandatory for all models.
    """
    
    def __init__(
        self,
        method: str = 'isotonic',  # 'isotonic' or 'sigmoid'
        cv: Union[int, str] = 'prefit'  # 'prefit' or number of folds
    ):
        """Initialize calibrator."""
        self.method = method
        self.cv = cv
        self.calibrator = None
        self.is_fitted = False
        
    def fit(
        self,
        y_prob: np.ndarray,
        y_true: np.ndarray
    ) -> 'ProbabilityCalibrator':
        """
        Fit calibrator on validation data.
        
        Args:
            y_prob: Predicted probabilities
            y_true: True labels
        """
        
        if self.method == 'isotonic':
            self.calibrator = IsotonicRegression(out_of_bounds='clip')
        elif self.method == 'sigmoid':
            self.calibrator = LogisticRegression()
        else:
            raise ValueError(f"Unknown method: {self.method}")
            
        # Fit calibrator
        self.calibrator.fit(y_prob.reshape(-1, 1), y_true)
        self.is_fitted = True
        
        # Calculate calibration metrics
        y_prob_cal = self.transform(y_prob)
        
        self.metrics = {
            'brier_score_before': brier_score_loss(y_true, y_prob),
            'brier_score_after': brier_score_loss(y_true, y_prob_cal),
            'log_loss_before': log_loss(y_true, y_prob),
            'log_loss_after': log_loss(y_true, y_prob_cal),
        }
        
        log.info(
            f"Calibration fitted: "
            f"Brier before={self.metrics['brier_score_before']:.4f}, "
            f"after={self.metrics['brier_score_after']:.4f}"
        )
        
        return self
        
    def transform(self, y_prob: np.ndarray) -> np.ndarray:
        """Transform probabilities using fitted calibrator."""
        
        if not self.is_fitted:
            raise ValueError("Calibrator not fitted yet")
            
        if self.method == 'isotonic':
            return self.calibrator.transform(y_prob.reshape(-1, 1))
        else:  # sigmoid
            return self.calibrator.predict_proba(y_prob.reshape(-1, 1))[:, 1]
            
    def fit_transform(
        self,
        y_prob: np.ndarray,
        y_true: np.ndarray
    ) -> np.ndarray:
        """Fit and transform in one call."""
        self.fit(y_prob, y_true)
        return self.transform(y_prob)
        
    def plot_calibration_curve(
        self,
        y_prob: np.ndarray,
        y_true: np.ndarray,
        n_bins: int = 10,
        save_path: Optional[str] = None
    ) -> None:
        """Plot calibration curve (reliability diagram)."""
        
        # Calculate calibration curves
        fraction_pos, mean_pred = calibration_curve(
            y_true, y_prob, n_bins=n_bins, strategy='uniform'
        )
        
        if self.is_fitted:
            y_prob_cal = self.transform(y_prob)
            fraction_pos_cal, mean_pred_cal = calibration_curve(
                y_true, y_prob_cal, n_bins=n_bins, strategy='uniform'
            )
        
        # Plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Calibration plot
        ax1.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
        ax1.plot(mean_pred, fraction_pos, 'o-', label='Before calibration')
        if self.is_fitted:
            ax1.plot(mean_pred_cal, fraction_pos_cal, 's-', label='After calibration')
        ax1.set_xlabel('Mean Predicted Probability')
        ax1.set_ylabel('Fraction of Positives')
        ax1.set_title('Calibration Plot (Reliability Diagram)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Histogram of predictions
        ax2.hist(y_prob, bins=30, alpha=0.5, label='Before calibration')
        if self.is_fitted:
            ax2.hist(y_prob_cal, bins=30, alpha=0.5, label='After calibration')
        ax2.set_xlabel('Predicted Probability')
        ax2.set_ylabel('Count')
        ax2.set_title('Distribution of Predictions')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            log.info(f"Saved calibration plot to {save_path}")
        else:
            plt.show()


class EVThresholdOptimizer:
    """
    Optimize decision threshold based on Expected Value.
    PRD Section 10: Threshold optimization by EV with costs.
    """
    
    def __init__(
        self,
        fee_bps: float = 10.0,  # Trading fee in basis points
        slippage_bps: float = 10.0,  # Slippage in basis points
        win_return: float = 0.01,  # Expected return on winning trade (1%)
        loss_return: float = 0.01,  # Expected loss on losing trade (1%)
    ):
        """Initialize optimizer with cost parameters."""
        
        self.fee_bps = fee_bps
        self.slippage_bps = slippage_bps
        self.total_cost_bps = fee_bps + slippage_bps
        self.cost_per_trade = self.total_cost_bps / 10000  # Convert to decimal
        
        self.win_return = win_return
        self.loss_return = loss_return
        
        self.optimal_threshold = 0.5
        self.threshold_metrics = None
        
    def calculate_ev(
        self,
        tp: int,  # True positives
        fp: int,  # False positives
        tn: int,  # True negatives
        fn: int,  # False negatives
    ) -> float:
        """
        Calculate expected value per trade.
        
        EV = P(win) * (win_return - cost) - P(loss) * (loss_return + cost)
        """
        
        n_trades = tp + fp  # Total predicted positive trades
        
        if n_trades == 0:
            return 0.0
            
        # Win/loss probabilities
        p_win = tp / n_trades if n_trades > 0 else 0
        p_loss = fp / n_trades if n_trades > 0 else 0
        
        # Expected value
        ev = (
            p_win * (self.win_return - self.cost_per_trade) -
            p_loss * (self.loss_return + self.cost_per_trade)
        )
        
        return ev
        
    def optimize_threshold(
        self,
        y_prob: np.ndarray,
        y_true: np.ndarray,
        thresholds: Optional[np.ndarray] = None
    ) -> Tuple[float, pd.DataFrame]:
        """
        Find optimal threshold that maximizes EV.
        
        Returns:
            Tuple of (optimal_threshold, metrics_dataframe)
        """
        
        if thresholds is None:
            thresholds = np.linspace(0.1, 0.9, 81)
            
        metrics_list = []
        
        for threshold in thresholds:
            # Generate predictions
            y_pred = (y_prob >= threshold).astype(int)
            
            # Calculate confusion matrix
            tp = np.sum((y_pred == 1) & (y_true == 1))
            fp = np.sum((y_pred == 1) & (y_true == 0))
            tn = np.sum((y_pred == 0) & (y_true == 0))
            fn = np.sum((y_pred == 0) & (y_true == 1))
            
            # Calculate metrics
            n_trades = tp + fp
            precision = tp / n_trades if n_trades > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            # Calculate EV
            ev = self.calculate_ev(tp, fp, tn, fn)
            
            # Calculate net profit (simulated)
            gross_profit = tp * self.win_return - fp * self.loss_return
            total_cost = n_trades * self.cost_per_trade
            net_profit = gross_profit - total_cost
            
            metrics_list.append({
                'threshold': threshold,
                'tp': tp,
                'fp': fp,
                'tn': tn,
                'fn': fn,
                'n_trades': n_trades,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'ev_per_trade': ev,
                'gross_profit': gross_profit,
                'total_cost': total_cost,
                'net_profit': net_profit,
            })
            
        # Convert to DataFrame
        self.threshold_metrics = pd.DataFrame(metrics_list)
        
        # Find optimal threshold
        idx_max = self.threshold_metrics['ev_per_trade'].idxmax()
        self.optimal_threshold = self.threshold_metrics.loc[idx_max, 'threshold']
        
        log.info(
            f"Optimal threshold: {self.optimal_threshold:.3f}, "
            f"EV: {self.threshold_metrics.loc[idx_max, 'ev_per_trade']:.6f}, "
            f"Trades: {self.threshold_metrics.loc[idx_max, 'n_trades']:.0f}"
        )
        
        return self.optimal_threshold, self.threshold_metrics
        
    def plot_threshold_analysis(
        self,
        save_path: Optional[str] = None
    ) -> None:
        """Plot threshold optimization results."""
        
        if self.threshold_metrics is None:
            raise ValueError("Run optimize_threshold first")
            
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # EV per trade
        ax = axes[0, 0]
        ax.plot(self.threshold_metrics['threshold'], 
                self.threshold_metrics['ev_per_trade'], 'b-', linewidth=2)
        ax.axvline(self.optimal_threshold, color='r', linestyle='--', 
                   label=f'Optimal: {self.optimal_threshold:.3f}')
        ax.set_xlabel('Threshold')
        ax.set_ylabel('EV per Trade')
        ax.set_title('Expected Value vs Threshold')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Net profit
        ax = axes[0, 1]
        ax.plot(self.threshold_metrics['threshold'], 
                self.threshold_metrics['net_profit'], 'g-', linewidth=2)
        ax.axvline(self.optimal_threshold, color='r', linestyle='--')
        ax.set_xlabel('Threshold')
        ax.set_ylabel('Net Profit')
        ax.set_title('Net Profit vs Threshold')
        ax.grid(True, alpha=0.3)
        
        # Number of trades
        ax = axes[1, 0]
        ax.plot(self.threshold_metrics['threshold'], 
                self.threshold_metrics['n_trades'], 'orange', linewidth=2)
        ax.axvline(self.optimal_threshold, color='r', linestyle='--')
        ax.set_xlabel('Threshold')
        ax.set_ylabel('Number of Trades')
        ax.set_title('Trade Count vs Threshold')
        ax.grid(True, alpha=0.3)
        
        # Precision vs Recall
        ax = axes[1, 1]
        ax.plot(self.threshold_metrics['threshold'], 
                self.threshold_metrics['precision'], 'purple', label='Precision', linewidth=2)
        ax.plot(self.threshold_metrics['threshold'], 
                self.threshold_metrics['recall'], 'brown', label='Recall', linewidth=2)
        ax.plot(self.threshold_metrics['threshold'], 
                self.threshold_metrics['f1'], 'navy', label='F1', linewidth=2)
        ax.axvline(self.optimal_threshold, color='r', linestyle='--')
        ax.set_xlabel('Threshold')
        ax.set_ylabel('Score')
        ax.set_title('Classification Metrics vs Threshold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.suptitle(f'Threshold Optimization Analysis (Optimal: {self.optimal_threshold:.3f})', 
                     fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            log.info(f"Saved threshold analysis to {save_path}")
        else:
            plt.show()


def calibrate_and_optimize(
    y_prob: np.ndarray,
    y_true: np.ndarray,
    calibration_method: str = 'isotonic',
    fee_bps: float = 10.0,
    slippage_bps: float = 10.0,
    plot: bool = False
) -> Tuple[np.ndarray, float, Dict]:
    """
    Complete calibration and threshold optimization pipeline.
    
    Args:
        y_prob: Predicted probabilities
        y_true: True labels
        calibration_method: 'isotonic' or 'sigmoid'
        fee_bps: Trading fee in basis points
        slippage_bps: Slippage in basis points
        plot: Whether to generate plots
        
    Returns:
        Tuple of (calibrated_probabilities, optimal_threshold, metrics)
    """
    
    # Step 1: Calibrate probabilities
    calibrator = ProbabilityCalibrator(method=calibration_method)
    y_prob_cal = calibrator.fit_transform(y_prob, y_true)
    
    if plot:
        calibrator.plot_calibration_curve(y_prob, y_true)
    
    # Step 2: Optimize threshold
    optimizer = EVThresholdOptimizer(
        fee_bps=fee_bps,
        slippage_bps=slippage_bps
    )
    optimal_threshold, threshold_metrics = optimizer.optimize_threshold(
        y_prob_cal, y_true
    )
    
    if plot:
        optimizer.plot_threshold_analysis()
    
    # Compile metrics
    metrics = {
        **calibrator.metrics,
        'optimal_threshold': optimal_threshold,
        'optimal_ev': threshold_metrics.loc[
            threshold_metrics['threshold'] == optimal_threshold, 
            'ev_per_trade'
        ].values[0],
        'optimal_n_trades': threshold_metrics.loc[
            threshold_metrics['threshold'] == optimal_threshold,
            'n_trades'
        ].values[0],
    }
    
    log.info(f"Calibration and optimization complete: {metrics}")
    
    return y_prob_cal, optimal_threshold, metrics


if __name__ == "__main__":
    # Example usage
    np.random.seed(42)
    
    # Generate sample data
    n_samples = 1000
    y_true = np.random.binomial(1, 0.3, n_samples)
    
    # Generate uncalibrated probabilities (systematically biased)
    y_prob = np.clip(
        y_true * 0.7 + np.random.beta(2, 5, n_samples) * 0.5,
        0, 1
    )
    
    # Run calibration and optimization
    y_prob_cal, threshold, metrics = calibrate_and_optimize(
        y_prob, y_true,
        calibration_method='isotonic',
        fee_bps=10,
        slippage_bps=10,
        plot=True
    )
    
    print("\nCalibration and Optimization Results:")
    print("-" * 50)
    for key, value in metrics.items():
        print(f"{key:25s}: {value:.6f}")