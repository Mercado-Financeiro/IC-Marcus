"""Model calibration utilities for XGBoost."""

import numpy as np
import pandas as pd
from typing import Optional, Union
from sklearn.calibration import CalibratedClassifierCV
from sklearn.base import BaseEstimator
import logging

log = logging.getLogger(__name__)


class ModelCalibrator:
    """Calibrate model probabilities for better reliability."""
    
    def __init__(self, method: str = 'isotonic', cv: int = 3):
        """
        Initialize calibrator.
        
        Args:
            method: Calibration method ('isotonic' or 'sigmoid')
            cv: Number of CV folds for calibration
        """
        self.method = method
        self.cv = cv
        self.calibrated_model = None
        
    def calibrate(self, model: BaseEstimator, X: pd.DataFrame, y: pd.Series) -> BaseEstimator:
        """
        Calibrate model predictions.
        
        Args:
            model: Trained model to calibrate
            X: Features for calibration
            y: Labels for calibration
            
        Returns:
            Calibrated model
        """
        log.info(f"Calibrating model using {self.method} method with {self.cv} folds")
        
        # Create calibrated classifier
        self.calibrated_model = CalibratedClassifierCV(
            estimator=model,
            method=self.method,
            cv=self.cv
        )
        
        # Fit calibration
        self.calibrated_model.fit(X, y)
        
        # Calculate calibration metrics
        y_pred_uncalibrated = model.predict_proba(X)[:, 1]
        y_pred_calibrated = self.calibrated_model.predict_proba(X)[:, 1]
        
        # Expected Calibration Error (ECE)
        ece_before = self._calculate_ece(y, y_pred_uncalibrated)
        ece_after = self._calculate_ece(y, y_pred_calibrated)
        
        log.info(f"Calibration complete. ECE before: {ece_before:.4f}, after: {ece_after:.4f}")
        
        return self.calibrated_model
    
    def _calculate_ece(self, y_true: np.ndarray, y_pred_proba: np.ndarray, 
                      n_bins: int = 10) -> float:
        """
        Calculate Expected Calibration Error.
        
        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            n_bins: Number of bins for calibration
            
        Returns:
            ECE score (lower is better)
        """
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (y_pred_proba > bin_lower) & (y_pred_proba <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = y_true[in_bin].mean()
                avg_confidence_in_bin = y_pred_proba[in_bin].mean()
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        
        return ece
    
    def plot_calibration_curve(self, y_true: np.ndarray, y_pred_proba: np.ndarray,
                              n_bins: int = 10, title: str = "Calibration Plot"):
        """
        Plot calibration curve.
        
        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            n_bins: Number of bins
            title: Plot title
            
        Returns:
            matplotlib figure
        """
        try:
            import matplotlib.pyplot as plt
            from sklearn.calibration import calibration_curve
            
            # Calculate calibration curve
            fraction_of_positives, mean_predicted_value = calibration_curve(
                y_true, y_pred_proba, n_bins=n_bins
            )
            
            # Create plot
            fig, ax = plt.subplots(figsize=(8, 6))
            
            # Plot perfect calibration line
            ax.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
            
            # Plot calibration curve
            ax.plot(mean_predicted_value, fraction_of_positives, 
                   'o-', label='Model calibration')
            
            ax.set_xlabel('Mean predicted probability')
            ax.set_ylabel('Fraction of positives')
            ax.set_title(title)
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            return fig
            
        except ImportError:
            log.warning("Matplotlib not available for plotting")
            return None
    
    def get_reliability_scores(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> dict:
        """
        Calculate various reliability metrics.
        
        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            
        Returns:
            Dictionary of reliability metrics
        """
        from sklearn.metrics import brier_score_loss, log_loss
        
        # Expected Calibration Error
        ece = self._calculate_ece(y_true, y_pred_proba)
        
        # Maximum Calibration Error
        mce = self._calculate_mce(y_true, y_pred_proba)
        
        # Brier Score (lower is better)
        brier = brier_score_loss(y_true, y_pred_proba)
        
        # Log Loss
        logloss = log_loss(y_true, y_pred_proba)
        
        return {
            'ece': ece,
            'mce': mce,
            'brier_score': brier,
            'log_loss': logloss
        }
    
    def _calculate_mce(self, y_true: np.ndarray, y_pred_proba: np.ndarray,
                      n_bins: int = 10) -> float:
        """
        Calculate Maximum Calibration Error.
        
        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            n_bins: Number of bins
            
        Returns:
            MCE score (lower is better)
        """
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        max_error = 0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (y_pred_proba > bin_lower) & (y_pred_proba <= bin_upper)
            
            if in_bin.sum() > 0:
                accuracy_in_bin = y_true[in_bin].mean()
                avg_confidence_in_bin = y_pred_proba[in_bin].mean()
                error = np.abs(avg_confidence_in_bin - accuracy_in_bin)
                max_error = max(max_error, error)
        
        return max_error