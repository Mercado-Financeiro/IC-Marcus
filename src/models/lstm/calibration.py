"""
Calibration module for LSTM probability outputs.
Provides isotonic and sigmoid calibration methods.
"""

import numpy as np
from typing import Dict, Tuple, Optional, Union
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
import warnings
warnings.filterwarnings('ignore')


class LSTMCalibrator:
    """Calibration wrapper for LSTM models."""
    
    def __init__(self, method: str = 'isotonic'):
        """
        Initialize calibrator.
        
        Args:
            method: 'isotonic', 'sigmoid', or 'both' (choose best)
        """
        self.method = method
        self.calibrator = None
        self.best_method = None
        self.is_fitted = False
        
    def fit(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> 'LSTMCalibrator':
        """
        Fit calibrator on validation data.
        
        Args:
            y_true: True labels
            y_pred_proba: Uncalibrated probabilities from LSTM
            
        Returns:
            Self
        """
        y_true = np.asarray(y_true).flatten()
        y_pred_proba = np.asarray(y_pred_proba).flatten()
        
        if self.method == 'both':
            # Try both methods and choose best
            calibrators = {}
            scores = {}
            
            # Isotonic
            iso_cal = IsotonicRegression(out_of_bounds='clip')
            iso_cal.fit(y_pred_proba, y_true)
            iso_probs = iso_cal.transform(y_pred_proba)
            scores['isotonic'] = brier_score_loss(y_true, iso_probs)
            calibrators['isotonic'] = iso_cal
            
            # Sigmoid (Platt)
            sig_cal = LogisticRegression(C=1e10, solver='lbfgs', max_iter=1000)
            sig_cal.fit(y_pred_proba.reshape(-1, 1), y_true)
            sig_probs = sig_cal.predict_proba(y_pred_proba.reshape(-1, 1))[:, 1]
            scores['sigmoid'] = brier_score_loss(y_true, sig_probs)
            calibrators['sigmoid'] = sig_cal
            
            # Choose best
            self.best_method = min(scores, key=scores.get)
            self.calibrator = calibrators[self.best_method]
            print(f"Selected calibration method: {self.best_method} (Brier: {scores[self.best_method]:.4f})")
            
        elif self.method == 'isotonic':
            self.calibrator = IsotonicRegression(out_of_bounds='clip')
            self.calibrator.fit(y_pred_proba, y_true)
            self.best_method = 'isotonic'
            
        elif self.method == 'sigmoid':
            self.calibrator = LogisticRegression(C=1e10, solver='lbfgs', max_iter=1000)
            self.calibrator.fit(y_pred_proba.reshape(-1, 1), y_true)
            self.best_method = 'sigmoid'
            
        else:
            raise ValueError(f"Unknown calibration method: {self.method}")
        
        self.is_fitted = True
        return self
    
    def transform(self, y_pred_proba: np.ndarray) -> np.ndarray:
        """
        Transform uncalibrated probabilities to calibrated ones.
        
        Args:
            y_pred_proba: Uncalibrated probabilities
            
        Returns:
            Calibrated probabilities
        """
        if not self.is_fitted:
            raise ValueError("Calibrator not fitted. Call fit() first.")
        
        y_pred_proba = np.asarray(y_pred_proba).flatten()
        
        if isinstance(self.calibrator, IsotonicRegression):
            return self.calibrator.transform(y_pred_proba)
        else:  # LogisticRegression
            return self.calibrator.predict_proba(y_pred_proba.reshape(-1, 1))[:, 1]
    
    def fit_transform(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> np.ndarray:
        """
        Fit and transform in one step.
        
        Args:
            y_true: True labels
            y_pred_proba: Uncalibrated probabilities
            
        Returns:
            Calibrated probabilities
        """
        self.fit(y_true, y_pred_proba)
        return self.transform(y_pred_proba)
    
    def evaluate(
        self,
        y_true: np.ndarray,
        y_pred_uncal: np.ndarray,
        y_pred_cal: Optional[np.ndarray] = None,
        n_bins: int = 10
    ) -> Dict[str, float]:
        """
        Evaluate calibration quality.
        
        Args:
            y_true: True labels
            y_pred_uncal: Uncalibrated probabilities
            y_pred_cal: Calibrated probabilities (computed if None)
            n_bins: Number of bins for calibration curve
            
        Returns:
            Dictionary with calibration metrics
        """
        if y_pred_cal is None:
            if not self.is_fitted:
                raise ValueError("Calibrator not fitted")
            y_pred_cal = self.transform(y_pred_uncal)
        
        metrics = {}
        
        # Brier scores
        metrics['brier_uncalibrated'] = brier_score_loss(y_true, y_pred_uncal)
        metrics['brier_calibrated'] = brier_score_loss(y_true, y_pred_cal)
        metrics['brier_improvement'] = metrics['brier_uncalibrated'] - metrics['brier_calibrated']
        
        # Expected Calibration Error (ECE)
        for name, probs in [('uncalibrated', y_pred_uncal), ('calibrated', y_pred_cal)]:
            fraction_pos, mean_pred = calibration_curve(y_true, probs, n_bins=n_bins)
            ece = np.mean(np.abs(fraction_pos - mean_pred))
            mce = np.max(np.abs(fraction_pos - mean_pred))
            metrics[f'ece_{name}'] = ece
            metrics[f'mce_{name}'] = mce
        
        metrics['ece_improvement'] = metrics['ece_uncalibrated'] - metrics['ece_calibrated']
        metrics['calibration_method'] = self.best_method or self.method
        
        return metrics
    
    def print_evaluation(
        self,
        y_true: np.ndarray,
        y_pred_uncal: np.ndarray,
        y_pred_cal: Optional[np.ndarray] = None
    ) -> None:
        """
        Print calibration evaluation report.
        
        Args:
            y_true: True labels
            y_pred_uncal: Uncalibrated probabilities
            y_pred_cal: Calibrated probabilities
        """
        metrics = self.evaluate(y_true, y_pred_uncal, y_pred_cal)
        
        print("\n" + "="*60)
        print("CALIBRATION REPORT")
        print("="*60)
        
        print(f"\nMethod: {metrics['calibration_method']}")
        
        print("\nBrier Score:")
        print(f"  Uncalibrated: {metrics['brier_uncalibrated']:.4f}")
        print(f"  Calibrated: {metrics['brier_calibrated']:.4f}")
        print(f"  Improvement: {metrics['brier_improvement']:.4f}")
        
        print("\nExpected Calibration Error (ECE):")
        print(f"  Uncalibrated: {metrics['ece_uncalibrated']:.4f}")
        print(f"  Calibrated: {metrics['ece_calibrated']:.4f}")
        print(f"  Improvement: {metrics['ece_improvement']:.4f}")
        
        print("\nMaximum Calibration Error (MCE):")
        print(f"  Uncalibrated: {metrics['mce_uncalibrated']:.4f}")
        print(f"  Calibrated: {metrics['mce_calibrated']:.4f}")
        
        if metrics['brier_improvement'] > 0:
            print("\n✓ Calibration improved Brier score")
        else:
            print("\n⚠️ Calibration did not improve Brier score")
        
        print("="*60 + "\n")


class LSTMCalibratorWrapper:
    """
    Wrapper to make LSTM model compatible with sklearn's CalibratedClassifierCV.
    """
    
    def __init__(self, lstm_model, device='cpu'):
        """
        Initialize wrapper.
        
        Args:
            lstm_model: Trained LSTM model
            device: Device for inference
        """
        self.model = lstm_model
        self.device = device
        self.classes_ = np.array([0, 1])
        
    def predict_proba(self, X):
        """
        Predict probabilities for sklearn compatibility.
        
        Args:
            X: Input features or sequences
            
        Returns:
            Probability array of shape (n_samples, 2)
        """
        import torch
        
        self.model.eval()
        
        # Handle different input types
        if isinstance(X, torch.Tensor):
            X_tensor = X.to(self.device)
        else:
            X_tensor = torch.FloatTensor(X).to(self.device)
        
        with torch.no_grad():
            # Get logits or probabilities from model
            outputs = self.model(X_tensor)
            
            # Apply sigmoid if model outputs logits
            if not hasattr(self.model, 'output_activation') or self.model.output_activation != 'sigmoid':
                probs = torch.sigmoid(outputs).cpu().numpy()
            else:
                probs = outputs.cpu().numpy()
        
        # Reshape to 1D if needed
        probs = probs.flatten()
        
        # Return as 2D array for sklearn
        return np.column_stack([1 - probs, probs])
    
    def fit(self, X, y):
        """Dummy fit method for sklearn compatibility."""
        return self
    
    def predict(self, X):
        """Predict classes."""
        probs = self.predict_proba(X)
        return (probs[:, 1] >= 0.5).astype(int)