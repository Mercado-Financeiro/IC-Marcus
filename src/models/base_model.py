"""
Base model abstract class with minimal, focused interface.
No over-engineering, just what's needed for production.
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd
import joblib
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class BaseModel(ABC):
    """
    Minimal base class for all models.
    Focus on core functionality without bloat.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize model with configuration.
        
        Args:
            config: Model configuration dictionary
        """
        self.config = config
        self.model = None
        self.threshold = 0.5
        self.calibrator = None
        self.feature_names = None
        self.is_fitted = False
        
    @abstractmethod
    def fit(self, 
            X_train: pd.DataFrame, 
            y_train: pd.Series,
            X_val: Optional[pd.DataFrame] = None,
            y_val: Optional[pd.Series] = None) -> 'BaseModel':
        """
        Train the model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            
        Returns:
            Self for method chaining
        """
        pass
    
    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict probabilities.
        
        Args:
            X: Features
            
        Returns:
            Array of shape (n_samples, 2) with probabilities for each class
        """
        pass
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict binary labels using current threshold.
        
        Args:
            X: Features
            
        Returns:
            Binary predictions
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
            
        probas = self.predict_proba(X)
        return (probas[:, 1] >= self.threshold).astype(int)
    
    def calibrate(self, 
                  X_cal: pd.DataFrame, 
                  y_cal: pd.Series,
                  method: str = 'isotonic') -> 'BaseModel':
        """
        Calibrate model probabilities.
        
        Args:
            X_cal: Calibration features
            y_cal: Calibration labels
            method: 'isotonic' or 'sigmoid' (Platt scaling)
            
        Returns:
            Self for method chaining
        """
        from sklearn.calibration import CalibratedClassifierCV
        from sklearn.isotonic import IsotonicRegression
        from sklearn.linear_model import LogisticRegression
        
        if not self.is_fitted:
            raise ValueError("Model must be fitted before calibration")
        
        # Get uncalibrated probabilities
        probas = self.predict_proba(X_cal)[:, 1]
        
        if method == 'isotonic':
            self.calibrator = IsotonicRegression(out_of_bounds='clip')
            self.calibrator.fit(probas, y_cal)
        elif method == 'sigmoid':
            # Platt scaling using logistic regression
            self.calibrator = LogisticRegression()
            self.calibrator.fit(probas.reshape(-1, 1), y_cal)
        else:
            raise ValueError(f"Unknown calibration method: {method}")
        
        logger.info(f"Model calibrated using {method} method")
        return self
    
    def tune_threshold_by_ev(self,
                            y_true: pd.Series,
                            y_proba: np.ndarray,
                            costs: Dict[str, float],
                            n_thresholds: int = 100) -> float:
        """
        Tune decision threshold by maximizing Expected Value (EV).
        
        Args:
            y_true: True labels
            y_proba: Predicted probabilities
            costs: Dictionary with 'tp', 'fp', 'tn', 'fn' costs
            n_thresholds: Number of thresholds to test
            
        Returns:
            Optimal threshold
        """
        thresholds = np.linspace(0.01, 0.99, n_thresholds)
        best_ev = -np.inf
        best_threshold = 0.5
        
        for threshold in thresholds:
            y_pred = (y_proba >= threshold).astype(int)
            
            # Calculate confusion matrix elements
            tp = np.sum((y_true == 1) & (y_pred == 1))
            fp = np.sum((y_true == 0) & (y_pred == 1))
            tn = np.sum((y_true == 0) & (y_pred == 0))
            fn = np.sum((y_true == 1) & (y_pred == 0))
            
            # Calculate EV
            ev = (tp * costs.get('tp', 1.0) + 
                  fp * costs.get('fp', -1.0) + 
                  tn * costs.get('tn', 0.0) + 
                  fn * costs.get('fn', 0.0))
            
            # Normalize by number of samples
            ev = ev / len(y_true)
            
            if ev > best_ev:
                best_ev = ev
                best_threshold = threshold
        
        self.threshold = best_threshold
        logger.info(f"Optimal threshold: {best_threshold:.4f}, EV: {best_ev:.4f}")
        return best_threshold
    
    def save(self, path: str) -> None:
        """
        Save model to disk.
        
        Args:
            path: Path to save the model
        """
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted model")
        
        model_data = {
            'model': self.model,
            'config': self.config,
            'threshold': self.threshold,
            'calibrator': self.calibrator,
            'feature_names': self.feature_names,
            'is_fitted': self.is_fitted
        }
        
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model_data, path)
        logger.info(f"Model saved to {path}")
    
    def load(self, path: str) -> 'BaseModel':
        """
        Load model from disk.
        
        Args:
            path: Path to load the model from
            
        Returns:
            Self for method chaining
        """
        model_data = joblib.load(path)
        
        self.model = model_data['model']
        self.config = model_data['config']
        self.threshold = model_data['threshold']
        self.calibrator = model_data['calibrator']
        self.feature_names = model_data['feature_names']
        self.is_fitted = model_data['is_fitted']
        
        logger.info(f"Model loaded from {path}")
        return self
    
    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance if available.
        
        Returns:
            DataFrame with feature names and importance scores
        """
        raise NotImplementedError("Feature importance not available for this model")
    
    def _apply_calibration(self, probas: np.ndarray) -> np.ndarray:
        """
        Apply calibration if available.
        
        Args:
            probas: Uncalibrated probabilities
            
        Returns:
            Calibrated probabilities
        """
        if self.calibrator is None:
            return probas
        
        if hasattr(self.calibrator, 'predict_proba'):
            # Platt scaling
            calibrated = self.calibrator.predict_proba(probas.reshape(-1, 1))[:, 1]
        else:
            # Isotonic regression
            calibrated = self.calibrator.transform(probas)
        
        return calibrated