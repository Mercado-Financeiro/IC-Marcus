"""Base XGBoost model implementation."""

import numpy as np
import pandas as pd
import xgboost as xgb
from typing import Optional, Union, Tuple, Dict, Any
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.calibration import CalibratedClassifierCV
import structlog
import joblib
from pathlib import Path

from .config import XGBoostConfig

log = structlog.get_logger()


class BaseXGBoost(BaseEstimator, ClassifierMixin):
    """Base XGBoost classifier with calibration support."""
    
    def __init__(self, config: Optional[XGBoostConfig] = None):
        """Initialize base XGBoost model.
        
        Args:
            config: XGBoost configuration
        """
        self.config = config or XGBoostConfig()
        self.model = None
        self.calibrator = None
        self.feature_names = None
        self.is_fitted = False
        
        # Initialize XGBoost classifier
        self._init_model()
        
        log.info(
            "base_xgboost_initialized",
            n_estimators=self.config.n_estimators,
            max_depth=self.config.max_depth,
            learning_rate=self.config.learning_rate
        )
    
    def _init_model(self):
        """Initialize XGBoost classifier."""
        params = self.config.to_dict()
        self.model = xgb.XGBClassifier(**params)
    
    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        eval_set: Optional[list] = None,
        early_stopping_rounds: Optional[int] = None,
        verbose: bool = False,
        sample_weight: Optional[np.ndarray] = None
    ) -> 'BaseXGBoost':
        """Fit the XGBoost model.
        
        Args:
            X: Training features
            y: Training labels
            eval_set: Validation set for early stopping
            early_stopping_rounds: Rounds for early stopping
            verbose: Whether to print training progress
            sample_weight: Sample weights
            
        Returns:
            Self
        """
        # Store feature names
        if isinstance(X, pd.DataFrame):
            self.feature_names = X.columns.tolist()
        else:
            self.feature_names = [f"f_{i}" for i in range(X.shape[1])]
        
        # Fit model
        fit_params = {
            'eval_set': eval_set,
            'verbose': verbose
        }
        
        if early_stopping_rounds is not None:
            fit_params['callbacks'] = [
                xgb.callback.EarlyStopping(
                    rounds=early_stopping_rounds,
                    save_best=True
                )
            ]
        
        if sample_weight is not None:
            fit_params['sample_weight'] = sample_weight
        
        self.model.fit(X, y, **fit_params)
        self.is_fitted = True
        
        log.info(
            "model_fitted",
            n_features=len(self.feature_names),
            n_samples=len(y),
            best_iteration=getattr(self.model, 'best_iteration', None)
        )
        
        return self
    
    def calibrate(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        method: str = 'isotonic',
        cv: int = 3
    ) -> 'BaseXGBoost':
        """Calibrate model predictions.
        
        Args:
            X: Calibration features
            y: Calibration labels
            method: Calibration method ('isotonic' or 'sigmoid')
            cv: Cross-validation folds for calibration
            
        Returns:
            Self
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before calibration")
        
        self.calibrator = CalibratedClassifierCV(
            estimator=self.model,
            method=method,
            cv=cv
        )
        
        self.calibrator.fit(X, y)
        
        log.info(
            "model_calibrated",
            method=method,
            cv=cv,
            n_samples=len(y)
        )
        
        return self
    
    def predict(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        threshold: float = 0.5
    ) -> np.ndarray:
        """Predict classes.
        
        Args:
            X: Features to predict
            threshold: Classification threshold
            
        Returns:
            Predicted classes
        """
        proba = self.predict_proba(X)
        
        if proba.ndim == 2:
            return (proba[:, 1] >= threshold).astype(int)
        else:
            return (proba >= threshold).astype(int)
    
    def predict_proba(
        self,
        X: Union[pd.DataFrame, np.ndarray]
    ) -> np.ndarray:
        """Predict probabilities.
        
        Args:
            X: Features to predict
            
        Returns:
            Predicted probabilities
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # Use calibrator if available
        if self.calibrator is not None:
            return self.calibrator.predict_proba(X)
        else:
            return self.model.predict_proba(X)
    
    def get_feature_importance(
        self,
        importance_type: str = 'gain'
    ) -> pd.DataFrame:
        """Get feature importance.
        
        Args:
            importance_type: Type of importance ('gain', 'weight', 'cover')
            
        Returns:
            DataFrame with feature importance
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted to get feature importance")
        
        importance = self.model.get_booster().get_score(
            importance_type=importance_type
        )
        
        # Create DataFrame
        importance_df = pd.DataFrame(
            list(importance.items()),
            columns=['feature', 'importance']
        )
        
        # Sort by importance
        importance_df = importance_df.sort_values(
            'importance',
            ascending=False
        )
        
        return importance_df
    
    def save(self, filepath: str):
        """Save model to file.
        
        Args:
            filepath: Path to save model
        """
        model_data = {
            'model': self.model,
            'calibrator': self.calibrator,
            'config': self.config,
            'feature_names': self.feature_names,
            'is_fitted': self.is_fitted
        }
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model_data, filepath)
        
        log.info("model_saved", filepath=filepath)
    
    def load(self, filepath: str):
        """Load model from file.
        
        Args:
            filepath: Path to load model from
        """
        model_data = joblib.load(filepath)
        
        self.model = model_data['model']
        self.calibrator = model_data.get('calibrator')
        self.config = model_data['config']
        self.feature_names = model_data.get('feature_names')
        self.is_fitted = model_data.get('is_fitted', True)
        
        log.info("model_loaded", filepath=filepath)
    
    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """Get parameters for sklearn compatibility.
        
        Args:
            deep: Whether to get nested parameters
            
        Returns:
            Parameters dictionary
        """
        return {'config': self.config}
    
    def set_params(self, **params) -> 'BaseXGBoost':
        """Set parameters for sklearn compatibility.
        
        Args:
            **params: Parameters to set
            
        Returns:
            Self
        """
        if 'config' in params:
            self.config = params['config']
            self._init_model()
        
        return self
    
    def score(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray]
    ) -> float:
        """Calculate accuracy score.
        
        Args:
            X: Features
            y: True labels
            
        Returns:
            Accuracy score
        """
        predictions = self.predict(X)
        return np.mean(predictions == y)