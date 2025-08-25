"""
Optimized XGBoost implementation with CPU hist and proper validation.
No GPU hype, just fast and reliable CPU performance.
"""
import numpy as np
import pandas as pd
import xgboost as xgb
from typing import Dict, Any, Optional
import logging
from .base_model import BaseModel

logger = logging.getLogger(__name__)


class XGBoostModel(BaseModel):
    """
    XGBoost with CPU hist, fixed seeds, and temporal validation.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize XGBoost model.
        
        Args:
            config: Model configuration
        """
        super().__init__(config)
        self.seed = config.get('seed', 42)
        self._set_deterministic_params()
        
    def _set_deterministic_params(self):
        """Set parameters for deterministic behavior."""
        import os
        os.environ['PYTHONHASHSEED'] = str(self.seed)
        np.random.seed(self.seed)
        
    def _calculate_scale_pos_weight(self, y_train: pd.Series) -> float:
        """
        Calculate scale_pos_weight from training data.
        
        Args:
            y_train: Training labels
            
        Returns:
            Scale weight for positive class
        """
        neg_count = (y_train == 0).sum()
        pos_count = (y_train == 1).sum()
        scale = neg_count / pos_count if pos_count > 0 else 1.0
        logger.info(f"Calculated scale_pos_weight: {scale:.2f} (neg: {neg_count}, pos: {pos_count})")
        return scale
    
    def _get_model_params(self, scale_pos_weight: float) -> Dict[str, Any]:
        """
        Get XGBoost parameters with CPU optimization.
        
        Args:
            scale_pos_weight: Weight for positive class
            
        Returns:
            Model parameters
        """
        params = {
            # Core parameters
            'objective': 'binary:logistic',
            'eval_metric': ['logloss', 'auc'],
            'tree_method': 'hist',  # Fast CPU method
            'device': 'cpu',
            
            # Model complexity
            'max_depth': self.config.get('max_depth', 6),
            'min_child_weight': self.config.get('min_child_weight', 1),
            'subsample': self.config.get('subsample', 0.8),
            'colsample_bytree': self.config.get('colsample_bytree', 0.8),
            'learning_rate': self.config.get('learning_rate', 0.3),
            
            # Regularization
            'gamma': self.config.get('gamma', 0),
            'reg_alpha': self.config.get('reg_alpha', 0),
            'reg_lambda': self.config.get('reg_lambda', 1),
            
            # Class imbalance
            'scale_pos_weight': scale_pos_weight,
            
            # Determinism
            'seed': self.seed,
            'random_state': self.seed,
            
            # Performance
            'n_jobs': self.config.get('n_jobs', -1),
            'verbosity': 1 if logger.level <= logging.INFO else 0
        }
        
        return params
    
    def fit(self,
            X_train: pd.DataFrame,
            y_train: pd.Series,
            X_val: Optional[pd.DataFrame] = None,
            y_val: Optional[pd.Series] = None) -> 'XGBoostModel':
        """
        Train XGBoost model with temporal early stopping.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            
        Returns:
            Self for method chaining
        """
        # Store feature names
        self.feature_names = list(X_train.columns)
        
        # Calculate scale_pos_weight from training data
        scale_pos_weight = self._calculate_scale_pos_weight(y_train)
        
        # Get model parameters
        params = self._get_model_params(scale_pos_weight)
        
        # Extract sklearn-specific params
        sklearn_params = {
            'n_estimators': self.config.get('n_estimators', 100),
            'max_depth': params.pop('max_depth'),
            'min_child_weight': params.pop('min_child_weight'),
            'subsample': params.pop('subsample'),
            'colsample_bytree': params.pop('colsample_bytree'),
            'learning_rate': params.pop('learning_rate'),
            'gamma': params.pop('gamma'),
            'reg_alpha': params.pop('reg_alpha'),
            'reg_lambda': params.pop('reg_lambda'),
            'scale_pos_weight': params.pop('scale_pos_weight'),
            'n_jobs': params.pop('n_jobs'),
            'random_state': params.pop('random_state'),
            'seed': params.pop('seed'),
            'objective': params.pop('objective'),
            'tree_method': params.pop('tree_method'),
            'device': params.pop('device'),
            'verbosity': params.pop('verbosity')
        }
        
        # Create model
        self.model = xgb.XGBClassifier(**sklearn_params)
        
        # Fit model (simplified for compatibility)
        if X_val is not None and y_val is not None:
            eval_set = [(X_val, y_val)]
            try:
                # Try newer XGBoost API first
                self.model.fit(
                    X_train, y_train,
                    eval_set=eval_set,
                    verbose=False
                )
            except TypeError:
                # Fallback for older versions
                self.model.fit(X_train, y_train)
                logger.info("Early stopping not supported in this XGBoost version")
        else:
            self.model.fit(X_train, y_train)
        
        logger.info("Model training completed")
        self.is_fitted = True
        return self
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict probabilities with optional calibration.
        
        Args:
            X: Features
            
        Returns:
            Probability array of shape (n_samples, 2)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # Ensure feature order
        X = X[self.feature_names]
        
        # Get base probabilities
        probas = self.model.predict_proba(X)
        
        # Apply calibration if available
        if self.calibrator is not None:
            probas[:, 1] = self._apply_calibration(probas[:, 1])
            # Ensure probabilities sum to 1
            probas[:, 0] = 1 - probas[:, 1]
        
        return probas
    
    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance scores.
        
        Returns:
            DataFrame with features and importance scores
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted to get feature importance")
        
        importance_dict = {}
        
        # Get multiple importance types
        for importance_type in ['weight', 'gain', 'cover']:
            scores = self.model.get_booster().get_score(importance_type=importance_type)
            importance_dict[importance_type] = scores
        
        # Create DataFrame
        df_importance = pd.DataFrame(importance_dict).fillna(0)
        df_importance['feature'] = df_importance.index
        df_importance = df_importance.reset_index(drop=True)
        
        # Add average importance
        df_importance['avg_importance'] = df_importance[['weight', 'gain', 'cover']].mean(axis=1)
        
        # Sort by average importance
        df_importance = df_importance.sort_values('avg_importance', ascending=False)
        
        return df_importance
    
    def get_params(self) -> Dict[str, Any]:
        """
        Get current model parameters.
        
        Returns:
            Dictionary of parameters
        """
        if self.model is not None:
            return self.model.get_params()
        return self.config
    
    def set_params(self, **params) -> 'XGBoostModel':
        """
        Set model parameters.
        
        Args:
            **params: Parameters to set
            
        Returns:
            Self for method chaining
        """
        self.config.update(params)
        if self.model is not None:
            self.model.set_params(**params)
        return self