"""XGBoost Optuna optimization orchestrator."""

import numpy as np
import pandas as pd
import xgboost as xgb
from typing import Dict, Tuple, Optional, Any
import optuna
from optuna.pruners import MedianPruner, HyperbandPruner, SuccessiveHalvingPruner
from optuna.samplers import TPESampler
from sklearn.metrics import f1_score, precision_recall_curve, auc
from sklearn.calibration import CalibratedClassifierCV
import warnings

from .config import XGBOptunaConfig, OptimizationMetrics
from .threshold import ThresholdOptimizer
from .calibration import ModelCalibrator
from .utils import get_logger, calculate_scale_pos_weight
from .training import create_objective, train_fold

try:
    from optuna.integration import XGBoostPruningCallback
    OPTUNA_XGB_INTEGRATION = True
except ImportError:
    OPTUNA_XGB_INTEGRATION = False

warnings.filterwarnings('ignore')
log = get_logger()


class XGBOptuna:
    """XGBoost classifier with Optuna optimization and mandatory calibration."""
    
    def __init__(self, config: Optional[XGBOptunaConfig] = None):
        """
        Initialize XGBoost optimizer.
        
        Args:
            config: Configuration object
        """
        self.config = config or XGBOptunaConfig()
        
        self.best_model = None
        self.best_params = None
        self.best_score = -np.inf
        self.calibrator = None
        self.threshold_optimizer = ThresholdOptimizer(self.config.threshold_strategies)
        self.model_calibrator = ModelCalibrator(
            method=self.config.calibration_method,
            cv=self.config.calibration_cv
        )
        
        # Tracking
        self.study = None
        self.metrics = None
        self.feature_names_ = None
        
        log.info(
            "xgb_optuna_initialized",
            n_trials=self.config.n_trials,
            cv_folds=self.config.cv_folds,
            embargo=self.config.embargo,
            pruner=self.config.pruner_type
        )
    
    
    def _get_pruner(self) -> optuna.pruners.BasePruner:
        """Get Optuna pruner based on configuration."""
        if self.config.pruner_type == 'median':
            return MedianPruner(n_startup_trials=5, n_warmup_steps=10)
        elif self.config.pruner_type == 'hyperband':
            return HyperbandPruner(min_resource=1, max_resource='auto')
        elif self.config.pruner_type == 'successive_halving':
            return SuccessiveHalvingPruner()
        else:
            return MedianPruner(n_startup_trials=5, n_warmup_steps=10)
    
    
    def optimize(self, X: pd.DataFrame, y: pd.Series, 
                 sample_weights: Optional[np.ndarray] = None):
        """
        Run Optuna optimization.
        
        Args:
            X: Features
            y: Labels
            sample_weights: Optional sample weights
        """
        # Store feature names
        self.feature_names_ = X.columns.tolist()
        
        # Create study
        self.study = optuna.create_study(
            direction='maximize',
            pruner=self._get_pruner(),
            sampler=TPESampler(seed=self.config.seed)
        )
        
        # Create objective function using training module
        objective = create_objective(X, y, self.config, self.threshold_optimizer, sample_weights)
        
        # Optimize
        self.study.optimize(
            objective,
            n_trials=self.config.n_trials,
            show_progress_bar=True
        )
        
        # Store best parameters
        self.best_params = self.study.best_params
        self.best_score = self.study.best_value
        
        log.info(
            "optimization_complete",
            best_score=self.best_score,
            best_params=self.best_params
        )
    
    def fit_final_model(self, X: pd.DataFrame, y: pd.Series,
                       sample_weights: Optional[np.ndarray] = None):
        """
        Fit final model with best parameters.
        
        Args:
            X: Features
            y: Labels
            sample_weights: Optional sample weights
        """
        if self.best_params is None:
            raise ValueError("Must run optimize() before fit_final_model()")
        
        # Add fixed parameters
        params = self.best_params.copy()
        params.update({
            'tree_method': self.config.tree_method,
            'device': self.config.device,
            'objective': 'binary:logistic',
            'eval_metric': 'aucpr',
            'use_label_encoder': False,
            'random_state': self.config.seed,
            'scale_pos_weight': calculate_scale_pos_weight(y)
        })
        
        # Create final model
        self.best_model = xgb.XGBClassifier(**params)
        
        # Split for validation
        split_idx = int(len(X) * 0.8)
        X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]
        
        if sample_weights is not None:
            w_train = sample_weights[:split_idx] if isinstance(sample_weights, np.ndarray) \
                     else sample_weights.iloc[:split_idx]
        else:
            w_train = None
        
        # Train with early stopping
        self.best_model.fit(
            X_train, y_train,
            sample_weight=w_train,
            eval_set=[(X_val, y_val)],
            eval_metric='aucpr',
            verbose=False,
            callbacks=[xgb.callback.EarlyStopping(
                rounds=self.config.early_stopping_rounds, save_best=True)]
        )
        
        # Calibrate model
        self.calibrator = self.model_calibrator.calibrate(self.best_model, X, y)
        
        # Optimize thresholds
        y_pred_proba = self.calibrator.predict_proba(X)[:, 1]
        
        if self.config.optimize_threshold:
            self.threshold_optimizer.optimize_all(y, y_pred_proba, 
                costs={'fee_bps': 5, 'slippage_bps': 5})
        
        log.info("final_model_trained_and_calibrated")
    
    def fit(self, X: pd.DataFrame, y: pd.Series,
            sample_weights: Optional[np.ndarray] = None):
        """
        Full training pipeline: optimize and fit final model.
        
        Args:
            X: Features
            y: Labels
            sample_weights: Optional sample weights
        """
        self.optimize(X, y, sample_weights)
        self.fit_final_model(X, y, sample_weights)
        return self
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict probabilities using calibrated model.
        
        Args:
            X: Features
            
        Returns:
            Probability predictions
        """
        if self.calibrator is None:
            raise ValueError("Model not fitted")
        
        return self.calibrator.predict_proba(X)
    
    def predict(self, X: pd.DataFrame, threshold_type: str = 'f1') -> np.ndarray:
        """
        Predict classes using specified threshold.
        
        Args:
            X: Features
            threshold_type: Type of threshold ('f1', 'ev', 'profit')
            
        Returns:
            Class predictions
        """
        proba = self.predict_proba(X)[:, 1]
        
        threshold = self.threshold_optimizer.thresholds.get(threshold_type, 0.5)
        return (proba >= threshold).astype(int)
    
    @property
    def feature_importances_(self):
        """Get feature importances from best model."""
        if self.best_model is None:
            raise ValueError("Model not fitted")
        return self.best_model.feature_importances_
    
    def get_params(self, deep: bool = True) -> Dict:
        """Get parameters for sklearn compatibility."""
        return {'config': self.config}
    
    def set_params(self, **params) -> 'XGBOptuna':
        """Set parameters for sklearn compatibility."""
        for key, value in params.items():
            if key == 'config':
                self.config = value
            elif hasattr(self.config, key):
                setattr(self.config, key, value)
        return self