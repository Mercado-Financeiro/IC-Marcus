"""Training utilities for XGBoost models."""

import numpy as np
import pandas as pd
import xgboost as xgb
from typing import Dict, Tuple, Optional, Any
import optuna
from sklearn.metrics import f1_score

from .utils import get_logger, check_constant_predictions, calculate_scale_pos_weight
from .threshold import ThresholdOptimizer

try:
    from optuna.integration import XGBoostPruningCallback
    OPTUNA_XGB_INTEGRATION = True
except ImportError:
    OPTUNA_XGB_INTEGRATION = False

log = get_logger()


def create_objective(
    X: pd.DataFrame, 
    y: pd.Series,
    config: Any,
    threshold_optimizer: ThresholdOptimizer,
    sample_weights: Optional[np.ndarray] = None
):
    """
    Create objective function for Optuna optimization.
    
    Args:
        X: Features
        y: Labels
        config: Configuration object
        threshold_optimizer: Threshold optimizer instance
        sample_weights: Optional sample weights
        
    Returns:
        Objective function
    """
    def objective(trial: optuna.Trial) -> float:
        """Inner objective function."""
        # Get hyperparameters
        params = create_search_space(trial, config)
        
        # Add scale_pos_weight for imbalanced data
        params['scale_pos_weight'] = calculate_scale_pos_weight(y)
        
        # Import temporal validator
        from src.features.validation.temporal import TemporalValidator, TemporalValidationConfig
        
        # Configure temporal validation
        val_config = TemporalValidationConfig(
            n_splits=config.cv_folds,
            embargo=config.embargo,
            check_leakage=True
        )
        validator = TemporalValidator(val_config)
        
        scores = []
        
        for fold_idx, (train_idx, val_idx) in enumerate(validator.split(X, y, strategy='purged_kfold')):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            # Sample weights
            if sample_weights is not None:
                if isinstance(sample_weights, pd.Series):
                    w_train = sample_weights.iloc[train_idx].values
                else:
                    w_train = sample_weights[train_idx]
            else:
                w_train = None
            
            # Train model for this fold
            model, y_pred_proba = train_fold(
                X_train, y_train, X_val, y_val, 
                params, config, w_train, trial
            )
            
            if model is None:
                return -1.0  # Failed training
            
            # Check for constant predictions
            if check_constant_predictions(y_pred_proba):
                log.warning("constant_predictions_in_trial", trial=trial.number)
                return -1.0
            
            # Optimize threshold and calculate F1
            threshold = threshold_optimizer.optimize_f1(y_val, y_pred_proba)
            y_pred = (y_pred_proba >= threshold).astype(int)
            f1 = f1_score(y_val, y_pred, zero_division=0)
            
            scores.append(f1)
            
            # Report intermediate value for pruning
            trial.report(f1, fold_idx)
            if trial.should_prune():
                raise optuna.TrialPruned()
        
        return np.mean(scores)
    
    return objective


def create_search_space(trial: optuna.Trial, config: Any) -> Dict:
    """
    Create Optuna search space from configuration.
    
    Args:
        trial: Optuna trial
        config: Configuration object
        
    Returns:
        Dictionary of hyperparameters
    """
    return {
        'n_estimators': trial.suggest_int('n_estimators', 
            config.n_estimators_min, config.n_estimators_max),
        'max_depth': trial.suggest_int('max_depth',
            config.max_depth_min, config.max_depth_max),
        'learning_rate': trial.suggest_float('learning_rate',
            config.learning_rate_min, config.learning_rate_max, log=True),
        'subsample': trial.suggest_float('subsample',
            config.subsample_min, config.subsample_max),
        'colsample_bytree': trial.suggest_float('colsample_bytree',
            config.colsample_bytree_min, config.colsample_bytree_max),
        'gamma': trial.suggest_float('gamma',
            config.gamma_min, config.gamma_max),
        'reg_alpha': trial.suggest_float('reg_alpha',
            config.reg_alpha_min, config.reg_alpha_max),
        'reg_lambda': trial.suggest_float('reg_lambda',
            config.reg_lambda_min, config.reg_lambda_max),
        'min_child_weight': trial.suggest_int('min_child_weight',
            config.min_child_weight_min, config.min_child_weight_max),
        'tree_method': config.tree_method,
        'device': config.device,
        'objective': 'binary:logistic',
        'eval_metric': 'aucpr',
        'use_label_encoder': False,
        'random_state': config.seed
    }


def train_fold(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    params: Dict,
    config: Any,
    sample_weight: Optional[np.ndarray] = None,
    trial: Optional[optuna.Trial] = None
) -> Tuple[Optional[xgb.XGBClassifier], Optional[np.ndarray]]:
    """
    Train XGBoost model for a single fold.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        params: Model parameters
        config: Configuration object
        sample_weight: Optional sample weights
        trial: Optional Optuna trial for pruning
        
    Returns:
        Tuple of (trained model, validation predictions)
    """
    try:
        # Create model
        model = xgb.XGBClassifier(**params)
        
        # Prepare callbacks
        callbacks = [xgb.callback.EarlyStopping(
            rounds=config.early_stopping_rounds, save_best=True)]
        
        if OPTUNA_XGB_INTEGRATION and trial is not None:
            callbacks.append(XGBoostPruningCallback(trial, "validation_0-aucpr"))
        
        # Fit model
        model.fit(
            X_train, y_train,
            sample_weight=sample_weight,
            eval_set=[(X_val, y_val)],
            eval_metric='aucpr',
            verbose=False,
            callbacks=callbacks
        )
        
        # Get predictions
        y_pred_proba = model.predict_proba(X_val)[:, 1]
        
        return model, y_pred_proba
        
    except Exception as e:
        log.error(f"Error training fold: {str(e)}")
        return None, None