"""
XGBoost with quantile regression for uncertainty estimation.

Implements quantile predictions to provide confidence bands around point estimates.
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from typing import Dict, List, Optional, Tuple, Any
from sklearn.base import BaseEstimator, ClassifierMixin
# Use temporal validator instead of sklearn TimeSeriesSplit
from src.features.validation.temporal import TemporalValidator, TemporalValidationConfig
from sklearn.metrics import roc_auc_score, brier_score_loss
import optuna
from optuna.pruners import MedianPruner, HyperbandPruner
import structlog
import joblib
from pathlib import Path

log = structlog.get_logger()


class XGBQuantileRegressor(BaseEstimator):
    """XGBoost quantile regressor for uncertainty estimation."""
    
    def __init__(
        self,
        quantiles: List[float] = [0.1, 0.25, 0.5, 0.75, 0.9],
        n_estimators: int = 500,
        max_depth: int = 6,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        min_child_weight: int = 3,
        gamma: float = 0.1,
        reg_alpha: float = 0.01,
        reg_lambda: float = 1.0,
        random_state: int = 42,
        n_jobs: int = -1,
        tree_method: str = "hist",
        early_stopping_rounds: int = 50,
    ):
        """
        Initialize XGBoost quantile regressor.
        
        Args:
            quantiles: List of quantiles to predict
            n_estimators: Number of trees
            max_depth: Maximum tree depth
            learning_rate: Learning rate
            subsample: Subsample ratio
            colsample_bytree: Column subsample ratio
            min_child_weight: Minimum child weight
            gamma: Minimum loss reduction
            reg_alpha: L1 regularization
            reg_lambda: L2 regularization
            random_state: Random seed
            n_jobs: Number of parallel threads
            tree_method: Tree construction algorithm
            early_stopping_rounds: Early stopping rounds
        """
        self.quantiles = sorted(quantiles)
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.min_child_weight = min_child_weight
        self.gamma = gamma
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.tree_method = tree_method
        self.early_stopping_rounds = early_stopping_rounds
        self.models_ = {}
        
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        eval_set: Optional[List[Tuple]] = None,
        verbose: bool = False
    ):
        """
        Fit quantile models.
        
        Args:
            X: Features
            y: Target values
            eval_set: Validation set for early stopping
            verbose: Verbosity flag
        """
        log.info("fitting_quantile_models", n_quantiles=len(self.quantiles))
        
        for quantile in self.quantiles:
            log.info(f"fitting_quantile_{quantile}")
            
            # Create model for this quantile
            params = {
                'objective': 'reg:quantileerror',
                'quantile_alpha': quantile,
                'n_estimators': self.n_estimators,
                'max_depth': self.max_depth,
                'learning_rate': self.learning_rate,
                'subsample': self.subsample,
                'colsample_bytree': self.colsample_bytree,
                'min_child_weight': self.min_child_weight,
                'gamma': self.gamma,
                'reg_alpha': self.reg_alpha,
                'reg_lambda': self.reg_lambda,
                'random_state': self.random_state,
                'n_jobs': self.n_jobs,
                'tree_method': self.tree_method,
                'early_stopping_rounds': self.early_stopping_rounds,
            }
            
            model = xgb.XGBRegressor(**params)
            
            # Fit model
            if eval_set and self.early_stopping_rounds:
                model.fit(
                    X, y,
                    eval_set=eval_set,
                    verbose=verbose
                )
            else:
                # Fit without early stopping if no eval_set
                params_no_early = params.copy()
                params_no_early.pop('early_stopping_rounds', None)
                model = xgb.XGBRegressor(**params_no_early)
                model.fit(X, y, verbose=verbose)
            
            self.models_[quantile] = model
        
        return self
    
    def predict(self, X: np.ndarray) -> Dict[float, np.ndarray]:
        """
        Predict quantiles.
        
        Args:
            X: Features
            
        Returns:
            Dictionary mapping quantiles to predictions
        """
        predictions = {}
        
        for quantile, model in self.models_.items():
            predictions[quantile] = model.predict(X)
        
        return predictions
    
    def predict_intervals(
        self,
        X: np.ndarray,
        alpha: float = 0.1
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Predict with confidence intervals.
        
        Args:
            X: Features
            alpha: Significance level (e.g., 0.1 for 90% CI)
            
        Returns:
            Tuple of (lower_bound, median, upper_bound)
        """
        predictions = self.predict(X)
        
        # Get quantiles for confidence interval
        lower_q = alpha / 2
        upper_q = 1 - alpha / 2
        
        # Find closest quantiles
        lower_idx = np.argmin(np.abs(np.array(self.quantiles) - lower_q))
        upper_idx = np.argmin(np.abs(np.array(self.quantiles) - upper_q))
        median_idx = np.argmin(np.abs(np.array(self.quantiles) - 0.5))
        
        lower_bound = predictions[self.quantiles[lower_idx]]
        upper_bound = predictions[self.quantiles[upper_idx]]
        median = predictions[self.quantiles[median_idx]]
        
        return lower_bound, median, upper_bound
    
    def get_uncertainty(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate prediction uncertainty.
        
        Args:
            X: Features
            
        Returns:
            Uncertainty scores (IQR or similar)
        """
        predictions = self.predict(X)
        
        # Use IQR as uncertainty measure
        q25_idx = np.argmin(np.abs(np.array(self.quantiles) - 0.25))
        q75_idx = np.argmin(np.abs(np.array(self.quantiles) - 0.75))
        
        q25 = predictions[self.quantiles[q25_idx]]
        q75 = predictions[self.quantiles[q75_idx]]
        
        return q75 - q25


class XGBQuantileClassifier(BaseEstimator, ClassifierMixin):
    """XGBoost classifier with quantile-based uncertainty."""
    
    def __init__(
        self,
        quantiles: List[float] = [0.1, 0.25, 0.5, 0.75, 0.9],
        threshold: float = 0.5,
        uncertainty_threshold: Optional[float] = None,
        **xgb_params
    ):
        """
        Initialize quantile classifier.
        
        Args:
            quantiles: Quantiles to predict
            threshold: Classification threshold
            uncertainty_threshold: Max uncertainty for predictions
            **xgb_params: XGBoost parameters
        """
        self.quantiles = quantiles
        self.threshold = threshold
        self.uncertainty_threshold = uncertainty_threshold
        self.xgb_params = xgb_params
        self.regressor_ = None
        self._estimator_type = "classifier"
        self.classes_ = np.array([0, 1])
        
    def fit(self, X, y, **fit_params):
        """Fit the quantile classifier."""
        # Create quantile regressor
        self.regressor_ = XGBQuantileRegressor(
            quantiles=self.quantiles,
            **self.xgb_params
        )
        
        # Fit on continuous target (can be probabilities)
        self.regressor_.fit(X, y, **fit_params)
        
        return self
    
    def predict_proba(self, X):
        """Predict probabilities with uncertainty."""
        # Get quantile predictions
        predictions = self.regressor_.predict(X)
        
        # Use median as main prediction
        median_idx = np.argmin(np.abs(np.array(self.quantiles) - 0.5))
        median_pred = predictions[self.quantiles[median_idx]]
        
        # Clip to [0, 1] for valid probabilities
        median_pred = np.clip(median_pred, 0, 1)
        
        # Create probability array
        proba = np.column_stack([1 - median_pred, median_pred])
        
        # Apply uncertainty filtering if threshold set
        if self.uncertainty_threshold is not None:
            uncertainty = self.regressor_.get_uncertainty(X)
            uncertain_mask = uncertainty > self.uncertainty_threshold
            
            # Set uncertain predictions to 50/50
            proba[uncertain_mask] = [0.5, 0.5]
        
        return proba
    
    def predict(self, X):
        """Predict classes."""
        proba = self.predict_proba(X)
        return (proba[:, 1] >= self.threshold).astype(int)
    
    def predict_with_uncertainty(self, X):
        """
        Predict with uncertainty bands.
        
        Returns:
            Dictionary with predictions, probabilities, and uncertainty
        """
        # Get all quantile predictions
        predictions = self.regressor_.predict(X)
        
        # Get confidence intervals
        lower, median, upper = self.regressor_.predict_intervals(X, alpha=0.1)
        
        # Get uncertainty
        uncertainty = self.regressor_.get_uncertainty(X)
        
        # Get class predictions
        proba = self.predict_proba(X)
        classes = self.predict(X)
        
        return {
            'class': classes,
            'probability': proba[:, 1],
            'lower_bound': np.clip(lower, 0, 1),
            'median': np.clip(median, 0, 1),
            'upper_bound': np.clip(upper, 0, 1),
            'uncertainty': uncertainty,
            'quantiles': {q: np.clip(pred, 0, 1) 
                         for q, pred in predictions.items()}
        }


class XGBQuantileOptuna:
    """Optuna optimization for XGBoost quantile models."""
    
    def __init__(
        self,
        quantiles: List[float] = [0.1, 0.25, 0.5, 0.75, 0.9],
        n_trials: int = 100,
        cv_folds: int = 5,
        embargo: int = 10,
        random_state: int = 42,
        pruner: str = "hyperband",
        direction: str = "maximize",
        metric: str = "roc_auc",
    ):
        """
        Initialize Optuna optimizer for quantile models.
        
        Args:
            quantiles: Quantiles to optimize
            n_trials: Number of optimization trials
            cv_folds: Number of CV folds
            embargo: Embargo period for purged CV
            random_state: Random seed
            pruner: Pruner type (median, hyperband)
            direction: Optimization direction
            metric: Metric to optimize
        """
        self.quantiles = quantiles
        self.n_trials = n_trials
        self.cv_folds = cv_folds
        self.embargo = embargo
        self.random_state = random_state
        self.pruner = pruner
        self.direction = direction
        self.metric = metric
        self.best_params_ = None
        self.best_model_ = None
        self.study_ = None
        
    def objective(self, trial, X, y, sample_weight=None):
        """Optuna objective function."""
        # Suggest hyperparameters
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'gamma': trial.suggest_float('gamma', 0, 1),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-4, 1.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-4, 10.0, log=True),
            'random_state': self.random_state,
        }
        
        # Create model
        model = XGBQuantileClassifier(
            quantiles=self.quantiles,
            **params
        )
        
        # Time series cross-validation with embargo
        val_config = TemporalValidationConfig(n_splits=self.cv_folds, embargo=10)
        validator = TemporalValidator(val_config)
        scores = []
        
        for fold, (train_idx, val_idx) in enumerate(validator.split(X, y, strategy='purged_kfold')):
            # Apply embargo
            if self.embargo > 0:
                val_start = val_idx[0]
                train_mask = train_idx < (val_start - self.embargo)
                train_idx = train_idx[train_mask]
            
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Apply sample weights if provided
            if sample_weight is not None:
                w_train = sample_weight[train_idx]
                model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False, sample_weight=w_train)
            else:
                model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
            
            # Evaluate
            if self.metric == "roc_auc":
                y_pred = model.predict_proba(X_val)[:, 1]
                score = roc_auc_score(y_val, y_pred)
            elif self.metric == "brier":
                y_pred = model.predict_proba(X_val)[:, 1]
                score = -brier_score_loss(y_val, y_pred)  # Negative for maximization
            else:
                raise ValueError(f"Unknown metric: {self.metric}")
            
            scores.append(score)
            
            # Report intermediate value for pruning
            trial.report(score, fold)
            
            # Handle pruning
            if trial.should_prune():
                raise optuna.TrialPruned()
        
        return np.mean(scores)
    
    def optimize(self, X, y, sample_weight=None, verbose=True):
        """
        Run Optuna optimization.
        
        Args:
            X: Features
            y: Labels
            sample_weight: Sample weights
            verbose: Whether to show progress
            
        Returns:
            Best model
        """
        log.info("starting_quantile_optimization", n_trials=self.n_trials)
        
        # Create pruner
        if self.pruner == "median":
            pruner = MedianPruner(n_startup_trials=10, n_warmup_steps=5)
        elif self.pruner == "hyperband":
            pruner = HyperbandPruner(min_resource=1, max_resource=self.cv_folds)
        else:
            pruner = None
        
        # Create study
        self.study_ = optuna.create_study(
            direction=self.direction,
            pruner=pruner,
            sampler=optuna.samplers.TPESampler(seed=self.random_state)
        )
        
        # Optimize
        self.study_.optimize(
            lambda trial: self.objective(trial, X, y, sample_weight),
            n_trials=self.n_trials,
            show_progress_bar=verbose
        )
        
        # Get best parameters
        self.best_params_ = self.study_.best_params
        log.info("optimization_complete", best_score=self.study_.best_value)
        
        # Train final model with best parameters
        self.best_model_ = XGBQuantileClassifier(
            quantiles=self.quantiles,
            **self.best_params_
        )
        
        if sample_weight is not None:
            self.best_model_.fit(X, y, sample_weight=sample_weight)
        else:
            self.best_model_.fit(X, y)
        
        return self.best_model_
    
    def save(self, path: str):
        """Save the best model and study."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save model
        joblib.dump(self.best_model_, path / "xgb_quantile_model.pkl")
        
        # Save parameters
        params = {
            'best_params': self.best_params_,
            'quantiles': self.quantiles,
            'metric': self.metric,
            'best_score': self.study_.best_value if self.study_ else None
        }
        joblib.dump(params, path / "xgb_quantile_params.pkl")
        
        log.info("model_saved", path=str(path))
    
    @classmethod
    def load(cls, path: str):
        """Load a saved model."""
        path = Path(path)
        
        # Load model
        model = joblib.load(path / "xgb_quantile_model.pkl")
        
        # Load parameters
        params = joblib.load(path / "xgb_quantile_params.pkl")
        
        # Create instance
        instance = cls(quantiles=params['quantiles'])
        instance.best_model_ = model
        instance.best_params_ = params['best_params']
        
        return instance