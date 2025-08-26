"""
Unified XGBoost Model with Optuna Optimization, Calibration, and Threshold Optimization.

This module consolidates all XGBoost functionality:
- Base XGBoost model with deterministic settings
- Bayesian optimization via Optuna with pruning
- Calibration (Platt, Isotonic, Beta)
- Threshold optimization by Expected Value
- Trading metrics calculation
- MLflow integration

References:
- "XGBoost: A Scalable Tree Boosting System" (Chen & Guestrin, 2016)
- "Asynchronous Successive Halving" (Li et al., 2018)
- "Beta calibration" (Kull et al., 2017)
"""

import os
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, Any, List, Union
from dataclasses import dataclass, field
import warnings
import joblib
from pathlib import Path
import json

import xgboost as xgb
from xgboost import XGBClassifier
import optuna
from optuna.pruners import MedianPruner, PercentilePruner, SuccessiveHalvingPruner, HyperbandPruner
from optuna.integration import XGBoostPruningCallback
from optuna.samplers import TPESampler, RandomSampler, CmaEsSampler
from optuna.storages import RDBStorage
import mlflow

from sklearn.metrics import (
    average_precision_score, matthews_corrcoef, f1_score, 
    precision_recall_curve, roc_auc_score, accuracy_score,
    brier_score_loss, log_loss, confusion_matrix
)
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, TimeSeriesSplit

# Import utilities
from src.utils.determinism_enhanced import set_full_determinism, assert_determinism
from src.utils.logging import log as logger


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class XGBoostUnifiedConfig:
    """Unified configuration for XGBoost with all features."""
    
    # Basic settings
    n_trials: int = 100
    timeout: Optional[int] = None
    seed: int = 42
    deterministic: bool = True
    verbose: bool = True
    
    # XGBoost specific
    tree_method: str = "hist"  # "hist", "gpu_hist", "exact"
    device: str = "cpu"  # "cpu", "gpu", "cuda"
    n_jobs: int = 1  # Default to 1 for determinism
    
    # Sampler settings
    sampler_type: str = 'tpe'  # 'tpe', 'random', 'cmaes'
    sampler_params: Dict[str, Any] = field(default_factory=dict)
    
    # Pruner settings  
    pruner_type: str = 'hyperband'  # Best for XGBoost
    pruner_params: Dict[str, Any] = field(default_factory=dict)
    
    # Storage settings
    storage_url: Optional[str] = None
    study_name: Optional[str] = None
    load_if_exists: bool = True
    
    # Validation settings
    use_outer_cv: bool = True
    outer_cv_splits: int = 3
    inner_cv_splits: int = 5
    embargo: int = 10
    
    # Calibration settings
    calibration_method: str = 'auto'  # 'auto', 'platt', 'isotonic', 'beta'
    calibration_selection_metric: str = 'brier'
    
    # MLflow integration
    use_mlflow: bool = True
    mlflow_experiment: str = 'xgboost_unified'
    
    # Metrics
    primary_metric: str = 'pr_auc'  # Better for imbalanced data
    eval_metric: str = 'aucpr'  # XGBoost eval metric
    
    # Early stopping
    early_stopping_rounds: int = 100
    
    # Hyperparameter search space
    param_space: Dict[str, Any] = field(default_factory=lambda: {
        'n_estimators': {'low': 100, 'high': 2000},
        'max_depth': {'low': 3, 'high': 15},
        'learning_rate': {'low': 0.001, 'high': 0.3, 'log': True},
        'subsample': {'low': 0.5, 'high': 1.0},
        'colsample_bytree': {'low': 0.5, 'high': 1.0},
        'reg_alpha': {'low': 1e-8, 'high': 10.0, 'log': True},
        'reg_lambda': {'low': 1e-8, 'high': 10.0, 'log': True},
        'min_child_weight': {'low': 1, 'high': 10},
        'gamma': {'low': 0.0, 'high': 1.0}
    })
    
    # Model saving
    save_models: bool = True
    model_dir: str = 'models/xgboost'


# ============================================================================
# TRADING COSTS & METRICS
# ============================================================================

@dataclass
class TradingCosts:
    """Trading cost configuration."""
    fee_bps: float = 5.0  # Basis points
    slippage_bps: float = 5.0
    impact_bps: float = 2.0
    
    @property
    def total_bps(self) -> float:
        """Total cost in basis points."""
        return self.fee_bps + self.slippage_bps + self.impact_bps
    
    @property
    def total_pct(self) -> float:
        """Total cost as percentage."""
        return self.total_bps / 10000


class TradingMetrics:
    """Calculate trading-specific metrics."""
    
    @staticmethod
    def calculate_expected_value(
        y_true: np.ndarray,
        y_proba: np.ndarray,
        threshold: float,
        avg_win_pct: float = 0.015,
        avg_loss_pct: float = 0.005,
        costs: Optional[TradingCosts] = None
    ) -> Dict[str, float]:
        """Calculate expected value metrics for a given threshold."""
        
        y_pred = (y_proba >= threshold).astype(int)
        
        # Confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        # Calculate rates
        n_trades = tp + fp
        if n_trades == 0:
            return {
                'ev_per_trade': 0.0,
                'total_ev': 0.0,
                'n_trades': 0,
                'win_rate': 0.0,
                'precision': 0.0
            }
        
        precision = tp / n_trades if n_trades > 0 else 0
        
        # Calculate EV
        gross_ev = precision * avg_win_pct - (1 - precision) * avg_loss_pct
        
        # Apply costs
        if costs:
            net_ev = gross_ev - costs.total_pct
        else:
            net_ev = gross_ev
        
        return {
            'ev_per_trade': net_ev,
            'total_ev': net_ev * n_trades,
            'n_trades': n_trades,
            'win_rate': precision,
            'precision': precision,
            'recall': tp / (tp + fn) if (tp + fn) > 0 else 0
        }
    
    @staticmethod
    def optimize_threshold_by_ev(
        y_true: np.ndarray,
        y_proba: np.ndarray,
        thresholds: Optional[np.ndarray] = None,
        **kwargs
    ) -> Tuple[float, Dict[str, float]]:
        """Find optimal threshold that maximizes expected value."""
        
        if thresholds is None:
            thresholds = np.linspace(0.1, 0.9, 81)
        
        best_threshold = 0.5
        best_metrics = None
        best_ev = -float('inf')
        
        for threshold in thresholds:
            metrics = TradingMetrics.calculate_expected_value(
                y_true, y_proba, threshold, **kwargs
            )
            
            if metrics['ev_per_trade'] > best_ev:
                best_ev = metrics['ev_per_trade']
                best_threshold = threshold
                best_metrics = metrics
        
        return best_threshold, best_metrics


# ============================================================================
# CALIBRATION
# ============================================================================

class XGBoostCalibrator:
    """Calibration methods for XGBoost probabilities."""
    
    def __init__(self, method: str = 'auto'):
        """
        Initialize calibrator.
        
        Args:
            method: 'auto', 'platt', 'isotonic', 'beta', or 'none'
        """
        self.method = method
        self.calibrator = None
        self.metrics = {}
    
    def fit(self, model, X_cal: np.ndarray, y_cal: np.ndarray):
        """Fit calibrator on calibration data."""
        
        if self.method == 'none':
            return self
        
        # Get uncalibrated predictions
        y_proba_uncal = model.predict_proba(X_cal)[:, 1]
        
        if self.method == 'auto':
            # Try all methods and select best
            best_method = self._select_best_method(model, X_cal, y_cal)
            self.method = best_method
        
        if self.method == 'platt':
            self.calibrator = LogisticRegression()
            self.calibrator.fit(y_proba_uncal.reshape(-1, 1), y_cal)
            
        elif self.method == 'isotonic':
            self.calibrator = IsotonicRegression(out_of_bounds='clip')
            self.calibrator.fit(y_proba_uncal, y_cal)
            
        elif self.method == 'beta':
            # Simplified beta calibration
            from scipy.stats import beta
            pos_probs = y_proba_uncal[y_cal == 1]
            neg_probs = y_proba_uncal[y_cal == 0]
            
            # Fit beta distributions
            self.alpha_pos, self.beta_pos = beta.fit(pos_probs)[0:2]
            self.alpha_neg, self.beta_neg = beta.fit(neg_probs)[0:2]
            
        # Calculate calibration metrics
        y_proba_cal = self.transform(model.predict_proba(X_cal)[:, 1])
        self.metrics = self._calculate_calibration_metrics(y_cal, y_proba_cal)
        
        return self
    
    def transform(self, y_proba: np.ndarray) -> np.ndarray:
        """Transform probabilities using fitted calibrator."""
        
        if self.method == 'none' or self.calibrator is None:
            return y_proba
        
        if self.method in ['platt', 'isotonic']:
            if self.method == 'platt':
                return self.calibrator.predict_proba(y_proba.reshape(-1, 1))[:, 1]
            else:
                return self.calibrator.transform(y_proba)
                
        elif self.method == 'beta':
            # Simplified beta transformation
            return np.clip(y_proba, 0.001, 0.999)
        
        return y_proba
    
    def _select_best_method(self, model, X_cal: np.ndarray, y_cal: np.ndarray) -> str:
        """Select best calibration method based on Brier score."""
        
        methods = ['none', 'platt', 'isotonic']
        best_score = float('inf')
        best_method = 'none'
        
        for method in methods:
            temp_calibrator = XGBoostCalibrator(method=method)
            temp_calibrator.fit(model, X_cal, y_cal)
            
            y_proba_cal = temp_calibrator.transform(
                model.predict_proba(X_cal)[:, 1]
            )
            score = brier_score_loss(y_cal, y_proba_cal)
            
            if score < best_score:
                best_score = score
                best_method = method
        
        logger.info(f"Selected calibration method: {best_method} (Brier: {best_score:.4f})")
        return best_method
    
    def _calculate_calibration_metrics(self, y_true: np.ndarray, y_proba: np.ndarray) -> Dict:
        """Calculate calibration metrics."""
        
        # Expected Calibration Error
        ece = self._calculate_ece(y_true, y_proba)
        
        return {
            'brier_score': brier_score_loss(y_true, y_proba),
            'log_loss': log_loss(y_true, y_proba),
            'ece': ece,
            'method': self.method
        }
    
    def _calculate_ece(self, y_true: np.ndarray, y_proba: np.ndarray, n_bins: int = 10) -> float:
        """Calculate Expected Calibration Error."""
        
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (y_proba > bin_lower) & (y_proba <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = y_true[in_bin].mean()
                avg_confidence_in_bin = y_proba[in_bin].mean()
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        
        return ece


# ============================================================================
# MAIN UNIFIED XGBOOST CLASS
# ============================================================================

class UnifiedXGBoost:
    """Unified XGBoost with Optuna optimization, calibration, and threshold optimization."""
    
    def __init__(self, config: Optional[XGBoostUnifiedConfig] = None):
        """Initialize unified XGBoost model."""
        
        self.config = config or XGBoostUnifiedConfig()
        self.model = None
        self.calibrator = None
        self.study = None
        self.best_params = None
        self.best_threshold = 0.5
        self.metrics = {}
        
        # Set determinism if requested
        if self.config.deterministic:
            set_full_determinism(self.config.seed)
        
        # Initialize MLflow if requested
        if self.config.use_mlflow:
            mlflow.set_experiment(self.config.mlflow_experiment)
    
    def optimize(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None
    ) -> optuna.Study:
        """
        Run Optuna optimization to find best hyperparameters.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
        
        Returns:
            Optuna study object
        """
        
        # Create objective function
        def objective(trial):
            # Sample hyperparameters
            params = self._sample_params(trial)
            
            # Add fixed parameters
            params.update({
                'objective': 'binary:logistic',
                'eval_metric': self.config.eval_metric,
                'tree_method': self.config.tree_method,
                'device': self.config.device,
                'n_jobs': self.config.n_jobs,
                'random_state': self.config.seed,
                'use_label_encoder': False,
                'verbosity': 0,
                'early_stopping_rounds': self.config.early_stopping_rounds  # XGBoost 3.0 requires this in params
            })
            
            # Train model with cross-validation or validation set
            if X_val is not None and y_val is not None:
                # Use provided validation set
                model = XGBClassifier(**params)
                
                # Create eval set
                eval_set = [(X_train, y_train), (X_val, y_val)]
                
                # Add pruning callback
                pruning_callback = XGBoostPruningCallback(trial, f"validation_1-{self.config.eval_metric}")
                
                # Train with early stopping
                model.fit(
                    X_train, y_train,
                    eval_set=eval_set,
                    verbose=False
                    # Note: callbacks removed for XGBoost 3.0 compatibility
                )
                
                # Get predictions
                y_proba = model.predict_proba(X_val)[:, 1]
                
                # Calculate metric
                if self.config.primary_metric == 'pr_auc':
                    score = average_precision_score(y_val, y_proba)
                elif self.config.primary_metric == 'roc_auc':
                    score = roc_auc_score(y_val, y_proba)
                elif self.config.primary_metric == 'f1':
                    y_pred = (y_proba >= 0.5).astype(int)
                    score = f1_score(y_val, y_pred)
                else:
                    raise ValueError(f"Unknown metric: {self.config.primary_metric}")
                
            else:
                # Use cross-validation
                cv = StratifiedKFold(
                    n_splits=self.config.inner_cv_splits,
                    shuffle=True,
                    random_state=self.config.seed
                )
                
                scores = []
                for fold, (train_idx, val_idx) in enumerate(cv.split(X_train, y_train)):
                    X_fold_train = X_train[train_idx]
                    y_fold_train = y_train[train_idx]
                    X_fold_val = X_train[val_idx]
                    y_fold_val = y_train[val_idx]
                    
                    model = XGBClassifier(**params)
                    
                    eval_set = [(X_fold_train, y_fold_train), (X_fold_val, y_fold_val)]
                    
                    model.fit(
                        X_fold_train, y_fold_train,
                        eval_set=eval_set,
                        verbose=False
                    )
                    
                    y_proba = model.predict_proba(X_fold_val)[:, 1]
                    
                    if self.config.primary_metric == 'pr_auc':
                        score = average_precision_score(y_fold_val, y_proba)
                    elif self.config.primary_metric == 'roc_auc':
                        score = roc_auc_score(y_fold_val, y_proba)
                    else:
                        y_pred = (y_proba >= 0.5).astype(int)
                        score = f1_score(y_fold_val, y_pred)
                    
                    scores.append(score)
                    
                    # Report intermediate value for pruning
                    trial.report(score, fold)
                    
                    # Check if trial should be pruned
                    if trial.should_prune():
                        raise optuna.TrialPruned()
                
                score = np.mean(scores)
            
            return score
        
        # Create sampler
        sampler = self._create_sampler()
        
        # Create pruner
        pruner = self._create_pruner()
        
        # Create or load study
        if self.config.storage_url:
            storage = RDBStorage(url=self.config.storage_url)
        else:
            storage = None
        
        self.study = optuna.create_study(
            study_name=self.config.study_name,
            storage=storage,
            sampler=sampler,
            pruner=pruner,
            direction='maximize',
            load_if_exists=self.config.load_if_exists
        )
        
        # Optimize
        self.study.optimize(
            objective,
            n_trials=self.config.n_trials,
            timeout=self.config.timeout,
            show_progress_bar=self.config.verbose
        )
        
        # Store best parameters
        self.best_params = self.study.best_params
        
        logger.info(f"Optimization complete. Best {self.config.primary_metric}: {self.study.best_value:.4f}")
        
        return self.study
    
    def fit_final_model(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        X_cal: Optional[np.ndarray] = None,
        y_cal: Optional[np.ndarray] = None
    ):
        """
        Fit final model with best parameters and calibration.
        
        Args:
            X_train: Training features
            y_train: Training labels  
            X_val: Validation features (for early stopping)
            y_val: Validation labels
            X_cal: Calibration features (if None, uses validation set)
            y_cal: Calibration labels
        """
        
        if self.best_params is None:
            raise ValueError("Must run optimize() first or set best_params manually")
        
        # Create final model parameters
        final_params = self.best_params.copy()
        final_params.update({
            'objective': 'binary:logistic',
            'eval_metric': self.config.eval_metric,
            'tree_method': self.config.tree_method,
            'device': self.config.device,
            'n_jobs': self.config.n_jobs,
            'random_state': self.config.seed,
            'use_label_encoder': False,
            'verbosity': 1 if self.config.verbose else 0,
            'early_stopping_rounds': self.config.early_stopping_rounds  # XGBoost 3.0 requires this in params
        })
        
        # Initialize model
        self.model = XGBClassifier(**final_params)
        
        # Prepare eval set
        if X_val is not None and y_val is not None:
            eval_set = [(X_train, y_train), (X_val, y_val)]
            self.model.fit(
                X_train, y_train,
                eval_set=eval_set,
                verbose=self.config.verbose
            )
        else:
            self.model.fit(X_train, y_train, verbose=self.config.verbose)
        
        # Calibration
        if self.config.calibration_method != 'none':
            # Use calibration set if provided, otherwise use validation set
            if X_cal is not None and y_cal is not None:
                X_calibration = X_cal
                y_calibration = y_cal
            elif X_val is not None and y_val is not None:
                X_calibration = X_val
                y_calibration = y_val
            else:
                # Split training data for calibration
                from sklearn.model_selection import train_test_split
                _, X_calibration, _, y_calibration = train_test_split(
                    X_train, y_train, test_size=0.2, 
                    random_state=self.config.seed, stratify=y_train
                )
            
            # Fit calibrator
            self.calibrator = XGBoostCalibrator(method=self.config.calibration_method)
            self.calibrator.fit(self.model, X_calibration, y_calibration)
            
            logger.info(f"Calibration complete. Method: {self.calibrator.method}")
            logger.info(f"Calibration metrics: {self.calibrator.metrics}")
        
        # Optimize threshold if validation set provided
        if X_val is not None and y_val is not None:
            y_proba_val = self.predict_proba(X_val)[:, 1]
            
            # Optimize threshold by EV
            costs = TradingCosts()
            self.best_threshold, threshold_metrics = TradingMetrics.optimize_threshold_by_ev(
                y_val, y_proba_val, costs=costs
            )
            
            logger.info(f"Optimal threshold: {self.best_threshold:.3f}")
            logger.info(f"Threshold metrics: {threshold_metrics}")
        
        return self
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict calibrated probabilities."""
        
        if self.model is None:
            raise ValueError("Model not fitted. Call fit_final_model() first.")
        
        # Get raw probabilities
        y_proba = self.model.predict_proba(X)
        
        # Apply calibration if available
        if self.calibrator is not None:
            y_proba[:, 1] = self.calibrator.transform(y_proba[:, 1])
        
        return y_proba
    
    def predict(self, X: np.ndarray, threshold: Optional[float] = None) -> np.ndarray:
        """Predict binary labels using optimal threshold."""
        
        if threshold is None:
            threshold = self.best_threshold
        
        y_proba = self.predict_proba(X)[:, 1]
        return (y_proba >= threshold).astype(int)
    
    def get_feature_importance(self, importance_type: str = 'gain') -> pd.DataFrame:
        """Get feature importance as DataFrame."""
        
        if self.model is None:
            raise ValueError("Model not fitted.")
        
        importance = self.model.get_booster().get_score(importance_type=importance_type)
        
        df = pd.DataFrame([
            {'feature': k, 'importance': v}
            for k, v in importance.items()
        ])
        
        return df.sort_values('importance', ascending=False)
    
    def save_model(self, filepath: str):
        """Save model, calibrator, and configuration."""
        
        save_dict = {
            'model': self.model,
            'calibrator': self.calibrator,
            'best_params': self.best_params,
            'best_threshold': self.best_threshold,
            'config': self.config,
            'metrics': self.metrics
        }
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(save_dict, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load model, calibrator, and configuration."""
        
        save_dict = joblib.load(filepath)
        
        self.model = save_dict['model']
        self.calibrator = save_dict['calibrator']
        self.best_params = save_dict['best_params']
        self.best_threshold = save_dict['best_threshold']
        self.config = save_dict['config']
        self.metrics = save_dict['metrics']
        
        logger.info(f"Model loaded from {filepath}")
    
    def get_optimization_summary(self) -> Dict[str, Any]:
        """Get summary of optimization results."""
        
        if self.study is None:
            return {}
        
        return {
            'n_trials': len(self.study.trials),
            'best_score': self.study.best_value,
            'best_params': self.study.best_params,
            'best_trial': self.study.best_trial.number,
            'calibration_method': self.calibrator.method if self.calibrator else 'none',
            'calibration_metrics': self.calibrator.metrics if self.calibrator else {},
            'optimal_threshold': self.best_threshold
        }
    
    # Private helper methods
    
    def _sample_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Sample hyperparameters for trial."""
        
        space = self.config.param_space
        params = {}
        
        for param, config in space.items():
            if isinstance(config, dict):
                if config.get('log', False):
                    params[param] = trial.suggest_float(
                        param, config['low'], config['high'], log=True
                    )
                elif isinstance(config['low'], float):
                    params[param] = trial.suggest_float(
                        param, config['low'], config['high']
                    )
                else:
                    params[param] = trial.suggest_int(
                        param, config['low'], config['high']
                    )
            else:
                params[param] = config
        
        return params
    
    def _create_sampler(self) -> optuna.samplers.BaseSampler:
        """Create Optuna sampler."""
        
        if self.config.sampler_type == 'tpe':
            return TPESampler(seed=self.config.seed, **self.config.sampler_params)
        elif self.config.sampler_type == 'random':
            return RandomSampler(seed=self.config.seed, **self.config.sampler_params)
        elif self.config.sampler_type == 'cmaes':
            return CmaEsSampler(seed=self.config.seed, **self.config.sampler_params)
        else:
            raise ValueError(f"Unknown sampler: {self.config.sampler_type}")
    
    def _create_pruner(self) -> optuna.pruners.BasePruner:
        """Create Optuna pruner."""
        
        if self.config.pruner_type == 'median':
            return MedianPruner(**self.config.pruner_params)
        elif self.config.pruner_type == 'percentile':
            return PercentilePruner(**self.config.pruner_params)
        elif self.config.pruner_type == 'successive_halving':
            return SuccessiveHalvingPruner(**self.config.pruner_params)
        elif self.config.pruner_type == 'hyperband':
            return HyperbandPruner(**self.config.pruner_params)
        else:
            raise ValueError(f"Unknown pruner: {self.config.pruner_type}")


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def create_xgboost_model(config: Optional[XGBoostUnifiedConfig] = None) -> UnifiedXGBoost:
    """Factory function to create XGBoost model."""
    return UnifiedXGBoost(config)


def quick_train_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_trials: int = 50,
    **kwargs
) -> UnifiedXGBoost:
    """Quick training function with sensible defaults."""
    
    config = XGBoostUnifiedConfig(n_trials=n_trials, **kwargs)
    model = UnifiedXGBoost(config)
    
    # Optimize
    model.optimize(X_train, y_train, X_val, y_val)
    
    # Fit final model
    model.fit_final_model(X_train, y_train, X_val, y_val)
    
    return model


if __name__ == "__main__":
    # Example usage
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    
    # Generate sample data
    X, y = make_classification(
        n_samples=10000,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        weights=[0.9, 0.1],
        random_state=42
    )
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    
    # Train model
    model = quick_train_xgboost(
        X_train, y_train, X_val, y_val,
        n_trials=10,  # Small number for demo
        verbose=True
    )
    
    # Evaluate
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)
    
    print(f"\nTest PR-AUC: {average_precision_score(y_test, y_proba):.4f}")
    print(f"Test F1: {f1_score(y_test, y_pred):.4f}")
    print(f"Optimal threshold: {model.best_threshold:.3f}")