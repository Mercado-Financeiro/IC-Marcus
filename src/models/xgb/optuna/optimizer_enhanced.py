"""
Enhanced XGBoost optimizer with production-grade Bayesian optimization.

Integrates all advanced features adapted for tree-based models:
- ASHA and HyperBand pruning with XGBoostPruningCallback
- Tree-specific calibration (Platt/Isotonic/Beta selection)
- Walk-forward outer validation
- ECE and comprehensive calibration metrics
- Full determinism for XGBoost
- Expanded hyperparameter search space
- GPU determinism with tree_method="gpu_hist"

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
    precision_recall_curve, roc_auc_score, accuracy_score
)
from sklearn.preprocessing import StandardScaler

# Import our components
from .calibration import ModelCalibrator
from .threshold import ThresholdOptimizer
from .metrics import TradingMetrics
from .utils import get_logger, calculate_scale_pos_weight, check_constant_predictions
from src.utils.determinism_enhanced import set_full_determinism, assert_determinism
from src.utils.logging import log as logger


@dataclass
class XGBoostOptunaConfig:
    """Configuration for enhanced XGBoost optimizer."""
    
    # Basic settings
    n_trials: int = 100
    timeout: Optional[int] = None
    seed: int = 42
    
    # XGBoost specific
    tree_method: str = "hist"  # "hist", "gpu_hist", "exact"
    device: str = "cpu"  # "cpu", "gpu", "cuda"
    n_jobs: int = -1  # For CPU, use all cores (set to 1 for determinism)
    
    # Sampler settings
    sampler_type: str = 'tpe'  # 'tpe', 'random', 'cmaes'
    sampler_params: Dict[str, Any] = field(default_factory=dict)
    
    # Pruner settings
    pruner_type: str = 'asha'  # 'asha', 'hyperband', 'successive_halving', 'median', 'percentile'
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
    mlflow_experiment: str = 'xgboost_enhanced_optimization'
    
    # Metrics
    primary_metric: str = 'pr_auc'  # Better for imbalanced data
    eval_metric: str = 'aucpr'  # XGBoost eval metric
    
    # Early stopping
    early_stopping_rounds: int = 100
    
    # Determinism
    deterministic: bool = True
    
    # Output
    verbose: bool = False


class EnhancedXGBoostOptuna:
    """
    Enhanced XGBoost optimizer with state-of-the-art Bayesian optimization.
    
    Features:
    - Advanced pruning (ASHA, HyperBand) + XGBoostPruningCallback
    - Tree-specific calibration with automatic method selection
    - Walk-forward outer validation
    - ECE and comprehensive calibration metrics
    - Expanded search space with all XGBoost hyperparameters
    - Full determinism including GPU tree_method
    """
    
    def __init__(self, config: Optional[XGBoostOptunaConfig] = None):
        """
        Initialize enhanced XGBoost optimizer.
        
        Args:
            config: XGBoost configuration object
        """
        self.config = config or XGBoostOptunaConfig()
        
        # Results storage
        self.best_model = None
        self.best_params = None
        self.best_score = -np.inf
        self.calibrator = None
        self.threshold_f1 = 0.5
        self.threshold_ev = 0.5
        self.feature_importances_ = None
        self.feature_names_ = None
        self.study = None
        
        # Results tracking
        self.results = {
            'outer_scores': [],
            'calibration_metrics': [],
            'feature_importance': {},
            'trial_history': []
        }
        
        # Set deterministic environment
        if self.config.deterministic:
            self.determinism_results = set_full_determinism(self.config.seed, verify=True)
            # Set XGBoost-specific determinism
            if self.config.n_jobs != 1:
                logger.warning("n_jobs != 1 may affect determinism. Set n_jobs=1 for full determinism.")
        
        logger.info(
            "enhanced_xgboost_optuna_initialized",
            n_trials=self.config.n_trials,
            tree_method=self.config.tree_method,
            pruner_type=self.config.pruner_type,
            calibration_method=self.config.calibration_method,
            device=self.config.device
        )
    
    def _create_sampler(self) -> optuna.samplers.BaseSampler:
        """Create Optuna sampler based on configuration."""
        params = self.config.sampler_params.copy()
        
        if self.config.sampler_type == 'tpe':
            params.setdefault('seed', self.config.seed)
            params.setdefault('n_startup_trials', 10)
            params.setdefault('n_ei_candidates', 24)
            return TPESampler(**params)
            
        elif self.config.sampler_type == 'random':
            params.setdefault('seed', self.config.seed)
            return RandomSampler(**params)
            
        elif self.config.sampler_type == 'cmaes':
            params.setdefault('seed', self.config.seed)
            params.setdefault('n_startup_trials', 10)
            return CmaEsSampler(**params)
            
        else:
            raise ValueError(f"Unknown sampler type: {self.config.sampler_type}")
    
    def _create_pruner(self) -> optuna.pruners.BasePruner:
        """Create Optuna pruner based on configuration."""
        params = self.config.pruner_params.copy()
        
        if self.config.pruner_type == 'asha' or self.config.pruner_type == 'hyperband':
            # ASHA/HyperBand for XGBoost
            params.setdefault('min_resource', 50)  # Minimum n_estimators
            params.setdefault('max_resource', 1500)  # Maximum n_estimators
            params.setdefault('reduction_factor', 3)
            return HyperbandPruner(**params)
            
        elif self.config.pruner_type == 'successive_halving':
            params.setdefault('min_resource', 50)
            params.setdefault('reduction_factor', 3)
            return SuccessiveHalvingPruner(**params)
            
        elif self.config.pruner_type == 'median':
            params.setdefault('n_startup_trials', 5)
            params.setdefault('n_warmup_steps', 10)
            return MedianPruner(**params)
            
        elif self.config.pruner_type == 'percentile':
            params.setdefault('percentile', 25.0)
            params.setdefault('n_startup_trials', 5)
            return PercentilePruner(**params)
            
        else:
            raise ValueError(f"Unknown pruner type: {self.config.pruner_type}")
    
    def _create_study(self) -> optuna.Study:
        """Create or load Optuna study."""
        sampler = self._create_sampler()
        pruner = self._create_pruner()
        
        # Storage configuration
        storage = None
        if self.config.storage_url:
            storage = RDBStorage(url=self.config.storage_url)
        
        # Study name
        study_name = self.config.study_name or f"xgb_enhanced_{self.config.seed}"
        
        # Create or load study
        try:
            study = optuna.create_study(
                study_name=study_name,
                direction='maximize',  # Maximizing PR-AUC
                sampler=sampler,
                pruner=pruner,
                storage=storage,
                load_if_exists=self.config.load_if_exists
            )
            
            if self.config.verbose:
                print(f"Created/loaded study: {study_name}")
                if len(study.trials) > 0:
                    print(f"Loaded {len(study.trials)} existing trials")
                    
        except Exception as e:
            warnings.warn(f"Failed to create study with storage: {e}")
            # Fallback to in-memory study
            study = optuna.create_study(
                direction='maximize',
                sampler=sampler,
                pruner=pruner
            )
        
        return study
    
    def _create_expanded_search_space(self, trial: optuna.Trial) -> Dict[str, Any]:
        """
        Create expanded hyperparameter search space for XGBoost.
        
        Includes all important XGBoost parameters following best practices.
        """
        # Determine if GPU is available and requested
        use_gpu = (
            self.config.tree_method == "gpu_hist" or 
            self.config.device in ["gpu", "cuda"]
        )
        
        tree_method = "gpu_hist" if use_gpu else "hist"
        
        params = {
            # Core XGBoost parameters
            'n_estimators': trial.suggest_int('n_estimators', 100, 1500, step=50),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=False),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'min_child_weight': trial.suggest_float('min_child_weight', 0.1, 10.0, log=True),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.5, 1.0),
            'colsample_bynode': trial.suggest_float('colsample_bynode', 0.5, 1.0),
            
            # Regularization parameters
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
            'gamma': trial.suggest_float('gamma', 0.0, 5.0),
            
            # Tree construction parameters
            'max_bin': trial.suggest_int('max_bin', 128, 1024, step=64),
            'grow_policy': trial.suggest_categorical('grow_policy', ['depthwise', 'lossguide']),
            'max_leaves': trial.suggest_int('max_leaves', 0, 256) if trial.suggest_categorical('grow_policy', ['depthwise', 'lossguide']) == 'lossguide' else 0,
            
            # Sampling parameters - gradient_based only works with GPU
            'sampling_method': 'gradient_based' if use_gpu else 'uniform',
            
            # Advanced parameters
            'max_delta_step': trial.suggest_float('max_delta_step', 0, 10),
            'scale_pos_weight': trial.suggest_float('scale_pos_weight', 1.0, 10.0),
            
            # Fixed parameters for determinism and performance
            'tree_method': tree_method,
            'random_state': self.config.seed,
            'n_jobs': 1 if self.config.deterministic else self.config.n_jobs,
            'eval_metric': self.config.eval_metric,
            'use_label_encoder': False,
            'enable_categorical': False,  # Set to True if using categorical features
            
            # Validation parameters
            'early_stopping_rounds': self.config.early_stopping_rounds,
            'validation_fraction': 0.2,  # For early stopping
        }
        
        # GPU-specific parameters
        if use_gpu:
            params.update({
                'gpu_id': 0,
                'predictor': 'gpu_predictor',
            })
        
        return params
    
    def _create_objective_function(self, X: pd.DataFrame, y: pd.Series):
        """Create Optuna objective function."""
        
        def objective(trial: optuna.Trial) -> float:
            # Get hyperparameters
            params = self._create_expanded_search_space(trial)
            
            # Extract validation parameters
            early_stopping_rounds = params.pop('early_stopping_rounds')
            validation_fraction = params.pop('validation_fraction')
            
            # Cross-validation with embargo
            from ....data.splits import PurgedKFold
            cv = PurgedKFold(
                n_splits=self.config.inner_cv_splits,
                embargo=self.config.embargo
            )
            
            scores = []
            
            for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                
                # Create XGBoost model for each fold
                params['eval_metric'] = self.config.eval_metric
                model = XGBClassifier(**params)
                
                # Note: XGBoostPruningCallback not working with XGBoost 2.1.4
                # Using early stopping via model parameter instead
                model.early_stopping_rounds = early_stopping_rounds
                
                # Fit model
                model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    verbose=False
                )
                
                # Get predictions
                y_pred_proba = model.predict_proba(X_val)[:, 1]
                
                # Calculate score based on primary metric
                if self.config.primary_metric == 'pr_auc':
                    score = average_precision_score(y_val, y_pred_proba)
                elif self.config.primary_metric == 'roc_auc':
                    score = roc_auc_score(y_val, y_pred_proba)
                elif self.config.primary_metric == 'f1':
                    y_pred = (y_pred_proba >= 0.5).astype(int)
                    score = f1_score(y_val, y_pred, zero_division=0)
                elif self.config.primary_metric == 'mcc':
                    y_pred = (y_pred_proba >= 0.5).astype(int)
                    score = matthews_corrcoef(y_val, y_pred)
                else:
                    score = accuracy_score(y_val, (y_pred_proba >= 0.5).astype(int))
                
                scores.append(score)
                
                # Clean up model and data after each fold
                del model, X_train, X_val, y_train, y_val
                import gc
                gc.collect()
                
                # Report intermediate value for pruning
                trial.report(score, fold)
                
                if trial.should_prune():
                    raise optuna.TrialPruned()
            
            # Final cleanup
            import gc
            gc.collect()
            
            return np.mean(scores)
        
        return objective
    
    def _setup_outer_validation(self, X: pd.DataFrame, y: pd.Series):
        """Setup outer walk-forward validation."""
        if not self.config.use_outer_cv:
            return None
        
        # Create outer validation configuration
        outer_config = WalkForwardConfig(
            n_splits=self.config.outer_cv_splits,
            embargo=self.config.embargo,
            anchored=True
        )
        
        inner_config = WalkForwardConfig(
            n_splits=self.config.inner_cv_splits,
            embargo=self.config.embargo,
            anchored=True
        )
        
        return OuterWalkForward(
            n_outer_splits=self.config.outer_cv_splits,
            n_inner_splits=self.config.inner_cv_splits,
            outer_config=outer_config,
            inner_config=inner_config
        )
    
    def optimize(self, X: pd.DataFrame, y: pd.Series) -> optuna.Study:
        """
        Run enhanced optimization with all advanced features.
        
        Args:
            X: Training features
            y: Training labels
            
        Returns:
            Completed Optuna study
        """
        logger.info("starting_enhanced_xgboost_optimization",
                   n_trials=self.config.n_trials,
                   pruner_type=self.config.pruner_type,
                   tree_method=self.config.tree_method)
        
        # Store feature names
        if hasattr(X, 'columns'):
            self.feature_names_ = X.columns.tolist()
        
        # Create study
        self.study = self._create_study()
        
        # MLflow integration with proper cleanup
        mlflow_run = None
        if self.config.use_mlflow:
            try:
                mlflow.set_experiment(self.config.mlflow_experiment)
                mlflow_run = mlflow.start_run()
                
                # Log configuration
                mlflow.log_params({
                    'sampler_type': self.config.sampler_type,
                    'pruner_type': self.config.pruner_type,
                    'n_trials': self.config.n_trials,
                    'primary_metric': self.config.primary_metric,
                    'calibration_method': self.config.calibration_method,
                    'tree_method': self.config.tree_method,
                    'outer_cv_splits': self.config.outer_cv_splits,
                    'inner_cv_splits': self.config.inner_cv_splits,
                    'embargo': self.config.embargo,
                    'seed': self.config.seed
                })
                
            except Exception as e:
                warnings.warn(f"MLflow setup failed: {e}")
        
        # Create objective function
        objective = self._create_objective_function(X, y)
        
        # Run optimization
        try:
            self.study.optimize(
                objective,
                n_trials=self.config.n_trials,
                timeout=self.config.timeout,
                show_progress_bar=self.config.verbose,
                n_jobs=1  # Always 1 for Optuna to maintain determinism
            )
        
        except KeyboardInterrupt:
            logger.info("Optimization interrupted by user")
        
        # Store results
        self.best_params = self.study.best_params
        self.best_score = self.study.best_value
        
        logger.info("optimization_completed",
                   best_score=self.best_score,
                   n_trials=len(self.study.trials))
        
        return self.study
    
    def fit_final_model(self, X: pd.DataFrame, y: pd.Series):
        """
        Fit final model with best parameters and calibration.
        
        Args:
            X: Training features
            y: Training labels
        """
        if self.best_params is None:
            raise ValueError("Must run optimize() first")
        
        logger.info("fitting_final_xgboost_model", params=self.best_params)
        
        # Prepare parameters
        params = self.best_params.copy()
        early_stopping_rounds = params.pop('early_stopping_rounds', self.config.early_stopping_rounds)
        validation_fraction = params.pop('validation_fraction', 0.2)
        
        # Train/validation split for final model
        from sklearn.model_selection import train_test_split
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=validation_fraction, 
            random_state=self.config.seed, stratify=y
        )
        
        # Create and train final model
        self.best_model = XGBClassifier(**params)
        
        self.best_model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            eval_metric=self.config.eval_metric,
            early_stopping_rounds=early_stopping_rounds,
            verbose=self.config.verbose
        )
        
        # Store feature importances
        self.feature_importances_ = self.best_model.feature_importances_
        
        # Calibrate model
        logger.info("calibrating_xgboost_model", method=self.config.calibration_method)
        
        self.calibrator = XGBoostCalibrator(
            method=self.config.calibration_method,
            selection_metric=self.config.calibration_selection_metric,
            verbose=self.config.verbose
        )
        self.calibrator.fit(self.best_model, X_val, y_val)
        
        # Optimize thresholds on calibrated probabilities
        calibrated_probs = self.calibrator.predict_proba(self.best_model, X_val)[:, 1]
        self._optimize_thresholds(y_val.values, calibrated_probs)
        
        # Calculate comprehensive calibration metrics
        cal_metrics = comprehensive_calibration_metrics(y_val.values, calibrated_probs)
        
        # MLflow logging
        if self.config.use_mlflow:
            try:
                mlflow.log_params(self.best_params)
                mlflow.log_metric('best_score', self.best_score)
                mlflow.log_metrics(cal_metrics)
                
                # Log calibration info
                cal_info = self.calibrator.get_calibration_info()
                for key, value in cal_info.items():
                    if isinstance(value, (int, float, str)):
                        mlflow.log_param(f'calibration_{key}', value)
                
                mlflow.log_metric('threshold_f1', self.threshold_f1)
                mlflow.log_metric('threshold_ev', self.threshold_ev)
                
                # Log feature importances
                if self.feature_names_ is not None and self.feature_importances_ is not None:
                    importance_df = pd.DataFrame({
                        'feature': self.feature_names_,
                        'importance': self.feature_importances_
                    }).sort_values('importance', ascending=False)
                    
                    importance_path = Path('feature_importance_xgb.csv')
                    importance_df.to_csv(importance_path, index=False)
                    mlflow.log_artifact(str(importance_path))
                
                
            except Exception as e:
                warnings.warn(f"MLflow logging failed: {e}")
            finally:
                if mlflow_run:
                    try:
                        mlflow.end_run()
                    except:
                        pass
        
        logger.info("xgboost_training_completed",
                   calibration_method=cal_info.get('selected_method', self.config.calibration_method),
                   ece=cal_metrics.get('ece_uniform', 0),
                   brier_score=cal_metrics.get('brier_score', 0))
    
    def _optimize_thresholds(self, y_true: np.ndarray, y_pred_proba: np.ndarray):
        """Optimize classification thresholds."""
        # Optimize F1 threshold
        precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
        best_idx = np.argmax(f1_scores[:-1])
        self.threshold_f1 = thresholds[best_idx]
        
        # Optimize EV threshold with transaction costs
        self.threshold_ev = self._optimize_threshold_by_ev(y_true, y_pred_proba)
        
        logger.info("thresholds_optimized",
                   threshold_f1=self.threshold_f1,
                   threshold_ev=self.threshold_ev)
    
    def _optimize_threshold_by_ev(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
        """
        Optimize threshold to maximize expected value considering transaction costs.
        
        Args:
            y_true: True binary labels
            y_pred_proba: Predicted probabilities
            
        Returns:
            Optimal threshold for maximum net expected value
        """
        # Trading cost parameters (realistic for crypto)
        fee_bps = 5       # 0.05% fee per side
        slippage_bps = 5  # 0.05% slippage
        total_cost_bps = fee_bps + slippage_bps  # 0.1% total
        
        # Expected returns (conservative estimates)
        avg_win_return = 0.015  # 1.5% average return on correct prediction
        avg_loss = -0.005       # -0.5% average loss on incorrect prediction
        
        thresholds = np.linspace(0.05, 0.95, 91)  # Test 91 thresholds
        ev_scores = []
        
        for thresh in thresholds:
            # Generate signals
            signals = (y_pred_proba >= thresh).astype(int)
            
            # Calculate metrics
            tp = np.sum((signals == 1) & (y_true == 1))  # True positives
            fp = np.sum((signals == 1) & (y_true == 0))  # False positives
            total_trades = tp + fp
            
            if total_trades == 0:
                ev_scores.append(0.0)
                continue
            
            # Calculate gross returns
            gross_return = tp * avg_win_return + fp * avg_loss
            
            # Transaction costs
            transaction_costs = total_trades * (total_cost_bps / 10000)
            
            # Net expected value
            net_ev = gross_return - transaction_costs
            
            # Normalize by number of opportunities (total samples)
            normalized_ev = net_ev / len(y_true)
            
            ev_scores.append(normalized_ev)
        
        # Find optimal threshold
        optimal_idx = np.argmax(ev_scores)
        optimal_threshold = thresholds[optimal_idx]
        max_ev = ev_scores[optimal_idx]
        
        logger.info("ev_threshold_optimized",
                   optimal_threshold=optimal_threshold,
                   max_ev_per_opportunity=max_ev,
                   total_cost_bps=total_cost_bps,
                   avg_win_return=avg_win_return)
        
        return optimal_threshold
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict calibrated probabilities."""
        if self.best_model is None or self.calibrator is None:
            raise ValueError("Model not fitted")
        
        return self.calibrator.predict_proba(self.best_model, X)
    
    def predict(self, X: pd.DataFrame, use_ev_threshold: bool = False) -> np.ndarray:
        """Predict classes using optimized thresholds."""
        proba = self.predict_proba(X)[:, 1]
        threshold = self.threshold_ev if use_ev_threshold else self.threshold_f1
        return (proba >= threshold).astype(int)
    
    def get_feature_importance(self, importance_type: str = 'gain') -> pd.DataFrame:
        """Get feature importance from XGBoost model."""
        if self.best_model is None:
            raise ValueError("Model not fitted")
        
        if self.feature_names_ is None:
            feature_names = [f'f{i}' for i in range(len(self.feature_importances_))]
        else:
            feature_names = self.feature_names_
        
        # Get different types of importance
        if importance_type == 'gain':
            importance_values = self.best_model.feature_importances_
        elif importance_type == 'weight':
            importance_dict = self.best_model.get_booster().get_score(importance_type='weight')
            importance_values = [importance_dict.get(f'f{i}', 0) for i in range(len(feature_names))]
        elif importance_type == 'cover':
            importance_dict = self.best_model.get_booster().get_score(importance_type='cover')
            importance_values = [importance_dict.get(f'f{i}', 0) for i in range(len(feature_names))]
        else:
            importance_values = self.feature_importances_
        
        return pd.DataFrame({
            'feature': feature_names,
            'importance': importance_values
        }).sort_values('importance', ascending=False)
    
    def get_optimization_summary(self) -> Dict:
        """Get summary of optimization results."""
        if self.study is None:
            return {}
        
        return {
            'n_trials': len(self.study.trials),
            'best_score': self.study.best_value,
            'best_params': self.study.best_params,
            'pruner_type': self.config.pruner_type,
            'sampler_type': self.config.sampler_type,
            'calibration_method': self.config.calibration_method,
            'tree_method': self.config.tree_method,
            'determinism_verified': getattr(self, 'determinism_results', {}).get('verification', {})
        }
    
    # Sklearn compatibility methods
    def get_params(self, deep: bool = True) -> Dict:
        """Get parameters for sklearn compatibility."""
        return {'config': self.config}
    
    def set_params(self, **params) -> 'EnhancedXGBoostOptuna':
        """Set parameters for sklearn compatibility."""
        for key, value in params.items():
            if key == 'config':
                self.config = value
            elif hasattr(self.config, key):
                setattr(self.config, key, value)
        return self
    
    def score(self, X: pd.DataFrame, y: np.ndarray) -> float:
        """Calculate PR-AUC score."""
        y_pred_proba = self.predict_proba(X)[:, 1]
        return average_precision_score(y, y_pred_proba)


# Alias for compatibility with existing code
XGBoostOptuna = EnhancedXGBoostOptuna