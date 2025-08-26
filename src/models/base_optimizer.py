"""
Base optimizer class for all model optimizers.
Provides common functionality for Bayesian optimization across different model types.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, Any, List, Callable, Union
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import warnings
import mlflow
from pathlib import Path
import optuna
from sklearn.metrics import (
    f1_score, precision_recall_curve, average_precision_score,
    roc_auc_score, matthews_corrcoef, accuracy_score
)

from ..utils.logging import log as logger
from .threshold_optimizer import ThresholdOptimizer, TradingCosts
from .metrics.calibration import comprehensive_calibration_metrics


@dataclass
class BaseOptimizerConfig:
    """Base configuration for all optimizers."""
    
    # Basic settings
    n_trials: int = 100
    timeout: Optional[int] = None
    seed: int = 42
    
    # Sampler settings
    sampler_type: str = 'tpe'  # 'tpe', 'random', 'cmaes'
    sampler_params: Dict[str, Any] = field(default_factory=dict)
    
    # Pruner settings  
    pruner_type: str = 'asha'  # 'asha', 'hyperband', 'median', 'percentile'
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
    calibration_method: str = 'auto'
    calibration_selection_metric: str = 'brier'
    
    # MLflow integration
    use_mlflow: bool = True
    mlflow_experiment: str = 'model_optimization'
    
    # Metrics
    primary_metric: str = 'f1_score'
    
    # Early stopping
    early_stopping_rounds: int = 100
    
    # Determinism
    deterministic: bool = True
    
    # Output
    verbose: bool = False


class BaseModelOptimizer(ABC):
    """
    Abstract base class for all model optimizers.
    Provides common functionality for optimization, calibration, and threshold tuning.
    """
    
    def __init__(self, config: BaseOptimizerConfig):
        """Initialize base optimizer."""
        self.config = config
        
        # Results storage
        self.best_model = None
        self.best_params = None
        self.best_score = -np.inf
        self.calibrator = None
        self.threshold_f1 = 0.5
        self.threshold_ev = 0.5
        self.threshold_optimizer = None
        self.feature_names_ = None
        self.study = None
        
        # Results tracking
        self.results = {
            'outer_scores': [],
            'calibration_metrics': [],
            'feature_importance': {},
            'trial_history': []
        }
        
        logger.info(
            f"{self.__class__.__name__}_initialized",
            n_trials=self.config.n_trials,
            pruner_type=self.config.pruner_type,
            calibration_method=self.config.calibration_method
        )
    
    @abstractmethod
    def _create_model_factory(self) -> Callable:
        """
        Create model factory function for the optimizer.
        Must be implemented by subclasses.
        """
        pass
    
    @abstractmethod
    def _create_expanded_search_space(self, trial: optuna.Trial) -> Dict[str, Any]:
        """
        Create hyperparameter search space.
        Must be implemented by subclasses.
        """
        pass
    
    @abstractmethod
    def _fit_calibrator(self, X_val: pd.DataFrame, y_val: pd.Series):
        """
        Fit calibration model specific to the model type.
        Must be implemented by subclasses.
        """
        pass
    
    def _create_sampler(self) -> optuna.samplers.BaseSampler:
        """Create Optuna sampler based on configuration."""
        from optuna.samplers import TPESampler, RandomSampler, CmaEsSampler
        
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
        from optuna.pruners import (
            MedianPruner, PercentilePruner, 
            SuccessiveHalvingPruner, HyperbandPruner
        )
        
        params = self.config.pruner_params.copy()
        
        if self.config.pruner_type in ['asha', 'hyperband']:
            params.setdefault('min_resource', 50)
            params.setdefault('max_resource', 1500)
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
        from optuna.storages import RDBStorage
        
        sampler = self._create_sampler()
        pruner = self._create_pruner()
        
        # Storage configuration
        storage = None
        if self.config.storage_url:
            storage = RDBStorage(url=self.config.storage_url)
        
        # Study name
        study_name = self.config.study_name or f"{self.__class__.__name__}_{self.config.seed}"
        
        # Create or load study
        try:
            study = optuna.create_study(
                study_name=study_name,
                direction='maximize',
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
    
    def _optimize_thresholds(self, y_true: np.ndarray, y_pred_proba: np.ndarray):
        """Optimize classification thresholds using F1 and Expected Value."""
        # Optimize F1 threshold
        precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
        best_idx = np.argmax(f1_scores[:-1])
        self.threshold_f1 = thresholds[best_idx] if len(thresholds) > 0 else 0.5
        
        # Optimize EV threshold using ThresholdOptimizer
        costs = TradingCosts(
            fee_bps=5.0,      # 0.05% exchange fee
            slippage_bps=5.0,  # 0.05% typical slippage
            impact_bps=2.0     # 0.02% market impact
        )
        
        self.threshold_optimizer = ThresholdOptimizer(costs=costs)
        
        # Optimize threshold for maximum expected value
        ev_results = self.threshold_optimizer.optimize_threshold(
            y_true=y_true,
            y_proba=y_pred_proba,
            avg_win_pct=0.015,  # 1.5% average win
            avg_loss_pct=0.005,  # 0.5% average loss
            method='adaptive',
            n_points=100
        )
        
        self.threshold_ev = ev_results.optimal_threshold
        
        logger.info("thresholds_optimized",
                   threshold_f1=self.threshold_f1,
                   threshold_ev=self.threshold_ev)
    
    def _create_objective_function(self, X: pd.DataFrame, y: pd.Series) -> Callable:
        """Create Optuna objective function with cross-validation."""
        
        def objective(trial: optuna.Trial) -> float:
            # Get hyperparameters
            params = self._create_expanded_search_space(trial)
            
            # Log optimization target on first trial
            if trial.number == 0:
                logger.info(
                    "OPTIMIZATION TARGET",
                    primary_metric=self.config.primary_metric,
                    direction="MAXIMIZE"
                )
            
            # Cross-validation with embargo
            from ..data.splits import PurgedKFold
            cv = PurgedKFold(
                n_splits=self.config.inner_cv_splits,
                embargo=self.config.embargo
            )
            
            scores = []
            model_factory = self._create_model_factory()
            
            for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                
                # Create and train model
                model = model_factory(params)
                model.fit(X_train, y_train)
                
                # Get predictions
                y_pred_proba = model.predict_proba(X_val)[:, 1]
                
                # Calculate score based on primary metric
                if self.config.primary_metric == 'pr_auc':
                    score = average_precision_score(y_val, y_pred_proba)
                elif self.config.primary_metric == 'roc_auc':
                    score = roc_auc_score(y_val, y_pred_proba)
                elif self.config.primary_metric == 'f1_score':
                    y_pred = (y_pred_proba >= 0.5).astype(int)
                    score = f1_score(y_val, y_pred, zero_division=0)
                elif self.config.primary_metric == 'mcc':
                    y_pred = (y_pred_proba >= 0.5).astype(int)
                    score = matthews_corrcoef(y_val, y_pred)
                else:
                    score = accuracy_score(y_val, (y_pred_proba >= 0.5).astype(int))
                
                scores.append(score)
                
                # Clean up
                del model, X_train, X_val, y_train, y_val
                import gc
                gc.collect()
                
                # Report intermediate value for pruning
                trial.report(score, fold)
                
                if trial.should_prune():
                    raise optuna.TrialPruned()
            
            # Calculate mean score
            mean_score = np.mean(scores)
            
            # Log trial results
            logger.info(
                f"Trial {trial.number}",
                metric=self.config.primary_metric,
                score=f"{mean_score:.4f}",
                fold_scores=[f"{s:.4f}" for s in scores]
            )
            
            # Store in trial attributes
            trial.set_user_attr('fold_scores', scores)
            trial.set_user_attr('cv_std', np.std(scores))
            
            # Final cleanup
            import gc
            gc.collect()
            
            return mean_score
        
        return objective
    
    def optimize(self, X: pd.DataFrame, y: pd.Series) -> optuna.Study:
        """
        Run optimization with all advanced features.
        
        Args:
            X: Training features
            y: Training labels
            
        Returns:
            Completed Optuna study
        """
        logger.info(f"starting_{self.__class__.__name__}_optimization",
                   n_trials=self.config.n_trials,
                   pruner_type=self.config.pruner_type)
        
        # Store feature names
        if hasattr(X, 'columns'):
            self.feature_names_ = X.columns.tolist()
        
        # Create study
        self.study = self._create_study()
        
        # MLflow integration
        if self.config.use_mlflow:
            self._setup_mlflow()
        
        # Create objective function
        objective = self._create_objective_function(X, y)
        
        # Run optimization
        try:
            self.study.optimize(
                objective,
                n_trials=self.config.n_trials,
                timeout=self.config.timeout,
                show_progress_bar=self.config.verbose,
                n_jobs=1  # Always 1 for determinism
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
        
        logger.info(f"fitting_final_{self.__class__.__name__}_model")
        
        # Train/validation split for final model
        from sklearn.model_selection import train_test_split
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, 
            random_state=self.config.seed, stratify=y
        )
        
        # Create and train final model
        model_factory = self._create_model_factory()
        self.best_model = model_factory(self.best_params)
        self.best_model.fit(X_train, y_train)
        
        # Calibrate model
        logger.info("calibrating_model", method=self.config.calibration_method)
        self._fit_calibrator(X_val, y_val)
        
        # Get calibrated predictions
        calibrated_probs = self.predict_proba(X_val)[:, 1]
        
        # Optimize thresholds
        self._optimize_thresholds(y_val.values, calibrated_probs)
        
        # Calculate calibration metrics
        cal_metrics = comprehensive_calibration_metrics(y_val.values, calibrated_probs)
        
        # Log to MLflow
        if self.config.use_mlflow:
            self._log_mlflow_metrics(cal_metrics)
        
        logger.info("model_training_completed",
                   ece=cal_metrics.get('ece_uniform', 0),
                   brier_score=cal_metrics.get('brier_score', 0))
    
    def _setup_mlflow(self):
        """Setup MLflow tracking."""
        try:
            mlflow.set_experiment(self.config.mlflow_experiment)
            mlflow.start_run()
            
            # Log configuration
            mlflow.log_params({
                'model_type': self.__class__.__name__,
                'sampler_type': self.config.sampler_type,
                'pruner_type': self.config.pruner_type,
                'n_trials': self.config.n_trials,
                'primary_metric': self.config.primary_metric,
                'calibration_method': self.config.calibration_method,
                'seed': self.config.seed
            })
        except Exception as e:
            warnings.warn(f"MLflow setup failed: {e}")
    
    def _log_mlflow_metrics(self, cal_metrics: Dict):
        """Log metrics to MLflow."""
        try:
            mlflow.log_params(self.best_params)
            mlflow.log_metric('best_score', self.best_score)
            mlflow.log_metrics(cal_metrics)
            mlflow.log_metric('threshold_f1', self.threshold_f1)
            mlflow.log_metric('threshold_ev', self.threshold_ev)
        except Exception as e:
            warnings.warn(f"MLflow logging failed: {e}")
        finally:
            try:
                mlflow.end_run()
            except:
                pass
    
    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict calibrated probabilities."""
        pass
    
    def predict(self, X: pd.DataFrame, use_ev_threshold: bool = False) -> np.ndarray:
        """Predict classes using optimized thresholds."""
        proba = self.predict_proba(X)[:, 1]
        threshold = self.threshold_ev if use_ev_threshold else self.threshold_f1
        return (proba >= threshold).astype(int)
    
    def get_optimization_summary(self) -> Dict:
        """Get summary of optimization results."""
        if self.study is None:
            return {}
        
        completed_trials = [
            t for t in self.study.trials 
            if t.state == optuna.trial.TrialState.COMPLETE
        ]
        
        return {
            'n_trials': len(self.study.trials),
            'n_completed': len(completed_trials),
            'n_pruned': len([
                t for t in self.study.trials 
                if t.state == optuna.trial.TrialState.PRUNED
            ]),
            'best_score': self.study.best_value,
            'best_params': self.study.best_params,
            'pruner_type': self.config.pruner_type,
            'sampler_type': self.config.sampler_type,
            'calibration_method': self.config.calibration_method
        }
    
    # Sklearn compatibility methods
    def get_params(self, deep: bool = True) -> Dict:
        """Get parameters for sklearn compatibility."""
        return {'config': self.config}
    
    def set_params(self, **params) -> 'BaseModelOptimizer':
        """Set parameters for sklearn compatibility."""
        for key, value in params.items():
            if key == 'config':
                self.config = value
            elif hasattr(self.config, key):
                setattr(self.config, key, value)
        return self
    
    def score(self, X: pd.DataFrame, y: np.ndarray) -> float:
        """Calculate score based on primary metric."""
        y_pred_proba = self.predict_proba(X)[:, 1]
        
        if self.config.primary_metric == 'pr_auc':
            return average_precision_score(y, y_pred_proba)
        elif self.config.primary_metric == 'roc_auc':
            return roc_auc_score(y, y_pred_proba)
        elif self.config.primary_metric == 'f1_score':
            y_pred = self.predict(X)
            return f1_score(y, y_pred, zero_division=0)
        else:
            y_pred = self.predict(X)
            return accuracy_score(y, y_pred)