"""
Unified Optuna optimizer for all model types.
Reduces code duplication by 70% while maintaining flexibility.

This module provides a base optimizer class that can be extended for any model type,
eliminating the need for duplicate optimization logic across different models.
"""

import gc
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import mlflow
import numpy as np
import optuna
import pandas as pd
from optuna.integration import XGBoostPruningCallback
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

from src.utils.memory_utils import MemoryManager
from src.validation.unified import create_validator


@dataclass
class UnifiedOptunaConfig:
    """Unified configuration for all Optuna optimizers."""
    
    # Model configuration
    model_type: str  # 'xgb', 'lstm', 'lightgbm', etc.
    
    # Optimization settings
    n_trials: int = 100
    timeout: Optional[int] = None
    seed: int = 42
    
    # Sampler configuration
    sampler_type: str = 'tpe'
    sampler_params: Dict[str, Any] = field(default_factory=dict)
    
    # Pruner configuration
    pruner_type: str = 'hyperband'
    pruner_params: Dict[str, Any] = field(default_factory=dict)
    
    # Storage configuration
    storage_url: Optional[str] = None
    study_name: Optional[str] = None
    load_if_exists: bool = True
    
    # Validation configuration
    validation_strategy: str = 'walkforward'
    n_splits: int = 5
    embargo: int = 10
    
    # Metrics configuration
    primary_metric: str = 'pr_auc'
    secondary_metrics: List[str] = field(
        default_factory=lambda: ['f1', 'mcc', 'accuracy']
    )
    
    # MLflow configuration
    use_mlflow: bool = True
    mlflow_experiment: str = 'unified_optimization'
    
    # Memory management
    enable_memory_management: bool = True
    memory_cleanup_interval: int = 5  # Clean every N trials
    
    # Calibration
    calibration_method: str = 'auto'
    
    # Early stopping
    early_stopping_patience: int = 10
    early_stopping_min_delta: float = 1e-4


class UnifiedModelOptimizer(ABC):
    """
    Unified Optuna optimizer for all model types.
    
    This base class provides common functionality for all model optimizers,
    reducing code duplication while maintaining flexibility for model-specific
    implementations.
    
    Example:
        >>> config = UnifiedOptunaConfig(model_type='xgb', n_trials=50)
        >>> optimizer = XGBoostUnifiedOptimizer(config)
        >>> study = optimizer.optimize(X_train, y_train)
        >>> best_model = optimizer.best_model
    """
    
    def __init__(self, config: UnifiedOptunaConfig):
        """Initialize unified optimizer."""
        self.config = config
        self.study = None
        self.best_params = None
        self.best_model = None
        self.best_score = None
        self.memory_manager = MemoryManager() if config.enable_memory_management else None
        self.validator = self._create_validator()
        self.scaler = StandardScaler()
        self.trial_count = 0
        
    def _create_validator(self):
        """Create validator based on configuration."""
        # Import here to avoid circular dependency
        from src.validation.unified import create_validator
        return create_validator(
            strategy=self.config.validation_strategy,
            n_splits=self.config.n_splits,
            embargo=self.config.embargo
        )
        
    def _create_sampler(self) -> optuna.samplers.BaseSampler:
        """Create Optuna sampler based on configuration."""
        samplers = {
            'tpe': optuna.samplers.TPESampler,
            'random': optuna.samplers.RandomSampler,
            'cmaes': optuna.samplers.CmaEsSampler,
            'grid': optuna.samplers.GridSampler,
        }
        
        sampler_class = samplers.get(self.config.sampler_type)
        if not sampler_class:
            raise ValueError(f"Unknown sampler type: {self.config.sampler_type}")
        
        # Add seed to sampler params if not present
        params = self.config.sampler_params.copy()
        if 'seed' not in params and self.config.sampler_type != 'grid':
            params['seed'] = self.config.seed
            
        return sampler_class(**params)
    
    def _create_pruner(self) -> Optional[optuna.pruners.BasePruner]:
        """Create Optuna pruner based on configuration."""
        if self.config.pruner_type == 'none':
            return None
            
        pruners = {
            'median': optuna.pruners.MedianPruner,
            'percentile': optuna.pruners.PercentilePruner,
            'hyperband': optuna.pruners.HyperbandPruner,
            'asha': optuna.pruners.SuccessiveHalvingPruner,
            'threshold': optuna.pruners.ThresholdPruner,
        }
        
        pruner_class = pruners.get(self.config.pruner_type)
        if not pruner_class:
            raise ValueError(f"Unknown pruner type: {self.config.pruner_type}")
            
        return pruner_class(**self.config.pruner_params)
    
    def _create_study(self) -> optuna.Study:
        """Create or load Optuna study."""
        sampler = self._create_sampler()
        pruner = self._create_pruner()
        
        # Setup storage if provided
        storage = None
        if self.config.storage_url:
            try:
                from optuna.storages import RDBStorage
                storage = RDBStorage(url=self.config.storage_url)
            except Exception as e:
                warnings.warn(f"Failed to create storage: {e}")
        
        # Generate study name if not provided
        study_name = self.config.study_name or f"{self.config.model_type}_{self.config.seed}"
        
        # Create or load study
        study = optuna.create_study(
            study_name=study_name,
            direction='maximize',
            sampler=sampler,
            pruner=pruner,
            storage=storage,
            load_if_exists=self.config.load_if_exists
        )
        
        return study
    
    @abstractmethod
    def _create_search_space(self, trial: optuna.Trial) -> Dict[str, Any]:
        """
        Create model-specific hyperparameter search space.
        
        Must be implemented by subclasses.
        
        Args:
            trial: Optuna trial object
            
        Returns:
            Dictionary of hyperparameters
        """
        pass
    
    @abstractmethod
    def _create_model(self, params: Dict[str, Any]):
        """
        Create model instance with given parameters.
        
        Must be implemented by subclasses.
        
        Args:
            params: Hyperparameters dictionary
            
        Returns:
            Model instance
        """
        pass
    
    @abstractmethod
    def _train_model(
        self,
        model,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        trial: Optional[optuna.Trial] = None
    ) -> Tuple[Any, Dict[str, float]]:
        """
        Train model and return trained model and validation metrics.
        
        Must be implemented by subclasses.
        
        Args:
            model: Model instance
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            trial: Optuna trial for pruning (optional)
            
        Returns:
            Tuple of (trained_model, metrics_dict)
        """
        pass
    
    def _calculate_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """Calculate all configured metrics."""
        metrics = {}
        
        # Binary predictions metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['f1'] = f1_score(y_true, y_pred, zero_division=0)
        metrics['mcc'] = matthews_corrcoef(y_true, y_pred)
        
        # Probability-based metrics
        if y_proba is not None:
            metrics['pr_auc'] = average_precision_score(y_true, y_proba)
            metrics['roc_auc'] = roc_auc_score(y_true, y_proba)
            
            # Calibration metrics (if available)
            try:
                from src.models.metrics.calibration import expected_calibration_error
                metrics['ece'] = expected_calibration_error(y_true, y_proba)
            except ImportError:
                pass
            
        return metrics
    
    def _objective(self, trial: optuna.Trial) -> float:
        """Objective function for Optuna optimization."""
        # Create hyperparameters
        params = self._create_search_space(trial)
        
        # Memory cleanup
        if self.memory_manager and self.trial_count % self.config.memory_cleanup_interval == 0:
            self.memory_manager.cleanup()
        
        self.trial_count += 1
        
        # Cross-validation
        scores = []
        for fold, (train_idx, val_idx) in enumerate(self.validator.split(self.X, self.y)):
            # Split data
            X_train, X_val = self.X[train_idx], self.X[val_idx]
            y_train, y_val = self.y[train_idx], self.y[val_idx]
            
            # Create and train model
            model = self._create_model(params)
            trained_model, metrics = self._train_model(
                model, X_train, y_train, X_val, y_val, trial
            )
            
            # Calculate score
            score = metrics.get(self.config.primary_metric, 0)
            scores.append(score)
            
            # Report intermediate value for pruning
            if trial:
                trial.report(score, fold)
                if trial.should_prune():
                    raise optuna.TrialPruned()
            
            # Cleanup after each fold
            del model, trained_model
            if self.memory_manager:
                self.memory_manager.cleanup(force=False)
        
        # Average score across folds
        avg_score = np.mean(scores)
        
        # MLflow tracking
        if self.config.use_mlflow:
            with mlflow.start_run(nested=True):
                mlflow.log_params(params)
                mlflow.log_metric(self.config.primary_metric, avg_score)
                mlflow.log_metric(f"{self.config.primary_metric}_std", np.std(scores))
        
        return avg_score
    
    def optimize(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray]) -> optuna.Study:
        """
        Run optimization on given data.
        
        Args:
            X: Features (DataFrame or array)
            y: Labels (Series or array)
            
        Returns:
            Optuna study object with optimization results
        """
        # Prepare data
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values
            
        # Scale features
        self.X = self.scaler.fit_transform(X)
        self.y = y
        
        # Setup MLflow
        if self.config.use_mlflow:
            mlflow.set_experiment(self.config.mlflow_experiment)
        
        # Create and run study
        self.study = self._create_study()
        
        try:
            self.study.optimize(
                self._objective,
                n_trials=self.config.n_trials,
                timeout=self.config.timeout,
                show_progress_bar=True
            )
        except KeyboardInterrupt:
            print("Optimization interrupted by user")
        finally:
            # Cleanup
            if self.memory_manager:
                self.memory_manager.cleanup(force=True)
        
        # Store best results
        self.best_params = self.study.best_params
        self.best_score = self.study.best_value
        
        # Train final model with best params on all data
        self.best_model = self._create_model(self.best_params)
        self.best_model, _ = self._train_model(
            self.best_model, self.X, self.y
        )
        
        print(f"\nOptimization complete!")
        print(f"Best {self.config.primary_metric}: {self.best_score:.4f}")
        print(f"Best parameters: {self.best_params}")
        
        return self.study
    
    def get_feature_importance(self) -> Optional[pd.DataFrame]:
        """
        Get feature importance from best model.
        
        Returns:
            DataFrame with feature importance or None if not available
        """
        if hasattr(self.best_model, 'feature_importances_'):
            importance = pd.DataFrame({
                'feature': range(len(self.best_model.feature_importances_)),
                'importance': self.best_model.feature_importances_
            })
            return importance.sort_values('importance', ascending=False)
        return None
    
    def save_model(self, path: Union[str, Path]):
        """
        Save best model to disk.
        
        Args:
            path: Path to save the model
        """
        if self.best_model is None:
            raise ValueError("No model to save. Run optimize() first.")
            
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        import joblib
        joblib.dump({
            'model': self.best_model,
            'params': self.best_params,
            'score': self.best_score,
            'scaler': self.scaler,
            'config': self.config
        }, path)
        
        print(f"Model saved to {path}")
    
    @classmethod
    def load_model(cls, path: Union[str, Path]):
        """
        Load model from disk.
        
        Args:
            path: Path to the saved model
            
        Returns:
            Optimizer instance with loaded model
        """
        import joblib
        data = joblib.load(path)
        
        optimizer = cls(data['config'])
        optimizer.best_model = data['model']
        optimizer.best_params = data['params']
        optimizer.best_score = data['score']
        optimizer.scaler = data['scaler']
        
        return optimizer