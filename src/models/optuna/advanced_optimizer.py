"""
Advanced Optuna optimizer with ASHA, SuccessiveHalving, and state-of-the-art features.

Implements production-grade Bayesian optimization with:
- Advanced pruning strategies (ASHA, SuccessiveHalving, HyperBand, Percentile)
- Temperature scaling calibration
- Walk-forward outer validation
- ECE and comprehensive calibration metrics
- Persistent storage with RDBStorage
- Full determinism and reproducibility

References:
- "Asynchronous Successive Halving" (Li et al., 2018)
- "Hyperband: A Novel Bandit-Based Approach" (Li et al., 2017)
- "On Calibration of Modern Neural Networks" (Guo et al., 2017)
"""

import os
import json
import pickle
import sqlite3
from pathlib import Path
from typing import Dict, Any, Optional, Union, List, Callable, Tuple
from dataclasses import dataclass, field
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import optuna
from optuna.pruners import (
    BasePruner, MedianPruner, PercentilePruner, 
    SuccessiveHalvingPruner, HyperbandPruner
)
from optuna.samplers import TPESampler, RandomSampler, CmaEsSampler
from optuna.storages import RDBStorage
import mlflow

# Import our modules
from ..calibration.temperature import TemperatureScaling, VectorScaling
from ..metrics.calibration import (
    expected_calibration_error, comprehensive_calibration_metrics
)
from ..validation.walkforward import WalkForwardValidator, WalkForwardConfig, OuterWalkForward
from ...utils.determinism_enhanced import set_full_determinism, DeterministicContext


class ASHAPruner(BasePruner):
    """
    Asynchronous Successive Halving Algorithm (ASHA) Pruner.
    
    More aggressive than standard SuccessiveHalving for faster optimization.
    Good for neural networks where early stopping is reliable.
    
    References:
    - "Asynchronous Successive Halving" (Li et al., 2018)
    - https://arxiv.org/abs/1810.05934
    """
    
    def __init__(
        self,
        min_resource: int = 1,
        max_resource: int = 100,
        reduction_factor: float = 3.0,
        min_early_stopping_rate: int = 0
    ):
        """
        Initialize ASHA pruner.
        
        Args:
            min_resource: Minimum resource (epochs/iterations)
            max_resource: Maximum resource
            reduction_factor: Factor to reduce population at each rung
            min_early_stopping_rate: Minimum trials before pruning starts
        """
        self.min_resource = min_resource
        self.max_resource = max_resource
        self.reduction_factor = reduction_factor
        self.min_early_stopping_rate = min_early_stopping_rate
        
        # Calculate rungs
        self.rungs = []
        resource = min_resource
        while resource <= max_resource:
            self.rungs.append(resource)
            resource = int(resource * reduction_factor)
        
        # Track trial performances at each rung
        self.rung_records = {rung: [] for rung in self.rungs}
    
    def prune(self, study: optuna.Study, trial: optuna.Trial) -> bool:
        """
        Determine if trial should be pruned.
        
        Args:
            study: Optuna study
            trial: Current trial
            
        Returns:
            True if trial should be pruned
        """
        current_resource = len(trial.intermediate_values)
        
        if current_resource < self.min_resource:
            return False
        
        if len(trial.intermediate_values) < self.min_early_stopping_rate:
            return False
        
        # Find appropriate rung
        rung = None
        for r in reversed(self.rungs):
            if current_resource >= r:
                rung = r
                break
        
        if rung is None:
            return False
        
        # Get current performance
        if current_resource not in trial.intermediate_values:
            return False
        
        current_value = trial.intermediate_values[current_resource]
        
        # Add to rung records
        self.rung_records[rung].append((trial.number, current_value))
        
        # Calculate threshold for this rung
        rung_values = [val for _, val in self.rung_records[rung]]
        
        if len(rung_values) < self.reduction_factor:
            return False
        
        # Determine direction (maximize vs minimize)
        if study.direction == optuna.study.StudyDirection.MAXIMIZE:
            threshold = np.percentile(rung_values, 100 / self.reduction_factor)
            should_prune = current_value < threshold
        else:
            threshold = np.percentile(rung_values, 100 - 100 / self.reduction_factor)
            should_prune = current_value > threshold
        
        return should_prune


@dataclass
class AdvancedOptimizerConfig:
    """Configuration for advanced Optuna optimizer."""
    
    # Basic settings
    n_trials: int = 100
    timeout: Optional[int] = None
    seed: int = 42
    
    # Sampler settings
    sampler_type: str = 'tpe'  # 'tpe', 'random', 'cmaes'
    sampler_params: Dict[str, Any] = field(default_factory=dict)
    
    # Pruner settings
    pruner_type: str = 'asha'  # 'asha', 'successive_halving', 'hyperband', 'median', 'percentile'
    pruner_params: Dict[str, Any] = field(default_factory=dict)
    
    # Storage settings
    storage_url: Optional[str] = None
    study_name: Optional[str] = None
    load_if_exists: bool = True
    
    # Validation settings
    use_outer_cv: bool = True
    outer_cv_splits: int = 3
    inner_cv_splits: int = 5
    embargo: int = 5
    
    # Calibration settings
    calibration_method: str = 'temperature'  # 'temperature', 'vector', 'isotonic', 'platt'
    calibration_params: Dict[str, Any] = field(default_factory=dict)
    
    # MLflow integration
    use_mlflow: bool = True
    mlflow_experiment: str = 'advanced_optimization'
    
    # Metrics
    primary_metric: str = 'f1_score'
    calibration_metrics: bool = True
    
    # Early stopping
    early_stopping_rounds: Optional[int] = 20
    min_improvement: float = 1e-4
    
    # Parallelization
    n_jobs: int = 1
    
    # Debugging
    verbose: bool = False
    debug_mode: bool = False


class AdvancedOptunaOptimizer:
    """
    Advanced Optuna optimizer with state-of-the-art features.
    
    Features:
    - ASHA, SuccessiveHalving, HyperBand pruning
    - Temperature scaling calibration
    - Walk-forward outer validation
    - ECE and comprehensive calibration metrics
    - Persistent storage
    - Full determinism
    """
    
    def __init__(self, config: AdvancedOptimizerConfig):
        """
        Initialize advanced optimizer.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.study = None
        self.best_params = None
        self.best_score = None
        self.best_model = None
        self.calibrator = None
        self.outer_validator = None
        
        # Results storage
        self.results = {
            'outer_scores': [],
            'calibration_metrics': [],
            'feature_importance': {},
            'trial_history': []
        }
        
        # Set deterministic environment
        self.determinism_results = set_full_determinism(
            seed=self.config.seed,
            verify=True
        )
        
        if self.config.verbose:
            print(f"Determinism status: {self.determinism_results['status']}")
    
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
    
    def _create_pruner(self) -> BasePruner:
        """Create Optuna pruner based on configuration."""
        params = self.config.pruner_params.copy()
        
        if self.config.pruner_type == 'asha':
            params.setdefault('min_resource', 5)
            params.setdefault('max_resource', 100)
            params.setdefault('reduction_factor', 3.0)
            return ASHAPruner(**params)
            
        elif self.config.pruner_type == 'successive_halving':
            params.setdefault('min_resource', 5)
            params.setdefault('reduction_factor', 3)
            return SuccessiveHalvingPruner(**params)
            
        elif self.config.pruner_type == 'hyperband':
            params.setdefault('min_resource', 5)
            params.setdefault('max_resource', 100)
            params.setdefault('reduction_factor', 3)
            return HyperbandPruner(**params)
            
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
        study_name = self.config.study_name or f"advanced_opt_{self.config.seed}"
        
        # Create or load study
        try:
            study = optuna.create_study(
                study_name=study_name,
                direction='maximize' if self._is_maximize_metric() else 'minimize',
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
                direction='maximize' if self._is_maximize_metric() else 'minimize',
                sampler=sampler,
                pruner=pruner
            )
        
        return study
    
    def _is_maximize_metric(self) -> bool:
        """Determine if primary metric should be maximized."""
        maximize_metrics = {
            'f1_score', 'precision', 'recall', 'accuracy', 
            'roc_auc', 'pr_auc', 'mcc'
        }
        return self.config.primary_metric in maximize_metrics
    
    def _setup_outer_validation(self, X: pd.DataFrame, y: pd.Series):
        """Setup outer walk-forward validation."""
        if not self.config.use_outer_cv:
            return
        
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
        
        self.outer_validator = OuterWalkForward(
            n_outer_splits=self.config.outer_cv_splits,
            n_inner_splits=self.config.inner_cv_splits,
            outer_config=outer_config,
            inner_config=inner_config
        )
    
    def _create_calibrator(self):
        """Create calibrator based on configuration."""
        params = self.config.calibration_params.copy()
        
        if self.config.calibration_method == 'temperature':
            params.setdefault('max_iter', 100)
            params.setdefault('lr', 0.01)
            return TemperatureScaling(**params)
            
        elif self.config.calibration_method == 'vector':
            params.setdefault('max_iter', 100)
            params.setdefault('lr', 0.01)
            return VectorScaling(**params)
            
        elif self.config.calibration_method == 'isotonic':
            from sklearn.calibration import IsotonicRegression
            return IsotonicRegression(out_of_bounds='clip')
            
        elif self.config.calibration_method == 'platt':
            from sklearn.linear_model import LogisticRegression
            return LogisticRegression()
            
        else:
            raise ValueError(f"Unknown calibration method: {self.config.calibration_method}")
    
    def _evaluate_calibration(
        self, 
        y_true: np.ndarray, 
        y_prob: np.ndarray
    ) -> Dict[str, float]:
        """Evaluate calibration metrics."""
        if not self.config.calibration_metrics:
            return {}
        
        try:
            metrics = comprehensive_calibration_metrics(y_true, y_prob)
            return metrics
        except Exception as e:
            warnings.warn(f"Failed to calculate calibration metrics: {e}")
            return {}
    
    def _objective_with_outer_cv(
        self,
        trial: optuna.Trial,
        X: pd.DataFrame,
        y: pd.Series,
        model_factory: Callable,
        param_factory: Callable
    ) -> float:
        """Objective function with outer cross-validation."""
        # Get hyperparameters
        params = param_factory(trial)
        
        outer_scores = []
        calibration_scores = []
        
        # Outer cross-validation
        for dev_idx, test_idx, inner_validator in self.outer_validator.split(X, y):
            X_dev = X.iloc[dev_idx]
            y_dev = y.iloc[dev_idx]
            X_test = X.iloc[test_idx]
            y_test = y.iloc[test_idx]
            
            # Inner optimization on development set
            best_inner_score = -np.inf if self._is_maximize_metric() else np.inf
            best_inner_model = None
            
            for train_idx, val_idx, _ in inner_validator.split(X_dev, y_dev):
                X_train = X_dev.iloc[train_idx]
                y_train = y_dev.iloc[train_idx]
                X_val = X_dev.iloc[val_idx]
                y_val = y_dev.iloc[val_idx]
                
                # Train model
                model = model_factory(params)
                model.fit(X_train, y_train)
                
                # Validate
                y_pred = model.predict(X_val)
                score = self._calculate_score(y_val, y_pred)
                
                # Update best inner model
                is_better = (score > best_inner_score if self._is_maximize_metric() 
                           else score < best_inner_score)
                
                if is_better:
                    best_inner_score = score
                    best_inner_model = model
                
                # Report intermediate value for pruning
                trial.report(score, len(outer_scores) * self.config.inner_cv_splits + len(calibration_scores))
                
                if trial.should_prune():
                    raise optuna.TrialPruned()
            
            # Test best inner model
            if best_inner_model is not None:
                # Get probabilities for calibration
                if hasattr(best_inner_model, 'predict_proba'):
                    y_prob = best_inner_model.predict_proba(X_test)[:, 1]
                else:
                    y_prob = best_inner_model.predict(X_test)
                
                # Calibrate on development set
                calibrator = self._create_calibrator()
                
                if hasattr(best_inner_model, 'predict_proba'):
                    dev_prob = best_inner_model.predict_proba(X_dev)[:, 1]
                else:
                    dev_prob = best_inner_model.predict(X_dev)
                
                if self.config.calibration_method in ['isotonic', 'platt']:
                    calibrator.fit(dev_prob.reshape(-1, 1), y_dev)
                    y_prob_cal = calibrator.predict_proba(y_prob.reshape(-1, 1))[:, 1]
                else:
                    # For temperature/vector scaling, need logits
                    logits = np.log(y_prob / (1 - y_prob + 1e-8))  # Approximate logits
                    calibrator.fit(logits, y_test)
                    y_prob_cal = calibrator.transform(np.log(y_prob / (1 - y_prob + 1e-8)))
                
                # Calculate outer score
                y_pred_test = (y_prob_cal >= 0.5).astype(int)
                outer_score = self._calculate_score(y_test, y_pred_test)
                outer_scores.append(outer_score)
                
                # Calculate calibration metrics
                if self.config.calibration_metrics:
                    cal_metrics = self._evaluate_calibration(y_test, y_prob_cal)
                    calibration_scores.append(cal_metrics)
        
        # Store results
        self.results['outer_scores'].extend(outer_scores)
        self.results['calibration_metrics'].extend(calibration_scores)
        
        return np.mean(outer_scores)
    
    def _calculate_score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate primary metric score."""
        from sklearn.metrics import (
            f1_score, precision_score, recall_score, accuracy_score,
            roc_auc_score, average_precision_score, matthews_corrcoef
        )
        
        if self.config.primary_metric == 'f1_score':
            return f1_score(y_true, y_pred, zero_division=0)
        elif self.config.primary_metric == 'precision':
            return precision_score(y_true, y_pred, zero_division=0)
        elif self.config.primary_metric == 'recall':
            return recall_score(y_true, y_pred, zero_division=0)
        elif self.config.primary_metric == 'accuracy':
            return accuracy_score(y_true, y_pred)
        elif self.config.primary_metric == 'mcc':
            return matthews_corrcoef(y_true, y_pred)
        else:
            raise ValueError(f"Unsupported metric for discrete predictions: {self.config.primary_metric}")
    
    def optimize(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        model_factory: Callable,
        param_factory: Callable,
        callbacks: Optional[List] = None
    ) -> optuna.Study:
        """
        Run optimization with advanced features.
        
        Args:
            X: Training features
            y: Training labels
            model_factory: Function that creates model from parameters
            param_factory: Function that creates parameter space for trial
            callbacks: Optional callbacks for Optuna
            
        Returns:
            Completed Optuna study
        """
        # Setup
        self._setup_outer_validation(X, y)
        
        # Create study
        self.study = self._create_study()
        
        # MLflow integration
        if self.config.use_mlflow:
            try:
                mlflow.set_experiment(self.config.mlflow_experiment)
                mlflow.start_run()
                
                # Log configuration
                mlflow.log_params({
                    'sampler_type': self.config.sampler_type,
                    'pruner_type': self.config.pruner_type,
                    'n_trials': self.config.n_trials,
                    'primary_metric': self.config.primary_metric,
                    'calibration_method': self.config.calibration_method,
                    'outer_cv_splits': self.config.outer_cv_splits,
                    'inner_cv_splits': self.config.inner_cv_splits,
                    'seed': self.config.seed
                })
                
            except Exception as e:
                warnings.warn(f"MLflow setup failed: {e}")
        
        # Create objective function
        if self.config.use_outer_cv:
            objective = lambda trial: self._objective_with_outer_cv(
                trial, X, y, model_factory, param_factory
            )
        else:
            # Simple objective without outer CV
            def simple_objective(trial):
                params = param_factory(trial)
                model = model_factory(params)
                
                # Simple train/validation split
                from sklearn.model_selection import train_test_split
                X_train, X_val, y_train, y_val = train_test_split(
                    X, y, test_size=0.2, random_state=self.config.seed, stratify=y
                )
                
                model.fit(X_train, y_train)
                y_pred = model.predict(X_val)
                return self._calculate_score(y_val, y_pred)
            
            objective = simple_objective
        
        # Optimization
        try:
            with DeterministicContext(seed=self.config.seed):
                self.study.optimize(
                    objective,
                    n_trials=self.config.n_trials,
                    timeout=self.config.timeout,
                    callbacks=callbacks,
                    show_progress_bar=self.config.verbose,
                    n_jobs=self.config.n_jobs
                )
        
        except KeyboardInterrupt:
            print("Optimization interrupted by user")
        
        # Store results
        self.best_params = self.study.best_params
        self.best_score = self.study.best_value
        
        # MLflow logging
        if self.config.use_mlflow:
            try:
                mlflow.log_params(self.best_params)
                mlflow.log_metric('best_score', self.best_score)
                
                # Log calibration metrics if available
                if self.results['calibration_metrics']:
                    avg_cal_metrics = {}
                    for key in self.results['calibration_metrics'][0].keys():
                        values = [m[key] for m in self.results['calibration_metrics'] if key in m]
                        avg_cal_metrics[f'avg_{key}'] = np.mean(values)
                    
                    mlflow.log_metrics(avg_cal_metrics)
                
                mlflow.end_run()
                
            except Exception as e:
                warnings.warn(f"MLflow logging failed: {e}")
        
        if self.config.verbose:
            print(f"Optimization completed. Best score: {self.best_score:.4f}")
            print(f"Best parameters: {self.best_params}")
        
        return self.study
    
    def get_optimization_history(self) -> pd.DataFrame:
        """Get optimization history as DataFrame."""
        if self.study is None:
            return pd.DataFrame()
        
        records = []
        for trial in self.study.trials:
            record = {
                'trial_number': trial.number,
                'value': trial.value,
                'state': trial.state.name,
                'datetime_start': trial.datetime_start,
                'datetime_complete': trial.datetime_complete,
                'duration': (trial.datetime_complete - trial.datetime_start).total_seconds() if trial.datetime_complete else None
            }
            
            # Add parameters
            for key, value in trial.params.items():
                record[f'param_{key}'] = value
            
            # Add intermediate values
            for step, value in trial.intermediate_values.items():
                record[f'intermediate_{step}'] = value
            
            records.append(record)
        
        return pd.DataFrame(records)
    
    def save_results(self, filepath: Union[str, Path]):
        """Save optimization results."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        results = {
            'config': self.config.__dict__,
            'best_params': self.best_params,
            'best_score': self.best_score,
            'study_trials': len(self.study.trials) if self.study else 0,
            'results': self.results,
            'determinism': self.determinism_results
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(results, f)
    
    def load_results(self, filepath: Union[str, Path]):
        """Load optimization results."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        self.best_params = data['best_params']
        self.best_score = data['best_score']
        self.results = data.get('results', {})
        
        return data