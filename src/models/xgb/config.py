"""Configuration classes for XGBoost models."""

from dataclasses import dataclass
from typing import Optional, Dict, Any


@dataclass
class XGBoostConfig:
    """Configuration for XGBoost base model."""
    
    # Basic parameters
    n_estimators: int = 100
    max_depth: int = 6
    learning_rate: float = 0.3
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    
    # Regularization
    reg_alpha: float = 0.0
    reg_lambda: float = 1.0
    gamma: float = 0.0
    min_child_weight: int = 1
    
    # Training parameters
    objective: str = 'binary:logistic'
    eval_metric: str = 'logloss'
    tree_method: str = 'hist'
    device: str = 'cpu'
    
    # Other parameters
    random_state: int = 42
    n_jobs: int = -1
    verbosity: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for XGBoost."""
        return {
            'n_estimators': self.n_estimators,
            'max_depth': self.max_depth,
            'learning_rate': self.learning_rate,
            'subsample': self.subsample,
            'colsample_bytree': self.colsample_bytree,
            'reg_alpha': self.reg_alpha,
            'reg_lambda': self.reg_lambda,
            'gamma': self.gamma,
            'min_child_weight': self.min_child_weight,
            'objective': self.objective,
            'eval_metric': self.eval_metric,
            'tree_method': self.tree_method,
            'device': self.device,
            'random_state': self.random_state,
            'n_jobs': self.n_jobs,
            'verbosity': self.verbosity
        }


@dataclass
class OptunaConfig:
    """Configuration for Optuna optimization."""
    
    n_trials: int = 100
    cv_folds: int = 5
    embargo: int = 10
    
    # Pruner settings
    pruner_type: str = 'hyperband'  # median, hyperband, successive_halving
    pruner_warmup_steps: int = 5
    pruner_interval_steps: int = 1
    
    # Sampler settings
    sampler_type: str = 'tpe'  # tpe, random, grid
    sampler_seed: int = 42
    
    # Optimization settings
    direction: str = 'maximize'
    metric: str = 'f1'
    timeout: Optional[int] = None
    
    # MLflow settings
    use_mlflow: bool = True
    experiment_name: str = 'xgboost_optuna'
    
    # Early stopping
    early_stopping_rounds: int = 50
    early_stopping_tolerance: float = 0.001


@dataclass
class ThresholdConfig:
    """Configuration for threshold optimization."""
    
    # Methods to use
    optimize_f1: bool = True
    optimize_ev: bool = True
    optimize_profit: bool = True
    
    # Search parameters
    threshold_min: float = 0.1
    threshold_max: float = 0.9
    threshold_steps: int = 181  # 0.1 to 0.9 with 0.005 step
    
    # Cost parameters
    transaction_cost_bps: float = 10
    slippage_bps: float = 5
    
    # Risk parameters
    kelly_fraction: float = 0.25
    max_position_size: float = 1.0


@dataclass
class ValidationConfig:
    """Configuration for temporal validation."""
    
    # Purged K-Fold settings
    n_splits: int = 5
    embargo_td: int = 10  # Time periods
    
    # Walk-forward settings
    use_walk_forward: bool = False
    walk_forward_window: int = 252
    walk_forward_test_size: int = 63
    
    # Combinatorial Purged CV
    use_combinatorial: bool = False
    n_test_groups: int = 2