"""Configuration for XGBoost with Optuna optimization."""

from dataclasses import dataclass
from typing import Optional, Dict, Any


@dataclass
class XGBOptunaConfig:
    """Configuration for XGBoost Optuna optimization."""
    
    # Optimization settings
    n_trials: int = 100
    cv_folds: int = 5
    embargo: int = 10
    pruner_type: str = 'hyperband'  # median, hyperband, successive_halving
    seed: int = 42
    
    # XGBoost hyperparameter ranges
    n_estimators_min: int = 100
    n_estimators_max: int = 1000
    max_depth_min: int = 3
    max_depth_max: int = 10
    learning_rate_min: float = 0.01
    learning_rate_max: float = 0.3
    subsample_min: float = 0.6
    subsample_max: float = 1.0
    colsample_bytree_min: float = 0.6
    colsample_bytree_max: float = 1.0
    gamma_min: float = 0
    gamma_max: float = 5
    reg_alpha_min: float = 0
    reg_alpha_max: float = 10
    reg_lambda_min: float = 0
    reg_lambda_max: float = 10
    min_child_weight_min: int = 1
    min_child_weight_max: int = 10
    
    # Training settings
    early_stopping_rounds: int = 50
    tree_method: str = 'hist'  # auto, exact, approx, hist
    device: str = 'cpu'  # cpu, cuda
    
    # Calibration settings
    calibration_method: str = 'isotonic'  # isotonic, sigmoid
    calibration_cv: int = 3
    
    # Threshold optimization
    optimize_threshold: bool = True
    threshold_strategies: list = None
    
    # MLflow settings
    use_mlflow: bool = True
    experiment_name: str = 'xgboost_optuna'
    
    def __post_init__(self):
        """Initialize default values."""
        if self.threshold_strategies is None:
            self.threshold_strategies = ['f1', 'ev', 'profit']
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            k: v for k, v in self.__dict__.items()
            if not k.startswith('_')
        }


@dataclass
class OptimizationMetrics:
    """Metrics from optimization."""
    
    best_score: float
    best_params: Dict[str, Any]
    cv_scores: list
    feature_importance: Dict[str, float]
    threshold_f1: float = 0.5
    threshold_ev: float = 0.5
    threshold_profit: float = 0.5
    
    # ML metrics
    accuracy: float = None
    precision: float = None
    recall: float = None
    f1: float = None
    auc: float = None
    pr_auc: float = None
    mcc: float = None
    brier: float = None
    
    # Trading metrics
    sharpe: float = None
    sortino: float = None
    calmar: float = None
    max_drawdown: float = None
    win_rate: float = None
    profit_factor: float = None
    expected_value: float = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            k: v for k, v in self.__dict__.items()
            if v is not None and not k.startswith('_')
        }