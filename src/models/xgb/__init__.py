"""XGBoost model components."""

from .optuna import (
    XGBOptuna,
    XGBOptunaConfig,
    OptimizationMetrics,
    ModelCalibrator,
    ThresholdOptimizer,
    TradingMetrics
)

# Alias for backward compatibility
XGBoostOptuna = XGBOptuna

__all__ = [
    'XGBoostOptuna',
    'XGBOptuna',
    'XGBOptunaConfig',
    'OptimizationMetrics',
    'ModelCalibrator',
    'ThresholdOptimizer',
    'TradingMetrics'
]