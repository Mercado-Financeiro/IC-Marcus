"""Utility functions for XGBoost Optuna optimization."""

import numpy as np
import pandas as pd
import logging
import structlog
from typing import Optional
import warnings

warnings.filterwarnings('ignore')


def get_logger():
    """Get logger with fallback to standard logging."""
    try:
        return structlog.get_logger()
    except:
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(__name__)


def check_constant_predictions(y_pred_proba: np.ndarray, 
                              threshold_std: float = 0.001,
                              min_unique: int = 5,
                              range_threshold: float = 0.01) -> bool:
    """
    Check if predictions are essentially constant.
    
    Args:
        y_pred_proba: Predicted probabilities
        threshold_std: Minimum standard deviation required
        min_unique: Minimum number of unique predictions
        range_threshold: Minimum range of predictions
        
    Returns:
        True if predictions are constant, False otherwise
    """
    if len(y_pred_proba) == 0:
        return True
    
    pred_std = np.std(y_pred_proba)
    pred_mean = np.mean(y_pred_proba)
    unique_preds = len(np.unique(np.round(y_pred_proba, 3)))
    pred_range = y_pred_proba.max() - y_pred_proba.min()
    
    # Multiple criteria for detecting constant predictions
    is_constant = (
        pred_std < threshold_std or
        unique_preds < min_unique or
        pred_range < range_threshold or
        np.allclose(y_pred_proba, pred_mean, atol=0.001)
    )
    
    if is_constant:
        log = get_logger()
        log.warning(
            "constant_predictions_detected",
            pred_std=pred_std,
            pred_mean=pred_mean,
            unique_count=unique_preds,
            pred_range=pred_range
        )
    
    return is_constant


def calculate_scale_pos_weight(y: pd.Series) -> float:
    """
    Calculate scale_pos_weight for imbalanced datasets.
    
    Args:
        y: Target labels
        
    Returns:
        scale_pos_weight value
    """
    n_negative = (y == 0).sum()
    n_positive = (y == 1).sum()
    
    if n_positive == 0:
        return 1.0
    
    return n_negative / n_positive


def validate_features(X: pd.DataFrame) -> pd.DataFrame:
    """
    Validate and clean features.
    
    Args:
        X: Features DataFrame
        
    Returns:
        Cleaned features
    """
    log = get_logger()
    
    # Check for NaN values
    if X.isnull().any().any():
        n_nan = X.isnull().sum().sum()
        log.warning(f"Found {n_nan} NaN values, filling with 0")
        X = X.fillna(0)
    
    # Check for infinite values
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    inf_mask = np.isinf(X[numeric_cols].values)
    
    if inf_mask.any():
        n_inf = inf_mask.sum()
        log.warning(f"Found {n_inf} infinite values, clipping")
        X[numeric_cols] = X[numeric_cols].replace([np.inf, -np.inf], 0)
    
    # Remove constant features
    constant_cols = []
    for col in X.columns:
        if X[col].nunique() <= 1:
            constant_cols.append(col)
    
    if constant_cols:
        log.warning(f"Removing {len(constant_cols)} constant features: {constant_cols}")
        X = X.drop(columns=constant_cols)
    
    return X


def calculate_feature_importance_stability(importances_list: list) -> dict:
    """
    Calculate stability metrics for feature importances across folds.
    
    Args:
        importances_list: List of feature importance dictionaries
        
    Returns:
        Dictionary with mean and std of importances
    """
    if not importances_list:
        return {}
    
    # Convert to DataFrame for easier calculation
    df = pd.DataFrame(importances_list)
    
    # Calculate statistics
    result = {
        'mean': df.mean().to_dict(),
        'std': df.std().to_dict(),
        'cv': (df.std() / df.mean()).to_dict()  # Coefficient of variation
    }
    
    # Find most stable features (low CV)
    cv_series = df.std() / df.mean()
    result['most_stable'] = cv_series.nsmallest(10).index.tolist()
    result['least_stable'] = cv_series.nlargest(10).index.tolist()
    
    return result


def create_sample_weights(y: pd.Series, method: str = 'balanced') -> np.ndarray:
    """
    Create sample weights for training.
    
    Args:
        y: Target labels
        method: Weighting method ('balanced', 'sqrt', 'linear')
        
    Returns:
        Array of sample weights
    """
    from sklearn.utils.class_weight import compute_sample_weight
    
    if method == 'balanced':
        return compute_sample_weight('balanced', y)
    elif method == 'sqrt':
        # Square root weighting
        weights = compute_sample_weight('balanced', y)
        return np.sqrt(weights)
    elif method == 'linear':
        # Linear weighting based on class frequency
        class_weights = len(y) / (len(np.unique(y)) * np.bincount(y))
        return class_weights[y]
    else:
        # No weighting
        return np.ones(len(y))


def get_memory_usage():
    """Get current memory usage in MB."""
    import psutil
    import os
    
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024  # MB


def log_experiment_info(experiment_name: str, params: dict, dataset_info: dict):
    """
    Log experiment information.
    
    Args:
        experiment_name: Name of the experiment
        params: Model parameters
        dataset_info: Information about the dataset
    """
    log = get_logger()
    
    log.info(
        "experiment_started",
        name=experiment_name,
        n_samples=dataset_info.get('n_samples'),
        n_features=dataset_info.get('n_features'),
        class_balance=dataset_info.get('class_balance'),
        memory_mb=get_memory_usage()
    )
    
    log.info("model_params", **params)