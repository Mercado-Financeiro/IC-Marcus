"""Utility functions for LSTM Optuna optimization."""

import numpy as np
import pandas as pd
import torch
from typing import Tuple, Optional, Union
import structlog
import logging
from pathlib import Path
import sys

# Add parent to path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))


class LoggerShim:
    """Compatibility shim for structlog-style logging when structlog is not available."""
    
    def __init__(self, logger):
        self._logger = logger
        
    def info(self, msg, **kwargs):
        extra_info = ' '.join(f'{k}={v}' for k, v in kwargs.items())
        self._logger.info(f"{msg} {extra_info}" if extra_info else msg)
        
    def warning(self, msg, **kwargs):
        extra_info = ' '.join(f'{k}={v}' for k, v in kwargs.items())
        self._logger.warning(f"{msg} {extra_info}" if extra_info else msg)
        
    def error(self, msg, **kwargs):
        extra_info = ' '.join(f'{k}={v}' for k, v in kwargs.items())
        self._logger.error(f"{msg} {extra_info}" if extra_info else msg)
        
    def debug(self, msg, **kwargs):
        extra_info = ' '.join(f'{k}={v}' for k, v in kwargs.items())
        self._logger.debug(f"{msg} {extra_info}" if extra_info else msg)


def get_logger():
    """Get logger with fallback to standard logging."""
    try:
        import structlog
        return structlog.get_logger()
    except ImportError:
        logging.basicConfig(
            level=logging.INFO, 
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        return LoggerShim(logging.getLogger(__name__))


def set_lstm_deterministic(seed: int = 42):
    """
    Set deterministic behavior for LSTM training.
    
    Args:
        seed: Random seed
    """
    # Import here to avoid circular dependency
    from src.utils.determinism import set_deterministic_environment
    
    # Use base determinism function
    set_deterministic_environment(seed)
    
    # PyTorch specific determinism
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    # Enhanced deterministic operations
    try:
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception as e:
        log = get_logger()
        log.warning("partial_determinism_only", error=str(e))
    
    # Set number of threads for reproducibility
    torch.set_num_threads(1)
    
    # Additional environment variables for determinism
    import os
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    
    log = get_logger()
    log.info("lstm_deterministic_mode_enabled", seed=seed, torch_version=torch.__version__)


def check_constant_predictions(
    y_pred_proba: np.ndarray, 
    threshold_std: float = 0.005, 
    min_unique: int = 10
) -> bool:
    """Check if predictions are constant (all same value).
    Check if predictions are essentially constant.
    
    Args:
        y_pred_proba: Predicted probabilities
        threshold_std: Minimum standard deviation required
        min_unique: Minimum number of unique rounded predictions
        
    Returns:
        True if predictions are constant, False otherwise
    """
    if len(y_pred_proba) == 0:
        return True
    
    # Calculate statistics
    pred_mean = np.mean(y_pred_proba)
    pred_std = np.std(y_pred_proba)
    unique_preds = len(np.unique(np.round(y_pred_proba, 4)))
    
    # Check if predictions are constant
    is_constant = pred_std < threshold_std or unique_preds < min_unique
    
    if is_constant:
        log = get_logger()
        log.warning(
            "constant_predictions_detected", 
            pred_mean=pred_mean, 
            pred_std=pred_std, 
            unique_count=unique_preds
        )
        # Log sample predictions for debugging
        sample_preds = y_pred_proba[:10].tolist() if len(y_pred_proba) >= 10 else y_pred_proba.tolist()
        log.warning("sample_predictions", samples=sample_preds)
    
    return is_constant


def create_sequences(
    X: Union[pd.DataFrame, np.ndarray], 
    y: Union[pd.Series, np.ndarray], 
    seq_len: int,
    stride: int = 1
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sequences for LSTM training.
    
    Args:
        X: Features
        y: Labels
        seq_len: Sequence length
        stride: Step size between sequences
        
    Returns:
        Tuple of (sequences, labels)
    """
    # Convert to numpy if needed
    if isinstance(X, pd.DataFrame):
        X = X.values
    if isinstance(y, pd.Series):
        y = y.values
    
    sequences = []
    labels = []
    
    for i in range(0, len(X) - seq_len + 1, stride):
        sequences.append(X[i:i+seq_len])
        labels.append(y[i+seq_len-1])
    
    return np.array(sequences), np.array(labels)


def get_device(device_str: str = 'auto') -> torch.device:
    """
    Get PyTorch device.
    
    Args:
        device_str: Device string ('auto', 'cpu', 'cuda', 'mps')
        
    Returns:
        torch.device
    """
    if device_str == 'auto':
        if torch.cuda.is_available():
            return torch.device('cuda')
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return torch.device('mps')
        else:
            return torch.device('cpu')
    else:
        try:
            return torch.device(device_str)
        except (RuntimeError, ValueError):
            # Invalid device, fallback to CPU
            return torch.device('cpu')


def calculate_metrics(
    y_true: np.ndarray, 
    y_pred_proba: np.ndarray,
    threshold: float = 0.5
) -> dict:
    """
    Calculate classification metrics.
    
    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        threshold: Classification threshold
        
    Returns:
        Dictionary of metrics
    """
    from sklearn.metrics import (
        f1_score, precision_score, recall_score, accuracy_score,
        roc_auc_score, average_precision_score,
        brier_score_loss, matthews_corrcoef
    )
    
    y_pred = (y_pred_proba >= threshold).astype(int)
    
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred),
        'auc': roc_auc_score(y_true, y_pred_proba),
        'roc_auc': roc_auc_score(y_true, y_pred_proba),
        'pr_auc': average_precision_score(y_true, y_pred_proba),
        'brier': brier_score_loss(y_true, y_pred_proba),
        'mcc': matthews_corrcoef(y_true, y_pred)
    }
    
    return metrics