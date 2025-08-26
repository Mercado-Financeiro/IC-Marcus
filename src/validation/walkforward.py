"""
Walk-forward validation with purged k-fold and embargo.
Prevents data leakage in time series data.
"""
import numpy as np
import pandas as pd
from typing import List, Tuple, Optional, Iterator
import logging
from sklearn.model_selection import BaseCrossValidator

logger = logging.getLogger(__name__)


class PurgedKFold(BaseCrossValidator):
    """
    Purged K-Fold cross-validation for time series data.
    
    This ensures no data leakage by:
    1. Purging samples that are influenced by the validation set
    2. Adding an embargo period after the validation set
    """
    
    def __init__(self, 
                 n_splits: int = 5,
                 embargo_pct: float = 0.01,
                 purge_pct: float = 0.01):
        """
        Initialize purged k-fold validator.
        
        Args:
            n_splits: Number of splits
            embargo_pct: Embargo period as percentage of total samples
            purge_pct: Purge period as percentage of total samples
        """
        self.n_splits = n_splits
        self.embargo_pct = embargo_pct
        self.purge_pct = purge_pct
        
    def get_n_splits(self, X=None, y=None, groups=None):
        """Return number of splits."""
        return self.n_splits
        
    def split(self, X, y=None, groups=None):
        """
        Generate indices for purged k-fold splits.
        
        Args:
            X: Features or indices
            y: Labels (not used)
            groups: Groups (not used)
            
        Yields:
            Tuple of train and validation indices
        """
        if hasattr(X, 'index'):
            indices = X.index.values
        else:
            indices = np.arange(len(X))
            
        n_samples = len(indices)
        embargo_size = int(n_samples * self.embargo_pct)
        purge_size = int(n_samples * self.purge_pct)
        
        # Calculate fold size
        fold_size = n_samples // self.n_splits
        
        for i in range(self.n_splits):
            # Validation set bounds
            val_start = i * fold_size
            val_end = min((i + 1) * fold_size, n_samples)
            
            # Skip if validation set is too small
            if val_end - val_start < 10:
                continue
                
            val_indices = indices[val_start:val_end]
            
            # Training set: all data before validation start (with purge)
            train_end = max(0, val_start - purge_size)
            train_indices_before = indices[:train_end] if train_end > 0 else np.array([])
            
            # Training set: all data after validation end (with embargo)
            train_start = min(n_samples, val_end + embargo_size)
            train_indices_after = indices[train_start:] if train_start < n_samples else np.array([])
            
            # Combine training indices
            train_indices = np.concatenate([train_indices_before, train_indices_after])
            
            # Skip if training set is too small
            if len(train_indices) < 20:
                continue
                
            logger.debug(f"Fold {i}: Train size={len(train_indices)}, Val size={len(val_indices)}")
            
            yield train_indices, val_indices


class WalkForwardValidator:
    """
    Walk-forward validation for time series models.
    """
    
    def __init__(self,
                 initial_window: int = 252,  # ~1 year of daily data
                 step_size: int = 21,        # ~1 month steps
                 validation_size: int = 21,   # ~1 month validation
                 embargo: int = 1,           # 1 day embargo
                 purge: int = 1):           # 1 day purge
        """
        Initialize walk-forward validator.
        
        Args:
            initial_window: Size of initial training window
            step_size: Step size for walking forward
            validation_size: Size of validation window
            embargo: Embargo period
            purge: Purge period
        """
        self.initial_window = initial_window
        self.step_size = step_size
        self.validation_size = validation_size
        self.embargo = embargo
        self.purge = purge
        
    def split(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate walk-forward splits.
        
        Args:
            X: Features DataFrame with datetime index
            y: Labels Series (optional)
            
        Yields:
            Tuple of (train_indices, val_indices)
        """
        if not isinstance(X.index, pd.DatetimeIndex):
            raise ValueError("X must have DatetimeIndex for walk-forward validation")
            
        n_samples = len(X)
        
        # Start from initial window
        current_pos = self.initial_window
        fold = 0
        
        while current_pos + self.validation_size + self.embargo < n_samples:
            # Training window
            train_start = max(0, current_pos - self.initial_window - self.step_size * fold)
            train_end = current_pos - self.purge
            
            # Validation window
            val_start = current_pos + self.embargo
            val_end = val_start + self.validation_size
            
            # Get indices
            train_indices = X.iloc[train_start:train_end].index
            val_indices = X.iloc[val_start:val_end].index
            
            # Convert to positional indices
            train_pos = X.index.get_indexer(train_indices)
            val_pos = X.index.get_indexer(val_indices)
            
            # Skip if windows are too small
            if len(train_pos) < 50 or len(val_pos) < 5:
                current_pos += self.step_size
                fold += 1
                continue
                
            logger.info(f"Walk-forward fold {fold}: "
                       f"Train {X.index[train_pos[0]]} to {X.index[train_pos[-1]]} ({len(train_pos)} samples), "
                       f"Val {X.index[val_pos[0]]} to {X.index[val_pos[-1]]} ({len(val_pos)} samples)")
            
            yield train_pos, val_pos
            
            # Move forward
            current_pos += self.step_size
            fold += 1


def validate_model_temporal(model,
                           X: pd.DataFrame,
                           y: pd.Series,
                           cv_method: str = 'purged_kfold',
                           n_splits: int = 5,
                           **cv_kwargs) -> Tuple[List[float], List[np.ndarray], List[np.ndarray]]:
    """
    Validate model using temporal cross-validation.
    
    Args:
        model: Model with fit/predict_proba methods
        X: Features DataFrame
        y: Labels Series
        cv_method: 'purged_kfold' or 'walk_forward'
        n_splits: Number of splits (for purged_kfold)
        **cv_kwargs: Additional arguments for cross-validator
        
    Returns:
        Tuple of (scores, predictions, true_labels) for each fold
    """
    if cv_method == 'purged_kfold':
        cv = PurgedKFold(n_splits=n_splits, **cv_kwargs)
    elif cv_method == 'walk_forward':
        cv = WalkForwardValidator(**cv_kwargs)
    else:
        raise ValueError(f"Unknown cv_method: {cv_method}")
    
    fold_scores = []
    fold_predictions = []
    fold_labels = []
    
    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
        # Get data for this fold
        X_train = X.iloc[train_idx]
        y_train = y.iloc[train_idx]
        X_val = X.iloc[val_idx]
        y_val = y.iloc[val_idx]
        
        # Fit model
        model_copy = model.__class__(model.config)
        model_copy.fit(X_train, y_train, X_val, y_val)
        
        # Predict
        y_pred_proba = model_copy.predict_proba(X_val)[:, 1]
        
        # Store results
        fold_predictions.append(y_pred_proba)
        fold_labels.append(y_val.values)
        
        # Calculate fold score (using log loss as default)
        from sklearn.metrics import log_loss
        score = log_loss(y_val, y_pred_proba)
        fold_scores.append(score)
        
        logger.info(f"Fold {fold}: Score = {score:.4f}")
    
    logger.info(f"Cross-validation complete. Mean score: {np.mean(fold_scores):.4f} ± {np.std(fold_scores):.4f}")
    
    return fold_scores, fold_predictions, fold_labels


def embargo_indices(indices: np.ndarray, 
                   val_start: int, 
                   val_end: int, 
                   embargo_size: int) -> np.ndarray:
    """
    Apply embargo to training indices.
    
    Args:
        indices: All available indices
        val_start: Start of validation period
        val_end: End of validation period
        embargo_size: Size of embargo period
        
    Returns:
        Filtered training indices with embargo applied
    """
    # Remove indices within embargo period
    embargo_mask = (indices < val_start - embargo_size) | (indices > val_end + embargo_size)
    return indices[embargo_mask]


def purge_indices(indices: np.ndarray,
                 val_indices: np.ndarray,
                 purge_size: int) -> np.ndarray:
    """
    Purge training indices that might be influenced by validation set.
    
    Args:
        indices: Training indices
        val_indices: Validation indices
        purge_size: Size of purge period
        
    Returns:
        Purged training indices
    """
    # For each validation index, remove nearby training indices
    purged_indices = indices.copy()
    
    for val_idx in val_indices:
        # Remove indices within purge distance
        purge_mask = np.abs(indices - val_idx) > purge_size
        purged_indices = purged_indices[purge_mask]
        indices = purged_indices  # Update for next iteration
    
    return purged_indices