"""
Unified validation module for time series cross-validation.
Consolidates all validation strategies in one place.

This module replaces:
- src/validation/walkforward.py
- src/models/validation/walkforward.py
- src/models/xgb/validators.py
- src/features/validation/temporal.py (validation parts)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterator, Tuple, Optional, Union, List
import numpy as np
import pandas as pd
import warnings
from sklearn.model_selection import BaseCrossValidator


@dataclass
class ValidationConfig:
    """Unified configuration for all validation strategies."""
    
    # Basic settings
    n_splits: int = 5
    strategy: str = 'walkforward'  # 'walkforward', 'purged_kfold', 'time_series_cv'
    
    # Temporal settings
    embargo: int = 10  # Embargo period between train and test
    purge: int = 5  # Additional purge for feature lookahead
    gap: int = 0  # Gap between train and validation
    
    # Window settings
    min_train_size: int = 252  # Minimum training samples
    max_train_size: Optional[int] = None  # Maximum training size (for sliding window)
    test_size: Optional[int] = None  # Fixed test size
    
    # Walk-forward specific
    anchored: bool = True  # True for expanding window, False for sliding
    
    # Purged k-fold specific
    embargo_pct: float = 0.01  # Embargo as percentage of total samples
    purge_pct: float = 0.01  # Purge as percentage of total samples
    
    def __post_init__(self):
        """Validate configuration."""
        if self.n_splits < 2:
            raise ValueError("n_splits must be at least 2")
        if self.embargo < 0 or self.purge < 0 or self.gap < 0:
            raise ValueError("embargo, purge, and gap must be non-negative")
        if self.min_train_size <= 0:
            raise ValueError("min_train_size must be positive")
        if not self.anchored and self.max_train_size is None:
            warnings.warn("Sliding window without max_train_size will use all available data")


class BaseValidator(ABC):
    """
    Abstract base class for all validation strategies.
    
    Provides common functionality and interface for time series validation.
    """
    
    def __init__(self, config: ValidationConfig):
        """Initialize validator with configuration."""
        self.config = config
        self.n_splits = config.n_splits
        
    @abstractmethod
    def split(self, X, y=None, groups=None) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate train/test indices.
        
        Args:
            X: Features array or DataFrame
            y: Labels (optional)
            groups: Group labels for grouped CV (optional)
            
        Yields:
            Tuple of (train_indices, test_indices)
        """
        pass
    
    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        """Return the number of splits."""
        return self.n_splits
    
    def validate_temporal_integrity(
        self, 
        train_idx: np.ndarray, 
        test_idx: np.ndarray,
        timestamps: Optional[pd.DatetimeIndex] = None
    ) -> bool:
        """
        Validate that there's no temporal leakage.
        
        Args:
            train_idx: Training indices
            test_idx: Test indices
            timestamps: Optional timestamps for validation
            
        Returns:
            True if valid, raises ValueError otherwise
        """
        # Check basic ordering
        if len(train_idx) > 0 and len(test_idx) > 0:
            if train_idx.max() >= test_idx.min():
                raise ValueError(
                    f"Temporal leakage detected! "
                    f"Max train index {train_idx.max()} >= Min test index {test_idx.min()}"
                )
        
        # Check embargo if timestamps provided
        if timestamps is not None and len(train_idx) > 0 and len(test_idx) > 0:
            train_end_time = timestamps[train_idx.max()]
            test_start_time = timestamps[test_idx.min()]
            
            # Calculate embargo in same units as timestamps
            if hasattr(test_start_time - train_end_time, 'days'):
                actual_embargo = (test_start_time - train_end_time).days
                if actual_embargo < self.config.embargo:
                    warnings.warn(
                        f"Embargo period violated! "
                        f"Required: {self.config.embargo} days, Actual: {actual_embargo} days"
                    )
        
        return True
    
    def _get_n_samples(self, X) -> int:
        """Get number of samples from X."""
        if hasattr(X, 'shape'):
            return X.shape[0]
        elif hasattr(X, '__len__'):
            return len(X)
        else:
            return int(X)


class WalkForwardValidator(BaseValidator):
    """
    Walk-forward validation for time series.
    
    Implements both expanding window (anchored) and sliding window strategies.
    
    Example:
        >>> config = ValidationConfig(n_splits=5, embargo=10, anchored=True)
        >>> validator = WalkForwardValidator(config)
        >>> for train_idx, test_idx in validator.split(X):
        ...     X_train, X_test = X[train_idx], X[test_idx]
        ...     # Train and evaluate model
    """
    
    def split(self, X, y=None, groups=None) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """Generate walk-forward splits with embargo."""
        n_samples = self._get_n_samples(X)
        indices = np.arange(n_samples)
        
        # Calculate test size if not specified
        if self.config.test_size is None:
            # Use remaining data after training for test, divided by number of splits
            available_for_test = n_samples - self.config.min_train_size
            test_size = available_for_test // (self.n_splits + 1)
        else:
            test_size = self.config.test_size
        
        # Generate splits
        for i in range(self.n_splits):
            # Calculate test window
            test_end = n_samples - i * test_size
            test_start = max(test_end - test_size, self.config.min_train_size + self.config.embargo)
            
            if test_start >= test_end:
                continue
            
            # Calculate train window with embargo and purge
            train_end = test_start - self.config.embargo - self.config.gap
            
            if self.config.anchored:
                # Expanding window: always start from beginning
                train_start = 0
            else:
                # Sliding window: use max_train_size if specified
                if self.config.max_train_size:
                    train_start = max(0, train_end - self.config.max_train_size)
                else:
                    train_start = 0
            
            # Apply purge to remove potentially leaked samples
            if self.config.purge > 0:
                train_end = train_end - self.config.purge
            
            # Check minimum training size
            if train_end - train_start < self.config.min_train_size:
                continue
            
            # Generate indices
            train_idx = indices[train_start:train_end]
            test_idx = indices[test_start:test_end]
            
            # Validate temporal integrity
            self.validate_temporal_integrity(train_idx, test_idx)
            
            yield train_idx, test_idx


class PurgedKFoldValidator(BaseValidator, BaseCrossValidator):
    """
    Purged K-Fold cross-validation for time series.
    
    Implements the purged k-fold strategy from "Advances in Financial Machine Learning"
    by Marcos López de Prado.
    
    This ensures no data leakage by:
    1. Purging training samples that could be influenced by the validation set
    2. Adding embargo periods after each validation set
    
    Example:
        >>> config = ValidationConfig(
        ...     n_splits=5, 
        ...     strategy='purged_kfold',
        ...     embargo_pct=0.01,
        ...     purge_pct=0.01
        ... )
        >>> validator = PurgedKFoldValidator(config)
        >>> for train_idx, val_idx in validator.split(X):
        ...     # Train and validate model
    """
    
    def split(self, X, y=None, groups=None) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """Generate purged k-fold splits."""
        n_samples = self._get_n_samples(X)
        indices = np.arange(n_samples)
        
        # Calculate embargo and purge sizes
        embargo_size = int(n_samples * self.config.embargo_pct)
        purge_size = int(n_samples * self.config.purge_pct)
        
        # Calculate fold sizes
        fold_size = n_samples // self.n_splits
        
        for fold in range(self.n_splits):
            # Define test fold boundaries
            test_start = fold * fold_size
            test_end = (fold + 1) * fold_size if fold < self.n_splits - 1 else n_samples
            
            # Create test indices
            test_idx = indices[test_start:test_end]
            
            # Create training indices with purging and embargo
            train_idx = []
            
            # Add samples before test fold (with purge)
            if test_start > purge_size:
                train_idx.extend(indices[:test_start - purge_size])
            
            # Add samples after test fold (with embargo)
            if test_end + embargo_size < n_samples:
                train_idx.extend(indices[test_end + embargo_size:])
            
            train_idx = np.array(train_idx)
            
            # Skip if insufficient training data
            if len(train_idx) < self.config.min_train_size:
                continue
            
            yield train_idx, test_idx


class TimeSeriesCV(BaseValidator):
    """
    Standard time series cross-validation.
    
    Similar to sklearn's TimeSeriesSplit but with embargo and purge support.
    
    Example:
        >>> config = ValidationConfig(
        ...     n_splits=5,
        ...     strategy='time_series_cv',
        ...     embargo=10
        ... )
        >>> validator = TimeSeriesCV(config)
        >>> for train_idx, test_idx in validator.split(X):
        ...     # Train and test model
    """
    
    def split(self, X, y=None, groups=None) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """Generate time series CV splits."""
        n_samples = self._get_n_samples(X)
        indices = np.arange(n_samples)
        
        # Calculate sizes
        n_splits = self.n_splits
        n_folds = n_splits + 1
        
        # Calculate test size
        if self.config.test_size:
            test_size = self.config.test_size
        else:
            test_size = (n_samples - self.config.min_train_size) // n_folds
        
        # Generate splits
        for i in range(n_splits):
            # Training set grows with each split
            train_end = self.config.min_train_size + i * test_size
            
            # Apply embargo
            test_start = train_end + self.config.embargo + self.config.gap
            test_end = test_start + test_size
            
            # Check bounds
            if test_end > n_samples:
                test_end = n_samples
            
            if test_start >= test_end:
                continue
            
            # Apply purge
            if self.config.purge > 0:
                train_end = train_end - self.config.purge
            
            train_idx = indices[:train_end]
            test_idx = indices[test_start:test_end]
            
            # Validate
            self.validate_temporal_integrity(train_idx, test_idx)
            
            yield train_idx, test_idx


def create_validator(
    strategy: str = 'walkforward',
    n_splits: int = 5,
    embargo: int = 10,
    **kwargs
) -> BaseValidator:
    """
    Factory function to create appropriate validator.
    
    Args:
        strategy: Validation strategy ('walkforward', 'purged_kfold', 'time_series_cv')
        n_splits: Number of splits
        embargo: Embargo period between train and test
        **kwargs: Additional configuration parameters
        
    Returns:
        Validator instance
        
    Example:
        >>> validator = create_validator('walkforward', n_splits=5, embargo=10)
        >>> for train_idx, test_idx in validator.split(X):
        ...     # Use indices for training and testing
    """
    # Map strategy names to classes
    validators = {
        'walkforward': WalkForwardValidator,
        'walk_forward': WalkForwardValidator,
        'purged_kfold': PurgedKFoldValidator,
        'purged': PurgedKFoldValidator,
        'time_series_cv': TimeSeriesCV,
        'timeseries': TimeSeriesCV,
        'ts_cv': TimeSeriesCV,
    }
    
    # Get validator class
    validator_class = validators.get(strategy.lower())
    if not validator_class:
        raise ValueError(
            f"Unknown validation strategy: {strategy}. "
            f"Available: {list(validators.keys())}"
        )
    
    # Create configuration
    config = ValidationConfig(
        strategy=strategy,
        n_splits=n_splits,
        embargo=embargo,
        **kwargs
    )
    
    # Return validator instance
    return validator_class(config)


# Convenience functions for common use cases
def walk_forward_split(X, n_splits=5, embargo=10, anchored=True, **kwargs):
    """Convenience function for walk-forward validation."""
    validator = create_validator(
        'walkforward',
        n_splits=n_splits,
        embargo=embargo,
        anchored=anchored,
        **kwargs
    )
    return validator.split(X)


def purged_kfold_split(X, n_splits=5, embargo_pct=0.01, purge_pct=0.01, **kwargs):
    """Convenience function for purged k-fold validation."""
    validator = create_validator(
        'purged_kfold',
        n_splits=n_splits,
        embargo_pct=embargo_pct,
        purge_pct=purge_pct,
        **kwargs
    )
    return validator.split(X)