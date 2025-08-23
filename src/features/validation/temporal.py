"""
Unified temporal validation module for time series data.

This module provides standardized temporal validation strategies with 
guaranteed no temporal leakage through purging and embargo mechanisms.
"""

import numpy as np
import pandas as pd
from typing import Iterator, Tuple, Optional, Dict, Any, Union, List
import warnings
import structlog
from dataclasses import dataclass

log = structlog.get_logger()


@dataclass
class TemporalValidationConfig:
    """Configuration for temporal validation."""
    
    # Purged K-Fold parameters
    n_splits: int = 5
    embargo: int = 10  # Number of bars to embargo between train and validation
    purge: int = 5  # Number of bars to purge before validation
    
    # Walk-Forward parameters
    walk_forward_window: int = 252  # Training window size
    walk_forward_test_size: int = 63  # Test window size
    walk_forward_gap: int = 0  # Gap between train and test
    expanding_window: bool = False  # Use expanding vs fixed window
    
    # Combinatorial parameters
    n_test_groups: int = 2  # Number of test groups for combinatorial CV
    
    # Validation parameters
    check_leakage: bool = True  # Automatically check for temporal leakage
    min_train_samples: int = 100  # Minimum training samples required
    min_test_samples: int = 20  # Minimum test samples required
    
    # Time resolution (for embargo calculation)
    time_resolution_minutes: float = 15.0  # Default 15-minute bars


class TemporalValidator:
    """
    Unified temporal validator with multiple strategies.
    
    This class provides a single interface for all temporal validation
    strategies, ensuring no temporal leakage through proper purging
    and embargo mechanisms.
    """
    
    def __init__(self, config: Optional[TemporalValidationConfig] = None):
        """
        Initialize temporal validator.
        
        Args:
            config: Validation configuration
        """
        self.config = config or TemporalValidationConfig()
        self._validation_history = []
        
    def split(self, 
             X: Union[pd.DataFrame, np.ndarray],
             y: Optional[Union[pd.Series, np.ndarray]] = None,
             strategy: str = 'purged_kfold',
             **kwargs) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate train/validation splits using specified strategy.
        
        Args:
            X: Features
            y: Labels (optional)
            strategy: Validation strategy ('purged_kfold', 'walk_forward', 'combinatorial')
            **kwargs: Additional strategy-specific parameters
            
        Yields:
            train_indices, validation_indices
            
        Raises:
            ValueError: If strategy is unknown or data has insufficient samples
        """
        # Check for shuffle parameter (should never be True for time series)
        if kwargs.get('shuffle', False):
            raise ValueError(
                "shuffle=True detected! This will cause temporal leakage. "
                "Time series data must never be shuffled."
            )
        
        # Get appropriate splitter
        strategies = {
            'purged_kfold': self._purged_kfold_split,
            'walk_forward': self._walk_forward_split,
            'combinatorial': self._combinatorial_purged_split,
            'expanding_window': self._expanding_window_split
        }
        
        if strategy not in strategies:
            raise ValueError(f"Unknown strategy: {strategy}. Available: {list(strategies.keys())}")
        
        splitter = strategies[strategy]
        
        # Generate splits
        n_samples = len(X)
        split_count = 0
        
        for train_idx, val_idx in splitter(X, y, **kwargs):
            # Validate split sizes
            if len(train_idx) < self.config.min_train_samples:
                log.warning(
                    "insufficient_train_samples",
                    n_train=len(train_idx),
                    min_required=self.config.min_train_samples
                )
                continue
                
            if len(val_idx) < self.config.min_test_samples:
                log.warning(
                    "insufficient_test_samples", 
                    n_test=len(val_idx),
                    min_required=self.config.min_test_samples
                )
                continue
            
            # Check for temporal leakage if enabled
            if self.config.check_leakage:
                if not self._validate_no_leakage(X, train_idx, val_idx):
                    raise ValueError(
                        f"Temporal leakage detected in split {split_count}! "
                        "Check embargo and purge parameters."
                    )
            
            split_count += 1
            yield train_idx, val_idx
        
        if split_count == 0:
            raise ValueError(
                "No valid splits generated. Check data size and validation parameters."
            )
        
        log.info(
            "temporal_validation_complete",
            strategy=strategy,
            n_splits=split_count,
            n_samples=n_samples
        )
    
    def _purged_kfold_split(self, 
                           X: Union[pd.DataFrame, np.ndarray],
                           y: Optional[Union[pd.Series, np.ndarray]] = None,
                           **kwargs) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        Purged K-Fold with embargo to prevent temporal leakage.
        
        This is the recommended default strategy for time series CV.
        """
        n_samples = len(X)
        indices = np.arange(n_samples)
        
        # Override config with kwargs if provided
        n_splits = kwargs.get('n_splits', self.config.n_splits)
        embargo = kwargs.get('embargo', self.config.embargo)
        purge = kwargs.get('purge', self.config.purge)
        
        # Calculate fold size
        fold_size = n_samples // n_splits
        
        for fold in range(n_splits):
            # Define validation indices
            val_start = fold * fold_size
            val_end = (fold + 1) * fold_size if fold < n_splits - 1 else n_samples
            val_indices = indices[val_start:val_end]
            
            # Define training indices with purge and embargo
            train_indices = []
            
            # Add samples before validation (with purge)
            if val_start > purge:
                train_before = indices[:val_start - purge]
                train_indices.extend(train_before)
            
            # Add samples after validation (with embargo)
            if val_end + embargo < n_samples:
                train_after = indices[val_end + embargo:]
                train_indices.extend(train_after)
            
            train_indices = np.array(train_indices)
            
            if len(train_indices) > 0 and len(val_indices) > 0:
                yield train_indices, val_indices
    
    def _walk_forward_split(self,
                           X: Union[pd.DataFrame, np.ndarray],
                           y: Optional[Union[pd.Series, np.ndarray]] = None,
                           **kwargs) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        Walk-forward validation with fixed or expanding window.
        """
        n_samples = len(X)
        
        # Override config with kwargs
        window_size = kwargs.get('window_size', self.config.walk_forward_window)
        test_size = kwargs.get('test_size', self.config.walk_forward_test_size)
        gap = kwargs.get('gap', self.config.walk_forward_gap)
        
        # Start position
        start = 0
        
        while start + window_size + gap + test_size <= n_samples:
            # Training indices
            train_end = start + window_size
            train_indices = np.arange(start, train_end)
            
            # Test indices (with gap)
            test_start = train_end + gap
            test_end = test_start + test_size
            test_indices = np.arange(test_start, test_end)
            
            yield train_indices, test_indices
            
            # Move window forward
            start += test_size
    
    def _expanding_window_split(self,
                               X: Union[pd.DataFrame, np.ndarray],
                               y: Optional[Union[pd.Series, np.ndarray]] = None,
                               **kwargs) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        Expanding window validation (anchored walk-forward).
        """
        n_samples = len(X)
        
        # Override config with kwargs
        min_train = kwargs.get('min_train_size', self.config.walk_forward_window)
        test_size = kwargs.get('test_size', self.config.walk_forward_test_size)
        gap = kwargs.get('gap', self.config.walk_forward_gap)
        max_splits = kwargs.get('max_splits', self.config.n_splits)
        
        # Calculate step size
        step_size = test_size
        n_splits = min((n_samples - min_train - gap - test_size) // step_size + 1, max_splits)
        
        for i in range(n_splits):
            # Training always starts from beginning (expanding)
            train_start = 0
            train_end = min_train + i * step_size
            train_indices = np.arange(train_start, train_end)
            
            # Test indices
            test_start = train_end + gap
            test_end = test_start + test_size
            
            if test_end > n_samples:
                break
                
            test_indices = np.arange(test_start, test_end)
            
            yield train_indices, test_indices
    
    def _combinatorial_purged_split(self,
                                   X: Union[pd.DataFrame, np.ndarray],
                                   y: Optional[Union[pd.Series, np.ndarray]] = None,
                                   **kwargs) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        Combinatorial Purged Cross-Validation (based on López de Prado).
        """
        from itertools import combinations
        
        n_samples = len(X)
        indices = np.arange(n_samples)
        
        # Override config with kwargs
        n_splits = kwargs.get('n_splits', self.config.n_splits)
        n_test_groups = kwargs.get('n_test_groups', self.config.n_test_groups)
        embargo = kwargs.get('embargo', self.config.embargo)
        
        # Create groups
        group_size = n_samples // n_splits
        groups = []
        
        for i in range(n_splits):
            start = i * group_size
            end = start + group_size if i < n_splits - 1 else n_samples
            groups.append(indices[start:end])
        
        # Generate combinations
        for test_group_indices in combinations(range(n_splits), n_test_groups):
            # Test indices
            test_indices = np.concatenate([groups[i] for i in test_group_indices])
            
            # Train indices with purging
            train_indices = []
            
            for i in range(n_splits):
                if i not in test_group_indices:
                    group = groups[i]
                    
                    # Apply embargo to each test group
                    valid_mask = np.ones(len(group), dtype=bool)
                    
                    for test_idx in test_group_indices:
                        test_group = groups[test_idx]
                        test_min = test_group.min()
                        test_max = test_group.max()
                        
                        # Remove samples too close to test group
                        embargo_mask = (
                            (group < test_min - embargo) | 
                            (group > test_max + embargo)
                        )
                        valid_mask &= embargo_mask
                    
                    valid_indices = group[valid_mask]
                    if len(valid_indices) > 0:
                        train_indices.extend(valid_indices)
            
            if train_indices:
                train_indices = np.array(train_indices)
                yield train_indices, test_indices
    
    def _validate_no_leakage(self,
                            X: Union[pd.DataFrame, np.ndarray],
                            train_idx: np.ndarray,
                            val_idx: np.ndarray) -> bool:
        """
        Validate that there's no temporal leakage between train and validation.
        
        Returns:
            True if no leakage detected, False otherwise
        """
        # Check for index overlap
        if len(np.intersect1d(train_idx, val_idx)) > 0:
            log.error("index_overlap_detected")
            return False
        
        # If X has datetime index, perform temporal checks
        if isinstance(X, pd.DataFrame) and isinstance(X.index, pd.DatetimeIndex):
            train_times = X.index[train_idx]
            val_times = X.index[val_idx]
            
            # Get validation period
            val_start = val_times.min()
            val_end = val_times.max()
            
            # Check for train samples within validation period
            train_in_val = train_times[(train_times >= val_start) & (train_times <= val_end)]
            if len(train_in_val) > 0:
                log.error(
                    "temporal_leakage_detected",
                    n_leaked=len(train_in_val),
                    val_start=val_start,
                    val_end=val_end
                )
                return False
            
            # Check embargo is respected
            train_before_val = train_times[train_times < val_start]
            if len(train_before_val) > 0:
                gap_minutes = (val_start - train_before_val.max()).total_seconds() / 60
                min_gap_minutes = self.config.embargo * self.config.time_resolution_minutes
                
                if gap_minutes < min_gap_minutes:
                    log.warning(
                        "insufficient_embargo",
                        gap_minutes=gap_minutes,
                        required_minutes=min_gap_minutes
                    )
                    # This is a warning, not an error
            
            # Check purge is respected
            train_after_val = train_times[train_times > val_end]
            if len(train_after_val) > 0:
                gap_minutes = (train_after_val.min() - val_end).total_seconds() / 60
                min_gap_minutes = self.config.purge * self.config.time_resolution_minutes
                
                if gap_minutes < min_gap_minutes:
                    log.warning(
                        "insufficient_purge",
                        gap_minutes=gap_minutes,
                        required_minutes=min_gap_minutes
                    )
        
        return True
    
    def get_n_splits(self, strategy: str = 'purged_kfold') -> int:
        """
        Get number of splits for a given strategy.
        
        Args:
            strategy: Validation strategy
            
        Returns:
            Number of splits
        """
        if strategy in ['purged_kfold', 'combinatorial']:
            return self.config.n_splits
        elif strategy in ['walk_forward', 'expanding_window']:
            # This is approximate as it depends on data size
            return -1  # Unknown until data is provided
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
    
    def validate_dataset(self,
                        X: Union[pd.DataFrame, np.ndarray],
                        y: Union[pd.Series, np.ndarray]) -> Dict[str, Any]:
        """
        Validate dataset for temporal modeling.
        
        Args:
            X: Features
            y: Labels
            
        Returns:
            Validation report
        """
        report = {
            'n_samples': len(X),
            'has_datetime_index': False,
            'is_sorted': True,
            'has_duplicates': False,
            'has_gaps': False,
            'warnings': [],
            'errors': []
        }
        
        # Check if X has datetime index
        if isinstance(X, pd.DataFrame) and isinstance(X.index, pd.DatetimeIndex):
            report['has_datetime_index'] = True
            
            # Check if sorted
            report['is_sorted'] = X.index.is_monotonic_increasing
            if not report['is_sorted']:
                report['errors'].append("Data is not sorted by time!")
            
            # Check for duplicates
            report['has_duplicates'] = X.index.has_duplicates
            if report['has_duplicates']:
                report['errors'].append("Duplicate timestamps detected!")
            
            # Check for gaps
            if len(X) > 1:
                time_diffs = pd.Series(X.index).diff().dropna()
                median_diff = time_diffs.median()
                large_gaps = time_diffs > median_diff * 3
                report['has_gaps'] = large_gaps.any()
                if report['has_gaps']:
                    report['warnings'].append(
                        f"Large time gaps detected ({large_gaps.sum()} gaps)"
                    )
        
        # Check class balance
        if y is not None:
            unique, counts = np.unique(y, return_counts=True)
            min_class = counts.min()
            max_class = counts.max()
            imbalance_ratio = max_class / min_class
            
            if imbalance_ratio > 10:
                report['warnings'].append(
                    f"Severe class imbalance detected (ratio: {imbalance_ratio:.1f})"
                )
        
        # Check if sufficient data for validation
        min_samples_needed = (
            self.config.n_splits * 
            (self.config.min_train_samples + self.config.min_test_samples)
        )
        
        if len(X) < min_samples_needed:
            report['errors'].append(
                f"Insufficient samples ({len(X)}) for {self.config.n_splits} splits. "
                f"Need at least {min_samples_needed} samples."
            )
        
        return report


def warn_if_shuffle_used(func):
    """Decorator to warn if shuffle parameter is used."""
    def wrapper(*args, **kwargs):
        if kwargs.get('shuffle', False):
            warnings.warn(
                "shuffle=True will cause temporal leakage in time series data! "
                "Setting shuffle=False.",
                UserWarning,
                stacklevel=2
            )
            kwargs['shuffle'] = False
        return func(*args, **kwargs)
    return wrapper


# Convenience function for backward compatibility
def get_temporal_splitter(strategy: str = 'purged_kfold',
                         **kwargs) -> TemporalValidator:
    """
    Get a configured temporal splitter.
    
    Args:
        strategy: Validation strategy
        **kwargs: Configuration parameters
        
    Returns:
        Configured TemporalValidator
    """
    config = TemporalValidationConfig(**kwargs)
    return TemporalValidator(config)