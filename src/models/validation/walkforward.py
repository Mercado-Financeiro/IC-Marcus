"""
Walk-forward validation for time series with proper embargo.

Implements anchored and sliding window walk-forward validation strategies
for financial time series to prevent data leakage.

References:
- "Advances in Financial Machine Learning" (López de Prado, 2018)
- "Machine Learning for Asset Managers" (López de Prado, 2020)
"""

import numpy as np
import pandas as pd
from typing import Generator, Tuple, Optional, Union, List
from dataclasses import dataclass
import warnings


@dataclass
class WalkForwardConfig:
    """Configuration for walk-forward validation."""
    
    n_splits: int = 5
    train_period: Optional[int] = None  # None for expanding window
    val_period: int = 252  # Default 1 year for daily data
    test_period: int = 63  # Default 3 months for daily data
    embargo: int = 5  # Embargo between train/val and val/test
    purge: int = 0  # Additional purge for feature lookahead
    gap: int = 0  # Gap between train and validation
    anchored: bool = True  # True for expanding, False for sliding
    min_train_size: int = 252  # Minimum training samples
    
    def __post_init__(self):
        """Validate configuration."""
        if self.n_splits < 2:
            raise ValueError("n_splits must be at least 2")
        if self.embargo < 0:
            raise ValueError("embargo must be non-negative")
        if self.val_period <= 0 or self.test_period <= 0:
            raise ValueError("val_period and test_period must be positive")
        if not self.anchored and self.train_period is None:
            raise ValueError("train_period must be specified for sliding window")


class WalkForwardValidator:
    """
    Walk-forward validator for time series with embargo.
    
    Provides both anchored (expanding) and sliding window strategies
    with proper embargo to prevent information leakage.
    """
    
    def __init__(self, config: Optional[WalkForwardConfig] = None):
        """
        Initialize walk-forward validator.
        
        Args:
            config: Configuration object
        """
        self.config = config or WalkForwardConfig()
        
    def split(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Optional[Union[pd.Series, np.ndarray]] = None,
        groups: Optional[np.ndarray] = None
    ) -> Generator[Tuple[np.ndarray, np.ndarray, np.ndarray], None, None]:
        """
        Generate train/validation/test indices for walk-forward validation.
        
        Args:
            X: Features
            y: Labels (optional, for compatibility)
            groups: Groups (not used, for sklearn compatibility)
            
        Yields:
            Tuples of (train_indices, val_indices, test_indices)
        """
        n_samples = len(X)
        
        # Calculate total period per split
        period_per_split = self.config.val_period + self.config.test_period + \
                          2 * self.config.embargo + self.config.gap
        
        # Determine split points
        if self.config.anchored:
            # Expanding window (anchored)
            train_start = 0
            
            # Ensure minimum training size
            min_start = self.config.min_train_size + self.config.embargo + \
                       self.config.gap + self.config.val_period + \
                       self.config.embargo + self.config.test_period
            
            if n_samples < min_start:
                raise ValueError(f"Not enough samples. Need at least {min_start}, got {n_samples}")
            
            # Calculate test end points
            test_ends = []
            for i in range(self.config.n_splits):
                test_end = n_samples - (self.config.n_splits - i - 1) * period_per_split
                test_ends.append(min(test_end, n_samples))
            
            for i, test_end in enumerate(test_ends):
                # Test period
                test_start = test_end - self.config.test_period
                test_indices = np.arange(test_start, test_end)
                
                # Validation period (with embargo before test)
                val_end = test_start - self.config.embargo
                val_start = val_end - self.config.val_period
                val_indices = np.arange(val_start, val_end)
                
                # Training period (with embargo/gap before validation)
                train_end = val_start - self.config.embargo - self.config.gap
                
                # Apply purge if specified
                if self.config.purge > 0:
                    train_end -= self.config.purge
                
                # Ensure minimum training size
                if train_end - train_start < self.config.min_train_size:
                    warnings.warn(f"Split {i}: Training size {train_end - train_start} "
                                f"is less than minimum {self.config.min_train_size}")
                    continue
                
                train_indices = np.arange(train_start, train_end)
                
                yield train_indices, val_indices, test_indices
                
        else:
            # Sliding window
            if self.config.train_period is None:
                raise ValueError("train_period must be specified for sliding window")
            
            window_size = self.config.train_period + self.config.embargo + \
                         self.config.gap + self.config.val_period + \
                         self.config.embargo + self.config.test_period
            
            if n_samples < window_size:
                raise ValueError(f"Not enough samples for sliding window. "
                               f"Need at least {window_size}, got {n_samples}")
            
            # Calculate step size
            step_size = (n_samples - window_size) // (self.config.n_splits - 1)
            
            for i in range(self.config.n_splits):
                window_start = i * step_size
                
                # Training period
                train_start = window_start
                train_end = train_start + self.config.train_period
                
                # Apply purge
                if self.config.purge > 0:
                    train_end -= self.config.purge
                
                train_indices = np.arange(train_start, train_end)
                
                # Validation period (with embargo/gap after train)
                val_start = train_end + self.config.embargo + self.config.gap
                val_end = val_start + self.config.val_period
                val_indices = np.arange(val_start, val_end)
                
                # Test period (with embargo after validation)
                test_start = val_end + self.config.embargo
                test_end = test_start + self.config.test_period
                
                # Ensure we don't exceed data bounds
                if test_end > n_samples:
                    if i == self.config.n_splits - 1:
                        # Adjust last split
                        test_end = n_samples
                        test_indices = np.arange(test_start, test_end)
                    else:
                        continue
                else:
                    test_indices = np.arange(test_start, test_end)
                
                yield train_indices, val_indices, test_indices
    
    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        """Get number of splits (sklearn compatibility)."""
        return self.config.n_splits


class OuterWalkForward:
    """
    Outer walk-forward validation for nested cross-validation.
    
    Used for model selection with proper time series validation.
    """
    
    def __init__(
        self,
        n_outer_splits: int = 3,
        n_inner_splits: int = 5,
        outer_config: Optional[WalkForwardConfig] = None,
        inner_config: Optional[WalkForwardConfig] = None
    ):
        """
        Initialize nested walk-forward validator.
        
        Args:
            n_outer_splits: Number of outer splits for model selection
            n_inner_splits: Number of inner splits for hyperparameter tuning
            outer_config: Configuration for outer loop
            inner_config: Configuration for inner loop
        """
        self.outer_config = outer_config or WalkForwardConfig(n_splits=n_outer_splits)
        self.inner_config = inner_config or WalkForwardConfig(n_splits=n_inner_splits)
        
        self.outer_validator = WalkForwardValidator(self.outer_config)
        
    def split(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Optional[Union[pd.Series, np.ndarray]] = None
    ) -> Generator[Tuple[np.ndarray, np.ndarray, WalkForwardValidator], None, None]:
        """
        Generate outer splits with inner validator.
        
        Args:
            X: Features
            y: Labels
            
        Yields:
            Tuples of (development_indices, test_indices, inner_validator)
        """
        for train_idx, val_idx, test_idx in self.outer_validator.split(X, y):
            # Combine train and validation for development set
            dev_idx = np.concatenate([train_idx, val_idx])
            
            # Create inner validator for this development set
            inner_validator = WalkForwardValidator(self.inner_config)
            
            yield dev_idx, test_idx, inner_validator


class BlockingTimeSeriesSplit:
    """
    Time series split with blocking to prevent leakage.
    
    Similar to sklearn's TimeSeriesSplit but with embargo/purge.
    """
    
    def __init__(
        self,
        n_splits: int = 5,
        embargo: int = 0,
        purge: int = 0,
        test_size: Optional[int] = None,
        gap: int = 0
    ):
        """
        Initialize blocking time series split.
        
        Args:
            n_splits: Number of splits
            embargo: Embargo period after each fold
            purge: Purge period before each fold
            test_size: Fixed test size (None for variable)
            gap: Gap between train and test
        """
        self.n_splits = n_splits
        self.embargo = embargo
        self.purge = purge
        self.test_size = test_size
        self.gap = gap
        
    def split(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Optional[Union[pd.Series, np.ndarray]] = None,
        groups: Optional[np.ndarray] = None
    ) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """
        Generate train/test indices.
        
        Args:
            X: Features
            y: Labels
            groups: Groups (not used)
            
        Yields:
            Tuples of (train_indices, test_indices)
        """
        n_samples = len(X)
        
        if self.test_size is None:
            # Variable test size
            test_size = n_samples // (self.n_splits + 1)
        else:
            test_size = self.test_size
        
        for i in range(self.n_splits):
            # Calculate test end
            test_end = n_samples - i * test_size
            test_start = test_end - test_size
            
            # Apply embargo and gap
            train_end = test_start - self.gap - self.embargo
            
            # Apply purge
            if self.purge > 0:
                train_end -= self.purge
            
            if train_end <= 0:
                continue
            
            train_indices = np.arange(0, train_end)
            test_indices = np.arange(test_start, test_end)
            
            yield train_indices, test_indices
    
    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        """Get number of splits."""
        return self.n_splits


def calculate_embargo_size(
    lookback: int,
    horizon: int,
    feature_lag: int = 0,
    safety_factor: float = 1.0
) -> int:
    """
    Calculate appropriate embargo size.
    
    Args:
        lookback: Maximum lookback period in features
        horizon: Prediction horizon
        feature_lag: Additional lag in features
        safety_factor: Multiplicative safety factor
        
    Returns:
        Recommended embargo size
    """
    base_embargo = max(lookback - 1, horizon)
    total_embargo = base_embargo + feature_lag
    return int(np.ceil(total_embargo * safety_factor))


def validate_no_leakage(
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    embargo: int,
    timestamps: Optional[pd.Series] = None
) -> bool:
    """
    Validate that there's no data leakage between train and test.
    
    Args:
        train_idx: Training indices
        test_idx: Test indices
        embargo: Expected embargo size
        timestamps: Optional timestamps for additional validation
        
    Returns:
        True if no leakage detected
    """
    if len(train_idx) == 0 or len(test_idx) == 0:
        return True
    
    # Check index-based embargo
    max_train_idx = np.max(train_idx)
    min_test_idx = np.min(test_idx)
    
    actual_gap = min_test_idx - max_train_idx - 1
    
    if actual_gap < embargo:
        warnings.warn(f"Potential leakage: gap={actual_gap} < embargo={embargo}")
        return False
    
    # Check timestamp-based embargo if provided
    if timestamps is not None:
        train_times = timestamps.iloc[train_idx]
        test_times = timestamps.iloc[test_idx]
        
        max_train_time = train_times.max()
        min_test_time = test_times.min()
        
        time_gap = min_test_time - max_train_time
        
        # Convert embargo to time units (assuming uniform spacing)
        if len(timestamps) > 1:
            avg_time_diff = (timestamps.iloc[-1] - timestamps.iloc[0]) / (len(timestamps) - 1)
            expected_time_gap = embargo * avg_time_diff
            
            if time_gap < expected_time_gap:
                warnings.warn(f"Potential temporal leakage: time_gap={time_gap} < expected={expected_time_gap}")
                return False
    
    return True


def plot_walk_forward_splits(
    validator: WalkForwardValidator,
    n_samples: int,
    figsize: Tuple[int, int] = (12, 6)
) -> None:
    """
    Visualize walk-forward splits.
    
    Args:
        validator: Walk-forward validator
        n_samples: Total number of samples
        figsize: Figure size
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create dummy data for splitting
    X_dummy = np.zeros((n_samples, 1))
    
    colors = {'train': 'blue', 'val': 'green', 'test': 'red'}
    
    for i, (train_idx, val_idx, test_idx) in enumerate(validator.split(X_dummy)):
        y_pos = i
        
        # Plot train
        if len(train_idx) > 0:
            ax.barh(y_pos, train_idx[-1] - train_idx[0], 
                   left=train_idx[0], height=0.8, 
                   color=colors['train'], alpha=0.7)
        
        # Plot validation
        if len(val_idx) > 0:
            ax.barh(y_pos, val_idx[-1] - val_idx[0], 
                   left=val_idx[0], height=0.8, 
                   color=colors['val'], alpha=0.7)
        
        # Plot test
        if len(test_idx) > 0:
            ax.barh(y_pos, test_idx[-1] - test_idx[0], 
                   left=test_idx[0], height=0.8, 
                   color=colors['test'], alpha=0.7)
    
    # Labels and legend
    ax.set_xlabel('Sample Index')
    ax.set_ylabel('Split')
    ax.set_title('Walk-Forward Validation Splits')
    ax.set_yticks(range(validator.config.n_splits))
    ax.set_yticklabels([f'Split {i+1}' for i in range(validator.config.n_splits)])
    
    # Create legend
    patches = [mpatches.Patch(color=color, label=label.capitalize(), alpha=0.7) 
              for label, color in colors.items()]
    ax.legend(handles=patches, loc='upper right')
    
    plt.tight_layout()
    plt.show()