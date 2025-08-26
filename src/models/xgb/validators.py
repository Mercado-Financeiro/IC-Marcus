"""Temporal validation for XGBoost models."""

import numpy as np
import pandas as pd
from typing import Iterator, Tuple, Optional, List, Union, Dict
from sklearn.model_selection import TimeSeriesSplit
import structlog

from .config import ValidationConfig

log = structlog.get_logger()


class TemporalValidator:
    """Temporal validation with purging and embargo."""
    
    def __init__(self, config: Optional[ValidationConfig] = None):
        """Initialize temporal validator.
        
        Args:
            config: Validation configuration
        """
        self.config = config or ValidationConfig()
        
    def purged_kfold_split(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Optional[Union[pd.Series, np.ndarray]] = None,
        groups: Optional[np.ndarray] = None
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """Generate purged k-fold splits with embargo.
        
        Args:
            X: Features
            y: Labels (optional)
            groups: Group labels (optional)
            
        Yields:
            Train and validation indices
        """
        n_samples = len(X)
        indices = np.arange(n_samples)
        
        # Create base splits
        fold_size = n_samples // self.config.n_splits
        
        for fold in range(self.config.n_splits):
            # Define validation indices
            val_start = fold * fold_size
            val_end = (fold + 1) * fold_size if fold < self.config.n_splits - 1 else n_samples
            val_indices = indices[val_start:val_end]
            
            # Define train indices with embargo
            train_before = indices[:max(0, val_start - self.config.embargo_td)]
            train_after = indices[min(n_samples, val_end + self.config.embargo_td):]
            train_indices = np.concatenate([train_before, train_after])
            
            if len(train_indices) > 0 and len(val_indices) > 0:
                yield train_indices, val_indices
    
    def walk_forward_split(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Optional[Union[pd.Series, np.ndarray]] = None
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """Generate walk-forward validation splits.
        
        Args:
            X: Features
            y: Labels (optional)
            
        Yields:
            Train and validation indices
        """
        n_samples = len(X)
        window_size = self.config.walk_forward_window
        test_size = self.config.walk_forward_test_size
        
        # Start position
        start = 0
        
        while start + window_size + test_size <= n_samples:
            # Train indices
            train_end = start + window_size
            train_indices = np.arange(start, train_end)
            
            # Test indices
            test_start = train_end
            test_end = test_start + test_size
            test_indices = np.arange(test_start, test_end)
            
            yield train_indices, test_indices
            
            # Move window forward
            start += test_size
    
    def combinatorial_purged_split(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Optional[Union[pd.Series, np.ndarray]] = None,
        n_test_groups: Optional[int] = None
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """Generate combinatorial purged cross-validation splits.
        
        Based on Marcos López de Prado's method.
        
        Args:
            X: Features
            y: Labels (optional)
            n_test_groups: Number of test groups
            
        Yields:
            Train and validation indices
        """
        if n_test_groups is None:
            n_test_groups = self.config.n_test_groups
        
        n_samples = len(X)
        n_groups = self.config.n_splits
        indices = np.arange(n_samples)
        
        # Create groups
        group_size = n_samples // n_groups
        groups = []
        
        for i in range(n_groups):
            start = i * group_size
            end = start + group_size if i < n_groups - 1 else n_samples
            groups.append(indices[start:end])
        
        # Generate combinations
        from itertools import combinations
        
        for test_group_indices in combinations(range(n_groups), n_test_groups):
            # Test indices
            test_indices = np.concatenate([groups[i] for i in test_group_indices])
            
            # Train indices with purging
            train_groups = []
            for i in range(n_groups):
                if i not in test_group_indices:
                    group = groups[i]
                    
                    # Apply embargo
                    for test_idx in test_group_indices:
                        test_group = groups[test_idx]
                        
                        # Remove samples too close to test group
                        min_test = test_group.min()
                        max_test = test_group.max()
                        
                        mask = (
                            (group < min_test - self.config.embargo_td) |
                            (group > max_test + self.config.embargo_td)
                        )
                        group = group[mask]
                    
                    if len(group) > 0:
                        train_groups.append(group)
            
            if train_groups:
                train_indices = np.concatenate(train_groups)
                yield train_indices, test_indices
    
    def time_series_split(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Optional[Union[pd.Series, np.ndarray]] = None,
        n_splits: Optional[int] = None
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """Standard time series split.
        
        Args:
            X: Features
            y: Labels (optional)
            n_splits: Number of splits
            
        Yields:
            Train and validation indices
        """
        if n_splits is None:
            n_splits = self.config.n_splits
        
        tscv = TimeSeriesSplit(n_splits=n_splits)
        
        for train_idx, val_idx in tscv.split(X):
            yield train_idx, val_idx
    
    def validate_no_leakage(
        self,
        train_indices: np.ndarray,
        val_indices: np.ndarray,
        timestamps: Optional[pd.Series] = None
    ) -> bool:
        """Validate that there's no temporal leakage.
        
        Args:
            train_indices: Training indices
            val_indices: Validation indices
            timestamps: Optional timestamps
            
        Returns:
            True if no leakage detected
        """
        # Check no overlap
        if len(np.intersect1d(train_indices, val_indices)) > 0:
            log.warning("Overlap detected between train and validation")
            return False
        
        # Check temporal order if timestamps provided
        if timestamps is not None:
            train_times = timestamps.iloc[train_indices]
            val_times = timestamps.iloc[val_indices]
            
            # Check if any training sample is after validation
            if train_times.max() > val_times.min():
                # This might be okay for purged k-fold
                # Check if embargo is respected
                time_diff = (val_times.min() - train_times.max()).total_seconds()
                
                if time_diff < 0:  # Training after validation
                    # Check if it's far enough (after validation ends)
                    if train_times.min() < val_times.max():
                        log.warning("Temporal leakage detected")
                        return False
        
        return True
    
    def get_cv_strategy(
        self,
        strategy: Optional[str] = None
    ) -> callable:
        """Get cross-validation strategy.
        
        Args:
            strategy: Strategy name
            
        Returns:
            CV split generator function
        """
        if strategy is None:
            if self.config.use_walk_forward:
                strategy = 'walk_forward'
            elif self.config.use_combinatorial:
                strategy = 'combinatorial'
            else:
                strategy = 'purged_kfold'
        
        strategies = {
            'purged_kfold': self.purged_kfold_split,
            'walk_forward': self.walk_forward_split,
            'combinatorial': self.combinatorial_purged_split,
            'time_series': self.time_series_split
        }
        
        if strategy not in strategies:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        return strategies[strategy]
    
    def calculate_cv_metrics(
        self,
        y_true_list: List[np.ndarray],
        y_pred_list: List[np.ndarray]
    ) -> Dict[str, float]:
        """Calculate aggregated CV metrics.
        
        Args:
            y_true_list: List of true labels per fold
            y_pred_list: List of predictions per fold
            
        Returns:
            Dictionary of aggregated metrics
        """
        from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
        
        metrics = {
            'accuracy': [],
            'f1_score': [],
            'roc_auc': []
        }
        
        for y_true, y_pred in zip(y_true_list, y_pred_list):
            metrics['accuracy'].append(accuracy_score(y_true, y_pred))
            metrics['f1_score'].append(f1_score(y_true, y_pred))
            
            # For ROC AUC, need probabilities
            if y_pred.ndim == 2:
                metrics['roc_auc'].append(roc_auc_score(y_true, y_pred[:, 1]))
        
        # Calculate mean and std
        result = {}
        for metric_name, values in metrics.items():
            if values:
                result[f'{metric_name}_mean'] = np.mean(values)
                result[f'{metric_name}_std'] = np.std(values)
        
        return result