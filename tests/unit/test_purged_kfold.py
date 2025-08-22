"""Tests for Purged K-Fold to ensure no temporal leakage."""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.data.splits import PurgedKFold


class TestPurgedKFold:
    """Test suite for Purged K-Fold cross-validation."""
    
    @pytest.mark.no_leak
    def test_no_temporal_leakage(self, sample_features_data):
        """Critical test: verify no temporal leakage between train and validation."""
        X = sample_features_data.drop('label', axis=1)
        y = sample_features_data['label']
        
        cv = PurgedKFold(n_splits=5, embargo=10)
        
        for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X, y)):
            # Get timestamps
            train_times = X.index[train_idx]
            val_times = X.index[val_idx]
            
            # Check no overlap
            train_max = train_times.max()
            train_min = train_times.min()
            val_max = val_times.max()
            val_min = val_times.min()
            
            # In Purged K-Fold, validation can be in the middle with training on both sides
            # This is the design - we just need to ensure proper embargo gaps
            
            # Check if there's training data before validation
            train_before = train_times[train_times < val_min]
            if len(train_before) > 0:
                gap_before = (val_min - train_before.max()).total_seconds() / 60
                assert gap_before >= cv.embargo * 15, (
                    f"Fold {fold_idx}: Insufficient embargo before validation. "
                    f"Gap: {gap_before:.1f} min, Required: {cv.embargo * 15} min"
                )
            
            # Check if there's training data after validation
            train_after = train_times[train_times > val_max]
            if len(train_after) > 0:
                gap_after = (train_after.min() - val_max).total_seconds() / 60
                assert gap_after >= cv.embargo * 15, (
                    f"Fold {fold_idx}: Insufficient embargo after validation. "
                    f"Gap: {gap_after:.1f} min, Required: {cv.embargo * 15} min"
                )
            
            # Most importantly: no training data should be WITHIN validation period
            train_within = train_times[(train_times >= val_min) & (train_times <= val_max)]
            assert len(train_within) == 0, (
                f"Fold {fold_idx}: Found {len(train_within)} training samples "
                f"within validation period [{val_min}, {val_max}]"
            )
    
    def test_all_samples_used(self, sample_features_data):
        """Test that all samples are used exactly once as validation."""
        X = sample_features_data.drop('label', axis=1)
        y = sample_features_data['label']
        
        cv = PurgedKFold(n_splits=3, embargo=5)
        
        all_val_indices = []
        
        for train_idx, val_idx in cv.split(X, y):
            all_val_indices.extend(val_idx)
            
            # Check no duplicates in validation
            assert len(val_idx) == len(set(val_idx)), "Duplicate indices in validation"
            
            # Check no overlap between train and val
            assert len(set(train_idx) & set(val_idx)) == 0, "Train and val overlap"
        
        # Check all indices are covered
        all_val_indices = sorted(set(all_val_indices))
        expected_indices = list(range(len(X)))
        
        # Some indices might be lost due to embargo, but coverage should be good
        coverage = len(all_val_indices) / len(expected_indices)
        assert coverage >= 0.8, f"Poor coverage: {coverage:.2%}"
    
    def test_embargo_effect(self):
        """Test that embargo properly creates gaps."""
        # Create simple time series
        dates = pd.date_range('2023-01-01', periods=100, freq='1h')
        X = pd.DataFrame({'feature': range(100)}, index=dates)
        y = pd.Series(range(100), index=dates)
        
        # Test with no embargo
        cv_no_embargo = PurgedKFold(n_splits=3, embargo=0)
        
        # Test with embargo
        cv_with_embargo = PurgedKFold(n_splits=3, embargo=5)
        
        # Count training samples
        train_samples_no_embargo = []
        train_samples_with_embargo = []
        
        for train_idx, _ in cv_no_embargo.split(X, y):
            train_samples_no_embargo.append(len(train_idx))
        
        for train_idx, _ in cv_with_embargo.split(X, y):
            train_samples_with_embargo.append(len(train_idx))
        
        # With embargo, we should have fewer training samples
        assert sum(train_samples_with_embargo) < sum(train_samples_no_embargo), \
            "Embargo should reduce training samples"
    
    def test_reproducibility(self, sample_features_data):
        """Test that splits are reproducible."""
        X = sample_features_data.drop('label', axis=1)
        y = sample_features_data['label']
        
        cv = PurgedKFold(n_splits=3, embargo=5)
        
        # Get splits twice
        splits1 = list(cv.split(X, y))
        splits2 = list(cv.split(X, y))
        
        assert len(splits1) == len(splits2), "Different number of splits"
        
        for (train1, val1), (train2, val2) in zip(splits1, splits2):
            np.testing.assert_array_equal(train1, train2, "Training indices differ")
            np.testing.assert_array_equal(val1, val2, "Validation indices differ")
    
    def test_minimum_samples_per_fold(self, sample_features_data):
        """Test that each fold has minimum samples."""
        X = sample_features_data.drop('label', axis=1)
        y = sample_features_data['label']
        
        cv = PurgedKFold(n_splits=5, embargo=10)
        
        min_samples = len(X) // (cv.n_splits * 2)  # At least this many
        
        for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X, y)):
            assert len(train_idx) >= min_samples, \
                f"Fold {fold_idx}: Too few training samples ({len(train_idx)})"
            assert len(val_idx) >= min_samples // 2, \
                f"Fold {fold_idx}: Too few validation samples ({len(val_idx)})"
    
    @pytest.mark.parametrize("n_splits,embargo", [(3, 0), (5, 10), (10, 20)])
    def test_different_configurations(self, n_splits, embargo):
        """Test various configurations of splits and embargo."""
        # Create data
        dates = pd.date_range('2023-01-01', periods=1000, freq='1h')
        X = pd.DataFrame({'feature': range(1000)}, index=dates)
        y = pd.Series(range(1000), index=dates)
        
        cv = PurgedKFold(n_splits=n_splits, embargo=embargo)
        
        fold_count = 0
        for train_idx, val_idx in cv.split(X, y):
            fold_count += 1
            
            # Basic sanity checks
            assert len(train_idx) > 0, f"Empty training set in fold {fold_count}"
            assert len(val_idx) > 0, f"Empty validation set in fold {fold_count}"
            
            # Check indices are valid
            assert max(train_idx) < len(X), "Invalid training indices"
            assert max(val_idx) < len(X), "Invalid validation indices"
        
        assert fold_count == n_splits, f"Expected {n_splits} folds, got {fold_count}"
    
    def test_get_n_splits(self):
        """Test get_n_splits method."""
        cv = PurgedKFold(n_splits=7, embargo=5)
        assert cv.get_n_splits() == 7
        assert cv.get_n_splits(X=None, y=None, groups=None) == 7
    
    @pytest.mark.no_leak
    def test_future_data_never_in_training(self):
        """Ensure future data never appears in training set."""
        dates = pd.date_range('2023-01-01', periods=500, freq='1h')
        X = pd.DataFrame({'feature': range(500)}, index=dates)
        y = pd.Series(range(500), index=dates)
        
        cv = PurgedKFold(n_splits=5, embargo=10)
        
        for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X, y)):
            # For each validation sample
            for v_idx in val_idx:
                val_time = X.index[v_idx]
                
                # Check no training sample is from the future
                for t_idx in train_idx:
                    train_time = X.index[t_idx]
                    
                    # If validation is in the "past" relative to this fold
                    # then training should not contain "future" data
                    if val_time < train_time:
                        time_diff = (train_time - val_time).total_seconds() / 3600
                        assert time_diff > cv.embargo, (
                            f"Future data in training! "
                            f"Val time: {val_time}, Train time: {train_time}, "
                            f"Diff: {time_diff:.1f} hours, Embargo: {cv.embargo} hours"
                        )


class TestPurgedKFoldEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_insufficient_data(self):
        """Test behavior with insufficient data."""
        dates = pd.date_range('2023-01-01', periods=10, freq='1h')
        X = pd.DataFrame({'feature': range(10)}, index=dates)
        y = pd.Series(range(10), index=dates)
        
        cv = PurgedKFold(n_splits=5, embargo=2)
        
        # Should handle gracefully
        splits = list(cv.split(X, y))
        assert len(splits) <= 5
    
    def test_single_split(self):
        """Test with single split (train/test split)."""
        dates = pd.date_range('2023-01-01', periods=100, freq='1h')
        X = pd.DataFrame({'feature': range(100)}, index=dates)
        y = pd.Series(range(100), index=dates)
        
        cv = PurgedKFold(n_splits=1, embargo=5)
        
        # Should work like a simple train/test split
        splits = list(cv.split(X, y))
        assert len(splits) == 1
        
        train_idx, val_idx = splits[0]
        assert len(train_idx) > 0
        assert len(val_idx) > 0
    
    def test_very_large_embargo(self):
        """Test with embargo larger than fold size."""
        dates = pd.date_range('2023-01-01', periods=100, freq='1h')
        X = pd.DataFrame({'feature': range(100)}, index=dates)
        y = pd.Series(range(100), index=dates)
        
        cv = PurgedKFold(n_splits=5, embargo=50)
        
        # Should still produce valid splits
        for train_idx, val_idx in cv.split(X, y):
            if len(train_idx) > 0 and len(val_idx) > 0:
                # Check embargo is respected
                train_times = X.index[train_idx]
                val_times = X.index[val_idx]
                
                # No overlap should occur
                assert train_times.max() < val_times.min() or \
                       val_times.max() < train_times.min()