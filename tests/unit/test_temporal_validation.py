"""Tests for temporal validation module."""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from src.features.validation.temporal import (
    TemporalValidator,
    TemporalValidationConfig,
    get_temporal_splitter
)


class TestTemporalValidator:
    """Test suite for temporal validation."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample time series data."""
        dates = pd.date_range('2024-01-01', periods=1000, freq='15min', tz='UTC')
        df = pd.DataFrame({
            'feature_1': np.random.randn(1000),
            'feature_2': np.random.randn(1000),
            'feature_3': np.random.randn(1000)
        }, index=dates)
        
        y = pd.Series(np.random.randint(0, 2, 1000), index=dates)
        
        return df, y
    
    def test_no_temporal_leakage_purged_kfold(self, sample_data):
        """Test that purged k-fold has no temporal leakage."""
        X, y = sample_data
        
        config = TemporalValidationConfig(
            n_splits=5,
            embargo=10,
            purge=5,
            check_leakage=True
        )
        validator = TemporalValidator(config)
        
        for train_idx, val_idx in validator.split(X, y, strategy='purged_kfold'):
            # Check no index overlap
            assert len(np.intersect1d(train_idx, val_idx)) == 0
            
            # Check temporal separation
            train_times = X.index[train_idx]
            val_times = X.index[val_idx]
            
            val_start = val_times.min()
            val_end = val_times.max()
            
            # No training samples should be within validation period
            train_in_val = train_times[(train_times >= val_start) & (train_times <= val_end)]
            assert len(train_in_val) == 0, "Training samples found within validation period!"
            
            # Check embargo is respected
            train_before_val = train_times[train_times < val_start]
            if len(train_before_val) > 0:
                gap_minutes = (val_start - train_before_val.max()).total_seconds() / 60
                # Embargo of 10 bars * 15 minutes = 150 minutes
                min_gap = config.purge * 15  # 5 bars * 15 min = 75 min
                assert gap_minutes >= min_gap - 1, f"Insufficient purge: {gap_minutes} < {min_gap}"
    
    def test_shuffle_raises_error(self, sample_data):
        """Test that shuffle=True raises an error."""
        X, y = sample_data
        validator = TemporalValidator()
        
        with pytest.raises(ValueError, match="shuffle=True detected"):
            list(validator.split(X, y, shuffle=True))
    
    def test_walk_forward_split(self, sample_data):
        """Test walk-forward validation."""
        X, y = sample_data
        
        config = TemporalValidationConfig(
            walk_forward_window=200,
            walk_forward_test_size=50,
            walk_forward_gap=10
        )
        validator = TemporalValidator(config)
        
        splits = list(validator.split(X, y, strategy='walk_forward'))
        
        # Check that we get multiple splits
        assert len(splits) > 0
        
        # Check each split
        for train_idx, test_idx in splits:
            # Training window should be 200
            assert len(train_idx) == 200
            # Test window should be 50
            assert len(test_idx) == 50
            
            # Check gap between train and test
            gap = test_idx[0] - train_idx[-1] - 1
            assert gap == 10, f"Gap should be 10, got {gap}"
    
    def test_expanding_window_split(self, sample_data):
        """Test expanding window validation."""
        X, y = sample_data
        
        config = TemporalValidationConfig(
            walk_forward_window=100,  # Min train size
            walk_forward_test_size=50,
            n_splits=5
        )
        validator = TemporalValidator(config)
        
        splits = list(validator.split(X, y, strategy='expanding_window'))
        
        # Check expanding behavior
        prev_train_size = 0
        for i, (train_idx, test_idx) in enumerate(splits):
            train_size = len(train_idx)
            
            # Training size should expand (or at least not shrink)
            assert train_size >= prev_train_size
            
            # Test size should be constant
            assert len(test_idx) == 50
            
            # First split should have minimum training size
            if i == 0:
                assert train_size >= 100
            
            prev_train_size = train_size
    
    def test_combinatorial_purged_split(self, sample_data):
        """Test combinatorial purged cross-validation."""
        X, y = sample_data
        
        config = TemporalValidationConfig(
            n_splits=5,
            n_test_groups=2,
            embargo=10,
            check_leakage=False  # Disable strict checking for combinatorial
        )
        validator = TemporalValidator(config)
        
        splits = list(validator.split(X, y, strategy='combinatorial'))
        
        # Should generate C(5,2) = 10 combinations
        assert len(splits) == 10
        
        # Check each split
        for train_idx, test_idx in splits:
            # Check no overlap
            assert len(np.intersect1d(train_idx, test_idx)) == 0
            
            # Both should have samples
            assert len(train_idx) > 0
            assert len(test_idx) > 0
    
    def test_insufficient_samples_error(self):
        """Test error when insufficient samples."""
        # Create tiny dataset
        dates = pd.date_range('2024-01-01', periods=10, freq='15min', tz='UTC')
        X = pd.DataFrame({'feature': np.random.randn(10)}, index=dates)
        y = pd.Series(np.random.randint(0, 2, 10), index=dates)
        
        config = TemporalValidationConfig(
            n_splits=5,
            min_train_samples=100,
            min_test_samples=20
        )
        validator = TemporalValidator(config)
        
        with pytest.raises(ValueError, match="No valid splits generated"):
            list(validator.split(X, y))
    
    def test_validate_dataset(self, sample_data):
        """Test dataset validation."""
        X, y = sample_data
        validator = TemporalValidator()
        
        report = validator.validate_dataset(X, y)
        
        # Check report structure
        assert 'n_samples' in report
        assert 'has_datetime_index' in report
        assert 'is_sorted' in report
        assert 'has_duplicates' in report
        assert 'warnings' in report
        assert 'errors' in report
        
        # Our sample data should be valid
        assert report['n_samples'] == 1000
        assert report['has_datetime_index'] is True
        assert report['is_sorted'] is True
        assert report['has_duplicates'] is False
        assert len(report['errors']) == 0
    
    def test_unsorted_data_detection(self):
        """Test detection of unsorted data."""
        # Create unsorted data
        dates = pd.date_range('2024-01-01', periods=100, freq='15min', tz='UTC')
        shuffled_dates = dates.to_list()
        np.random.shuffle(shuffled_dates)
        
        X = pd.DataFrame({'feature': np.random.randn(100)}, index=shuffled_dates)
        y = pd.Series(np.random.randint(0, 2, 100), index=shuffled_dates)
        
        validator = TemporalValidator()
        report = validator.validate_dataset(X, y)
        
        # Should detect unsorted data
        assert report['is_sorted'] is False
        assert any('not sorted' in error for error in report['errors'])
    
    def test_duplicate_timestamps_detection(self):
        """Test detection of duplicate timestamps."""
        # Create data with duplicates
        dates = pd.date_range('2024-01-01', periods=99, freq='15min', tz='UTC')
        dates = dates.append(pd.DatetimeIndex([dates[50]]))  # Add duplicate
        
        X = pd.DataFrame({'feature': np.random.randn(100)}, index=dates)
        y = pd.Series(np.random.randint(0, 2, 100), index=dates)
        
        validator = TemporalValidator()
        report = validator.validate_dataset(X, y)
        
        # Should detect duplicates
        assert report['has_duplicates'] is True
        assert any('Duplicate' in error for error in report['errors'])
    
    def test_get_n_splits(self):
        """Test getting number of splits."""
        config = TemporalValidationConfig(n_splits=7)
        validator = TemporalValidator(config)
        
        assert validator.get_n_splits('purged_kfold') == 7
        assert validator.get_n_splits('combinatorial') == 7
        
        # Walk-forward splits depend on data size
        assert validator.get_n_splits('walk_forward') == -1
    
    def test_backward_compatibility(self):
        """Test backward compatibility function."""
        splitter = get_temporal_splitter('purged_kfold', n_splits=3, embargo=5)
        
        assert isinstance(splitter, TemporalValidator)
        assert splitter.config.n_splits == 3
        assert splitter.config.embargo == 5


class TestTemporalLeakageGuards:
    """Test suite for temporal leakage prevention."""
    
    def test_no_future_information_in_training(self):
        """Ensure no future information leaks into training."""
        # Create data with clear temporal pattern
        dates = pd.date_range('2024-01-01', periods=500, freq='1h', tz='UTC')
        
        # Create feature that increases over time
        X = pd.DataFrame({
            'time_feature': np.arange(500),
            'random_feature': np.random.randn(500)
        }, index=dates)
        
        y = pd.Series(np.random.randint(0, 2, 500), index=dates)
        
        config = TemporalValidationConfig(
            n_splits=3,
            embargo=24,  # 24 hours embargo
            check_leakage=True
        )
        validator = TemporalValidator(config)
        
        for train_idx, val_idx in validator.split(X, y):
            # Max time feature in training should be less than min in validation
            max_train_time = X.iloc[train_idx]['time_feature'].max()
            min_val_time = X.iloc[val_idx]['time_feature'].min()
            
            # Account for the fact that in purged k-fold, 
            # training can come from after validation
            train_times = X.index[train_idx]
            val_times = X.index[val_idx]
            
            val_start = val_times.min()
            val_end = val_times.max()
            
            # Check training samples before validation
            train_before = train_times[train_times < val_start]
            if len(train_before) > 0:
                # These should have time features less than validation
                max_before = X.loc[train_before, 'time_feature'].max()
                assert max_before < min_val_time
            
            # Check training samples after validation
            train_after = train_times[train_times > val_end]
            if len(train_after) > 0:
                # These should have time features greater than validation
                min_after = X.loc[train_after, 'time_feature'].min()
                max_val_time = X.iloc[val_idx]['time_feature'].max()
                assert min_after > max_val_time
    
    def test_embargo_enforcement(self):
        """Test that embargo is properly enforced."""
        # Create hourly data
        dates = pd.date_range('2024-01-01', periods=240, freq='1h', tz='UTC')
        X = pd.DataFrame({'feature': np.random.randn(240)}, index=dates)
        y = pd.Series(np.random.randint(0, 2, 240), index=dates)
        
        # Set embargo to 5 hours
        config = TemporalValidationConfig(
            n_splits=3,
            embargo=5,  # 5 bars = 5 hours
            time_resolution_minutes=60  # 1 hour bars
        )
        validator = TemporalValidator(config)
        
        for train_idx, val_idx in validator.split(X, y):
            train_times = X.index[train_idx]
            val_times = X.index[val_idx]
            
            val_start = val_times.min()
            val_end = val_times.max()
            
            # Check gap before validation
            train_before = train_times[train_times < val_start]
            if len(train_before) > 0:
                gap_hours = (val_start - train_before.max()).total_seconds() / 3600
                # Should be at least 5 hours
                assert gap_hours >= 4.9, f"Embargo violation: {gap_hours} < 5 hours"
            
            # Check gap after validation
            train_after = train_times[train_times > val_end]
            if len(train_after) > 0:
                gap_hours = (train_after.min() - val_end).total_seconds() / 3600
                assert gap_hours >= 4.9, f"Embargo violation: {gap_hours} < 5 hours"