"""Tests for adaptive labeling system."""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

from src.features.adaptive_labeling import AdaptiveLabeler, resolve_funding_minutes


class TestResolveFundingMinutes:
    """Test funding period resolution."""
    
    def test_known_symbols(self):
        """Test funding periods for known symbols."""
        assert resolve_funding_minutes("BTCUSDT") == 480
        assert resolve_funding_minutes("ETHUSDT") == 480
        assert resolve_funding_minutes("SOLUSDT") == 480
    
    def test_unknown_symbol(self):
        """Test default funding period for unknown symbol."""
        assert resolve_funding_minutes("UNKNOWNUSDT") == 480  # Default
    
    def test_with_timestamp(self):
        """Test with timestamp parameter."""
        timestamp = pd.Timestamp("2024-01-01 12:00:00")
        result = resolve_funding_minutes("BTCUSDT", timestamp)
        assert result == 480


class TestAdaptiveLabeler:
    """Test adaptive labeling system."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample OHLC data for testing."""
        np.random.seed(42)
        dates = pd.date_range('2024-01-01', periods=1000, freq='15min')
        
        # Generate realistic OHLC data
        base_price = 50000
        returns = np.random.normal(0, 0.001, len(dates))
        close_prices = base_price * np.exp(np.cumsum(returns))
        
        df = pd.DataFrame({
            'open': close_prices * (1 + np.random.uniform(-0.0005, 0.0005, len(dates))),
            'high': close_prices * (1 + np.random.uniform(0, 0.002, len(dates))),
            'low': close_prices * (1 - np.random.uniform(0, 0.002, len(dates))),
            'close': close_prices,
            'volume': np.random.lognormal(10, 0.5, len(dates))
        }, index=dates)
        
        # Ensure high >= close >= low and high >= open >= low
        df['high'] = df[['open', 'high', 'close']].max(axis=1) * 1.001
        df['low'] = df[['open', 'low', 'close']].min(axis=1) * 0.999
        
        return df
    
    @pytest.fixture
    def labeler(self):
        """Create adaptive labeler instance."""
        return AdaptiveLabeler(
            horizon_bars=4,
            k=1.0,
            vol_estimator='yang_zhang',
            neutral_zone=True
        )
    
    def test_initialization(self, labeler):
        """Test labeler initialization."""
        assert labeler.horizon_bars == 4
        assert labeler.k == 1.0
        assert labeler.vol_estimator == 'yang_zhang'
        assert labeler.neutral_zone is True
        assert '60m' in labeler.horizon_map
        assert labeler.horizon_map['60m'] == 4
    
    def test_horizon_map(self):
        """Test horizon mapping with custom funding period."""
        labeler = AdaptiveLabeler(funding_period_minutes=240)  # 4 hours
        
        assert labeler.horizon_map['15m'] == 1
        assert labeler.horizon_map['60m'] == 4
        assert '240m' in labeler.horizon_map
        assert labeler.horizon_map['240m'] == 16  # 240 / 15
    
    def test_calculate_volatility(self, labeler, sample_data):
        """Test volatility calculation."""
        volatility = labeler.calculate_volatility(sample_data, window=20)
        
        assert isinstance(volatility, pd.Series)
        assert len(volatility) == len(sample_data)
        assert volatility.dropna().min() > 0  # Volatility should be positive
        assert not volatility.dropna().isna().any()
    
    def test_calculate_volatility_invalid_estimator(self, sample_data):
        """Test volatility calculation with invalid estimator."""
        labeler = AdaptiveLabeler(vol_estimator='invalid')
        
        with pytest.raises(ValueError, match="Estimator invalid not supported"):
            labeler.calculate_volatility(sample_data)
    
    def test_calculate_adaptive_threshold(self, labeler, sample_data):
        """Test adaptive threshold calculation."""
        threshold = labeler.calculate_adaptive_threshold(sample_data, window=20)
        
        assert isinstance(threshold, pd.Series)
        assert len(threshold) == len(sample_data)
        
        # Check bounds
        valid_thresholds = threshold.dropna()
        assert valid_thresholds.min() >= 0.001  # Lower bound
        assert valid_thresholds.max() <= 0.10   # Upper bound
    
    def test_create_labels_with_neutral_zone(self, labeler, sample_data):
        """Test label creation with neutral zone."""
        labels = labeler.create_labels(sample_data)
        
        assert isinstance(labels, pd.Series)
        assert len(labels) == len(sample_data)
        
        # Check label values
        unique_labels = labels.dropna().unique()
        expected_labels = {-1, 0, 1}
        assert set(unique_labels).issubset(expected_labels)
    
    def test_create_labels_without_neutral_zone(self, sample_data):
        """Test label creation without neutral zone."""
        labeler = AdaptiveLabeler(neutral_zone=False)
        labels = labeler.create_labels(sample_data)
        
        unique_labels = labels.dropna().unique()
        expected_labels = {-1, 1}
        assert set(unique_labels).issubset(expected_labels)
        assert 0 not in unique_labels
    
    def test_get_label_distribution(self, labeler, sample_data):
        """Test label distribution statistics."""
        labels = labeler.create_labels(sample_data)
        distribution = labeler.get_label_distribution(labels)
        
        assert isinstance(distribution, dict)
        assert 'counts' in distribution
        assert 'proportions' in distribution
        assert 'total' in distribution
        assert 'balance_ratio' in distribution
        
        assert distribution['total'] > 0
        assert 0 <= distribution['balance_ratio'] <= 1
        
        # Check proportions sum to 1
        proportions_sum = sum(distribution['proportions'].values())
        assert abs(proportions_sum - 1.0) < 1e-6
    
    @patch('src.features.adaptive_labeling.RandomForestClassifier')
    def test_optimize_k_for_horizon(self, mock_rf, labeler, sample_data):
        """Test k optimization for specific horizon."""
        # Mock RandomForest
        mock_model = MagicMock()
        mock_model.predict.return_value = np.random.randint(0, 2, 100)
        mock_model.predict_proba.return_value = np.random.rand(100, 2)
        mock_rf.return_value = mock_model
        
        # Create simple features
        features = pd.DataFrame({
            'feature1': np.random.randn(len(sample_data)),
            'feature2': np.random.randn(len(sample_data))
        }, index=sample_data.index)
        
        optimal_k = labeler.optimize_k_for_horizon(
            sample_data, features, '60m', cv_splits=2, metric='f1'
        )
        
        assert isinstance(optimal_k, float)
        assert 0.5 <= optimal_k <= 2.0  # Within expected range
        
        # Check that horizon was set correctly
        assert labeler.horizon_bars == 4  # 60m = 4 bars
    
    def test_set_horizon(self, labeler):
        """Test horizon setting."""
        labeler.set_horizon('120m')
        assert labeler.horizon_bars == 8  # 120m = 8 bars
        
        with pytest.raises(ValueError, match="Horizon invalid not supported"):
            labeler.set_horizon('invalid')
    
    def test_get_threshold_stats(self, labeler, sample_data):
        """Test threshold statistics."""
        stats = labeler.get_threshold_stats(sample_data)
        
        assert isinstance(stats, dict)
        expected_keys = ['mean', 'std', 'min', 'max', 'median', 'q25', 'q75']
        for key in expected_keys:
            assert key in stats
            assert isinstance(stats[key], (int, float))
        
        # Sanity checks
        assert stats['min'] <= stats['q25'] <= stats['median'] <= stats['q75'] <= stats['max']
        assert stats['std'] >= 0
    
    def test_different_volatility_estimators(self, sample_data):
        """Test different volatility estimators."""
        estimators = ['atr', 'garman_klass', 'yang_zhang', 'parkinson', 'realized']
        
        for estimator in estimators:
            labeler = AdaptiveLabeler(vol_estimator=estimator)
            volatility = labeler.calculate_volatility(sample_data, window=20)
            
            assert isinstance(volatility, pd.Series)
            assert volatility.dropna().min() > 0
            assert not volatility.dropna().isna().any()
    
    def test_edge_cases(self, labeler):
        """Test edge cases and error conditions."""
        # Empty data
        empty_df = pd.DataFrame(columns=['open', 'high', 'low', 'close'])
        labels = labeler.create_labels(empty_df)
        assert len(labels) == 0 or labels.dropna().empty
        
        # Insufficient data for horizon
        small_df = pd.DataFrame({
            'open': [100, 101],
            'high': [102, 103],
            'low': [99, 100],
            'close': [101, 102]
        })
        
        labels = labeler.create_labels(small_df)
        # Should handle gracefully with mostly NaN
        assert labels.dropna().empty or len(labels.dropna()) < len(small_df)
    
    def test_label_consistency(self, labeler, sample_data):
        """Test that labels are consistent with thresholds."""
        labels = labeler.create_labels(sample_data)
        threshold = labeler.calculate_adaptive_threshold(sample_data)
        
        # Calculate future returns manually
        future_return = (
            sample_data['close'].shift(-labeler.horizon_bars) / sample_data['close'] - 1
        )
        
        # Check consistency for valid indices
        valid_idx = ~(labels.isna() | threshold.isna() | future_return.isna())
        
        if valid_idx.sum() > 0:
            labels_valid = labels[valid_idx]
            threshold_valid = threshold[valid_idx]
            future_return_valid = future_return[valid_idx]
            
            # Check that long labels correspond to positive returns above threshold
            long_mask = labels_valid == 1
            if long_mask.sum() > 0:
                long_returns = future_return_valid[long_mask]
                long_thresholds = threshold_valid[long_mask]
                # Most long signals should be above threshold
                assert (long_returns > long_thresholds).mean() > 0.8
    
    def test_memory_efficiency(self, sample_data):
        """Test that labeler doesn't keep unnecessary references."""
        labeler = AdaptiveLabeler()
        
        # Process data
        labels = labeler.create_labels(sample_data)
        
        # Check that labeler doesn't store the input data
        assert not hasattr(labeler, 'data')
        assert not hasattr(labeler, 'cached_volatility')
        
        # Labels should be independent
        del sample_data
        assert len(labels) > 0  # Labels should still exist