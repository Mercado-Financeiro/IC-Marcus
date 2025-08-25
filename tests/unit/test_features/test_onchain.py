"""Unit tests for on-chain features module."""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch
from hypothesis import given, strategies as st
import warnings

from src.features.onchain import OnChainFeatures, OnChainConfig


class TestOnChainConfig:
    """Test cases for OnChainConfig dataclass."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = OnChainConfig()
        
        assert config.nvt_window == 90
        assert config.mvrv_window == 365
        assert config.sopr_window == 7
        assert config.whale_threshold == 1000
        assert config.exchange_flow_window == 24
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = OnChainConfig(
            nvt_window=30,
            mvrv_window=180,
            sopr_window=14,
            whale_threshold=500,
            exchange_flow_window=12
        )
        
        assert config.nvt_window == 30
        assert config.mvrv_window == 180
        assert config.sopr_window == 14
        assert config.whale_threshold == 500
        assert config.exchange_flow_window == 12


class TestOnChainFeatures:
    """Test cases for OnChainFeatures class."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample OHLCV data."""
        n_samples = 500
        dates = pd.date_range('2023-01-01', periods=n_samples, freq='h')
        
        # Generate realistic-looking price data
        np.random.seed(42)
        price = 50000 + np.cumsum(np.random.randn(n_samples) * 100)
        
        df = pd.DataFrame({
            'timestamp': dates,
            'open': price + np.random.randn(n_samples) * 50,
            'high': price + np.abs(np.random.randn(n_samples)) * 100,
            'low': price - np.abs(np.random.randn(n_samples)) * 100,
            'close': price,
            'volume': np.abs(np.random.randn(n_samples)) * 1000000 + 100000
        })
        
        # Ensure high >= low and high >= close/open
        df['high'] = df[['high', 'open', 'close']].max(axis=1)
        df['low'] = df[['low', 'open', 'close']].min(axis=1)
        
        return df
    
    @pytest.fixture
    def onchain_features(self):
        """Create OnChainFeatures instance."""
        config = OnChainConfig(nvt_window=10, mvrv_window=30)
        return OnChainFeatures(config)
    
    def test_initialization(self, onchain_features):
        """Test OnChainFeatures initialization."""
        assert onchain_features.config.nvt_window == 10
        assert onchain_features.config.mvrv_window == 30
    
    def test_calculate_nvt_ratio(self, onchain_features, sample_data):
        """Test NVT ratio calculation."""
        result = onchain_features.calculate_nvt_ratio(sample_data)
        
        assert isinstance(result, pd.Series)
        assert len(result) == len(sample_data)
        assert not result.isna().all()
        assert not np.isinf(result).any()
        
        # Check that values are reasonable
        assert result.min() > -10
        assert result.max() < 20
    
    def test_nvt_ratio_with_zero_volume(self, onchain_features):
        """Test NVT ratio with zero volume."""
        df = pd.DataFrame({
            'close': [100, 200, 300],
            'volume': [0, 0, 0]
        })
        
        result = onchain_features.calculate_nvt_ratio(df)
        
        assert not np.isinf(result).any()
        assert not result.isna().all()
    
    def test_calculate_mvrv(self, onchain_features, sample_data):
        """Test MVRV calculation."""
        result = onchain_features.calculate_mvrv(sample_data)
        
        assert isinstance(result, pd.Series)
        assert len(result) == len(sample_data)
        
        # Z-score should be mostly between -3 and 3
        non_nan = result.dropna()
        assert (non_nan.abs() < 5).sum() / len(non_nan) > 0.9
    
    def test_calculate_sopr(self, onchain_features, sample_data):
        """Test SOPR calculation."""
        result = onchain_features.calculate_sopr(sample_data)
        
        assert isinstance(result, pd.Series)
        assert len(result) == len(sample_data)
        
        # SOPR should be around 1 on average
        assert 0.5 < result.mean() < 2.0
    
    def test_calculate_active_addresses(self, onchain_features, sample_data):
        """Test active addresses estimation."""
        result = onchain_features.calculate_active_addresses(sample_data)
        
        assert isinstance(result, pd.Series)
        assert len(result) == len(sample_data)
        assert (result >= 0).all()
        assert not result.isna().all()
    
    def test_calculate_hash_rate_proxy(self, onchain_features, sample_data):
        """Test hash rate proxy calculation."""
        result = onchain_features.calculate_hash_rate_proxy(sample_data)
        
        assert isinstance(result, pd.Series)
        assert len(result) == len(sample_data)
        assert not result.isna().all()
        
        # Should be normalized (z-score)
        non_nan = result.dropna()
        if len(non_nan) > 90:
            assert abs(non_nan.mean()) < 1
    
    def test_detect_whale_movements(self, onchain_features, sample_data):
        """Test whale movement detection."""
        result = onchain_features.detect_whale_movements(sample_data)
        
        assert isinstance(result, pd.Series)
        assert len(result) == len(sample_data)
        assert ((result >= 0) & (result <= 1)).all()
    
    def test_whale_movements_with_extreme_volume(self, onchain_features):
        """Test whale detection with extreme volume spikes."""
        df = pd.DataFrame({
            'close': [100] * 100,
            'volume': [1000] * 100
        })
        # Add extreme volume spike
        df.loc[50, 'volume'] = 1000000
        
        result = onchain_features.detect_whale_movements(df)
        
        # Should detect spike around index 50
        assert result.iloc[48:53].max() > 0
    
    def test_calculate_exchange_flows(self, onchain_features, sample_data):
        """Test exchange flow calculation."""
        flows = onchain_features.calculate_exchange_flows(sample_data)
        
        assert isinstance(flows, dict)
        assert 'exchange_inflow' in flows
        assert 'exchange_outflow' in flows
        assert 'exchange_netflow' in flows
        
        for key, series in flows.items():
            assert isinstance(series, pd.Series)
            assert len(series) == len(sample_data)
    
    def test_calculate_puell_multiple(self, onchain_features, sample_data):
        """Test Puell Multiple calculation."""
        result = onchain_features.calculate_puell_multiple(sample_data)
        
        assert isinstance(result, pd.Series)
        assert len(result) == len(sample_data)
        assert not result.isna().all()
        assert (result > 0).any()
    
    def test_calculate_dormancy_flow(self, onchain_features, sample_data):
        """Test dormancy flow calculation."""
        result = onchain_features.calculate_dormancy_flow(sample_data)
        
        assert isinstance(result, pd.Series)
        assert len(result) == len(sample_data)
        assert not result.isna().all()
    
    def test_calculate_all_features(self, onchain_features, sample_data):
        """Test calculation of all features."""
        result = onchain_features.calculate_all_features(sample_data)
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(sample_data)
        
        # Check that new columns were added
        expected_features = [
            'nvt_ratio', 'mvrv_zscore', 'asopr',
            'active_addresses', 'hash_rate_proxy',
            'whale_signal', 'exchange_inflow',
            'exchange_outflow', 'exchange_netflow',
            'puell_multiple', 'dormancy_flow'
        ]
        
        for feature in expected_features:
            assert feature in result.columns
        
        # Check no infinite values
        for col in expected_features:
            assert not np.isinf(result[col]).any()
    
    def test_get_feature_importance_weights(self, onchain_features):
        """Test feature importance weights."""
        weights = onchain_features.get_feature_importance_weights()
        
        assert isinstance(weights, dict)
        assert len(weights) > 0
        assert all(0 <= w <= 1 for w in weights.values())
        assert abs(sum(weights.values()) - 1.0) < 0.01  # Should sum to ~1
    
    @given(
        n_samples=st.integers(min_value=100, max_value=1000),
        price_base=st.floats(min_value=100, max_value=100000),
        volume_base=st.floats(min_value=1000, max_value=10000000)
    )
    def test_features_with_random_data(self, n_samples, price_base, volume_base):
        """Property-based test with random data."""
        # Generate random data
        df = pd.DataFrame({
            'close': np.abs(np.random.randn(n_samples)) * price_base + price_base,
            'high': np.abs(np.random.randn(n_samples)) * price_base + price_base * 1.1,
            'low': np.abs(np.random.randn(n_samples)) * price_base + price_base * 0.9,
            'volume': np.abs(np.random.randn(n_samples)) * volume_base + volume_base
        })
        
        features = OnChainFeatures(OnChainConfig(nvt_window=10))
        
        # Should not raise exceptions
        result = features.calculate_all_features(df)
        
        assert len(result) == n_samples
        assert not result.isna().all().any()
    
    def test_empty_dataframe(self, onchain_features):
        """Test with empty dataframe."""
        df = pd.DataFrame(columns=['close', 'high', 'low', 'volume'])
        
        result = onchain_features.calculate_all_features(df)
        
        assert len(result) == 0
        assert all(col in result.columns for col in ['nvt_ratio', 'mvrv_zscore'])
    
    def test_single_row_dataframe(self, onchain_features):
        """Test with single row dataframe."""
        df = pd.DataFrame({
            'close': [100],
            'high': [105],
            'low': [95],
            'volume': [1000]
        })
        
        result = onchain_features.calculate_all_features(df)
        
        assert len(result) == 1
        # Most features will be NaN due to rolling windows
        assert 'nvt_ratio' in result.columns
    
    def test_missing_columns(self, onchain_features):
        """Test error handling with missing columns."""
        df = pd.DataFrame({
            'close': [100, 200, 300]
        })
        
        with pytest.raises(KeyError):
            onchain_features.calculate_nvt_ratio(df, volume_col='volume')
    
    @pytest.mark.parametrize("window", [7, 14, 30, 90])
    def test_different_windows(self, sample_data, window):
        """Test with different window sizes."""
        config = OnChainConfig(nvt_window=window, mvrv_window=window*2)
        features = OnChainFeatures(config)
        
        result = features.calculate_nvt_ratio(sample_data)
        
        assert len(result) == len(sample_data)
        # Earlier values might be NaN due to rolling window
        assert not result.iloc[window:].isna().all()


class TestOnChainFeaturesEdgeCases:
    """Test edge cases and error conditions."""
    
    @pytest.fixture
    def features(self):
        """Create features instance."""
        return OnChainFeatures()
    
    def test_constant_price(self, features):
        """Test with constant price."""
        df = pd.DataFrame({
            'close': [100] * 100,
            'high': [100] * 100,
            'low': [100] * 100,
            'volume': np.random.rand(100) * 1000
        })
        
        result = features.calculate_all_features(df)
        
        assert len(result) == 100
        # Some features should handle constant price gracefully
        assert not np.isinf(result['nvt_ratio']).any()
    
    def test_extreme_volatility(self, features):
        """Test with extreme price volatility."""
        df = pd.DataFrame({
            'close': [100 * (2 ** i) for i in range(10)],
            'high': [105 * (2 ** i) for i in range(10)],
            'low': [95 * (2 ** i) for i in range(10)],
            'volume': [1000] * 10
        })
        
        result = features.calculate_all_features(df)
        
        assert len(result) == 10
        # Should handle exponential growth
        assert not np.isinf(result.select_dtypes(include=[np.number])).any().any()
    
    def test_negative_volume(self, features):
        """Test handling of negative volume (should not happen but test robustness)."""
        df = pd.DataFrame({
            'close': [100, 200, 300],
            'high': [105, 205, 305],
            'low': [95, 195, 295],
            'volume': [-1000, 2000, 3000]
        })
        
        # Should handle gracefully (abs or similar)
        result = features.calculate_all_features(df)
        assert len(result) == 3