"""Unit tests for enhanced volatility features module."""

import pytest
import pandas as pd
import numpy as np

from src.features.volatility_features import VolatilityFeatures, VolatilityEstimators


@pytest.fixture
def sample_ohlcv_with_returns():
    """Create sample OHLCV data with returns for testing."""
    dates = pd.date_range(start="2024-01-01", periods=100, freq="15min")
    np.random.seed(42)
    
    close_prices = 50000 + np.cumsum(np.random.randn(100) * 100)
    
    df = pd.DataFrame({
        "open": close_prices + np.random.randn(100) * 50,
        "high": close_prices + np.abs(np.random.randn(100) * 100),
        "low": close_prices - np.abs(np.random.randn(100) * 100),
        "close": close_prices,
        "volume": np.abs(np.random.randn(100) * 1000000) + 100000
    }, index=dates)
    
    # Add returns
    df["returns"] = np.log(df["close"] / df["close"].shift(1))
    
    return df


class TestVolatilityFeatures:
    """Test volatility features calculation."""
    
    def test_initialization(self):
        """Test VolatilityFeatures initialization."""
        # Default initialization
        vf = VolatilityFeatures()
        assert vf.lookback_periods == [5, 10, 20, 50, 100, 200]
        
        # Custom periods
        custom_periods = [10, 20, 30]
        vf = VolatilityFeatures(lookback_periods=custom_periods)
        assert vf.lookback_periods == custom_periods
    
    def test_calculate_historical_volatility(self, sample_ohlcv_with_returns):
        """Test historical volatility calculation."""
        vf = VolatilityFeatures(lookback_periods=[10, 20])
        df = vf.calculate_historical_volatility(sample_ohlcv_with_returns)
        
        # Check volatility columns exist
        assert "volatility_10" in df.columns
        assert "volatility_20" in df.columns
        assert "volatility_ann_10" in df.columns
        assert "volatility_ann_20" in df.columns
        
        # Verify calculation
        expected_vol_10 = df["returns"].rolling(10).std()
        pd.testing.assert_series_equal(
            df["volatility_10"],
            expected_vol_10,
            check_names=False
        )
        
        # Check annualized volatility
        expected_ann_10 = expected_vol_10 * np.sqrt(365 * 24 * 4)
        pd.testing.assert_series_equal(
            df["volatility_ann_10"],
            expected_ann_10,
            check_names=False
        )
    
    def test_calculate_parkinson_volatility(self, sample_ohlcv_with_returns):
        """Test Parkinson volatility calculation."""
        vf = VolatilityFeatures()
        df = vf.calculate_parkinson_volatility(
            sample_ohlcv_with_returns,
            periods=[10, 20]
        )
        
        # Check columns exist
        assert "parkinson_vol_10" in df.columns
        assert "parkinson_vol_20" in df.columns
        
        # Check values are positive
        assert (df["parkinson_vol_10"].dropna() >= 0).all()
        assert (df["parkinson_vol_20"].dropna() >= 0).all()
        
        # Check no infinite values
        assert not np.isinf(df["parkinson_vol_10"].dropna()).any()
    
    def test_calculate_garman_klass_volatility(self, sample_ohlcv_with_returns):
        """Test Garman-Klass volatility calculation."""
        vf = VolatilityFeatures()
        df = vf.calculate_garman_klass_volatility(
            sample_ohlcv_with_returns,
            periods=[10, 20]
        )
        
        # Check columns exist
        assert "gk_vol_10" in df.columns
        assert "gk_vol_20" in df.columns
        
        # Check values are non-negative
        assert (df["gk_vol_10"].dropna() >= 0).all()
        assert (df["gk_vol_20"].dropna() >= 0).all()
        
        # Check no infinite values
        assert not np.isinf(df["gk_vol_10"].dropna()).any()
    
    def test_calculate_volatility_ratios(self, sample_ohlcv_with_returns):
        """Test volatility ratio calculation."""
        vf = VolatilityFeatures(lookback_periods=[5, 10, 20, 50, 100])
        
        # First calculate historical volatility
        df = vf.calculate_historical_volatility(sample_ohlcv_with_returns)
        
        # Then calculate ratios
        df = vf.calculate_volatility_ratios(df)
        
        # Check ratio columns
        assert "vol_ratio_10_50" in df.columns
        assert "vol_ratio_20_100" in df.columns
        assert "vol_ratio_5_20" in df.columns
        
        # Check momentum columns
        assert "vol_momentum_5" in df.columns
        assert "vol_momentum_20" in df.columns
        
        # Verify ratio calculation
        expected_ratio = df["volatility_10"] / (df["volatility_50"] + 1e-10)
        pd.testing.assert_series_equal(
            df["vol_ratio_10_50"],
            expected_ratio,
            check_names=False,
            rtol=1e-5
        )
    
    def test_calculate_volatility_percentiles(self, sample_ohlcv_with_returns):
        """Test volatility percentile calculation."""
        vf = VolatilityFeatures(lookback_periods=[20])
        
        # Calculate historical volatility first
        df = vf.calculate_historical_volatility(sample_ohlcv_with_returns)
        
        # Calculate percentiles
        df = vf.calculate_volatility_percentiles(df, lookback=50)
        
        # Check columns
        assert "vol_percentile_50" in df.columns
        assert "high_vol_regime" in df.columns
        assert "low_vol_regime" in df.columns
        assert "normal_vol_regime" in df.columns
        
        # Check percentiles are between 0 and 1
        percentiles = df["vol_percentile_50"].dropna()
        assert (percentiles >= 0).all()
        assert (percentiles <= 1).all()
        
        # Check regime indicators are binary
        assert set(df["high_vol_regime"].dropna().unique()).issubset({0, 1})
        assert set(df["low_vol_regime"].dropna().unique()).issubset({0, 1})
    
    def test_calculate_all(self, sample_ohlcv_with_returns):
        """Test complete volatility feature calculation."""
        vf = VolatilityFeatures(lookback_periods=[10, 20, 50])
        df = vf.calculate_all(sample_ohlcv_with_returns)
        
        # Check all feature types are present
        assert "volatility_10" in df.columns
        assert "volatility_ann_20" in df.columns
        assert "parkinson_vol_10" in df.columns
        assert "gk_vol_10" in df.columns
        assert "vol_ratio_10_50" in df.columns
        assert "vol_percentile_252" in df.columns
        assert "high_vol_regime" in df.columns
        
        # Check data integrity
        assert len(df) == len(sample_ohlcv_with_returns)
        assert not np.isinf(df.select_dtypes(include=[np.number]).values).any()
    
    def test_missing_returns_error(self):
        """Test error when returns column is missing."""
        vf = VolatilityFeatures()
        
        # Create data without returns
        df = pd.DataFrame({
            "open": [100, 101, 102],
            "high": [102, 103, 104],
            "low": [99, 100, 101],
            "close": [101, 102, 103],
            "volume": [1000, 1100, 1200]
        })
        
        # Should raise error for historical volatility
        with pytest.raises(ValueError, match="returns"):
            vf.calculate_historical_volatility(df)
    
    def test_edge_cases(self, sample_ohlcv_with_returns):
        """Test edge cases and error handling."""
        vf = VolatilityFeatures(lookback_periods=[5])
        
        # Test with zero/negative prices
        df_edge = sample_ohlcv_with_returns.copy()
        df_edge.loc[df_edge.index[10], "low"] = 0
        df_edge.loc[df_edge.index[11], "high"] = df_edge.loc[df_edge.index[11], "low"]
        
        # Should handle gracefully
        result = vf.calculate_all(df_edge)
        assert not np.isinf(result.select_dtypes(include=[np.number]).values).any()
        
        # Test with very small dataset
        small_df = sample_ohlcv_with_returns.iloc[:3]
        result = vf.calculate_all(small_df)
        assert len(result) == 3


class TestVolatilityEstimators:
    """Test enhanced volatility estimators."""
    
    @pytest.fixture
    def sample_ohlc_data(self):
        """Create realistic OHLC data for estimator testing."""
        np.random.seed(42)
        dates = pd.date_range('2024-01-01', periods=200, freq='15min')
        
        # Generate realistic price data
        base_price = 50000
        returns = np.random.normal(0, 0.001, len(dates))
        close_prices = base_price * np.exp(np.cumsum(returns))
        
        # Create OHLC ensuring proper relationships
        open_prices = np.roll(close_prices, 1)
        open_prices[0] = close_prices[0]
        
        # Add intraday volatility
        intraday_range = np.random.uniform(0.0005, 0.003, len(dates))
        high_prices = np.maximum(open_prices, close_prices) * (1 + intraday_range/2)
        low_prices = np.minimum(open_prices, close_prices) * (1 - intraday_range/2)
        
        df = pd.DataFrame({
            'open': open_prices,
            'high': high_prices,
            'low': low_prices,
            'close': close_prices,
            'volume': np.random.lognormal(10, 0.5, len(dates))
        }, index=dates)
        
        return df
    
    def test_atr_calculation(self, sample_ohlc_data):
        """Test ATR calculation."""
        atr = VolatilityEstimators.atr(sample_ohlc_data, window=14)
        
        assert isinstance(atr, pd.Series)
        assert len(atr) == len(sample_ohlc_data)
        
        # ATR should be positive when valid
        valid_atr = atr.dropna()
        assert (valid_atr > 0).all()
        
        # Should be normalized (fraction of price)
        assert valid_atr.max() < 0.1  # Less than 10%
    
    def test_garman_klass_calculation(self, sample_ohlc_data):
        """Test Garman-Klass estimator."""
        gk = VolatilityEstimators.garman_klass(sample_ohlc_data, window=14)
        
        assert isinstance(gk, pd.Series)
        assert len(gk) == len(sample_ohlc_data)
        
        valid_gk = gk.dropna()
        assert (valid_gk > 0).all()
        assert (valid_gk >= 1e-8).all()  # Minimum clipping
    
    def test_yang_zhang_calculation(self, sample_ohlc_data):
        """Test Yang-Zhang estimator."""
        yz = VolatilityEstimators.yang_zhang(sample_ohlc_data, window=14)
        
        assert isinstance(yz, pd.Series)
        assert len(yz) == len(sample_ohlc_data)
        
        valid_yz = yz.dropna()
        assert (valid_yz > 0).all()
        assert (valid_yz >= 1e-8).all()
    
    def test_parkinson_calculation(self, sample_ohlc_data):
        """Test Parkinson estimator."""
        park = VolatilityEstimators.parkinson(sample_ohlc_data, window=14)
        
        assert isinstance(park, pd.Series)
        assert len(park) == len(sample_ohlc_data)
        
        valid_park = park.dropna()
        assert (valid_park > 0).all()
        assert (valid_park >= 1e-8).all()
    
    def test_realized_volatility_calculation(self, sample_ohlc_data):
        """Test realized volatility."""
        rv = VolatilityEstimators.realized_volatility(sample_ohlc_data, window=14)
        
        assert isinstance(rv, pd.Series)
        assert len(rv) == len(sample_ohlc_data)
        
        valid_rv = rv.dropna()
        assert (valid_rv >= 0).all()  # Can be zero
    
    def test_estimators_correlation(self, sample_ohlc_data):
        """Test that estimators are reasonably correlated."""
        window = 20
        
        atr = VolatilityEstimators.atr(sample_ohlc_data, window)
        gk = VolatilityEstimators.garman_klass(sample_ohlc_data, window)
        yz = VolatilityEstimators.yang_zhang(sample_ohlc_data, window)
        
        # Find common valid indices
        valid_idx = ~(atr.isna() | gk.isna() | yz.isna())
        
        if valid_idx.sum() > 50:
            # Should be positively correlated
            corr_atr_gk = atr[valid_idx].corr(gk[valid_idx])
            corr_atr_yz = atr[valid_idx].corr(yz[valid_idx])
            
            assert corr_atr_gk > 0.2
            assert corr_atr_yz > 0.2


class TestEnhancedVolatilityFeatures:
    """Test the enhanced VolatilityFeatures class."""
    
    @pytest.fixture
    def enhanced_sample_data(self):
        """Create sample data for enhanced features."""
        np.random.seed(42)
        dates = pd.date_range('2024-01-01', periods=150, freq='15min')
        
        base_price = 50000
        returns = np.random.normal(0, 0.001, len(dates))
        close_prices = base_price * np.exp(np.cumsum(returns))
        
        df = pd.DataFrame({
            'open': close_prices * (1 + np.random.uniform(-0.0005, 0.0005, len(dates))),
            'high': close_prices * (1 + np.random.uniform(0, 0.002, len(dates))),
            'low': close_prices * (1 - np.random.uniform(0, 0.002, len(dates))),
            'close': close_prices,
            'volume': np.random.lognormal(10, 0.5, len(dates)),
            'returns': returns
        }, index=dates)
        
        # Fix OHLC relationships
        df['high'] = df[['open', 'high', 'close']].max(axis=1) * 1.001
        df['low'] = df[['open', 'low', 'close']].min(axis=1) * 0.999
        
        return df
    
    def test_calculate_advanced_volatility(self, enhanced_sample_data):
        """Test advanced volatility calculation."""
        vf = VolatilityFeatures(lookback_periods=[5, 20])
        result = vf.calculate_advanced_volatility(enhanced_sample_data)
        
        # Should add advanced volatility features
        expected_features = [
            'atr_vol_5', 'atr_vol_20',
            'gk_vol_5', 'gk_vol_20', 
            'yz_vol_5', 'yz_vol_20',
            'park_vol_5', 'park_vol_20'
        ]
        
        for feature in expected_features:
            assert feature in result.columns
            
            # Check quality
            valid_values = result[feature].dropna()
            if len(valid_values) > 0:
                assert (valid_values > 0).all()
    
    def test_enhanced_calculate_all(self, enhanced_sample_data):
        """Test enhanced calculate_all method."""
        vf = VolatilityFeatures(lookback_periods=[10, 30])
        result = vf.calculate_all(enhanced_sample_data)
        
        # Should include both traditional and advanced features
        assert 'volatility_10' in result.columns  # Traditional
        assert 'atr_vol_10' in result.columns     # Advanced
        assert 'yz_vol_30' in result.columns      # Advanced
        
        # Should maintain data integrity
        assert len(result) == len(enhanced_sample_data)
        
        # Should not have infinite values
        numeric_cols = result.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            assert not np.isinf(result[col]).any()
    
    def test_vol_estimators_integration(self):
        """Test that VolatilityEstimators is properly integrated."""
        vf = VolatilityFeatures()
        
        # Should have vol_estimators attribute
        assert hasattr(vf, 'vol_estimators')
        assert vf.vol_estimators is not None
        assert isinstance(vf.vol_estimators, VolatilityEstimators)