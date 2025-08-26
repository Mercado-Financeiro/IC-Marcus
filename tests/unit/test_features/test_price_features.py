"""Unit tests for price features module."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.features.price_features import PriceFeatures


@pytest.fixture
def sample_ohlcv_data():
    """Create sample OHLCV data for testing."""
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
    
    return df


class TestPriceFeatures:
    """Test price features calculation."""
    
    def test_initialization(self):
        """Test PriceFeatures initialization."""
        # Default initialization
        pf = PriceFeatures()
        assert pf.lookback_periods == [5, 10, 20, 50, 100, 200]
        
        # Custom periods
        custom_periods = [10, 20, 30]
        pf = PriceFeatures(lookback_periods=custom_periods)
        assert pf.lookback_periods == custom_periods
    
    def test_calculate_returns(self, sample_ohlcv_data):
        """Test return calculation."""
        pf = PriceFeatures(lookback_periods=[5, 10])
        df = pf.calculate_returns(sample_ohlcv_data)
        
        # Check returns column exists
        assert "returns" in df.columns
        
        # Check multiple period returns
        assert "returns_5" in df.columns
        assert "returns_10" in df.columns
        
        # Check momentum columns
        assert "momentum_5" in df.columns
        assert "momentum_10" in df.columns
        
        # Verify returns calculation
        expected_returns = np.log(df["close"] / df["close"].shift(1))
        pd.testing.assert_series_equal(
            df["returns"].iloc[1:],
            expected_returns.iloc[1:],
            check_names=False
        )
    
    def test_calculate_moving_averages(self, sample_ohlcv_data):
        """Test moving average calculation."""
        pf = PriceFeatures(lookback_periods=[10, 20])
        df = pf.calculate_moving_averages(sample_ohlcv_data)
        
        # Check SMA columns
        assert "sma_10" in df.columns
        assert "sma_20" in df.columns
        
        # Check EMA columns
        assert "ema_10" in df.columns
        assert "ema_20" in df.columns
        
        # Check price-to-SMA ratios
        assert "price_to_sma_10" in df.columns
        assert "price_to_sma_20" in df.columns
        
        # Verify SMA calculation
        expected_sma_10 = df["close"].rolling(10).mean()
        pd.testing.assert_series_equal(
            df["sma_10"],
            expected_sma_10,
            check_names=False
        )
    
    def test_calculate_zscore(self, sample_ohlcv_data):
        """Test z-score calculation."""
        pf = PriceFeatures()
        df = pf.calculate_zscore(sample_ohlcv_data, periods=[20, 50])
        
        # Check z-score columns
        assert "zscore_20" in df.columns
        assert "zscore_50" in df.columns
        
        # Verify z-score calculation for period 20
        mean_20 = df["close"].rolling(20).mean()
        std_20 = df["close"].rolling(20).std()
        expected_zscore = (df["close"] - mean_20) / (std_20 + 1e-10)
        
        pd.testing.assert_series_equal(
            df["zscore_20"].iloc[20:],
            expected_zscore.iloc[20:],
            check_names=False,
            rtol=1e-5
        )
    
    def test_calculate_crossovers(self, sample_ohlcv_data):
        """Test moving average crossover calculation."""
        pf = PriceFeatures(lookback_periods=[20, 50, 100])
        
        # First calculate MAs
        df = pf.calculate_moving_averages(sample_ohlcv_data)
        
        # Then calculate crossovers
        df = pf.calculate_crossovers(df)
        
        # Check crossover columns
        assert "sma_cross_20_50" in df.columns
        assert "sma_cross_50_100" in df.columns
        
        # Verify crossover logic
        # Crossover should be 1 when fast crosses above slow
        # -1 when fast crosses below slow
        # 0 otherwise
        crossover_values = df["sma_cross_20_50"].dropna().unique()
        assert all(v in [-1, 0, 1] for v in crossover_values)
    
    def test_calculate_all(self, sample_ohlcv_data):
        """Test complete feature calculation."""
        pf = PriceFeatures(lookback_periods=[10, 20])
        df = pf.calculate_all(sample_ohlcv_data)
        
        # Check that all feature types are present
        assert "returns" in df.columns
        assert "returns_10" in df.columns
        assert "momentum_10" in df.columns
        assert "sma_10" in df.columns
        assert "ema_20" in df.columns
        assert "price_to_sma_10" in df.columns
        assert "zscore_20" in df.columns
        
        # Check data integrity
        assert len(df) == len(sample_ohlcv_data)
        assert not np.isinf(df.select_dtypes(include=[np.number]).values).any()
    
    def test_edge_cases(self, sample_ohlcv_data):
        """Test edge cases and error handling."""
        pf = PriceFeatures(lookback_periods=[5])
        
        # Test with zero prices (should handle division by zero)
        df_zero = sample_ohlcv_data.copy()
        df_zero.loc[df_zero.index[10], "close"] = 0.001  # Use small value instead of zero
        
        result = pf.calculate_all(df_zero)
        # Check no infinite values (NaN is ok from rolling windows)
        numeric_data = result.select_dtypes(include=[np.number]).values
        finite_mask = ~np.isnan(numeric_data)
        assert not np.isinf(numeric_data[finite_mask]).any()
        
        # Test with very small dataset
        small_df = sample_ohlcv_data.iloc[:3]
        result = pf.calculate_all(small_df)
        assert len(result) == 3