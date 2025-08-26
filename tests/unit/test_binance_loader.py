"""Tests for Binance data loader."""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import sys
from pathlib import Path
import tempfile
import shutil

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.data.binance_loader import CryptoDataLoader


class TestCryptoDataLoader:
    """Test suite for crypto data loader."""
    
    def test_initialization(self):
        """Test loader initialization."""
        loader = CryptoDataLoader()
        
        assert loader.exchange_name == "binance"
        assert loader.use_cache == True
        assert loader.cache_dir.exists()
    
    def test_cache_path_generation(self):
        """Test cache path generation."""
        loader = CryptoDataLoader(cache_dir="test_cache")
        
        path = loader._get_cache_path("BTCUSDT", "15m")
        
        assert "BTCUSDT" in str(path)
        assert "15m" in str(path)
        assert path.suffix == ".parquet"
    
    def test_data_hash_calculation(self, sample_ohlcv_data):
        """Test data hash calculation."""
        loader = CryptoDataLoader()
        
        hash1 = loader._calculate_data_hash(sample_ohlcv_data)
        hash2 = loader._calculate_data_hash(sample_ohlcv_data)
        
        # Same data should produce same hash
        assert hash1 == hash2
        assert len(hash1) == 16  # We take first 16 chars
        
        # Different data should produce different hash
        modified_data = sample_ohlcv_data.copy()
        modified_data['close'] = modified_data['close'] * 1.01
        hash3 = loader._calculate_data_hash(modified_data)
        
        assert hash1 != hash3
    
    @patch('src.data.binance_loader.ccxt.binance')
    def test_fetch_from_api_success(self, mock_exchange_class, sample_ohlcv_data):
        """Test successful API fetch."""
        # Mock exchange instance
        mock_exchange = MagicMock()
        mock_exchange_class.return_value = mock_exchange
        
        # Mock fetch_ohlcv to return sample data
        mock_ohlcv = [
            [1609459200000, 100, 105, 95, 102, 1000],  # timestamp, o, h, l, c, v
            [1609462800000, 102, 106, 101, 104, 1100],
        ]
        mock_exchange.fetch_ohlcv.return_value = mock_ohlcv
        
        loader = CryptoDataLoader()
        loader.exchange = mock_exchange
        
        df = loader._fetch_from_api("BTCUSDT", "15m", "2021-01-01", "2021-01-02")
        
        assert df is not None
        assert len(df) == 2
        assert 'open' in df.columns
        assert 'close' in df.columns
        assert df.index.name == 'timestamp'
        assert df.index.tz is not None  # Should be timezone-aware
    
    @patch('src.data.binance_loader.yf.Ticker')
    def test_yfinance_fallback(self, mock_ticker_class):
        """Test yfinance fallback when Binance fails."""
        # Mock yfinance ticker
        mock_ticker = MagicMock()
        mock_ticker_class.return_value = mock_ticker
        
        # Create sample data for yfinance
        dates = pd.date_range('2023-01-01', periods=10, freq='15min')
        mock_history = pd.DataFrame({
            'Open': [100] * 10,
            'High': [105] * 10,
            'Low': [95] * 10,
            'Close': [102] * 10,
            'Volume': [1000] * 10
        }, index=dates)
        
        mock_ticker.history.return_value = mock_history
        
        loader = CryptoDataLoader()
        loader.exchange = None  # Simulate Binance not available
        
        df = loader._fetch_from_yfinance("BTCUSDT", "15m", "2023-01-01", "2023-01-02")
        
        assert df is not None
        assert 'open' in df.columns
        assert 'close' in df.columns
        assert df.index.tz is not None
    
    def test_filter_by_dates(self, sample_ohlcv_data):
        """Test date filtering."""
        loader = CryptoDataLoader()
        
        start = "2023-06-01"
        end = "2023-08-01"
        
        filtered = loader._filter_by_dates(sample_ohlcv_data, start, end)
        
        assert len(filtered) <= len(sample_ohlcv_data)
        assert filtered.index.min() >= pd.Timestamp(start, tz='UTC')
        assert filtered.index.max() <= pd.Timestamp(end, tz='UTC')
    
    def test_timeframe_to_ms(self):
        """Test timeframe conversion to milliseconds."""
        loader = CryptoDataLoader()
        
        assert loader._timeframe_to_ms("1m") == 60 * 1000
        assert loader._timeframe_to_ms("5m") == 5 * 60 * 1000
        assert loader._timeframe_to_ms("1h") == 60 * 60 * 1000
        assert loader._timeframe_to_ms("1d") == 24 * 60 * 60 * 1000
    
    def test_symbol_conversion(self):
        """Test symbol conversion for yfinance."""
        loader = CryptoDataLoader()
        
        assert loader._convert_symbol_to_yfinance("BTCUSDT") == "BTC-USD"
        assert loader._convert_symbol_to_yfinance("ETHUSDT") == "ETH-USD"
        assert loader._convert_symbol_to_yfinance("BTCUSD") == "BTCUSD"
    
    def test_timeframe_conversion(self):
        """Test timeframe conversion for yfinance."""
        loader = CryptoDataLoader()
        
        assert loader._convert_timeframe_to_yfinance("1m") == "1m"
        assert loader._convert_timeframe_to_yfinance("5m") == "5m"
        assert loader._convert_timeframe_to_yfinance("1h") == "1h"
        assert loader._convert_timeframe_to_yfinance("4h") == "1h"  # yfinance doesn't have 4h
    
    def test_data_validation_success(self, sample_ohlcv_data):
        """Test successful data validation."""
        loader = CryptoDataLoader()
        
        # Ensure data meets validation criteria
        validated = loader.validate_data(sample_ohlcv_data.copy())
        
        assert validated is not None
        assert not validated.index.has_duplicates
        assert validated.index.is_monotonic_increasing
    
    def test_data_validation_failure(self):
        """Test data validation with invalid data."""
        loader = CryptoDataLoader()
        
        # Create invalid data (negative prices)
        dates = pd.date_range('2023-01-01', periods=10, freq='15min', tz='UTC')
        invalid_df = pd.DataFrame({
            'open': [-100] * 10,  # Invalid negative price
            'high': [105] * 10,
            'low': [95] * 10,
            'close': [102] * 10,
            'volume': [1000] * 10
        }, index=dates)
        
        with pytest.raises(Exception):  # Should raise validation error
            loader.validate_data(invalid_df)
    
    def test_cache_functionality(self, temp_cache_dir, sample_ohlcv_data):
        """Test cache save and load."""
        loader = CryptoDataLoader(cache_dir=temp_cache_dir)
        
        # Mock the API fetch to return sample data
        with patch.object(loader, '_fetch_from_api', return_value=sample_ohlcv_data):
            # First call should fetch from API and save to cache
            df1 = loader.fetch_ohlcv("BTCUSDT", "15m", "2023-01-01", "2023-12-31")
            
            # Check cache file was created
            cache_path = loader._get_cache_path("BTCUSDT", "15m")
            assert cache_path.exists()
            
            # Second call should load from cache
            df2 = loader.fetch_ohlcv("BTCUSDT", "15m", "2023-01-01", "2023-12-31")
            
            # Data should be identical
            pd.testing.assert_frame_equal(df1, df2)
    
    def test_check_data_quality(self, sample_ohlcv_data):
        """Test data quality checks."""
        loader = CryptoDataLoader()
        
        metrics = loader.check_data_quality(sample_ohlcv_data)
        
        # Check required metrics
        assert 'total_rows' in metrics
        assert 'start_date' in metrics
        assert 'end_date' in metrics
        assert 'missing_values' in metrics
        assert 'gaps_detected' in metrics
        assert 'data_hash' in metrics
        assert 'monotonic' in metrics
        assert 'duplicates' in metrics
        
        # Check metric values
        assert metrics['total_rows'] == len(sample_ohlcv_data)
        assert metrics['monotonic'] == True
        assert metrics['duplicates'] == False
    
    def test_data_quality_with_gaps(self):
        """Test data quality detection with gaps."""
        loader = CryptoDataLoader()
        
        # Create data with gaps
        dates1 = pd.date_range('2023-01-01', periods=10, freq='15min', tz='UTC')
        dates2 = pd.date_range('2023-01-02', periods=10, freq='15min', tz='UTC')
        dates = pd.Index(list(dates1) + list(dates2))
        
        df = pd.DataFrame({
            'open': [100] * 20,
            'high': [105] * 20,
            'low': [95] * 20,
            'close': [102] * 20,
            'volume': [1000] * 20
        }, index=dates)
        
        metrics = loader.check_data_quality(df)
        
        assert metrics['gaps_detected'] > 0
        assert metrics['largest_gap'] > pd.Timedelta(hours=1)


class TestCryptoDataLoaderIntegration:
    """Integration tests for data loader."""
    
    @pytest.mark.slow
    @patch('src.data.binance_loader.ccxt.binance')
    def test_full_pipeline_with_mock_exchange(self, mock_exchange_class):
        """Test full data fetching pipeline with mocked exchange."""
        # Setup mock exchange
        mock_exchange = MagicMock()
        mock_exchange_class.return_value = mock_exchange
        
        # Create realistic OHLCV data
        n_candles = 100
        timestamps = [1609459200000 + i * 900000 for i in range(n_candles)]  # 15min intervals
        prices = 30000 + np.cumsum(np.random.randn(n_candles) * 100)
        
        mock_ohlcv = []
        for i, ts in enumerate(timestamps):
            o = prices[i] + np.random.randn() * 50
            h = prices[i] + abs(np.random.randn() * 100)
            l = prices[i] - abs(np.random.randn() * 100)
            c = prices[i]
            v = abs(np.random.randn() * 1000000) + 100000
            mock_ohlcv.append([ts, o, h, l, c, v])
        
        mock_exchange.fetch_ohlcv.return_value = mock_ohlcv
        
        # Test with temp cache
        with tempfile.TemporaryDirectory() as temp_dir:
            loader = CryptoDataLoader(cache_dir=temp_dir)
            loader.exchange = mock_exchange
            
            df = loader.fetch_ohlcv("BTCUSDT", "15m", "2021-01-01", "2021-01-02")
            
            # Validate result
            assert len(df) == n_candles
            assert df.index.is_monotonic_increasing
            assert not df.index.has_duplicates
            assert all(df['high'] >= df['low'])
            assert all(df['high'] >= df['open'])
            assert all(df['low'] <= df['open'])
            
            # Check cache was created
            cache_path = loader._get_cache_path("BTCUSDT", "15m")
            assert cache_path.exists()
    
    def test_error_handling(self):
        """Test error handling in various scenarios."""
        loader = CryptoDataLoader()
        loader.exchange = None  # No exchange available
        
        # Mock yfinance to also fail
        with patch('src.data.binance_loader.yf.Ticker') as mock_ticker:
            mock_ticker.return_value.history.return_value = pd.DataFrame()
            
            # Should raise error when both sources fail
            with pytest.raises(ValueError, match="Não foi possível"):
                loader.fetch_ohlcv("INVALID", "15m", "2023-01-01", "2023-01-02", validate=False)