"""Unit tests for enhanced calendar features module."""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch

from src.features.calendar_features import CalendarFeatures, Crypto24x7Features


@pytest.fixture
def sample_datetime_data():
    """Create sample datetime data for testing."""
    dates = pd.date_range(
        start="2024-01-01 00:00:00", 
        end="2024-01-07 23:45:00", 
        freq="15min",
        tz="UTC"
    )
    
    df = pd.DataFrame({
        'close': np.random.uniform(49000, 51000, len(dates)),
        'volume': np.random.uniform(100000, 1000000, len(dates))
    }, index=dates)
    
    return df


class TestCalendarFeatures:
    """Test base calendar features."""
    
    def test_initialization(self):
        """Test CalendarFeatures initialization."""
        cf = CalendarFeatures()
        assert cf.include_crypto_features is True
        
        cf_no_crypto = CalendarFeatures(include_crypto_features=False)
        assert cf_no_crypto.include_crypto_features is False
    
    def test_calculate_time_components(self, sample_datetime_data):
        """Test time component calculation."""
        cf = CalendarFeatures()
        result = cf.calculate_time_components(sample_datetime_data)
        
        # Check that time component features are added
        expected_features = [
            'hour', 'day_of_week', 'day_of_month', 'month',
            'quarter', 'year', 'week_of_year', 'day_of_year'
        ]
        
        for feature in expected_features:
            assert feature in result.columns
        
        # Check data integrity
        assert len(result) == len(sample_datetime_data)
        assert (result['hour'] >= 0).all()
        assert (result['hour'] <= 23).all()
        assert (result['day_of_week'] >= 0).all()
        assert (result['day_of_week'] <= 6).all()
    
    def test_calculate_all(self, sample_datetime_data):
        """Test complete calendar feature calculation."""
        cf = CalendarFeatures()
        result = cf.calculate_all(sample_datetime_data)
        
        # Should include basic calendar and funding features
        assert 'hour' in result.columns  # Time components
        assert 'funding_proximity' in result.columns  # Funding features
        
        # Check data integrity
        assert len(result) == len(sample_datetime_data)
        numeric_cols = result.select_dtypes(include=[np.number]).columns
        assert not result[numeric_cols].isnull().all().any()


class TestFundingFeatures:
    """Test funding-related features."""
    
    def test_create_funding_features(self, sample_datetime_data):
        """Test funding feature creation."""
        from src.features.calendar_features import Crypto24x7Features
        
        result = Crypto24x7Features.create_funding_features(sample_datetime_data)
        
        # Check funding features are present
        expected_features = [
            'minutes_to_funding', 'bars_to_funding', 'funding_proximity',
            'is_funding_time', 'pre_funding_window', 'funding_cycle'
        ]
        
        for feature in expected_features:
            assert feature in result.columns
        
        # Check value ranges
        assert (result['funding_proximity'] >= 0).all()
        assert (result['funding_proximity'] <= 1).all()
        # pre_funding_window is int64 (0 or 1), not bool
        assert set(result['pre_funding_window'].unique()).issubset({0, 1})
    
    def test_different_funding_periods(self, sample_datetime_data):
        """Test with different funding periods."""
        from src.features.calendar_features import Crypto24x7Features
        
        periods_and_expected = [
            (240, 240),   # 4 hours
            (480, 480),   # 8 hours
            (720, 720),   # 12 hours
        ]
        
        for period_minutes, expected_period in periods_and_expected:
            result = Crypto24x7Features.create_funding_features(sample_datetime_data, funding_period_minutes=period_minutes)
            assert 'funding_proximity' in result.columns
            assert len(result) == len(sample_datetime_data)
    
class TestSessionFeatures:
    """Test session-related features."""
    
    def test_create_session_features(self, sample_datetime_data):
        """Test session feature creation."""
        from src.features.calendar_features import Crypto24x7Features
        
        result = Crypto24x7Features.create_session_features(sample_datetime_data)
        
        # Check session features
        expected_features = [
            'session_asia',
            'session_europe', 
            'session_americas',
            'overlap_asia_europe',
            'overlap_europe_americas'
        ]
        
        for feature in expected_features:
            assert feature in result.columns
        
        # Check data integrity
        assert len(result) == len(sample_datetime_data)
    
    def test_edge_cases(self):
        """Test edge cases and error handling."""
        from src.features.calendar_features import Crypto24x7Features
        
        # Empty data with proper datetime index
        empty_df = pd.DataFrame(columns=['close', 'volume'])
        empty_df.index = pd.DatetimeIndex([], tz='UTC')
        result = Crypto24x7Features.create_funding_features(empty_df)
        assert len(result) == 0
        
        # Single timestamp
        single_time = pd.DataFrame({
            'close': [50000],
            'volume': [100000]
        }, index=[pd.Timestamp('2024-01-01 12:00:00', tz='UTC')])
        
        result = Crypto24x7Features.create_funding_features(single_time)
        assert len(result) == 1
        assert 'funding_proximity' in result.columns
    
    @pytest.mark.parametrize("symbol,expected_funding", [
        ("BTCUSDT", 480),
        ("ETHUSDT", 480), 
        ("SOLUSDT", 480),
        ("UNKNOWN", 480)  # Default
    ])
    def test_funding_period_resolution(self, symbol, expected_funding):
        """Test funding period resolution for different symbols."""
        from src.features.adaptive_labeling import resolve_funding_minutes
        
        result = resolve_funding_minutes(symbol)
        assert result == expected_funding