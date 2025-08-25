"""Tests for feature validation system."""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch

from src.features.validation import (
    FeatureValidator, 
    validate_inputs, 
    validate_outputs,
    ColumnMissingError, 
    DataInconsistencyError,
    InsufficientDataError,
    InvalidDataTypeError,
    InvalidRangeError
)


class TestFeatureValidator:
    """Test the central FeatureValidator class."""
    
    def test_validate_columns_exist_success(self):
        """Test successful column validation."""
        df = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
        
        # Should not raise
        FeatureValidator.validate_columns_exist(df, ['a', 'b'])
    
    def test_validate_columns_exist_failure(self):
        """Test column validation failure."""
        df = pd.DataFrame({'a': [1, 2]})
        
        with pytest.raises(ColumnMissingError) as exc_info:
            FeatureValidator.validate_columns_exist(df, ['a', 'missing'], 'test_function')
        
        assert 'missing' in str(exc_info.value)
        assert 'test_function' in str(exc_info.value)
    
    def test_validate_numeric_columns_success(self):
        """Test successful numeric validation."""
        df = pd.DataFrame({
            'int_col': [1, 2, 3],
            'float_col': [1.1, 2.2, 3.3],
            'mixed': [1, 2.5, 3]
        })
        
        # Should not raise
        FeatureValidator.validate_numeric_columns(df, ['int_col', 'float_col', 'mixed'])
    
    def test_validate_numeric_columns_with_nan(self):
        """Test numeric validation with NaN values."""
        df = pd.DataFrame({
            'col_with_nan': [1.0, 2.0, np.nan]
        })
        
        # Should not raise when NaN allowed
        FeatureValidator.validate_numeric_columns(df, ['col_with_nan'], allow_nan=True)
        
        # Should log warning when NaN not allowed but not raise
        FeatureValidator.validate_numeric_columns(df, ['col_with_nan'], allow_nan=False)
    
    def test_validate_numeric_columns_failure(self):
        """Test numeric validation failure."""
        df = pd.DataFrame({'string_col': ['a', 'b', 'c']})
        
        with pytest.raises(InvalidDataTypeError):
            FeatureValidator.validate_numeric_columns(df, ['string_col'])
    
    def test_validate_ohlcv_basic_success(self):
        """Test successful OHLCV validation."""
        df = pd.DataFrame({
            'open': [100.0, 101.0, 102.0],
            'high': [105.0, 106.0, 107.0], 
            'low': [95.0, 96.0, 97.0],
            'close': [103.0, 104.0, 105.0],
            'volume': [1000, 1100, 1200]
        })
        
        # Should not raise
        FeatureValidator.validate_ohlcv_basic(df)
    
    def test_validate_ohlcv_basic_missing_columns(self):
        """Test OHLCV validation with missing columns."""
        df = pd.DataFrame({
            'open': [100.0, 101.0],
            'high': [105.0, 106.0]
            # Missing low, close, volume
        })
        
        with pytest.raises(ColumnMissingError):
            FeatureValidator.validate_ohlcv_basic(df)
    
    def test_validate_ohlc_consistency_success(self):
        """Test successful OHLC consistency validation."""
        df = pd.DataFrame({
            'open': [100.0, 101.0, 102.0],
            'high': [105.0, 106.0, 107.0],  # high >= open, close
            'low': [95.0, 96.0, 97.0],      # low <= open, close
            'close': [103.0, 104.0, 105.0]
        })
        
        # Should not raise
        FeatureValidator.validate_ohlc_consistency(df, strict=False)
        FeatureValidator.validate_ohlc_consistency(df, strict=True)
    
    def test_validate_ohlc_consistency_failure(self):
        """Test OHLC consistency validation failure."""
        df = pd.DataFrame({
            'open': [100.0, 101.0],
            'high': [95.0, 96.0],   # high < open (inconsistent!)
            'low': [105.0, 106.0],  # low > open (inconsistent!)
            'close': [103.0, 104.0]
        })
        
        # Should not raise in non-strict mode (warnings only)
        FeatureValidator.validate_ohlc_consistency(df, strict=False)
        
        # Should raise in strict mode
        with pytest.raises(DataInconsistencyError):
            FeatureValidator.validate_ohlc_consistency(df, strict=True)
    
    def test_validate_volume_data_success(self):
        """Test successful volume validation."""
        df = pd.DataFrame({'volume': [1000, 1100, 1200]})
        
        # Should not raise
        FeatureValidator.validate_volume_data(df, strict=False)
        FeatureValidator.validate_volume_data(df, strict=True)
    
    def test_validate_volume_data_negative(self):
        """Test volume validation with negative values."""
        df = pd.DataFrame({'volume': [1000, -100, 1200]})  # Negative volume
        
        # Should not raise in non-strict mode (warnings only)
        FeatureValidator.validate_volume_data(df, strict=False)
        
        # Should raise in strict mode
        with pytest.raises(DataInconsistencyError):
            FeatureValidator.validate_volume_data(df, strict=True)
    
    def test_validate_sufficient_data_success(self):
        """Test sufficient data validation success."""
        df = pd.DataFrame({'a': [1, 2, 3, 4, 5]})
        
        # Should not raise
        FeatureValidator.validate_sufficient_data(df, min_rows=3)
        FeatureValidator.validate_sufficient_data(df, min_rows=5)
    
    def test_validate_sufficient_data_failure(self):
        """Test insufficient data validation."""
        df = pd.DataFrame({'a': [1, 2]})
        
        with pytest.raises(InsufficientDataError) as exc_info:
            FeatureValidator.validate_sufficient_data(df, min_rows=5, function_name='test_func')
        
        assert 'test_func' in str(exc_info.value)
        assert '5' in str(exc_info.value)
        assert '2' in str(exc_info.value)
    
    def test_validate_lookback_periods(self):
        """Test lookback periods validation and filtering."""
        df_length = 50
        periods = [10, 20, 30, 100, 200]  # Some exceed df_length
        
        valid_periods = FeatureValidator.validate_lookback_periods(
            periods, df_length, 'test_func'
        )
        
        assert valid_periods == [10, 20, 30]
        assert 100 not in valid_periods
        assert 200 not in valid_periods
    
    def test_validate_returns_data_success(self):
        """Test returns validation success."""
        df = pd.DataFrame({
            'returns': np.random.normal(0, 0.02, 100)  # Normal returns ~2% std
        })
        
        # Should not raise
        FeatureValidator.validate_returns_data(df, 'returns')
    
    def test_validate_returns_data_extreme(self):
        """Test returns validation with extreme values."""
        df = pd.DataFrame({
            'returns': [0.01, 0.02, 0.8, 0.03]  # 80% return is extreme
        })
        
        with pytest.raises(DataInconsistencyError) as exc_info:
            FeatureValidator.validate_returns_data(df, 'returns')
        
        assert 'extreme' in str(exc_info.value).lower()
    
    def test_validate_feature_output_success(self):
        """Test feature output validation success."""
        input_df = pd.DataFrame({'a': [1, 2, 3]})
        output_df = pd.DataFrame({
            'a': [1, 2, 3],           # Original column preserved
            'new_feature': [4, 5, 6]  # New feature added
        })
        
        # Should not raise
        FeatureValidator.validate_feature_output(
            input_df, output_df, expected_new_cols=['new_feature']
        )
    
    def test_validate_feature_output_missing_original(self):
        """Test output validation when original columns are missing."""
        input_df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        output_df = pd.DataFrame({'a': [1, 2, 3]})  # Missing column 'b'
        
        with pytest.raises(DataInconsistencyError) as exc_info:
            FeatureValidator.validate_feature_output(input_df, output_df)
        
        assert 'missing in output' in str(exc_info.value).lower()


class TestValidationDecorators:
    """Test validation decorators."""
    
    def test_validate_inputs_decorator_success(self):
        """Test successful input validation decorator."""
        
        class TestClass:
            @validate_inputs(['volume'], min_rows=2)
            def test_method(self, df):
                return df.copy()
        
        df = pd.DataFrame({'volume': [100, 200, 300]})
        test_obj = TestClass()
        
        # Should not raise
        result = test_obj.test_method(df)
        assert len(result) == 3
    
    def test_validate_inputs_decorator_failure(self):
        """Test input validation decorator failure."""
        
        class TestClass:
            @validate_inputs(['missing_col'], min_rows=2)
            def test_method(self, df):
                return df.copy()
        
        df = pd.DataFrame({'volume': [100, 200, 300]})
        test_obj = TestClass()
        
        with pytest.raises(ColumnMissingError):
            test_obj.test_method(df)
    
    def test_validate_inputs_insufficient_data(self):
        """Test input validation with insufficient data."""
        
        class TestClass:
            @validate_inputs(['volume'], min_rows=10)
            def test_method(self, df):
                return df.copy()
        
        df = pd.DataFrame({'volume': [100, 200]})  # Only 2 rows
        test_obj = TestClass()
        
        with pytest.raises(InsufficientDataError):
            test_obj.test_method(df)
    
    def test_validate_outputs_decorator(self):
        """Test output validation decorator."""
        
        class TestClass:
            @validate_outputs(['new_col'], allow_row_reduction=False)
            def test_method(self, df):
                df = df.copy()
                df['new_col'] = df['volume'] * 2
                return df
        
        df = pd.DataFrame({'volume': [100, 200, 300]})
        test_obj = TestClass()
        
        # Should not raise and should validate output
        result = test_obj.test_method(df)
        assert 'new_col' in result.columns
        assert len(result) == 3
    
    @patch('src.features.validation.decorators.log')
    def test_validate_inputs_with_ohlcv(self, mock_log):
        """Test OHLCV validation in decorator."""
        
        class TestClass:
            @validate_inputs(['open', 'high', 'low', 'close', 'volume'], 
                           validate_ohlcv=True, validate_consistency=True)
            def test_method(self, df):
                return df.copy()
        
        df = pd.DataFrame({
            'open': [100.0, 101.0],
            'high': [105.0, 106.0],
            'low': [95.0, 96.0],
            'close': [103.0, 104.0],
            'volume': [1000, 1100]  # Added volume
        })
        
        test_obj = TestClass()
        result = test_obj.test_method(df)
        
        # Should succeed
        assert len(result) == 2
    
    def test_dependent_columns_warning(self):
        """Test that dependent columns generate warnings when missing."""
        
        class TestClass:
            @validate_inputs(['volume'], dependent_cols=['volume_sma_20'])
            def test_method(self, df):
                return df.copy()
        
        df = pd.DataFrame({'volume': [100, 200, 300]})  # Missing volume_sma_20
        test_obj = TestClass()
        
        # Should not raise but should generate warning
        with patch('src.features.validation.decorators.log') as mock_log:
            result = test_obj.test_method(df)
            
            # Verify warning was logged
            mock_log.warning.assert_called()
            call_args = mock_log.warning.call_args[1]
            # Check that the warning contains the missing dependent column
            assert 'volume_sma_20' in call_args['missing']


class TestValidationIntegration:
    """Integration tests with real feature classes."""
    
    def test_volume_features_with_validation(self):
        """Test that VolumeFeatures works with validation."""
        from src.features.microstructure.volume import VolumeFeatures
        
        # Valid data
        df = pd.DataFrame({
            'close': [100.0, 101.0, 102.0, 103.0, 104.0] * 4,  # 20 rows
            'volume': [1000, 1100, 1200, 1300, 1400] * 4
        }, index=pd.date_range('2023-01-01', periods=20, freq='15min'))
        
        volume_calc = VolumeFeatures([5, 10])
        result = volume_calc.calculate_all(df)
        
        # Should work and create features
        assert len(result) >= len(df)
        assert 'dollar_volume' in result.columns
        assert 'volume_sma_5' in result.columns
    
    def test_volume_features_validation_failure(self):
        """Test that VolumeFeatures validation catches errors."""
        from src.features.microstructure.volume import VolumeFeatures
        
        # Invalid data (insufficient rows - VolumeFeatures requires min_rows=10)
        df = pd.DataFrame({
            'close': [100.0, 101.0, 102.0],
            'volume': [1000, 1100, 1200]
        })
        
        volume_calc = VolumeFeatures()
        
        with pytest.raises(InsufficientDataError):
            volume_calc.calculate_all(df)
    
    def test_technical_indicators_validation(self):
        """Test TechnicalIndicators with validation."""
        from src.features.technical_indicators import TechnicalIndicators
        
        # Valid OHLCV data with sufficient rows for technical indicators
        np.random.seed(42)  # For reproducibility
        df = pd.DataFrame({
            'open': np.random.randn(30) * 2 + 100,
            'high': np.random.randn(30) * 2 + 105,
            'low': np.random.randn(30) * 2 + 95, 
            'close': np.random.randn(30) * 2 + 100,
            'volume': np.random.exponential(1000, 30)
        }, index=pd.date_range('2023-01-01', periods=30, freq='15min'))
        
        # Ensure OHLC consistency
        df['high'] = np.maximum(df['high'], df[['open', 'close']].max(axis=1))
        df['low'] = np.minimum(df['low'], df[['open', 'close']].min(axis=1))
        
        tech_calc = TechnicalIndicators(['rsi'])
        result = tech_calc.calculate_all(df)
        
        # Should work and create RSI features
        assert 'rsi_14' in result.columns or 'rsi_7' in result.columns


if __name__ == '__main__':
    pytest.main([__file__])