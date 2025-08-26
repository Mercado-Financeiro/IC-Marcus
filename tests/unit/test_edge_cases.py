"""Comprehensive tests for edge cases and extreme conditions."""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock

from src.features.adaptive_labeling import AdaptiveLabeler
from src.utils.logging_config import safe_divide, validate_dataframe
from src.backtest.engine import BacktestEngine


class TestSafeDivision:
    """Test safe division helper."""
    
    def test_divide_by_zero(self):
        """Test division by zero returns default."""
        result = safe_divide(10, 0, default=0)
        assert result == 0
        
    def test_divide_by_near_zero(self):
        """Test division by very small number."""
        result = safe_divide(10, 1e-11, default=0)
        assert result == 0
        
    def test_normal_division(self):
        """Test normal division works."""
        result = safe_divide(10, 2, default=0)
        assert result == 5.0
        
    def test_infinite_result(self):
        """Test infinite result returns default."""
        with patch('numpy.isfinite', return_value=False):
            result = safe_divide(np.inf, 1, default=0)
            assert result == 0
            
    def test_nan_result(self):
        """Test NaN result returns default."""
        result = safe_divide(np.nan, 1, default=0)
        assert result == 0


class TestDataFrameValidation:
    """Test DataFrame validation."""
    
    def test_empty_dataframe_raises(self):
        """Test empty DataFrame raises error."""
        df = pd.DataFrame()
        with pytest.raises(ValueError, match="is empty"):
            validate_dataframe(df, "test_df")
            
    def test_all_null_columns_dropped(self):
        """Test all-null columns are dropped."""
        df = pd.DataFrame({
            'good': [1, 2, 3],
            'bad': [np.nan, np.nan, np.nan]
        })
        # Mock the logger to avoid actual logging
        with patch('src.utils.logging_config.log'):
            validate_dataframe(df, "test_df")
        assert 'bad' not in df.columns
        assert 'good' in df.columns
        
    def test_infinite_values_replaced(self):
        """Test infinite values are replaced with NaN."""
        df = pd.DataFrame({
            'values': [1, np.inf, -np.inf, 2]
        })
        with patch('src.utils.logging_config.log'):
            validate_dataframe(df, "test_df")
        assert not np.isinf(df['values']).any()
        assert df['values'].isna().sum() == 2  # inf values replaced with NaN


class TestAdaptiveLabelingEdgeCases:
    """Test edge cases in adaptive labeling."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.labeler = AdaptiveLabeler(horizon_bars=4, k=1.0)
        
    def test_empty_dataframe(self):
        """Test with empty DataFrame."""
        df = pd.DataFrame()
        with pytest.raises(ValueError, match="is empty"):
            self.labeler.create_labels(df)
            
    def test_missing_close_column(self):
        """Test with missing close column."""
        df = pd.DataFrame({
            'open': [100, 101, 102],
            'high': [105, 106, 107],
            'low': [95, 96, 97]
        })
        with pytest.raises(ValueError, match="must have 'close' column"):
            self.labeler.create_labels(df)
            
    def test_insufficient_data_for_horizon(self):
        """Test with insufficient data for horizon."""
        df = pd.DataFrame({
            'close': [100, 101]  # Only 2 bars, need at least 5 for horizon_bars=4
        })
        with pytest.raises(ValueError, match="Need at least"):
            self.labeler.create_labels(df)
            
    def test_zero_prices(self):
        """Test with zero prices in close."""
        df = pd.DataFrame({
            'open': [100, 101, 102, 103, 104, 105],
            'high': [105, 106, 107, 108, 109, 110],
            'low': [95, 96, 97, 98, 99, 100],
            'close': [100, 0, 102, 103, 104, 105]  # Zero price
        })
        with patch('src.utils.logging_config.log') as mock_log:
            labels = self.labeler.create_labels(df)
            # Check that warning was logged
            mock_log.warning.assert_called()
            
    def test_all_nan_labels(self):
        """Test get_label_distribution with all NaN labels."""
        labels = pd.Series([np.nan, np.nan, np.nan])
        result = self.labeler.get_label_distribution(labels)
        assert result['total'] == 0
        assert result['balance_ratio'] == 0
        
    def test_single_class_labels(self):
        """Test get_label_distribution with single class."""
        labels = pd.Series([1, 1, 1, 1])
        with patch('src.utils.logging_config.log') as mock_log:
            result = self.labeler.get_label_distribution(labels)
            assert result['balance_ratio'] == 0  # Single class has no balance
            mock_log.warning.assert_called()
            
    def test_severe_imbalance(self):
        """Test get_label_distribution with severe imbalance."""
        labels = pd.Series([1] * 100 + [-1] * 5)  # 95:5 ratio
        with patch('src.utils.logging_config.log') as mock_log:
            result = self.labeler.get_label_distribution(labels)
            assert result['balance_ratio'] < 0.1
            # Check that severe imbalance warning was logged
            mock_log.warning.assert_called()
            
    def test_optimize_k_invalid_params(self):
        """Test optimize_k_for_horizon with invalid parameters."""
        df = pd.DataFrame({
            'open': np.random.randn(100) + 100,
            'high': np.random.randn(100) + 105,
            'low': np.random.randn(100) + 95,
            'close': np.random.randn(100) + 100
        })
        X = pd.DataFrame(np.random.randn(100, 5))
        
        # Test with invalid cv_splits
        with pytest.raises(ValueError, match="cv_splits must be >= 2"):
            self.labeler.optimize_k_for_horizon(df, X, '15m', cv_splits=1)
            
        # Test with invalid k_range
        with pytest.raises(ValueError, match="Invalid k_range"):
            self.labeler.optimize_k_for_horizon(df, X, '15m', k_range=(0, 0))
            
        # Test with unsupported metric
        with pytest.raises(ValueError, match="Metric .* not supported"):
            self.labeler.optimize_k_for_horizon(df, X, '15m', metric='invalid')
            
        # Test with invalid horizon
        with pytest.raises(ValueError, match="Horizon .* not supported"):
            self.labeler.optimize_k_for_horizon(df, X, 'invalid_horizon')
            
    def test_state_restoration_on_error(self):
        """Test that state is restored even on error."""
        df = pd.DataFrame({
            'open': np.random.randn(100) + 100,
            'high': np.random.randn(100) + 105,
            'low': np.random.randn(100) + 95,
            'close': np.random.randn(100) + 100
        })
        X = pd.DataFrame(np.random.randn(100, 5))
        
        original_k = self.labeler.k
        original_horizon = self.labeler.horizon_bars
        
        # Force an error during optimization
        with patch.object(self.labeler, '_create_labels_with_params', side_effect=Exception("Test error")):
            with pytest.raises(Exception, match="Test error"):
                self.labeler.optimize_k_for_horizon(df, X, '15m')
                
        # Check state was restored
        assert self.labeler.k == original_k
        assert self.labeler.horizon_bars == original_horizon


class TestBacktestEngineEdgeCases:
    """Test edge cases in backtest engine."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.engine = BacktestEngine()
        
    def test_zero_initial_capital(self):
        """Test with zero initial capital."""
        engine = BacktestEngine(initial_capital=0)
        df = pd.DataFrame({
            'open': [100, 101, 102],
            'high': [105, 106, 107],
            'low': [95, 96, 97],
            'close': [100, 101, 102],
            'volume': [1000, 1100, 1200]
        }, index=pd.date_range('2024-01-01', periods=3, freq='15min'))
        
        signals = pd.Series([1, -1, 0], index=df.index)
        
        # Should handle zero capital gracefully
        results = engine.run_backtest(df, signals)
        assert results['equity'].iloc[0] == 0
        
    def test_zero_volatility(self):
        """Test position sizing with zero volatility."""
        size = self.engine.calculate_position_size(
            method='volatility_target',
            capital=10000,
            signal_strength=1,
            volatility=0,
            vol_target=0.2
        )
        # Should use max leverage when volatility is zero
        assert size == 10000 * self.engine.max_leverage
        
    def test_zero_execution_price(self):
        """Test with zero execution price."""
        df = pd.DataFrame({
            'open': [100, 0, 102],  # Zero price
            'high': [105, 106, 107],
            'low': [95, 96, 97],
            'close': [100, 101, 102],
            'volume': [1000, 1100, 1200]
        }, index=pd.date_range('2024-01-01', periods=3, freq='15min'))
        
        signals = pd.Series([0, 1, 0], index=df.index)
        
        results = self.engine.run_backtest(df, signals)
        # Position units should be 0 when price is 0
        assert results['position_units'].iloc[1] == 0
        
    def test_extreme_slippage(self):
        """Test with extreme slippage values."""
        engine = BacktestEngine(slippage_bps=10000)  # 100% slippage
        df = pd.DataFrame({
            'open': [100, 101, 102],
            'high': [105, 106, 107],
            'low': [95, 96, 97],
            'close': [100, 101, 102],
            'volume': [1000, 1100, 1200]
        }, index=pd.date_range('2024-01-01', periods=3, freq='15min'))
        
        signals = pd.Series([0, 1, -1], index=df.index)
        
        results = engine.run_backtest(df, signals)
        # Should handle extreme slippage without breaking
        assert results is not None
        
    def test_metrics_with_no_trades(self):
        """Test metrics calculation with no trades."""
        df = pd.DataFrame({
            'open': [100, 101, 102],
            'high': [105, 106, 107],
            'low': [95, 96, 97],
            'close': [100, 101, 102],
            'volume': [1000, 1100, 1200]
        }, index=pd.date_range('2024-01-01', periods=3, freq='15min'))
        
        signals = pd.Series([0, 0, 0], index=df.index)  # No signals
        
        results = self.engine.run_backtest(df, signals)
        metrics = self.engine.calculate_metrics(results)
        
        # Should handle no trades gracefully
        assert metrics['win_rate'] == 0
        assert metrics['profit_factor'] == 0
        assert metrics['total_trades'] == 0
        
    def test_metrics_with_zero_returns(self):
        """Test metrics with zero returns."""
        df = pd.DataFrame({
            'equity': [10000, 10000, 10000],  # No change
            'pnl': [0, 0, 0],
            'trades': [0, 0, 0]
        })
        
        metrics = self.engine.calculate_metrics(df)
        
        # Should handle zero returns
        assert metrics['sharpe_ratio'] == 0
        assert metrics['sortino_ratio'] == 0
        assert metrics['calmar_ratio'] == 0


class TestIntegrationEdgeCases:
    """Integration tests for edge cases across modules."""
    
    def test_full_pipeline_with_minimal_data(self):
        """Test full pipeline with minimal valid data."""
        # Create minimal valid dataset
        df = pd.DataFrame({
            'open': np.random.randn(10) + 100,
            'high': np.random.randn(10) + 105,
            'low': np.random.randn(10) + 95,
            'close': np.random.randn(10) + 100,
            'volume': np.random.randn(10) * 100 + 1000
        }, index=pd.date_range('2024-01-01', periods=10, freq='15min'))
        
        # Create labels
        labeler = AdaptiveLabeler(horizon_bars=2)
        labels = labeler.create_labels(df)
        
        # Create signals from labels
        signals = labels.fillna(0)
        
        # Run backtest
        engine = BacktestEngine()
        results = engine.run_backtest(df, signals)
        metrics = engine.calculate_metrics(results)
        
        # Should complete without errors
        assert results is not None
        assert metrics is not None
        assert 'sharpe_ratio' in metrics
        
    def test_pipeline_with_extreme_values(self):
        """Test pipeline with extreme values."""
        # Create dataset with extreme values
        df = pd.DataFrame({
            'open': [1e-10, 1e10, 100, 100, 100],
            'high': [1e-10, 1e10, 105, 105, 105],
            'low': [1e-10, 1e10, 95, 95, 95],
            'close': [1e-10, 1e10, 100, 100, 100],
            'volume': [1, 1e10, 1000, 1000, 1000]
        }, index=pd.date_range('2024-01-01', periods=5, freq='15min'))
        
        # Should handle extreme values gracefully
        labeler = AdaptiveLabeler(horizon_bars=1)
        with patch('src.utils.logging_config.log'):
            labels = labeler.create_labels(df)
            
        # Check that labels are created despite extreme values
        assert labels is not None
        assert len(labels) == len(df)