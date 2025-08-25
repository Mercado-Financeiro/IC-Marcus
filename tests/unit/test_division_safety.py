"""Tests for division safety across the codebase."""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock

from src.utils.logging_config import safe_divide
from src.backtest.engine import BacktestEngine
from src.features.adaptive_labeling import AdaptiveLabeler


class TestSafeDivideFunction:
    """Test the safe_divide utility function comprehensively."""
    
    def test_basic_operations(self):
        """Test basic division operations."""
        assert safe_divide(10, 2) == 5.0
        assert safe_divide(100, 10) == 10.0
        assert safe_divide(-10, 2) == -5.0
        assert safe_divide(10, -2) == -5.0
        
    def test_zero_division(self):
        """Test division by zero."""
        assert safe_divide(10, 0, default=0) == 0
        assert safe_divide(10, 0, default=-1) == -1
        assert safe_divide(0, 0, default=999) == 999
        
    def test_near_zero_division(self):
        """Test division by numbers close to zero."""
        assert safe_divide(10, 1e-11, default=0) == 0
        assert safe_divide(10, -1e-11, default=0) == 0
        assert safe_divide(10, 1e-9, default=0) != 0  # Should work
        
    def test_special_values(self):
        """Test with special float values."""
        assert safe_divide(np.inf, 10, default=0) != 0  # inf/10 is valid
        assert safe_divide(10, np.inf, default=0) == 0  # 10/inf → 0
        assert safe_divide(np.nan, 10, default=0) == 0
        assert safe_divide(10, np.nan, default=0) == 0
        
    def test_logging_on_edge_cases(self):
        """Test that logging occurs on edge cases."""
        with patch('src.utils.logging_config.get_logger') as mock_logger:
            mock_log = MagicMock()
            mock_logger.return_value = mock_log
            
            # Division by zero
            safe_divide(10, 0)
            mock_log.warning.assert_called()
            
            # Near zero denominator
            safe_divide(10, 1e-11)
            assert mock_log.warning.call_count >= 1


class TestBacktestDivisionSafety:
    """Test division safety in backtest engine."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.engine = BacktestEngine()
        
    def test_position_sizing_divisions(self):
        """Test position sizing with edge case divisions."""
        # Test volatility target with zero volatility
        size = self.engine.calculate_position_size(
            method='volatility_target',
            capital=10000,
            signal_strength=1,
            volatility=0,
            vol_target=0.2
        )
        assert size > 0  # Should fallback to max leverage
        
        # Test Kelly with edge probabilities
        size = self.engine.calculate_position_size(
            method='kelly',
            capital=10000,
            signal_strength=1,
            p=0,  # Zero win probability
            q=1
        )
        assert size == 0  # Kelly should be 0 with p=0
        
    def test_cost_calculations_divisions(self):
        """Test cost calculations with division edge cases."""
        # Test with zero holding period
        costs = self.engine.calculate_costs(
            trade_value=10000,
            side=1,
            holding_period=0
        )
        assert costs >= 0  # Should handle zero holding period
        
        # Test with extreme holding period
        costs = self.engine.calculate_costs(
            trade_value=10000,
            side=1,
            holding_period=1e10
        )
        assert np.isfinite(costs)  # Should remain finite
        
    def test_metrics_divisions(self):
        """Test metric calculations with division edge cases."""
        # Create results with edge cases
        results = pd.DataFrame({
            'equity': [10000, 10000, 10000],  # No change
            'pnl': [0, 0, 0],
            'trades': [0, 0, 0],
            'realized_pnl': [0, 0, 0]
        })
        
        metrics = self.engine.calculate_metrics(results)
        
        # All ratios should handle zero denominators
        assert metrics['sharpe_ratio'] == 0
        assert metrics['sortino_ratio'] == 0
        assert metrics['calmar_ratio'] == 0
        assert metrics['win_rate'] == 0
        assert metrics['profit_factor'] == 0
        
    def test_slippage_calculation(self):
        """Test slippage calculation with extreme values."""
        # Test with zero slippage basis points
        engine = BacktestEngine(slippage_bps=0)
        
        df = pd.DataFrame({
            'open': [100, 101, 102],
            'high': [105, 106, 107],
            'low': [95, 96, 97],
            'close': [100, 101, 102],
            'volume': [1000, 1100, 1200]
        }, index=pd.date_range('2024-01-01', periods=3, freq='15min'))
        
        signals = pd.Series([0, 1, -1], index=df.index)
        results = engine.run_backtest(df, signals)
        
        # Should work with zero slippage
        assert results is not None
        
    def test_annualization_divisions(self):
        """Test annualization with edge case periods."""
        # Test with very short period (1 bar)
        results = pd.DataFrame({
            'equity': [10000, 10100]
        })
        
        metrics = self.engine.calculate_metrics(results)
        
        # Should handle short periods
        assert np.isfinite(metrics['annualized_return'])
        assert metrics['annualized_return'] >= 0


class TestAdaptiveLabelingDivisionSafety:
    """Test division safety in adaptive labeling."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.labeler = AdaptiveLabeler()
        
    def test_threshold_calculation_divisions(self):
        """Test threshold calculation with edge cases."""
        # Create data with zero volatility
        df = pd.DataFrame({
            'open': [100] * 20,
            'high': [100] * 20,
            'low': [100] * 20,
            'close': [100] * 20
        })
        
        # Should handle zero volatility
        threshold = self.labeler.calculate_adaptive_threshold(df)
        assert threshold is not None
        assert (threshold >= 0.001).all()  # Should be clipped to minimum
        
    def test_label_creation_divisions(self):
        """Test label creation with price edge cases."""
        # Data with some zero prices
        df = pd.DataFrame({
            'open': [100, 100, 100, 100, 100],
            'high': [105, 105, 105, 105, 105],
            'low': [95, 95, 95, 95, 95],
            'close': [100, 0, 100, 100, 100]  # Zero price
        })
        
        with patch('src.utils.logging_config.log'):
            labels = self.labeler.create_labels(df)
            
        # Should handle zero prices
        assert labels is not None
        
    def test_balance_ratio_calculation(self):
        """Test balance ratio with edge cases."""
        # Empty labels
        labels = pd.Series([])
        result = self.labeler.get_label_distribution(labels)
        assert result['balance_ratio'] == 0
        
        # Single class
        labels = pd.Series([1, 1, 1])
        result = self.labeler.get_label_distribution(labels)
        assert result['balance_ratio'] == 0
        
        # Extreme imbalance
        labels = pd.Series([1] * 1000 + [-1])
        result = self.labeler.get_label_distribution(labels)
        assert 0 <= result['balance_ratio'] <= 1
        
    def test_optimization_score_divisions(self):
        """Test k optimization with edge case scores."""
        df = pd.DataFrame({
            'open': np.random.randn(100) + 100,
            'high': np.random.randn(100) + 105,
            'low': np.random.randn(100) + 95,
            'close': np.random.randn(100) + 100
        })
        X = pd.DataFrame(np.random.randn(100, 5))
        
        # Mock the model to return extreme scores
        with patch('sklearn.ensemble.RandomForestClassifier') as mock_rf:
            mock_model = MagicMock()
            mock_model.predict.return_value = np.array([0] * 10)
            mock_model.predict_proba.return_value = np.array([[1, 0]] * 10)
            mock_rf.return_value = mock_model
            
            # Should handle edge scores
            result = self.labeler.optimize_k_for_horizon(df, X, '15m', cv_splits=2)
            assert np.isfinite(result)


class TestDivisionSafetyIntegration:
    """Integration tests for division safety."""
    
    def test_full_pipeline_zero_prices(self):
        """Test full pipeline with zero prices."""
        # Create data with zero prices
        df = pd.DataFrame({
            'open': [100, 0, 100, 100, 100, 100],
            'high': [105, 0, 105, 105, 105, 105],
            'low': [95, 0, 95, 95, 95, 95],
            'close': [100, 0, 100, 100, 100, 100],
            'volume': [1000, 0, 1000, 1000, 1000, 1000]
        }, index=pd.date_range('2024-01-01', periods=6, freq='15min'))
        
        # Create labeler and labels
        labeler = AdaptiveLabeler(horizon_bars=1)
        with patch('src.utils.logging_config.log'):
            labels = labeler.create_labels(df)
        
        # Run backtest
        engine = BacktestEngine()
        signals = labels.fillna(0)
        results = engine.run_backtest(df, signals)
        metrics = engine.calculate_metrics(results)
        
        # Should complete despite zero prices
        assert results is not None
        assert metrics is not None
        
    def test_pipeline_constant_prices(self):
        """Test pipeline with constant prices (zero volatility)."""
        # Create data with constant prices
        df = pd.DataFrame({
            'open': [100] * 20,
            'high': [100] * 20,
            'low': [100] * 20,
            'close': [100] * 20,
            'volume': [1000] * 20
        }, index=pd.date_range('2024-01-01', periods=20, freq='15min'))
        
        # Create labeler and labels
        labeler = AdaptiveLabeler(horizon_bars=2)
        labels = labeler.create_labels(df)
        
        # Run backtest
        engine = BacktestEngine()
        signals = labels.fillna(0)
        results = engine.run_backtest(df, signals)
        metrics = engine.calculate_metrics(results)
        
        # Should handle zero volatility
        assert metrics['sharpe_ratio'] == 0
        assert metrics['sortino_ratio'] == 0
        
    def test_pipeline_extreme_ratios(self):
        """Test pipeline with extreme price ratios."""
        # Create data with extreme ratios
        df = pd.DataFrame({
            'open': [1e-10, 1e10, 100, 100, 100, 100],
            'high': [1e-10, 1e10, 105, 105, 105, 105],
            'low': [1e-10, 1e10, 95, 95, 95, 95],
            'close': [1e-10, 1e10, 100, 100, 100, 100],
            'volume': [1, 1e10, 1000, 1000, 1000, 1000]
        }, index=pd.date_range('2024-01-01', periods=6, freq='15min'))
        
        # Should handle extreme ratios
        labeler = AdaptiveLabeler(horizon_bars=1)
        with patch('src.utils.logging_config.log'):
            labels = labeler.create_labels(df)
            
        assert labels is not None
        
        # Run backtest with extreme values
        engine = BacktestEngine()
        signals = pd.Series([0, 0, 1, -1, 0, 0], index=df.index)
        results = engine.run_backtest(df, signals)
        
        # Should complete without crashing
        assert results is not None