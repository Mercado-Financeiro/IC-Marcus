"""Unit tests for XGBoost threshold optimization."""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch

from src.models.xgb.threshold import ThresholdOptimizer
from src.models.xgb.config import ThresholdConfig


class TestThresholdConfig:
    """Test cases for ThresholdConfig."""
    
    def test_default_config(self):
        """Test default threshold configuration."""
        config = ThresholdConfig()
        
        assert config.optimize_f1 is True
        assert config.optimize_ev is True
        assert config.optimize_profit is True
        assert config.threshold_min == 0.1
        assert config.threshold_max == 0.9
        assert config.transaction_cost_bps == 10
    
    def test_custom_config(self):
        """Test custom threshold configuration."""
        config = ThresholdConfig(
            optimize_f1=False,
            transaction_cost_bps=20,
            kelly_fraction=0.5
        )
        
        assert config.optimize_f1 is False
        assert config.transaction_cost_bps == 20
        assert config.kelly_fraction == 0.5


class TestThresholdOptimizer:
    """Test cases for ThresholdOptimizer."""
    
    @pytest.fixture
    def sample_predictions(self):
        """Create sample predictions and labels."""
        np.random.seed(42)
        n_samples = 1000
        
        # Create realistic probabilities
        y_true = np.random.randint(0, 2, n_samples)
        y_pred_proba = np.random.beta(2, 2, n_samples)
        
        # Make predictions somewhat correlated with truth
        y_pred_proba[y_true == 1] += np.random.normal(0.2, 0.1, (y_true == 1).sum())
        y_pred_proba = np.clip(y_pred_proba, 0, 1)
        
        # Generate returns
        returns = np.random.normal(0.001, 0.02, n_samples)
        returns[y_true == 1] += 0.01  # Positive bias for correct predictions
        
        return y_true, y_pred_proba, returns
    
    @pytest.fixture
    def optimizer(self):
        """Create threshold optimizer."""
        config = ThresholdConfig()
        return ThresholdOptimizer(config)
    
    def test_initialization(self, optimizer):
        """Test optimizer initialization."""
        assert optimizer.threshold_f1 == 0.5
        assert optimizer.threshold_ev == 0.5
        assert optimizer.threshold_profit == 0.5
        assert optimizer.optimization_results == {}
    
    def test_optimize_f1_threshold(self, optimizer, sample_predictions):
        """Test F1 score threshold optimization."""
        y_true, y_pred_proba, _ = sample_predictions
        
        threshold = optimizer.optimize_f1_threshold(y_true, y_pred_proba)
        
        assert 0.0 <= threshold <= 1.0
        assert threshold != 0.5  # Should find a better threshold than default
        
        # Verify F1 is better at optimized threshold
        from sklearn.metrics import f1_score
        
        f1_default = f1_score(y_true, y_pred_proba >= 0.5)
        f1_optimized = f1_score(y_true, y_pred_proba >= threshold)
        
        assert f1_optimized >= f1_default
    
    def test_optimize_ev_threshold(self, optimizer, sample_predictions):
        """Test expected value threshold optimization."""
        y_true, y_pred_proba, returns = sample_predictions
        
        threshold = optimizer.optimize_ev_threshold(y_true, y_pred_proba, returns)
        
        assert optimizer.config.threshold_min <= threshold <= optimizer.config.threshold_max
        
        # Calculate EV at optimized threshold
        predictions = (y_pred_proba >= threshold).astype(int)
        strategy_returns = predictions * returns
        
        # Should have positive or reasonable EV
        assert strategy_returns.mean() > -0.01  # Not too negative
    
    def test_optimize_profit_threshold(self, optimizer, sample_predictions):
        """Test profit threshold optimization with Kelly criterion."""
        y_true, y_pred_proba, returns = sample_predictions
        
        threshold = optimizer.optimize_profit_threshold(y_true, y_pred_proba, returns)
        
        assert optimizer.config.threshold_min <= threshold <= optimizer.config.threshold_max
    
    def test_optimize_all(self, optimizer, sample_predictions):
        """Test optimizing all thresholds."""
        y_true, y_pred_proba, returns = sample_predictions
        
        results = optimizer.optimize_all(y_true, y_pred_proba, returns)
        
        assert 'threshold_f1' in results
        assert 'threshold_ev' in results
        assert 'threshold_profit' in results
        
        # Check that optimizer stores results
        assert optimizer.threshold_f1 == results['threshold_f1']
        assert optimizer.threshold_ev == results['threshold_ev']
        assert optimizer.threshold_profit == results['threshold_profit']
    
    def test_optimize_all_without_returns(self, optimizer, sample_predictions):
        """Test optimization without returns (only F1)."""
        y_true, y_pred_proba, _ = sample_predictions
        
        results = optimizer.optimize_all(y_true, y_pred_proba, returns=None)
        
        assert 'threshold_f1' in results
        assert 'threshold_ev' not in results
        assert 'threshold_profit' not in results
    
    def test_optimize_with_config_disabled(self, sample_predictions):
        """Test optimization with some methods disabled."""
        y_true, y_pred_proba, returns = sample_predictions
        
        config = ThresholdConfig(
            optimize_f1=True,
            optimize_ev=False,
            optimize_profit=False
        )
        optimizer = ThresholdOptimizer(config)
        
        results = optimizer.optimize_all(y_true, y_pred_proba, returns)
        
        assert 'threshold_f1' in results
        assert 'threshold_ev' not in results
        assert 'threshold_profit' not in results
    
    def test_evaluate_threshold(self, optimizer, sample_predictions):
        """Test threshold evaluation."""
        y_true, y_pred_proba, returns = sample_predictions
        
        metrics = optimizer.evaluate_threshold(
            y_true, y_pred_proba, 0.5, returns
        )
        
        assert 'threshold' in metrics
        assert metrics['threshold'] == 0.5
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1_score' in metrics
        assert 'mcc' in metrics
        assert 'mean_return' in metrics
        assert 'sharpe_ratio' in metrics
        
        # Check metric ranges
        assert 0 <= metrics['precision'] <= 1
        assert 0 <= metrics['recall'] <= 1
        assert 0 <= metrics['f1_score'] <= 1
        assert -1 <= metrics['mcc'] <= 1
    
    def test_evaluate_threshold_without_returns(self, optimizer, sample_predictions):
        """Test threshold evaluation without returns."""
        y_true, y_pred_proba, _ = sample_predictions
        
        metrics = optimizer.evaluate_threshold(
            y_true, y_pred_proba, 0.5, returns=None
        )
        
        assert 'mean_return' not in metrics
        assert 'sharpe_ratio' not in metrics
        assert 'f1_score' in metrics
    
    def test_get_threshold_analysis(self, optimizer, sample_predictions):
        """Test threshold analysis across multiple thresholds."""
        y_true, y_pred_proba, returns = sample_predictions
        
        analysis = optimizer.get_threshold_analysis(
            y_true, y_pred_proba, returns
        )
        
        assert isinstance(analysis, pd.DataFrame)
        assert len(analysis) == 17  # Default is 17 thresholds
        assert 'threshold' in analysis.columns
        assert 'f1_score' in analysis.columns
        assert 'precision' in analysis.columns
        assert 'recall' in analysis.columns
        
        # Check thresholds are ordered
        assert analysis['threshold'].is_monotonic_increasing
    
    def test_edge_case_all_zeros(self, optimizer):
        """Test with all predictions being zero."""
        y_true = np.ones(100)
        y_pred_proba = np.zeros(100)
        
        threshold = optimizer.optimize_f1_threshold(y_true, y_pred_proba)
        
        # Should handle gracefully
        assert 0.0 <= threshold <= 1.0
    
    def test_edge_case_all_ones(self, optimizer):
        """Test with all predictions being one."""
        y_true = np.zeros(100)
        y_pred_proba = np.ones(100)
        
        threshold = optimizer.optimize_f1_threshold(y_true, y_pred_proba)
        
        assert 0.0 <= threshold <= 1.0
    
    def test_edge_case_perfect_predictions(self, optimizer):
        """Test with perfect predictions."""
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_pred_proba = np.array([0.1, 0.1, 0.1, 0.9, 0.9, 0.9])
        
        threshold = optimizer.optimize_f1_threshold(y_true, y_pred_proba)
        
        # Should find threshold between 0.1 and 0.9
        assert 0.1 < threshold < 0.9
    
    def test_kelly_criterion_calculation(self, optimizer):
        """Test Kelly criterion in profit optimization."""
        # Create scenario with known win rate
        y_true = np.array([1, 1, 1, 0, 0])  # 60% win rate
        y_pred_proba = np.array([0.7, 0.7, 0.7, 0.7, 0.7])
        returns = np.array([0.02, 0.02, 0.02, -0.01, -0.01])
        
        threshold = optimizer.optimize_profit_threshold(
            y_true, y_pred_proba, returns
        )
        
        assert 0.0 <= threshold <= 1.0
    
    @pytest.mark.parametrize("n_samples", [10, 100, 1000])
    def test_different_sample_sizes(self, optimizer, n_samples):
        """Test with different sample sizes."""
        np.random.seed(42)
        y_true = np.random.randint(0, 2, n_samples)
        y_pred_proba = np.random.rand(n_samples)
        
        threshold = optimizer.optimize_f1_threshold(y_true, y_pred_proba)
        
        assert 0.0 <= threshold <= 1.0
    
    def test_threshold_optimization_deterministic(self, optimizer):
        """Test that optimization is deterministic."""
        np.random.seed(42)
        y_true = np.random.randint(0, 2, 100)
        y_pred_proba = np.random.rand(100)
        
        threshold1 = optimizer.optimize_f1_threshold(y_true, y_pred_proba)
        threshold2 = optimizer.optimize_f1_threshold(y_true, y_pred_proba)
        
        assert threshold1 == threshold2
    
    def test_custom_threshold_range(self, sample_predictions):
        """Test with custom threshold range."""
        y_true, y_pred_proba, _ = sample_predictions
        
        config = ThresholdConfig(
            threshold_min=0.3,
            threshold_max=0.7,
            threshold_steps=41
        )
        optimizer = ThresholdOptimizer(config)
        
        threshold = optimizer.optimize_f1_threshold(y_true, y_pred_proba)
        
        assert 0.3 <= threshold <= 0.7