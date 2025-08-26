"""
Test file to demonstrate CI/CD workflow for LSTM profit pipeline
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch


def test_threshold_optimizer_import():
    """Test that ThresholdOptimizer can be imported."""
    from src.models.threshold_optimizer import ThresholdOptimizer, TradingCosts
    
    costs = TradingCosts(fee_bps=5.0)
    optimizer = ThresholdOptimizer(costs)
    assert optimizer is not None
    assert optimizer.costs.fee_bps == 5.0


def test_lstm_has_ev_threshold():
    """Test that LSTM optimizer has EV threshold support."""
    from src.models.lstm.optuna.optimizer_v2 import EnhancedLSTMOptuna
    from src.models.lstm.optuna.config import LSTMOptunaConfig
    
    config = LSTMOptunaConfig(seed=42, verbose=False)
    lstm = EnhancedLSTMOptuna(config)
    
    # Check required attributes
    assert hasattr(lstm, 'threshold_optimizer')
    assert hasattr(lstm, 'threshold_ev')
    assert hasattr(lstm, 'threshold_f1')


def test_ensemble_supports_lstm():
    """Test that ensemble supports both XGBoost and LSTM."""
    from src.models.ensemble.multi_horizon import MultiHorizonEnsemble, MultiHorizonConfig
    
    config = MultiHorizonConfig(
        model_types=['xgboost', 'lstm'],
        horizons=[1, 5],
        n_trials_per_model=2
    )
    
    ensemble = MultiHorizonEnsemble(config)
    assert 'lstm' in config.model_types
    assert 'xgboost' in config.model_types
    assert len(config.horizons) == 2


def test_expected_value_calculation():
    """Test EV calculation with realistic parameters."""
    from src.models.threshold_optimizer import ThresholdOptimizer, TradingCosts
    
    # Create test data
    np.random.seed(42)
    y_true = np.array([1, 0, 1, 1, 0, 1, 0, 0, 1, 1])
    y_proba = np.array([0.9, 0.2, 0.8, 0.7, 0.3, 0.6, 0.4, 0.1, 0.85, 0.75])
    
    costs = TradingCosts(fee_bps=5.0, slippage_bps=5.0, impact_bps=2.0)
    optimizer = ThresholdOptimizer(costs)
    
    # Compute EV for a specific threshold
    ev, metrics = optimizer.compute_ev(
        y_true=y_true,
        y_proba=y_proba,
        threshold=0.5,
        avg_win_pct=0.015,
        avg_loss_pct=0.005
    )
    
    assert 'net_return_per_trade' in metrics
    assert 'precision' in metrics
    assert 'n_trades' in metrics
    assert metrics['n_trades'] >= 0


def test_profit_pipeline_scripts_exist():
    """Test that required scripts exist."""
    import os
    
    scripts = [
        'scripts/train_profit_lstm.py',
        'run_profit_pipeline.sh',
        'test_integration_final.py'
    ]
    
    for script in scripts:
        assert os.path.exists(script), f"Script {script} not found"


@pytest.mark.parametrize("model_type", ["xgboost", "lstm"])
def test_model_types(model_type):
    """Test different model types in the pipeline."""
    assert model_type in ["xgboost", "lstm", "ensemble"]


def test_trading_costs_calculation():
    """Test trading costs calculation."""
    from src.models.threshold_optimizer import TradingCosts
    
    costs = TradingCosts(
        fee_bps=5.0,      # 0.05%
        slippage_bps=5.0,  # 0.05%
        impact_bps=2.0     # 0.02%
    )
    
    # Total roundtrip should be 2x the sum
    expected_total = (5.0 + 5.0 + 2.0) * 2
    assert costs.total_bps == expected_total
    assert costs.total_pct == expected_total / 10000


def test_calibration_import():
    """Test that calibration modules can be imported."""
    from src.models.calibration.temperature import TemperatureScaling
    
    # Just verify import works
    assert TemperatureScaling is not None


if __name__ == "__main__":
    # Run tests locally
    pytest.main([__file__, "-v"])