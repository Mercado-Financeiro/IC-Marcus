"""Pytest configuration and shared fixtures."""

import sys
from pathlib import Path
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import tempfile
import shutil

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))


# Set deterministic seeds for tests
@pytest.fixture(autouse=True)
def set_seeds():
    """Set all random seeds for reproducible tests."""
    import random
    import os
    
    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    os.environ['PYTHONHASHSEED'] = str(SEED)
    
    try:
        import torch
        torch.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
    
    yield
    
    # Reset after test
    np.random.seed(None)


@pytest.fixture
def sample_ohlcv_data():
    """Generate sample OHLCV data for testing."""
    dates = pd.date_range(
        start='2023-01-01', 
        end='2023-12-31', 
        freq='15min',
        tz='UTC'
    )
    
    n = len(dates)
    
    # Generate realistic price data
    close = 30000 + np.cumsum(np.random.randn(n) * 100)
    high = close + np.abs(np.random.randn(n) * 50)
    low = close - np.abs(np.random.randn(n) * 50)
    open_ = close + np.random.randn(n) * 30
    volume = np.abs(np.random.randn(n) * 1000000) + 100000
    
    df = pd.DataFrame({
        'open': open_,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume
    }, index=dates)
    
    # Ensure high >= low, open, close
    df['high'] = df[['high', 'low', 'open', 'close']].max(axis=1)
    df['low'] = df[['high', 'low', 'open', 'close']].min(axis=1)
    
    return df


@pytest.fixture
def sample_features_data(sample_ohlcv_data):
    """Generate sample data with features."""
    df = sample_ohlcv_data.copy()
    
    # Add some basic features
    df['returns'] = df['close'].pct_change()
    df['sma_20'] = df['close'].rolling(20).mean()
    df['volatility_20'] = df['returns'].rolling(20).std()
    df['rsi_14'] = 50 + np.random.randn(len(df)) * 10  # Simplified RSI
    df['volume_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
    
    # Add label
    df['label'] = np.random.choice([-1, 0, 1], size=len(df))
    
    return df.dropna()


@pytest.fixture
def sample_predictions():
    """Generate sample predictions for testing."""
    n = 1000
    y_true = np.random.choice([0, 1], size=n, p=[0.4, 0.6])
    y_pred_proba = np.random.beta(2, 2, size=n)
    
    # Make probabilities somewhat correlated with true labels
    y_pred_proba[y_true == 1] += 0.2
    y_pred_proba = np.clip(y_pred_proba, 0, 1)
    
    return y_true, y_pred_proba


@pytest.fixture
def temp_cache_dir():
    """Create temporary directory for cache testing."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def mlflow_test_dir():
    """Create temporary MLflow directory."""
    temp_dir = tempfile.mkdtemp()
    import mlflow
    mlflow.set_tracking_uri(f"file://{temp_dir}")
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def mock_config():
    """Mock project configuration."""
    from dataclasses import dataclass
    from typing import List
    
    @dataclass
    class MockConfig:
        symbols: List[str] = None
        timeframes: List[str] = None
        start_date: str = "2023-01-01"
        end_date: str = "2023-12-31"
        lookback_periods: List[int] = None
        n_splits: int = 3
        embargo: int = 5
        test_size: float = 0.2
        fee_bps: float = 5.0
        slippage_bps: float = 5.0
        
        def __post_init__(self):
            if self.symbols is None:
                self.symbols = ["BTCUSDT"]
            if self.timeframes is None:
                self.timeframes = ["15m"]
            if self.lookback_periods is None:
                self.lookback_periods = [5, 10, 20]
    
    return MockConfig()


# Markers for different test types
def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "determinism: marks tests that verify deterministic behavior"
    )
    config.addinivalue_line(
        "markers", "no_leak: marks tests that verify no temporal leakage"
    )


# Hypothesis settings (optional)
try:
    from hypothesis import settings, Verbosity
    
    settings.register_profile(
        "dev",
        max_examples=10,
        verbosity=Verbosity.verbose,
        deadline=None
    )
    
    settings.register_profile(
        "ci",
        max_examples=100,
        verbosity=Verbosity.normal,
        deadline=5000
    )
    
    # Use dev profile by default
    settings.load_profile("dev")
except ImportError:
    pass  # Hypothesis is optional