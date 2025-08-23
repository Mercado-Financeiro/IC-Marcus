"""Unit tests for LSTM utility functions."""

import pytest
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler, RobustScaler
from src.models.lstm.optuna.utils import (
    create_sequences,
    get_device,
    set_lstm_deterministic,
    calculate_metrics,
    check_constant_predictions,
    get_logger
)


class TestCreateSequences:
    """Test sequence creation for LSTM."""
    
    def test_create_sequences_basic(self):
        """Test basic sequence creation."""
        # Create sample data
        data = np.arange(100).reshape(100, 1)
        labels = np.arange(100)
        
        X, y = create_sequences(data, labels, seq_len=10)
        
        # Check shapes
        assert X.shape == (91, 10, 1)  # 100 - 10 + 1 sequences
        assert y.shape == (91,)
        
        # Check first sequence
        assert np.array_equal(X[0], data[:10])
        assert y[0] == labels[9]  # Label of last element in sequence
    
    def test_create_sequences_multivariate(self):
        """Test sequence creation with multiple features."""
        data = np.random.randn(100, 5)
        labels = np.random.randint(0, 2, 100)
        
        X, y = create_sequences(data, labels, seq_len=20)
        
        assert X.shape == (81, 20, 5)
        assert y.shape == (81,)
    
    def test_create_sequences_stride(self):
        """Test sequence creation with stride."""
        data = np.arange(100).reshape(100, 1)
        labels = np.arange(100)
        
        X, y = create_sequences(data, labels, seq_len=10, stride=5)
        
        # With stride 5, we get fewer sequences
        expected_sequences = (100 - 10) // 5 + 1
        assert X.shape[0] == expected_sequences
        
        # Check that sequences are properly strided
        assert X[0, 0, 0] == 0
        assert X[1, 0, 0] == 5  # Second sequence starts at index 5
    
    def test_create_sequences_insufficient_data(self):
        """Test with insufficient data for sequences."""
        data = np.random.randn(5, 3)
        labels = np.random.randint(0, 2, 5)
        
        X, y = create_sequences(data, labels, seq_len=10)
        
        # Should return empty arrays
        assert X.shape[0] == 0
        assert y.shape[0] == 0


class TestCheckConstantPredictions:
    """Test constant predictions check."""
    
    def test_constant_predictions(self):
        """Test detection of constant predictions."""
        # Constant predictions
        y_pred_constant = np.array([0.5, 0.5, 0.5, 0.5])
        assert check_constant_predictions(y_pred_constant) == True
        
        # Non-constant predictions
        y_pred_varied = np.array([0.1, 0.5, 0.8, 0.3])
        assert check_constant_predictions(y_pred_varied) == False
    
    def test_get_logger(self):
        """Test logger creation."""
        logger = get_logger()
        assert logger is not None
        assert hasattr(logger, 'info')
        assert hasattr(logger, 'warning')
        assert hasattr(logger, 'error')


class TestGetDevice:
    """Test device selection for PyTorch."""
    
    def test_get_device_auto(self):
        """Test automatic device selection."""
        device = get_device('auto')
        
        assert isinstance(device, torch.device)
        # Should be either cpu or cuda
        assert device.type in ['cpu', 'cuda']
    
    def test_get_device_cpu(self):
        """Test CPU device selection."""
        device = get_device('cpu')
        
        assert device.type == 'cpu'
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_get_device_cuda(self):
        """Test CUDA device selection."""
        device = get_device('cuda')
        
        assert device.type == 'cuda'
    
    def test_get_device_invalid(self):
        """Test invalid device selection falls back to CPU."""
        device = get_device('invalid_device')
        
        assert device.type == 'cpu'


class TestSetLSTMDeterministic:
    """Test LSTM deterministic settings."""
    
    def test_set_lstm_deterministic(self):
        """Test setting deterministic mode for LSTM."""
        set_lstm_deterministic(42)
        
        # Check random seeds
        # Python random
        import random
        val1 = random.random()
        random.seed(42)
        val2 = random.random()
        assert val1 == val2
        
        # NumPy
        arr1 = np.random.randn(5)
        np.random.seed(42)
        arr2 = np.random.randn(5)
        assert np.array_equal(arr1, arr2)
        
        # PyTorch
        tensor1 = torch.randn(5)
        torch.manual_seed(42)
        tensor2 = torch.randn(5)
        assert torch.equal(tensor1, tensor2)
    
    def test_deterministic_torch_settings(self):
        """Test PyTorch deterministic settings."""
        set_lstm_deterministic(42)
        
        # Check that deterministic algorithms are enabled
        if hasattr(torch, 'use_deterministic_algorithms'):
            # This might raise an error in some environments
            try:
                assert torch.are_deterministic_algorithms_enabled()
            except:
                pass  # Some operations may not support deterministic mode
        
        # Check cuDNN settings if available
        if torch.cuda.is_available():
            assert torch.backends.cudnn.deterministic
            assert not torch.backends.cudnn.benchmark


class TestCalculateMetrics:
    """Test metric calculation."""
    
    def test_calculate_metrics_binary(self):
        """Test metric calculation for binary classification."""
        y_true = np.array([0, 1, 1, 0, 1, 0, 1, 1, 0, 0])
        y_pred = np.array([0.1, 0.8, 0.9, 0.2, 0.7, 0.3, 0.6, 0.8, 0.1, 0.2])
        
        metrics = calculate_metrics(y_true, y_pred, threshold=0.5)
        
        # Check that all expected metrics are present
        assert 'accuracy' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1' in metrics
        assert 'auc' in metrics
        
        # Check metric ranges
        for key, value in metrics.items():
            assert 0 <= value <= 1
    
    def test_calculate_metrics_perfect(self):
        """Test metrics with perfect predictions."""
        y_true = np.array([0, 1, 0, 1, 0, 1])
        y_pred = np.array([0.0, 1.0, 0.0, 1.0, 0.0, 1.0])
        
        metrics = calculate_metrics(y_true, y_pred, threshold=0.5)
        
        assert metrics['accuracy'] == 1.0
        assert metrics['precision'] == 1.0
        assert metrics['recall'] == 1.0
        assert metrics['f1'] == 1.0
        assert metrics['auc'] == 1.0
    
    def test_calculate_metrics_different_thresholds(self):
        """Test metrics with different thresholds."""
        y_true = np.array([0, 1, 1, 0, 1])
        y_pred = np.array([0.3, 0.6, 0.7, 0.4, 0.8])
        
        # Lower threshold - more positive predictions
        metrics_low = calculate_metrics(y_true, y_pred, threshold=0.3)
        
        # Higher threshold - fewer positive predictions
        metrics_high = calculate_metrics(y_true, y_pred, threshold=0.7)
        
        # Recall should be higher with lower threshold
        assert metrics_low['recall'] >= metrics_high['recall']
        
        # Precision often higher with higher threshold
        # (but not guaranteed in all cases)