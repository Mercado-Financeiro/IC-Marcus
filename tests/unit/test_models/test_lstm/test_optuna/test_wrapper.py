"""Unit tests for LSTM scikit-learn wrapper."""

import pytest
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.exceptions import NotFittedError
from unittest.mock import Mock, MagicMock
from src.models.lstm.optuna.wrapper import LSTMWrapper


class TestLSTMWrapper:
    """Test LSTM scikit-learn compatible wrapper."""
    
    @pytest.fixture
    def mock_lstm_model(self):
        """Create a mock LSTM model."""
        model = Mock(spec=nn.Module)
        model.eval = Mock()
        # Return probabilities between 0 and 1
        model.return_value = torch.FloatTensor([[0.7], [0.3], [0.8], [0.2]])
        return model
    
    @pytest.fixture
    def wrapper(self, mock_lstm_model, torch_device):
        """Create wrapper instance."""
        return LSTMWrapper(
            lstm_model=mock_lstm_model,
            seq_len=5,
            device=torch_device,
            scaler=None
        )
    
    def test_wrapper_creation(self, wrapper):
        """Test wrapper initialization."""
        assert wrapper.seq_len == 5
        assert wrapper.device.type == 'cpu'
        assert wrapper.scaler is None
        assert wrapper._estimator_type == "classifier"
        assert len(wrapper.classes_) == 2
        assert wrapper.n_classes_ == 2
    
    def test_fit_method(self, wrapper, sample_features_data):
        """Test fit method (already trained model)."""
        X = sample_features_data.drop('label', axis=1)
        y = sample_features_data['label']
        
        result = wrapper.fit(X, y)
        
        assert result is wrapper
        assert wrapper.n_features_in_ == X.shape[1]
    
    def test_predict_proba(self, wrapper, sample_features_data):
        """Test probability prediction."""
        wrapper.n_features_in_ = 5
        X = sample_features_data.drop('label', axis=1).iloc[:20]
        
        proba = wrapper.predict_proba(X)
        
        # Check shape
        assert proba.shape[1] == 2  # Binary classification
        
        # Check probabilities sum to 1
        assert np.allclose(proba.sum(axis=1), 1.0)
        
        # Check range
        assert np.all(proba >= 0)
        assert np.all(proba <= 1)
    
    def test_predict(self, wrapper, sample_features_data):
        """Test class prediction."""
        wrapper.n_features_in_ = 5
        X = sample_features_data.drop('label', axis=1).iloc[:20]
        
        predictions = wrapper.predict(X)
        
        # Check predictions are binary
        assert set(np.unique(predictions)).issubset({0, 1})
    
    def test_decision_function(self, wrapper, sample_features_data):
        """Test decision function (log odds)."""
        wrapper.n_features_in_ = 5
        X = sample_features_data.drop('label', axis=1).iloc[:20]
        
        decision = wrapper.decision_function(X)
        
        # Check shape
        assert len(decision) > 0
        
        # Decision values can be any real number
        assert decision.dtype == np.float64
    
    def test_not_fitted_error(self):
        """Test that NotFittedError is raised when model is None."""
        wrapper = LSTMWrapper(
            lstm_model=None,
            seq_len=5,
            device=torch.device('cpu'),
            scaler=None
        )
        
        X = np.random.randn(10, 5)
        
        with pytest.raises(NotFittedError):
            wrapper.predict_proba(X)
    
    def test_create_sequences(self, wrapper):
        """Test sequence creation from input data."""
        # Test with DataFrame
        df = pd.DataFrame(np.random.randn(20, 5))
        sequences = wrapper._create_sequences(df)
        
        assert sequences.shape == (16, 5, 5)  # (samples, seq_len, features)
        
        # Test with numpy array
        arr = np.random.randn(20, 5)
        sequences = wrapper._create_sequences(arr)
        
        assert sequences.shape == (16, 5, 5)
    
    def test_create_sequences_with_padding(self, wrapper):
        """Test sequence creation with insufficient data."""
        # Less data than seq_len
        arr = np.random.randn(3, 5)  # Only 3 samples, need 5
        sequences = wrapper._create_sequences(arr)
        
        # Should pad and create 1 sequence
        assert sequences.shape[0] >= 1
        assert sequences.shape[1] == 5  # seq_len
        assert sequences.shape[2] == 5  # features
    
    def test_with_scaler(self, mock_lstm_model, torch_device):
        """Test wrapper with feature scaler."""
        from sklearn.preprocessing import StandardScaler
        
        scaler = Mock(spec=StandardScaler)
        scaler.transform = Mock(side_effect=lambda x: x * 2)
        
        wrapper = LSTMWrapper(
            lstm_model=mock_lstm_model,
            seq_len=5,
            device=torch_device,
            scaler=scaler
        )
        
        wrapper.n_features_in_ = 5
        X = np.random.randn(20, 5)
        
        _ = wrapper.predict_proba(X)
        
        # Check scaler was called
        assert scaler.transform.called
    
    def test_score_method(self, wrapper, sample_features_data):
        """Test score method (accuracy)."""
        wrapper.n_features_in_ = 5
        X = sample_features_data.drop('label', axis=1).iloc[:20]
        y = np.random.randint(0, 2, len(X))
        
        score = wrapper.score(X, y)
        
        assert 0 <= score <= 1
        assert isinstance(score, float)
    
    def test_sklearn_compatibility(self, wrapper):
        """Test scikit-learn API compatibility."""
        # Check required attributes
        assert hasattr(wrapper, 'fit')
        assert hasattr(wrapper, 'predict')
        assert hasattr(wrapper, 'predict_proba')
        assert hasattr(wrapper, 'score')
        assert hasattr(wrapper, 'classes_')
        assert hasattr(wrapper, '_estimator_type')
        
        # Check estimator type
        assert wrapper._estimator_type == 'classifier'
    
    def test_with_different_seq_lengths(self, torch_device):
        """Test wrapper with different sequence lengths."""
        for seq_len in [3, 10, 20]:
            # Create a new mock for each seq_len
            mock_model = Mock(spec=nn.Module)
            mock_model.eval = Mock()
            expected_outputs = 30 - seq_len + 1
            # Return the correct number of probabilities
            mock_model.return_value = torch.FloatTensor([[0.7]] * expected_outputs)
            
            wrapper = LSTMWrapper(
                lstm_model=mock_model,
                seq_len=seq_len,
                device=torch_device,
                scaler=None
            )
            
            wrapper.n_features_in_ = 5
            X = np.random.randn(30, 5)
            
            proba = wrapper.predict_proba(X)
            assert proba.shape[0] == expected_outputs