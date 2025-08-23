"""Unit tests for LSTM model architectures."""

import pytest
import torch
import torch.nn as nn
import numpy as np
from src.models.lstm.optuna.model import LSTMModel, AttentionLSTM


class TestLSTMModel:
    """Test basic LSTM model."""
    
    def test_model_creation(self):
        """Test creating LSTM model."""
        model = LSTMModel(
            input_size=10,
            hidden_size=32,
            num_layers=2,
            dropout=0.2,
            output_size=1
        )
        
        assert isinstance(model, nn.Module)
        assert model.hidden_size == 32
        assert model.num_layers == 2
        assert not model.bidirectional
    
    def test_forward_pass(self, sample_sequences):
        """Test forward pass through model."""
        X, _ = sample_sequences
        batch_size, seq_len, n_features = X.shape
        
        model = LSTMModel(
            input_size=n_features,
            hidden_size=16,
            num_layers=1,
            dropout=0.1
        )
        
        # Convert to tensor
        X_tensor = torch.FloatTensor(X)
        
        # Forward pass
        output = model(X_tensor)
        
        # Check output shape
        assert output.shape == (batch_size, 1)
        
        # Check output range (sigmoid activation)
        assert torch.all(output >= 0)
        assert torch.all(output <= 1)
    
    def test_bidirectional_model(self):
        """Test bidirectional LSTM model."""
        model = LSTMModel(
            input_size=10,
            hidden_size=32,
            num_layers=2,
            dropout=0.2,
            bidirectional=True
        )
        
        assert model.bidirectional
        
        # Test forward pass
        X = torch.randn(16, 20, 10)  # batch, seq, features
        output = model(X)
        assert output.shape == (16, 1)
    
    def test_init_hidden(self):
        """Test hidden state initialization."""
        model = LSTMModel(
            input_size=10,
            hidden_size=32,
            num_layers=2,
            dropout=0.2
        )
        
        batch_size = 16
        device = torch.device('cpu')
        
        h0, c0 = model.init_hidden(batch_size, device)
        
        assert h0.shape == (2, batch_size, 32)  # num_layers, batch, hidden
        assert c0.shape == (2, batch_size, 32)
        assert torch.all(h0 == 0)
        assert torch.all(c0 == 0)
    
    def test_different_layer_configs(self):
        """Test models with different layer configurations."""
        configs = [
            (1, 0.0),   # Single layer, no dropout
            (2, 0.3),   # Two layers with dropout
            (3, 0.5),   # Three layers with high dropout
        ]
        
        for num_layers, dropout in configs:
            model = LSTMModel(
                input_size=10,
                hidden_size=20,
                num_layers=num_layers,
                dropout=dropout
            )
            
            X = torch.randn(8, 15, 10)
            output = model(X)
            assert output.shape == (8, 1)


class TestAttentionLSTM:
    """Test LSTM with attention mechanism."""
    
    def test_attention_model_creation(self):
        """Test creating attention LSTM model."""
        model = AttentionLSTM(
            input_size=10,
            hidden_size=32,
            num_layers=2,
            dropout=0.2,
            output_size=1
        )
        
        assert isinstance(model, nn.Module)
        assert hasattr(model, 'attention')
        assert model.hidden_size == 32
        assert model.num_layers == 2
    
    def test_attention_forward_pass(self, sample_sequences):
        """Test forward pass with attention."""
        X, _ = sample_sequences
        batch_size, seq_len, n_features = X.shape
        
        model = AttentionLSTM(
            input_size=n_features,
            hidden_size=16,
            num_layers=1,
            dropout=0.1
        )
        
        # Convert to tensor
        X_tensor = torch.FloatTensor(X)
        
        # Forward pass
        output = model(X_tensor)
        
        # Check output shape
        assert output.shape == (batch_size, 1)
        
        # Check output range
        assert torch.all(output >= 0)
        assert torch.all(output <= 1)
    
    def test_attention_weights(self):
        """Test that attention weights sum to 1."""
        model = AttentionLSTM(
            input_size=10,
            hidden_size=20,
            num_layers=1,
            dropout=0.1
        )
        
        X = torch.randn(8, 15, 10)
        
        # Get LSTM output
        lstm_out, _ = model.lstm(X)
        
        # Calculate attention weights
        attention_weights = torch.softmax(
            model.attention(lstm_out).squeeze(-1), 
            dim=1
        )
        
        # Check that weights sum to 1 for each sample
        weight_sums = attention_weights.sum(dim=1)
        assert torch.allclose(weight_sums, torch.ones(8), atol=1e-6)
    
    def test_gradient_flow(self):
        """Test that gradients flow through the model."""
        model = AttentionLSTM(
            input_size=5,
            hidden_size=10,
            num_layers=1,
            dropout=0.0
        )
        
        X = torch.randn(4, 10, 5, requires_grad=True)
        y = torch.randint(0, 2, (4,)).float()
        
        # Forward pass
        output = model(X).squeeze()
        
        # Calculate loss
        loss = nn.BCELoss()(output, y)
        
        # Backward pass
        loss.backward()
        
        # Check that input has gradients
        assert X.grad is not None
        assert not torch.all(X.grad == 0)