"""Unit tests for LSTM base model."""

import pytest
import numpy as np
import torch
import torch.nn as nn
from unittest.mock import Mock, patch, MagicMock
import tempfile
from pathlib import Path

from src.models.lstm.base import BaseLSTM, LSTMNetwork


class TestLSTMNetwork:
    """Test cases for LSTMNetwork class."""
    
    @pytest.fixture
    def network_params(self):
        """Network parameters fixture."""
        return {
            'input_size': 10,
            'hidden_size': 32,
            'num_layers': 2,
            'dropout': 0.2,
            'bidirectional': False,
            'output_size': 1
        }
    
    @pytest.fixture
    def lstm_network(self, network_params):
        """LSTM network fixture."""
        return LSTMNetwork(**network_params)
    
    def test_network_initialization(self, lstm_network, network_params):
        """Test network initialization."""
        assert lstm_network.hidden_size == network_params['hidden_size']
        assert lstm_network.num_layers == network_params['num_layers']
        assert isinstance(lstm_network.lstm, nn.LSTM)
        assert isinstance(lstm_network.fc1, nn.Linear)
        assert isinstance(lstm_network.fc2, nn.Linear)
    
    def test_forward_pass(self, lstm_network):
        """Test forward pass through network."""
        batch_size = 16
        seq_length = 20
        input_size = 10
        
        # Create dummy input
        x = torch.randn(batch_size, seq_length, input_size)
        
        # Forward pass
        output, hidden = lstm_network(x)
        
        # Check output shapes
        assert output.shape == (batch_size, 1)
        assert hidden[0].shape == (2, batch_size, 32)  # hidden state
        assert hidden[1].shape == (2, batch_size, 32)  # cell state
    
    def test_forward_pass_single_sample(self, lstm_network):
        """Test forward pass with single sample."""
        x = torch.randn(1, 20, 10)
        output, hidden = lstm_network(x)
        
        assert output.shape == (1, 1)
        assert not torch.isnan(output).any()
    
    def test_bidirectional_network(self):
        """Test bidirectional LSTM network."""
        network = LSTMNetwork(
            input_size=10,
            hidden_size=32,
            bidirectional=True,
            output_size=1
        )
        
        x = torch.randn(8, 20, 10)
        output, hidden = network(x)
        
        assert output.shape == (8, 1)
        # Bidirectional doubles the hidden state size
        assert hidden[0].shape == (4, 8, 32)  # 2 layers * 2 directions
    
    def test_init_hidden(self, lstm_network):
        """Test hidden state initialization."""
        batch_size = 16
        device = torch.device('cpu')
        
        h0, c0 = lstm_network.init_hidden(batch_size, device)
        
        assert h0.shape == (2, batch_size, 32)
        assert c0.shape == (2, batch_size, 32)
        assert torch.all(h0 == 0)
        assert torch.all(c0 == 0)
    
    @pytest.mark.parametrize("num_layers", [1, 2, 3])
    def test_different_layers(self, num_layers):
        """Test network with different number of layers."""
        network = LSTMNetwork(
            input_size=10,
            hidden_size=32,
            num_layers=num_layers,
            dropout=0.2 if num_layers > 1 else 0
        )
        
        x = torch.randn(8, 20, 10)
        output, hidden = network(x)
        
        assert output.shape == (8, 1)
        assert hidden[0].shape == (num_layers, 8, 32)


class TestBaseLSTM:
    """Test cases for BaseLSTM class."""
    
    @pytest.fixture
    def base_lstm(self):
        """Base LSTM fixture."""
        return BaseLSTM(
            input_size=10,
            hidden_size=32,
            num_layers=2,
            dropout=0.2,
            learning_rate=0.001,
            device='cpu'
        )
    
    @pytest.fixture
    def sample_data(self):
        """Sample data fixture."""
        n_samples = 100
        seq_length = 20
        n_features = 10
        
        X = np.random.randn(n_samples, seq_length, n_features)
        y = np.random.randint(0, 2, n_samples)
        
        return X, y
    
    def test_initialization(self, base_lstm):
        """Test BaseLSTM initialization."""
        assert base_lstm.input_size == 10
        assert base_lstm.hidden_size == 32
        assert base_lstm.num_layers == 2
        assert isinstance(base_lstm.model, LSTMNetwork)
        assert isinstance(base_lstm.optimizer, torch.optim.Adam)
        assert isinstance(base_lstm.criterion, nn.BCEWithLogitsLoss)
    
    def test_device_selection(self):
        """Test device selection."""
        # CPU device
        lstm_cpu = BaseLSTM(input_size=10, device='cpu')
        assert lstm_cpu.device == torch.device('cpu')
        
        # CUDA device (if available)
        if torch.cuda.is_available():
            lstm_cuda = BaseLSTM(input_size=10, device='cuda')
            assert lstm_cuda.device == torch.device('cuda')
    
    def test_train_epoch(self, base_lstm, sample_data):
        """Test training for one epoch."""
        X, y = sample_data
        
        # Create simple dataloader
        dataset = torch.utils.data.TensorDataset(
            torch.FloatTensor(X),
            torch.FloatTensor(y)
        )
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=16,
            shuffle=True
        )
        
        # Train one epoch
        loss = base_lstm.train_epoch(dataloader)
        
        assert isinstance(loss, float)
        assert loss > 0
        assert not np.isnan(loss)
    
    def test_evaluate(self, base_lstm, sample_data):
        """Test model evaluation."""
        X, y = sample_data
        
        # Create dataloader
        dataset = torch.utils.data.TensorDataset(
            torch.FloatTensor(X),
            torch.FloatTensor(y)
        )
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=16,
            shuffle=False
        )
        
        # Evaluate
        loss, preds, targets = base_lstm.evaluate(dataloader)
        
        assert isinstance(loss, float)
        assert loss > 0
        assert len(preds) == len(y)
        assert len(targets) == len(y)
        assert np.all((preds >= 0) & (preds <= 1))
    
    def test_predict_proba(self, base_lstm, sample_data):
        """Test probability prediction."""
        X, _ = sample_data
        
        # Predict probabilities
        proba = base_lstm.predict_proba(X)
        
        assert proba.shape == (len(X), 2)
        assert np.all((proba >= 0) & (proba <= 1))
        assert np.allclose(proba.sum(axis=1), 1.0)
    
    def test_predict(self, base_lstm, sample_data):
        """Test class prediction."""
        X, _ = sample_data
        
        # Predict classes
        preds = base_lstm.predict(X, threshold=0.5)
        
        assert preds.shape == (len(X),)
        assert np.all((preds == 0) | (preds == 1))
    
    @pytest.mark.parametrize("threshold", [0.3, 0.5, 0.7])
    def test_predict_with_different_thresholds(self, base_lstm, sample_data, threshold):
        """Test prediction with different thresholds."""
        X, _ = sample_data
        
        preds = base_lstm.predict(X, threshold=threshold)
        
        assert preds.shape == (len(X),)
        assert np.all((preds == 0) | (preds == 1))
    
    def test_save_load_model(self, base_lstm, sample_data):
        """Test model saving and loading."""
        X, _ = sample_data
        
        # Get predictions before saving
        preds_before = base_lstm.predict_proba(X)
        
        # Save model
        with tempfile.NamedTemporaryFile(suffix='.pt') as tmp:
            base_lstm.save_model(tmp.name)
            
            # Create new model and load
            new_lstm = BaseLSTM(
                input_size=10,
                hidden_size=32,
                num_layers=2
            )
            new_lstm.load_model(tmp.name)
            
            # Get predictions after loading
            preds_after = new_lstm.predict_proba(X)
            
            # Check predictions are the same
            np.testing.assert_array_almost_equal(preds_before, preds_after)
    
    def test_gradient_clipping(self, base_lstm):
        """Test that gradient clipping is applied."""
        # Create data with extreme values to cause large gradients
        X = torch.randn(10, 20, 10) * 100
        y = torch.ones(10)
        
        dataset = torch.utils.data.TensorDataset(X, y)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=10)
        
        # Train and check gradients are clipped
        loss = base_lstm.train_epoch(dataloader)
        
        # Check that no gradients exploded
        for param in base_lstm.model.parameters():
            if param.grad is not None:
                assert torch.all(torch.abs(param.grad) <= 1.1)  # Small margin for numerical errors
    
    def test_empty_dataloader(self, base_lstm):
        """Test handling of empty dataloader."""
        empty_dataloader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(
                torch.FloatTensor([]),
                torch.FloatTensor([])
            ),
            batch_size=1
        )
        
        loss = base_lstm.train_epoch(empty_dataloader)
        assert loss == 0
    
    def test_single_sample_prediction(self, base_lstm):
        """Test prediction with single sample."""
        X = np.random.randn(1, 20, 10)
        
        proba = base_lstm.predict_proba(X)
        pred = base_lstm.predict(X)
        
        assert proba.shape == (1, 2)
        assert pred.shape == (1,)


class TestLSTMIntegration:
    """Integration tests for LSTM components."""
    
    def test_end_to_end_training(self):
        """Test end-to-end training workflow."""
        # Create model
        model = BaseLSTM(
            input_size=5,
            hidden_size=16,
            num_layers=1,
            learning_rate=0.01
        )
        
        # Generate synthetic data
        n_samples = 50
        X = np.random.randn(n_samples, 10, 5)
        y = (np.random.randn(n_samples) > 0).astype(int)
        
        # Split data
        n_train = 40
        X_train, X_val = X[:n_train], X[n_train:]
        y_train, y_val = y[:n_train], y[n_train:]
        
        # Create dataloaders
        train_dataset = torch.utils.data.TensorDataset(
            torch.FloatTensor(X_train),
            torch.FloatTensor(y_train)
        )
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=8,
            shuffle=True
        )
        
        val_dataset = torch.utils.data.TensorDataset(
            torch.FloatTensor(X_val),
            torch.FloatTensor(y_val)
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=8,
            shuffle=False
        )
        
        # Train for a few epochs
        for epoch in range(3):
            train_loss = model.train_epoch(train_loader)
            val_loss, _, _ = model.evaluate(val_loader)
            
            assert train_loss > 0
            assert val_loss > 0
        
        # Make predictions
        predictions = model.predict(X_val)
        assert len(predictions) == len(X_val)