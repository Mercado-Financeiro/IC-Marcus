"""Unit tests for LSTM training utilities."""

import pytest
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from unittest.mock import Mock, MagicMock, patch
from src.models.lstm.optuna.training import (
    train_epoch,
    validate,
    train_model
)


class TestTrainEpoch:
    """Test training epoch function."""
    
    @pytest.fixture
    def mock_model(self):
        """Create mock model."""
        model = Mock(spec=nn.Module)
        model.train = Mock()
        model.parameters = Mock(return_value=[torch.randn(10, 10)])
        # Return predictions
        model.return_value = torch.sigmoid(torch.randn(32, 1))
        return model
    
    @pytest.fixture
    def mock_loader(self):
        """Create mock data loader."""
        # Create batches of data
        batches = []
        for _ in range(5):
            X = torch.randn(32, 10, 5)  # batch, seq, features
            y = torch.randint(0, 2, (32,)).float()
            batches.append((X, y))
        return batches
    
    @pytest.fixture
    def mock_optimizer(self):
        """Create mock optimizer."""
        optimizer = Mock()
        optimizer.zero_grad = Mock()
        optimizer.step = Mock()
        return optimizer
    
    def test_train_epoch(self, mock_model, mock_loader, mock_optimizer, torch_device):
        """Test single training epoch."""
        criterion = nn.BCELoss()
        
        avg_loss, accuracy = train_epoch(
            mock_model,
            mock_loader,
            criterion,
            mock_optimizer,
            torch_device
        )
        
        assert isinstance(avg_loss, float)
        assert isinstance(accuracy, float)
        assert 0 <= accuracy <= 1
        assert avg_loss >= 0
        
        # Check that optimizer was used
        assert mock_optimizer.zero_grad.called
        assert mock_optimizer.step.called
    
    def test_train_epoch_with_gradient_clipping(
        self, mock_model, mock_loader, mock_optimizer, torch_device
    ):
        """Test training with gradient clipping."""
        criterion = nn.BCELoss()
        
        with patch('torch.nn.utils.clip_grad_norm_') as mock_clip:
            avg_loss, accuracy = train_epoch(
                mock_model,
                mock_loader,
                criterion,
                mock_optimizer,
                torch_device,
                gradient_clipping=1.0
            )
            
            # Check gradient clipping was applied
            assert mock_clip.called


class TestValidate:
    """Test validation function."""
    
    @pytest.fixture
    def mock_model(self):
        """Create mock model for validation."""
        model = Mock(spec=nn.Module)
        model.eval = Mock()
        # Return predictions
        model.return_value = torch.sigmoid(torch.randn(32, 1))
        return model
    
    @pytest.fixture
    def mock_loader(self):
        """Create mock validation data loader."""
        batches = []
        for _ in range(3):
            X = torch.randn(32, 10, 5)
            y = torch.randint(0, 2, (32,)).float()
            batches.append((X, y))
        return batches
    
    def test_validate(self, mock_model, mock_loader, torch_device):
        """Test model validation."""
        criterion = nn.BCELoss()
        
        with patch('torch.no_grad'):
            avg_loss, accuracy, all_preds, all_labels = validate(
                mock_model,
                mock_loader,
                criterion,
                torch_device
            )
        
        assert isinstance(avg_loss, float)
        assert isinstance(accuracy, float)
        assert 0 <= accuracy <= 1
        assert avg_loss >= 0
        
        assert isinstance(all_preds, np.ndarray)
        assert isinstance(all_labels, np.ndarray)
        assert len(all_preds) == len(all_labels)
        
        # Check model was in eval mode
        assert mock_model.eval.called


class TestTrainModel:
    """Test full model training."""
    
    @pytest.fixture
    def mock_model(self):
        """Create mock model."""
        model = Mock(spec=nn.Module)
        model.train = Mock()
        model.eval = Mock()
        model.parameters = Mock(return_value=[torch.randn(10, 10)])
        model.return_value = torch.sigmoid(torch.randn(32, 1))
        return model
    
    def test_train_model(self, mock_model):
        """Test model training function."""
        X_train = torch.randn(100, 10, 5)
        y_train = torch.randint(0, 2, (100,)).float()
        X_val = torch.randn(20, 10, 5)
        y_val = torch.randint(0, 2, (20,)).float()
        
        device = torch.device('cpu')
        
        with patch('src.models.lstm.optuna.training.train_epoch', return_value=(0.5, 0.7)):
            with patch('src.models.lstm.optuna.training.validate', 
                      return_value=(0.4, 0.75, np.random.rand(20), np.random.randint(0, 2, 20))):
                trained_model = train_model(
                    mock_model,
                    X_train, y_train,
                    X_val, y_val,
                    device,
                    epochs=2,
                    batch_size=32,
                    learning_rate=0.001
                )
        
        assert trained_model is not None


class TestEarlyStopping:
    """Test early stopping mechanism."""
    
    def test_early_stopping_creation(self):
        """Test early stopping initialization."""
        early_stop = EarlyStopping(patience=5, verbose=True)
        
        assert early_stop.patience == 5
        assert early_stop.verbose
        assert early_stop.best_score is None
        assert early_stop.counter == 0
        assert not early_stop.early_stop
    
    def test_early_stopping_improvement(self):
        """Test early stopping with improving scores."""
        early_stop = EarlyStopping(patience=3)
        model = Mock()
        
        # First call - should save
        early_stop(0.5, model)
        assert early_stop.best_score == -0.5
        assert early_stop.counter == 0
        
        # Better score - should reset counter
        early_stop(0.4, model)
        assert early_stop.best_score == -0.4
        assert early_stop.counter == 0
        assert not early_stop.early_stop
    
    def test_early_stopping_no_improvement(self):
        """Test early stopping with no improvement."""
        early_stop = EarlyStopping(patience=2)
        model = Mock()
        
        # Initial score
        early_stop(0.5, model)
        assert early_stop.counter == 0
        
        # Worse score
        early_stop(0.6, model)
        assert early_stop.counter == 1
        assert not early_stop.early_stop
        
        # Another worse score - should trigger
        early_stop(0.7, model)
        assert early_stop.counter == 2
        assert early_stop.early_stop
    
    def test_early_stopping_with_delta(self):
        """Test early stopping with minimum delta."""
        early_stop = EarlyStopping(patience=2, delta=0.01)
        model = Mock()
        
        early_stop(0.5, model)
        
        # Small improvement less than delta
        early_stop(0.495, model)
        assert early_stop.counter == 1  # Not enough improvement
        
        # Significant improvement
        early_stop(0.48, model)
        assert early_stop.counter == 0  # Reset counter
    
    def test_save_checkpoint(self, tmp_path):
        """Test checkpoint saving."""
        early_stop = EarlyStopping(patience=3, path=str(tmp_path / 'checkpoint.pt'))
        
        model = Mock(spec=nn.Module)
        model.state_dict = Mock(return_value={'weight': torch.randn(10, 10)})
        
        early_stop(0.5, model)
        
        # Check file was created
        assert (tmp_path / 'checkpoint.pt').exists()


