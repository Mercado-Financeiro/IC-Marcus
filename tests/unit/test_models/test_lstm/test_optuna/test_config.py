"""Unit tests for LSTM Optuna configuration module."""

import pytest
from src.models.lstm.optuna.config import LSTMOptunaConfig, TrainingMetrics


class TestLSTMOptunaConfig:
    """Test LSTM Optuna configuration."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = LSTMOptunaConfig()
        
        assert config.n_trials == 50
        assert config.cv_folds == 3
        assert config.embargo == 10
        assert config.pruner_type == 'median'
        assert config.early_stopping_patience == 10
        assert config.max_epochs == 100
        assert config.batch_size == 32
        assert config.seed == 42
        assert config.device == 'auto'
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = LSTMOptunaConfig(
            n_trials=100,
            cv_folds=5,
            max_epochs=200,
            device='cuda'
        )
        
        assert config.n_trials == 100
        assert config.cv_folds == 5
        assert config.max_epochs == 200
        assert config.device == 'cuda'
    
    def test_config_to_dict(self):
        """Test configuration to dictionary conversion."""
        config = LSTMOptunaConfig(n_trials=10)
        config_dict = config.to_dict()
        
        assert isinstance(config_dict, dict)
        assert config_dict['n_trials'] == 10
        assert 'cv_folds' in config_dict
        assert 'seed' in config_dict
    
    def test_hyperparameter_ranges(self):
        """Test hyperparameter range settings."""
        config = LSTMOptunaConfig()
        
        # Sequence length
        assert config.seq_len_min < config.seq_len_max
        assert config.seq_len_min > 0
        
        # Hidden size
        assert config.hidden_size_min < config.hidden_size_max
        assert config.hidden_size_min > 0
        
        # Layers
        assert config.num_layers_min < config.num_layers_max
        assert config.num_layers_min > 0
        
        # Dropout
        assert 0 <= config.dropout_min < config.dropout_max <= 1
        
        # Learning rate
        assert config.learning_rate_min < config.learning_rate_max
        assert config.learning_rate_min > 0


class TestTrainingMetrics:
    """Test training metrics container."""
    
    def test_metrics_creation(self):
        """Test creating training metrics."""
        metrics = TrainingMetrics(
            train_loss=0.5,
            val_loss=0.6,
            train_accuracy=0.8,
            val_accuracy=0.75,
            train_f1=0.7,
            val_f1=0.65,
            train_auc=0.85,
            val_auc=0.8,
            epoch=10
        )
        
        assert metrics.train_loss == 0.5
        assert metrics.val_loss == 0.6
        assert metrics.train_accuracy == 0.8
        assert metrics.val_accuracy == 0.75
        assert metrics.epoch == 10
    
    def test_metrics_to_dict(self):
        """Test metrics to dictionary conversion."""
        metrics = TrainingMetrics(
            train_loss=0.5,
            val_loss=0.6,
            train_accuracy=0.8,
            val_accuracy=0.75,
            train_f1=0.7,
            val_f1=0.65,
            train_auc=0.85,
            val_auc=0.8,
            epoch=10
        )
        
        metrics_dict = metrics.to_dict()
        
        assert isinstance(metrics_dict, dict)
        assert metrics_dict['train_loss'] == 0.5
        assert metrics_dict['val_loss'] == 0.6
        assert metrics_dict['epoch'] == 10
        assert len(metrics_dict) == 9