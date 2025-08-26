"""Unit tests for LSTM Optuna optimizer."""

import pytest
import numpy as np
import pandas as pd
import torch
import optuna
from unittest.mock import Mock, MagicMock, patch
from sklearn.model_selection import TimeSeriesSplit
from src.models.lstm.optuna.optimizer import LSTMOptuna
from src.models.lstm.optuna.config import LSTMOptunaConfig


class TestLSTMOptuna:
    """Test LSTM Optuna optimizer."""
    
    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return LSTMOptunaConfig(
            n_trials=2,
            cv_folds=2,
            max_epochs=3,
            batch_size=16,
            device='cpu',
            early_stopping_patience=2
        )
    
    @pytest.fixture
    def lstm_optuna(self, config):
        """Create LSTMOptuna instance."""
        return LSTMOptuna(config)
    
    def test_initialization(self, lstm_optuna, config):
        """Test LSTMOptuna initialization."""
        assert lstm_optuna.config == config
        assert lstm_optuna.best_model is None
        assert lstm_optuna.best_params is None
        assert lstm_optuna.best_score == -np.inf
        assert lstm_optuna.study is None
        assert lstm_optuna.scaler is None
    
    def test_objective_function(self, lstm_optuna, sample_features_data):
        """Test Optuna objective function."""
        X = sample_features_data.drop('label', axis=1).values
        y = sample_features_data['label'].values
        
        # Mock trial
        trial = Mock(spec=optuna.Trial)
        trial.suggest_int = Mock(side_effect=[10, 32, 2])  # seq_len, hidden_size, num_layers
        trial.suggest_float = Mock(side_effect=[0.2, 0.001])  # dropout, learning_rate
        trial.suggest_categorical = Mock(side_effect=['adam', True])  # optimizer, bidirectional
        trial.report = Mock()
        trial.should_prune = Mock(return_value=False)
        
        with patch.object(lstm_optuna, '_train_model', return_value=(Mock(), 0.8)):
            score = lstm_optuna._objective(trial, X, y)
        
        assert isinstance(score, float)
        assert trial.suggest_int.called
        assert trial.suggest_float.called
        assert trial.suggest_categorical.called
    
    def test_optimize(self, lstm_optuna, sample_features_data):
        """Test optimization process."""
        X = sample_features_data.drop('label', axis=1).values
        y = (sample_features_data['label'] > 0).astype(int).values
        
        # Mock the training process
        mock_model = Mock()
        with patch.object(lstm_optuna, '_train_model', return_value=(mock_model, 0.75)):
            with patch.object(lstm_optuna, '_train_final_model', return_value=mock_model):
                lstm_optuna.optimize(X, y)
        
        assert lstm_optuna.study is not None
        assert lstm_optuna.best_params is not None
        assert lstm_optuna.best_model is not None
        assert lstm_optuna.best_score > -np.inf
    
    def test_fit(self, lstm_optuna, sample_features_data):
        """Test fit method."""
        X = sample_features_data.drop('label', axis=1)
        y = (sample_features_data['label'] > 0).astype(int)
        
        mock_model = Mock()
        with patch.object(lstm_optuna, 'optimize'):
            with patch.object(lstm_optuna, '_create_wrapper', return_value=mock_model):
                result = lstm_optuna.fit(X, y)
        
        assert result is lstm_optuna
        assert lstm_optuna.wrapper is not None
        assert lstm_optuna.feature_names_ is not None
    
    def test_predict(self, lstm_optuna, sample_features_data):
        """Test predict method."""
        X = sample_features_data.drop('label', axis=1)
        
        # Create mock wrapper
        mock_wrapper = Mock()
        mock_wrapper.predict = Mock(return_value=np.array([0, 1, 0, 1]))
        lstm_optuna.wrapper = mock_wrapper
        lstm_optuna.feature_names_ = X.columns.tolist()
        
        predictions = lstm_optuna.predict(X.iloc[:4])
        
        assert len(predictions) == 4
        mock_wrapper.predict.called_once()
    
    def test_predict_proba(self, lstm_optuna, sample_features_data):
        """Test predict_proba method."""
        X = sample_features_data.drop('label', axis=1)
        
        # Create mock wrapper
        mock_wrapper = Mock()
        mock_wrapper.predict_proba = Mock(
            return_value=np.array([[0.3, 0.7], [0.8, 0.2]])
        )
        lstm_optuna.wrapper = mock_wrapper
        lstm_optuna.feature_names_ = X.columns.tolist()
        
        proba = lstm_optuna.predict_proba(X.iloc[:2])
        
        assert proba.shape == (2, 2)
        mock_wrapper.predict_proba.called_once()
    
    def test_create_wrapper(self, lstm_optuna):
        """Test wrapper creation."""
        mock_model = Mock()
        lstm_optuna.best_model = mock_model
        lstm_optuna.best_params = {'seq_len': 10}
        lstm_optuna.scaler = Mock()
        lstm_optuna.config.device = 'cpu'
        
        wrapper = lstm_optuna._create_wrapper()
        
        assert wrapper is not None
        assert wrapper.lstm_model == mock_model
        assert wrapper.seq_len == 10
        assert wrapper.scaler == lstm_optuna.scaler
    
    def test_train_model(self, lstm_optuna, sample_sequences):
        """Test model training."""
        X, y = sample_sequences
        params = {
            'seq_len': 5,
            'hidden_size': 16,
            'num_layers': 1,
            'dropout': 0.1,
            'learning_rate': 0.001,
            'optimizer_type': 'adam',
            'bidirectional': False
        }
        
        with patch('src.models.lstm.optuna.training.train_epoch', return_value=(0.5, 0.7)):
            with patch('src.models.lstm.optuna.training.evaluate', 
                      return_value=(0.4, 0.75, np.random.rand(10), np.random.randint(0, 2, 10))):
                model, score = lstm_optuna._train_model(
                    X[:20], y[:20], X[20:], y[20:], params, max_epochs=2
                )
        
        assert model is not None
        assert isinstance(score, float)
    
    def test_train_final_model(self, lstm_optuna, sample_sequences):
        """Test final model training on all data."""
        X, y = sample_sequences
        params = {
            'seq_len': 5,
            'hidden_size': 16,
            'num_layers': 1,
            'dropout': 0.1,
            'learning_rate': 0.001,
            'optimizer_type': 'adam',
            'bidirectional': False
        }
        
        with patch('src.models.lstm.optuna.training.train_epoch', return_value=(0.5, 0.7)):
            model = lstm_optuna._train_final_model(X, y, params, max_epochs=2)
        
        assert model is not None
    
    def test_get_params(self, lstm_optuna):
        """Test get_params for sklearn compatibility."""
        params = lstm_optuna.get_params()
        
        assert isinstance(params, dict)
        assert 'config' in params
    
    def test_set_params(self, lstm_optuna):
        """Test set_params for sklearn compatibility."""
        new_config = LSTMOptunaConfig(n_trials=10)
        lstm_optuna.set_params(config=new_config)
        
        assert lstm_optuna.config.n_trials == 10
    
    def test_cross_validation(self, lstm_optuna, sample_features_data):
        """Test cross-validation during optimization."""
        X = sample_features_data.drop('label', axis=1).values
        y = (sample_features_data['label'] > 0).astype(int).values
        
        # Test that CV is used properly
        with patch('sklearn.model_selection.TimeSeriesSplit') as mock_tscv:
            mock_tscv.return_value.split = Mock(
                return_value=[
                    (np.arange(50), np.arange(50, 70)),
                    (np.arange(70), np.arange(70, 90))
                ]
            )
            
            trial = Mock(spec=optuna.Trial)
            trial.suggest_int = Mock(side_effect=[10, 32, 2])
            trial.suggest_float = Mock(side_effect=[0.2, 0.001])
            trial.suggest_categorical = Mock(side_effect=['adam', False])
            trial.report = Mock()
            trial.should_prune = Mock(return_value=False)
            
            with patch.object(lstm_optuna, '_train_model', return_value=(Mock(), 0.8)):
                score = lstm_optuna._objective(trial, X[:90], y[:90])
            
            # Check that TimeSeriesSplit was created with correct params
            mock_tscv.assert_called_once_with(n_splits=2)
    
    def test_feature_importance(self, lstm_optuna):
        """Test feature importance (should raise NotImplementedError for LSTM)."""
        with pytest.raises(NotImplementedError):
            lstm_optuna.feature_importances_
    
    def test_score_method(self, lstm_optuna, sample_features_data):
        """Test score method."""
        X = sample_features_data.drop('label', axis=1)
        y = (sample_features_data['label'] > 0).astype(int)
        
        mock_wrapper = Mock()
        mock_wrapper.score = Mock(return_value=0.85)
        lstm_optuna.wrapper = mock_wrapper
        lstm_optuna.feature_names_ = X.columns.tolist()
        
        score = lstm_optuna.score(X, y)
        
        assert score == 0.85
        mock_wrapper.score.called_once()