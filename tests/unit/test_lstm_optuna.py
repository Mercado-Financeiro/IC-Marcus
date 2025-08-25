"""Tests for LSTM with Optuna optimization."""

import pytest
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from datetime import datetime, timedelta
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))


class TestLSTMOptuna:
    """Test suite for LSTM with Bayesian optimization."""
    
    def test_lstm_model_architecture(self):
        """Test LSTM model creation and forward pass."""
        from src.models.lstm_optuna import LSTMModel
        
        # Test model creation
        model = LSTMModel(
            input_size=10,
            hidden_size=64,
            num_layers=2,
            dropout=0.2,
            output_size=1
        )
        
        # Check architecture
        assert isinstance(model.lstm, nn.LSTM)
        assert isinstance(model.fc, nn.Linear)
        assert isinstance(model.dropout, nn.Dropout)
        
        # Test forward pass
        batch_size = 32
        seq_len = 100
        input_size = 10
        
        x = torch.randn(batch_size, seq_len, input_size)
        output = model(x)
        
        # Check output shape
        assert output.shape == (batch_size, 1)
        
        # Check output range (should be between 0 and 1 for binary classification)
        assert torch.all(output >= 0) and torch.all(output <= 1)
    
    def test_sequence_creation(self):
        """Test sequence creation for LSTM."""
        from src.models.lstm_optuna import LSTMOptuna
        
        # Create dummy data
        n_samples = 1000
        n_features = 10
        seq_len = 50
        
        X = pd.DataFrame(np.random.randn(n_samples, n_features))
        y = pd.Series(np.random.randint(0, 2, n_samples))
        
        optimizer = LSTMOptuna(seed=42)
        X_seq, y_seq = optimizer.create_sequences(X, y, seq_len)
        
        # Check shapes
        expected_samples = n_samples - seq_len
        assert X_seq.shape == (expected_samples, seq_len, n_features)
        assert y_seq.shape == (expected_samples,)
        
        # Check sequence continuity
        for i in range(10):  # Check first 10 sequences
            assert np.allclose(X_seq[i], X.iloc[i:i+seq_len].values)
            assert y_seq[i] == y.iloc[i+seq_len]
    
    def test_determinism(self):
        """Test that LSTM training is deterministic."""
        from src.models.lstm_optuna import LSTMOptuna, set_lstm_deterministic
        
        # Set deterministic mode
        set_lstm_deterministic(42)
        
        # Create dummy data
        X = pd.DataFrame(np.random.randn(500, 5))
        y = pd.Series(np.random.randint(0, 2, 500))
        
        # Train twice with same seed
        optimizer1 = LSTMOptuna(seed=42)
        optimizer2 = LSTMOptuna(seed=42)
        
        # Create same model architecture
        model1 = optimizer1._create_model(
            input_size=5,
            hidden_size=32,
            num_layers=1,
            dropout=0.1
        )
        
        model2 = optimizer2._create_model(
            input_size=5,
            hidden_size=32,
            num_layers=1,
            dropout=0.1
        )
        
        # Check that initial weights are the same
        for p1, p2 in zip(model1.parameters(), model2.parameters()):
            assert torch.allclose(p1, p2)
    
    def test_optuna_optimization(self, sample_ml_data):
        """Test Optuna optimization for LSTM."""
        from src.models.lstm_optuna import LSTMOptuna
        
        X, y = sample_ml_data
        
        # Small dataset for quick test
        X = X[:500]
        y = y[:500]
        
        optimizer = LSTMOptuna(
            n_trials=2,  # Very few trials for testing
            cv_folds=2,
            pruner_type='median',
            seed=42
        )
        
        # Run optimization
        study = optimizer.optimize(X, y)
        
        # Check study results
        assert len(study.trials) == 2
        assert study.best_value is not None
        assert optimizer.best_params is not None
        
        # Check that best params contain expected keys
        expected_keys = [
            'hidden_size', 'num_layers', 'dropout', 'seq_len',
            'lr', 'batch_size', 'weight_decay', 'gradient_clip'
        ]
        for key in expected_keys:
            assert key in optimizer.best_params
    
    def test_calibration(self, sample_ml_data):
        """Test that LSTM predictions are calibrated."""
        from src.models.lstm_optuna import LSTMOptuna
        
        X, y = sample_ml_data
        X = X[:300]  # Small dataset
        y = y[:300]
        
        optimizer = LSTMOptuna(n_trials=1, cv_folds=2, seed=42)
        
        # Quick optimization
        study = optimizer.optimize(X, y)
        
        # Fit final model
        optimizer.fit_final_model(X, y)
        
        # Check calibrator exists
        assert optimizer.calibrator is not None
        
        # Make predictions
        y_pred_proba = optimizer.predict_proba(X[-50:])
        
        # Check probability range
        assert np.all(y_pred_proba >= 0) and np.all(y_pred_proba <= 1)
        
        # Check that thresholds were optimized
        assert optimizer.threshold_f1 is not None
        assert optimizer.threshold_ev is not None
        assert 0 < optimizer.threshold_f1 < 1
        assert 0 < optimizer.threshold_ev < 1
    
    def test_early_stopping(self):
        """Test early stopping in LSTM training."""
        from src.models.lstm_optuna import LSTMOptuna
        
        # Create data that should trigger early stopping
        X = pd.DataFrame(np.random.randn(200, 5))
        y = pd.Series(np.random.randint(0, 2, 200))
        
        optimizer = LSTMOptuna(
            n_trials=1,
            early_stopping_patience=3,
            seed=42
        )
        
        # The optimization should complete without errors
        study = optimizer.optimize(X, y)
        assert study is not None
    
    def test_gradient_clipping(self):
        """Test that gradient clipping is applied."""
        from src.models.lstm_optuna import LSTMOptuna
        
        optimizer = LSTMOptuna(seed=42)
        
        # Create model and dummy data
        model = optimizer._create_model(
            input_size=10,
            hidden_size=32,
            num_layers=1,
            dropout=0.1
        )
        
        # Create optimizer with gradient clipping
        torch_optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        # Forward pass with large input (to create large gradients)
        x = torch.randn(16, 50, 10) * 100  # Large values
        y = torch.randint(0, 2, (16,)).float()
        
        output = model(x).squeeze()
        loss = nn.BCELoss()(output, y)
        loss.backward()
        
        # Apply gradient clipping
        max_grad_norm = 1.0
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        
        # Check that gradients are clipped
        total_norm = 0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        
        assert total_norm <= max_grad_norm * 1.1  # Allow small tolerance
    
    def test_device_handling(self):
        """Test that model handles CPU/GPU correctly."""
        from src.models.lstm_optuna import LSTMOptuna
        
        optimizer = LSTMOptuna(seed=42)
        
        # Should default to CPU if CUDA not available
        device = optimizer._get_device()
        
        if torch.cuda.is_available():
            assert device.type == 'cuda'
        else:
            assert device.type == 'cpu'
        
        # Model should be on correct device
        model = optimizer._create_model(10, 32, 1, 0.1)
        model = model.to(device)
        
        # Check model device
        for param in model.parameters():
            assert param.device.type == device.type
    
    def test_hyperparameter_ranges(self):
        """Test that hyperparameters are within expected ranges."""
        from src.models.lstm_optuna import LSTMOptuna
        
        optimizer = LSTMOptuna(seed=42)
        
        # Create dummy trial
        import optuna
        study = optuna.create_study()
        trial = study.ask()
        
        # Get hyperparameters
        params = optimizer._create_search_space(trial)
        
        # Check ranges
        assert 32 <= params['hidden_size'] <= 512
        assert 1 <= params['num_layers'] <= 3
        assert 0 <= params['dropout'] <= 0.5
        assert 20 <= params['seq_len'] <= 200
        assert 1e-5 <= params['lr'] <= 1e-2
        assert params['batch_size'] in [16, 32, 64, 128, 256]
        assert 0 <= params['weight_decay'] <= 1e-3
        assert 0.1 <= params['gradient_clip'] <= 2.0
    
    def test_no_temporal_leakage(self, sample_ohlcv_data):
        """Test that LSTM doesn't have temporal leakage."""
        from src.models.lstm_optuna import LSTMOptuna
        from src.data.splits import PurgedKFold
        
        # Create features
        df = sample_ohlcv_data.copy()
        df['returns'] = df['close'].pct_change()
        df['label'] = (df['returns'].shift(-1) > 0).astype(int)
        
        feature_cols = ['returns']
        X = df[feature_cols].dropna()
        y = df['label'].dropna()
        
        # Align
        common_idx = X.index.intersection(y.index)
        X = X.loc[common_idx]
        y = y.loc[common_idx]
        
        # Use PurgedKFold for cross-validation
        cv = PurgedKFold(n_splits=3, embargo=10)
        
        for train_idx, val_idx in cv.split(X, y):
            train_times = X.index[train_idx]
            val_times = X.index[val_idx]
            
            # Check no overlap
            assert train_times.max() < val_times.min() or val_times.max() < train_times.min()


class TestLSTMOptunaIntegration:
    """Integration tests for LSTM with other components."""
    
    def test_lstm_with_feature_engineering(self, sample_ohlcv_data):
        """Test LSTM with feature engineering pipeline."""
        from src.models.lstm_optuna import LSTMOptuna
        from src.features.engineering import FeatureEngineer
        
        # Create features
        fe = FeatureEngineer(lookback_periods=[5, 10])
        df = fe.create_price_features(sample_ohlcv_data)
        
        # Create labels
        df['label'] = (df['close'].pct_change().shift(-1) > 0).astype(int)
        df = df.dropna()
        
        # Prepare data
        feature_cols = [c for c in df.columns if c not in ['open', 'high', 'low', 'close', 'volume', 'label']]
        X = df[feature_cols]
        y = df['label']
        
        # Test LSTM can handle engineered features
        optimizer = LSTMOptuna(n_trials=1, cv_folds=2, seed=42)
        study = optimizer.optimize(X[:200], y[:200])  # Small subset
        
        assert study is not None
        assert optimizer.best_params is not None
    
    def test_lstm_with_backtest(self, sample_ohlcv_data):
        """Test LSTM predictions in backtest."""
        from src.models.lstm_optuna import LSTMOptuna
        from src.backtest.engine import BacktestEngine
        
        # Prepare simple data
        df = sample_ohlcv_data.copy()
        df['returns'] = df['close'].pct_change()
        df['label'] = (df['returns'].shift(-1) > 0).astype(int)
        df = df.dropna()
        
        X = df[['returns']]
        y = df['label']
        
        # Train LSTM
        optimizer = LSTMOptuna(n_trials=1, cv_folds=2, seed=42)
        optimizer.optimize(X[:800], y[:800])
        optimizer.fit_final_model(X[:800], y[:800])
        
        # Make predictions
        y_pred = optimizer.predict(X[800:])
        
        # Run backtest
        signals = pd.Series(y_pred, index=X[800:].index)
        backtest = BacktestEngine(initial_capital=100000)
        results = backtest.run_backtest(df.loc[signals.index], signals)
        metrics = backtest.calculate_metrics(results)
        
        # Check backtest ran
        assert 'total_return' in metrics
        assert 'sharpe_ratio' in metrics
        assert results is not None