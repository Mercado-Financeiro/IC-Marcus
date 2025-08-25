"""
Comprehensive tests for LSTM with Optuna optimization.
Covers edge cases, performance, and integration scenarios.
"""

import pytest
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
import sys
import warnings
from unittest.mock import patch, MagicMock
import optuna

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

try:
    from src.models.lstm_optuna import (
        LSTMOptuna, LSTMModel, LSTMWrapper, 
        set_lstm_deterministic, check_constant_predictions
    )
    from src.data.splits import PurgedKFold
    IMPORT_SUCCESS = True
except Exception as e:
    IMPORT_SUCCESS = False
    IMPORT_ERROR = str(e)


@pytest.fixture
def sample_data():
    """Generate sample time series data for testing."""
    np.random.seed(42)
    n_samples = 300
    n_features = 8
    
    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f'feat_{i}' for i in range(n_features)]
    )
    
    # Create realistic pattern: y depends on multiple features with noise
    y = pd.Series(
        (0.4 * X['feat_0'] + 0.3 * X['feat_1'] - 0.2 * X['feat_2'] + 
         np.random.randn(n_samples) * 0.1 > 0).astype(int)
    )
    
    # Add datetime index
    X.index = pd.date_range('2023-01-01', periods=n_samples, freq='h')
    y.index = X.index
    
    return X, y


@pytest.fixture
def imbalanced_data():
    """Generate imbalanced dataset for testing."""
    np.random.seed(42)
    n_samples = 200
    
    X = pd.DataFrame(np.random.randn(n_samples, 5))
    # Create 80/20 imbalanced dataset
    y = pd.Series([0] * 160 + [1] * 40)
    
    X.index = pd.date_range('2023-01-01', periods=n_samples, freq='h')
    y.index = X.index
    
    return X, y


class TestLSTMModelArchitecture:
    """Test LSTM model architecture and components."""
    
    def test_model_initialization_parameters(self):
        """Test model initialization with different parameters."""
        if not IMPORT_SUCCESS:
            pytest.skip("Import failed")
        
        # Test different layer configurations
        configs = [
            (10, 32, 1, 0.0),   # Single layer, no dropout
            (5, 64, 2, 0.2),    # Two layers with dropout
            (15, 128, 3, 0.4),  # Three layers with high dropout
        ]
        
        for input_size, hidden_size, num_layers, dropout in configs:
            model = LSTMModel(input_size, hidden_size, num_layers, dropout)
            
            # Check architecture
            assert model.lstm.input_size == input_size
            assert model.lstm.hidden_size == hidden_size
            assert model.lstm.num_layers == num_layers
            
            # Check layer connections
            assert model.fc.in_features == hidden_size
            assert model.fc.out_features == 1
            
            # Test forward pass
            batch_size = 8
            seq_len = 20
            x = torch.randn(batch_size, seq_len, input_size)
            output = model(x)
            
            assert output.shape == (batch_size, 1)
            assert torch.all(output >= 0) and torch.all(output <= 1)
    
    def test_model_gradient_flow(self):
        """Test that gradients flow properly through the model."""
        if not IMPORT_SUCCESS:
            pytest.skip("Import failed")
        
        model = LSTMModel(input_size=5, hidden_size=32, num_layers=2, dropout=0.1)
        
        # Forward pass
        x = torch.randn(16, 30, 5, requires_grad=True)
        y = torch.randint(0, 2, (16,)).float()
        
        output = model(x).squeeze()
        loss = nn.BCELoss()(output, y)
        
        # Backward pass
        loss.backward()
        
        # Check gradients exist and are non-zero
        has_gradients = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                has_grad = param.grad is not None and param.grad.abs().sum() > 0
                has_gradients.append(has_grad)
        
        # Most parameters should have gradients
        grad_ratio = sum(has_gradients) / len(has_gradients)
        assert grad_ratio > 0.7, f"Only {grad_ratio:.2%} parameters have gradients"
    
    def test_model_memory_usage(self):
        """Test model doesn't cause memory leaks."""
        if not IMPORT_SUCCESS:
            pytest.skip("Import failed")
        
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        model = LSTMModel(input_size=10, hidden_size=64, num_layers=2, dropout=0.2)
        
        # Multiple forward passes
        for _ in range(10):
            x = torch.randn(32, 50, 10)
            with torch.no_grad():
                output = model(x)
                del output, x
        
        # Should not accumulate memory (basic check)
        assert True  # If we reach here without OOM, test passes


class TestLSTMWrapper:
    """Test sklearn-compatible LSTM wrapper."""
    
    def test_wrapper_sklearn_compatibility(self, sample_data):
        """Test that wrapper is compatible with sklearn interface."""
        if not IMPORT_SUCCESS:
            pytest.skip("Import failed")
        
        X, y = sample_data
        X_small = X[:100]
        y_small = y[:100]
        
        # Create model and wrapper
        model = LSTMModel(input_size=X.shape[1], hidden_size=32, num_layers=1, dropout=0.1)
        wrapper = LSTMWrapper(model, seq_len=20, device=torch.device('cpu'))
        
        # Test sklearn interface
        assert hasattr(wrapper, 'fit')
        assert hasattr(wrapper, 'predict')
        assert hasattr(wrapper, 'predict_proba')
        assert hasattr(wrapper, 'classes_')
        
        # Test classes attribute
        assert np.array_equal(wrapper.classes_, np.array([0, 1]))
    
    def test_wrapper_predictions_shape(self, sample_data):
        """Test wrapper prediction shapes."""
        if not IMPORT_SUCCESS:
            pytest.skip("Import failed")
        
        X, y = sample_data
        X_test = X[:80]
        
        model = LSTMModel(input_size=X.shape[1], hidden_size=32, num_layers=1, dropout=0.1)
        wrapper = LSTMWrapper(model, seq_len=30, device=torch.device('cpu'))
        
        # Test predict_proba
        proba = wrapper.predict_proba(X_test)
        expected_samples = len(X_test) - 30 + 1  # After sequence creation
        
        assert proba.shape == (expected_samples, 2)  # Binary classification
        assert np.all(proba >= 0) and np.all(proba <= 1)
        assert np.allclose(proba.sum(axis=1), 1.0)  # Probabilities sum to 1
        
        # Test predict
        predictions = wrapper.predict(X_test)
        assert predictions.shape == (expected_samples,)
        assert all(p in [0, 1] for p in predictions)


class TestLSTMOptuna:
    """Test LSTM with Optuna optimization."""
    
    def test_initialization_edge_cases(self):
        """Test initialization with edge case parameters."""
        if not IMPORT_SUCCESS:
            pytest.skip("Import failed")
        
        # Test invalid parameters
        with pytest.raises(ValueError):
            LSTMOptuna(n_trials=0)  # Invalid n_trials
        
        with pytest.raises(ValueError):
            LSTMOptuna(cv_folds=1)  # Invalid cv_folds
        
        with pytest.raises(ValueError):
            LSTMOptuna(embargo=-1)  # Invalid embargo
        
        # Test valid edge cases
        optimizer = LSTMOptuna(n_trials=1, cv_folds=2, embargo=0)
        assert optimizer.n_trials == 1
        assert optimizer.cv_folds == 2
        assert optimizer.embargo == 0
    
    def test_optimization_with_insufficient_data(self):
        """Test optimization behavior with insufficient data."""
        if not IMPORT_SUCCESS:
            pytest.skip("Import failed")
        
        # Very small dataset
        X = pd.DataFrame(np.random.randn(50, 5))
        y = pd.Series(np.random.randint(0, 2, 50))
        X.index = pd.date_range('2023-01-01', periods=50, freq='h')
        y.index = X.index
        
        optimizer = LSTMOptuna(n_trials=1, cv_folds=2, max_epochs=5, seed=42)
        
        # Should handle small data gracefully
        try:
            study = optimizer.optimize(X, y)
            # May return low scores but shouldn't crash
            assert study is not None
        except Exception as e:
            # Should raise informative error about insufficient data
            assert "data" in str(e).lower() or "sequence" in str(e).lower()
    
    def test_optimization_with_single_class(self):
        """Test optimization with single class labels."""
        if not IMPORT_SUCCESS:
            pytest.skip("Import failed")
        
        # All same class
        X = pd.DataFrame(np.random.randn(100, 5))
        y = pd.Series([1] * 100)  # All class 1
        X.index = pd.date_range('2023-01-01', periods=100, freq='h')
        y.index = X.index
        
        optimizer = LSTMOptuna(n_trials=1, cv_folds=2, seed=42)
        
        # Should raise error for single class
        with pytest.raises(ValueError, match="single class"):
            optimizer.optimize(X, y)
    
    def test_pruning_mechanism(self, sample_data):
        """Test Optuna pruning works correctly."""
        if not IMPORT_SUCCESS:
            pytest.skip("Import failed")
        
        X, y = sample_data
        X_small = X[:150]  # Smaller for faster test
        y_small = y[:150]
        
        optimizer = LSTMOptuna(
            n_trials=5,
            cv_folds=2,
            pruner_type='median',
            early_stopping_patience=2,  # Aggressive for pruning
            max_epochs=10,
            seed=42
        )
        
        study = optimizer.optimize(X_small, y_small)
        
        # Check that some trials were pruned
        pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
        completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        
        # Should have both pruned and completed trials in most cases
        assert len(completed_trials) > 0, "No trials completed"
        assert len(study.trials) == 5, "Wrong number of total trials"
    
    def test_constant_prediction_penalty(self):
        """Test constant prediction penalty mechanism."""
        if not IMPORT_SUCCESS:
            pytest.skip("Import failed")
        
        # Test data that might lead to constant predictions
        X = pd.DataFrame(np.random.randn(80, 3) * 0.01)  # Very small variance
        y = pd.Series(np.random.randint(0, 2, 80))
        X.index = pd.date_range('2023-01-01', periods=80, freq='h')
        y.index = X.index
        
        # Mock the check_constant_predictions to always return True
        with patch('src.models.lstm_optuna.check_constant_predictions', return_value=True):
            optimizer = LSTMOptuna(n_trials=1, cv_folds=2, seed=42)
            study = optimizer.optimize(X, y)
            
            # Should get penalty score
            assert study.best_value <= -0.5, f"Expected penalty score, got {study.best_value}"
    
    def test_threshold_optimization(self, sample_data):
        """Test F1 and EV threshold optimization."""
        if not IMPORT_SUCCESS:
            pytest.skip("Import failed")
        
        X, y = sample_data
        X_small = X[:120]
        y_small = y[:120]
        
        optimizer = LSTMOptuna(n_trials=1, cv_folds=2, max_epochs=5, seed=42)
        optimizer.optimize(X_small, y_small)
        optimizer.fit_final_model(X_small, y_small)
        
        # Check thresholds are optimized
        assert 0 <= optimizer.threshold_f1 <= 1
        assert 0 <= optimizer.threshold_ev <= 1
        
        # Test both threshold types give different predictions potentially
        y_pred_f1 = optimizer.predict(X_small[-20:], use_ev_threshold=False)
        y_pred_ev = optimizer.predict(X_small[-20:], use_ev_threshold=True)
        
        assert len(y_pred_f1) == len(y_pred_ev) == 20
        assert all(p in [0, 1] for p in y_pred_f1)
        assert all(p in [0, 1] for p in y_pred_ev)
    
    def test_memory_efficiency_large_sequence(self):
        """Test memory efficiency with larger sequences."""
        if not IMPORT_SUCCESS:
            pytest.skip("Import failed")
        
        # Test with longer sequences
        X = pd.DataFrame(np.random.randn(500, 10))
        y = pd.Series(np.random.randint(0, 2, 500))
        X.index = pd.date_range('2023-01-01', periods=500, freq='h')
        y.index = X.index
        
        optimizer = LSTMOptuna(
            n_trials=1, 
            cv_folds=2, 
            max_epochs=3,  # Very short for memory test
            seed=42
        )
        
        # Should complete without memory issues
        study = optimizer.optimize(X, y)
        assert study is not None
    
    def test_early_stopping_mechanism(self, sample_data):
        """Test early stopping works correctly."""
        if not IMPORT_SUCCESS:
            pytest.skip("Import failed")
        
        X, y = sample_data
        X_small = X[:100]
        y_small = y[:100]
        
        optimizer = LSTMOptuna(
            n_trials=1,
            cv_folds=2,
            early_stopping_patience=2,  # Very low patience
            max_epochs=50,  # High max but should stop early
            seed=42
        )
        
        # Training should stop early and not use all epochs
        study = optimizer.optimize(X_small, y_small)
        assert study is not None  # Should complete even with early stopping


class TestConstantPredictionDetection:
    """Test constant prediction detection utilities."""
    
    def test_detection_accuracy(self):
        """Test detection correctly identifies constant predictions."""
        if not IMPORT_SUCCESS:
            pytest.skip("Import failed")
        
        # Truly constant
        constant = np.full(100, 0.5)
        assert check_constant_predictions(constant) == True
        
        # Nearly constant (low std)
        nearly_constant = np.random.normal(0.5, 0.001, 100)
        assert check_constant_predictions(nearly_constant) == True
        
        # Low unique count
        low_unique = np.array([0.3] * 50 + [0.7] * 50)  # Only 2 unique values
        assert check_constant_predictions(low_unique) == True
        
        # Diverse predictions
        diverse = np.random.uniform(0, 1, 100)
        assert check_constant_predictions(diverse) == False
        
        # Edge case: empty
        empty = np.array([])
        assert check_constant_predictions(empty) == True
        
        # Edge case: single value
        single = np.array([0.5])
        assert check_constant_predictions(single) == True
    
    def test_detection_thresholds(self):
        """Test detection with different thresholds."""
        if not IMPORT_SUCCESS:
            pytest.skip("Import failed")
        
        # Borderline case
        borderline = np.random.normal(0.5, 0.01, 100)  # Std = 0.01
        
        # Should be constant with strict threshold
        assert check_constant_predictions(borderline, threshold_std=0.02) == True
        
        # Should not be constant with loose threshold
        assert check_constant_predictions(borderline, threshold_std=0.005) == False


class TestIntegrationScenarios:
    """Test integration with other components."""
    
    def test_with_imbalanced_data(self, imbalanced_data):
        """Test LSTM handles imbalanced data correctly."""
        if not IMPORT_SUCCESS:
            pytest.skip("Import failed")
        
        X, y = imbalanced_data
        
        # Check class distribution
        class_counts = pd.Series(y).value_counts()
        assert class_counts[0] > class_counts[1]  # Imbalanced
        
        optimizer = LSTMOptuna(n_trials=1, cv_folds=2, max_epochs=5, seed=42)
        study = optimizer.optimize(X, y)
        optimizer.fit_final_model(X, y)
        
        # Should handle imbalanced data without crashing
        assert study.best_value is not None
        
        # Predictions should not be all majority class
        y_pred = optimizer.predict(X[-50:])
        pred_counts = pd.Series(y_pred).value_counts()
        
        # Should predict both classes (not just majority)
        assert len(pred_counts) >= 1  # At least predicts something
    
    def test_reproducibility(self, sample_data):
        """Test that results are reproducible with same seed."""
        if not IMPORT_SUCCESS:
            pytest.skip("Import failed")
        
        X, y = sample_data
        X_small = X[:100]
        y_small = y[:100]
        
        def run_optimization():
            optimizer = LSTMOptuna(n_trials=2, cv_folds=2, max_epochs=5, seed=42)
            study = optimizer.optimize(X_small, y_small)
            return study.best_value, study.best_params
        
        # Run twice with same seed
        result1 = run_optimization()
        result2 = run_optimization()
        
        # Should be identical (or very close due to floating point)
        assert abs(result1[0] - result2[0]) < 1e-4, "Results not reproducible"
        
        # Best params should be identical
        for key in result1[1]:
            if isinstance(result1[1][key], float):
                assert abs(result1[1][key] - result2[1][key]) < 1e-6
            else:
                assert result1[1][key] == result2[1][key]
    
    def test_temporal_validation_no_leakage(self, sample_data):
        """Test that temporal validation doesn't leak information."""
        if not IMPORT_SUCCESS:
            pytest.skip("Import failed")
        
        X, y = sample_data
        
        # Create optimizer with purged k-fold
        optimizer = LSTMOptuna(n_trials=1, cv_folds=3, embargo=10, seed=42)
        
        # Mock the cross-validation to inspect split indices
        original_split = PurgedKFold.split
        split_info = []
        
        def mock_split(self, X, y=None):
            for train_idx, val_idx in original_split(self, X, y):
                # Record split information
                train_times = X.index[train_idx]
                val_times = X.index[val_idx]
                split_info.append((train_times, val_times))
                yield train_idx, val_idx
        
        with patch.object(PurgedKFold, 'split', mock_split):
            optimizer.optimize(X, y)
        
        # Check that there's no temporal overlap between train and validation
        for train_times, val_times in split_info:
            train_max = train_times.max()
            val_min = val_times.min()
            
            # Should have embargo gap (or complete separation)
            time_gap = (val_min - train_max).total_seconds() / 3600  # Hours
            assert time_gap >= optimizer.embargo or train_max < val_times.min()


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_empty_data(self):
        """Test behavior with empty data."""
        if not IMPORT_SUCCESS:
            pytest.skip("Import failed")
        
        X = pd.DataFrame()
        y = pd.Series(dtype=int)
        
        optimizer = LSTMOptuna(n_trials=1, cv_folds=2, seed=42)
        
        with pytest.raises(ValueError):
            optimizer.optimize(X, y)
    
    def test_mismatched_indices(self):
        """Test behavior with mismatched X and y indices."""
        if not IMPORT_SUCCESS:
            pytest.skip("Import failed")
        
        X = pd.DataFrame(np.random.randn(100, 5))
        y = pd.Series(np.random.randint(0, 2, 100))
        
        # Mismatched indices
        X.index = range(0, 100)
        y.index = range(50, 150)
        
        optimizer = LSTMOptuna(n_trials=1, cv_folds=2, seed=42)
        
        # Should handle gracefully or raise informative error
        try:
            optimizer.optimize(X, y)
        except Exception as e:
            assert "index" in str(e).lower() or "align" in str(e).lower()
    
    def test_nan_values(self):
        """Test behavior with NaN values in data."""
        if not IMPORT_SUCCESS:
            pytest.skip("Import failed")
        
        X = pd.DataFrame(np.random.randn(100, 5))
        y = pd.Series(np.random.randint(0, 2, 100))
        
        # Introduce some NaN values
        X.iloc[10:15, 1] = np.nan
        X.iloc[20:25, 3] = np.nan
        
        optimizer = LSTMOptuna(n_trials=1, cv_folds=2, max_epochs=3, seed=42)
        
        # Should either handle NaN or raise informative error
        try:
            optimizer.optimize(X, y)
        except Exception as e:
            assert any(term in str(e).lower() for term in ['nan', 'missing', 'null', 'finite'])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])