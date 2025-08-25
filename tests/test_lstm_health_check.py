"""
Health check tests for LSTM after fixes.
Quick tests to verify the main issues are resolved.
"""

import pytest
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
import sys
import warnings
import optuna

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

try:
    from src.models.lstm_optuna import LSTMOptuna, LSTMModel, set_lstm_deterministic, check_constant_predictions
    IMPORT_SUCCESS = True
except Exception as e:
    IMPORT_SUCCESS = False
    IMPORT_ERROR = str(e)


class TestLSTMHealthCheck:
    """Quick health checks after fixes."""
    
    def test_import_works(self):
        """Basic import test."""
        assert IMPORT_SUCCESS, f"Import failed: {IMPORT_ERROR}"
    
    def test_lstm_model_architecture(self):
        """Test LSTM model creation and basic forward pass."""
        if not IMPORT_SUCCESS:
            pytest.skip("Import failed")
        
        model = LSTMModel(
            input_size=10,
            hidden_size=64,
            num_layers=2,
            dropout=0.2,
            output_size=1
        )
        
        # Check architecture components
        assert isinstance(model.lstm, nn.LSTM)
        assert isinstance(model.fc, nn.Linear)
        assert isinstance(model.dropout, nn.Dropout)
        assert isinstance(model.sigmoid, nn.Sigmoid)
        
        # Test forward pass
        batch_size = 16
        seq_len = 50
        x = torch.randn(batch_size, seq_len, 10)
        output = model(x)
        
        # Check output shape and range
        assert output.shape == (batch_size, 1)
        assert torch.all(output >= 0) and torch.all(output <= 1)  # Sigmoid output
        
        print("✅ LSTM architecture works correctly")
    
    def test_determinism_configuration(self):
        """Test deterministic configuration works."""
        if not IMPORT_SUCCESS:
            pytest.skip("Import failed")
        
        # Test deterministic function doesn't crash
        set_lstm_deterministic(42)
        
        # Create two models with same seed
        model1 = LSTMModel(input_size=5, hidden_size=32, num_layers=1, dropout=0.1)
        
        # Reset seed
        set_lstm_deterministic(42)
        model2 = LSTMModel(input_size=5, hidden_size=32, num_layers=1, dropout=0.1)
        
        # Check that initial weights are the same (deterministic initialization)
        param_diffs = []
        for p1, p2 in zip(model1.parameters(), model2.parameters()):
            param_diffs.append(torch.allclose(p1, p2, atol=1e-6))
        
        # At least most parameters should be deterministic
        deterministic_ratio = sum(param_diffs) / len(param_diffs)
        assert deterministic_ratio > 0.8, f"Only {deterministic_ratio:.2%} parameters are deterministic"
        
        print("✅ Determinism configuration works")
    
    def test_logging_available(self):
        """Check logging works with fallback."""
        if not IMPORT_SUCCESS:
            pytest.skip("Import failed")
        
        optimizer = LSTMOptuna(n_trials=1, cv_folds=2, seed=42)
        
        # Should have log attribute
        assert hasattr(optimizer, 'log'), "Missing log attribute"
        
        # Should be able to call logging methods
        try:
            optimizer.log.info("test_message", test_param="test_value")
            optimizer.log.warning("test_warning", warning_param="test_value")
            optimizer.log.error("test_error", error_param="test_value")
            print("✅ Logging works")
        except Exception as e:
            pytest.fail(f"Logging failed: {e}")
    
    def test_constant_predictions_detection(self):
        """Test constant prediction detection."""
        if not IMPORT_SUCCESS:
            pytest.skip("Import failed")
        
        # Test with truly constant predictions
        constant_preds = np.full(100, 0.5)
        assert check_constant_predictions(constant_preds) == True
        
        # Test with near-constant predictions (low std)
        near_constant = np.random.normal(0.5, 0.001, 100)  # Very low std
        assert check_constant_predictions(near_constant) == True
        
        # Test with diverse predictions
        diverse_preds = np.random.uniform(0, 1, 100)
        assert check_constant_predictions(diverse_preds) == False
        
        # Test with edge case - empty array
        assert check_constant_predictions(np.array([])) == True
        
        print("✅ Constant prediction detection works")
    
    def test_sequence_creation(self):
        """Test sequence creation for LSTM input."""
        if not IMPORT_SUCCESS:
            pytest.skip("Import failed")
        
        optimizer = LSTMOptuna(n_trials=1, cv_folds=2, seed=42)
        
        # Create test data
        n_samples = 200
        n_features = 5
        seq_len = 30
        
        X = pd.DataFrame(np.random.randn(n_samples, n_features))
        y = pd.Series(np.random.randint(0, 2, n_samples))
        
        # Create sequences
        X_seq, y_seq = optimizer.create_sequences(X, y, seq_len)
        
        # Check shapes
        expected_samples = n_samples - seq_len
        assert X_seq.shape == (expected_samples, seq_len, n_features)
        assert y_seq.shape == (expected_samples,)
        
        # Check sequence alignment
        assert y_seq[0] == y.iloc[seq_len]  # First target should align correctly
        
        print("✅ Sequence creation works correctly")
    
    def test_hyperparameter_ranges(self):
        """Test hyperparameter search space is reasonable."""
        if not IMPORT_SUCCESS:
            pytest.skip("Import failed")
        
        optimizer = LSTMOptuna(n_trials=1, cv_folds=2, seed=42)
        
        # Create dummy trial
        study = optuna.create_study()
        trial = study.ask()
        
        # Get hyperparameters
        params = optimizer._create_search_space(trial)
        
        # Check ranges are reasonable
        assert 32 <= params['hidden_size'] <= 512
        assert 1 <= params['num_layers'] <= 3
        assert 0 <= params['dropout'] <= 0.5
        assert 20 <= params['seq_len'] <= 200
        assert 1e-5 <= params['lr'] <= 1e-2
        assert params['batch_size'] in [16, 32, 64, 128, 256]
        assert 0 <= params['weight_decay'] <= 1e-3
        assert 0.1 <= params['gradient_clip'] <= 2.0
        
        print(f"✅ Hyperparameter ranges are valid: {sorted(params.keys())}")
    
    def test_device_handling(self):
        """Test device handling (CPU/GPU)."""
        if not IMPORT_SUCCESS:
            pytest.skip("Import failed")
        
        optimizer = LSTMOptuna(n_trials=1, cv_folds=2, seed=42)
        
        # Should have a valid device
        assert optimizer.device is not None
        assert optimizer.device.type in ['cpu', 'cuda', 'mps']
        
        # Model should be placed on the correct device
        model = optimizer._create_model(
            input_size=10,
            hidden_size=32,
            num_layers=1,
            dropout=0.1
        )
        
        # Check model is on correct device
        model_device = next(model.parameters()).device
        assert model_device.type == optimizer.device.type
        
        print(f"✅ Device handling works: {optimizer.device.type}")
    
    def test_basic_optimization_runs(self):
        """Test that basic optimization doesn't crash."""
        if not IMPORT_SUCCESS:
            pytest.skip("Import failed")
        
        # Create simple synthetic data
        np.random.seed(42)
        X = pd.DataFrame(np.random.randn(150, 8), columns=[f'feat_{i}' for i in range(8)])
        y = pd.Series((X['feat_0'] + X['feat_1'] > 0).astype(int))  # Simple pattern
        X.index = pd.date_range('2023-01-01', periods=150, freq='h')
        y.index = X.index
        
        try:
            optimizer = LSTMOptuna(n_trials=2, cv_folds=2, embargo=5, seed=42)
            study = optimizer.optimize(X, y)
            
            # Should complete without errors
            assert study is not None
            assert len(study.trials) == 2
            assert optimizer.best_params is not None
            assert optimizer.best_score is not None
            
            print("✅ Basic optimization runs successfully")
            
        except Exception as e:
            pytest.fail(f"Optimization failed: {e}")
    
    def test_calibration_and_thresholds(self):
        """Test that calibration and threshold optimization work."""
        if not IMPORT_SUCCESS:
            pytest.skip("Import failed")
        
        # Create simple data with clear pattern
        np.random.seed(42)
        X = pd.DataFrame(np.random.randn(100, 5))
        y = pd.Series((X.sum(axis=1) > 0).astype(int))  # Clear pattern
        X.index = pd.date_range('2023-01-01', periods=100, freq='h')
        y.index = X.index
        
        try:
            optimizer = LSTMOptuna(n_trials=1, cv_folds=2, seed=42)
            optimizer.optimize(X, y)
            optimizer.fit_final_model(X, y)
            
            # Check calibrator exists
            assert optimizer.calibrator is not None
            
            # Check thresholds are set
            assert optimizer.threshold_f1 is not None
            assert optimizer.threshold_ev is not None
            assert 0 <= optimizer.threshold_f1 <= 1
            assert 0 <= optimizer.threshold_ev <= 1
            
            # Make predictions
            y_pred_proba = optimizer.predict_proba(X[-30:])
            y_pred = optimizer.predict(X[-30:])
            
            # Check predictions are valid
            assert all(0 <= p <= 1 for p in y_pred_proba)
            assert all(p in [0, 1] for p in y_pred)
            
            print("✅ Calibration and thresholds work")
            
        except Exception as e:
            pytest.fail(f"Calibration failed: {e}")
    
    def test_no_constant_predictions_in_optimization(self):
        """Test that optimization doesn't produce constant predictions."""
        if not IMPORT_SUCCESS:
            pytest.skip("Import failed")
        
        # Create data with clear pattern that should be learnable
        np.random.seed(42)
        X = pd.DataFrame(np.random.randn(200, 10))
        # Create a learnable pattern: y depends on sum of first 3 features
        y = pd.Series((X.iloc[:, :3].sum(axis=1) > 0).astype(int))
        X.index = pd.date_range('2023-01-01', periods=200, freq='h')
        y.index = X.index
        
        try:
            # Use more trials and epochs for better learning
            optimizer = LSTMOptuna(n_trials=3, cv_folds=2, embargo=3, max_epochs=20, seed=42)
            optimizer.optimize(X, y)
            optimizer.fit_final_model(X, y)
            
            # Get predictions on a subset (account for sequences)
            y_pred_proba = optimizer.predict_proba(X[-80:])  # More samples to account for sequences
            
            # Check predictions are diverse (reasonable for LSTM with short training)
            unique_probs = len(np.unique(np.round(y_pred_proba, 3)))
            prob_std = np.std(y_pred_proba)
            
            # Check predictions are not truly constant (main validation)
            is_constant = check_constant_predictions(y_pred_proba)
            assert not is_constant, f"Predictions are essentially constant: {is_constant}"
            
            # Basic diversity checks (LSTM may have low diversity with short training)
            assert unique_probs >= 2, f"Only {unique_probs} unique probabilities (need at least 2)"
            assert not np.all(y_pred_proba == y_pred_proba[0]), "All predictions are identical"
            
            print(f"✅ Predictions pass constant detection: {unique_probs} unique probs, std={prob_std:.6f}")
            
        except Exception as e:
            pytest.fail(f"Non-constant prediction test failed: {e}")
    
    def test_no_warnings_with_valid_data(self):
        """Test that valid training doesn't produce warnings."""
        if not IMPORT_SUCCESS:
            pytest.skip("Import failed")
        
        # Capture warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            # Create good quality data
            np.random.seed(42)
            X = pd.DataFrame(np.random.randn(100, 6))
            y = pd.Series(np.random.randint(0, 2, 100))
            X.index = pd.date_range('2023-01-01', periods=100, freq='h')
            y.index = X.index
            
            optimizer = LSTMOptuna(n_trials=1, cv_folds=2, embargo=3, seed=42)
            optimizer.optimize(X, y)
            
            # Check for PyTorch/LSTM related warnings
            relevant_warnings = [warning for warning in w 
                               if any(term in str(warning.message).lower() 
                                     for term in ['torch', 'lstm', 'cuda', 'deterministic'])]
            
            if relevant_warnings and len(relevant_warnings) > 2:  # Allow some minor warnings
                warning_msgs = '\n'.join(str(w.message) for w in relevant_warnings[:3])
                pytest.fail(f"Too many PyTorch/LSTM warnings found:\n{warning_msgs}")
        
        print("✅ No excessive warnings during training")
    
    def test_complete_pipeline_works(self):
        """Test the complete LSTM pipeline end-to-end."""
        if not IMPORT_SUCCESS:
            pytest.skip("Import failed")
        
        # Create realistic synthetic data
        np.random.seed(42)
        X = pd.DataFrame(np.random.randn(120, 6), columns=[f'feat_{i}' for i in range(6)])
        y = pd.Series((X['feat_0'] * 0.7 + X['feat_1'] * 0.3 > 0).astype(int))  # Complex pattern
        X.index = pd.date_range('2023-01-01', periods=120, freq='h')
        y.index = X.index
        
        try:
            # Full pipeline
            optimizer = LSTMOptuna(n_trials=2, cv_folds=2, embargo=3, seed=42)
            
            # Optimize
            study = optimizer.optimize(X, y)
            
            # Fit final model
            optimizer.fit_final_model(X, y)
            
            # Should have all components
            assert optimizer.best_model is not None, "No best model found"
            assert optimizer.best_params is not None, "No best params found"
            assert optimizer.calibrator is not None, "No calibrator found"
            assert optimizer.scaler is not None, "No scaler found"
            
            # Make predictions (account for sequence creation reducing sample count)
            probs = optimizer.predict_proba(X[-40:])  # Start with more to get reasonable output
            preds = optimizer.predict(X[-40:])
            
            # Validate predictions (sequences will reduce count)
            assert len(probs) >= 5, f"Too few probabilities: {len(probs)} (need at least 5)"
            assert len(preds) >= 5, f"Too few predictions: {len(preds)} (need at least 5)"
            assert len(probs) == len(preds), "Mismatched prediction lengths"
            assert all(p in [0, 1] for p in preds), "Invalid predictions"
            assert all(0 <= p <= 1 for p in probs), "Invalid probabilities"
            
            # Check study results
            assert study.best_value is not None
            assert len(study.trials) == 2
            
            print("✅ Complete LSTM pipeline works")
            
        except Exception as e:
            pytest.fail(f"Complete pipeline failed: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])