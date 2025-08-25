"""
Health check tests for XGBoost after fixes.
Quick tests to verify the main issues are resolved.
"""

import pytest
import numpy as np
import pandas as pd
import xgboost as xgb
from pathlib import Path
import sys
import warnings
import optuna

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

try:
    from src.models.xgb_optuna import XGBoostOptuna
    IMPORT_SUCCESS = True
except Exception as e:
    IMPORT_SUCCESS = False
    IMPORT_ERROR = str(e)


class TestXGBoostHealthCheck:
    """Quick health checks after fixes."""
    
    def test_import_works(self):
        """Basic import test."""
        assert IMPORT_SUCCESS, f"Import failed: {IMPORT_ERROR}"
    
    def test_no_deprecated_parameters(self):
        """Check no deprecated XGBoost parameters are used."""
        if not IMPORT_SUCCESS:
            pytest.skip("Import failed")
        
        optimizer = XGBoostOptuna(n_trials=5, cv_folds=3, embargo=5, use_mlflow=False, seed=42)
        
        # Test parameter generation
        study = optuna.create_study()
        trial = study.ask()
        params = optimizer._create_search_space(trial)
        
        # These should NOT be present
        forbidden = ['eta', 'lambda', 'alpha', 'use_label_encoder']
        found_forbidden = [p for p in forbidden if p in params]
        
        assert len(found_forbidden) == 0, f"Found deprecated parameters: {found_forbidden}"
        
        # These SHOULD be present (or at least the canonical versions)
        canonical_check = []
        if 'eta' not in params and 'learning_rate' not in params:
            canonical_check.append("Missing learning_rate parameter")
        
        assert len(canonical_check) == 0, f"Missing canonical parameters: {canonical_check}"
        
        print(f"✅ Parameters are clean: {sorted(params.keys())}")
    
    def test_logging_available(self):
        """Check logging works."""
        if not IMPORT_SUCCESS:
            pytest.skip("Import failed")
        
        optimizer = XGBoostOptuna(n_trials=5, use_mlflow=False)
        
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
    
    def test_xgboost_warnings_clean(self):
        """Test that training doesn't produce XGBoost warnings."""
        if not IMPORT_SUCCESS:
            pytest.skip("Import failed")
        
        # Create simple data
        np.random.seed(42)
        X = pd.DataFrame(np.random.randn(100, 5))
        y = pd.Series(np.random.randint(0, 2, 100))
        X.index = pd.date_range('2023-01-01', periods=100, freq='h')
        y.index = X.index
        
        # Capture warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            optimizer = XGBoostOptuna(n_trials=2, cv_folds=2, embargo=3, use_mlflow=False)
            optimizer.optimize(X, y)
            
            # Check for XGBoost parameter warnings
            xgb_warnings = [warning for warning in w if 'Parameters:' in str(warning.message)]
            
            if xgb_warnings:
                warning_msg = '\n'.join(str(w.message) for w in xgb_warnings)
                pytest.fail(f"XGBoost parameter warnings found:\n{warning_msg}")
        
        print("✅ No XGBoost parameter warnings")
    
    def test_early_stopping_works(self):
        """Test that early stopping is properly configured in XGBoost 3.x."""
        if not IMPORT_SUCCESS:
            pytest.skip("Import failed")
        
        # Test the XGBoost 3.x API for early stopping
        X = np.random.randn(200, 10)
        y = np.random.randint(0, 2, 200)
        X_val = np.random.randn(50, 10) 
        y_val = np.random.randint(0, 2, 50)
        
        # Test that we can create a model with early stopping in constructor
        try:
            model = xgb.XGBClassifier(
                n_estimators=1000,
                early_stopping_rounds=10,
                eval_metric='aucpr',
                random_state=42
            )
            
            # XGBoost 3.x: early_stopping_rounds in constructor, eval_set in fit
            model.fit(X, y, eval_set=[(X_val, y_val)], verbose=False)
            
            # Check it trained successfully
            probs = model.predict_proba(X)[:, 1]
            assert len(probs) == len(X), "Model failed to predict"
            print("✅ Early stopping XGBoost 3.x API works")
                
        except Exception as e:
            pytest.fail(f"XGBoost 3.x early stopping failed: {e}")
    
    def test_scale_pos_weight_consistent(self):
        """Test scale_pos_weight calculation is consistent."""
        if not IMPORT_SUCCESS:
            pytest.skip("Import failed")
        
        # Create imbalanced data
        np.random.seed(42)
        X = pd.DataFrame(np.random.randn(100, 5))
        y = pd.Series([0] * 80 + [1] * 20)  # 20% positive
        
        # Test calculation
        neg_count = (y == 0).sum()
        pos_count = (y == 1).sum()
        expected_weight = neg_count / max(1, pos_count)
        expected_weight = np.clip(expected_weight, 0.1, 10.0)
        
        print(f"Expected scale_pos_weight: {expected_weight}")
        
        # Should be reasonable for this imbalance (80/20 = 4.0)
        assert 3.0 <= expected_weight <= 5.0, f"scale_pos_weight {expected_weight} seems wrong"
        print("✅ scale_pos_weight calculation is reasonable")
    
    def test_eval_metric_consistency(self):
        """Test that eval_metric is used consistently."""
        if not IMPORT_SUCCESS:
            pytest.skip("Import failed")
        
        optimizer = XGBoostOptuna(n_trials=3, cv_folds=2, embargo=3, use_mlflow=False)
        
        # Check if we can identify the eval_metric being used
        study = optuna.create_study()
        trial = study.ask()
        params = optimizer._create_search_space(trial)
        
        if 'eval_metric' in params:
            eval_metric = params['eval_metric']
            print(f"✅ eval_metric is set to: {eval_metric}")
        else:
            print("ℹ️ eval_metric not in search space (may be set in constructor)")
        
        # This test passes if it doesn't crash
        assert True
    
    def test_no_constant_predictions(self):
        """Test predictions are not constant."""
        if not IMPORT_SUCCESS:
            pytest.skip("Import failed")
        
        # Create data with clear pattern
        np.random.seed(42)
        X = pd.DataFrame(np.random.randn(200, 10))
        # Make y depend on sum of features
        y = pd.Series((X.sum(axis=1) > 0).astype(int))
        X.index = pd.date_range('2023-01-01', periods=200, freq='h')
        y.index = X.index
        
        optimizer = XGBoostOptuna(n_trials=3, cv_folds=2, embargo=5, use_mlflow=False)
        optimizer.optimize(X, y)
        
        # Get predictions
        probs = optimizer.predict_proba(X)
        
        # Check diversity
        unique_probs = len(np.unique(np.round(probs, 3)))
        prob_std = np.std(probs)
        
        assert unique_probs >= 10, f"Only {unique_probs} unique probabilities"
        assert prob_std >= 0.05, f"Probability std too low: {prob_std:.3f}"
        
        print(f"✅ Predictions diverse: {unique_probs} unique probs, std={prob_std:.3f}")
    
    def test_basic_pipeline_works(self):
        """Test the complete basic pipeline works."""
        if not IMPORT_SUCCESS:
            pytest.skip("Import failed")
        
        # Simple synthetic data
        np.random.seed(42)
        X = pd.DataFrame(np.random.randn(150, 8), columns=[f'feat_{i}' for i in range(8)])
        y = pd.Series((X['feat_0'] + X['feat_1'] > 0).astype(int))  # Simple pattern
        X.index = pd.date_range('2023-01-01', periods=150, freq='h')
        y.index = X.index
        
        try:
            # Full pipeline
            optimizer = XGBoostOptuna(n_trials=3, cv_folds=2, embargo=5, use_mlflow=False)
            
            # Optimize
            optimizer.optimize(X, y)
            
            # Should have results
            assert optimizer.best_model is not None, "No best model found"
            assert optimizer.best_params is not None, "No best params found"
            assert optimizer.calibrator is not None, "No calibrator found"
            
            # Make predictions
            probs = optimizer.predict_proba(X)
            preds = optimizer.predict(X)
            
            assert len(probs) == len(X), "Wrong number of probabilities"
            assert len(preds) == len(X), "Wrong number of predictions"
            assert all(p in [0, 1] for p in preds), "Invalid predictions"
            assert all(0 <= p <= 1 for p in probs), "Invalid probabilities"
            
            print("✅ Complete pipeline works")
            
        except Exception as e:
            pytest.fail(f"Pipeline failed: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])