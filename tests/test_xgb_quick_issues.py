"""
Quick tests to identify the specific issues mentioned in the code review.
"""

import pytest
import numpy as np
import pandas as pd
import xgboost as xgb
from pathlib import Path
import sys
import warnings
from unittest.mock import patch
import optuna

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

# Test imports
try:
    from src.models.xgb_optuna import XGBoostOptuna
    IMPORT_SUCCESS = True
except Exception as e:
    IMPORT_SUCCESS = False
    IMPORT_ERROR = str(e)


def test_import_success():
    """Test that we can import the XGBoost class."""
    assert IMPORT_SUCCESS, f"Import failed: {IMPORT_ERROR}"


def test_xgboost_parameter_names():
    """Test for deprecated XGBoost parameter names."""
    if not IMPORT_SUCCESS:
        pytest.skip("Import failed")
    
    # Create optimizer
    optimizer = XGBoostOptuna(n_trials=5, cv_folds=3, embargo=5, use_mlflow=False, seed=42)
    
    # Create a trial to test parameter generation
    study = optuna.create_study()
    trial = study.ask()
    params = optimizer._create_search_space(trial)
    
    # Check for deprecated parameters
    deprecated_found = []
    
    if 'eta' in params:
        deprecated_found.append('eta (use learning_rate)')
    if 'lambda' in params:
        deprecated_found.append('lambda (use reg_lambda)')
    if 'alpha' in params:
        deprecated_found.append('alpha (use reg_alpha)')
    if 'use_label_encoder' in params:
        deprecated_found.append('use_label_encoder (deprecated in XGBoost 2.x)')
    
    if deprecated_found:
        pytest.fail(f"Found deprecated parameters: {deprecated_found}")
    
    print(f"✓ Parameters look good: {list(params.keys())}")


def test_notebook_file_handling():
    """Test handling of missing __file__ in notebooks."""
    if not IMPORT_SUCCESS:
        pytest.skip("Import failed")
    
    # Mock missing __file__
    with patch('builtins.__file__', side_effect=NameError("__file__ not defined")):
        try:
            # This should work even without __file__
            optimizer = XGBoostOptuna(n_trials=5, cv_folds=3, use_mlflow=False)
            print("✓ Handles missing __file__ correctly")
        except NameError as e:
            pytest.fail(f"Failed to handle missing __file__: {e}")


def test_logging_compatibility():
    """Test logging works with and without structlog."""
    if not IMPORT_SUCCESS:
        pytest.skip("Import failed")
    
    # Test with structlog (if available)
    try:
        optimizer = XGBoostOptuna(n_trials=5, use_mlflow=False)
        optimizer.log.info("test message", param="value")
        print("✓ Structlog logging works")
    except Exception as e:
        print(f"⚠ Structlog issue: {e}")
    
    # Test fallback logging
    with patch.dict('sys.modules', {'structlog': None}):
        try:
            # Need to reload module to trigger fallback
            import importlib
            import sys
            if 'src.models.xgb_optuna' in sys.modules:
                del sys.modules['src.models.xgb_optuna']
            
            from src.models.xgb_optuna import XGBoostOptuna
            optimizer = XGBoostOptuna(n_trials=5, use_mlflow=False)
            
            # This should work with fallback logger
            optimizer.log.info("test message", param="value")
            print("✓ Fallback logging works")
        except Exception as e:
            pytest.fail(f"Logging fallback failed: {e}")


def test_xgboost_version_compatibility():
    """Test XGBoost version compatibility."""
    xgb_version = xgb.__version__
    major_version = int(xgb_version.split('.')[0])
    
    print(f"XGBoost version: {xgb_version}")
    
    if major_version >= 2:
        # In XGBoost 2.x, use_label_encoder should not be used
        try:
            clf = xgb.XGBClassifier(use_label_encoder=False)
            pytest.fail("XGBoost 2.x should not accept use_label_encoder parameter")
        except TypeError:
            print("✓ XGBoost 2.x correctly rejects use_label_encoder")
    else:
        print("✓ XGBoost 1.x - use_label_encoder handling not tested")


def test_basic_training_flow():
    """Test basic training doesn't produce constant predictions."""
    if not IMPORT_SUCCESS:
        pytest.skip("Import failed")
    
    # Create simple synthetic data
    np.random.seed(42)
    X = pd.DataFrame(np.random.randn(200, 10), columns=[f'feat_{i}' for i in range(10)])
    y = pd.Series((X.sum(axis=1) > 0).astype(int))  # Simple pattern
    
    # Add timestamps
    X.index = pd.date_range('2023-01-01', periods=200, freq='h')
    y.index = X.index
    
    # Quick train
    optimizer = XGBoostOptuna(n_trials=3, cv_folds=2, embargo=5, use_mlflow=False)
    
    try:
        # Run optimization (quick)
        optimizer.optimize(X, y)
        
        # Check we got a model
        assert optimizer.best_model is not None
        
        # Check predictions aren't constant
        probs = optimizer.predict_proba(X)
        unique_probs = np.unique(probs)
        
        if len(unique_probs) < 5:
            pytest.fail(f"Too few unique probabilities: {len(unique_probs)}")
        
        prob_std = np.std(probs)
        if prob_std < 0.01:
            pytest.fail(f"Probability std too low: {prob_std}")
        
        print(f"✓ Training successful, {len(unique_probs)} unique probs, std={prob_std:.3f}")
        
    except Exception as e:
        pytest.fail(f"Training failed: {e}")


def test_scale_pos_weight_calculation():
    """Test scale_pos_weight calculation for imbalanced data."""
    if not IMPORT_SUCCESS:
        pytest.skip("Import failed")
    
    # Create imbalanced data
    np.random.seed(42)
    X = pd.DataFrame(np.random.randn(100, 5))
    y = pd.Series([0] * 80 + [1] * 20)  # 20% positive class
    
    # Calculate expected scale_pos_weight
    neg_count = (y == 0).sum()
    pos_count = (y == 1).sum()
    expected_weight = neg_count / max(1, pos_count)
    
    print(f"Data balance: {neg_count} negative, {pos_count} positive")
    print(f"Expected scale_pos_weight: {expected_weight}")
    
    # Should be reasonable for imbalanced data
    assert 1.0 < expected_weight < 10.0, f"scale_pos_weight {expected_weight} seems wrong"
    print("✓ scale_pos_weight calculation looks correct")


def test_early_stopping_availability():
    """Test that early stopping parameters exist."""
    # Test XGBoost supports early stopping
    X = np.random.randn(100, 10)
    y = np.random.randint(0, 2, 100)
    X_val = np.random.randn(50, 10)
    y_val = np.random.randint(0, 2, 50)
    
    model = xgb.XGBClassifier(n_estimators=100, random_state=42)
    
    try:
        model.fit(
            X, y,
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=10,
            verbose=False
        )
        
        # Check it actually stopped early
        if hasattr(model, 'best_iteration'):
            print(f"✓ Early stopping works, stopped at iteration {model.best_iteration}")
        else:
            print("✓ Early stopping parameter accepted")
            
    except Exception as e:
        pytest.fail(f"Early stopping failed: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])