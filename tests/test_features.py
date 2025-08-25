"""
Test feature consistency and validation.
"""
import pytest
import pandas as pd
import numpy as np


class TestFeatureConsistency:
    """Test feature handling consistency."""
    
    def test_feature_names_consistency(self):
        """Test that feature names are preserved correctly."""
        from src.models.xgb import XGBoostModel
        
        # Create test data
        X = pd.DataFrame({
            'feature_a': np.random.randn(100),
            'feature_b': np.random.randn(100),
            'feature_c': np.random.randn(100)
        })
        y = pd.Series(np.random.randint(0, 2, 100))
        
        # Train model
        config = {'seed': 42, 'n_estimators': 10}
        model = XGBoostModel(config)
        model.fit(X, y)
        
        # Check feature names are stored
        assert model.feature_names == ['feature_a', 'feature_b', 'feature_c']
        
        # Test prediction with same order
        pred1 = model.predict_proba(X)
        
        # Test prediction with different order
        X_reordered = X[['feature_c', 'feature_a', 'feature_b']]
        pred2 = model.predict_proba(X_reordered)
        
        # Should be identical (model reorders internally)
        np.testing.assert_array_equal(pred1, pred2)
    
    def test_missing_features_error(self):
        """Test that missing features raise appropriate errors."""
        from src.models.xgb import XGBoostModel
        
        X_train = pd.DataFrame({
            'feature_a': np.random.randn(100),
            'feature_b': np.random.randn(100)
        })
        y_train = pd.Series(np.random.randint(0, 2, 100))
        
        # Train model
        config = {'seed': 42, 'n_estimators': 10}
        model = XGBoostModel(config)
        model.fit(X_train, y_train)
        
        # Test prediction with missing feature
        X_test = pd.DataFrame({
            'feature_a': np.random.randn(50)
            # Missing feature_b
        })
        
        with pytest.raises(KeyError):
            model.predict_proba(X_test)
    
    def test_extra_features_ignored(self):
        """Test that extra features are handled correctly."""
        from src.models.xgb import XGBoostModel
        
        X_train = pd.DataFrame({
            'feature_a': np.random.randn(100),
            'feature_b': np.random.randn(100)
        })
        y_train = pd.Series(np.random.randint(0, 2, 100))
        
        # Train model
        config = {'seed': 42, 'n_estimators': 10}
        model = XGBoostModel(config)
        model.fit(X_train, y_train)
        
        # Test prediction with extra feature
        X_test = pd.DataFrame({
            'feature_a': np.random.randn(50),
            'feature_b': np.random.randn(50),
            'extra_feature': np.random.randn(50)  # Extra feature
        })
        
        # Should work fine (extra features ignored)
        pred = model.predict_proba(X_test)
        assert pred.shape == (50, 2)