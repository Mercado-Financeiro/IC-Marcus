"""Unit tests for XGBoost base model."""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
import tempfile
from pathlib import Path
from sklearn.datasets import make_classification

from src.models.xgb.base import BaseXGBoost
from src.models.xgb.config import XGBoostConfig


class TestXGBoostConfig:
    """Test cases for XGBoostConfig."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = XGBoostConfig()
        
        assert config.n_estimators == 100
        assert config.max_depth == 6
        assert config.learning_rate == 0.3
        assert config.random_state == 42
        assert config.objective == 'binary:logistic'
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = XGBoostConfig(
            n_estimators=200,
            max_depth=8,
            learning_rate=0.1
        )
        
        assert config.n_estimators == 200
        assert config.max_depth == 8
        assert config.learning_rate == 0.1
    
    def test_config_to_dict(self):
        """Test configuration to dictionary conversion."""
        config = XGBoostConfig()
        config_dict = config.to_dict()
        
        assert isinstance(config_dict, dict)
        assert 'n_estimators' in config_dict
        assert 'max_depth' in config_dict
        assert config_dict['n_estimators'] == 100


class TestBaseXGBoost:
    """Test cases for BaseXGBoost model."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample classification data."""
        X, y = make_classification(
            n_samples=100,
            n_features=10,
            n_informative=5,
            n_redundant=3,
            n_classes=2,
            random_state=42
        )
        return X, y
    
    @pytest.fixture
    def base_model(self):
        """Create base XGBoost model."""
        config = XGBoostConfig(n_estimators=10, max_depth=3)
        return BaseXGBoost(config)
    
    def test_initialization(self, base_model):
        """Test model initialization."""
        assert base_model.model is not None
        assert base_model.calibrator is None
        assert base_model.is_fitted is False
        assert base_model.config.n_estimators == 10
    
    def test_fit(self, base_model, sample_data):
        """Test model fitting."""
        X, y = sample_data
        
        base_model.fit(X, y)
        
        assert base_model.is_fitted is True
        assert base_model.feature_names is not None
        assert len(base_model.feature_names) == X.shape[1]
    
    def test_fit_with_dataframe(self, base_model):
        """Test fitting with pandas DataFrame."""
        X = pd.DataFrame(
            np.random.randn(50, 5),
            columns=['f1', 'f2', 'f3', 'f4', 'f5']
        )
        y = pd.Series(np.random.randint(0, 2, 50))
        
        base_model.fit(X, y)
        
        assert base_model.feature_names == ['f1', 'f2', 'f3', 'f4', 'f5']
        assert base_model.is_fitted is True
    
    def test_fit_with_eval_set(self, base_model, sample_data):
        """Test fitting with validation set."""
        X, y = sample_data
        X_train, X_val = X[:80], X[80:]
        y_train, y_val = y[:80], y[80:]
        
        base_model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=5
        )
        
        assert base_model.is_fitted is True
    
    def test_predict(self, base_model, sample_data):
        """Test prediction."""
        X, y = sample_data
        
        base_model.fit(X, y)
        predictions = base_model.predict(X)
        
        assert len(predictions) == len(y)
        assert np.all((predictions == 0) | (predictions == 1))
    
    def test_predict_proba(self, base_model, sample_data):
        """Test probability prediction."""
        X, y = sample_data
        
        base_model.fit(X, y)
        probabilities = base_model.predict_proba(X)
        
        assert probabilities.shape == (len(y), 2)
        assert np.all((probabilities >= 0) & (probabilities <= 1))
        assert np.allclose(probabilities.sum(axis=1), 1.0)
    
    def test_predict_with_threshold(self, base_model, sample_data):
        """Test prediction with custom threshold."""
        X, y = sample_data
        
        base_model.fit(X, y)
        
        predictions_low = base_model.predict(X, threshold=0.3)
        predictions_high = base_model.predict(X, threshold=0.7)
        
        # Lower threshold should produce more positive predictions
        assert predictions_low.sum() >= predictions_high.sum()
    
    def test_calibration(self, base_model, sample_data):
        """Test model calibration."""
        X, y = sample_data
        X_train, X_cal = X[:70], X[70:]
        y_train, y_cal = y[:70], y[70:]
        
        # Fit base model
        base_model.fit(X_train, y_train)
        
        # Calibrate
        base_model.calibrate(X_cal, y_cal, method='isotonic', cv=3)
        
        assert base_model.calibrator is not None
        
        # Predictions should now use calibrator
        probabilities = base_model.predict_proba(X_cal)
        assert probabilities.shape == (len(y_cal), 2)
    
    def test_calibration_before_fit_raises(self, base_model, sample_data):
        """Test that calibration before fitting raises error."""
        X, y = sample_data
        
        with pytest.raises(ValueError, match="Model must be fitted"):
            base_model.calibrate(X, y)
    
    def test_predict_before_fit_raises(self, base_model, sample_data):
        """Test that prediction before fitting raises error."""
        X, _ = sample_data
        
        with pytest.raises(ValueError, match="Model must be fitted"):
            base_model.predict(X)
    
    def test_feature_importance(self, base_model, sample_data):
        """Test feature importance extraction."""
        X, y = sample_data
        
        base_model.fit(X, y)
        importance = base_model.get_feature_importance(importance_type='gain')
        
        assert isinstance(importance, pd.DataFrame)
        assert 'feature' in importance.columns
        assert 'importance' in importance.columns
        assert len(importance) > 0
    
    def test_feature_importance_before_fit_raises(self, base_model):
        """Test that feature importance before fitting raises error."""
        with pytest.raises(ValueError, match="Model must be fitted"):
            base_model.get_feature_importance()
    
    def test_save_load(self, base_model, sample_data):
        """Test model saving and loading."""
        X, y = sample_data
        
        # Fit model
        base_model.fit(X, y)
        predictions_before = base_model.predict_proba(X)
        
        # Save model
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as tmp:
            base_model.save(tmp.name)
            
            # Create new model and load
            new_model = BaseXGBoost()
            new_model.load(tmp.name)
            
            # Check predictions are the same
            predictions_after = new_model.predict_proba(X)
            np.testing.assert_array_almost_equal(
                predictions_before,
                predictions_after,
                decimal=5
            )
            
            # Clean up
            Path(tmp.name).unlink()
    
    def test_save_with_calibration(self, base_model, sample_data):
        """Test saving model with calibration."""
        X, y = sample_data
        
        # Fit and calibrate
        base_model.fit(X[:70], y[:70])
        base_model.calibrate(X[70:], y[70:])
        
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as tmp:
            base_model.save(tmp.name)
            
            # Load and check calibrator is preserved
            new_model = BaseXGBoost()
            new_model.load(tmp.name)
            
            assert new_model.calibrator is not None
            
            # Clean up
            Path(tmp.name).unlink()
    
    def test_sklearn_compatibility(self, base_model, sample_data):
        """Test sklearn compatibility methods."""
        X, y = sample_data
        
        # Get params
        params = base_model.get_params()
        assert 'config' in params
        
        # Set params
        new_config = XGBoostConfig(n_estimators=20)
        base_model.set_params(config=new_config)
        assert base_model.config.n_estimators == 20
        
        # Score
        base_model.fit(X, y)
        score = base_model.score(X, y)
        assert 0 <= score <= 1
    
    def test_fit_with_sample_weight(self, base_model, sample_data):
        """Test fitting with sample weights."""
        X, y = sample_data
        weights = np.random.rand(len(y))
        
        base_model.fit(X, y, sample_weight=weights)
        
        assert base_model.is_fitted is True
    
    @pytest.mark.parametrize("n_features", [5, 10, 20])
    def test_different_feature_sizes(self, n_features):
        """Test with different number of features."""
        X, y = make_classification(
            n_samples=50,
            n_features=n_features,
            n_classes=2,
            random_state=42
        )
        
        model = BaseXGBoost(XGBoostConfig(n_estimators=5))
        model.fit(X, y)
        
        predictions = model.predict(X)
        assert len(predictions) == len(y)
    
    def test_empty_data_handling(self, base_model):
        """Test handling of empty data."""
        X = np.array([]).reshape(0, 10)
        y = np.array([])
        
        # Should handle gracefully or raise appropriate error
        with pytest.raises(Exception):  # XGBoost will raise an error
            base_model.fit(X, y)
    
    def test_single_sample_prediction(self, base_model, sample_data):
        """Test prediction with single sample."""
        X, y = sample_data
        
        base_model.fit(X, y)
        
        # Single sample
        single_sample = X[0:1]
        prediction = base_model.predict(single_sample)
        proba = base_model.predict_proba(single_sample)
        
        assert len(prediction) == 1
        assert proba.shape == (1, 2)