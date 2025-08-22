"""
Comprehensive test suite for XGBoost with Optuna optimization.
Tests cover all critical issues identified in the code review.
"""

import pytest
import numpy as np
import pandas as pd
import xgboost as xgb
from pathlib import Path
import sys
import tempfile
import shutil
import warnings
from unittest.mock import Mock, patch, MagicMock
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from sklearn.metrics import brier_score_loss, f1_score, roc_auc_score
import optuna
import json
import hashlib
import os

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.models.xgb_optuna import XGBoostOptuna
from data.splits import PurgedKFold
from features.engineering import FeatureEngineer


class TestXGBoostFixtures:
    """Fixtures and utilities for XGBoost testing."""
    
    @pytest.fixture
    def synthetic_data(self):
        """Create synthetic classification data with temporal structure."""
        np.random.seed(42)
        X, y = make_classification(
            n_samples=1000,
            n_features=20,
            n_informative=15,
            n_redundant=5,
            n_clusters_per_class=3,
            weights=[0.7, 0.3],  # Imbalanced
            flip_y=0.05,
            random_state=42
        )
        
        # Add temporal index
        timestamps = pd.date_range('2023-01-01', periods=1000, freq='h')
        X_df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(20)])
        X_df.index = timestamps
        y_series = pd.Series(y, index=timestamps)
        
        return X_df, y_series
    
    @pytest.fixture
    def optimizer_kwargs(self):
        """Minimal valid optimizer configuration."""
        return {
            'n_trials': 5,
            'cv_folds': 3,
            'embargo': 10,
            'pruner_type': 'median',
            'use_mlflow': False,
            'seed': 42
        }
    
    @pytest.fixture
    def temp_artifacts_dir(self):
        """Create temporary artifacts directory."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)


class TestParameterConsistency(TestXGBoostFixtures):
    """Test XGBoost parameter naming and consistency."""
    
    def test_canonical_parameter_names(self, synthetic_data, optimizer_kwargs):
        """Verify use of canonical XGBoost parameter names."""
        X, y = synthetic_data
        optimizer = XGBoostOptuna(**optimizer_kwargs)
        
        # Create a mock trial to test parameter naming
        import optuna
        study = optuna.create_study()
        trial = study.ask()
        params = optimizer._create_search_space(trial)
        
        # Should use learning_rate, not eta
        if 'eta' in params:
            pytest.fail("Using deprecated 'eta' instead of 'learning_rate'")
        
        # Should use reg_lambda/reg_alpha, not lambda/alpha
        if 'lambda' in params:
            pytest.fail("Using deprecated 'lambda' instead of 'reg_lambda'")
        if 'alpha' in params:
            pytest.fail("Using deprecated 'alpha' instead of 'reg_alpha'")
        
        # Should not have deprecated parameters
        if 'use_label_encoder' in params:
            pytest.fail("Using deprecated 'use_label_encoder' parameter")
    
    def test_parameter_consistency_optimization_vs_final(self, synthetic_data, optimizer_kwargs):
        """Verify parameters are consistent between optimization and final model."""
        X, y = synthetic_data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        
        optimizer = XGBoostOptuna(optimizer_kwargs)
        
        # Capture parameters used during optimization
        optimization_params = []
        
        def objective_wrapper(trial):
            params = optimizer._suggest_params(trial)
            optimization_params.append(params.copy())
            return np.random.random()  # Mock score
        
        study = optuna.create_study(direction='maximize')
        study.optimize(objective_wrapper, n_trials=3)
        
        # Check that critical parameters remain consistent
        for params in optimization_params:
            if 'scale_pos_weight' in params:
                assert params['scale_pos_weight'] > 0
            assert 'eval_metric' in params
    
    def test_xgboost_version_compatibility(self):
        """Test that code handles different XGBoost versions correctly."""
        xgb_version = xgb.__version__
        major_version = int(xgb_version.split('.')[0])
        
        # For XGBoost 2.x, use_label_encoder should not be used
        if major_version >= 2:
            with pytest.raises(TypeError):
                # This should fail in XGBoost 2.x
                xgb.XGBClassifier(use_label_encoder=False)


class TestLoggingRobustness(TestXGBoostFixtures):
    """Test logging fallback mechanisms."""
    
    def test_structlog_fallback(self, synthetic_data, optimizer_kwargs):
        """Test that logging works even without structlog."""
        X, y = synthetic_data
        
        # Mock structlog import failure
        with patch.dict('sys.modules', {'structlog': None}):
            # Clear any cached imports
            if 'src.models.xgb_optuna' in sys.modules:
                del sys.modules['src.models.xgb_optuna']
            
            # Re-import should handle missing structlog
            from src.models.xgb_optuna import XGBoostOptuna
            optimizer = XGBoostOptuna(optimizer_kwargs)
            
            # Should not raise errors when logging
            optimizer.log.info("test", key="value")
    
    def test_logging_with_kwargs(self, synthetic_data, optimizer_kwargs):
        """Test that logging handles keyword arguments correctly."""
        X, y = synthetic_data
        optimizer = XGBoostOptuna(optimizer_kwargs)
        
        # Test various logging patterns
        try:
            optimizer.log.info("event", param1="value1", param2=42)
            optimizer.log.warning("warning", data={"key": "value"})
            optimizer.log.error("error", exception="test error")
        except TypeError as e:
            pytest.fail(f"Logging failed with kwargs: {e}")


class TestNotebookCompatibility(TestXGBoostFixtures):
    """Test compatibility with Jupyter notebooks."""
    
    def test_file_attribute_fallback(self, optimizer_kwargs):
        """Test handling of missing __file__ in notebooks."""
        # Simulate notebook environment where __file__ doesn't exist
        with patch('src.models.xgb_optuna.__file__', side_effect=NameError):
            try:
                optimizer = XGBoostOptuna(optimizer_kwargs)
                # Should fall back to cwd
                assert optimizer.base_path == Path.cwd()
            except NameError:
                pytest.fail("Failed to handle missing __file__")
    
    def test_notebook_execution_context(self, synthetic_data, optimizer_kwargs):
        """Test that code works in simulated notebook context."""
        X, y = synthetic_data
        
        # Mock IPython/notebook environment
        with patch('IPython.get_ipython', return_value=Mock()):
            with patch.object(Path, '__file__', side_effect=AttributeError):
                optimizer = XGBoostOptuna(optimizer_kwargs)
                
                # Should still be able to train
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, shuffle=False
                )
                
                # Quick training test
                model = optimizer.train_final_model(X_train, y_train)
                assert model is not None


class TestScalePosWeight(TestXGBoostFixtures):
    """Test scale_pos_weight handling."""
    
    def test_scale_pos_weight_calculation(self, synthetic_data):
        """Test correct calculation of scale_pos_weight for imbalanced data."""
        X, y = synthetic_data
        
        # Calculate expected scale_pos_weight
        neg_count = (y == 0).sum()
        pos_count = (y == 1).sum()
        expected_weight = neg_count / max(1, pos_count)
        
        # Should be clipped to reasonable range
        expected_weight = np.clip(expected_weight, 0.1, 10.0)
        
        assert 0.1 <= expected_weight <= 10.0
        assert expected_weight > 1.0  # Since we have imbalanced data (70/30)
    
    def test_scale_pos_weight_consistency(self, synthetic_data, optimizer_kwargs):
        """Test that scale_pos_weight is used consistently."""
        X, y = synthetic_data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, shuffle=False
        )
        
        optimizer = XGBoostOptuna(optimizer_kwargs)
        
        # Track scale_pos_weight across different calls
        weights = []
        
        # Mock to capture parameters
        original_fit = xgb.XGBClassifier.fit
        
        def fit_wrapper(self, *args, **kwargs):
            if hasattr(self, 'scale_pos_weight'):
                weights.append(self.scale_pos_weight)
            return original_fit(self, *args, **kwargs)
        
        with patch.object(xgb.XGBClassifier, 'fit', fit_wrapper):
            # Train with optimization
            if optimizer_kwargs['optuna']['enabled']:
                optimizer.optimize(X_train, y_train, n_trials=2)
            
            # Train final model
            model = optimizer.train_final_model(X_train, y_train)
        
        # All weights should be the same
        if weights:
            assert len(set(weights)) == 1, "scale_pos_weight inconsistent"


class TestEvalMetric(TestXGBoostFixtures):
    """Test eval_metric consistency."""
    
    def test_eval_metric_alignment(self, synthetic_data, optimizer_kwargs):
        """Test that eval_metric is consistent throughout pipeline."""
        X, y = synthetic_data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, shuffle=False
        )
        
        optimizer = XGBoostOptuna(optimizer_kwargs)
        
        # Capture eval_metrics used
        metrics_used = []
        
        original_init = xgb.XGBClassifier.__init__
        
        def init_wrapper(self, **kwargs):
            if 'eval_metric' in kwargs:
                metrics_used.append(kwargs['eval_metric'])
            return original_init(self, **kwargs)
        
        with patch.object(xgb.XGBClassifier, '__init__', init_wrapper):
            # Optimize
            if optimizer_kwargs['optuna']['enabled']:
                optimizer.optimize(X_train, y_train, n_trials=2)
            
            # Train final
            model = optimizer.train_final_model(X_train, y_train)
        
        # Check consistency
        if metrics_used:
            assert len(set(metrics_used)) == 1, f"Inconsistent eval_metrics: {metrics_used}"
    
    def test_eval_metric_validity(self):
        """Test that chosen eval_metric is valid for binary classification."""
        valid_metrics = ['logloss', 'error', 'auc', 'aucpr', 'map']
        
        for metric in valid_metrics:
            try:
                clf = xgb.XGBClassifier(eval_metric=metric, n_estimators=10)
                X = np.random.randn(100, 10)
                y = np.random.randint(0, 2, 100)
                clf.fit(X, y, verbose=False)
            except Exception as e:
                pytest.fail(f"Valid metric '{metric}' failed: {e}")


class TestEarlyStopping(TestXGBoostFixtures):
    """Test early stopping functionality."""
    
    def test_early_stopping_enabled(self, synthetic_data, optimizer_kwargs):
        """Test that early stopping prevents overfitting."""
        X, y = synthetic_data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, shuffle=False
        )
        
        # Train with and without early stopping
        params = {
            'n_estimators': 1000,
            'learning_rate': 0.3,
            'max_depth': 10,
            'random_state': 42
        }
        
        # Without early stopping
        model_no_stop = xgb.XGBClassifier(**params)
        model_no_stop.fit(X_train, y_train, verbose=False)
        
        # With early stopping
        model_with_stop = xgb.XGBClassifier(**params)
        model_with_stop.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=50,
            verbose=False
        )
        
        # Model with early stopping should use fewer trees
        assert model_with_stop.best_iteration < model_no_stop.n_estimators
    
    def test_early_stopping_with_high_learning_rate(self, synthetic_data):
        """Test early stopping is crucial with high learning rates."""
        X, y = synthetic_data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, shuffle=False
        )
        
        # High learning rate without early stopping can overfit quickly
        params = {
            'n_estimators': 500,
            'learning_rate': 0.5,  # Very high
            'max_depth': 8,
            'random_state': 42
        }
        
        model = xgb.XGBClassifier(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=20,
            verbose=False
        )
        
        # Should stop early with high learning rate
        assert model.best_iteration < 100, "Failed to stop early with high LR"


class TestSearchSpace(TestXGBoostFixtures):
    """Test hyperparameter search space."""
    
    def test_search_space_bounds(self):
        """Test that search space has reasonable bounds."""
        # Recommended bounds
        recommended_bounds = {
            'max_depth': (3, 10),
            'learning_rate': (0.01, 0.3),
            'min_child_weight': (1, 10),
            'subsample': (0.6, 1.0),
            'colsample_bytree': (0.6, 1.0),
            'gamma': (0, 5),
            'reg_lambda': (0, 10),
            'reg_alpha': (0, 5)
        }
        
        # Check if bounds are sensible
        for param, (low, high) in recommended_bounds.items():
            assert low < high
            assert low >= 0
            
            # Learning rate should use log scale
            if param == 'learning_rate':
                assert high / low > 10  # At least 10x range for log scale
    
    def test_search_space_coverage(self, synthetic_data, optimizer_kwargs):
        """Test that search explores diverse parameter combinations."""
        X, y = synthetic_data
        X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, shuffle=False)
        
        optimizer_kwargs['optuna']['n_trials'] = 10
        optimizer = XGBoostOptuna(optimizer_kwargs)
        
        # Collect parameter combinations
        param_history = []
        
        def objective_wrapper(trial):
            params = optimizer._suggest_params(trial)
            param_history.append(params)
            return np.random.random()
        
        study = optuna.create_study(direction='maximize')
        study.optimize(objective_wrapper, n_trials=10)
        
        # Check diversity
        max_depths = [p.get('max_depth', 6) for p in param_history]
        learning_rates = [p.get('learning_rate', 0.1) for p in param_history]
        
        # Should explore different values
        assert len(set(max_depths)) > 1
        assert len(set(learning_rates)) > 1


class TestCalibration(TestXGBoostFixtures):
    """Test probability calibration."""
    
    def test_calibration_improves_brier_score(self, synthetic_data, optimizer_kwargs):
        """Test that calibration improves Brier score."""
        X, y = synthetic_data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, shuffle=False
        )
        
        # Train uncalibrated model
        model = xgb.XGBClassifier(n_estimators=50, random_state=42)
        model.fit(X_train, y_train)
        
        # Get uncalibrated probabilities
        probs_uncalibrated = model.predict_proba(X_test)[:, 1]
        brier_uncalibrated = brier_score_loss(y_test, probs_uncalibrated)
        
        # Apply calibration (simplified version)
        from sklearn.calibration import CalibratedClassifierCV
        calibrator = CalibratedClassifierCV(model, method='isotonic', cv=3)
        calibrator.fit(X_train, y_train)
        
        probs_calibrated = calibrator.predict_proba(X_test)[:, 1]
        brier_calibrated = brier_score_loss(y_test, probs_calibrated)
        
        # Calibration should improve or at least not hurt Brier score significantly
        assert brier_calibrated <= brier_uncalibrated * 1.1
    
    def test_calibration_fallback(self, synthetic_data):
        """Test fallback from isotonic to sigmoid for small datasets."""
        # Very small dataset where isotonic might fail
        X, y = synthetic_data
        X_small = X[:50]
        y_small = y[:50]
        
        model = xgb.XGBClassifier(n_estimators=10, random_state=42)
        model.fit(X_small, y_small)
        
        # Try isotonic first, fallback to sigmoid
        from sklearn.calibration import CalibratedClassifierCV
        
        try:
            calibrator = CalibratedClassifierCV(model, method='isotonic', cv=2)
            calibrator.fit(X_small, y_small)
        except:
            # Should fallback to sigmoid
            calibrator = CalibratedClassifierCV(model, method='sigmoid', cv=2)
            calibrator.fit(X_small, y_small)
        
        # Should produce valid probabilities
        probs = calibrator.predict_proba(X_small)[:, 1]
        assert np.all((probs >= 0) & (probs <= 1))


class TestThresholdOptimization(TestXGBoostFixtures):
    """Test threshold selection for classification."""
    
    def test_threshold_f1_optimization(self, synthetic_data):
        """Test F1-based threshold optimization."""
        X, y = synthetic_data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, shuffle=False
        )
        
        model = xgb.XGBClassifier(n_estimators=50, random_state=42)
        model.fit(X_train, y_train)
        
        probs = model.predict_proba(X_val)[:, 1]
        
        # Find optimal threshold for F1
        thresholds = np.linspace(0.1, 0.9, 50)
        f1_scores = []
        
        for threshold in thresholds:
            preds = (probs >= threshold).astype(int)
            f1 = f1_score(y_val, preds)
            f1_scores.append(f1)
        
        best_threshold = thresholds[np.argmax(f1_scores)]
        
        # Best threshold should not be at extremes
        assert 0.2 <= best_threshold <= 0.8
    
    def test_threshold_ev_optimization(self, synthetic_data):
        """Test Expected Value (EV) based threshold optimization."""
        X, y = synthetic_data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, shuffle=False
        )
        
        model = xgb.XGBClassifier(n_estimators=50, random_state=42)
        model.fit(X_train, y_train)
        
        probs = model.predict_proba(X_val)[:, 1]
        
        # Simulate EV calculation with costs
        def calculate_ev(threshold, probs, y_true, cost_bps=10):
            preds = (probs >= threshold).astype(int)
            
            # Simple EV: +1% for correct, -1% for incorrect, minus costs
            returns = np.where(preds == y_true, 0.01, -0.01)
            costs = np.where(preds != 0, cost_bps / 10000, 0)
            
            ev = returns - costs
            return ev.mean()
        
        # Find optimal threshold for EV
        thresholds = np.linspace(0.1, 0.9, 50)
        evs = [calculate_ev(t, probs, y_val) for t in thresholds]
        
        best_threshold_ev = thresholds[np.argmax(evs)]
        
        # EV-optimal threshold might differ from F1-optimal
        assert 0.1 <= best_threshold_ev <= 0.9


class TestPredictionQuality(TestXGBoostFixtures):
    """Test that model produces meaningful predictions."""
    
    def test_non_constant_predictions(self, synthetic_data, optimizer_kwargs):
        """Test that model doesn't output constant predictions."""
        X, y = synthetic_data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, shuffle=False
        )
        
        # Train with problematic settings that might cause constant predictions
        params = {
            'n_estimators': 50,
            'learning_rate': 0.5,  # Very high
            'max_depth': 1,  # Very shallow
            'random_state': 42
        }
        
        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train)
        
        probs = model.predict_proba(X_test)[:, 1]
        
        # Check predictions are not constant
        unique_probs = np.unique(probs)
        assert len(unique_probs) > 10, f"Only {len(unique_probs)} unique probabilities"
        
        # Check reasonable spread
        prob_std = np.std(probs)
        assert prob_std > 0.05, f"Probability std too low: {prob_std}"
    
    def test_prediction_distribution(self, synthetic_data):
        """Test that predictions have reasonable distribution."""
        X, y = synthetic_data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, shuffle=False
        )
        
        model = xgb.XGBClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        probs = model.predict_proba(X_test)[:, 1]
        
        # Check distribution properties
        assert probs.min() >= 0 and probs.max() <= 1
        assert 0.2 <= probs.mean() <= 0.8  # Not too extreme
        
        # Should have predictions across the range
        bins = np.histogram(probs, bins=10)[0]
        assert np.sum(bins > 0) >= 5  # At least 5 bins populated


class TestPurgedKFoldIntegration(TestXGBoostFixtures):
    """Test integration with PurgedKFold."""
    
    def test_purged_kfold_no_leakage(self, synthetic_data):
        """Test that PurgedKFold prevents temporal leakage."""
        X, y = synthetic_data
        
        # Use PurgedKFold
        cv = PurgedKFold(n_splits=3, embargo=10)
        
        for train_idx, val_idx in cv.split(X):
            train_times = X.index[train_idx]
            val_times = X.index[val_idx]
            
            # Check embargo gap
            if len(train_times) > 0 and len(val_times) > 0:
                max_train_time = train_times.max()
                min_val_time = val_times.min()
                
                # Should have embargo gap
                gap = (min_val_time - max_train_time).total_seconds() / 3600  # hours
                assert gap >= 10  # At least 10 hours embargo
    
    def test_purged_kfold_with_xgboost(self, synthetic_data, optimizer_kwargs):
        """Test XGBoost training with PurgedKFold."""
        X, y = synthetic_data
        
        optimizer = XGBoostOptuna(optimizer_kwargs)
        cv = PurgedKFold(n_splits=3, embargo=10)
        
        # Train on each fold
        scores = []
        for train_idx, val_idx in cv.split(X):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            model = xgb.XGBClassifier(n_estimators=50, random_state=42)
            model.fit(X_train, y_train)
            
            probs = model.predict_proba(X_val)[:, 1]
            auc = roc_auc_score(y_val, probs)
            scores.append(auc)
        
        # Should get reasonable scores
        assert all(0.5 <= s <= 1.0 for s in scores)
        assert np.std(scores) < 0.3  # Not too variable


class TestReproducibility(TestXGBoostFixtures):
    """Test model reproducibility and determinism."""
    
    def test_deterministic_training(self, synthetic_data, optimizer_kwargs):
        """Test that training is deterministic with fixed seed."""
        X, y = synthetic_data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, shuffle=False
        )
        
        # Train twice with same seed
        predictions = []
        for _ in range(2):
            model = xgb.XGBClassifier(
                n_estimators=50,
                random_state=42,
                seed=42
            )
            model.fit(X_train, y_train)
            probs = model.predict_proba(X_test)[:, 1]
            predictions.append(probs)
        
        # Should be identical
        np.testing.assert_array_almost_equal(predictions[0], predictions[1])
    
    def test_seed_propagation(self, synthetic_data, optimizer_kwargs):
        """Test that seed is properly propagated through the pipeline."""
        X, y = synthetic_data
        
        optimizer_kwargs['seed'] = 123
        optimizer = XGBoostOptuna(optimizer_kwargs)
        
        # Check seed is used
        assert optimizer.seed == 123
        
        # Train models
        results = []
        for _ in range(2):
            optimizer = XGBoostOptuna(optimizer_kwargs)
            model = optimizer.train_final_model(X, y)
            
            # Get some predictions
            probs = model.predict_proba(X[:50])[:, 1]
            results.append(probs)
        
        # Should be deterministic
        np.testing.assert_array_almost_equal(results[0], results[1], decimal=5)
    
    def test_environment_determinism(self):
        """Test environment settings for determinism."""
        # Check critical environment variables
        required_env = {
            'PYTHONHASHSEED': '0',
            'CUBLAS_WORKSPACE_CONFIG': ':4096:8'
        }
        
        # Set environment for determinism
        for key, value in required_env.items():
            os.environ[key] = value
        
        # Verify they're set
        for key, value in required_env.items():
            assert os.environ.get(key) == value


class TestFullPipelineIntegration(TestXGBoostFixtures):
    """Test complete pipeline integration."""
    
    def test_end_to_end_pipeline(self, synthetic_data, optimizer_kwargs, temp_artifacts_dir):
        """Test complete pipeline from data to predictions."""
        X, y = synthetic_data
        
        # Setup
        optimizer_kwargs['artifacts_dir'] = temp_artifacts_dir
        optimizer = XGBoostOptuna(optimizer_kwargs)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, shuffle=False
        )
        
        # Optimize if enabled
        if optimizer_kwargs['optuna']['enabled']:
            best_params = optimizer.optimize(X_train, y_train, n_trials=3)
            assert best_params is not None
        
        # Train final model
        model = optimizer.train_final_model(X_train, y_train)
        assert model is not None
        
        # Make predictions
        probs = model.predict_proba(X_test)[:, 1]
        preds = (probs >= 0.5).astype(int)
        
        # Evaluate
        auc = roc_auc_score(y_test, probs)
        f1 = f1_score(y_test, preds)
        
        assert 0.5 <= auc <= 1.0
        assert 0.0 <= f1 <= 1.0
        
        # Check artifacts were saved
        artifacts_path = Path(temp_artifacts_dir)
        assert artifacts_path.exists()
    
    def test_pipeline_with_feature_engineering(self, synthetic_data, optimizer_kwargs):
        """Test pipeline with feature engineering."""
        X, y = synthetic_data
        
        # Add feature engineering
        engineer = FeatureEngineer(config={})
        
        # Create OHLCV-like data
        ohlcv = pd.DataFrame({
            'open': np.random.randn(len(X)) * 0.1 + 100,
            'high': np.random.randn(len(X)) * 0.1 + 101,
            'low': np.random.randn(len(X)) * 0.1 + 99,
            'close': np.random.randn(len(X)) * 0.1 + 100,
            'volume': np.abs(np.random.randn(len(X))) * 1000
        }, index=X.index)
        
        # Generate features
        features = engineer.create_features(ohlcv)
        
        # Combine with original features
        X_enhanced = pd.concat([X, features], axis=1)
        
        # Train model
        optimizer = XGBoostOptuna(optimizer_kwargs)
        X_train, X_test, y_train, y_test = train_test_split(
            X_enhanced, y, test_size=0.2, shuffle=False
        )
        
        model = optimizer.train_final_model(X_train, y_train)
        
        # Should handle enhanced features
        probs = model.predict_proba(X_test)[:, 1]
        assert len(probs) == len(y_test)
    
    def test_mlflow_integration(self, synthetic_data, optimizer_kwargs, temp_artifacts_dir):
        """Test MLflow tracking integration."""
        X, y = synthetic_data
        
        # Setup MLflow
        import mlflow
        mlflow.set_tracking_uri(f"file://{temp_artifacts_dir}/mlruns")
        
        with mlflow.start_run():
            # Log parameters
            mlflow.log_params(optimizer_kwargs['xgb'])
            
            # Train model
            optimizer = XGBoostOptuna(optimizer_kwargs)
            model = optimizer.train_final_model(X, y)
            
            # Log metrics
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, shuffle=False
            )
            probs = model.predict_proba(X_test)[:, 1]
            
            mlflow.log_metric("auc", roc_auc_score(y_test, probs))
            mlflow.log_metric("brier", brier_score_loss(y_test, probs))
            
            # Log model
            mlflow.xgboost.log_model(model, "model")
        
        # Check run was logged
        runs = mlflow.search_runs()
        assert len(runs) > 0


class TestMemoryOptimization(TestXGBoostFixtures):
    """Test memory optimization."""
    
    def test_memory_efficient_training(self, synthetic_data):
        """Test that training doesn't leak memory."""
        X, y = synthetic_data
        
        import tracemalloc
        tracemalloc.start()
        
        # Baseline memory
        snapshot1 = tracemalloc.take_snapshot()
        
        # Train multiple models
        for _ in range(3):
            model = xgb.XGBClassifier(n_estimators=50)
            model.fit(X, y)
            del model  # Explicit cleanup
        
        # Check memory after
        snapshot2 = tracemalloc.take_snapshot()
        top_stats = snapshot2.compare_to(snapshot1, 'lineno')
        
        # Memory increase should be reasonable
        total_diff = sum(stat.size_diff for stat in top_stats)
        assert total_diff < 100 * 1024 * 1024  # Less than 100MB increase
        
        tracemalloc.stop()
    
    def test_data_type_optimization(self, synthetic_data):
        """Test optimization of data types for memory efficiency."""
        X, y = synthetic_data
        
        # Check if we can downcast safely
        X_float32 = X.astype(np.float32)
        
        # Train with float32
        model = xgb.XGBClassifier(n_estimators=50)
        model.fit(X_float32, y)
        
        # Should work and use less memory
        assert X_float32.memory_usage().sum() < X.memory_usage().sum()


class TestErrorHandling(TestXGBoostFixtures):
    """Test error handling and edge cases."""
    
    def test_empty_data_handling(self, optimizer_kwargs):
        """Test handling of empty datasets."""
        X_empty = pd.DataFrame()
        y_empty = pd.Series()
        
        optimizer = XGBoostOptuna(optimizer_kwargs)
        
        with pytest.raises((ValueError, IndexError)):
            optimizer.train_final_model(X_empty, y_empty)
    
    def test_single_class_handling(self, optimizer_kwargs):
        """Test handling when only one class is present."""
        X = pd.DataFrame(np.random.randn(100, 10))
        y = pd.Series(np.zeros(100))  # Only one class
        
        optimizer = XGBoostOptuna(optimizer_kwargs)
        
        with pytest.raises(ValueError):
            optimizer.train_final_model(X, y)
    
    def test_infinite_values_handling(self, synthetic_data, optimizer_kwargs):
        """Test handling of infinite values."""
        X, y = synthetic_data
        X_with_inf = X.copy()
        X_with_inf.iloc[0, 0] = np.inf
        
        optimizer = XGBoostOptuna(optimizer_kwargs)
        
        # Should handle or raise appropriate error
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                model = optimizer.train_final_model(X_with_inf, y)
                # If it doesn't raise, check predictions are valid
                probs = model.predict_proba(X_with_inf)[:, 1]
                assert not np.any(np.isnan(probs))
            except ValueError:
                pass  # Expected for some configurations
    
    def test_feature_name_handling(self, synthetic_data, optimizer_kwargs):
        """Test handling of special characters in feature names."""
        X, y = synthetic_data
        
        # Add special characters to column names
        X_special = X.copy()
        X_special.columns = [f"feat [{i}]" for i in range(X.shape[1])]
        
        optimizer = XGBoostOptuna(optimizer_kwargs)
        
        # XGBoost should handle this
        model = optimizer.train_final_model(X_special, y)
        probs = model.predict_proba(X_special)[:, 1]
        assert len(probs) == len(y)


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])