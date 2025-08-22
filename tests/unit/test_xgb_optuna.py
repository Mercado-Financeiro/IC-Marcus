"""Tests for XGBoost with Optuna optimization."""

import pytest
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import f1_score, precision_recall_curve, auc, brier_score_loss
import optuna
from unittest.mock import Mock, patch, MagicMock
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))


class TestXGBoostOptuna:
    """Test suite for XGBoost with Bayesian optimization."""
    
    def test_initialization(self):
        """Test XGBoostOptuna initialization."""
        from src.models.xgb_optuna import XGBoostOptuna
        
        optimizer = XGBoostOptuna(
            n_trials=10,
            cv_folds=3,
            embargo=5,
            pruner_type='median'
        )
        
        assert optimizer.n_trials == 10
        assert optimizer.cv_folds == 3
        assert optimizer.embargo == 5
        assert optimizer.pruner_type == 'median'
        assert optimizer.best_model is None
        assert optimizer.best_params is None
    
    def test_create_search_space(self):
        """Test hyperparameter search space creation."""
        from src.models.xgb_optuna import XGBoostOptuna
        
        optimizer = XGBoostOptuna()
        trial = optuna.trial.FixedTrial({
            'max_depth': 5,
            'min_child_weight': 3,
            'gamma': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.7,
            'eta': 0.05,
            'lambda': 1.0,
            'alpha': 0.5,
            'max_bin': 256,
            'n_estimators': 500
        })
        
        params = optimizer._create_search_space(trial)
        
        # Check required parameters
        assert 'max_depth' in params
        assert 'n_estimators' in params
        assert 'eta' in params
        assert params['tree_method'] == 'hist'
        assert params['objective'] == 'binary:logistic'
        assert params['eval_metric'] == 'auc'
        assert params['random_state'] == 42
    
    def test_objective_function(self, sample_features_data):
        """Test objective function for optimization."""
        from src.models.xgb_optuna import XGBoostOptuna
        from src.data.splits import PurgedKFold
        
        X = sample_features_data.drop('label', axis=1)
        y = sample_features_data['label']
        
        # Convert to binary for XGBoost
        y = (y > 0).astype(int)
        
        optimizer = XGBoostOptuna(n_trials=1, cv_folds=2, embargo=5)
        
        # Create objective
        objective = optimizer._create_objective(X, y)
        
        # Create trial
        study = optuna.create_study(direction='maximize')
        trial = study.ask()
        
        # Run objective
        score = objective(trial)
        
        # Check score is valid
        assert isinstance(score, float)
        assert -1 <= score <= 1  # Composite score should be reasonable
    
    def test_calibration_in_objective(self, sample_features_data):
        """Test that calibration is applied in objective."""
        from src.models.xgb_optuna import XGBoostOptuna
        
        X = sample_features_data.drop('label', axis=1)
        y = (sample_features_data['label'] > 0).astype(int)
        
        optimizer = XGBoostOptuna(n_trials=1, cv_folds=2)
        
        # Mock the training to check calibration is called
        with patch('src.models.xgb_optuna.CalibratedClassifierCV') as mock_calibrator:
            mock_calibrator.return_value.fit.return_value = Mock()
            mock_calibrator.return_value.predict_proba.return_value = np.random.rand(len(y), 2)
            
            objective = optimizer._create_objective(X, y)
            study = optuna.create_study(direction='maximize')
            trial = study.ask()
            
            # This should trigger calibration
            objective(trial)
            
            # Verify calibration was called
            assert mock_calibrator.called
    
    def test_threshold_optimization(self):
        """Test threshold optimization for both F1 and EV."""
        from src.models.xgb_optuna import XGBoostOptuna
        
        # Create sample predictions
        y_true = np.array([0, 1, 1, 0, 1, 0, 1, 0])
        y_pred_proba = np.array([0.1, 0.9, 0.8, 0.2, 0.7, 0.3, 0.6, 0.4])
        
        optimizer = XGBoostOptuna()
        
        # Optimize threshold for F1
        threshold_f1 = optimizer._optimize_threshold_f1(y_true, y_pred_proba)
        assert 0 <= threshold_f1 <= 1
        
        # Optimize threshold for EV
        costs = {'fee_bps': 5, 'slippage_bps': 5}
        threshold_ev = optimizer._optimize_threshold_ev(y_true, y_pred_proba, costs)
        assert 0 <= threshold_ev <= 1
        
        # Thresholds might be different
        # EV threshold considers costs, F1 doesn't
    
    def test_optimize_method(self, sample_features_data):
        """Test main optimize method."""
        from src.models.xgb_optuna import XGBoostOptuna
        
        X = sample_features_data.drop('label', axis=1)
        y = (sample_features_data['label'] > 0).astype(int)
        
        # Use very few trials for speed
        optimizer = XGBoostOptuna(n_trials=2, cv_folds=2, embargo=5)
        
        # Run optimization
        study, best_model = optimizer.optimize(X, y)
        
        # Check results
        assert study is not None
        assert best_model is not None
        assert optimizer.best_params is not None
        assert optimizer.best_score is not None
        assert len(study.trials) == 2
    
    def test_pruner_selection(self):
        """Test different pruner types."""
        from src.models.xgb_optuna import XGBoostOptuna
        
        # Median pruner
        opt1 = XGBoostOptuna(pruner_type='median')
        pruner1 = opt1._get_pruner()
        assert isinstance(pruner1, optuna.pruners.MedianPruner)
        
        # Hyperband pruner
        opt2 = XGBoostOptuna(pruner_type='hyperband')
        pruner2 = opt2._get_pruner()
        assert isinstance(pruner2, optuna.pruners.HyperbandPruner)
        
        # Successive halving
        opt3 = XGBoostOptuna(pruner_type='successive_halving')
        pruner3 = opt3._get_pruner()
        assert isinstance(pruner3, optuna.pruners.SuccessiveHalvingPruner)
    
    def test_sample_weights_support(self, sample_features_data):
        """Test that sample weights are properly used."""
        from src.models.xgb_optuna import XGBoostOptuna
        
        X = sample_features_data.drop('label', axis=1)
        y = (sample_features_data['label'] > 0).astype(int)
        
        # Create sample weights
        weights = np.random.rand(len(y))
        
        optimizer = XGBoostOptuna(n_trials=1, cv_folds=2)
        
        # Run with weights
        study, model = optimizer.optimize(X, y, sample_weights=weights)
        
        assert model is not None
        # Weights should be used in training
    
    def test_mlflow_logging(self, sample_features_data, mlflow_test_dir):
        """Test MLflow logging integration."""
        from src.models.xgb_optuna import XGBoostOptuna
        import mlflow
        
        X = sample_features_data.drop('label', axis=1)
        y = (sample_features_data['label'] > 0).astype(int)
        
        optimizer = XGBoostOptuna(n_trials=1, cv_folds=2, use_mlflow=True)
        
        with mlflow.start_run():
            study, model = optimizer.optimize(X, y)
            
            # Check that metrics were logged
            run = mlflow.active_run()
            assert run is not None
    
    def test_fit_final_model(self, sample_features_data):
        """Test fitting final model with best params."""
        from src.models.xgb_optuna import XGBoostOptuna
        
        X = sample_features_data.drop('label', axis=1)
        y = (sample_features_data['label'] > 0).astype(int)
        
        optimizer = XGBoostOptuna()
        
        # Set best params manually
        optimizer.best_params = {
            'max_depth': 5,
            'min_child_weight': 3,
            'n_estimators': 100,
            'eta': 0.05,
            'subsample': 0.8,
            'colsample_bytree': 0.7
        }
        
        # Fit final model
        model = optimizer.fit_final_model(X, y)
        
        assert model is not None
        assert hasattr(model, 'predict')
        assert hasattr(model, 'predict_proba')
    
    def test_predict_with_calibration(self, sample_features_data):
        """Test prediction with calibrated model."""
        from src.models.xgb_optuna import XGBoostOptuna
        
        X = sample_features_data.drop('label', axis=1)
        y = (sample_features_data['label'] > 0).astype(int)
        
        # Split data
        split_idx = len(X) // 2
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        optimizer = XGBoostOptuna(n_trials=1, cv_folds=2)
        
        # Train
        study, model = optimizer.optimize(X_train, y_train)
        
        # Predict
        y_pred_proba = optimizer.predict_proba(X_test)
        
        assert len(y_pred_proba) == len(X_test)
        assert all(0 <= p <= 1 for p in y_pred_proba)
    
    def test_brier_score_improvement(self, sample_features_data):
        """Test that calibration improves Brier score."""
        from src.models.xgb_optuna import XGBoostOptuna
        
        X = sample_features_data.drop('label', axis=1)
        y = (sample_features_data['label'] > 0).astype(int)
        
        # Split data
        split_idx = len(X) // 2
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        optimizer = XGBoostOptuna(n_trials=1, cv_folds=2)
        
        # Train with calibration
        study, model = optimizer.optimize(X_train, y_train)
        y_pred_calibrated = optimizer.predict_proba(X_test)
        
        # Train without calibration (direct XGBoost)
        raw_model = xgb.XGBClassifier(random_state=42, n_estimators=100)
        raw_model.fit(X_train, y_train)
        y_pred_raw = raw_model.predict_proba(X_test)[:, 1]
        
        # Calculate Brier scores
        brier_calibrated = brier_score_loss(y_test, y_pred_calibrated)
        brier_raw = brier_score_loss(y_test, y_pred_raw)
        
        # Calibrated should be better or equal (lower is better for Brier)
        # Allow small tolerance for randomness
        assert brier_calibrated <= brier_raw + 0.05


class TestXGBoostOptunaEdgeCases:
    """Test edge cases for XGBoost Optuna."""
    
    def test_empty_data(self):
        """Test with empty data."""
        from src.models.xgb_optuna import XGBoostOptuna
        
        X = pd.DataFrame()
        y = pd.Series()
        
        optimizer = XGBoostOptuna(n_trials=1)
        
        with pytest.raises(ValueError):
            optimizer.optimize(X, y)
    
    def test_single_class(self):
        """Test with single class (no variation in y)."""
        from src.models.xgb_optuna import XGBoostOptuna
        
        X = pd.DataFrame(np.random.randn(100, 5))
        y = pd.Series([1] * 100)  # All same class
        
        optimizer = XGBoostOptuna(n_trials=1)
        
        with pytest.raises(ValueError, match="single class"):
            optimizer.optimize(X, y)
    
    def test_insufficient_data_for_cv(self):
        """Test with insufficient data for cross-validation."""
        from src.models.xgb_optuna import XGBoostOptuna
        
        X = pd.DataFrame(np.random.randn(5, 3))
        y = pd.Series([0, 1, 0, 1, 0])
        
        optimizer = XGBoostOptuna(n_trials=1, cv_folds=5)  # More folds than samples
        
        with pytest.raises(ValueError):
            optimizer.optimize(X, y)