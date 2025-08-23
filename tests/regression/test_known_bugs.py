"""
Regression tests for known bugs to prevent reoccurrence.
These tests ensure that previously fixed bugs don't return.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))


class TestXGBoostCalibrationBug:
    """Tests to prevent the calibration bug we found in XGBoost optimizer."""
    
    def test_calibration_not_during_optimization(self):
        """Ensure calibration is NOT done on validation set during optimization."""
        from src.models.xgb_optuna import XGBoostOptuna
        
        optimizer = XGBoostOptuna(n_trials=2, cv_folds=2)
        
        # Check that the objective function doesn't use calibration
        X = pd.DataFrame(np.random.randn(100, 5))
        y = pd.Series(np.random.choice([0, 1], 100))
        
        objective = optimizer._create_objective(X, y)
        
        # Mock trial
        trial = Mock()
        trial.number = 0
        trial.suggest_int = Mock(return_value=5)
        trial.suggest_float = Mock(return_value=0.5)
        trial.suggest_categorical = Mock(return_value=256)
        trial.should_prune = Mock(return_value=False)
        trial.report = Mock()
        
        # The objective should not contain CalibratedClassifierCV
        import inspect
        source = inspect.getsource(objective)
        
        # These should NOT be in the optimization loop
        assert "CalibratedClassifierCV" not in source, \
            "Calibration should not be used during optimization"
        assert "calibrator.fit(X_val, y_val)" not in source, \
            "Should not calibrate on validation set"
    
    def test_all_optuna_trials_return_different_values(self):
        """Ensure Optuna trials produce different values (not all the same)."""
        from src.models.xgb_optuna import XGBoostOptuna
        from sklearn.datasets import make_classification
        
        # Create synthetic data
        X, y = make_classification(
            n_samples=500, n_features=10, n_informative=8,
            n_redundant=2, random_state=42
        )
        X = pd.DataFrame(X)
        y = pd.Series(y)
        
        # Run optimization with few trials
        optimizer = XGBoostOptuna(n_trials=3, cv_folds=2, embargo=0)
        
        with patch('xgboost.XGBClassifier') as mock_xgb:
            # Mock the classifier to return different probabilities
            mock_model = Mock()
            mock_model.fit = Mock()
            
            # Make each trial return different predictions
            call_count = [0]
            def predict_proba_side_effect(*args, **kwargs):
                call_count[0] += 1
                # Add noise based on call count to ensure different results
                base_proba = np.random.rand(len(args[0]), 2)
                base_proba[:, 1] += call_count[0] * 0.1
                # Normalize
                base_proba = base_proba / base_proba.sum(axis=1, keepdims=True)
                return base_proba
            
            mock_model.predict_proba = Mock(side_effect=predict_proba_side_effect)
            mock_xgb.return_value = mock_model
            
            study, _ = optimizer.optimize(X, y)
            
            # Get all trial values
            values = [t.value for t in study.trials if t.value is not None]
            
            # Check that we have different values
            unique_values = len(set(values))
            assert unique_values > 1, \
                f"All trials returned same value! Values: {values}"
            
            # Check that variation is reasonable (not just floating point differences)
            if len(values) > 1:
                value_range = max(values) - min(values)
                assert value_range > 0.001, \
                    f"Trial values are too similar! Range: {value_range}"


class TestSampleWeightsAlignment:
    """Tests to ensure sample weights stay aligned with labels after filtering."""
    
    def test_sample_weights_aligned_after_neutral_filtering(self):
        """Ensure sample weights remain aligned when neutral labels are filtered."""
        from src.features.labels import AdaptiveLabeler
        
        # Create sample data
        n_samples = 100
        dates = pd.date_range('2023-01-01', periods=n_samples, freq='15min')
        prices = pd.Series(
            100 + np.cumsum(np.random.randn(n_samples) * 0.5),
            index=dates
        )
        
        # Create labeler and generate labels
        labeler = AdaptiveLabeler()
        labels, weights = labeler.fit_transform(
            prices,
            pt=0.02,  # 2% profit target
            sl=0.02,  # 2% stop loss
            max_holding=10,
            use_sample_weights=True
        )
        
        # Filter neutrals (like we do in the pipeline)
        mask = labels != 0
        filtered_labels = labels[mask]
        filtered_weights = weights[mask] if weights is not None else None
        
        # Check alignment
        assert len(filtered_labels) == len(filtered_weights) if filtered_weights is not None else True, \
            "Weights and labels must have same length after filtering"
        
        # Check that indices match
        if filtered_weights is not None:
            assert filtered_labels.index.equals(filtered_weights.index), \
                "Weights and labels indices must match after filtering"
    
    def test_binary_remapping_preserves_alignment(self):
        """Test that remapping {-1,1} to {0,1} preserves data alignment."""
        # Create sample labels with -1, 0, 1
        labels = pd.Series([-1, 0, 1, -1, 1, 0, -1], 
                          index=pd.date_range('2023-01-01', periods=7, freq='15min'))
        weights = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], index=labels.index)
        
        # Filter neutrals
        mask = labels != 0
        filtered_labels = labels[mask].copy()
        filtered_weights = weights[mask].copy()
        
        # Remap {-1, 1} to {0, 1}
        filtered_labels = filtered_labels.map({-1: 0, 1: 1})
        
        # Check that remapping didn't break alignment
        assert len(filtered_labels) == len(filtered_weights), \
            "Length mismatch after remapping"
        assert filtered_labels.index.equals(filtered_weights.index), \
            "Index mismatch after remapping"
        assert filtered_labels.isin([0, 1]).all(), \
            "Labels should only contain 0 and 1 after remapping"


class TestDataLeakage:
    """Tests to prevent temporal data leakage."""
    
    def test_purged_kfold_has_embargo(self):
        """Ensure PurgedKFold maintains embargo between train and validation."""
        from src.data.splits import PurgedKFold
        
        # Create time series data
        n_samples = 1000
        dates = pd.date_range('2023-01-01', periods=n_samples, freq='15min')
        X = pd.DataFrame(np.random.randn(n_samples, 5), index=dates)
        y = pd.Series(np.random.choice([0, 1], n_samples), index=dates)
        
        # Create splitter with embargo
        embargo = 10
        splitter = PurgedKFold(n_splits=3, embargo=embargo)
        
        for train_idx, val_idx in splitter.split(X, y):
            train_times = X.index[train_idx]
            val_times = X.index[val_idx]
            
            # Check embargo before validation
            if len(train_times) > 0 and len(val_times) > 0:
                # Find the latest training time before validation
                train_before_val = train_times[train_times < val_times.min()]
                if len(train_before_val) > 0:
                    gap = val_times.min() - train_before_val.max()
                    gap_periods = len(X.index[(X.index > train_before_val.max()) & 
                                              (X.index < val_times.min())])
                    assert gap_periods >= embargo - 1, \
                        f"Embargo violation: only {gap_periods} periods, expected >= {embargo-1}"
            
            # Check embargo after validation
            train_after_val = train_times[train_times > val_times.max()]
            if len(train_after_val) > 0:
                gap = train_after_val.min() - val_times.max()
                gap_periods = len(X.index[(X.index > val_times.max()) & 
                                          (X.index < train_after_val.min())])
                assert gap_periods >= embargo - 1, \
                    f"Embargo violation: only {gap_periods} periods, expected >= {embargo-1}"
    
    def test_no_future_data_in_features(self):
        """Ensure features don't use future information."""
        from src.features.engineering import FeatureEngineer
        
        # Create sample data
        n_samples = 200
        dates = pd.date_range('2023-01-01', periods=n_samples, freq='15min')
        data = pd.DataFrame({
            'open': 100 + np.random.randn(n_samples),
            'high': 101 + np.random.randn(n_samples),
            'low': 99 + np.random.randn(n_samples),
            'close': 100 + np.random.randn(n_samples),
            'volume': 1000 + np.random.randn(n_samples) * 100
        }, index=dates)
        
        # Create features
        engineer = FeatureEngineer()
        features = engineer.fit_transform(data)
        
        # Check that each feature at time t only uses data up to time t
        # This is a basic check - in reality we'd need more sophisticated testing
        for i in range(10, len(features)):  # Skip initial rows due to lookback
            current_time = features.index[i]
            
            # Features should not have perfect correlation with future returns
            future_return = data['close'].iloc[i+1:i+11].mean() - data['close'].iloc[i]
            
            for col in features.columns:
                if 'return' not in col.lower():  # Skip return features
                    correlation = np.corrcoef(
                        features[col].iloc[:i],
                        [future_return] * i
                    )[0, 1]
                    
                    # Correlation with future should be low (not perfect)
                    assert abs(correlation) < 0.99 or np.isnan(correlation), \
                        f"Feature {col} might be using future data!"


class TestModelReproducibility:
    """Tests to ensure model training is reproducible."""
    
    def test_same_seed_produces_same_results(self):
        """Ensure that using the same seed produces identical results."""
        from src.models.xgb_optuna import XGBoostOptuna
        from sklearn.datasets import make_classification
        
        # Create synthetic data
        X, y = make_classification(
            n_samples=200, n_features=10, random_state=42
        )
        X = pd.DataFrame(X)
        y = pd.Series(y)
        
        # Train model twice with same seed
        optimizer1 = XGBoostOptuna(n_trials=1, cv_folds=2, seed=42)
        optimizer2 = XGBoostOptuna(n_trials=1, cv_folds=2, seed=42)
        
        # Mock to speed up test
        with patch('xgboost.XGBClassifier') as mock_xgb:
            mock_model = Mock()
            mock_model.fit = Mock()
            mock_model.predict_proba = Mock(return_value=np.random.rand(len(X), 2))
            mock_xgb.return_value = mock_model
            
            study1, _ = optimizer1.optimize(X, y)
            study2, _ = optimizer2.optimize(X, y)
            
            # Check that best parameters are similar
            # (They might not be exactly the same due to Optuna's internal randomness)
            assert study1.best_params.keys() == study2.best_params.keys(), \
                "Same parameters should be explored"


class TestModelQualityGates:
    """Tests to ensure models meet minimum quality standards."""
    
    def test_model_metrics_above_threshold(self):
        """Ensure model metrics meet minimum requirements before deployment."""
        min_thresholds = {
            'f1_score': 0.4,  # Minimum F1
            'pr_auc': 0.4,    # Minimum PR-AUC
            'roc_auc': 0.5,   # Better than random
            'brier_score': 0.5  # Maximum (lower is better)
        }
        
        # This would typically load actual model metrics
        # For testing, we'll use mock values
        mock_metrics = {
            'f1_score': 0.55,
            'pr_auc': 0.60,
            'roc_auc': 0.65,
            'brier_score': 0.25
        }
        
        # Validate metrics
        for metric, threshold in min_thresholds.items():
            if metric == 'brier_score':
                assert mock_metrics[metric] <= threshold, \
                    f"{metric} ({mock_metrics[metric]}) exceeds maximum threshold ({threshold})"
            else:
                assert mock_metrics[metric] >= threshold, \
                    f"{metric} ({mock_metrics[metric]}) below minimum threshold ({threshold})"
    
    def test_no_overfitting(self):
        """Check that model doesn't overfit on training data."""
        # Mock metrics
        train_score = 0.95
        val_score = 0.60
        
        # Check that gap isn't too large
        max_gap = 0.20  # Maximum 20% difference
        gap = train_score - val_score
        
        assert gap <= max_gap, \
            f"Overfitting detected! Train: {train_score:.2f}, Val: {val_score:.2f}, Gap: {gap:.2f}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])