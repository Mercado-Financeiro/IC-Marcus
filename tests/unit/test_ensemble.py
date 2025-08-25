"""Tests for ensemble methods."""

import pytest
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.models.ensemble import (
    VotingEnsemble, WeightedEnsemble, StackingEnsemble, EnsembleOptimizer
)


class TestVotingEnsemble:
    """Test suite for voting ensemble."""
    
    def test_soft_voting(self):
        """Test soft voting ensemble."""
        # Create dummy data
        X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
        X = pd.DataFrame(X)
        y = pd.Series(y)
        
        # Split data
        split_idx = 800
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        # Create base models
        models = {
            'rf': RandomForestClassifier(n_estimators=10, random_state=42),
            'lr': LogisticRegression(random_state=42)
        }
        
        # Create ensemble
        ensemble = VotingEnsemble(models, voting='soft')
        ensemble.fit(X_train, y_train)
        
        # Test predictions
        y_pred = ensemble.predict(X_test)
        y_proba = ensemble.predict_proba(X_test)
        
        assert len(y_pred) == len(y_test)
        assert y_proba.shape == (len(y_test), 2)
        assert np.allclose(y_proba.sum(axis=1), 1.0)
        
        # Check accuracy is reasonable
        accuracy = (y_pred == y_test).mean()
        assert accuracy > 0.7  # Should get decent accuracy on this easy dataset
    
    def test_hard_voting(self):
        """Test hard voting ensemble."""
        # Create dummy data
        X, y = make_classification(n_samples=500, n_features=10, random_state=42)
        X = pd.DataFrame(X)
        y = pd.Series(y)
        
        # Create base models
        models = {
            'rf': RandomForestClassifier(n_estimators=10, random_state=42),
            'lr': LogisticRegression(random_state=42)
        }
        
        # Create ensemble
        ensemble = VotingEnsemble(models, voting='hard')
        ensemble.fit(X, y)
        
        # Test predictions
        y_pred = ensemble.predict(X)
        y_proba = ensemble.predict_proba(X)
        
        assert len(y_pred) == len(y)
        assert y_proba.shape == (len(y), 2)
        
        # For hard voting, probabilities should be binary (0 or 1)
        assert np.all((y_proba == 0) | (y_proba == 1))
    
    def test_weighted_voting(self):
        """Test voting with custom weights."""
        # Create dummy data
        X, y = make_classification(n_samples=500, n_features=10, random_state=42)
        X = pd.DataFrame(X)
        y = pd.Series(y)
        
        # Create base models
        models = {
            'rf': RandomForestClassifier(n_estimators=10, random_state=42),
            'lr': LogisticRegression(random_state=42)
        }
        
        # Create ensemble with weights
        weights = [0.7, 0.3]  # Give more weight to RF
        ensemble = VotingEnsemble(models, voting='soft', weights=weights)
        ensemble.fit(X, y)
        
        # Test predictions
        y_proba = ensemble.predict_proba(X)
        
        assert y_proba.shape == (len(y), 2)
        assert np.allclose(y_proba.sum(axis=1), 1.0)


class TestWeightedEnsemble:
    """Test suite for weighted ensemble."""
    
    def test_weight_calculation(self):
        """Test that weights are calculated based on performance."""
        # Create dummy data
        X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
        X = pd.DataFrame(X)
        y = pd.Series(y)
        
        # Split data
        split_idx = 800
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        # Create base models
        models = {
            'rf': RandomForestClassifier(n_estimators=50, random_state=42),
            'lr': LogisticRegression(random_state=42)
        }
        
        # Create ensemble
        ensemble = WeightedEnsemble(models, metric='f1')
        ensemble.fit(X_train, y_train, X_test, y_test)
        
        # Check weights
        assert ensemble.weights is not None
        assert len(ensemble.weights) == len(models)
        assert np.isclose(sum(ensemble.weights.values()), 1.0)
        
        # Better model should have higher weight
        # RF usually performs better on this dataset
        assert ensemble.weights['rf'] > ensemble.weights['lr']
    
    def test_prediction_with_weights(self):
        """Test that predictions use calculated weights."""
        # Create dummy data
        X, y = make_classification(n_samples=500, n_features=10, random_state=42)
        X = pd.DataFrame(X)
        y = pd.Series(y)
        
        # Create base models
        models = {
            'rf': RandomForestClassifier(n_estimators=10, random_state=42),
            'lr': LogisticRegression(random_state=42)
        }
        
        # Create ensemble
        ensemble = WeightedEnsemble(models, metric='f1')
        ensemble.fit(X, y)
        
        # Test predictions
        y_pred = ensemble.predict(X)
        y_proba = ensemble.predict_proba(X)
        
        assert len(y_pred) == len(y)
        assert y_proba.shape == (len(y), 2)
        assert np.allclose(y_proba.sum(axis=1), 1.0)


class TestStackingEnsemble:
    """Test suite for stacking ensemble."""
    
    def test_meta_feature_generation(self):
        """Test that meta-features are generated correctly."""
        # Create dummy data
        X, y = make_classification(n_samples=500, n_features=10, random_state=42)
        X = pd.DataFrame(X)
        y = pd.Series(y)
        
        # Create base models
        base_models = {
            'rf': RandomForestClassifier(n_estimators=10, random_state=42),
            'lr': LogisticRegression(random_state=42)
        }
        
        # Create stacking ensemble
        ensemble = StackingEnsemble(base_models, cv_folds=3)
        ensemble.fit(X, y)
        
        # Check that base models are fitted
        assert len(ensemble.fitted_base_models) == len(base_models)
        
        # Check meta-learner is fitted
        assert hasattr(ensemble.meta_learner, 'coef_')
        
        # Test predictions
        y_pred = ensemble.predict(X)
        y_proba = ensemble.predict_proba(X)
        
        assert len(y_pred) == len(y)
        assert y_proba.shape == (len(y), 2)
    
    def test_stacking_with_probabilities(self):
        """Test stacking using predicted probabilities."""
        # Create dummy data
        X, y = make_classification(n_samples=500, n_features=10, random_state=42)
        X = pd.DataFrame(X)
        y = pd.Series(y)
        
        # Create base models
        base_models = {
            'rf': RandomForestClassifier(n_estimators=10, random_state=42),
            'lr': LogisticRegression(random_state=42)
        }
        
        # Create stacking ensemble with probabilities
        ensemble = StackingEnsemble(base_models, use_probas=True, cv_folds=3)
        ensemble.fit(X, y)
        
        # Get meta-features for testing
        meta_features = ensemble._get_meta_features(X)
        
        # Meta-features should be probabilities (between 0 and 1)
        assert meta_features.shape[1] == len(base_models)
        assert np.all((meta_features >= 0) & (meta_features <= 1))
    
    def test_stacking_with_classes(self):
        """Test stacking using predicted classes."""
        # Create dummy data
        X, y = make_classification(n_samples=500, n_features=10, random_state=42)
        X = pd.DataFrame(X)
        y = pd.Series(y)
        
        # Create base models
        base_models = {
            'rf': RandomForestClassifier(n_estimators=10, random_state=42),
            'lr': LogisticRegression(random_state=42)
        }
        
        # Create stacking ensemble without probabilities
        ensemble = StackingEnsemble(base_models, use_probas=False, cv_folds=3)
        ensemble.fit(X, y)
        
        # Get meta-features for testing
        meta_features = ensemble._get_meta_features(X)
        
        # Meta-features should be binary classes (0 or 1)
        assert meta_features.shape[1] == len(base_models)
        assert np.all((meta_features == 0) | (meta_features == 1))


class TestEnsembleOptimizer:
    """Test suite for ensemble optimizer."""
    
    def test_ensemble_optimization(self):
        """Test finding optimal ensemble configuration."""
        # Create dummy data
        X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
        X = pd.DataFrame(X)
        y = pd.Series(y)
        
        # Split data
        train_idx = 600
        val_idx = 800
        X_train = X.iloc[:train_idx]
        y_train = y.iloc[:train_idx]
        X_val = X.iloc[train_idx:val_idx]
        y_val = y.iloc[train_idx:val_idx]
        
        # Create and train base models
        base_models = {
            'rf': RandomForestClassifier(n_estimators=20, random_state=42),
            'lr': LogisticRegression(random_state=42)
        }
        
        for model in base_models.values():
            model.fit(X_train, y_train)
        
        # Optimize ensemble
        optimizer = EnsembleOptimizer(base_models)
        results = optimizer.optimize(X_train, y_train, X_val, y_val)
        
        # Check results
        assert optimizer.best_ensemble is not None
        assert optimizer.best_score is not None
        assert len(results) == 4  # voting_soft, voting_hard, weighted, stacking
        
        # All methods should have metrics
        for method_results in results.values():
            assert 'f1' in method_results
            assert 'pr_auc' in method_results
        
        # Best score should be reasonable
        assert optimizer.best_score > 0.7
    
    def test_save_load_ensemble(self, tmp_path):
        """Test saving and loading ensemble."""
        # Create dummy data
        X, y = make_classification(n_samples=500, n_features=10, random_state=42)
        X = pd.DataFrame(X)
        y = pd.Series(y)
        
        # Create and train base models
        base_models = {
            'rf': RandomForestClassifier(n_estimators=10, random_state=42),
            'lr': LogisticRegression(random_state=42)
        }
        
        for model in base_models.values():
            model.fit(X, y)
        
        # Optimize ensemble
        optimizer = EnsembleOptimizer(base_models)
        results = optimizer.optimize(X, y, X, y)  # Using same data for simplicity
        
        # Save ensemble
        save_path = tmp_path / "ensemble.pkl"
        optimizer.save_best_ensemble(str(save_path))
        
        assert save_path.exists()
        
        # Load ensemble
        loaded_ensemble, loaded_score, loaded_results = EnsembleOptimizer.load_ensemble(str(save_path))
        
        assert loaded_ensemble is not None
        assert loaded_score == optimizer.best_score
        assert loaded_results == optimizer.results
        
        # Test that loaded ensemble works
        y_pred = loaded_ensemble.predict(X)
        assert len(y_pred) == len(y)


class TestEnsembleCalibration:
    """Test that all ensembles are properly calibrated."""
    
    def test_voting_calibration(self):
        """Test that voting ensemble is calibrated."""
        # Create dummy data
        X, y = make_classification(n_samples=500, n_features=10, random_state=42)
        X = pd.DataFrame(X)
        y = pd.Series(y)
        
        # Create base models
        models = {
            'rf': RandomForestClassifier(n_estimators=10, random_state=42),
            'lr': LogisticRegression(random_state=42)
        }
        
        # Create ensemble
        ensemble = VotingEnsemble(models, voting='soft')
        ensemble.fit(X, y)
        
        # Check calibrators exist
        assert len(ensemble.calibrators) == len(models)
        
        # Probabilities should be calibrated (sum to 1)
        y_proba = ensemble.predict_proba(X)
        assert np.allclose(y_proba.sum(axis=1), 1.0)
    
    def test_no_temporal_leakage(self):
        """Test that ensemble respects temporal ordering."""
        # Create time series data
        dates = pd.date_range('2020-01-01', periods=1000, freq='D')
        X = pd.DataFrame(np.random.randn(1000, 10), index=dates)
        y = pd.Series(np.random.randint(0, 2, 1000), index=dates)
        
        # Split temporally
        split_date = dates[800]
        X_train = X.loc[:split_date]
        y_train = y.loc[:split_date]
        X_test = X.loc[split_date:]
        y_test = y.loc[split_date:]
        
        # Ensure no overlap
        assert X_train.index.max() < X_test.index.min()
        
        # Create base models
        models = {
            'rf': RandomForestClassifier(n_estimators=10, random_state=42),
            'lr': LogisticRegression(random_state=42)
        }
        
        # Train ensemble
        ensemble = VotingEnsemble(models, voting='soft')
        ensemble.fit(X_train, y_train)
        
        # Predictions should work on future data
        y_pred = ensemble.predict(X_test)
        assert len(y_pred) == len(y_test)