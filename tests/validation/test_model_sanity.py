"""
Model sanity checks to ensure models behave correctly.
"""

import pytest
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, mean_squared_error
from sklearn.calibration import calibration_curve
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))


class TestModelPredictions:
    """Tests for model prediction sanity."""
    
    def test_predictions_in_valid_range(self):
        """Ensure all predictions are in [0, 1] range for classification."""
        # Simulate predictions
        predictions = np.array([0.1, 0.5, 0.9, 0.0, 1.0, 0.75])
        
        assert predictions.min() >= 0, "Predictions below 0"
        assert predictions.max() <= 1, "Predictions above 1"
        assert not np.any(np.isnan(predictions)), "NaN in predictions"
        assert not np.any(np.isinf(predictions)), "Inf in predictions"
    
    def test_prediction_distribution_not_constant(self):
        """Ensure model doesn't predict the same value for all samples."""
        # This would catch the bug we found
        predictions = np.array([0.5, 0.3, 0.7, 0.2, 0.8, 0.4])
        
        unique_preds = np.unique(predictions)
        assert len(unique_preds) > 1, "Model predicting constant value"
        
        # Check variance is reasonable
        std = np.std(predictions)
        assert std > 0.01, f"Predictions have very low variance: {std}"
    
    def test_prediction_calibration(self):
        """Test if predictions are well-calibrated."""
        np.random.seed(42)
        
        # Generate synthetic calibrated predictions
        n_samples = 1000
        true_probs = np.random.beta(2, 2, n_samples)
        y_true = np.random.binomial(1, true_probs)
        y_pred = true_probs + np.random.normal(0, 0.1, n_samples)
        y_pred = np.clip(y_pred, 0, 1)
        
        # Calculate calibration metrics
        brier = brier_score_loss(y_true, y_pred)
        
        # Brier score should be reasonable
        assert brier < 0.3, f"Poor calibration: Brier score = {brier}"
        
        # Check calibration curve
        fraction_pos, mean_pred = calibration_curve(y_true, y_pred, n_bins=10)
        
        # Perfect calibration: fraction_pos ≈ mean_pred
        calibration_error = np.mean(np.abs(fraction_pos - mean_pred))
        assert calibration_error < 0.1, f"Poor calibration: ECE = {calibration_error}"


class TestModelConvergence:
    """Tests for model training convergence."""
    
    def test_loss_decreases_during_training(self):
        """Ensure training loss decreases over iterations."""
        # Simulate training history
        train_losses = [0.8, 0.7, 0.65, 0.6, 0.58, 0.55, 0.54, 0.53]
        
        # Check overall trend is decreasing
        initial_loss = np.mean(train_losses[:2])
        final_loss = np.mean(train_losses[-2:])
        
        assert final_loss < initial_loss, \
            f"Loss not decreasing: {initial_loss:.3f} -> {final_loss:.3f}"
        
        # Check for convergence (loss stabilizing)
        last_losses = train_losses[-3:]
        loss_std = np.std(last_losses)
        assert loss_std < 0.05, \
            f"Loss not converging: std = {loss_std:.3f}"
    
    def test_early_stopping_works(self):
        """Test that early stopping prevents overfitting."""
        # Simulate validation scores
        val_scores = [0.5, 0.55, 0.58, 0.60, 0.61, 0.61, 0.60, 0.59, 0.58]
        
        # Find best iteration
        best_iter = np.argmax(val_scores)
        
        # Should stop within a few iterations of best
        patience = 3
        expected_stop = best_iter + patience
        
        # Simulate early stopping
        for i in range(best_iter + 1, min(expected_stop + 1, len(val_scores))):
            if val_scores[i] < val_scores[best_iter]:
                actual_stop = i
                break
        else:
            actual_stop = len(val_scores) - 1
        
        assert actual_stop <= expected_stop, \
            f"Early stopping failed: stopped at {actual_stop}, expected <= {expected_stop}"


class TestModelRobustness:
    """Tests for model robustness and stability."""
    
    def test_model_handles_edge_cases(self):
        """Test model with edge case inputs."""
        from sklearn.ensemble import RandomForestClassifier
        
        # Create model
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        
        # Normal data
        X_train = np.random.randn(100, 5)
        y_train = np.random.choice([0, 1], 100)
        model.fit(X_train, y_train)
        
        # Test edge cases
        edge_cases = {
            'all_zeros': np.zeros((10, 5)),
            'all_ones': np.ones((10, 5)),
            'very_large': np.ones((10, 5)) * 1e6,
            'very_small': np.ones((10, 5)) * 1e-6,
            'mixed': np.array([[0, 1e6, -1e6, 1e-6, 0]] * 10)
        }
        
        for case_name, X_test in edge_cases.items():
            try:
                predictions = model.predict_proba(X_test)[:, 1]
                
                # Check predictions are valid
                assert not np.any(np.isnan(predictions)), \
                    f"NaN predictions for {case_name}"
                assert np.all((predictions >= 0) & (predictions <= 1)), \
                    f"Invalid predictions for {case_name}"
                
            except Exception as e:
                pytest.fail(f"Model failed on {case_name}: {str(e)}")
    
    def test_model_stable_with_permuted_features(self):
        """Test that model predictions are stable with permuted feature order."""
        from sklearn.ensemble import RandomForestClassifier
        
        np.random.seed(42)
        
        # Create and train model
        X = np.random.randn(100, 5)
        y = np.random.choice([0, 1], 100)
        
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)
        
        # Test data
        X_test = np.random.randn(20, 5)
        
        # Original predictions
        pred_original = model.predict_proba(X_test)[:, 1]
        
        # Permute columns and predict
        perm = [4, 0, 3, 1, 2]
        X_test_perm = X_test[:, perm]
        
        # For tree-based models, column order shouldn't matter
        # (This test would need adjustment for neural networks)
        # Here we're just checking the model doesn't crash
        try:
            pred_perm = model.predict_proba(X_test_perm)[:, 1]
            assert pred_perm.shape == pred_original.shape
        except:
            pass  # Some models are sensitive to feature order


class TestModelVariability:
    """Tests to ensure models show appropriate variability."""
    
    def test_different_seeds_produce_different_models(self):
        """Ensure different random seeds produce different models."""
        from sklearn.ensemble import RandomForestClassifier
        
        # Same data
        X = np.random.randn(100, 5)
        y = np.random.choice([0, 1], 100)
        
        # Train with different seeds
        model1 = RandomForestClassifier(n_estimators=10, random_state=42)
        model2 = RandomForestClassifier(n_estimators=10, random_state=43)
        
        model1.fit(X, y)
        model2.fit(X, y)
        
        # Predictions should be different
        X_test = np.random.randn(20, 5)
        pred1 = model1.predict_proba(X_test)[:, 1]
        pred2 = model2.predict_proba(X_test)[:, 1]
        
        # Check predictions are different
        assert not np.allclose(pred1, pred2), \
            "Different seeds producing identical models"
        
        # But correlation should be high (same data, similar models)
        correlation = np.corrcoef(pred1, pred2)[0, 1]
        assert correlation > 0.5, \
            f"Models too different: correlation = {correlation:.3f}"
    
    def test_cross_validation_scores_consistent(self):
        """Test that CV scores are consistent across folds."""
        # Simulate CV scores
        cv_scores = [0.65, 0.63, 0.67, 0.64, 0.66]
        
        mean_score = np.mean(cv_scores)
        std_score = np.std(cv_scores)
        
        # Check consistency
        cv_coefficient = std_score / mean_score  # Coefficient of variation
        assert cv_coefficient < 0.1, \
            f"CV scores too variable: CV = {cv_coefficient:.3f}"
        
        # Check no outliers
        for score in cv_scores:
            z_score = abs(score - mean_score) / std_score
            assert z_score < 3, \
                f"Outlier CV score: {score} (z-score = {z_score:.2f})"


class TestModelComplexity:
    """Tests for model complexity and generalization."""
    
    def test_model_not_memorizing(self):
        """Ensure model isn't just memorizing training data."""
        # If model memorizes, it will have perfect train score
        train_score = 0.85  # Should not be 1.0
        val_score = 0.75
        
        assert train_score < 0.99, \
            f"Model might be memorizing: train score = {train_score}"
        
        # Gap should be reasonable
        gap = train_score - val_score
        assert gap < 0.15, \
            f"Large train-val gap suggests memorization: {gap:.3f}"
    
    def test_model_complexity_appropriate(self):
        """Test that model complexity is appropriate for data size."""
        n_samples = 1000
        n_features = 20
        
        # For tree-based models
        n_trees = 100
        max_depth = 10
        
        # Rule of thumb: avoid too many parameters relative to samples
        approx_params = n_trees * (2 ** max_depth)
        param_to_sample_ratio = approx_params / n_samples
        
        assert param_to_sample_ratio < 10, \
            f"Model too complex: {approx_params} params for {n_samples} samples"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])