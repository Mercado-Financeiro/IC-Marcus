"""
Ensemble methods for combining multiple ML models.

Implements:
- Voting Ensemble (hard and soft voting)
- Weighted Average Ensemble
- Stacking Ensemble with meta-learner
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Union
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
import pickle
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


class VotingEnsemble(BaseEstimator, ClassifierMixin):
    """
    Voting ensemble that combines predictions from multiple models.
    
    Supports both hard voting (majority vote) and soft voting (average probabilities).
    """
    
    def __init__(self, models: Dict[str, BaseEstimator], voting: str = 'soft',
                 weights: Optional[List[float]] = None):
        """
        Args:
            models: Dictionary of {name: model} pairs
            voting: 'hard' for majority vote, 'soft' for probability averaging
            weights: Optional weights for each model (must sum to 1 for soft voting)
        """
        self.models = models
        self.voting = voting
        self.weights = weights
        self.calibrators = {}
        
        if weights is not None:
            assert len(weights) == len(models), "Number of weights must match number of models"
            if voting == 'soft':
                assert np.isclose(sum(weights), 1.0), "Weights must sum to 1 for soft voting"
    
    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        Fit all models in the ensemble.
        
        Args:
            X: Training features
            y: Training labels
        """
        log.info(f"Fitting {len(self.models)} models in voting ensemble")
        
        for name, model in self.models.items():
            log.info(f"Fitting {name}...")
            model.fit(X, y)
            
            # Calibrate each model
            log.info(f"Calibrating {name}...")
            self.calibrators[name] = CalibratedClassifierCV(
                model, method='isotonic', cv='prefit'
            )
            self.calibrators[name].fit(X, y)
        
        self.classes_ = np.unique(y)
        log.info("Voting ensemble fitted successfully")
        return self
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict class probabilities.
        
        Args:
            X: Features to predict
            
        Returns:
            Array of shape (n_samples, n_classes) with class probabilities
        """
        if self.voting == 'hard':
            # For hard voting, return binary probabilities based on vote
            predictions = self.predict(X)
            n_samples = len(predictions)
            proba = np.zeros((n_samples, 2))
            proba[range(n_samples), predictions] = 1.0
            return proba
        
        else:  # soft voting
            probas = []
            
            for name, calibrator in self.calibrators.items():
                proba = calibrator.predict_proba(X)
                probas.append(proba)
            
            # Average probabilities
            if self.weights is None:
                avg_proba = np.mean(probas, axis=0)
            else:
                avg_proba = np.average(probas, axis=0, weights=self.weights)
            
            return avg_proba
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict class labels.
        
        Args:
            X: Features to predict
            
        Returns:
            Array of predicted class labels
        """
        if self.voting == 'hard':
            # Collect predictions from each model
            predictions = []
            
            for name, model in self.models.items():
                pred = model.predict(X)
                predictions.append(pred)
            
            # Majority vote
            predictions = np.array(predictions).T
            
            if self.weights is None:
                # Simple majority
                ensemble_pred = np.apply_along_axis(
                    lambda x: np.bincount(x).argmax(), 1, predictions
                )
            else:
                # Weighted majority
                ensemble_pred = []
                for row in predictions:
                    weighted_votes = np.zeros(len(self.classes_))
                    for i, vote in enumerate(row):
                        weighted_votes[vote] += self.weights[i]
                    ensemble_pred.append(weighted_votes.argmax())
                ensemble_pred = np.array(ensemble_pred)
            
            return ensemble_pred
        
        else:  # soft voting
            proba = self.predict_proba(X)
            return (proba[:, 1] >= 0.5).astype(int)


class WeightedEnsemble(BaseEstimator, ClassifierMixin):
    """
    Weighted average ensemble based on out-of-sample performance.
    
    Weights are determined by model performance metrics (e.g., F1 score, Sharpe ratio).
    """
    
    def __init__(self, models: Dict[str, BaseEstimator], metric: str = 'f1'):
        """
        Args:
            models: Dictionary of {name: model} pairs
            metric: Metric to use for weighting ('f1', 'sharpe', 'pr_auc')
        """
        self.models = models
        self.metric = metric
        self.weights = None
        self.calibrators = {}
    
    def fit(self, X: pd.DataFrame, y: pd.Series, X_val: pd.DataFrame = None, 
            y_val: pd.Series = None):
        """
        Fit models and determine optimal weights based on validation performance.
        
        Args:
            X: Training features
            y: Training labels
            X_val: Validation features (optional, uses CV if not provided)
            y_val: Validation labels (optional)
        """
        log.info(f"Fitting weighted ensemble with metric: {self.metric}")
        
        performances = {}
        
        for name, model in self.models.items():
            log.info(f"Fitting and evaluating {name}...")
            
            # Fit model
            model.fit(X, y)
            
            # Get validation predictions
            if X_val is not None and y_val is not None:
                y_pred = model.predict(X_val)
                y_pred_proba = model.predict_proba(X_val)[:, 1] if hasattr(model, 'predict_proba') else y_pred
            else:
                # Use cross-validation
                from sklearn.model_selection import cross_val_predict
                y_pred_proba = cross_val_predict(
                    model, X, y, cv=3, method='predict_proba'
                )[:, 1]
                y_pred = (y_pred_proba >= 0.5).astype(int)
            
            # Calculate performance metric
            if self.metric == 'f1':
                from sklearn.metrics import f1_score
                score = f1_score(y_val if y_val is not None else y, y_pred)
            elif self.metric == 'pr_auc':
                from sklearn.metrics import precision_recall_curve, auc
                precision, recall, _ = precision_recall_curve(
                    y_val if y_val is not None else y, y_pred_proba
                )
                score = auc(recall, precision)
            else:
                raise ValueError(f"Unknown metric: {self.metric}")
            
            performances[name] = score
            log.info(f"{name} {self.metric}: {score:.4f}")
            
            # Calibrate model
            self.calibrators[name] = CalibratedClassifierCV(
                model, method='isotonic', cv='prefit'
            )
            self.calibrators[name].fit(X, y)
        
        # Calculate weights based on performance
        total_score = sum(performances.values())
        self.weights = {
            name: score / total_score 
            for name, score in performances.items()
        }
        
        log.info(f"Optimal weights: {self.weights}")
        self.classes_ = np.unique(y)
        
        return self
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict class probabilities using weighted average.
        
        Args:
            X: Features to predict
            
        Returns:
            Weighted average of class probabilities
        """
        weighted_proba = None
        
        for name, calibrator in self.calibrators.items():
            proba = calibrator.predict_proba(X)
            weight = self.weights[name]
            
            if weighted_proba is None:
                weighted_proba = proba * weight
            else:
                weighted_proba += proba * weight
        
        return weighted_proba
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict class labels.
        
        Args:
            X: Features to predict
            
        Returns:
            Array of predicted class labels
        """
        proba = self.predict_proba(X)
        return (proba[:, 1] >= 0.5).astype(int)


class StackingEnsemble(BaseEstimator, ClassifierMixin):
    """
    Stacking ensemble with a meta-learner trained on base model predictions.
    
    Uses cross-validation to generate out-of-fold predictions for training the meta-learner.
    """
    
    def __init__(self, base_models: Dict[str, BaseEstimator], 
                 meta_learner: Optional[BaseEstimator] = None,
                 use_probas: bool = True, cv_folds: int = 5):
        """
        Args:
            base_models: Dictionary of {name: model} pairs for base models
            meta_learner: Model to use as meta-learner (default: LogisticRegression)
            use_probas: If True, use predicted probabilities as meta-features
            cv_folds: Number of CV folds for generating meta-features
        """
        self.base_models = base_models
        self.meta_learner = meta_learner or LogisticRegression(random_state=42)
        self.use_probas = use_probas
        self.cv_folds = cv_folds
        self.fitted_base_models = {}
        self.calibrator = None
    
    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        Fit base models and meta-learner.
        
        Args:
            X: Training features
            y: Training labels
        """
        log.info(f"Fitting stacking ensemble with {len(self.base_models)} base models")
        
        # Generate meta-features using cross-validation
        meta_features = []
        
        for name, model in self.base_models.items():
            log.info(f"Generating meta-features for {name}...")
            
            if self.use_probas and hasattr(model, 'predict_proba'):
                # Get out-of-fold probability predictions
                oof_preds = cross_val_predict(
                    model, X, y, cv=self.cv_folds, 
                    method='predict_proba', n_jobs=-1
                )
                # For binary classification, use probability of positive class
                meta_features.append(oof_preds[:, 1])
            else:
                # Get out-of-fold class predictions
                oof_preds = cross_val_predict(
                    model, X, y, cv=self.cv_folds, n_jobs=-1
                )
                meta_features.append(oof_preds)
            
            # Fit model on full training data
            log.info(f"Fitting {name} on full training data...")
            model.fit(X, y)
            self.fitted_base_models[name] = model
        
        # Stack meta-features
        meta_X = np.column_stack(meta_features)
        log.info(f"Meta-features shape: {meta_X.shape}")
        
        # Train meta-learner
        log.info("Training meta-learner...")
        self.meta_learner.fit(meta_X, y)
        
        # Calibrate the entire stacking ensemble
        log.info("Calibrating stacking ensemble...")
        self.calibrator = CalibratedClassifierCV(
            self, method='isotonic', cv='prefit'
        )
        self.calibrator.fit(X, y)
        
        self.classes_ = np.unique(y)
        log.info("Stacking ensemble fitted successfully")
        
        return self
    
    def _get_meta_features(self, X: pd.DataFrame) -> np.ndarray:
        """
        Generate meta-features from base model predictions.
        
        Args:
            X: Features to predict
            
        Returns:
            Meta-features array
        """
        meta_features = []
        
        for name, model in self.fitted_base_models.items():
            if self.use_probas and hasattr(model, 'predict_proba'):
                preds = model.predict_proba(X)[:, 1]
            else:
                preds = model.predict(X)
            meta_features.append(preds)
        
        return np.column_stack(meta_features)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict class probabilities.
        
        Args:
            X: Features to predict
            
        Returns:
            Class probabilities from meta-learner
        """
        meta_X = self._get_meta_features(X)
        
        if hasattr(self.meta_learner, 'predict_proba'):
            return self.meta_learner.predict_proba(meta_X)
        else:
            # If meta-learner doesn't support probabilities, return binary
            preds = self.meta_learner.predict(meta_X)
            proba = np.zeros((len(preds), 2))
            proba[range(len(preds)), preds] = 1.0
            return proba
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict class labels.
        
        Args:
            X: Features to predict
            
        Returns:
            Predicted class labels
        """
        meta_X = self._get_meta_features(X)
        return self.meta_learner.predict(meta_X)


class EnsembleOptimizer:
    """
    Optimizer for finding the best ensemble configuration.
    
    Tests different ensemble methods and configurations to find optimal combination.
    """
    
    def __init__(self, base_models: Dict[str, BaseEstimator]):
        """
        Args:
            base_models: Dictionary of trained base models
        """
        self.base_models = base_models
        self.best_ensemble = None
        self.best_score = None
        self.results = {}
    
    def optimize(self, X_train: pd.DataFrame, y_train: pd.Series,
                 X_val: pd.DataFrame, y_val: pd.Series,
                 metrics: List[str] = ['f1', 'pr_auc']) -> Dict:
        """
        Find optimal ensemble configuration.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            metrics: List of metrics to evaluate
            
        Returns:
            Dictionary with results for each ensemble type
        """
        log.info("Starting ensemble optimization...")
        
        # Test Voting Ensemble (soft)
        log.info("Testing soft voting ensemble...")
        voting_soft = VotingEnsemble(self.base_models, voting='soft')
        voting_soft.fit(X_train, y_train)
        self._evaluate_ensemble('voting_soft', voting_soft, X_val, y_val, metrics)
        
        # Test Voting Ensemble (hard)
        log.info("Testing hard voting ensemble...")
        voting_hard = VotingEnsemble(self.base_models, voting='hard')
        voting_hard.fit(X_train, y_train)
        self._evaluate_ensemble('voting_hard', voting_hard, X_val, y_val, metrics)
        
        # Test Weighted Ensemble
        log.info("Testing weighted ensemble...")
        weighted = WeightedEnsemble(self.base_models, metric='f1')
        weighted.fit(X_train, y_train, X_val, y_val)
        self._evaluate_ensemble('weighted', weighted, X_val, y_val, metrics)
        
        # Test Stacking Ensemble
        log.info("Testing stacking ensemble...")
        stacking = StackingEnsemble(self.base_models, use_probas=True)
        stacking.fit(X_train, y_train)
        self._evaluate_ensemble('stacking', stacking, X_val, y_val, metrics)
        
        # Find best ensemble
        best_name = max(self.results, key=lambda k: self.results[k]['f1'])
        self.best_score = self.results[best_name]['f1']
        
        log.info(f"\nBest ensemble: {best_name} with F1={self.best_score:.4f}")
        
        # Train best ensemble on full data
        if best_name == 'voting_soft':
            self.best_ensemble = VotingEnsemble(self.base_models, voting='soft')
        elif best_name == 'voting_hard':
            self.best_ensemble = VotingEnsemble(self.base_models, voting='hard')
        elif best_name == 'weighted':
            self.best_ensemble = WeightedEnsemble(self.base_models, metric='f1')
        else:  # stacking
            self.best_ensemble = StackingEnsemble(self.base_models, use_probas=True)
        
        # Combine train and validation for final fit
        X_full = pd.concat([X_train, X_val])
        y_full = pd.concat([y_train, y_val])
        self.best_ensemble.fit(X_full, y_full)
        
        return self.results
    
    def _evaluate_ensemble(self, name: str, ensemble: BaseEstimator,
                          X_val: pd.DataFrame, y_val: pd.Series,
                          metrics: List[str]):
        """
        Evaluate an ensemble on validation data.
        
        Args:
            name: Name of the ensemble
            ensemble: Fitted ensemble model
            X_val: Validation features
            y_val: Validation labels
            metrics: List of metrics to calculate
        """
        from sklearn.metrics import (
            f1_score, precision_recall_curve, auc, 
            roc_auc_score, brier_score_loss
        )
        
        y_pred = ensemble.predict(X_val)
        y_proba = ensemble.predict_proba(X_val)[:, 1]
        
        results = {}
        
        if 'f1' in metrics:
            results['f1'] = f1_score(y_val, y_pred)
        
        if 'pr_auc' in metrics:
            precision, recall, _ = precision_recall_curve(y_val, y_proba)
            results['pr_auc'] = auc(recall, precision)
        
        if 'roc_auc' in metrics:
            results['roc_auc'] = roc_auc_score(y_val, y_proba)
        
        if 'brier' in metrics:
            results['brier'] = brier_score_loss(y_val, y_proba)
        
        self.results[name] = results
        
        log.info(f"{name} results: {results}")
    
    def save_best_ensemble(self, path: str):
        """
        Save the best ensemble to disk.
        
        Args:
            path: Path to save the ensemble
        """
        if self.best_ensemble is None:
            raise ValueError("No ensemble optimized yet")
        
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'wb') as f:
            pickle.dump({
                'ensemble': self.best_ensemble,
                'score': self.best_score,
                'results': self.results
            }, f)
        
        log.info(f"Best ensemble saved to {path}")
    
    @staticmethod
    def load_ensemble(path: str) -> Tuple[BaseEstimator, float, Dict]:
        """
        Load a saved ensemble from disk.
        
        Args:
            path: Path to the saved ensemble
            
        Returns:
            Tuple of (ensemble, best_score, results)
        """
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        return data['ensemble'], data['score'], data['results']