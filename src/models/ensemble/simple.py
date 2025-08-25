"""
Simple but effective ensemble of XGBoost and LSTM.
Weighted average with dynamic weights based on recent performance.
"""
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple
import logging
from pathlib import Path
import joblib

logger = logging.getLogger(__name__)


class SimpleEnsemble:
    """
    Simple weighted ensemble of models.
    
    Key features:
    - Weighted average of probabilities
    - Dynamic weight adjustment based on recent performance
    - Optional voting for high-precision mode
    - Automatic calibration of ensemble output
    """
    
    def __init__(self,
                 models: Optional[Dict[str, Any]] = None,
                 weights: Optional[Dict[str, float]] = None,
                 weight_decay: float = 0.95,
                 min_weight: float = 0.1,
                 max_weight: float = 0.9,
                 use_voting: bool = False,
                 voting_threshold: float = 0.7):
        """
        Initialize ensemble.
        
        Args:
            models: Dictionary of model_name -> model instance
            weights: Initial weights for each model
            weight_decay: Decay factor for EWMA of performance
            min_weight: Minimum weight for any model
            max_weight: Maximum weight for any model
            use_voting: If True, require agreement for positive signals
            voting_threshold: Threshold for voting consensus
        """
        self.models = models or {}
        self.weights = weights or {}
        self.weight_decay = weight_decay
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.use_voting = use_voting
        self.voting_threshold = voting_threshold
        
        # Performance tracking
        self.performance_history = {name: [] for name in self.models}
        self.dynamic_weights = self.weights.copy()
        
        # Ensemble calibration
        self.calibrator = None
        self.threshold = 0.5
        
        # Initialize equal weights if not provided
        if self.models and not self.weights:
            n_models = len(self.models)
            equal_weight = 1.0 / n_models
            self.weights = {name: equal_weight for name in self.models}
            self.dynamic_weights = self.weights.copy()
    
    def add_model(self, name: str, model: Any, weight: float = None):
        """
        Add a model to the ensemble.
        
        Args:
            name: Model identifier
            model: Model instance with predict_proba method
            weight: Initial weight (default: equal weight)
        """
        self.models[name] = model
        self.performance_history[name] = []
        
        if weight is None:
            # Recalculate equal weights
            n_models = len(self.models)
            equal_weight = 1.0 / n_models
            for model_name in self.models:
                self.weights[model_name] = equal_weight
                self.dynamic_weights[model_name] = equal_weight
        else:
            self.weights[name] = weight
            self.dynamic_weights[name] = weight
            # Renormalize
            self._normalize_weights()
        
        logger.info(f"Added model '{name}' with weight {self.dynamic_weights[name]:.3f}")
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict probabilities using ensemble.
        
        Args:
            X: Feature DataFrame
            
        Returns:
            Array of shape (n_samples, 2) with class probabilities
        """
        if not self.models:
            raise ValueError("No models in ensemble")
        
        # Collect predictions from all models
        predictions = {}
        for name, model in self.models.items():
            try:
                pred = model.predict_proba(X)
                if pred.ndim == 1:
                    # Convert to 2D if needed
                    pred = np.column_stack([1 - pred, pred])
                predictions[name] = pred
            except Exception as e:
                logger.warning(f"Model '{name}' prediction failed: {e}")
                continue
        
        if not predictions:
            raise ValueError("All model predictions failed")
        
        # Apply ensemble strategy
        if self.use_voting:
            ensemble_proba = self._voting_ensemble(predictions)
        else:
            ensemble_proba = self._weighted_average_ensemble(predictions)
        
        # Apply calibration if available
        if self.calibrator is not None:
            ensemble_proba[:, 1] = self._apply_calibration(ensemble_proba[:, 1])
            ensemble_proba[:, 0] = 1 - ensemble_proba[:, 1]
        
        return ensemble_proba
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict binary labels.
        
        Args:
            X: Feature DataFrame
            
        Returns:
            Binary predictions
        """
        proba = self.predict_proba(X)
        return (proba[:, 1] >= self.threshold).astype(int)
    
    def _weighted_average_ensemble(self, predictions: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Weighted average of model predictions.
        
        Args:
            predictions: Dictionary of model predictions
            
        Returns:
            Ensemble probabilities
        """
        # Initialize ensemble prediction
        n_samples = next(iter(predictions.values())).shape[0]
        ensemble_proba = np.zeros((n_samples, 2))
        
        # Weighted sum
        total_weight = 0
        for name, pred in predictions.items():
            weight = self.dynamic_weights.get(name, self.weights.get(name, 0))
            ensemble_proba += weight * pred
            total_weight += weight
        
        # Normalize
        if total_weight > 0:
            ensemble_proba /= total_weight
        
        return ensemble_proba
    
    def _voting_ensemble(self, predictions: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Voting ensemble - requires consensus for positive signals.
        
        Args:
            predictions: Dictionary of model predictions
            
        Returns:
            Ensemble probabilities
        """
        # Get weighted average as base
        ensemble_proba = self._weighted_average_ensemble(predictions)
        
        # Count positive votes
        n_samples = ensemble_proba.shape[0]
        positive_votes = np.zeros(n_samples)
        total_weight = 0
        
        for name, pred in predictions.items():
            weight = self.dynamic_weights.get(name, self.weights.get(name, 0))
            positive_votes += weight * (pred[:, 1] >= 0.5)
            total_weight += weight
        
        if total_weight > 0:
            positive_votes /= total_weight
        
        # Apply voting threshold
        # If not enough models agree, set probability to low value
        mask = positive_votes < self.voting_threshold
        ensemble_proba[mask, 1] = ensemble_proba[mask, 1] * 0.3  # Reduce confidence
        ensemble_proba[mask, 0] = 1 - ensemble_proba[mask, 1]
        
        return ensemble_proba
    
    def update_weights(self, 
                      y_true: np.ndarray,
                      predictions: Dict[str, np.ndarray]):
        """
        Update model weights based on recent performance.
        
        Args:
            y_true: True labels
            predictions: Dictionary of model predictions
        """
        from sklearn.metrics import log_loss
        
        # Calculate performance for each model
        performances = {}
        for name, pred in predictions.items():
            try:
                # Use log loss (lower is better)
                loss = log_loss(y_true, pred[:, 1])
                # Convert to performance score (higher is better)
                performance = 1.0 / (1.0 + loss)
                performances[name] = performance
                
                # Update history with EWMA
                if self.performance_history[name]:
                    ewma = (self.weight_decay * self.performance_history[name][-1] + 
                           (1 - self.weight_decay) * performance)
                else:
                    ewma = performance
                
                self.performance_history[name].append(ewma)
                
            except Exception as e:
                logger.warning(f"Failed to update weight for '{name}': {e}")
        
        # Update weights based on relative performance
        if performances:
            # Calculate new weights proportional to performance
            total_perf = sum(performances.values())
            
            for name in self.models:
                if name in performances:
                    new_weight = performances[name] / total_perf
                    # Apply min/max constraints
                    new_weight = np.clip(new_weight, self.min_weight, self.max_weight)
                    self.dynamic_weights[name] = new_weight
            
            # Renormalize
            self._normalize_weights()
            
            logger.info("Updated ensemble weights: " + 
                       ", ".join([f"{n}: {w:.3f}" for n, w in self.dynamic_weights.items()]))
    
    def _normalize_weights(self):
        """Normalize weights to sum to 1."""
        total = sum(self.dynamic_weights.values())
        if total > 0:
            for name in self.dynamic_weights:
                self.dynamic_weights[name] /= total
    
    def calibrate(self, X: pd.DataFrame, y: pd.Series, method: str = 'isotonic'):
        """
        Calibrate ensemble predictions.
        
        Args:
            X: Calibration features
            y: Calibration labels
            method: 'isotonic' or 'sigmoid'
        """
        from sklearn.isotonic import IsotonicRegression
        from sklearn.linear_model import LogisticRegression
        
        # Get ensemble predictions
        proba = self.predict_proba(X)[:, 1]
        
        if method == 'isotonic':
            self.calibrator = IsotonicRegression(out_of_bounds='clip')
            self.calibrator.fit(proba, y)
        elif method == 'sigmoid':
            self.calibrator = LogisticRegression()
            self.calibrator.fit(proba.reshape(-1, 1), y)
        else:
            raise ValueError(f"Unknown calibration method: {method}")
        
        logger.info(f"Ensemble calibrated using {method} method")
    
    def _apply_calibration(self, probas: np.ndarray) -> np.ndarray:
        """Apply calibration to probabilities."""
        if self.calibrator is None:
            return probas
        
        if hasattr(self.calibrator, 'predict_proba'):
            # Platt scaling
            return self.calibrator.predict_proba(probas.reshape(-1, 1))[:, 1]
        else:
            # Isotonic regression
            return self.calibrator.transform(probas)
    
    def tune_threshold(self, y_true: pd.Series, y_proba: np.ndarray,
                      costs: Dict[str, float]) -> float:
        """
        Tune decision threshold for optimal expected value.
        
        Args:
            y_true: True labels
            y_proba: Predicted probabilities
            costs: Cost structure
            
        Returns:
            Optimal threshold
        """
        from src.metrics.core import find_optimal_threshold_by_metric
        
        threshold, best_ev = find_optimal_threshold_by_metric(
            y_true.values, y_proba, metric='ev', costs=costs
        )
        
        self.threshold = threshold
        logger.info(f"Optimal ensemble threshold: {threshold:.4f} (EV: {best_ev:.4f})")
        
        return threshold
    
    def save(self, path: str):
        """Save ensemble configuration."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        ensemble_data = {
            'weights': self.weights,
            'dynamic_weights': self.dynamic_weights,
            'performance_history': self.performance_history,
            'threshold': self.threshold,
            'config': {
                'weight_decay': self.weight_decay,
                'min_weight': self.min_weight,
                'max_weight': self.max_weight,
                'use_voting': self.use_voting,
                'voting_threshold': self.voting_threshold
            }
        }
        
        joblib.dump(ensemble_data, path)
        logger.info(f"Ensemble saved to {path}")
    
    def load(self, path: str):
        """Load ensemble configuration."""
        ensemble_data = joblib.load(path)
        
        self.weights = ensemble_data['weights']
        self.dynamic_weights = ensemble_data['dynamic_weights']
        self.performance_history = ensemble_data['performance_history']
        self.threshold = ensemble_data['threshold']
        
        config = ensemble_data['config']
        self.weight_decay = config['weight_decay']
        self.min_weight = config['min_weight']
        self.max_weight = config['max_weight']
        self.use_voting = config['use_voting']
        self.voting_threshold = config['voting_threshold']
        
        logger.info(f"Ensemble loaded from {path}")
    
    def get_weights_summary(self) -> pd.DataFrame:
        """
        Get summary of model weights.
        
        Returns:
            DataFrame with weight information
        """
        data = []
        for name in self.models:
            data.append({
                'model': name,
                'initial_weight': self.weights.get(name, 0),
                'current_weight': self.dynamic_weights.get(name, 0),
                'performance_ewma': self.performance_history[name][-1] 
                                  if self.performance_history[name] else None
            })
        
        return pd.DataFrame(data)


def create_xgb_lstm_ensemble(xgb_model, lstm_model,
                            xgb_weight: float = 0.6,
                            lstm_weight: float = 0.4) -> SimpleEnsemble:
    """
    Create a simple XGBoost + LSTM ensemble.
    
    Args:
        xgb_model: Trained XGBoost model
        lstm_model: Trained LSTM model
        xgb_weight: Initial weight for XGBoost
        lstm_weight: Initial weight for LSTM
        
    Returns:
        Configured ensemble
    """
    ensemble = SimpleEnsemble(
        weight_decay=0.95,
        min_weight=0.2,
        max_weight=0.8,
        use_voting=False  # Start with weighted average
    )
    
    ensemble.add_model('xgboost', xgb_model, xgb_weight)
    ensemble.add_model('lstm', lstm_model, lstm_weight)
    
    logger.info(f"Created XGB+LSTM ensemble (weights: XGB={xgb_weight}, LSTM={lstm_weight})")
    
    return ensemble