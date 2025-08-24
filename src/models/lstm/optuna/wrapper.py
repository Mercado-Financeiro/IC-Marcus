"""Scikit-learn compatible wrapper for LSTM models."""

import numpy as np
import pandas as pd
import torch
from typing import Optional, Union
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.exceptions import NotFittedError


class LSTMWrapper(BaseEstimator, ClassifierMixin):
    """Scikit-learn compatible wrapper for LSTM model."""
    
    def __init__(
        self,
        lstm_model: torch.nn.Module,
        seq_len: int,
        device: torch.device,
        scaler: Optional[object] = None,
        stride: int = 1,
    ):
        """
        Initialize LSTM wrapper.
        
        Args:
            lstm_model: Trained LSTM model
            seq_len: Sequence length for input
            device: Device to run model on
            scaler: Optional scaler for features
        """
        self.lstm_model = lstm_model
        self.seq_len = seq_len
        self.device = device
        self.scaler = scaler
        self.stride = max(1, int(stride))
        
        # Required sklearn classifier attributes
        self._estimator_type = "classifier"
        self.classes_ = np.array([0, 1])
        self.n_features_in_ = None
        self.n_classes_ = 2
        
    def fit(self, X, y):
        """
        Fit method (model is already trained).
        
        Args:
            X: Training features
            y: Training labels
            
        Returns:
            self
        """
        # Model is already trained, but we need to set sklearn attributes
        if hasattr(X, 'shape'):
            self.n_features_in_ = X.shape[1]
        elif hasattr(X, '__len__'):
            self.n_features_in_ = len(X.columns) if hasattr(X, 'columns') else None
        return self
    
    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Predict probabilities.
        
        Args:
            X: Features to predict
            
        Returns:
            Array of shape (n_samples, 2) with class probabilities
        """
        self._check_is_fitted()
        self.lstm_model.eval()
        
        # Convert to sequences
        X_seq = self._create_sequences(X)
        
        # Scale if needed
        if self.scaler is not None:
            X_seq_reshaped = X_seq.reshape(-1, X_seq.shape[-1])
            X_seq_reshaped = self.scaler.transform(X_seq_reshaped)
            X_seq = X_seq_reshaped.reshape(X_seq.shape)
        
        # Convert to tensor
        X_tensor = torch.FloatTensor(X_seq).to(self.device)
        
        # Predict
        with torch.no_grad():
            proba = self.lstm_model(X_tensor).cpu().numpy()
        
        # Flatten if needed
        if proba.ndim > 1:
            proba = proba.flatten()
        
        # Return probabilities for both classes (binary classification)
        proba_both = np.column_stack([1 - proba, proba])
        return proba_both
    
    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Predict classes.
        
        Args:
            X: Features to predict
            
        Returns:
            Array of predicted classes
        """
        proba = self.predict_proba(X)
        return (proba[:, 1] >= 0.5).astype(int)
    
    def decision_function(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Decision function for binary classification.
        
        Args:
            X: Features to predict
            
        Returns:
            Array of decision values (log odds)
        """
        # Return log odds of positive class
        proba = self.predict_proba(X)[:, 1]
        # Convert to log odds for decision function
        proba = np.clip(proba, 1e-10, 1 - 1e-10)  # Avoid log(0)
        return np.log(proba / (1 - proba)).astype(np.float64)
    
    def _check_is_fitted(self):
        """Check if model is fitted."""
        if self.lstm_model is None:
            raise NotFittedError("Model is not fitted yet.")
        return True
    
    def _create_sequences(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Create sequences from input data.
        
        Args:
            X: Input features
            
        Returns:
            Array of sequences
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        # If we don't have enough data for full sequences,
        # pad with the first values
        if len(X) < self.seq_len:
            padding = np.tile(X[0], (self.seq_len - len(X), 1))
            X = np.vstack([padding, X])
        
        # Create sequences with stride
        sequences = []
        for i in range(0, len(X) - self.seq_len + 1, self.stride):
            sequences.append(X[i : i + self.seq_len])

        return np.array(sequences)
    
    def score(self, X: Union[pd.DataFrame, np.ndarray], y: np.ndarray) -> float:
        """
        Calculate accuracy score.
        
        Args:
            X: Features
            y: True labels
            
        Returns:
            Accuracy score
        """
        y_pred = self.predict(X)
        # Adjust y to match the size of predictions (due to sequencing)
        if len(y_pred) < len(y):
            y = y[-len(y_pred):]
        elif len(y_pred) > len(y):
            y_pred = y_pred[:len(y)]
        return np.mean(y_pred == y)
