"""
Temperature scaling for neural network calibration.

Based on "On Calibration of Modern Neural Networks" (Guo et al., 2017)
https://arxiv.org/abs/1706.04599

Temperature scaling is a simple post-processing calibration method that
learns a single scalar parameter T on the validation set.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Optional, Tuple, Union
from sklearn.base import BaseEstimator
import warnings


class TemperatureScaling(BaseEstimator):
    """
    Temperature scaling calibration for neural networks.
    
    This method learns a single temperature parameter T that rescales
    the logits: calibrated_logits = logits / T
    
    The temperature is optimized on a validation set to minimize NLL.
    """
    
    def __init__(
        self,
        max_iter: int = 100,
        lr: float = 0.01,
        tol: float = 1e-4,
        device: str = 'cpu',
        verbose: bool = False
    ):
        """
        Initialize temperature scaling calibrator.
        
        Args:
            max_iter: Maximum optimization iterations
            lr: Learning rate for temperature optimization
            tol: Convergence tolerance
            device: Device for computation ('cpu', 'cuda')
            verbose: Whether to print optimization progress
        """
        self.max_iter = max_iter
        self.lr = lr
        self.tol = tol
        self.device = device
        self.verbose = verbose
        self.temperature_ = 1.0
        self.converged_ = False
        self.n_iter_ = 0
        self.loss_history_ = []
        
    def fit(self, logits: np.ndarray, y_true: np.ndarray) -> 'TemperatureScaling':
        """
        Learn temperature parameter on validation data.
        
        Args:
            logits: Raw model outputs (before sigmoid/softmax) of shape (n_samples,) or (n_samples, n_classes)
            y_true: True labels of shape (n_samples,)
            
        Returns:
            Self
        """
        # Convert to tensors
        if isinstance(logits, np.ndarray):
            logits = torch.FloatTensor(logits)
        if isinstance(y_true, np.ndarray):
            y_true = torch.LongTensor(y_true) if logits.dim() > 1 else torch.FloatTensor(y_true)
            
        logits = logits.to(self.device)
        y_true = y_true.to(self.device)
        
        # Initialize temperature as learnable parameter
        temperature = nn.Parameter(torch.ones(1, device=self.device))
        
        # Optimizer
        optimizer = optim.LBFGS([temperature], lr=self.lr, max_iter=self.max_iter)
        
        # Loss function
        if logits.dim() == 1 or (logits.dim() == 2 and logits.shape[1] == 1):
            # Binary classification
            logits = logits.squeeze()
            criterion = nn.BCEWithLogitsLoss()
        else:
            # Multi-class classification
            criterion = nn.CrossEntropyLoss()
        
        # Optimization
        def closure():
            optimizer.zero_grad()
            # Apply temperature scaling
            scaled_logits = logits / temperature
            
            # Calculate loss
            if logits.dim() == 1:
                loss = criterion(scaled_logits, y_true)
            else:
                loss = criterion(scaled_logits, y_true)
                
            loss.backward()
            self.loss_history_.append(loss.item())
            return loss
        
        # Initial loss
        with torch.no_grad():
            initial_loss = criterion(logits, y_true).item()
            self.loss_history_.append(initial_loss)
        
        # Optimize
        prev_loss = initial_loss
        for i in range(self.max_iter):
            optimizer.step(closure)
            current_loss = self.loss_history_[-1]
            
            if self.verbose and i % 10 == 0:
                print(f"Iter {i}: Loss = {current_loss:.4f}, T = {temperature.item():.4f}")
            
            # Check convergence
            if abs(prev_loss - current_loss) < self.tol:
                self.converged_ = True
                self.n_iter_ = i + 1
                break
                
            prev_loss = current_loss
        
        # Store learned temperature
        self.temperature_ = temperature.item()
        
        if self.verbose:
            print(f"Optimization finished. Final T = {self.temperature_:.4f}")
            print(f"Initial NLL: {initial_loss:.4f}, Final NLL: {self.loss_history_[-1]:.4f}")
        
        return self
    
    def transform(self, logits: np.ndarray) -> np.ndarray:
        """
        Apply temperature scaling to logits.
        
        Args:
            logits: Raw model outputs of shape (n_samples,) or (n_samples, n_classes)
            
        Returns:
            Calibrated probabilities
        """
        # Apply temperature scaling
        scaled_logits = logits / self.temperature_
        
        # Convert to probabilities
        if scaled_logits.ndim == 1 or (scaled_logits.ndim == 2 and scaled_logits.shape[1] == 1):
            # Binary classification
            probs = 1 / (1 + np.exp(-scaled_logits.squeeze()))
        else:
            # Multi-class classification
            exp_logits = np.exp(scaled_logits - np.max(scaled_logits, axis=1, keepdims=True))
            probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        
        return probs
    
    def fit_transform(self, logits: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        """Fit and transform in one step."""
        return self.fit(logits, y_true).transform(logits)
    
    def predict_proba(self, logits: np.ndarray) -> np.ndarray:
        """
        Sklearn-compatible probability prediction.
        
        Args:
            logits: Raw model outputs
            
        Returns:
            Calibrated probabilities
        """
        probs = self.transform(logits)
        
        # Ensure 2D output for sklearn compatibility
        if probs.ndim == 1:
            probs = np.column_stack([1 - probs, probs])
            
        return probs


class VectorScaling(BaseEstimator):
    """
    Vector scaling (extension of temperature scaling).
    
    Instead of a single temperature, learns a vector of weights
    and biases for calibration.
    """
    
    def __init__(
        self,
        max_iter: int = 100,
        lr: float = 0.01,
        tol: float = 1e-4,
        regularization: float = 0.0,
        device: str = 'cpu',
        verbose: bool = False
    ):
        """
        Initialize vector scaling calibrator.
        
        Args:
            max_iter: Maximum optimization iterations
            lr: Learning rate
            tol: Convergence tolerance
            regularization: L2 regularization strength
            device: Device for computation
            verbose: Whether to print progress
        """
        self.max_iter = max_iter
        self.lr = lr
        self.tol = tol
        self.regularization = regularization
        self.device = device
        self.verbose = verbose
        self.weights_ = None
        self.bias_ = None
        self.converged_ = False
        
    def fit(self, logits: np.ndarray, y_true: np.ndarray) -> 'VectorScaling':
        """
        Learn scaling weights and biases.
        
        Args:
            logits: Raw model outputs
            y_true: True labels
            
        Returns:
            Self
        """
        # Convert to tensors
        if isinstance(logits, np.ndarray):
            logits = torch.FloatTensor(logits)
        if isinstance(y_true, np.ndarray):
            y_true = torch.LongTensor(y_true) if logits.dim() > 1 else torch.FloatTensor(y_true)
            
        logits = logits.to(self.device)
        y_true = y_true.to(self.device)
        
        # Initialize parameters
        n_features = 1 if logits.dim() == 1 else logits.shape[1]
        weights = nn.Parameter(torch.ones(n_features, device=self.device))
        bias = nn.Parameter(torch.zeros(n_features, device=self.device))
        
        # Optimizer
        optimizer = optim.Adam([weights, bias], lr=self.lr)
        
        # Loss function
        if logits.dim() == 1:
            criterion = nn.BCEWithLogitsLoss()
        else:
            criterion = nn.CrossEntropyLoss()
        
        # Training loop
        prev_loss = float('inf')
        for i in range(self.max_iter):
            optimizer.zero_grad()
            
            # Apply vector scaling
            scaled_logits = logits * weights + bias
            
            # Calculate loss
            loss = criterion(scaled_logits, y_true)
            
            # Add regularization
            if self.regularization > 0:
                reg_loss = self.regularization * (torch.sum(weights**2) + torch.sum(bias**2))
                loss = loss + reg_loss
            
            loss.backward()
            optimizer.step()
            
            current_loss = loss.item()
            
            if self.verbose and i % 10 == 0:
                print(f"Iter {i}: Loss = {current_loss:.4f}")
            
            # Check convergence
            if abs(prev_loss - current_loss) < self.tol:
                self.converged_ = True
                break
                
            prev_loss = current_loss
        
        # Store learned parameters
        self.weights_ = weights.detach().cpu().numpy()
        self.bias_ = bias.detach().cpu().numpy()
        
        return self
    
    def transform(self, logits: np.ndarray) -> np.ndarray:
        """Apply vector scaling."""
        if self.weights_ is None or self.bias_ is None:
            raise ValueError("Model not fitted yet")
        
        # Apply scaling
        scaled_logits = logits * self.weights_ + self.bias_
        
        # Convert to probabilities
        if scaled_logits.ndim == 1:
            probs = 1 / (1 + np.exp(-scaled_logits))
        else:
            exp_logits = np.exp(scaled_logits - np.max(scaled_logits, axis=1, keepdims=True))
            probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        
        return probs


class EnsembleTemperatureScaling(BaseEstimator):
    """
    Temperature scaling for ensemble models.
    
    Learns a temperature parameter for each model in the ensemble.
    """
    
    def __init__(
        self,
        n_models: int,
        max_iter: int = 100,
        lr: float = 0.01,
        device: str = 'cpu',
        verbose: bool = False
    ):
        """
        Initialize ensemble temperature scaling.
        
        Args:
            n_models: Number of models in ensemble
            max_iter: Maximum optimization iterations
            lr: Learning rate
            device: Device for computation
            verbose: Whether to print progress
        """
        self.n_models = n_models
        self.max_iter = max_iter
        self.lr = lr
        self.device = device
        self.verbose = verbose
        self.temperatures_ = np.ones(n_models)
        
    def fit(self, logits_list: list, y_true: np.ndarray) -> 'EnsembleTemperatureScaling':
        """
        Learn temperature for each model.
        
        Args:
            logits_list: List of logits from each model
            y_true: True labels
            
        Returns:
            Self
        """
        if len(logits_list) != self.n_models:
            raise ValueError(f"Expected {self.n_models} models, got {len(logits_list)}")
        
        # Fit temperature for each model
        for i, logits in enumerate(logits_list):
            ts = TemperatureScaling(
                max_iter=self.max_iter,
                lr=self.lr,
                device=self.device,
                verbose=False
            )
            ts.fit(logits, y_true)
            self.temperatures_[i] = ts.temperature_
            
            if self.verbose:
                print(f"Model {i}: T = {ts.temperature_:.4f}")
        
        return self
    
    def transform(self, logits_list: list) -> np.ndarray:
        """
        Apply temperature scaling to ensemble.
        
        Args:
            logits_list: List of logits from each model
            
        Returns:
            Averaged calibrated probabilities
        """
        if len(logits_list) != self.n_models:
            raise ValueError(f"Expected {self.n_models} models, got {len(logits_list)}")
        
        calibrated_probs = []
        for i, logits in enumerate(logits_list):
            scaled_logits = logits / self.temperatures_[i]
            
            # Convert to probabilities
            if scaled_logits.ndim == 1:
                probs = 1 / (1 + np.exp(-scaled_logits))
            else:
                exp_logits = np.exp(scaled_logits - np.max(scaled_logits, axis=1, keepdims=True))
                probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
            
            calibrated_probs.append(probs)
        
        # Average probabilities
        return np.mean(calibrated_probs, axis=0)


def compare_calibration_methods(
    logits: np.ndarray,
    y_true: np.ndarray,
    methods: list = ['temperature', 'vector', 'isotonic', 'platt']
) -> dict:
    """
    Compare different calibration methods.
    
    Args:
        logits: Raw model outputs
        y_true: True labels
        methods: List of methods to compare
        
    Returns:
        Dictionary with calibrated probabilities for each method
    """
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.isotonic import IsotonicRegression
    from sklearn.linear_model import LogisticRegression
    
    results = {}
    
    # Uncalibrated probabilities
    if logits.ndim == 1:
        results['uncalibrated'] = 1 / (1 + np.exp(-logits))
    else:
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        results['uncalibrated'] = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
    
    # Temperature scaling
    if 'temperature' in methods:
        ts = TemperatureScaling()
        ts.fit(logits, y_true)
        results['temperature'] = ts.transform(logits)
        results['temperature_T'] = ts.temperature_
    
    # Vector scaling
    if 'vector' in methods:
        vs = VectorScaling()
        vs.fit(logits, y_true)
        results['vector'] = vs.transform(logits)
    
    # Isotonic regression
    if 'isotonic' in methods:
        uncal_probs = results['uncalibrated']
        if uncal_probs.ndim > 1:
            uncal_probs = uncal_probs[:, 1]
        iso = IsotonicRegression(out_of_bounds='clip')
        iso.fit(uncal_probs, y_true)
        results['isotonic'] = iso.transform(uncal_probs)
    
    # Platt scaling
    if 'platt' in methods:
        uncal_probs = results['uncalibrated']
        if uncal_probs.ndim > 1:
            uncal_probs = uncal_probs[:, 1]
        platt = LogisticRegression()
        platt.fit(uncal_probs.reshape(-1, 1), y_true)
        results['platt'] = platt.predict_proba(uncal_probs.reshape(-1, 1))[:, 1]
    
    return results