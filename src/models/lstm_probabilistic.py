"""
Probabilistic LSTM with NLL loss for uncertainty estimation.

Implements LSTM that outputs distribution parameters (mean, variance) 
and uses Negative Log-Likelihood loss for training.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, Any, List, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.distributions import Normal
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import StandardScaler
import optuna
from optuna.pruners import MedianPruner, HyperbandPruner
import warnings
from pathlib import Path
import joblib
import structlog

warnings.filterwarnings('ignore')
log = structlog.get_logger()


class ProbabilisticLSTM(nn.Module):
    """LSTM that outputs distribution parameters for probabilistic predictions."""
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        output_size: int = 1,
        min_std: float = 1e-3
    ):
        """
        Initialize Probabilistic LSTM.
        
        Args:
            input_size: Number of input features
            hidden_size: Number of hidden units
            num_layers: Number of LSTM layers
            dropout: Dropout rate
            output_size: Number of output dimensions
            min_std: Minimum standard deviation to avoid numerical issues
        """
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.min_std = min_std
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=False
        )
        
        # Attention mechanism (optional)
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=4,
            dropout=dropout,
            batch_first=True
        )
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Output layers for mean and log_std
        self.fc_mean = nn.Linear(hidden_size, output_size)
        self.fc_log_std = nn.Linear(hidden_size, output_size)
        
        # Optional: Layer normalization
        self.layer_norm = nn.LayerNorm(hidden_size)
        
    def forward(self, x, return_distribution=False):
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_size)
            return_distribution: If True, return Normal distribution object
            
        Returns:
            If return_distribution=False: tuple of (mean, std)
            If return_distribution=True: Normal distribution
        """
        # LSTM forward pass
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Apply attention
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        
        # Combine LSTM and attention outputs
        combined = lstm_out + attn_out
        
        # Layer normalization
        combined = self.layer_norm(combined)
        
        # Use last timestep
        last_hidden = combined[:, -1, :]
        
        # Apply dropout
        out = self.dropout(last_hidden)
        
        # Get mean and log_std
        mean = self.fc_mean(out)
        log_std = self.fc_log_std(out)
        
        # Convert log_std to std with minimum value
        std = torch.exp(log_std) + self.min_std
        
        if return_distribution:
            return Normal(mean, std)
        else:
            return mean, std


class NegativeLogLikelihood(nn.Module):
    """Negative Log-Likelihood loss for probabilistic predictions."""
    
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction
        
    def forward(self, mean, std, target):
        """
        Calculate NLL loss.
        
        Args:
            mean: Predicted mean
            std: Predicted standard deviation
            target: True values
            
        Returns:
            NLL loss
        """
        dist = Normal(mean, std)
        nll = -dist.log_prob(target)
        
        if self.reduction == 'mean':
            return nll.mean()
        elif self.reduction == 'sum':
            return nll.sum()
        else:
            return nll


class QuantileLoss(nn.Module):
    """Quantile loss for quantile regression."""
    
    def __init__(self, quantiles: List[float] = [0.1, 0.5, 0.9]):
        super().__init__()
        self.quantiles = quantiles
        
    def forward(self, predictions, target):
        """
        Calculate quantile loss.
        
        Args:
            predictions: Predicted quantiles (batch_size, n_quantiles)
            target: True values (batch_size, 1)
            
        Returns:
            Quantile loss
        """
        losses = []
        for i, q in enumerate(self.quantiles):
            errors = target - predictions[:, i:i+1]
            losses.append(torch.max((q - 1) * errors, q * errors))
        
        return torch.mean(torch.sum(torch.cat(losses, dim=1), dim=1))


class LSTMProbabilistic(BaseEstimator, RegressorMixin):
    """Scikit-learn compatible wrapper for Probabilistic LSTM."""
    
    def __init__(
        self,
        seq_len: int = 60,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        learning_rate: float = 1e-3,
        batch_size: int = 32,
        epochs: int = 100,
        patience: int = 10,
        min_std: float = 1e-3,
        mc_dropout_samples: int = 10,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        random_state: int = 42,
        verbose: bool = True
    ):
        """
        Initialize Probabilistic LSTM wrapper.
        
        Args:
            seq_len: Sequence length for LSTM
            hidden_size: Number of hidden units
            num_layers: Number of LSTM layers
            dropout: Dropout rate
            learning_rate: Learning rate
            batch_size: Batch size
            epochs: Maximum epochs
            patience: Early stopping patience
            min_std: Minimum standard deviation
            mc_dropout_samples: Number of MC dropout samples for uncertainty
            device: Device to use (cuda/cpu)
            random_state: Random seed
            verbose: Verbosity
        """
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.min_std = min_std
        self.mc_dropout_samples = mc_dropout_samples
        self.device = device
        self.random_state = random_state
        self.verbose = verbose
        
        self.model_ = None
        self.scaler_ = None
        self.history_ = {'train_loss': [], 'val_loss': []}
        
        # Set seeds
        torch.manual_seed(random_state)
        np.random.seed(random_state)
        
    def _create_sequences(self, X: np.ndarray, y: Optional[np.ndarray] = None):
        """Create sequences for LSTM input."""
        n_samples = len(X) - self.seq_len + 1
        
        if n_samples <= 0:
            raise ValueError(f"Not enough samples for sequence length {self.seq_len}")
        
        X_seq = np.zeros((n_samples, self.seq_len, X.shape[1]))
        
        for i in range(n_samples):
            X_seq[i] = X[i:i+self.seq_len]
        
        if y is not None:
            y_seq = y[self.seq_len-1:]
            return X_seq, y_seq
        
        return X_seq
    
    def fit(self, X, y, X_val=None, y_val=None):
        """
        Fit the probabilistic LSTM.
        
        Args:
            X: Training features
            y: Training targets
            X_val: Validation features (optional)
            y_val: Validation targets (optional)
            
        Returns:
            self
        """
        # Convert to numpy if needed
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values
        
        # Scale features
        self.scaler_ = StandardScaler()
        X_scaled = self.scaler_.fit_transform(X)
        
        # Create sequences
        X_seq, y_seq = self._create_sequences(X_scaled, y)
        
        # Reshape y if needed
        if len(y_seq.shape) == 1:
            y_seq = y_seq.reshape(-1, 1)
        
        # Create validation sequences if provided
        if X_val is not None and y_val is not None:
            if isinstance(X_val, pd.DataFrame):
                X_val = X_val.values
            if isinstance(y_val, pd.Series):
                y_val = y_val.values
            
            X_val_scaled = self.scaler_.transform(X_val)
            X_val_seq, y_val_seq = self._create_sequences(X_val_scaled, y_val)
            
            if len(y_val_seq.shape) == 1:
                y_val_seq = y_val_seq.reshape(-1, 1)
        
        # Create model
        input_size = X_seq.shape[2]
        output_size = y_seq.shape[1]
        
        self.model_ = ProbabilisticLSTM(
            input_size=input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
            output_size=output_size,
            min_std=self.min_std
        ).to(self.device)
        
        # Loss and optimizer
        criterion = NegativeLogLikelihood()
        optimizer = optim.Adam(self.model_.parameters(), lr=self.learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )
        
        # Create data loaders
        train_dataset = TensorDataset(
            torch.FloatTensor(X_seq),
            torch.FloatTensor(y_seq)
        )
        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=False  # Never shuffle time series!
        )
        
        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.epochs):
            # Training
            self.model_.train()
            train_losses = []
            
            for batch_X, batch_y in train_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                
                optimizer.zero_grad()
                
                mean, std = self.model_(batch_X)
                loss = criterion(mean, std, batch_y)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model_.parameters(), 1.0)
                optimizer.step()
                
                train_losses.append(loss.item())
            
            avg_train_loss = np.mean(train_losses)
            self.history_['train_loss'].append(avg_train_loss)
            
            # Validation
            if X_val is not None and y_val is not None:
                self.model_.eval()
                with torch.no_grad():
                    X_val_tensor = torch.FloatTensor(X_val_seq).to(self.device)
                    y_val_tensor = torch.FloatTensor(y_val_seq).to(self.device)
                    
                    mean_val, std_val = self.model_(X_val_tensor)
                    val_loss = criterion(mean_val, std_val, y_val_tensor).item()
                
                self.history_['val_loss'].append(val_loss)
                scheduler.step(val_loss)
                
                if self.verbose and epoch % 10 == 0:
                    log.info(f"Epoch {epoch}: train_loss={avg_train_loss:.4f}, val_loss={val_loss:.4f}")
                
                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    # Save best model
                    self.best_model_state_ = self.model_.state_dict()
                else:
                    patience_counter += 1
                    if patience_counter >= self.patience:
                        if self.verbose:
                            log.info(f"Early stopping at epoch {epoch}")
                        break
            else:
                if self.verbose and epoch % 10 == 0:
                    log.info(f"Epoch {epoch}: train_loss={avg_train_loss:.4f}")
        
        # Load best model if validation was used
        if hasattr(self, 'best_model_state_'):
            self.model_.load_state_dict(self.best_model_state_)
        
        return self
    
    def predict(self, X, return_std=False):
        """
        Make predictions.
        
        Args:
            X: Features
            return_std: If True, return standard deviation as well
            
        Returns:
            Predictions (and std if requested)
        """
        if self.model_ is None:
            raise ValueError("Model not fitted yet")
        
        # Convert and scale
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        X_scaled = self.scaler_.transform(X)
        X_seq = self._create_sequences(X_scaled)
        
        # Predict
        self.model_.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_seq).to(self.device)
            mean, std = self.model_(X_tensor)
            
            mean = mean.cpu().numpy()
            std = std.cpu().numpy()
        
        if return_std:
            return mean, std
        return mean
    
    def predict_distribution(self, X):
        """
        Predict full distribution.
        
        Args:
            X: Features
            
        Returns:
            Dictionary with mean, std, and quantiles
        """
        mean, std = self.predict(X, return_std=True)
        
        # Calculate quantiles using normal distribution
        quantiles = {}
        for q in [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]:
            z = torch.distributions.Normal(0, 1).icdf(torch.tensor(q)).item()
            quantiles[f'q{int(q*100)}'] = mean + z * std
        
        return {
            'mean': mean,
            'std': std,
            'lower_95': quantiles['q5'],
            'lower_90': quantiles['q10'],
            'median': quantiles['q50'],
            'upper_90': quantiles['q90'],
            'upper_95': quantiles['q95'],
            'quantiles': quantiles
        }
    
    def predict_with_uncertainty(self, X):
        """
        Predict with uncertainty using MC Dropout.
        
        Args:
            X: Features
            
        Returns:
            Dictionary with predictions and uncertainties
        """
        if self.model_ is None:
            raise ValueError("Model not fitted yet")
        
        # Convert and scale
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        X_scaled = self.scaler_.transform(X)
        X_seq = self._create_sequences(X_scaled)
        X_tensor = torch.FloatTensor(X_seq).to(self.device)
        
        # Enable dropout for MC sampling
        self.model_.train()
        
        # Collect MC samples
        mc_predictions = []
        mc_stds = []
        
        with torch.no_grad():
            for _ in range(self.mc_dropout_samples):
                mean, std = self.model_(X_tensor)
                mc_predictions.append(mean.cpu().numpy())
                mc_stds.append(std.cpu().numpy())
        
        mc_predictions = np.array(mc_predictions)
        mc_stds = np.array(mc_stds)
        
        # Calculate statistics
        mean_pred = mc_predictions.mean(axis=0)
        
        # Epistemic uncertainty (model uncertainty)
        epistemic_uncertainty = mc_predictions.std(axis=0)
        
        # Aleatoric uncertainty (data uncertainty)
        aleatoric_uncertainty = mc_stds.mean(axis=0)
        
        # Total uncertainty
        total_uncertainty = np.sqrt(epistemic_uncertainty**2 + aleatoric_uncertainty**2)
        
        return {
            'mean': mean_pred,
            'epistemic_uncertainty': epistemic_uncertainty,
            'aleatoric_uncertainty': aleatoric_uncertainty,
            'total_uncertainty': total_uncertainty,
            'lower_bound': mean_pred - 2 * total_uncertainty,
            'upper_bound': mean_pred + 2 * total_uncertainty
        }
    
    def save(self, path: str):
        """Save model and scaler."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save model
        torch.save(self.model_.state_dict(), path / 'model.pt')
        
        # Save scaler
        joblib.dump(self.scaler_, path / 'scaler.pkl')
        
        # Save config
        config = {
            'seq_len': self.seq_len,
            'hidden_size': self.hidden_size,
            'num_layers': self.num_layers,
            'dropout': self.dropout,
            'min_std': self.min_std,
            'input_size': self.model_.lstm.input_size,
            'output_size': self.model_.fc_mean.out_features
        }
        joblib.dump(config, path / 'config.pkl')
        
        log.info(f"Model saved to {path}")
    
    def load(self, path: str):
        """Load model and scaler."""
        path = Path(path)
        
        # Load config
        config = joblib.load(path / 'config.pkl')
        
        # Create model
        self.model_ = ProbabilisticLSTM(
            input_size=config['input_size'],
            hidden_size=config['hidden_size'],
            num_layers=config['num_layers'],
            dropout=config['dropout'],
            output_size=config['output_size'],
            min_std=config['min_std']
        ).to(self.device)
        
        # Load weights
        self.model_.load_state_dict(torch.load(path / 'model.pt'))
        
        # Load scaler
        self.scaler_ = joblib.load(path / 'scaler.pkl')
        
        log.info(f"Model loaded from {path}")


class LSTMProbabilisticOptuna:
    """Optuna optimization for Probabilistic LSTM."""
    
    def __init__(
        self,
        n_trials: int = 50,
        cv_folds: int = 3,
        seq_len_range: Tuple[int, int] = (30, 120),
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        random_state: int = 42
    ):
        """
        Initialize Optuna optimizer.
        
        Args:
            n_trials: Number of trials
            cv_folds: Number of CV folds
            seq_len_range: Range for sequence length
            device: Device to use
            random_state: Random seed
        """
        self.n_trials = n_trials
        self.cv_folds = cv_folds
        self.seq_len_range = seq_len_range
        self.device = device
        self.random_state = random_state
        self.best_model_ = None
        self.best_params_ = None
        self.study_ = None
        
    def objective(self, trial, X, y):
        """Optuna objective function."""
        # Suggest hyperparameters
        params = {
            'seq_len': trial.suggest_int('seq_len', *self.seq_len_range),
            'hidden_size': trial.suggest_categorical('hidden_size', [64, 128, 256]),
            'num_layers': trial.suggest_int('num_layers', 1, 3),
            'dropout': trial.suggest_float('dropout', 0.1, 0.5),
            'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True),
            'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64]),
            'device': self.device,
            'random_state': self.random_state,
            'epochs': 50,  # Reduced for optimization
            'patience': 5,
            'verbose': False
        }
        
        # Time series cross-validation
        from sklearn.model_selection import TimeSeriesSplit
        tscv = TimeSeriesSplit(n_splits=self.cv_folds)
        
        scores = []
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            # Split data
            X_train = X[train_idx]
            y_train = y[train_idx]
            X_val = X[val_idx]
            y_val = y[val_idx]
            
            # Create and train model
            model = LSTMProbabilistic(**params)
            
            try:
                model.fit(X_train, y_train, X_val, y_val)
                
                # Evaluate
                predictions = model.predict(X_val)
                
                # Use NLL as metric
                from sklearn.metrics import mean_squared_error
                score = -mean_squared_error(y_val[model.seq_len-1:], predictions.flatten())
                scores.append(score)
                
            except Exception as e:
                log.warning(f"Trial failed: {e}")
                return float('-inf')
            
            # Report intermediate value
            trial.report(score, fold)
            
            # Handle pruning
            if trial.should_prune():
                raise optuna.TrialPruned()
        
        return np.mean(scores)
    
    def optimize(self, X, y):
        """Run optimization."""
        log.info(f"Starting optimization with {self.n_trials} trials")
        
        # Create study
        self.study_ = optuna.create_study(
            direction='maximize',
            pruner=HyperbandPruner(),
            sampler=optuna.samplers.TPESampler(seed=self.random_state)
        )
        
        # Optimize
        self.study_.optimize(
            lambda trial: self.objective(trial, X, y),
            n_trials=self.n_trials
        )
        
        # Get best parameters
        self.best_params_ = self.study_.best_params
        log.info(f"Best parameters: {self.best_params_}")
        
        # Train final model with best parameters
        self.best_model_ = LSTMProbabilistic(
            **self.best_params_,
            epochs=100,
            device=self.device,
            random_state=self.random_state
        )
        
        # Use last 20% for validation
        val_size = int(0.2 * len(X))
        X_train = X[:-val_size]
        y_train = y[:-val_size]
        X_val = X[-val_size:]
        y_val = y[-val_size:]
        
        self.best_model_.fit(X_train, y_train, X_val, y_val)
        
        return self.best_model_