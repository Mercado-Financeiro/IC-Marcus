"""Configuration and dataclasses for LSTM Optuna optimization."""

from dataclasses import dataclass
from typing import Optional, Dict, Any


@dataclass
class LSTMOptunaConfig:
    """Configuration for LSTM Optuna optimization."""
    
    # Optuna settings
    n_trials: int = 50
    cv_folds: int = 3
    embargo: int = 10
    pruner_type: str = 'median'
    
    # Training settings
    early_stopping_patience: int = 10
    max_epochs: int = 100
    batch_size: int = 32
    
    # Model settings
    seq_len_min: int = 10
    seq_len_max: int = 60
    hidden_size_min: int = 32
    hidden_size_max: int = 256
    num_layers_min: int = 1
    num_layers_max: int = 3
    dropout_min: float = 0.1
    dropout_max: float = 0.5
    
    # Optimizer settings
    learning_rate_min: float = 1e-4
    learning_rate_max: float = 1e-2
    weight_decay_min: float = 1e-5
    weight_decay_max: float = 1e-3
    gradient_clip_min: float = 0.5
    gradient_clip_max: float = 2.0
    
    # System settings
    use_mlflow: bool = False
    seed: int = 42
    device: str = 'auto'  # 'auto', 'cpu', 'cuda'
    verbose: bool = False
    
    # Validation settings
    threshold_std: float = 0.005
    min_unique: int = 10
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            'n_trials': self.n_trials,
            'cv_folds': self.cv_folds,
            'embargo': self.embargo,
            'pruner_type': self.pruner_type,
            'early_stopping_patience': self.early_stopping_patience,
            'max_epochs': self.max_epochs,
            'batch_size': self.batch_size,
            'seed': self.seed,
            'device': self.device,
            'use_mlflow': self.use_mlflow,
            'verbose': self.verbose
        }


@dataclass
class TrainingMetrics:
    """Container for training metrics."""
    
    train_loss: float
    val_loss: float
    train_accuracy: float
    val_accuracy: float
    train_f1: float
    val_f1: float
    train_auc: float
    val_auc: float
    epoch: int
    
    def to_dict(self) -> Dict[str, float]:
        """Convert metrics to dictionary."""
        return {
            'train_loss': self.train_loss,
            'val_loss': self.val_loss,
            'train_accuracy': self.train_accuracy,
            'val_accuracy': self.val_accuracy,
            'train_f1': self.train_f1,
            'val_f1': self.val_f1,
            'train_auc': self.train_auc,
            'val_auc': self.val_auc,
            'epoch': self.epoch
        }