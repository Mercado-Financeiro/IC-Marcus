"""Training callbacks for LSTM models."""

import numpy as np
import torch
from typing import Optional, Dict, Any, List
from pathlib import Path
import structlog

log = structlog.get_logger()


class EarlyStopping:
    """Early stopping callback to prevent overfitting."""
    
    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.0001,
        mode: str = 'min',
        verbose: bool = True
    ):
        """Initialize early stopping.
        
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
            mode: 'min' for loss, 'max' for metrics like accuracy
            verbose: Whether to print messages
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.verbose = verbose
        
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_score_min = np.Inf if mode == 'min' else -np.Inf
    
    def __call__(self, val_score: float, model: Optional[torch.nn.Module] = None) -> bool:
        """Check if should stop training.
        
        Args:
            val_score: Validation score
            model: Model to save if best score
            
        Returns:
            True if should stop training
        """
        score = -val_score if self.mode == 'min' else val_score
        
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_score, model)
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.verbose:
                log.info(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    log.info("Early stopping triggered")
        else:
            self.best_score = score
            self.save_checkpoint(val_score, model)
            self.counter = 0
        
        return self.early_stop
    
    def save_checkpoint(self, val_score: float, model: Optional[torch.nn.Module]):
        """Save model when validation score improves."""
        if self.verbose:
            if self.mode == 'min':
                log.info(f"Validation score decreased ({self.val_score_min:.6f} --> {val_score:.6f})")
            else:
                log.info(f"Validation score increased ({self.val_score_min:.6f} --> {val_score:.6f})")
        
        if model is not None:
            self.best_model_state = model.state_dict().copy()
        
        self.val_score_min = val_score
    
    def reset(self):
        """Reset early stopping state."""
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_score_min = np.Inf if self.mode == 'min' else -np.Inf


class ModelCheckpoint:
    """Callback to save model checkpoints during training."""
    
    def __init__(
        self,
        filepath: str = 'checkpoint.pt',
        monitor: str = 'val_loss',
        mode: str = 'min',
        save_best_only: bool = True,
        save_weights_only: bool = False,
        verbose: bool = True
    ):
        """Initialize model checkpoint.
        
        Args:
            filepath: Path to save checkpoint
            monitor: Metric to monitor
            mode: 'min' or 'max'
            save_best_only: Only save best model
            save_weights_only: Only save weights (not optimizer state)
            verbose: Whether to print messages
        """
        self.filepath = Path(filepath)
        self.monitor = monitor
        self.mode = mode
        self.save_best_only = save_best_only
        self.save_weights_only = save_weights_only
        self.verbose = verbose
        
        self.best = np.Inf if mode == 'min' else -np.Inf
        
        # Create directory if needed
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
    
    def __call__(
        self,
        epoch: int,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        metrics: Optional[Dict[str, float]] = None
    ):
        """Save checkpoint if conditions are met.
        
        Args:
            epoch: Current epoch
            model: Model to save
            optimizer: Optimizer to save
            metrics: Dictionary of metrics
        """
        if metrics is None:
            current = 0
        else:
            current = metrics.get(self.monitor, 0)
        
        # Check if should save
        save = False
        if self.save_best_only:
            if self.mode == 'min' and current < self.best:
                save = True
                self.best = current
            elif self.mode == 'max' and current > self.best:
                save = True
                self.best = current
        else:
            save = True
        
        if save:
            # Prepare checkpoint
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'best_score': self.best,
                'metrics': metrics
            }
            
            if not self.save_weights_only and optimizer is not None:
                checkpoint['optimizer_state_dict'] = optimizer.state_dict()
            
            # Save checkpoint
            torch.save(checkpoint, self.filepath)
            
            if self.verbose:
                log.info(
                    "checkpoint_saved",
                    epoch=epoch,
                    filepath=str(self.filepath),
                    metric=self.monitor,
                    value=current
                )
    
    def load_best(
        self,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None
    ) -> Dict[str, Any]:
        """Load best checkpoint.
        
        Args:
            model: Model to load weights into
            optimizer: Optimizer to load state into
            
        Returns:
            Checkpoint dictionary
        """
        if not self.filepath.exists():
            raise FileNotFoundError(f"No checkpoint found at {self.filepath}")
        
        checkpoint = torch.load(self.filepath)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.verbose:
            log.info(
                "checkpoint_loaded",
                filepath=str(self.filepath),
                epoch=checkpoint.get('epoch', -1)
            )
        
        return checkpoint


class LearningRateScheduler:
    """Learning rate scheduler callback."""
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        mode: str = 'reduce_on_plateau',
        factor: float = 0.5,
        patience: int = 5,
        min_lr: float = 1e-7,
        verbose: bool = True
    ):
        """Initialize learning rate scheduler.
        
        Args:
            optimizer: Optimizer to schedule
            mode: Scheduling mode
            factor: Factor to reduce LR by
            patience: Patience for plateau scheduler
            min_lr: Minimum learning rate
            verbose: Whether to print messages
        """
        self.optimizer = optimizer
        self.mode = mode
        self.verbose = verbose
        
        if mode == 'reduce_on_plateau':
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=factor,
                patience=patience,
                min_lr=min_lr,
                verbose=verbose
            )
        elif mode == 'cosine':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=50,
                eta_min=min_lr
            )
        elif mode == 'exponential':
            self.scheduler = torch.optim.lr_scheduler.ExponentialLR(
                optimizer,
                gamma=0.95
            )
        else:
            raise ValueError(f"Unknown scheduler mode: {mode}")
    
    def step(self, metrics: Optional[float] = None):
        """Update learning rate.
        
        Args:
            metrics: Metric value for plateau scheduler
        """
        if self.mode == 'reduce_on_plateau':
            if metrics is not None:
                self.scheduler.step(metrics)
        else:
            self.scheduler.step()
        
        if self.verbose:
            current_lr = self.optimizer.param_groups[0]['lr']
            log.info(f"Learning rate: {current_lr:.2e}")
    
    def get_last_lr(self) -> List[float]:
        """Get last learning rate."""
        return [group['lr'] for group in self.optimizer.param_groups]


class TrainingHistory:
    """Callback to track training history."""
    
    def __init__(self):
        """Initialize training history."""
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_metrics': [],
            'val_metrics': [],
            'lr': []
        }
    
    def on_epoch_end(
        self,
        epoch: int,
        train_loss: float,
        val_loss: Optional[float] = None,
        train_metrics: Optional[Dict[str, float]] = None,
        val_metrics: Optional[Dict[str, float]] = None,
        lr: Optional[float] = None
    ):
        """Record epoch results.
        
        Args:
            epoch: Current epoch
            train_loss: Training loss
            val_loss: Validation loss
            train_metrics: Training metrics
            val_metrics: Validation metrics
            lr: Learning rate
        """
        self.history['train_loss'].append(train_loss)
        
        if val_loss is not None:
            self.history['val_loss'].append(val_loss)
        
        if train_metrics is not None:
            self.history['train_metrics'].append(train_metrics)
        
        if val_metrics is not None:
            self.history['val_metrics'].append(val_metrics)
        
        if lr is not None:
            self.history['lr'].append(lr)
    
    def get_best_epoch(self, metric: str = 'val_loss', mode: str = 'min') -> int:
        """Get best epoch based on metric.
        
        Args:
            metric: Metric to use
            mode: 'min' or 'max'
            
        Returns:
            Best epoch index
        """
        if metric not in self.history or len(self.history[metric]) == 0:
            return -1
        
        values = self.history[metric]
        if mode == 'min':
            return np.argmin(values)
        else:
            return np.argmax(values)
    
    def plot(self, save_path: Optional[str] = None):
        """Plot training history.
        
        Args:
            save_path: Path to save plot
        """
        try:
            import matplotlib.pyplot as plt
            
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            
            # Loss plot
            axes[0, 0].plot(self.history['train_loss'], label='Train')
            if self.history['val_loss']:
                axes[0, 0].plot(self.history['val_loss'], label='Validation')
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].legend()
            axes[0, 0].grid(True)
            
            # Learning rate
            if self.history['lr']:
                axes[0, 1].plot(self.history['lr'])
                axes[0, 1].set_xlabel('Epoch')
                axes[0, 1].set_ylabel('Learning Rate')
                axes[0, 1].grid(True)
            
            # Metrics
            if self.history['val_metrics']:
                metrics_df = pd.DataFrame(self.history['val_metrics'])
                for col in metrics_df.columns[:4]:  # Plot first 4 metrics
                    axes[1, 0].plot(metrics_df[col], label=col)
                axes[1, 0].set_xlabel('Epoch')
                axes[1, 0].set_ylabel('Metric Value')
                axes[1, 0].legend()
                axes[1, 0].grid(True)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path)
                log.info(f"Training history plot saved to {save_path}")
            
            plt.show()
            
        except ImportError:
            log.warning("Matplotlib not available for plotting")