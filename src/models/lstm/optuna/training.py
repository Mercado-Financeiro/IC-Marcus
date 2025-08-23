"""Training utilities for LSTM models."""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional, Dict, Any
from torch.utils.data import DataLoader
import structlog

log = structlog.get_logger()


class EarlyStopping:
    """Early stopping to prevent overfitting."""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.0001, mode: str = 'min', 
                 verbose: bool = False, delta: float = None, path: str = None):
        """
        Initialize early stopping.
        
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
            mode: 'min' for loss, 'max' for metrics
            verbose: Whether to print messages
            delta: Alternative name for min_delta (for compatibility)
            path: Path to save checkpoint (optional)
        """
        self.patience = patience
        self.min_delta = delta if delta is not None else min_delta
        self.delta = self.min_delta  # Alias for compatibility
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.verbose = verbose
        self.path = path
        
    def __call__(self, score: float, model=None) -> bool:
        """
        Check if should stop training.
        
        Args:
            score: Current score
            model: Model to save (optional)
            
        Returns:
            True if should stop
        """
        if self.best_score is None:
            self.best_score = score
            if model is not None and self.path is not None:
                self.save_checkpoint(score, model)
            return False
        
        if self.mode == 'min':
            improved = score < self.best_score - self.min_delta
        else:
            improved = score > self.best_score + self.min_delta
        
        if improved:
            self.best_score = score
            self.counter = 0
            if model is not None and self.path is not None:
                self.save_checkpoint(score, model)
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop
    
    def save_checkpoint(self, score: float, model):
        """Save model checkpoint."""
        if self.verbose:
            print(f'Saving model checkpoint with score: {score:.6f}')
        if self.path and model is not None:
            torch.save(model.state_dict(), self.path)


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    gradient_clip: float = 1.0,
    gradient_clipping: float = None
) -> Tuple[float, float]:
    """
    Train model for one epoch.
    
    Args:
        model: LSTM model
        train_loader: DataLoader for training data
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on
        gradient_clip: Gradient clipping value
        
    Returns:
        Tuple of (average loss, accuracy)
    """
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_X, batch_y in train_loader:
        # Move to device
        batch_X = batch_X.to(device)
        batch_y = batch_y.to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(batch_X)
        
        # Reshape if needed
        if outputs.dim() > 1:
            outputs = outputs.squeeze()
        
        # Calculate loss
        loss = criterion(outputs, batch_y.float())
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping (support both parameter names)
        clip_value = gradient_clipping if gradient_clipping is not None else gradient_clip
        if clip_value > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
        
        # Update weights
        optimizer.step()
        
        # Track metrics
        total_loss += loss.item()
        
        # Calculate accuracy
        predicted = (outputs > 0.5).float()
        correct += (predicted == batch_y).sum().item()
        total += batch_y.size(0)
    
    avg_loss = total_loss / len(train_loader)
    accuracy = correct / total if total > 0 else 0
    
    return avg_loss, accuracy


def validate(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """
    Validate model.
    
    Args:
        model: LSTM model
        val_loader: DataLoader for validation data
        criterion: Loss function
        device: Device to validate on
        
    Returns:
        Tuple of (average loss, accuracy, predictions, targets)
    """
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            # Move to device
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            
            # Forward pass
            outputs = model(batch_X)
            
            # Reshape if needed
            if outputs.dim() > 1:
                outputs = outputs.squeeze()
            
            # Calculate loss
            loss = criterion(outputs, batch_y.float())
            total_loss += loss.item()
            
            # Calculate accuracy
            predicted = (outputs > 0.5).float()
            correct += (predicted == batch_y).sum().item()
            total += batch_y.size(0)
            
            # Store predictions and targets
            all_predictions.extend(outputs.cpu().numpy())
            all_targets.extend(batch_y.cpu().numpy())
    
    avg_loss = total_loss / len(val_loader) if len(val_loader) > 0 else 0
    accuracy = correct / total if total > 0 else 0
    
    return avg_loss, accuracy, np.array(all_predictions), np.array(all_targets)


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    n_epochs: int = 100,
    early_stopping_patience: int = 10,
    gradient_clip: float = 1.0,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Train model with validation and early stopping.
    
    Args:
        model: LSTM model
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on
        n_epochs: Maximum number of epochs
        early_stopping_patience: Patience for early stopping
        gradient_clip: Gradient clipping value
        scheduler: Optional learning rate scheduler
        verbose: Whether to print progress
        
    Returns:
        Dictionary with training history
    """
    early_stopping = EarlyStopping(patience=early_stopping_patience)
    
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': [],
        'best_val_loss': float('inf'),
        'best_epoch': 0
    }
    
    best_model_state = None
    
    for epoch in range(n_epochs):
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, gradient_clip
        )
        
        # Validate
        val_loss, val_acc, _, _ = validate(
            model, val_loader, criterion, device
        )
        
        # Update history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        
        # Check for best model
        if val_loss < history['best_val_loss']:
            history['best_val_loss'] = val_loss
            history['best_epoch'] = epoch
            best_model_state = model.state_dict().copy()
        
        # Learning rate scheduler step
        if scheduler is not None:
            scheduler.step(val_loss)
        
        # Early stopping
        if early_stopping(val_loss):
            if verbose:
                log.info(f"Early stopping at epoch {epoch}")
            break
        
        # Print progress
        if verbose and epoch % 10 == 0:
            log.info(
                f"Epoch {epoch}: "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
            )
    
    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return history


def create_data_loader(X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool = True) -> DataLoader:
    """
    Create PyTorch DataLoader from numpy arrays.
    
    Args:
        X: Features
        y: Labels
        batch_size: Batch size
        shuffle: Whether to shuffle data
        
    Returns:
        DataLoader
    """
    from torch.utils.data import TensorDataset, DataLoader
    
    X_tensor = torch.FloatTensor(X)
    y_tensor = torch.FloatTensor(y)
    
    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    
    return loader