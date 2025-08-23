"""Data loading and preprocessing for LSTM models."""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset, Dataset
from typing import Tuple, Optional, Union, List
from sklearn.preprocessing import StandardScaler, RobustScaler
import structlog

log = structlog.get_logger()


def create_sequences(
    data: np.ndarray,
    target: np.ndarray,
    seq_length: int,
    stride: int = 1
) -> Tuple[np.ndarray, np.ndarray]:
    """Create sequences for LSTM training.
    
    Args:
        data: Input features of shape (n_samples, n_features)
        target: Target values of shape (n_samples,)
        seq_length: Length of each sequence
        stride: Stride between sequences
        
    Returns:
        Tuple of (sequences, targets) where sequences has shape
        (n_sequences, seq_length, n_features)
    """
    n_samples = len(data)
    n_sequences = (n_samples - seq_length) // stride + 1
    
    sequences = []
    targets = []
    
    for i in range(0, n_samples - seq_length + 1, stride):
        seq = data[i:i + seq_length]
        tgt = target[i + seq_length - 1]
        
        sequences.append(seq)
        targets.append(tgt)
    
    sequences = np.array(sequences)
    targets = np.array(targets)
    
    log.info(
        "sequences_created",
        input_shape=data.shape,
        output_shape=sequences.shape,
        n_sequences=n_sequences,
        seq_length=seq_length
    )
    
    return sequences, targets


class TimeSeriesDataset(Dataset):
    """Custom dataset for time series data."""
    
    def __init__(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        seq_length: int = 60,
        stride: int = 1,
        transform: Optional[callable] = None
    ):
        """Initialize time series dataset.
        
        Args:
            features: Input features
            targets: Target values
            seq_length: Sequence length
            stride: Stride between sequences
            transform: Optional transform to apply
        """
        self.features = features
        self.targets = targets
        self.seq_length = seq_length
        self.stride = stride
        self.transform = transform
        
        # Create sequences
        self.sequences, self.labels = create_sequences(
            features, targets, seq_length, stride
        )
    
    def __len__(self) -> int:
        """Get dataset length."""
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get item by index."""
        sequence = self.sequences[idx]
        label = self.labels[idx]
        
        if self.transform:
            sequence = self.transform(sequence)
        
        return torch.FloatTensor(sequence), torch.FloatTensor([label])


class LSTMDataLoader:
    """Data loader for LSTM models with preprocessing."""
    
    def __init__(
        self,
        seq_length: int = 60,
        stride: int = 1,
        batch_size: int = 32,
        scaler_type: str = 'standard',
        validation_split: float = 0.2,
        shuffle: bool = False,
        num_workers: int = 0
    ):
        """Initialize LSTM data loader.
        
        Args:
            seq_length: Length of sequences
            stride: Stride between sequences
            batch_size: Batch size for training
            scaler_type: Type of scaler ('standard', 'robust', or None)
            validation_split: Fraction for validation
            shuffle: Whether to shuffle data
            num_workers: Number of workers for data loading
        """
        self.seq_length = seq_length
        self.stride = stride
        self.batch_size = batch_size
        self.scaler_type = scaler_type
        self.validation_split = validation_split
        self.shuffle = shuffle
        self.num_workers = num_workers
        
        # Initialize scaler
        if scaler_type == 'standard':
            self.scaler = StandardScaler()
        elif scaler_type == 'robust':
            self.scaler = RobustScaler()
        else:
            self.scaler = None
    
    def prepare_data(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
        fit_scaler: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data for LSTM training.
        
        Args:
            X: Input features
            y: Target values
            fit_scaler: Whether to fit the scaler
            
        Returns:
            Tuple of (scaled_features, targets)
        """
        # Convert to numpy if needed
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values
        
        # Scale features
        if self.scaler is not None:
            if fit_scaler:
                X_scaled = self.scaler.fit_transform(X)
                log.info("scaler_fitted", scaler_type=self.scaler_type)
            else:
                X_scaled = self.scaler.transform(X)
        else:
            X_scaled = X
        
        return X_scaled, y
    
    def create_dataloaders(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
        train_idx: Optional[np.ndarray] = None,
        val_idx: Optional[np.ndarray] = None
    ) -> Tuple[DataLoader, Optional[DataLoader]]:
        """Create train and validation dataloaders.
        
        Args:
            X: Input features
            y: Target values
            train_idx: Training indices (optional)
            val_idx: Validation indices (optional)
            
        Returns:
            Tuple of (train_loader, val_loader)
        """
        # Prepare data
        X_scaled, y_processed = self.prepare_data(X, y, fit_scaler=True)
        
        # Split data if indices not provided
        if train_idx is None and val_idx is None:
            n_samples = len(X_scaled)
            n_train = int(n_samples * (1 - self.validation_split))
            
            # Time series split (no shuffling)
            train_idx = np.arange(n_train)
            val_idx = np.arange(n_train, n_samples)
        
        # Create training sequences
        if train_idx is not None:
            X_train = X_scaled[train_idx]
            y_train = y_processed[train_idx]
            
            train_dataset = TimeSeriesDataset(
                X_train, y_train,
                seq_length=self.seq_length,
                stride=self.stride
            )
            
            train_loader = DataLoader(
                train_dataset,
                batch_size=self.batch_size,
                shuffle=self.shuffle,
                num_workers=self.num_workers,
                pin_memory=torch.cuda.is_available()
            )
        else:
            train_loader = None
        
        # Create validation sequences
        if val_idx is not None:
            X_val = X_scaled[val_idx]
            y_val = y_processed[val_idx]
            
            val_dataset = TimeSeriesDataset(
                X_val, y_val,
                seq_length=self.seq_length,
                stride=self.stride
            )
            
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=torch.cuda.is_available()
            )
        else:
            val_loader = None
        
        log.info(
            "dataloaders_created",
            train_size=len(train_loader.dataset) if train_loader else 0,
            val_size=len(val_loader.dataset) if val_loader else 0,
            batch_size=self.batch_size
        )
        
        return train_loader, val_loader
    
    def create_test_dataloader(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Optional[Union[np.ndarray, pd.Series]] = None
    ) -> DataLoader:
        """Create test dataloader.
        
        Args:
            X: Input features
            y: Target values (optional for inference)
            
        Returns:
            Test dataloader
        """
        # Prepare data (don't fit scaler)
        X_scaled, _ = self.prepare_data(X, y if y is not None else np.zeros(len(X)), fit_scaler=False)
        
        # Create dummy targets if not provided
        if y is None:
            y = np.zeros(len(X_scaled))
        else:
            if isinstance(y, pd.Series):
                y = y.values
        
        # Create dataset
        test_dataset = TimeSeriesDataset(
            X_scaled, y,
            seq_length=self.seq_length,
            stride=self.stride
        )
        
        # Create dataloader
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available()
        )
        
        return test_loader
    
    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """Inverse transform scaled data.
        
        Args:
            X: Scaled data
            
        Returns:
            Original scale data
        """
        if self.scaler is not None:
            return self.scaler.inverse_transform(X)
        return X