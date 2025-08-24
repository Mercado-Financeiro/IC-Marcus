"""
Optimized LSTM baseline implementation.
Simple, fast, and compatible with BaseModel interface.
No attention, no PRD, just reliable predictions.
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, Any, Optional, Tuple
import logging
from pathlib import Path
import joblib
from sklearn.preprocessing import StandardScaler

from .base_model import BaseModel

logger = logging.getLogger(__name__)


class SimpleLSTM(nn.Module):
    """Simple LSTM network without bells and whistles."""
    
    def __init__(self, 
                 input_size: int,
                 hidden_size: int = 64,
                 num_layers: int = 1,
                 dropout: float = 0.2):
        """
        Initialize simple LSTM.
        
        Args:
            input_size: Number of input features
            hidden_size: Number of hidden units
            num_layers: Number of LSTM layers
            dropout: Dropout rate
        """
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Output layers
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        """Forward pass."""
        # LSTM
        lstm_out, _ = self.lstm(x)
        
        # Take last timestep
        last_out = lstm_out[:, -1, :]
        
        # Dropout and output
        out = self.dropout(last_out)
        out = self.fc(out)
        
        return out


class LSTMBaseline(BaseModel):
    """
    LSTM baseline model with minimal complexity.
    Compatible with ensemble and BaseModel interface.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize LSTM baseline.
        
        Args:
            config: Model configuration
        """
        super().__init__(config)
        
        # Model parameters
        self.sequence_length = config.get('sequence_length', 32)  # Reduced from 64
        self.hidden_size = config.get('hidden_size', 64)  # Reduced from 128
        self.num_layers = config.get('num_layers', 1)  # Reduced from 2
        self.dropout = config.get('dropout', 0.2)
        self.learning_rate = config.get('learning_rate', 0.0005)
        self.batch_size = config.get('batch_size', 512)  # Increased for efficiency
        self.epochs = config.get('epochs', 30)  # Reduced from 100
        self.early_stopping_patience = config.get('early_stopping_patience', 5)
        
        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Model components
        self.model = None
        self.scaler = StandardScaler()
        self.optimizer = None
        self.criterion = nn.BCEWithLogitsLoss()
        
        # Training state
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        
        # Set random seeds for reproducibility
        self._set_seeds()
        
    def _set_seeds(self):
        """Set random seeds for reproducibility."""
        seed = self.config.get('seed', 42)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    def _create_sequences(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Create sequences for LSTM input.
        
        Args:
            X: Feature DataFrame
            y: Target Series (optional)
            
        Returns:
            Tuple of (sequences, targets)
        """
        X_array = X.values if isinstance(X, pd.DataFrame) else X
        n_samples = len(X_array) - self.sequence_length + 1
        
        if n_samples <= 0:
            raise ValueError(f"Not enough samples for sequence_length={self.sequence_length}")
        
        # Create sequences
        sequences = np.zeros((n_samples, self.sequence_length, X_array.shape[1]))
        for i in range(n_samples):
            sequences[i] = X_array[i:i + self.sequence_length]
        
        # Create targets if provided
        if y is not None:
            y_array = y.values if isinstance(y, pd.Series) else y
            targets = y_array[self.sequence_length - 1:]
        else:
            targets = None
        
        return sequences, targets
    
    def fit(self,
            X_train: pd.DataFrame,
            y_train: pd.Series,
            X_val: Optional[pd.DataFrame] = None,
            y_val: Optional[pd.Series] = None) -> 'LSTMBaseline':
        """
        Train LSTM model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            
        Returns:
            Self for method chaining
        """
        logger.info("Training LSTM baseline model...")
        
        # Store feature names
        self.feature_names = list(X_train.columns)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
        
        if X_val is not None:
            X_val_scaled = self.scaler.transform(X_val)
            X_val_scaled = pd.DataFrame(X_val_scaled, columns=X_val.columns, index=X_val.index)
        
        # Create sequences
        X_train_seq, y_train_seq = self._create_sequences(X_train_scaled, y_train)
        
        if X_val is not None and y_val is not None:
            X_val_seq, y_val_seq = self._create_sequences(X_val_scaled, y_val)
        else:
            X_val_seq, y_val_seq = None, None
        
        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train_seq).to(self.device)
        y_train_tensor = torch.FloatTensor(y_train_seq).unsqueeze(1).to(self.device)
        
        if X_val_seq is not None:
            X_val_tensor = torch.FloatTensor(X_val_seq).to(self.device)
            y_val_tensor = torch.FloatTensor(y_val_seq).unsqueeze(1).to(self.device)
        
        # Create data loaders
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        
        if X_val_seq is not None:
            val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
            val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        else:
            val_loader = None
        
        # Initialize model
        input_size = X_train_seq.shape[2]
        self.model = SimpleLSTM(
            input_size=input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout
        ).to(self.device)
        
        # Initialize optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=1e-5
        )
        
        # Training loop
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        
        for epoch in range(self.epochs):
            # Training
            self.model.train()
            train_loss = 0
            for batch_x, batch_y in train_loader:
                self.optimizer.zero_grad()
                
                outputs = self.model(batch_x)
                loss = self.criterion(outputs, batch_y)
                
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                self.optimizer.step()
                train_loss += loss.item()
            
            avg_train_loss = train_loss / len(train_loader)
            self.train_losses.append(avg_train_loss)
            
            # Validation
            if val_loader is not None:
                self.model.eval()
                val_loss = 0
                with torch.no_grad():
                    for batch_x, batch_y in val_loader:
                        outputs = self.model(batch_x)
                        loss = self.criterion(outputs, batch_y)
                        val_loss += loss.item()
                
                avg_val_loss = val_loss / len(val_loader)
                self.val_losses.append(avg_val_loss)
                
                # Early stopping
                if avg_val_loss < self.best_val_loss:
                    self.best_val_loss = avg_val_loss
                    self.patience_counter = 0
                    # Save best model state
                    self.best_model_state = self.model.state_dict()
                else:
                    self.patience_counter += 1
                
                if self.patience_counter >= self.early_stopping_patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break
                
                if (epoch + 1) % 10 == 0:
                    logger.info(f"Epoch {epoch+1}/{self.epochs} - "
                              f"Train Loss: {avg_train_loss:.4f}, "
                              f"Val Loss: {avg_val_loss:.4f}")
            else:
                if (epoch + 1) % 10 == 0:
                    logger.info(f"Epoch {epoch+1}/{self.epochs} - "
                              f"Train Loss: {avg_train_loss:.4f}")
        
        # Load best model state if available
        if hasattr(self, 'best_model_state'):
            self.model.load_state_dict(self.best_model_state)
        
        self.is_fitted = True
        
        # Clear GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.info("LSTM training completed")
        return self
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict probabilities.
        
        Args:
            X: Feature DataFrame
            
        Returns:
            Probability array of shape (n_samples, 2)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # Ensure feature order
        X = X[self.feature_names]
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
        
        # Create sequences
        X_seq, _ = self._create_sequences(X_scaled)
        
        # Convert to tensor
        X_tensor = torch.FloatTensor(X_seq).to(self.device)
        
        # Predict
        self.model.eval()
        with torch.no_grad():
            logits = self.model(X_tensor)
            probs = torch.sigmoid(logits).cpu().numpy()
        
        # Convert to 2D array with both class probabilities
        probs_2d = np.column_stack([1 - probs.flatten(), probs.flatten()])
        
        # Apply calibration if available
        if self.calibrator is not None:
            probs_2d[:, 1] = self._apply_calibration(probs_2d[:, 1])
            probs_2d[:, 0] = 1 - probs_2d[:, 1]
        
        return probs_2d
    
    def save(self, path: str) -> None:
        """
        Save model to disk.
        
        Args:
            path: Path to save the model
        """
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted model")
        
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        # Save PyTorch model separately
        model_path = Path(path).with_suffix('.pth')
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'model_config': {
                'input_size': self.model.lstm.input_size,
                'hidden_size': self.hidden_size,
                'num_layers': self.num_layers,
                'dropout': self.dropout
            }
        }, model_path)
        
        # Save other components
        model_data = {
            'config': self.config,
            'threshold': self.threshold,
            'calibrator': self.calibrator,
            'feature_names': self.feature_names,
            'is_fitted': self.is_fitted,
            'scaler': self.scaler,
            'sequence_length': self.sequence_length
        }
        
        joblib.dump(model_data, path)
        logger.info(f"LSTM model saved to {path}")
    
    def load(self, path: str) -> 'LSTMBaseline':
        """
        Load model from disk.
        
        Args:
            path: Path to load the model from
            
        Returns:
            Self for method chaining
        """
        # Load model data
        model_data = joblib.load(path)
        
        self.config = model_data['config']
        self.threshold = model_data['threshold']
        self.calibrator = model_data['calibrator']
        self.feature_names = model_data['feature_names']
        self.is_fitted = model_data['is_fitted']
        self.scaler = model_data['scaler']
        self.sequence_length = model_data['sequence_length']
        
        # Load PyTorch model
        model_path = Path(path).with_suffix('.pth')
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Recreate model
        model_config = checkpoint['model_config']
        self.model = SimpleLSTM(
            input_size=model_config['input_size'],
            hidden_size=model_config['hidden_size'],
            num_layers=model_config['num_layers'],
            dropout=model_config['dropout']
        ).to(self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        logger.info(f"LSTM model loaded from {path}")
        return self