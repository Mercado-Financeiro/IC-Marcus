"""Base LSTM model implementation."""

import numpy as np
import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict, Any
import structlog

log = structlog.get_logger()


class LSTMNetwork(nn.Module):
    """LSTM neural network for time series prediction with improved loss handling."""
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        bidirectional: bool = False,
        output_size: int = 1,
        use_sigmoid: bool = False  # Backward compatibility
    ):
        """Initialize LSTM network.
        
        Args:
            input_size: Number of input features
            hidden_size: Number of hidden units
            num_layers: Number of LSTM layers
            dropout: Dropout rate
            bidirectional: Use bidirectional LSTM
            output_size: Number of output classes
            use_sigmoid: Whether to apply sigmoid activation (False for BCEWithLogitsLoss)
        """
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        self.use_sigmoid = use_sigmoid
        self.output_activation = 'sigmoid' if use_sigmoid else None
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=bidirectional
        )
        
        # Batch normalization
        self.batch_norm = nn.BatchNorm1d(hidden_size * self.num_directions)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Output layers
        self.fc1 = nn.Linear(hidden_size * self.num_directions, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights."""
        for name, param in self.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)
            elif 'fc' in name and 'weight' in name:
                nn.init.xavier_uniform_(param)
    
    def forward(
        self, 
        x: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch, seq_len, features)
            hidden: Initial hidden state
            
        Returns:
            Output tensor and final hidden state
        """
        # LSTM forward pass
        lstm_out, hidden = self.lstm(x, hidden)
        
        # Take the last output
        if self.bidirectional:
            # Concatenate forward and backward outputs
            last_output = lstm_out[:, -1, :]
        else:
            last_output = lstm_out[:, -1, :]
        
        # Apply batch norm if batch size > 1
        if last_output.shape[0] > 1:
            last_output = self.batch_norm(last_output)
        
        # Dropout and fully connected layers
        out = self.dropout(last_output)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        
        # Apply sigmoid if configured (for backward compatibility)
        if self.use_sigmoid:
            out = torch.sigmoid(out)
        
        return out, hidden
    
    def init_hidden(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """Initialize hidden state.
        
        Args:
            batch_size: Batch size
            device: Device to create tensor on
            
        Returns:
            Initial hidden and cell states
        """
        h0 = torch.zeros(
            self.num_layers * self.num_directions,
            batch_size,
            self.hidden_size,
            device=device
        )
        c0 = torch.zeros(
            self.num_layers * self.num_directions,
            batch_size,
            self.hidden_size,
            device=device
        )
        return h0, c0


class BaseLSTM:
    """Base LSTM model with training and prediction methods."""
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        learning_rate: float = 0.001,
        device: Optional[str] = None,
        use_pos_weight: bool = True,  # New: enable pos_weight for imbalanced data
        use_sigmoid: bool = False  # Backward compatibility
    ):
        """Initialize base LSTM model.
        
        Args:
            input_size: Number of input features
            hidden_size: Number of hidden units
            num_layers: Number of LSTM layers
            dropout: Dropout rate
            learning_rate: Learning rate for optimizer
            device: Device to use ('cuda' or 'cpu')
            use_pos_weight: Whether to use pos_weight in loss for imbalanced data
            use_sigmoid: Whether to use sigmoid activation (False for BCEWithLogitsLoss)
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.use_pos_weight = use_pos_weight
        self.use_sigmoid = use_sigmoid
        
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Initialize network
        self.model = LSTMNetwork(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            use_sigmoid=use_sigmoid  # Pass sigmoid flag
        ).to(self.device)
        
        # Loss and optimizer
        self.pos_weight = None  # Will be set during training
        if use_sigmoid:
            self.criterion = nn.BCELoss()  # For backward compatibility
        else:
            self.criterion = nn.BCEWithLogitsLoss()  # Better for training
        
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=learning_rate
        )
        
        log.info(
            "base_lstm_initialized",
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            device=str(self.device)
        )
    
    def calculate_pos_weight(self, y_train: np.ndarray) -> torch.Tensor:
        """Calculate positive class weight for imbalanced data.
        
        Args:
            y_train: Training labels
            
        Returns:
            pos_weight tensor
        """
        pos_count = y_train.sum()
        neg_count = len(y_train) - pos_count
        
        if pos_count == 0:
            return torch.tensor(1.0)
        
        pos_weight = neg_count / pos_count
        log.info(
            "pos_weight_calculated",
            pos_ratio=float(pos_count/len(y_train)),
            pos_weight=float(pos_weight)
        )
        return torch.tensor(pos_weight, device=self.device)
    
    def train_epoch(
        self,
        dataloader: torch.utils.data.DataLoader,
        y_train: Optional[np.ndarray] = None
    ) -> float:
        """Train for one epoch.
        
        Args:
            dataloader: Training dataloader
            y_train: Optional training labels to calculate pos_weight
            
        Returns:
            Average loss for the epoch
        """
        # Calculate and set pos_weight if needed
        if self.use_pos_weight and y_train is not None and self.pos_weight is None:
            self.pos_weight = self.calculate_pos_weight(y_train)
            if isinstance(self.criterion, nn.BCEWithLogitsLoss):
                self.criterion.pos_weight = self.pos_weight
        
        self.model.train()
        total_loss = 0
        n_batches = 0
        
        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs, _ = self.model(batch_x)
            loss = self.criterion(outputs.squeeze(), batch_y)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # Update weights
            self.optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
        
        avg_loss = total_loss / n_batches if n_batches > 0 else 0
        return avg_loss
    
    def evaluate(
        self,
        dataloader: torch.utils.data.DataLoader
    ) -> Tuple[float, np.ndarray, np.ndarray]:
        """Evaluate model on validation/test data.
        
        Args:
            dataloader: Validation/test dataloader
            
        Returns:
            Tuple of (loss, predictions, targets)
        """
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_targets = []
        n_batches = 0
        
        with torch.no_grad():
            for batch_x, batch_y in dataloader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                # Forward pass
                outputs, _ = self.model(batch_x)
                loss = self.criterion(outputs.squeeze(), batch_y)
                
                # Store predictions
                preds = torch.sigmoid(outputs).cpu().numpy()
                all_preds.extend(preds.squeeze())
                all_targets.extend(batch_y.cpu().numpy())
                
                total_loss += loss.item()
                n_batches += 1
        
        avg_loss = total_loss / n_batches if n_batches > 0 else 0
        
        return avg_loss, np.array(all_preds), np.array(all_targets)
    
    def predict_proba(
        self,
        X: np.ndarray
    ) -> np.ndarray:
        """Predict probabilities.
        
        Args:
            X: Input features of shape (n_samples, seq_len, n_features)
            
        Returns:
            Predicted probabilities
        """
        self.model.eval()
        
        # Convert to tensor
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        with torch.no_grad():
            # Handle batching for large datasets
            batch_size = 256
            predictions = []
            
            for i in range(0, len(X_tensor), batch_size):
                batch = X_tensor[i:i + batch_size]
                outputs, _ = self.model(batch)
                probs = torch.sigmoid(outputs).cpu().numpy()
                predictions.extend(probs.squeeze())
        
        predictions = np.array(predictions)
        
        # Return as 2D array for sklearn compatibility
        if len(predictions.shape) == 1:
            predictions = np.column_stack([1 - predictions, predictions])
        
        return predictions
    
    def predict(
        self,
        X: np.ndarray,
        threshold: float = 0.5
    ) -> np.ndarray:
        """Predict classes.
        
        Args:
            X: Input features
            threshold: Classification threshold
            
        Returns:
            Predicted classes
        """
        proba = self.predict_proba(X)
        
        if proba.shape[1] == 2:
            return (proba[:, 1] >= threshold).astype(int)
        else:
            return (proba >= threshold).astype(int)
    
    def save_model(self, path: str):
        """Save model to file.
        
        Args:
            path: Path to save model
        """
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': {
                'input_size': self.input_size,
                'hidden_size': self.hidden_size,
                'num_layers': self.num_layers,
                'dropout': self.dropout,
                'learning_rate': self.learning_rate
            }
        }, path)
        
        log.info("model_saved", path=path)
    
    def load_model(self, path: str):
        """Load model from file.
        
        Args:
            path: Path to load model from
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        log.info("model_loaded", path=path)