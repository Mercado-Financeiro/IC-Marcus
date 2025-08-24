"""LSTM model architecture for time series classification."""

import torch
import torch.nn as nn
from typing import Tuple


class LSTMModel(nn.Module):
    """LSTM model for time series classification.

    Extended to optionally support layer normalization and a simple residual
    connection from the last input vector into the post-LSTM representation.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        output_size: int = 1,
        bidirectional: bool = False,
        use_sigmoid: bool = False,  # For BCEWithLogitsLoss, set to False
        use_layer_norm: bool = False,
        use_residual: bool = False,
    ):
        """
        Initialize LSTM model.
        
        Args:
            input_size: Number of input features
            hidden_size: Number of hidden units
            num_layers: Number of LSTM layers
            dropout: Dropout rate
            output_size: Number of output classes (1 for binary)
            bidirectional: Whether to use bidirectional LSTM
        """
        super(LSTMModel, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=bidirectional
        )
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout)

        # Optional LayerNorm on the last hidden representation
        self.use_layer_norm = use_layer_norm
        ln_dim = hidden_size * (2 if bidirectional else 1)
        self.layer_norm = nn.LayerNorm(ln_dim) if use_layer_norm else None

        # Fully connected layer
        fc_input_size = hidden_size * 2 if bidirectional else hidden_size
        self.fc = nn.Linear(fc_input_size, output_size)

        # Activation for binary classification (optional)
        self.use_sigmoid = use_sigmoid
        self.sigmoid = nn.Sigmoid() if use_sigmoid else None

        # Optional residual connection from input (last time step)
        self.use_residual = use_residual
        self.residual_proj = None
        if use_residual:
            # Project last input to hidden dimension (match bi/uni dir)
            self.residual_proj = nn.Linear(input_size, fc_input_size)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_size)
            
        Returns:
            Output tensor of shape (batch_size, output_size)
        """
        # LSTM forward pass
        lstm_out, (h_n, c_n) = self.lstm(x)

        # Use last hidden state
        last_hidden = lstm_out[:, -1, :]

        # Optional residual from last input
        if self.use_residual and self.residual_proj is not None:
            last_input = x[:, -1, :]
            last_hidden = last_hidden + self.residual_proj(last_input)

        # Optional layer normalization
        if self.layer_norm is not None:
            last_hidden = self.layer_norm(last_hidden)

        # Apply dropout
        out = self.dropout(last_hidden)
        
        # Fully connected layer
        out = self.fc(out)
        
        # Apply sigmoid only if configured (not needed for BCEWithLogitsLoss)
        if self.use_sigmoid and self.sigmoid is not None:
            out = self.sigmoid(out)
        
        return out
    
    def init_hidden(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Initialize hidden state.
        
        Args:
            batch_size: Batch size
            device: Device to create tensors on
            
        Returns:
            Tuple of (hidden_state, cell_state)
        """
        num_directions = 2 if self.bidirectional else 1
        h0 = torch.zeros(
            self.num_layers * num_directions,
            batch_size,
            self.hidden_size,
            device=device
        )
        c0 = torch.zeros(
            self.num_layers * num_directions,
            batch_size,
            self.hidden_size,
            device=device
        )
        return h0, c0


class AttentionLSTM(nn.Module):
    """LSTM with attention mechanism."""
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        output_size: int = 1,
        num_heads: int = 1  # kept for API compatibility; not used in this simple attention
    ):
        """
        Initialize Attention LSTM model.
        
        Args:
            input_size: Number of input features
            hidden_size: Number of hidden units
            num_layers: Number of LSTM layers
            dropout: Dropout rate
            output_size: Number of output classes
        """
        super(AttentionLSTM, self).__init__()
        
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
        
        # Attention layer (single-head additive attention). The num_heads
        # parameter is accepted for compatibility but unused here.
        self.attention = nn.Linear(hidden_size, 1)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Output layer
        self.fc = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with attention.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_size)
            
        Returns:
            Output tensor of shape (batch_size, output_size)
        """
        # LSTM forward pass
        lstm_out, _ = self.lstm(x)
        
        # Calculate attention weights
        attention_weights = torch.softmax(
            self.attention(lstm_out).squeeze(-1), 
            dim=1
        )
        
        # Apply attention weights
        context = torch.sum(
            lstm_out * attention_weights.unsqueeze(-1), 
            dim=1
        )
        
        # Apply dropout
        out = self.dropout(context)
        
        # Output layer
        out = self.fc(out)
        out = self.sigmoid(out)
        
        return out
