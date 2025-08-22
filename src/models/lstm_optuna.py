"""LSTM with Bayesian optimization using Optuna."""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, Any, List
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import f1_score, precision_recall_curve, auc, roc_auc_score, brier_score_loss
from sklearn.base import BaseEstimator, ClassifierMixin
import optuna
from optuna.pruners import MedianPruner, HyperbandPruner
from optuna.samplers import TPESampler
import warnings
from pathlib import Path
import sys

# Add parent to path
sys.path.append(str(Path(__file__).parent.parent))

from data.splits import PurgedKFold
from utils.determinism import set_deterministic_environment

warnings.filterwarnings('ignore')

# Try to import logging
try:
    import structlog
    log = structlog.get_logger()
except ImportError:
    import logging
    logging.basicConfig(level=logging.INFO)
    log = logging.getLogger(__name__)


def set_lstm_deterministic(seed: int = 42):
    """Set deterministic behavior for LSTM training."""
    set_deterministic_environment(seed)
    
    # Additional PyTorch specific settings
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Deterministic operations
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Set number of threads for reproducibility
    torch.set_num_threads(1)
    
    log.info("lstm_deterministic_mode_enabled", seed=seed)


class LSTMModel(nn.Module):
    """LSTM model for time series classification."""
    
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, 
                 dropout: float, output_size: int = 1):
        """Initialize LSTM model.
        
        Args:
            input_size: Number of input features
            hidden_size: Number of hidden units
            num_layers: Number of LSTM layers
            dropout: Dropout rate
            output_size: Number of output classes (1 for binary)
        """
        super(LSTMModel, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=False
        )
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout)
        
        # Fully connected layer
        self.fc = nn.Linear(hidden_size, output_size)
        
        # Activation for binary classification
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        """Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_size)
            
        Returns:
            Output tensor of shape (batch_size, output_size)
        """
        # LSTM forward pass
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Use last hidden state
        last_hidden = lstm_out[:, -1, :]
        
        # Apply dropout
        out = self.dropout(last_hidden)
        
        # Fully connected layer
        out = self.fc(out)
        
        # Apply sigmoid for binary classification
        out = self.sigmoid(out)
        
        return out


class LSTMWrapper(BaseEstimator, ClassifierMixin):
    """Scikit-learn compatible wrapper for LSTM model."""
    
    def __init__(self, lstm_model, seq_len: int, device, scaler=None):
        self.lstm_model = lstm_model
        self.seq_len = seq_len
        self.device = device
        self.scaler = scaler
        self._estimator_type = "classifier"  # Explicitly mark as classifier
        self.classes_ = np.array([0, 1])  # Binary classification
        
    def fit(self, X, y):
        # Model is already trained
        return self
    
    def predict_proba(self, X):
        """Predict probabilities."""
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
        
        # Return probabilities for both classes
        proba_both = np.column_stack([1 - proba, proba])
        return proba_both
    
    def predict(self, X):
        """Predict classes."""
        proba = self.predict_proba(X)
        return (proba[:, 1] >= 0.5).astype(int)
    
    def _create_sequences(self, X):
        """Create sequences from dataframe."""
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        # If we don't have enough data for full sequences,
        # pad with the first values
        if len(X) < self.seq_len:
            padding = np.tile(X[0], (self.seq_len - len(X), 1))
            X = np.vstack([padding, X])
        
        # Create sequences
        sequences = []
        for i in range(len(X) - self.seq_len + 1):
            sequences.append(X[i:i+self.seq_len])
        
        return np.array(sequences)


class LSTMOptuna:
    """LSTM classifier with Optuna optimization and mandatory calibration."""
    
    def __init__(
        self,
        n_trials: int = 50,
        cv_folds: int = 3,
        embargo: int = 10,
        pruner_type: str = 'median',
        early_stopping_patience: int = 10,
        max_epochs: int = 100,
        use_mlflow: bool = False,
        seed: int = 42
    ):
        """Initialize LSTM optimizer.
        
        Args:
            n_trials: Number of Optuna trials
            cv_folds: Number of CV folds
            embargo: Embargo period for Purged K-Fold
            pruner_type: Type of pruner (median, hyperband)
            early_stopping_patience: Patience for early stopping
            max_epochs: Maximum number of epochs
            use_mlflow: Whether to log to MLflow
            seed: Random seed
        """
        self.n_trials = n_trials
        self.cv_folds = cv_folds
        self.embargo = embargo
        self.pruner_type = pruner_type
        self.early_stopping_patience = early_stopping_patience
        self.max_epochs = max_epochs
        self.use_mlflow = use_mlflow
        self.seed = seed
        
        self.best_model = None
        self.best_params = None
        self.best_score = None
        self.calibrator = None
        self.threshold_f1 = 0.5
        self.threshold_ev = 0.5
        self.scaler = None
        
        # Set deterministic mode
        set_lstm_deterministic(seed)
        
        # Set device
        self.device = self._get_device()
        
        log.info(
            "lstm_optuna_initialized",
            n_trials=n_trials,
            cv_folds=cv_folds,
            embargo=embargo,
            device=self.device.type
        )
    
    def _get_device(self) -> torch.device:
        """Get the device to use (GPU if available)."""
        if torch.cuda.is_available():
            device = torch.device('cuda')
            log.info("Using GPU for LSTM training")
        else:
            device = torch.device('cpu')
            log.info("Using CPU for LSTM training")
        return device
    
    def create_sequences(self, X: pd.DataFrame, y: pd.Series, seq_len: int) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for LSTM.
        
        Args:
            X: Features
            y: Labels
            seq_len: Sequence length
            
        Returns:
            Tuple of (sequences, labels)
        """
        sequences = []
        labels = []
        
        for i in range(len(X) - seq_len):
            seq = X.iloc[i:i+seq_len].values
            label = y.iloc[i+seq_len]
            sequences.append(seq)
            labels.append(label)
        
        return np.array(sequences), np.array(labels)
    
    def _create_model(self, input_size: int, hidden_size: int, 
                     num_layers: int, dropout: float) -> LSTMModel:
        """Create LSTM model.
        
        Args:
            input_size: Number of input features
            hidden_size: Number of hidden units
            num_layers: Number of LSTM layers
            dropout: Dropout rate
            
        Returns:
            LSTM model
        """
        # Ensure deterministic initialization
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)
        
        model = LSTMModel(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            output_size=1
        )
        return model.to(self.device)
    
    def _create_search_space(self, trial: optuna.Trial) -> Dict:
        """Create hyperparameter search space.
        
        Args:
            trial: Optuna trial
            
        Returns:
            Dictionary of hyperparameters
        """
        return {
            'hidden_size': trial.suggest_int('hidden_size', 32, 512),
            'num_layers': trial.suggest_int('num_layers', 1, 3),
            'dropout': trial.suggest_float('dropout', 0.0, 0.5),
            'seq_len': trial.suggest_int('seq_len', 20, 200),
            'lr': trial.suggest_float('lr', 1e-5, 1e-2, log=True),
            'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64, 128, 256]),
            'weight_decay': trial.suggest_float('weight_decay', 0, 1e-3),
            'gradient_clip': trial.suggest_float('gradient_clip', 0.1, 2.0)
        }
    
    def _get_pruner(self) -> optuna.pruners.BasePruner:
        """Get Optuna pruner based on type.
        
        Returns:
            Optuna pruner
        """
        if self.pruner_type == 'median':
            return MedianPruner(n_startup_trials=5, n_warmup_steps=10)
        elif self.pruner_type == 'hyperband':
            return HyperbandPruner(min_resource=1, max_resource=self.max_epochs)
        else:
            return MedianPruner()
    
    def _train_epoch(self, model, train_loader, criterion, optimizer, gradient_clip):
        """Train for one epoch."""
        model.train()
        total_loss = 0
        
        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(self.device)
            batch_y = batch_y.to(self.device)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            
            optimizer.step()
            total_loss += loss.item()
        
        return total_loss / len(train_loader)
    
    def _validate(self, model, val_loader, criterion):
        """Validate model."""
        model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                total_loss += loss.item()
                
                all_preds.extend(outputs.cpu().numpy())
                all_labels.extend(batch_y.cpu().numpy())
        
        return total_loss / len(val_loader), np.array(all_preds), np.array(all_labels)
    
    def _optimize_threshold_f1(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
        """Optimize threshold for F1 score."""
        precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
        f1_scores = np.nan_to_num(f1_scores)
        
        if len(f1_scores) > 0 and len(thresholds) > 0:
            best_idx = np.argmax(f1_scores[:-1])
            return float(thresholds[best_idx])
        
        return 0.5
    
    def _optimize_threshold_ev(self, y_true: np.ndarray, y_pred_proba: np.ndarray, 
                              costs: Dict) -> float:
        """Optimize threshold for Expected Value."""
        thresholds = np.linspace(0.05, 0.95, 181)
        best_ev = -np.inf
        best_threshold = 0.5
        
        total_cost_bps = costs.get('fee_bps', 5) + costs.get('slippage_bps', 5)
        
        for threshold in thresholds:
            signals = (y_pred_proba >= threshold).astype(int)
            returns = np.where(signals == y_true, 0.01, -0.01)
            
            trades = np.diff(np.concatenate([[0], signals]))
            n_trades = np.abs(trades).sum()
            
            if len(signals) > 0:
                avg_return = returns.mean()
                trade_cost = (n_trades / len(signals)) * total_cost_bps / 10000
                ev = avg_return - trade_cost
                
                if ev > best_ev:
                    best_ev = ev
                    best_threshold = threshold
        
        return float(best_threshold)
    
    def _create_objective(self, X: pd.DataFrame, y: pd.Series):
        """Create objective function for Optuna."""
        
        def objective(trial):
            # Get hyperparameters
            params = self._create_search_space(trial)
            
            # Create cross-validator
            cv = PurgedKFold(n_splits=self.cv_folds, embargo=self.embargo)
            
            scores = []
            
            for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X, y)):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                
                # Create sequences
                X_train_seq, y_train_seq = self.create_sequences(X_train, y_train, params['seq_len'])
                X_val_seq, y_val_seq = self.create_sequences(X_val, y_val, params['seq_len'])
                
                # Skip if not enough data
                if len(X_train_seq) < params['batch_size'] or len(X_val_seq) < 10:
                    continue
                
                # Normalize features
                from sklearn.preprocessing import StandardScaler
                scaler = StandardScaler()
                
                X_train_seq_reshaped = X_train_seq.reshape(-1, X_train_seq.shape[-1])
                X_train_seq_reshaped = scaler.fit_transform(X_train_seq_reshaped)
                X_train_seq = X_train_seq_reshaped.reshape(X_train_seq.shape)
                
                X_val_seq_reshaped = X_val_seq.reshape(-1, X_val_seq.shape[-1])
                X_val_seq_reshaped = scaler.transform(X_val_seq_reshaped)
                X_val_seq = X_val_seq_reshaped.reshape(X_val_seq.shape)
                
                # Convert to tensors
                X_train_t = torch.FloatTensor(X_train_seq).to(self.device)
                y_train_t = torch.FloatTensor(y_train_seq).unsqueeze(1).to(self.device)
                X_val_t = torch.FloatTensor(X_val_seq).to(self.device)
                y_val_t = torch.FloatTensor(y_val_seq).unsqueeze(1).to(self.device)
                
                # Create data loaders
                train_dataset = TensorDataset(X_train_t, y_train_t)
                train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True)
                
                val_dataset = TensorDataset(X_val_t, y_val_t)
                val_loader = DataLoader(val_dataset, batch_size=params['batch_size'], shuffle=False)
                
                # Create model
                model = self._create_model(
                    input_size=X.shape[1],
                    hidden_size=params['hidden_size'],
                    num_layers=params['num_layers'],
                    dropout=params['dropout']
                )
                
                # Setup training
                criterion = nn.BCELoss()
                optimizer = optim.Adam(
                    model.parameters(),
                    lr=params['lr'],
                    weight_decay=params['weight_decay']
                )
                
                # Training loop with early stopping
                best_val_loss = float('inf')
                patience_counter = 0
                
                for epoch in range(self.max_epochs):
                    # Train
                    train_loss = self._train_epoch(
                        model, train_loader, criterion, optimizer, 
                        params['gradient_clip']
                    )
                    
                    # Validate
                    val_loss, val_preds, val_labels = self._validate(model, val_loader, criterion)
                    
                    # Early stopping
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        patience_counter = 0
                    else:
                        patience_counter += 1
                        if patience_counter >= self.early_stopping_patience:
                            break
                    
                    # Report for pruning
                    trial.report(val_loss, epoch)
                    
                    if trial.should_prune():
                        raise optuna.TrialPruned()
                
                # Calculate final score
                _, val_preds, val_labels = self._validate(model, val_loader, criterion)
                
                # Optimize threshold and calculate F1
                threshold = self._optimize_threshold_f1(val_labels.flatten(), val_preds.flatten())
                y_pred = (val_preds.flatten() >= threshold).astype(int)
                f1 = f1_score(val_labels.flatten(), y_pred)
                
                scores.append(f1)
            
            return np.mean(scores) if scores else 0.0
        
        return objective
    
    def optimize(self, X: pd.DataFrame, y: pd.Series) -> optuna.Study:
        """Run Bayesian optimization.
        
        Args:
            X: Features
            y: Labels
            
        Returns:
            Optuna study
        """
        # Validate input
        if len(X) == 0 or len(y) == 0:
            raise ValueError("Empty data provided")
        
        if len(np.unique(y)) < 2:
            raise ValueError("Only single class in labels")
        
        log.info(
            "starting_lstm_optimization",
            n_samples=len(X),
            n_features=X.shape[1],
            class_balance=pd.Series(y).value_counts().to_dict()
        )
        
        # Create study
        study = optuna.create_study(
            direction='maximize',
            sampler=TPESampler(seed=self.seed),
            pruner=self._get_pruner(),
            study_name='lstm_optimization'
        )
        
        # Create objective
        objective = self._create_objective(X, y)
        
        # Optimize
        study.optimize(
            objective,
            n_trials=self.n_trials,
            show_progress_bar=True
        )
        
        # Save best parameters
        self.best_params = study.best_params
        self.best_score = study.best_value
        
        log.info(
            "lstm_optimization_complete",
            best_score=self.best_score,
            best_params=self.best_params,
            n_trials=len(study.trials)
        )
        
        # Fit final model
        self.fit_final_model(X, y)
        
        return study
    
    def fit_final_model(self, X: pd.DataFrame, y: pd.Series):
        """Fit final model with best parameters.
        
        Args:
            X: Features
            y: Labels
        """
        if self.best_params is None:
            # Use default parameters if not optimized
            self.best_params = {
                'hidden_size': 64,
                'num_layers': 2,
                'dropout': 0.2,
                'seq_len': 50,
                'lr': 0.001,
                'batch_size': 32,
                'weight_decay': 0.0001,
                'gradient_clip': 1.0
            }
        
        # Create sequences
        X_seq, y_seq = self.create_sequences(X, y, self.best_params['seq_len'])
        
        # Normalize
        from sklearn.preprocessing import StandardScaler
        self.scaler = StandardScaler()
        
        X_seq_reshaped = X_seq.reshape(-1, X_seq.shape[-1])
        X_seq_reshaped = self.scaler.fit_transform(X_seq_reshaped)
        X_seq = X_seq_reshaped.reshape(X_seq.shape)
        
        # Convert to tensors
        X_tensor = torch.FloatTensor(X_seq).to(self.device)
        y_tensor = torch.FloatTensor(y_seq).unsqueeze(1).to(self.device)
        
        # Create data loader
        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=self.best_params['batch_size'], shuffle=True)
        
        # Create model
        self.best_model = self._create_model(
            input_size=X.shape[1],
            hidden_size=self.best_params['hidden_size'],
            num_layers=self.best_params['num_layers'],
            dropout=self.best_params['dropout']
        )
        
        # Train
        criterion = nn.BCELoss()
        optimizer = optim.Adam(
            self.best_model.parameters(),
            lr=self.best_params['lr'],
            weight_decay=self.best_params['weight_decay']
        )
        
        for epoch in range(min(self.max_epochs, 50)):  # Limit training for final model
            self._train_epoch(
                self.best_model, loader, criterion, optimizer,
                self.best_params['gradient_clip']
            )
        
        # Create sklearn wrapper for calibration
        wrapper = LSTMWrapper(
            self.best_model, 
            self.best_params['seq_len'],
            self.device,
            self.scaler
        )
        
        # Calibrate
        self.calibrator = CalibratedClassifierCV(
            wrapper, method='isotonic', cv=3
        )
        self.calibrator.fit(X, y)
        
        # Optimize thresholds
        y_pred_proba = self.predict_proba(X)
        self.threshold_f1 = self._optimize_threshold_f1(y, y_pred_proba)
        
        costs = {'fee_bps': 5, 'slippage_bps': 5}
        self.threshold_ev = self._optimize_threshold_ev(y, y_pred_proba, costs)
        
        log.info(
            "lstm_final_model_fitted",
            threshold_f1=self.threshold_f1,
            threshold_ev=self.threshold_ev
        )
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict probabilities with calibrated model.
        
        Args:
            X: Features
            
        Returns:
            Calibrated probabilities
        """
        if self.calibrator is None:
            raise ValueError("Model not fitted. Run optimize or fit_final_model first.")
        
        return self.calibrator.predict_proba(X)[:, 1]
    
    def predict(self, X: pd.DataFrame, use_ev_threshold: bool = False) -> np.ndarray:
        """Predict classes using optimized threshold.
        
        Args:
            X: Features
            use_ev_threshold: Whether to use EV threshold instead of F1
            
        Returns:
            Predicted classes
        """
        proba = self.predict_proba(X)
        threshold = self.threshold_ev if use_ev_threshold else self.threshold_f1
        return (proba >= threshold).astype(int)