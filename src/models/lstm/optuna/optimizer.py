"""LSTM Optuna optimization orchestrator."""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, Tuple, Optional, Any
import optuna
from optuna.pruners import MedianPruner, HyperbandPruner
from optuna.samplers import TPESampler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, precision_recall_curve, auc

from .config import LSTMOptunaConfig
from .model import LSTMModel, AttentionLSTM
from .wrapper import LSTMWrapper
from .training import train_model
from .utils import (
    get_logger, set_lstm_deterministic, check_constant_predictions,
    create_sequences, get_device, calculate_metrics
)

log = get_logger()


class LSTMOptuna:
    """LSTM classifier with Optuna optimization and mandatory calibration."""
    
    def __init__(self, config: Optional[LSTMOptunaConfig] = None):
        """
        Initialize LSTM optimizer.
        
        Args:
            config: Configuration object
        """
        self.config = config or LSTMOptunaConfig()
        
        self.best_model = None
        self.best_params = None
        self.best_score = -np.inf
        self.calibrator = None
        self.threshold_f1 = 0.5
        self.threshold_ev = 0.5
        self.scaler = StandardScaler()
        self.study = None
        self.wrapper = None
        self.feature_names_ = None
        
        # Set device
        self.device = get_device(self.config.device)
        
        # Set deterministic environment
        set_lstm_deterministic(self.config.seed)
        
        log.info(
            "lstm_optuna_initialized",
            n_trials=self.config.n_trials,
            cv_folds=self.config.cv_folds,
            device=str(self.device)
        )
    
    def _objective(self, trial: optuna.Trial, X: np.ndarray, y: np.ndarray) -> float:
        """Objective function for Optuna optimization."""
        # Suggest hyperparameters
        params = {
            'seq_len': trial.suggest_int('seq_len', self.config.seq_len_min, self.config.seq_len_max),
            'hidden_size': trial.suggest_int('hidden_size', self.config.hidden_size_min, self.config.hidden_size_max),
            'num_layers': trial.suggest_int('num_layers', self.config.num_layers_min, self.config.num_layers_max),
            'dropout': trial.suggest_float('dropout', self.config.dropout_min, self.config.dropout_max),
            'learning_rate': trial.suggest_float('learning_rate', self.config.learning_rate_min, self.config.learning_rate_max, log=True),
            'optimizer_type': trial.suggest_categorical('optimizer_type', ['adam', 'sgd', 'rmsprop']),
            'bidirectional': trial.suggest_categorical('bidirectional', [True, False])
        }
        
        # Perform cross-validation with embargo
        from src.features.validation.temporal import TemporalValidator, TemporalValidationConfig
        val_config = TemporalValidationConfig(n_splits=self.config.cv_folds, embargo=10)
        validator = TemporalValidator(val_config)
        scores = []
        
        for fold, (train_idx, val_idx) in enumerate(validator.split(X, y, strategy='purged_kfold')):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Train model
            model, score = self._train_model(X_train, y_train, X_val, y_val, params)
            scores.append(score)
            
            # Report intermediate value for pruning
            trial.report(score, fold)
            if trial.should_prune():
                raise optuna.TrialPruned()
        
        return np.mean(scores)
    
    def _train_model(self, X_train: np.ndarray, y_train: np.ndarray, 
                     X_val: np.ndarray, y_val: np.ndarray, 
                     params: Dict, max_epochs: int = None) -> Tuple[nn.Module, float]:
        """Train a single model with given parameters."""
        if max_epochs is None:
            max_epochs = self.config.max_epochs
            
        # Create sequences
        seq_len = params['seq_len']
        X_train_seq, y_train_seq = create_sequences(X_train, y_train, seq_len)
        X_val_seq, y_val_seq = create_sequences(X_val, y_val, seq_len)
        
        if len(X_train_seq) == 0 or len(X_val_seq) == 0:
            return None, 0.0
        
        # Create model
        n_features = X_train_seq.shape[-1]
        model = LSTMModel(
            input_size=n_features,
            hidden_size=params['hidden_size'],
            num_layers=params['num_layers'],
            dropout=params['dropout'],
            bidirectional=params.get('bidirectional', False),
            output_size=1
        ).to(self.device)
        
        # Train using the training module
        trained_model = train_model(
            model, X_train_seq, y_train_seq, X_val_seq, y_val_seq,
            self.device, epochs=max_epochs, batch_size=self.config.batch_size,
            learning_rate=params['learning_rate']
        )
        
        # Calculate validation score
        model.eval()
        with torch.no_grad():
            X_val_tensor = torch.FloatTensor(X_val_seq).to(self.device)
            y_pred = model(X_val_tensor).cpu().numpy().flatten()
            score = f1_score(y_val_seq, (y_pred > 0.5).astype(int), zero_division=0)
        
        return model, score
    
    def _train_final_model(self, X: np.ndarray, y: np.ndarray, 
                          params: Dict, max_epochs: int = None) -> nn.Module:
        """Train final model on all data."""
        if max_epochs is None:
            max_epochs = self.config.max_epochs
            
        # Create sequences
        seq_len = params['seq_len']
        X_seq, y_seq = create_sequences(X, y, seq_len)
        
        if len(X_seq) == 0:
            return None
        
        # Create model
        n_features = X_seq.shape[-1]
        model = LSTMModel(
            input_size=n_features,
            hidden_size=params['hidden_size'],
            num_layers=params['num_layers'],
            dropout=params['dropout'],
            bidirectional=params.get('bidirectional', False),
            output_size=1
        ).to(self.device)
        
        # Train on all data (using 80/20 split for validation)
        split_idx = int(len(X_seq) * 0.8)
        X_train = X_seq[:split_idx]
        y_train = y_seq[:split_idx]
        X_val = X_seq[split_idx:]
        y_val = y_seq[split_idx:]
        
        trained_model = train_model(
            model, X_train, y_train, X_val, y_val,
            self.device, epochs=max_epochs, batch_size=self.config.batch_size,
            learning_rate=params['learning_rate']
        )
        
        return trained_model
    
    def _create_wrapper(self) -> LSTMWrapper:
        """Create scikit-learn compatible wrapper."""
        return LSTMWrapper(
            lstm_model=self.best_model,
            seq_len=self.best_params['seq_len'],
            device=self.device,
            scaler=self.scaler
        )
    
    def _create_search_space(self, trial: optuna.Trial) -> Dict:
        """
        Create Optuna search space.
        
        Args:
            trial: Optuna trial
            
        Returns:
            Dictionary of hyperparameters
        """
        return {
            'seq_len': trial.suggest_int(
                'seq_len', 
                self.config.seq_len_min, 
                self.config.seq_len_max, 
                step=5
            ),
            'hidden_size': trial.suggest_int(
                'hidden_size',
                self.config.hidden_size_min,
                self.config.hidden_size_max,
                step=16
            ),
            'num_layers': trial.suggest_int(
                'num_layers',
                self.config.num_layers_min,
                self.config.num_layers_max
            ),
            'dropout': trial.suggest_float(
                'dropout',
                self.config.dropout_min,
                self.config.dropout_max
            ),
            'learning_rate': trial.suggest_loguniform(
                'learning_rate',
                self.config.learning_rate_min,
                self.config.learning_rate_max
            ),
            'weight_decay': trial.suggest_loguniform(
                'weight_decay',
                self.config.weight_decay_min,
                self.config.weight_decay_max
            ),
            'gradient_clip': trial.suggest_float(
                'gradient_clip',
                self.config.gradient_clip_min,
                self.config.gradient_clip_max
            ),
            'batch_size': trial.suggest_categorical(
                'batch_size',
                [16, 32, 64, 128]
            ),
            'use_attention': trial.suggest_categorical(
                'use_attention',
                [False, True]
            )
        }
    
    def _get_pruner(self) -> optuna.pruners.BasePruner:
        """Get Optuna pruner based on configuration."""
        if self.config.pruner_type == 'median':
            return MedianPruner(n_startup_trials=5, n_warmup_steps=10)
        elif self.config.pruner_type == 'hyperband':
            return HyperbandPruner(min_resource=1, max_resource=self.config.max_epochs)
        else:
            return MedianPruner()
    
    def _create_model(self, params: Dict, input_size: int) -> nn.Module:
        """
        Create LSTM model.
        
        Args:
            params: Hyperparameters
            input_size: Number of input features
            
        Returns:
            LSTM model
        """
        if params.get('use_attention', False):
            model = AttentionLSTM(
                input_size=input_size,
                hidden_size=params['hidden_size'],
                num_layers=params['num_layers'],
                dropout=params['dropout']
            )
        else:
            model = LSTMModel(
                input_size=input_size,
                hidden_size=params['hidden_size'],
                num_layers=params['num_layers'],
                dropout=params['dropout']
            )
        
        return model.to(self.device)
    
    def _create_objective(self, X: pd.DataFrame, y: pd.Series):
        """
        Create Optuna objective function.
        
        Args:
            X: Training features
            y: Training labels
            
        Returns:
            Objective function
        """
        # Import here to avoid circular dependency
        from src.data.splits import PurgedKFold
        
        def objective(trial: optuna.Trial) -> float:
            # Get hyperparameters
            params = self._create_search_space(trial)
            
            # Create sequences
            X_seq, y_seq = create_sequences(X, y, params['seq_len'])
            
            # Scale features
            X_seq_reshaped = X_seq.reshape(-1, X_seq.shape[-1])
            X_seq_reshaped = self.scaler.fit_transform(X_seq_reshaped)
            X_seq = X_seq_reshaped.reshape(X_seq.shape)
            
            # Cross-validation
            cv_scores = []
            cv = PurgedKFold(
                n_splits=self.config.cv_folds,
                embargo=self.config.embargo
            )
            
            for fold, (train_idx, val_idx) in enumerate(cv.split(X_seq, y_seq)):
                # Split data
                X_train, X_val = X_seq[train_idx], X_seq[val_idx]
                y_train, y_val = y_seq[train_idx], y_seq[val_idx]
                
                # Create dataloaders
                train_dataset = TensorDataset(
                    torch.FloatTensor(X_train),
                    torch.FloatTensor(y_train)
                )
                val_dataset = TensorDataset(
                    torch.FloatTensor(X_val),
                    torch.FloatTensor(y_val)
                )
                
                train_loader = DataLoader(
                    train_dataset,
                    batch_size=params['batch_size'],
                    shuffle=False  # Important for time series
                )
                val_loader = DataLoader(
                    val_dataset,
                    batch_size=params['batch_size'],
                    shuffle=False
                )
                
                # Create model
                model = self._create_model(params, X_seq.shape[-1])
                
                # Create optimizer and criterion
                optimizer = optim.Adam(
                    model.parameters(),
                    lr=params['learning_rate'],
                    weight_decay=params['weight_decay']
                )
                criterion = nn.BCELoss()
                
                # Train model
                history = train_model(
                    model=model,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    criterion=criterion,
                    optimizer=optimizer,
                    device=self.device,
                    n_epochs=self.config.max_epochs,
                    early_stopping_patience=self.config.early_stopping_patience,
                    gradient_clip=params['gradient_clip'],
                    verbose=False
                )
                
                # Get validation predictions
                model.eval()
                val_preds = []
                with torch.no_grad():
                    for batch_X, _ in val_loader:
                        batch_X = batch_X.to(self.device)
                        outputs = model(batch_X)
                        val_preds.extend(outputs.cpu().numpy().flatten())
                
                val_preds = np.array(val_preds)
                
                # Check for constant predictions
                if check_constant_predictions(val_preds):
                    return 0.0
                
                # Calculate F1 score
                val_pred_binary = (val_preds >= 0.5).astype(int)
                score = f1_score(y_val, val_pred_binary)
                cv_scores.append(score)
                
                # Report intermediate value for pruning
                trial.report(score, fold)
                
                # Handle pruning
                if trial.should_prune():
                    raise optuna.TrialPruned()
            
            return np.mean(cv_scores)
        
        return objective
    
    def optimize(self, X: pd.DataFrame, y: pd.Series) -> optuna.Study:
        """
        Run Optuna optimization.
        
        Args:
            X: Training features
            y: Training labels
            
        Returns:
            Optuna study
        """
        log.info("starting_optuna_optimization", n_trials=self.config.n_trials)
        
        # Create study
        study = optuna.create_study(
            direction='maximize',
            pruner=self._get_pruner(),
            sampler=TPESampler(seed=self.config.seed)
        )
        
        # Create and optimize objective
        objective = self._create_objective(X, y)
        study.optimize(
            objective,
            n_trials=self.config.n_trials,
            show_progress_bar=True
        )
        
        # Store best parameters
        self.best_params = study.best_params
        self.best_score = study.best_value
        
        log.info(
            "optimization_completed",
            best_score=self.best_score,
            best_params=self.best_params
        )
        
        return study
    
    def fit_final_model(self, X: pd.DataFrame, y: pd.Series):
        """
        Fit final model with best parameters.
        
        Args:
            X: Training features
            y: Training labels
        """
        if self.best_params is None:
            raise ValueError("Must run optimize() first")
        
        log.info("fitting_final_model", params=self.best_params)
        
        # Create sequences
        X_seq, y_seq = create_sequences(X, y, self.best_params['seq_len'])
        
        # Scale features
        X_seq_reshaped = X_seq.reshape(-1, X_seq.shape[-1])
        X_seq_reshaped = self.scaler.fit_transform(X_seq_reshaped)
        X_seq = X_seq_reshaped.reshape(X_seq.shape)
        
        # Create model
        model = self._create_model(self.best_params, X_seq.shape[-1])
        
        # Create dataloader
        dataset = TensorDataset(
            torch.FloatTensor(X_seq),
            torch.FloatTensor(y_seq)
        )
        dataloader = DataLoader(
            dataset,
            batch_size=self.best_params['batch_size'],
            shuffle=False
        )
        
        # Train model
        optimizer = optim.Adam(
            model.parameters(),
            lr=self.best_params['learning_rate'],
            weight_decay=self.best_params['weight_decay']
        )
        criterion = nn.BCELoss()
        
        history = train_model(
            model=model,
            train_loader=dataloader,
            val_loader=dataloader,  # Using same for simplicity
            criterion=criterion,
            optimizer=optimizer,
            device=self.device,
            n_epochs=self.config.max_epochs,
            early_stopping_patience=self.config.early_stopping_patience * 2,
            gradient_clip=self.best_params['gradient_clip'],
            verbose=True
        )
        
        # Store model
        self.best_model = model
        
        # Create wrapper for sklearn compatibility
        wrapper = LSTMWrapper(
            lstm_model=model,
            seq_len=self.best_params['seq_len'],
            device=self.device,
            scaler=self.scaler
        )
        
        # Calibrate model
        log.info("calibrating_model")
        self.calibrator = CalibratedClassifierCV(
            wrapper,
            method='isotonic',
            cv='prefit'
        )
        self.calibrator.fit(X, y)
        
        # Optimize thresholds
        self._optimize_thresholds(X, y)
        
        log.info("model_training_completed")
    
    def _optimize_thresholds(self, X: pd.DataFrame, y: pd.Series):
        """Optimize classification thresholds."""
        # Get predictions
        y_pred_proba = self.predict_proba(X)[:, 1]
        
        # Optimize F1 threshold
        precision, recall, thresholds = precision_recall_curve(y, y_pred_proba)
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
        best_idx = np.argmax(f1_scores[:-1])
        self.threshold_f1 = thresholds[best_idx]
        
        # For EV threshold, use a simple heuristic
        self.threshold_ev = np.percentile(y_pred_proba, 70)
        
        log.info(
            "thresholds_optimized",
            threshold_f1=self.threshold_f1,
            threshold_ev=self.threshold_ev
        )
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict probabilities using calibrated model."""
        if self.calibrator is None:
            raise ValueError("Model not fitted")
        
        return self.calibrator.predict_proba(X)
    
    def predict(self, X: pd.DataFrame, use_ev_threshold: bool = False) -> np.ndarray:
        """Predict classes."""
        proba = self.predict_proba(X)[:, 1]
        threshold = self.threshold_ev if use_ev_threshold else self.threshold_f1
        return (proba >= threshold).astype(int)
    
    def get_params(self, deep: bool = True) -> Dict:
        """Get parameters for sklearn compatibility."""
        return {'config': self.config}
    
    def set_params(self, **params) -> 'LSTMOptuna':
        """Set parameters for sklearn compatibility."""
        for key, value in params.items():
            if key == 'config':
                self.config = value
            elif hasattr(self.config, key):
                setattr(self.config, key, value)
        return self
    
    def score(self, X: pd.DataFrame, y: np.ndarray) -> float:
        """Calculate accuracy score."""
        if self.wrapper is not None:
            return self.wrapper.score(X, y)
        y_pred = self.predict(X)
        return np.mean(y_pred == y)
    
    @property
    def feature_importances_(self):
        """Feature importances not available for LSTM."""
        raise NotImplementedError("Feature importances not available for LSTM models")