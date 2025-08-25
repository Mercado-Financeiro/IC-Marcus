"""LSTM Optuna optimization orchestrator."""

import gc
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, Tuple, Optional, Any
import optuna
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, precision_recall_curve, auc, brier_score_loss

# Import our custom modules
try:
    from ...metrics.quality_gates import QualityGates
    from ...metrics.pr_auc import calculate_pr_auc_normalized
    from ...calibration.beta import BetaCalibration
    CUSTOM_MODULES = True
except ImportError:
    CUSTOM_MODULES = False

from .config import LSTMOptunaConfig
from .model import LSTMModel, AttentionLSTM
from .wrapper import LSTMWrapper
from .training import train_model
from .utils import (
    get_logger, set_lstm_deterministic, check_constant_predictions,
    create_sequences, get_device, calculate_metrics
)
from ....utils.memory_utils import (
    memory_cleanup, cleanup_after_trial, safe_delete, 
    cleanup_model, DataLoaderCleanup, log_memory_usage
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
        """Objective function for Optuna optimization with PR-AUC focus."""
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
        
        # Initialize quality gates if available
        gates = QualityGates() if CUSTOM_MODULES else None
        
        # Perform cross-validation with embargo
        from src.features.validation.temporal import TemporalValidator, TemporalValidationConfig
        val_config = TemporalValidationConfig(n_splits=self.config.cv_folds, embargo=10)
        validator = TemporalValidator(val_config)
        scores = []
        pr_aucs = []
        
        for fold, (train_idx, val_idx) in enumerate(validator.split(X, y, strategy='purged_kfold')):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Train model and get PR-AUC
            model, pr_auc, val_proba = self._train_model_with_pr_auc(X_train, y_train, X_val, y_val, params, trial, fold)
            
            if model is None:
                # Training failed, return bad score
                return -1.0
            
            # Calculate normalized PR-AUC
            prevalence = y_val.mean()
            if CUSTOM_MODULES:
                _, pr_auc_norm, _ = calculate_pr_auc_normalized(y_val, val_proba)
            else:
                pr_auc_norm = (pr_auc - prevalence) / (1 - prevalence) if prevalence < 1 else 0
            
            # Check quality gates
            if CUSTOM_MODULES and gates:
                pr_gate = gates.check_pr_auc_gate(y_val, val_proba, return_details=False)
                if not pr_gate['passed'] and fold > 0:  # Give first fold a chance
                    log.warning(f"Trial {trial.number} failed PR-AUC gate at fold {fold}")
                    raise optuna.TrialPruned()
            
            scores.append(pr_auc_norm)
            pr_aucs.append(pr_auc)
            
            # Clean up model and data after fold
            if model is not None:
                cleanup_model(model)
            safe_delete(X_train, X_val, y_train, y_val)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            
            # CRITICAL: Report PR-AUC for pruning
            trial.report(pr_auc_norm, fold)
            
            # CRITICAL: Check if should prune
            if trial.should_prune():
                log.info(f"Pruning trial {trial.number} at fold {fold} (PR-AUC: {pr_auc:.4f})")
                raise optuna.TrialPruned()
        
        # Final cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        # Return mean normalized PR-AUC as optimization target
        mean_score = np.mean(scores)
        log.info(f"Trial {trial.number} completed: mean PR-AUC = {np.mean(pr_aucs):.4f}, normalized = {mean_score:.4f}")
        return mean_score
    
    def _train_model_with_pr_auc(self, X_train: np.ndarray, y_train: np.ndarray, 
                                 X_val: np.ndarray, y_val: np.ndarray, 
                                 params: Dict, trial: optuna.Trial, fold: int,
                                 max_epochs: int = None) -> Tuple[nn.Module, float, np.ndarray]:
        """Train model with PR-AUC optimization and proper pruning."""
        if max_epochs is None:
            max_epochs = self.config.max_epochs
            
        # Create sequences
        seq_len = params['seq_len']
        X_train_seq, y_train_seq = create_sequences(X_train, y_train, seq_len)
        X_val_seq, y_val_seq = create_sequences(X_val, y_val, seq_len)
        
        if len(X_train_seq) == 0 or len(X_val_seq) == 0:
            return None, 0.0, np.array([])
        
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
        
        # Setup optimizer and loss
        optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'])
        criterion = nn.BCEWithLogitsLoss()
        
        # Training loop with epoch-level pruning
        best_pr_auc = 0
        patience_counter = 0
        
        for epoch in range(max_epochs):
            # Train epoch
            model.train()
            train_loss = 0
            for batch_start in range(0, len(X_train_seq), self.config.batch_size):
                batch_end = min(batch_start + self.config.batch_size, len(X_train_seq))
                X_batch = torch.FloatTensor(X_train_seq[batch_start:batch_end]).to(self.device)
                y_batch = torch.FloatTensor(y_train_seq[batch_start:batch_end]).to(self.device)
                
                optimizer.zero_grad()
                outputs = model(X_batch).squeeze()
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                del X_batch, y_batch, outputs
            
            # Validate and calculate PR-AUC
            model.eval()
            with torch.no_grad():
                X_val_tensor = torch.FloatTensor(X_val_seq).to(self.device)
                y_pred_logits = model(X_val_tensor).cpu().numpy().flatten()
                y_pred_proba = 1 / (1 + np.exp(-y_pred_logits))  # Sigmoid
                del X_val_tensor
            
            # Calculate PR-AUC
            precision, recall, _ = precision_recall_curve(y_val_seq, y_pred_proba)
            pr_auc = auc(recall, precision)
            
            # CRITICAL: Report to Optuna for pruning (every 5 epochs)
            if epoch % 5 == 0:
                # Calculate normalized PR-AUC for reporting
                prevalence = y_val_seq.mean()
                pr_auc_norm = (pr_auc - prevalence) / (1 - prevalence) if prevalence < 1 else 0
                
                # Report current performance
                intermediate_value = pr_auc_norm
                trial.report(intermediate_value, fold * max_epochs + epoch)
                
                # Check if should prune
                if trial.should_prune():
                    log.info(f"Pruning trial {trial.number} at epoch {epoch} (PR-AUC: {pr_auc:.4f})")
                    cleanup_model(model)
                    raise optuna.TrialPruned()
            
            # Early stopping based on PR-AUC
            if pr_auc > best_pr_auc:
                best_pr_auc = pr_auc
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.config.early_stopping_patience:
                    break
        
        # Final validation
        model.eval()
        with torch.no_grad():
            X_val_tensor = torch.FloatTensor(X_val_seq).to(self.device)
            y_pred_logits = model(X_val_tensor).cpu().numpy().flatten()
            y_pred_proba = 1 / (1 + np.exp(-y_pred_logits))  # Sigmoid
            del X_val_tensor
        
        # Clean up training data
        safe_delete(X_train_seq, y_train_seq, X_val_seq, y_val_seq)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        return model, best_pr_auc, y_pred_proba
    
    def _train_model(self, X_train: np.ndarray, y_train: np.ndarray, 
                     X_val: np.ndarray, y_val: np.ndarray, 
                     params: Dict, max_epochs: int = None) -> Tuple[nn.Module, float]:
        """Legacy training method for compatibility."""
        # Use new method but return only model and score
        model, pr_auc, _ = self._train_model_with_pr_auc(
            X_train, y_train, X_val, y_val, params, 
            optuna.Trial(None, None), 0, max_epochs
        )
        return model, pr_auc
    
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
        
        @cleanup_after_trial
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
                
                with DataLoaderCleanup(DataLoader(
                    train_dataset,
                    batch_size=params['batch_size'],
                    shuffle=False  # Important for time series
                )) as train_loader, DataLoaderCleanup(DataLoader(
                    val_dataset,
                    batch_size=params['batch_size'],
                    shuffle=False
                )) as val_loader:
                    
                    # Create model
                    model = self._create_model(params, X_seq.shape[-1])
                    
                    # Create optimizer and criterion
                    optimizer = optim.Adam(
                        model.parameters(),
                        lr=params['learning_rate'],
                        weight_decay=params['weight_decay']
                    )
                    # Calculate pos_weight for imbalanced data
                    pos_count = y_train.sum()
                    neg_count = len(y_train) - pos_count
                    pos_weight = torch.tensor(neg_count / pos_count if pos_count > 0 else 1.0).to(self.device)
                    
                    # Use BCEWithLogitsLoss for better numerical stability
                    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
                    
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
                            # Clean up batch tensor
                            del batch_X, outputs
                    
                    val_preds = np.array(val_preds)
                    
                    # Check for constant predictions
                    if check_constant_predictions(val_preds):
                        return 0.0
                    
                    # Calculate PR-AUC as primary metric
                    precision, recall, _ = precision_recall_curve(y_val, val_preds)
                    pr_auc = auc(recall, precision)
                    
                    # Normalize PR-AUC
                    prevalence = y_val.mean()
                    if CUSTOM_MODULES:
                        _, pr_auc_norm, _ = calculate_pr_auc_normalized(y_val, val_preds)
                    else:
                        pr_auc_norm = (pr_auc - prevalence) / (1 - prevalence) if prevalence < 1 else 0
                    
                    cv_scores.append(pr_auc_norm)
                    
                    # CRITICAL: Report PR-AUC for pruning
                    trial.report(pr_auc_norm, fold)
                
                    # Handle pruning
                    if trial.should_prune():
                        # Clean up memory before pruning
                        cleanup_model(model)
                        safe_delete(optimizer, criterion, train_dataset, val_dataset)
                        safe_delete(X_train, X_val, y_train, y_val, val_preds)
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        gc.collect()
                        raise optuna.TrialPruned()
                    
                    # Clean up after each fold
                    cleanup_model(model)
                    safe_delete(optimizer, criterion, train_dataset, val_dataset, val_preds)
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    gc.collect()
            
            # Final cleanup after all folds
            del X_seq, y_seq
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            
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
        
        # Create and optimize objective with memory management
        objective = self._create_objective(X, y)
        
        # Add memory monitoring callback
        def memory_callback(study, trial):
            if trial.number % 5 == 0:  # Log every 5 trials for LSTM
                log_memory_usage(f"LSTM Trial {trial.number}")
        
        study.optimize(
            objective,
            n_trials=self.config.n_trials,
            show_progress_bar=True,
            callbacks=[memory_callback]
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