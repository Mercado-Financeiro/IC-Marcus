"""
Enhanced LSTM Optuna optimizer with production-grade features.

Integrates all advanced features:
- ASHA and SuccessiveHalving pruning
- Temperature scaling calibration  
- Walk-forward outer validation
- ECE and comprehensive calibration metrics
- Full determinism and reproducibility
- Expanded hyperparameter search space
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, Tuple, Optional, Any, List
import optuna
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, precision_recall_curve, auc
import warnings
import mlflow
from pathlib import Path

# Import our advanced components
from ...optuna.advanced_optimizer import AdvancedOptunaOptimizer, AdvancedOptimizerConfig
from ...calibration.temperature import TemperatureScaling, VectorScaling
from ...metrics.calibration import comprehensive_calibration_metrics, expected_calibration_error
from ...validation.walkforward import WalkForwardValidator, WalkForwardConfig
from ....utils.determinism_enhanced import set_full_determinism, DeterministicContext

# Import existing LSTM components
from .config import LSTMOptunaConfig
from .model import LSTMModel, AttentionLSTM
from .wrapper import LSTMWrapper
from .training import train_model, EarlyStopping
from .utils import (
    get_logger, create_sequences, get_device, calculate_metrics
)

log = get_logger()


class EnhancedLSTMOptuna:
    """
    Enhanced LSTM optimizer with state-of-the-art Bayesian optimization.
    
    Features:
    - Advanced pruning (ASHA, SuccessiveHalving, HyperBand)
    - Temperature scaling calibration
    - Walk-forward outer validation
    - ECE and comprehensive calibration metrics
    - Expanded search space with all hyperparameters
    - Full determinism and reproducibility
    """
    
    def __init__(self, config: Optional[LSTMOptunaConfig] = None):
        """
        Initialize enhanced LSTM optimizer.
        
        Args:
            config: LSTM configuration object
        """
        self.config = config or LSTMOptunaConfig()
        
        # Create advanced optimizer configuration
        self.optuna_config = AdvancedOptimizerConfig(
            n_trials=getattr(self.config, 'n_trials', 50),
            timeout=getattr(self.config, 'timeout', 7200),
            seed=getattr(self.config, 'seed', 42),
            sampler_type='tpe',
            pruner_type='asha',  # Use ASHA by default for LSTM
            pruner_params={
                'min_resource': 5,
                'max_resource': 100,
                'reduction_factor': 3.0
            },
            storage_url=getattr(self.config, 'storage_url', None),
            study_name=f"lstm_enhanced_{self.config.seed}",
            use_outer_cv=True,
            outer_cv_splits=3,
            inner_cv_splits=3,  # Reduced for LSTM (expensive)
            embargo=getattr(self.config, 'embargo', 10),
            calibration_method='temperature',  # Best for neural networks
            use_mlflow=True,
            mlflow_experiment='lstm_enhanced_optimization',
            primary_metric='f1_score',
            calibration_metrics=True,
            verbose=getattr(self.config, 'verbose', False)
        )
        
        # Initialize components
        self.optimizer = AdvancedOptunaOptimizer(self.optuna_config)
        self.device = get_device(getattr(self.config, 'device', 'auto'))
        self.scaler = StandardScaler()
        
        # Results storage
        self.best_model = None
        self.best_params = None
        self.best_score = -np.inf
        self.calibrator = None
        self.threshold_f1 = 0.5
        self.threshold_ev = 0.5
        self.feature_names_ = None
        self.wrapper = None
        
        # Set deterministic environment
        self.determinism_results = set_full_determinism(self.config.seed, verify=True)
        
        log.info(
            "enhanced_lstm_optuna_initialized",
            n_trials=self.optuna_config.n_trials,
            pruner_type=self.optuna_config.pruner_type,
            calibration_method=self.optuna_config.calibration_method,
            device=str(self.device)
        )
    
    def _create_expanded_search_space(self, trial: optuna.Trial) -> Dict[str, Any]:
        """
        Create expanded hyperparameter search space.
        
        Includes all parameters suggested in your feedback:
        - Bidirectional LSTM
        - Weight decay
        - Gradient clipping
        - Sequence stride
        - Advanced architectures
        """
        params = {
            # Architecture parameters
            'seq_len': trial.suggest_int('seq_len', 20, 200, step=10),
            'hidden_size': trial.suggest_int('hidden_size', 32, 256, step=16),
            'num_layers': trial.suggest_int('num_layers', 1, 4),
            'dropout': trial.suggest_float('dropout', 0.0, 0.5),
            'bidirectional': trial.suggest_categorical('bidirectional', [True, False]),
            
            # Training parameters
            'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True),
            'weight_decay': trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True),
            'gradient_clip': trial.suggest_float('gradient_clip', 0.1, 5.0),
            'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64, 128]),
            
            # Sequence parameters
            'stride': trial.suggest_int('stride', 1, 5),
            
            # Optimizer parameters
            'optimizer_type': trial.suggest_categorical('optimizer_type', ['adam', 'adamw', 'rmsprop']),
            'beta1': trial.suggest_float('beta1', 0.8, 0.99) if 'adam' in trial.suggest_categorical('optimizer_type', ['adam', 'adamw', 'rmsprop']) else 0.9,
            'beta2': trial.suggest_float('beta2', 0.9, 0.999) if 'adam' in trial.suggest_categorical('optimizer_type', ['adam', 'adamw', 'rmsprop']) else 0.999,
            
            # Scheduler parameters
            'scheduler_type': trial.suggest_categorical('scheduler_type', ['cosine', 'plateau', 'step', 'none']),
            'scheduler_patience': trial.suggest_int('scheduler_patience', 5, 15) if trial.suggest_categorical('scheduler_type', ['cosine', 'plateau', 'step', 'none']) == 'plateau' else 10,
            
            # Advanced architecture
            'use_attention': trial.suggest_categorical('use_attention', [True, False]),
            'attention_heads': trial.suggest_int('attention_heads', 1, 8) if trial.suggest_categorical('use_attention', [True, False]) else 1,
            
            # Regularization
            'layer_norm': trial.suggest_categorical('layer_norm', [True, False]),
            'residual_connection': trial.suggest_categorical('residual_connection', [True, False]),
            
            # Loss parameters
            'label_smoothing': trial.suggest_float('label_smoothing', 0.0, 0.1),
            'focal_loss_gamma': trial.suggest_float('focal_loss_gamma', 0.0, 3.0),
            
            # Early stopping
            'early_stopping_patience': trial.suggest_int('early_stopping_patience', 10, 30),
            'early_stopping_min_delta': trial.suggest_float('early_stopping_min_delta', 1e-6, 1e-3, log=True)
        }
        
        return params
    
    def _create_model_factory(self):
        """Create model factory function for the optimizer."""
        def model_factory(params: Dict[str, Any]) -> LSTMWrapper:
            """Factory function to create LSTM model with given parameters."""
            
            def fit_and_predict_model(X_train, y_train, X_val=None, y_val=None):
                """Fit model and return predictions."""
                
                # Create sequences with stride
                X_seq_train, y_seq_train = create_sequences(
                    X_train, y_train, 
                    params['seq_len'], 
                    stride=params['stride']
                )
                
                if X_val is not None and y_val is not None:
                    X_seq_val, y_seq_val = create_sequences(
                        X_val, y_val, 
                        params['seq_len'], 
                        stride=params['stride']
                    )
                else:
                    # Use part of training data for validation
                    split_idx = int(len(X_seq_train) * 0.8)
                    X_seq_val = X_seq_train[split_idx:]
                    y_seq_val = y_seq_train[split_idx:]
                    X_seq_train = X_seq_train[:split_idx]
                    y_seq_train = y_seq_train[:split_idx]
                
                if len(X_seq_train) == 0 or len(X_seq_val) == 0:
                    # Return dummy model if no sequences
                    class DummyModel:
                        def predict(self, X):
                            return np.zeros(len(X))
                        def predict_proba(self, X):
                            pred = self.predict(X)
                            return np.column_stack([1-pred, pred])
                    return DummyModel()
                
                # Scale features
                X_seq_train_flat = X_seq_train.reshape(-1, X_seq_train.shape[-1])
                X_seq_val_flat = X_seq_val.reshape(-1, X_seq_val.shape[-1])
                
                scaler = StandardScaler()
                X_seq_train_flat_scaled = scaler.fit_transform(X_seq_train_flat)
                X_seq_val_flat_scaled = scaler.transform(X_seq_val_flat)
                
                X_seq_train_scaled = X_seq_train_flat_scaled.reshape(X_seq_train.shape)
                X_seq_val_scaled = X_seq_val_flat_scaled.reshape(X_seq_val.shape)
                
                # Create model
                n_features = X_seq_train_scaled.shape[-1]
                
                if params['use_attention']:
                    model = AttentionLSTM(
                        input_size=n_features,
                        hidden_size=params['hidden_size'],
                        num_layers=params['num_layers'],
                        dropout=params['dropout'],
                        bidirectional=params['bidirectional'],
                        num_heads=params.get('attention_heads', 1)
                    )
                else:
                    model = LSTMModel(
                        input_size=n_features,
                        hidden_size=params['hidden_size'],
                        num_layers=params['num_layers'],
                        dropout=params['dropout'],
                        bidirectional=params['bidirectional'],
                        use_layer_norm=params.get('layer_norm', False),
                        use_residual=params.get('residual_connection', False)
                    )
                
                model = model.to(self.device)
                
                # Create data loaders
                train_dataset = TensorDataset(
                    torch.FloatTensor(X_seq_train_scaled),
                    torch.FloatTensor(y_seq_train)
                )
                val_dataset = TensorDataset(
                    torch.FloatTensor(X_seq_val_scaled),
                    torch.FloatTensor(y_seq_val)
                )
                
                train_loader = DataLoader(
                    train_dataset, 
                    batch_size=params['batch_size'], 
                    shuffle=False
                )
                val_loader = DataLoader(
                    val_dataset, 
                    batch_size=params['batch_size'], 
                    shuffle=False
                )
                
                # Create optimizer
                if params['optimizer_type'] == 'adam':
                    optimizer = optim.Adam(
                        model.parameters(),
                        lr=params['learning_rate'],
                        weight_decay=params['weight_decay'],
                        betas=(params.get('beta1', 0.9), params.get('beta2', 0.999))
                    )
                elif params['optimizer_type'] == 'adamw':
                    optimizer = optim.AdamW(
                        model.parameters(),
                        lr=params['learning_rate'],
                        weight_decay=params['weight_decay'],
                        betas=(params.get('beta1', 0.9), params.get('beta2', 0.999))
                    )
                else:  # rmsprop
                    optimizer = optim.RMSprop(
                        model.parameters(),
                        lr=params['learning_rate'],
                        weight_decay=params['weight_decay']
                    )
                
                # Create scheduler
                scheduler = None
                if params['scheduler_type'] == 'cosine':
                    scheduler = optim.lr_scheduler.CosineAnnealingLR(
                        optimizer, T_max=50
                    )
                elif params['scheduler_type'] == 'plateau':
                    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                        optimizer, 
                        patience=params['scheduler_patience'],
                        factor=0.5
                    )
                elif params['scheduler_type'] == 'step':
                    scheduler = optim.lr_scheduler.StepLR(
                        optimizer, step_size=20, gamma=0.5
                    )
                
                # Create loss function with advanced features
                pos_count = y_seq_train.sum()
                neg_count = len(y_seq_train) - pos_count
                pos_weight = torch.tensor(
                    neg_count / pos_count if pos_count > 0 else 1.0
                ).to(self.device)
                
                if params.get('focal_loss_gamma', 0) > 0:
                    # Focal loss (simplified)
                    class FocalLoss(nn.Module):
                        def __init__(self, gamma=2.0, pos_weight=None):
                            super().__init__()
                            self.gamma = gamma
                            self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
                        
                        def forward(self, inputs, targets):
                            bce_loss = self.bce(inputs, targets)
                            pt = torch.exp(-bce_loss)
                            focal_loss = (1 - pt) ** self.gamma * bce_loss
                            return focal_loss
                    
                    criterion = FocalLoss(
                        gamma=params['focal_loss_gamma'],
                        pos_weight=pos_weight
                    )
                else:
                    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
                
                # Train model
                history = train_model(
                    model=model,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    criterion=criterion,
                    optimizer=optimizer,
                    device=self.device,
                    n_epochs=100,  # Will be pruned if necessary
                    early_stopping_patience=params['early_stopping_patience'],
                    gradient_clip=params['gradient_clip'],
                    scheduler=scheduler,
                    verbose=False
                )
                
                # Create wrapper for sklearn compatibility
                wrapper = LSTMWrapper(
                    lstm_model=model,
                    seq_len=params['seq_len'],
                    stride=params['stride'],
                    device=self.device,
                    scaler=scaler
                )
                
                return wrapper
            
            # Create a model class that implements the required interface
            class ParametrizedLSTM:
                def __init__(self):
                    self.model = None
                    self.is_fitted = False
                
                def fit(self, X, y):
                    self.model = fit_and_predict_model(X, y)
                    self.is_fitted = True
                    return self
                
                def predict(self, X):
                    if not self.is_fitted:
                        raise ValueError("Model not fitted")
                    return self.model.predict(X)
                
                def predict_proba(self, X):
                    if not self.is_fitted:
                        raise ValueError("Model not fitted")
                    return self.model.predict_proba(X)
            
            return ParametrizedLSTM()
        
        return model_factory
    
    def optimize(self, X: pd.DataFrame, y: pd.Series) -> optuna.Study:
        """
        Run enhanced optimization with all advanced features.
        
        Args:
            X: Training features
            y: Training labels
            
        Returns:
            Completed Optuna study
        """
        log.info("starting_enhanced_optimization", 
                n_trials=self.optuna_config.n_trials,
                pruner_type=self.optuna_config.pruner_type)
        
        # Store feature names
        if hasattr(X, 'columns'):
            self.feature_names_ = X.columns.tolist()
        
        # Create model factory
        model_factory = self._create_model_factory()
        
        # Run optimization
        study = self.optimizer.optimize(
            X=X,
            y=y,
            model_factory=model_factory,
            param_factory=self._create_expanded_search_space
        )
        
        # Store results
        self.best_params = study.best_params
        self.best_score = study.best_value
        
        log.info("optimization_completed",
                best_score=self.best_score,
                n_trials=len(study.trials))
        
        return study
    
    def fit_final_model(self, X: pd.DataFrame, y: pd.Series):
        """
        Fit final model with best parameters and calibration.
        
        Args:
            X: Training features
            y: Training labels
        """
        if self.best_params is None:
            raise ValueError("Must run optimize() first")
        
        log.info("fitting_final_model", params=self.best_params)
        
        # Create and fit model with best parameters
        model_factory = self._create_model_factory()
        model = model_factory(self.best_params)
        model.fit(X, y)
        
        self.best_model = model
        
        # Temperature scaling calibration
        log.info("calibrating_model_with_temperature_scaling")
        
        # Get model predictions for calibration
        y_pred_proba = model.predict_proba(X)[:, 1]
        
        # Convert probabilities to logits for temperature scaling
        epsilon = 1e-7
        logits = np.log((y_pred_proba + epsilon) / (1 - y_pred_proba + epsilon))
        
        # Fit temperature scaling
        self.calibrator = TemperatureScaling(verbose=self.optuna_config.verbose)
        self.calibrator.fit(logits, y.values)
        
        # Optimize thresholds on calibrated probabilities
        calibrated_probs = self.calibrator.transform(logits)
        self._optimize_thresholds(y.values, calibrated_probs)
        
        # Calculate comprehensive calibration metrics
        cal_metrics = comprehensive_calibration_metrics(y.values, calibrated_probs)
        
        if self.optuna_config.use_mlflow:
            try:
                with mlflow.start_run():
                    mlflow.log_metrics(cal_metrics)
                    mlflow.log_metric('temperature', self.calibrator.temperature_)
                    mlflow.log_metric('threshold_f1', self.threshold_f1)
                    mlflow.log_metric('threshold_ev', self.threshold_ev)
            except Exception as e:
                warnings.warn(f"MLflow logging failed: {e}")
        
        log.info("model_training_completed",
                temperature=self.calibrator.temperature_,
                ece=cal_metrics.get('ece_uniform', 0),
                brier_score=cal_metrics.get('brier_score', 0))
    
    def _optimize_thresholds(self, y_true: np.ndarray, y_pred_proba: np.ndarray):
        """Optimize classification thresholds."""
        # Optimize F1 threshold
        precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
        best_idx = np.argmax(f1_scores[:-1])
        self.threshold_f1 = thresholds[best_idx]
        
        # Optimize EV threshold (using a simple heuristic)
        # In practice, this should use actual expected value calculation
        self.threshold_ev = np.percentile(y_pred_proba, 70)
        
        log.info("thresholds_optimized",
                threshold_f1=self.threshold_f1,
                threshold_ev=self.threshold_ev)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict calibrated probabilities."""
        if self.best_model is None or self.calibrator is None:
            raise ValueError("Model not fitted")
        
        # Get raw probabilities
        raw_probs = self.best_model.predict_proba(X)[:, 1]
        
        # Convert to logits and apply temperature scaling
        epsilon = 1e-7
        logits = np.log((raw_probs + epsilon) / (1 - raw_probs + epsilon))
        calibrated_probs = self.calibrator.transform(logits)
        
        # Return as 2D array for sklearn compatibility
        return np.column_stack([1 - calibrated_probs, calibrated_probs])
    
    def predict(self, X: pd.DataFrame, use_ev_threshold: bool = False) -> np.ndarray:
        """Predict classes using optimized thresholds."""
        proba = self.predict_proba(X)[:, 1]
        threshold = self.threshold_ev if use_ev_threshold else self.threshold_f1
        return (proba >= threshold).astype(int)
    
    def get_calibration_curve(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """Get calibration curve data for plotting."""
        if self.calibrator is None:
            raise ValueError("Model not fitted")
        
        y_pred_proba = self.predict_proba(X)[:, 1]
        ece, bin_stats = expected_calibration_error(y.values, y_pred_proba, n_bins=10)
        
        return {
            'bin_stats': bin_stats,
            'ece': ece,
            'temperature': self.calibrator.temperature_
        }
    
    def get_optimization_summary(self) -> Dict:
        """Get summary of optimization results."""
        if not hasattr(self, 'optimizer') or self.optimizer.study is None:
            return {}
        
        study = self.optimizer.study
        
        return {
            'n_trials': len(study.trials),
            'best_score': study.best_value,
            'best_params': study.best_params,
            'pruner_type': self.optuna_config.pruner_type,
            'sampler_type': self.optuna_config.sampler_type,
            'calibration_method': self.optuna_config.calibration_method,
            'determinism_verified': all(
                v for v in self.determinism_results.get('verification', {}).values()
                if v is not None
            ),
            'optimization_history': self.optimizer.get_optimization_history()
        }
    
    # Sklearn compatibility methods
    def get_params(self, deep: bool = True) -> Dict:
        """Get parameters for sklearn compatibility."""
        return {'config': self.config}
    
    def set_params(self, **params) -> 'EnhancedLSTMOptuna':
        """Set parameters for sklearn compatibility."""
        for key, value in params.items():
            if key == 'config':
                self.config = value
            elif hasattr(self.config, key):
                setattr(self.config, key, value)
        return self
    
    def score(self, X: pd.DataFrame, y: np.ndarray) -> float:
        """Calculate F1 score."""
        y_pred = self.predict(X)
        return f1_score(y, y_pred, zero_division=0)
    
    @property
    def feature_importances_(self):
        """Feature importances not available for LSTM."""
        raise NotImplementedError("Feature importances not available for LSTM models")