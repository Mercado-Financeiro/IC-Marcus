"""
Adapters to refactor existing configurations to use the new base system.

These adapters provide drop-in replacements for existing configuration classes,
maintaining backward compatibility while using the new unified system under the hood.
"""

import os
import warnings
from typing import Optional, Dict, Any
from pathlib import Path

from .base import (
    XGBoostConfig as NewXGBoostConfig,
    LSTMConfig as NewLSTMConfig,
    OptunaConfig as NewOptunaConfig,
    APIConfig as NewAPIConfig,
    SystemConfig as NewSystemConfig,
    TrainingConfig as NewTrainingConfig
)
from .migrations import BackwardCompatibilityWrapper, ConfigMigrator


# API Configuration Adapter (replaces src/api/config.py)
class Settings(NewAPIConfig):
    """
    Drop-in replacement for src/api/config.py Settings class.
    Maintains backward compatibility while using new configuration system.
    """
    
    def __init__(self, **kwargs):
        """Initialize with backward compatibility."""
        # Map old environment variables
        if not kwargs.get('mlflow_tracking_uri'):
            kwargs['mlflow_tracking_uri'] = os.getenv("MLFLOW_TRACKING_URI", "artifacts/mlruns")
        if not kwargs.get('model_name'):
            kwargs['model_name'] = os.getenv("MODEL_NAME", "crypto_xgb")
        if not kwargs.get('model_stage'):
            kwargs['model_stage'] = os.getenv("MODEL_STAGE", "Production")
        if not kwargs.get('api_key'):
            kwargs['api_key'] = os.getenv("API_KEY", "")
        
        # Handle Redis settings for backward compatibility
        redis_host = os.getenv("REDIS_HOST", "localhost")
        if redis_host and redis_host != "localhost":
            kwargs['cache_enabled'] = True
        
        redis_port = os.getenv("REDIS_PORT", "6379")
        cache_ttl = os.getenv("CACHE_TTL", "300")
        if cache_ttl:
            kwargs['cache_ttl'] = int(cache_ttl)
        
        # Max batch size (not in new config, but we can add as extra)
        self.max_batch_size = int(kwargs.pop('max_batch_size', 100))
        
        # Redis settings (keep for compatibility but not used in new system)
        self.redis_host = redis_host
        self.redis_port = int(redis_port)
        
        super().__init__(**kwargs)
    
    def __getattr__(self, name):
        """Provide backward compatibility for old attribute names."""
        # Map old names to new ones
        mapping = {
            'model_version': 'model_stage',  # model_version -> model_stage
        }
        
        if name in mapping:
            warnings.warn(
                f"Attribute '{name}' is deprecated, use '{mapping[name]}' instead",
                DeprecationWarning,
                stacklevel=2
            )
            return getattr(self, mapping[name])
        
        # For attributes not in the new config, return default values
        if name == 'model_version':
            return 'latest'
        
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")


# Create global settings instance for backward compatibility
settings = Settings()


# LSTM Configuration Adapter (replaces src/models/lstm/optuna/config.py)
class LSTMOptunaConfig(NewLSTMConfig):
    """
    Drop-in replacement for src/models/lstm/optuna/config.py LSTMOptunaConfig.
    Maintains backward compatibility while using new configuration system.
    """
    
    def __init__(self, **kwargs):
        """Initialize with backward compatibility for old parameter names."""
        # Map old parameters to new ones
        param_mapping = {
            'seq_len_min': 'seq_len',
            'seq_len_max': 'seq_len',
            'hidden_size_min': 'hidden_size',
            'hidden_size_max': 'hidden_size',
            'num_layers_min': 'num_layers',
            'num_layers_max': 'num_layers',
            'dropout_min': 'dropout',
            'dropout_max': 'dropout'
        }
        
        # Handle min/max parameters by using max value
        for old_param, new_param in param_mapping.items():
            if old_param in kwargs:
                if old_param.endswith('_max'):
                    # Use max value
                    kwargs[new_param] = kwargs.pop(old_param)
                elif old_param.endswith('_min') and new_param not in kwargs:
                    # Use min value only if max not provided
                    kwargs[new_param] = kwargs.pop(old_param)
                else:
                    kwargs.pop(old_param)  # Remove min value if max exists
        
        # Store Optuna-specific parameters
        self.n_trials = kwargs.pop('n_trials', 50)
        self.cv_folds = kwargs.pop('cv_folds', 3)
        self.embargo = kwargs.pop('embargo', 10)
        self.pruner_type = kwargs.pop('pruner_type', 'median')
        self.early_stopping_patience = kwargs.pop('early_stopping_patience', 10)
        self.max_epochs = kwargs.pop('max_epochs', 100)
        self.batch_size = kwargs.pop('batch_size', 32)
        
        # Learning rate parameters
        self.learning_rate_min = kwargs.pop('learning_rate_min', 1e-4)
        self.learning_rate_max = kwargs.pop('learning_rate_max', 1e-2)
        self.weight_decay_min = kwargs.pop('weight_decay_min', 1e-5)
        self.weight_decay_max = kwargs.pop('weight_decay_max', 1e-3)
        self.gradient_clip_min = kwargs.pop('gradient_clip_min', 0.5)
        self.gradient_clip_max = kwargs.pop('gradient_clip_max', 2.0)
        
        # System settings
        self.use_mlflow = kwargs.pop('use_mlflow', False)
        self.seed = kwargs.pop('seed', 42)
        self.device = kwargs.pop('device', 'auto')
        self.verbose = kwargs.pop('verbose', False)
        
        # Validation settings
        self.threshold_std = kwargs.pop('threshold_std', 0.005)
        self.min_unique = kwargs.pop('min_unique', 10)
        
        # Store ranges for backward compatibility
        self.seq_len_min = kwargs.get('seq_len', 10)
        self.seq_len_max = kwargs.get('seq_len', 60)
        self.hidden_size_min = kwargs.get('hidden_size', 32)
        self.hidden_size_max = kwargs.get('hidden_size', 256)
        self.num_layers_min = kwargs.get('num_layers', 1)
        self.num_layers_max = kwargs.get('num_layers', 3)
        self.dropout_min = kwargs.get('dropout', 0.1)
        self.dropout_max = kwargs.get('dropout', 0.5)
        
        super().__init__(**kwargs)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with all parameters."""
        base_dict = super().to_dict()
        
        # Add Optuna-specific parameters
        base_dict.update({
            'n_trials': self.n_trials,
            'cv_folds': self.cv_folds,
            'embargo': self.embargo,
            'pruner_type': self.pruner_type,
            'early_stopping_patience': self.early_stopping_patience,
            'max_epochs': self.max_epochs,
            'batch_size': self.batch_size,
            'seed': self.seed,
            'device': self.device,
            'use_mlflow': self.use_mlflow,
            'verbose': self.verbose
        })
        
        return base_dict


# XGBoost Configuration Adapter
class XGBoostOptunaConfig(NewXGBoostConfig):
    """
    Drop-in replacement for XGBoost Optuna configurations.
    Maintains backward compatibility while using new configuration system.
    """
    
    def __init__(self, **kwargs):
        """Initialize with backward compatibility."""
        # Store Optuna-specific parameters
        self.n_trials = kwargs.pop('n_trials', 100)
        self.timeout = kwargs.pop('timeout', None)
        self.seed = kwargs.pop('seed', 42)
        
        # Sampler and pruner settings
        self.sampler_type = kwargs.pop('sampler_type', 'tpe')
        self.sampler_params = kwargs.pop('sampler_params', {})
        self.pruner_type = kwargs.pop('pruner_type', 'hyperband')
        self.pruner_params = kwargs.pop('pruner_params', {})
        
        # Storage settings
        self.storage_url = kwargs.pop('storage_url', None)
        self.study_name = kwargs.pop('study_name', None)
        self.load_if_exists = kwargs.pop('load_if_exists', True)
        
        # Validation settings
        self.use_outer_cv = kwargs.pop('use_outer_cv', True)
        self.outer_cv_splits = kwargs.pop('outer_cv_splits', 3)
        self.inner_cv_splits = kwargs.pop('inner_cv_splits', 5)
        self.embargo = kwargs.pop('embargo', 10)
        
        # Calibration settings
        self.calibration_method = kwargs.pop('calibration_method', 'auto')
        self.calibration_selection_metric = kwargs.pop('calibration_selection_metric', 'brier')
        
        # MLflow settings
        self.use_mlflow = kwargs.pop('use_mlflow', True)
        self.mlflow_experiment = kwargs.pop('mlflow_experiment', 'xgboost_enhanced_optimization')
        
        # Metrics
        self.primary_metric = kwargs.pop('primary_metric', 'pr_auc')
        self.eval_metric = kwargs.pop('eval_metric', 'aucpr')
        
        # Determinism
        self.deterministic = kwargs.pop('deterministic', True)
        self.verbose = kwargs.pop('verbose', False)
        
        # Device and parallelism
        if 'device' in kwargs:
            device = kwargs.pop('device')
            if device in ['gpu', 'cuda']:
                kwargs['gpu_id'] = 0
                kwargs['tree_method'] = 'gpu_hist'
        
        # n_jobs for determinism
        self.n_jobs = kwargs.pop('n_jobs', 1)
        
        super().__init__(**kwargs)


# Training Configuration Adapter
class TrainingConfig(NewTrainingConfig):
    """
    Generic training configuration adapter.
    Can be used as a base for various training scripts.
    """
    
    def __init__(self, **kwargs):
        """Initialize training configuration with defaults."""
        # Map common parameters
        if 'epochs' in kwargs:
            kwargs['max_epochs'] = kwargs.pop('epochs')
        
        if 'lr' in kwargs:
            kwargs['learning_rate'] = kwargs.pop('lr')
        
        if 'patience' in kwargs:
            kwargs['early_stopping_patience'] = kwargs.pop('patience')
        
        super().__init__(**kwargs)


# Advanced Optimizer Configuration Adapter
class AdvancedOptimizerConfig(NewOptunaConfig):
    """
    Adapter for AdvancedOptimizerConfig used in advanced optimization modules.
    """
    
    def __init__(self, **kwargs):
        """Initialize with advanced optimizer parameters."""
        # Map advanced parameters
        self.use_outer_cv = kwargs.pop('use_outer_cv', False)
        self.outer_cv_splits = kwargs.pop('outer_cv_splits', 3)
        self.inner_cv_splits = kwargs.pop('inner_cv_splits', 5)
        self.embargo = kwargs.pop('embargo', 10)
        
        self.calibration_method = kwargs.pop('calibration_method', None)
        self.use_mlflow = kwargs.pop('use_mlflow', False)
        self.mlflow_experiment = kwargs.pop('mlflow_experiment', None)
        
        self.primary_metric = kwargs.pop('primary_metric', 'f1_score')
        self.calibration_metrics = kwargs.pop('calibration_metrics', False)
        self.verbose = kwargs.pop('verbose', False)
        
        super().__init__(**kwargs)


# Factory function to create compatible configurations
def create_config(config_type: str, **kwargs) -> Any:
    """
    Factory function to create configurations with backward compatibility.
    
    Args:
        config_type: Type of configuration to create
        **kwargs: Configuration parameters
        
    Returns:
        Configuration instance
    """
    config_classes = {
        'settings': Settings,
        'api': Settings,
        'lstm': LSTMOptunaConfig,
        'lstm_optuna': LSTMOptunaConfig,
        'xgboost': XGBoostOptunaConfig,
        'xgboost_optuna': XGBoostOptunaConfig,
        'training': TrainingConfig,
        'advanced_optimizer': AdvancedOptimizerConfig,
        'optuna': AdvancedOptimizerConfig
    }
    
    config_class = config_classes.get(config_type.lower())
    if not config_class:
        raise ValueError(f"Unknown configuration type: {config_type}")
    
    return config_class(**kwargs)


# Provide module-level backward compatibility
def __getattr__(name):
    """Module-level attribute access for backward compatibility."""
    # Map old class names to new ones
    class_mapping = {
        'Config': TrainingConfig,
        'ModelConfig': NewXGBoostConfig,  # Default to XGBoost
        'SystemConfig': NewSystemConfig
    }
    
    if name in class_mapping:
        warnings.warn(
            f"Importing '{name}' directly is deprecated. "
            f"Please use the specific configuration class.",
            DeprecationWarning,
            stacklevel=2
        )
        return class_mapping[name]
    
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")