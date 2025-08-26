"""
Configuration migration utilities for backward compatibility.

This module provides utilities to migrate from old configuration formats
to the new unified configuration system while maintaining backward compatibility.
"""

from typing import Any, Dict, Optional, Type, Union
from pathlib import Path
import warnings
from dataclasses import asdict, is_dataclass

from .base import (
    BaseConfig, XGBoostConfig, LSTMConfig, OptunaConfig,
    TrainingConfig, SystemConfig, APIConfig, DataConfig
)


class ConfigMigrator:
    """Handles migration from old configuration formats to new unified system."""
    
    @staticmethod
    def migrate_xgboost_config(old_config: Any) -> XGBoostConfig:
        """
        Migrate from old XGBoostOptunaConfig to new XGBoostConfig.
        
        Args:
            old_config: Old configuration object or dict
            
        Returns:
            New XGBoostConfig instance
        """
        if isinstance(old_config, dict):
            data = old_config
        elif is_dataclass(old_config):
            data = asdict(old_config)
        elif hasattr(old_config, '__dict__'):
            data = old_config.__dict__
        else:
            data = {}
        
        # Map old field names to new ones
        mapping = {
            'n_estimators': 'n_estimators',
            'max_depth': 'max_depth',
            'learning_rate': 'learning_rate',
            'tree_method': 'tree_method',
            'device': 'gpu_id',  # Convert device to gpu_id if needed
            'n_jobs': 'n_jobs',
            'early_stopping_rounds': 'early_stopping_rounds',
            'deterministic': 'deterministic',
            'seed': 'seed'
        }
        
        new_data = {}
        for old_key, new_key in mapping.items():
            if old_key in data:
                value = data[old_key]
                # Special handling for device -> gpu_id
                if old_key == 'device' and new_key == 'gpu_id':
                    if value == 'cuda' or value == 'gpu':
                        new_data['gpu_id'] = 0
                        new_data['tree_method'] = 'gpu_hist'
                    else:
                        new_data['gpu_id'] = None
                else:
                    new_data[new_key] = value
        
        # Handle calibration settings
        if 'calibration_method' in data:
            new_data['calibration_method'] = data['calibration_method']
            new_data['use_calibration'] = True
        
        return XGBoostConfig(**new_data)
    
    @staticmethod
    def migrate_lstm_config(old_config: Any) -> LSTMConfig:
        """
        Migrate from old LSTMOptunaConfig to new LSTMConfig.
        
        Args:
            old_config: Old configuration object or dict
            
        Returns:
            New LSTMConfig instance
        """
        if isinstance(old_config, dict):
            data = old_config
        elif is_dataclass(old_config):
            data = asdict(old_config)
        elif hasattr(old_config, '__dict__'):
            data = old_config.__dict__
        else:
            data = {}
        
        # Extract relevant fields
        new_data = {
            'hidden_size': data.get('hidden_size_max', 128),  # Use max as default
            'num_layers': data.get('num_layers_max', 2),
            'dropout': data.get('dropout_max', 0.2),
            'seq_len': data.get('seq_len_max', 20)
        }
        
        # Note: batch_size and device are not part of LSTMConfig base model,
        # they would be in a separate training or system config
        
        return LSTMConfig(**new_data)
    
    @staticmethod
    def migrate_optuna_config(old_config: Any) -> OptunaConfig:
        """
        Migrate from old Optuna configurations to new OptunaConfig.
        
        Args:
            old_config: Old configuration object or dict
            
        Returns:
            New OptunaConfig instance
        """
        if isinstance(old_config, dict):
            data = old_config
        elif is_dataclass(old_config):
            data = asdict(old_config)
        elif hasattr(old_config, '__dict__'):
            data = old_config.__dict__
        else:
            data = {}
        
        new_data = {
            'n_trials': data.get('n_trials', 100),
            'timeout': data.get('timeout'),
            'sampler_type': data.get('sampler_type', 'tpe'),
            'pruner_type': data.get('pruner_type', 'median'),
            'storage': data.get('storage_url'),
            'study_name': data.get('study_name'),
            'load_if_exists': data.get('load_if_exists', True),
            'n_jobs': data.get('n_jobs', 1)
        }
        
        # Handle sampler and pruner params
        if 'sampler_params' in data:
            new_data['sampler_params'] = data['sampler_params']
        if 'pruner_params' in data:
            new_data['pruner_params'] = data['pruner_params']
        
        # Determine direction from metric
        if 'primary_metric' in data:
            metric = data['primary_metric']
            if metric in ['loss', 'error', 'mse', 'mae']:
                new_data['direction'] = 'minimize'
            else:
                new_data['direction'] = 'maximize'
        
        return OptunaConfig(**new_data)
    
    @staticmethod
    def migrate_api_config(old_settings: Any) -> APIConfig:
        """
        Migrate from old Settings class to new APIConfig.
        
        Args:
            old_settings: Old Settings object or dict
            
        Returns:
            New APIConfig instance
        """
        if isinstance(old_settings, dict):
            data = old_settings
        elif hasattr(old_settings, 'model_dump'):
            data = old_settings.model_dump()
        elif hasattr(old_settings, '__dict__'):
            data = old_settings.__dict__
        else:
            data = {}
        
        new_data = {
            'host': data.get('host', '0.0.0.0'),
            'port': data.get('port', 8000),
            'reload': data.get('reload', False),
            'api_key': data.get('api_key'),
            'mlflow_tracking_uri': data.get('mlflow_tracking_uri', 'artifacts/mlruns'),
            'model_name': data.get('model_name', 'crypto_model'),
            'model_stage': data.get('model_stage', 'Production'),
            'cache_ttl': data.get('cache_ttl', 300),
            'rate_limit_requests': data.get('rate_limit_requests', 100),
            'rate_limit_window': data.get('rate_limit_window', 60),
            'ws_heartbeat_interval': data.get('ws_heartbeat_interval', 30),
            'ws_max_connections': data.get('ws_max_connections', 100)
        }
        
        # Handle Redis settings
        if 'redis_host' in data:
            new_data['cache_enabled'] = True
        
        return APIConfig(**new_data)
    
    @staticmethod
    def auto_migrate(old_config: Any, target_class: Optional[Type[BaseConfig]] = None) -> BaseConfig:
        """
        Automatically detect and migrate configuration.
        
        Args:
            old_config: Old configuration object
            target_class: Optional target configuration class
            
        Returns:
            Migrated configuration
        """
        # Try to detect config type from class name or attributes
        if target_class:
            class_name = target_class.__name__
        elif hasattr(old_config, '__class__'):
            class_name = old_config.__class__.__name__
        else:
            # Try to infer from attributes
            if hasattr(old_config, 'tree_method') or 'tree_method' in str(old_config):
                class_name = 'XGBoostConfig'
            elif hasattr(old_config, 'hidden_size') or 'hidden_size' in str(old_config):
                class_name = 'LSTMConfig'
            elif hasattr(old_config, 'n_trials') or 'n_trials' in str(old_config):
                class_name = 'OptunaConfig'
            elif hasattr(old_config, 'api_key') or 'api_key' in str(old_config):
                class_name = 'APIConfig'
            else:
                warnings.warn(f"Could not detect config type for {old_config}")
                return BaseConfig()
        
        # Map to migration method
        migrations = {
            'XGBoostOptunaConfig': ConfigMigrator.migrate_xgboost_config,
            'XGBoostConfig': ConfigMigrator.migrate_xgboost_config,
            'LSTMOptunaConfig': ConfigMigrator.migrate_lstm_config,
            'LSTMConfig': ConfigMigrator.migrate_lstm_config,
            'OptunaConfig': ConfigMigrator.migrate_optuna_config,
            'AdvancedOptimizerConfig': ConfigMigrator.migrate_optuna_config,
            'Settings': ConfigMigrator.migrate_api_config,
            'APIConfig': ConfigMigrator.migrate_api_config
        }
        
        # Try to find matching migration
        for pattern, migrate_func in migrations.items():
            if pattern in class_name:
                try:
                    return migrate_func(old_config)
                except Exception as e:
                    warnings.warn(f"Migration failed for {class_name}: {e}")
                    break
        
        # Default: try to create base config
        warnings.warn(f"No specific migration for {class_name}, using BaseConfig")
        return BaseConfig()


class BackwardCompatibilityWrapper:
    """
    Wrapper to provide backward compatibility for old configuration usage.
    
    This allows old code to continue working while gradually migrating
    to the new configuration system.
    """
    
    def __init__(self, new_config: BaseConfig, old_class_name: str):
        """
        Initialize compatibility wrapper.
        
        Args:
            new_config: New configuration instance
            old_class_name: Name of old configuration class for reference
        """
        self._config = new_config
        self._old_class = old_class_name
        self._warned = set()
    
    def __getattr__(self, name: str) -> Any:
        """
        Get attribute from new config with deprecation warning.
        
        Args:
            name: Attribute name
            
        Returns:
            Attribute value from new configuration
        """
        if name not in self._warned:
            warnings.warn(
                f"Accessing '{name}' through {self._old_class} is deprecated. "
                f"Please use {self._config.__class__.__name__} instead.",
                DeprecationWarning,
                stacklevel=2
            )
            self._warned.add(name)
        
        return getattr(self._config, name)
    
    def __setattr__(self, name: str, value: Any) -> None:
        """Set attribute with deprecation warning."""
        if name.startswith('_'):
            # Internal attributes
            super().__setattr__(name, value)
        else:
            if name not in self._warned:
                warnings.warn(
                    f"Setting '{name}' through {self._old_class} is deprecated. "
                    f"Please use {self._config.__class__.__name__} instead.",
                    DeprecationWarning,
                    stacklevel=2
                )
                self._warned.add(name)
            
            setattr(self._config, name, value)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for compatibility."""
        return self._config.to_dict()
    
    def __repr__(self) -> str:
        """String representation."""
        return f"BackwardCompatibilityWrapper({self._config})"


def create_compatible_config(old_class_name: str, **kwargs) -> BackwardCompatibilityWrapper:
    """
    Create a backward-compatible configuration.
    
    Args:
        old_class_name: Name of old configuration class
        **kwargs: Configuration parameters
        
    Returns:
        Wrapped configuration with backward compatibility
    """
    # Map old class names to new ones
    class_mapping = {
        'XGBoostOptunaConfig': XGBoostConfig,
        'LSTMOptunaConfig': LSTMConfig,
        'AdvancedOptimizerConfig': OptunaConfig,
        'Settings': APIConfig,
        'TrainingConfig': TrainingConfig
    }
    
    new_class = class_mapping.get(old_class_name, BaseConfig)
    new_config = new_class(**kwargs)
    
    return BackwardCompatibilityWrapper(new_config, old_class_name)