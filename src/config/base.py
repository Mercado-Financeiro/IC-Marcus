"""
Base configuration system with Pydantic for type validation and serialization.

This module provides a hierarchical configuration system where all configurations
inherit from BaseConfig, ensuring consistency, type safety, and easy serialization.
Memory-efficient loading with lazy initialization and caching is supported.
"""

import os
import json
import yaml
from pathlib import Path
from typing import Any, Dict, Optional, Type, TypeVar, Union, List, ClassVar
from datetime import datetime
from functools import lru_cache
from abc import ABC

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from pydantic_settings import BaseSettings


T = TypeVar('T', bound='BaseConfig')


class BaseConfig(BaseModel):
    """
    Base configuration class that all configs should inherit from.
    
    Features:
    - Automatic validation using Pydantic
    - JSON/YAML serialization and deserialization
    - Environment variable support
    - Type hints and documentation
    - Memory-efficient caching
    """
    
    model_config = ConfigDict(
        # Allow field names that might conflict with model methods
        protected_namespaces=(),
        # Enable environment variable loading
        env_file=".env",
        env_file_encoding="utf-8",
        # Allow extra fields for forward compatibility
        extra="forbid",
        # Validate on assignment for runtime safety
        validate_assignment=True,
        # Use enum values instead of names
        use_enum_values=True,
        # Validate default values
        validate_default=True,
        # Enable JSON schema generation
        json_schema_extra={
            "example": {}
        }
    )
    
    # Metadata
    config_version: str = Field(default="1.0.0", description="Configuration version for compatibility checking")
    created_at: Optional[datetime] = Field(default_factory=datetime.now, description="When config was created")
    description: Optional[str] = Field(default=None, description="Human-readable description of this config")
    
    # Caching
    _cache: ClassVar[Dict[str, Any]] = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return self.model_dump(exclude_none=False)
    
    def to_json(self, path: Optional[Union[str, Path]] = None, **kwargs) -> str:
        """
        Serialize configuration to JSON.
        
        Args:
            path: Optional path to save JSON file
            **kwargs: Additional arguments for json.dumps
            
        Returns:
            JSON string representation
        """
        json_str = self.model_dump_json(indent=2, **kwargs)
        
        if path:
            Path(path).write_text(json_str)
            
        return json_str
    
    def to_yaml(self, path: Optional[Union[str, Path]] = None) -> str:
        """
        Serialize configuration to YAML.
        
        Args:
            path: Optional path to save YAML file
            
        Returns:
            YAML string representation
        """
        import yaml
        
        yaml_str = yaml.dump(
            self.model_dump(mode='json'),
            default_flow_style=False,
            sort_keys=False
        )
        
        if path:
            Path(path).write_text(yaml_str)
            
        return yaml_str
    
    @classmethod
    def from_dict(cls: Type[T], data: Dict[str, Any]) -> T:
        """Create configuration from dictionary."""
        return cls(**data)
    
    @classmethod
    def from_json(cls: Type[T], path: Union[str, Path]) -> T:
        """Load configuration from JSON file."""
        data = json.loads(Path(path).read_text())
        return cls(**data)
    
    @classmethod
    def from_yaml(cls: Type[T], path: Union[str, Path]) -> T:
        """Load configuration from YAML file."""
        data = yaml.safe_load(Path(path).read_text())
        return cls(**data)
    
    @classmethod
    @lru_cache(maxsize=32)
    def from_file(cls: Type[T], path: Union[str, Path]) -> T:
        """
        Load configuration from file with caching.
        
        Supports JSON and YAML based on file extension.
        Results are cached for memory efficiency.
        """
        path = Path(path)
        
        if path.suffix in ['.json', '.jsonc']:
            return cls.from_json(path)
        elif path.suffix in ['.yaml', '.yml']:
            return cls.from_yaml(path)
        else:
            raise ValueError(f"Unsupported file type: {path.suffix}")
    
    def merge(self, other: 'BaseConfig', deep: bool = True) -> 'BaseConfig':
        """
        Merge this configuration with another.
        
        Args:
            other: Configuration to merge with
            deep: Whether to do deep merge for nested configs
            
        Returns:
            New merged configuration
        """
        if not isinstance(other, self.__class__):
            raise TypeError(f"Cannot merge {self.__class__} with {type(other)}")
        
        self_dict = self.model_dump()
        other_dict = other.model_dump()
        
        if deep:
            merged = self._deep_merge(self_dict, other_dict)
        else:
            merged = {**self_dict, **other_dict}
        
        return self.__class__(**merged)
    
    @staticmethod
    def _deep_merge(dict1: Dict, dict2: Dict) -> Dict:
        """Recursively merge two dictionaries."""
        result = dict1.copy()
        
        for key, value in dict2.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = BaseConfig._deep_merge(result[key], value)
            else:
                result[key] = value
                
        return result
    
    def validate_compatibility(self, required_version: str) -> bool:
        """Check if config version is compatible with required version."""
        from packaging import version
        
        return version.parse(self.config_version) >= version.parse(required_version)


class SystemConfig(BaseConfig):
    """System-level configuration for compute resources and environment."""
    
    # Device settings
    device: str = Field(default="auto", description="Device to use: 'auto', 'cpu', 'cuda', 'mps'")
    n_jobs: int = Field(default=1, description="Number of parallel jobs (-1 for all CPUs)")
    
    # Memory settings
    max_memory_gb: Optional[float] = Field(default=None, description="Maximum memory usage in GB")
    enable_memory_profiling: bool = Field(default=False, description="Enable memory profiling")
    
    # Determinism
    seed: int = Field(default=42, description="Random seed for reproducibility")
    deterministic: bool = Field(default=True, description="Enable deterministic mode")
    
    # Paths
    data_dir: Path = Field(default=Path("data"), description="Data directory")
    artifacts_dir: Path = Field(default=Path("artifacts"), description="Artifacts directory")
    cache_dir: Path = Field(default=Path(".cache"), description="Cache directory")
    
    # Logging
    log_level: str = Field(default="INFO", description="Logging level")
    verbose: bool = Field(default=False, description="Enable verbose output")
    
    @field_validator('device')
    @classmethod
    def validate_device(cls, v: str) -> str:
        """Validate and normalize device setting."""
        valid_devices = ['auto', 'cpu', 'cuda', 'gpu', 'mps']
        if v.lower() not in valid_devices:
            raise ValueError(f"Device must be one of {valid_devices}")
        
        # Normalize gpu -> cuda
        if v.lower() == 'gpu':
            return 'cuda'
            
        return v.lower()
    
    @field_validator('n_jobs')
    @classmethod
    def validate_n_jobs(cls, v: int) -> int:
        """Validate n_jobs setting."""
        if v == 0:
            raise ValueError("n_jobs cannot be 0")
        if v < -1:
            raise ValueError("n_jobs must be -1 or positive")
        return v


class DataConfig(BaseConfig):
    """Configuration for data loading and preprocessing."""
    
    # Data source
    symbols: List[str] = Field(default=["BTCUSDT"], description="Trading symbols")
    timeframe: str = Field(default="15m", description="Data timeframe")
    start_date: str = Field(default="2023-01-01", description="Start date for data")
    end_date: str = Field(default="2023-12-31", description="End date for data")
    
    # Preprocessing
    handle_missing: str = Field(default="interpolate", description="How to handle missing values")
    normalize: bool = Field(default=True, description="Whether to normalize features")
    remove_outliers: bool = Field(default=True, description="Whether to remove outliers")
    outlier_threshold: float = Field(default=3.0, description="Z-score threshold for outliers")
    
    # Labels
    label_type: str = Field(default="binary", description="Label type: 'binary', 'multiclass', 'regression'")
    label_horizon: int = Field(default=5, description="Prediction horizon in bars")
    label_threshold: float = Field(default=0.002, description="Threshold for binary labels")
    
    # Validation
    min_samples: int = Field(default=1000, description="Minimum samples required")
    check_stationarity: bool = Field(default=True, description="Check for stationarity")


class FeatureConfig(BaseConfig):
    """Configuration for feature engineering and selection."""
    
    # Feature types to include
    use_technical: bool = Field(default=True, description="Include technical indicators")
    use_price_features: bool = Field(default=True, description="Include price-based features")
    use_volume_features: bool = Field(default=True, description="Include volume-based features")
    use_microstructure: bool = Field(default=True, description="Include microstructure features")
    
    # Feature engineering
    lookback_periods: List[int] = Field(default=[5, 10, 20, 50], description="Lookback periods for features")
    rolling_windows: List[int] = Field(default=[10, 20, 50], description="Rolling window sizes")
    
    # Feature selection
    selection_method: str = Field(default="mutual_info", description="Feature selection method")
    max_features: Optional[int] = Field(default=None, description="Maximum number of features")
    min_importance: float = Field(default=0.01, description="Minimum feature importance")
    remove_correlated: bool = Field(default=True, description="Remove highly correlated features")
    correlation_threshold: float = Field(default=0.95, description="Correlation threshold")


class ValidationConfig(BaseConfig):
    """Configuration for model validation and cross-validation."""
    
    # Cross-validation
    cv_method: str = Field(default="time_series", description="CV method: 'time_series', 'purged', 'combo'")
    n_splits: int = Field(default=5, description="Number of CV splits")
    test_size: float = Field(default=0.2, description="Test set size")
    
    # Time series specific
    embargo: int = Field(default=10, description="Embargo period in bars")
    purge: int = Field(default=2, description="Purge period in bars")
    gap: int = Field(default=0, description="Gap between train and test")
    
    # Walk-forward
    use_walk_forward: bool = Field(default=True, description="Use walk-forward validation")
    walk_forward_window: int = Field(default=252, description="Walk-forward window size")
    walk_forward_step: int = Field(default=63, description="Walk-forward step size")
    
    # Metrics
    primary_metric: str = Field(default="f1_score", description="Primary optimization metric")
    metrics: List[str] = Field(
        default=["accuracy", "precision", "recall", "f1_score", "roc_auc", "pr_auc"],
        description="Metrics to calculate"
    )


class TrainingConfig(BaseConfig):
    """Base configuration for model training."""
    
    # Basic training
    batch_size: int = Field(default=32, description="Batch size for training")
    learning_rate: float = Field(default=0.001, description="Learning rate")
    max_epochs: int = Field(default=100, description="Maximum training epochs")
    
    # Early stopping
    early_stopping: bool = Field(default=True, description="Enable early stopping")
    early_stopping_patience: int = Field(default=10, description="Early stopping patience")
    early_stopping_min_delta: float = Field(default=0.0001, description="Minimum improvement")
    
    # Optimization
    optimizer: str = Field(default="adam", description="Optimizer type")
    weight_decay: float = Field(default=0.0001, description="Weight decay (L2 regularization)")
    gradient_clip: Optional[float] = Field(default=1.0, description="Gradient clipping value")
    
    # Learning rate schedule
    use_scheduler: bool = Field(default=True, description="Use learning rate scheduler")
    scheduler_type: str = Field(default="cosine", description="Scheduler type")
    warmup_steps: int = Field(default=0, description="Warmup steps")
    
    # Class weights
    use_class_weights: bool = Field(default=True, description="Use class weights for imbalanced data")
    class_weight_strategy: str = Field(default="balanced", description="Class weight strategy")


class ModelConfig(BaseConfig):
    """Base configuration for model architecture."""
    
    model_type: str = Field(description="Type of model: 'xgboost', 'lstm', 'ensemble'")
    model_name: str = Field(default="model", description="Name for the model")
    
    # Save/load
    save_path: Optional[Path] = Field(default=None, description="Path to save model")
    load_path: Optional[Path] = Field(default=None, description="Path to load model")
    
    # Calibration
    use_calibration: bool = Field(default=True, description="Use probability calibration")
    calibration_method: str = Field(default="isotonic", description="Calibration method")
    
    # Ensemble
    ensemble_method: Optional[str] = Field(default=None, description="Ensemble method if applicable")
    n_estimators: Optional[int] = Field(default=None, description="Number of estimators for ensemble")


class XGBoostConfig(ModelConfig):
    """XGBoost-specific configuration."""
    
    model_type: str = Field(default="xgboost", description="Model type")
    
    # Core parameters
    n_estimators: int = Field(default=100, description="Number of trees")
    max_depth: int = Field(default=6, description="Maximum tree depth")
    learning_rate: float = Field(default=0.3, description="Learning rate (eta)")
    
    # Tree-specific
    min_child_weight: int = Field(default=1, description="Minimum child weight")
    gamma: float = Field(default=0, description="Minimum loss reduction")
    subsample: float = Field(default=1.0, description="Subsample ratio")
    colsample_bytree: float = Field(default=1.0, description="Column subsample ratio")
    colsample_bylevel: float = Field(default=1.0, description="Column subsample by level")
    colsample_bynode: float = Field(default=1.0, description="Column subsample by node")
    
    # Regularization
    reg_alpha: float = Field(default=0, description="L1 regularization")
    reg_lambda: float = Field(default=1, description="L2 regularization")
    
    # Training
    tree_method: str = Field(default="hist", description="Tree construction algorithm")
    scale_pos_weight: Optional[float] = Field(default=None, description="Balance for unbalanced classes")
    
    # Early stopping
    early_stopping_rounds: Optional[int] = Field(default=None, description="Early stopping rounds")
    
    # GPU settings
    gpu_id: Optional[int] = Field(default=None, description="GPU device ID")
    predictor: str = Field(default="auto", description="Predictor type")
    
    @field_validator('subsample', 'colsample_bytree', 'colsample_bylevel', 'colsample_bynode')
    @classmethod
    def validate_ratios(cls, v: float) -> float:
        """Validate subsample ratios are in (0, 1]."""
        if not 0 < v <= 1:
            raise ValueError(f"Subsample ratios must be in (0, 1], got {v}")
        return v


class LSTMConfig(ModelConfig):
    """LSTM-specific configuration."""
    
    model_type: str = Field(default="lstm", description="Model type")
    
    # Architecture
    hidden_size: int = Field(default=128, description="Hidden layer size")
    num_layers: int = Field(default=2, description="Number of LSTM layers")
    dropout: float = Field(default=0.2, description="Dropout rate")
    bidirectional: bool = Field(default=False, description="Use bidirectional LSTM")
    
    # Sequence
    seq_len: int = Field(default=20, description="Sequence length")
    
    # Attention
    use_attention: bool = Field(default=True, description="Use attention mechanism")
    attention_heads: int = Field(default=4, description="Number of attention heads")
    
    # Batch normalization
    use_batch_norm: bool = Field(default=True, description="Use batch normalization")
    
    # Input/Output
    input_size: Optional[int] = Field(default=None, description="Input feature size")
    output_size: int = Field(default=2, description="Output size (classes)")
    
    @field_validator('dropout')
    @classmethod
    def validate_dropout(cls, v: float) -> float:
        """Validate dropout is in [0, 1)."""
        if not 0 <= v < 1:
            raise ValueError(f"Dropout must be in [0, 1), got {v}")
        return v


class OptunaConfig(BaseConfig):
    """Configuration for Optuna hyperparameter optimization."""
    
    # Basic settings
    n_trials: int = Field(default=100, description="Number of optimization trials")
    timeout: Optional[int] = Field(default=3600, description="Timeout in seconds")
    n_jobs: int = Field(default=1, description="Parallel trials")
    
    # Sampler
    sampler_type: str = Field(default="tpe", description="Sampler: 'tpe', 'random', 'cmaes'")
    sampler_params: Dict[str, Any] = Field(default_factory=dict, description="Sampler parameters")
    
    # Pruner
    pruner_type: str = Field(default="median", description="Pruner: 'median', 'asha', 'hyperband'")
    pruner_params: Dict[str, Any] = Field(default_factory=dict, description="Pruner parameters")
    
    # Storage
    storage: Optional[str] = Field(default=None, description="Optuna storage URL")
    study_name: Optional[str] = Field(default=None, description="Study name")
    load_if_exists: bool = Field(default=True, description="Load existing study")
    
    # Direction
    direction: str = Field(default="maximize", description="Optimization direction")
    
    # Logging
    optuna_verbose: bool = Field(default=False, description="Optuna verbosity")
    show_progress_bar: bool = Field(default=True, description="Show progress bar")


class APIConfig(BaseSettings):
    """API-specific configuration with environment variable support."""
    
    model_config = ConfigDict(
        env_prefix="API_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False
    )
    
    # Server settings
    host: str = Field(default="0.0.0.0", description="API host")
    port: int = Field(default=8000, description="API port")
    reload: bool = Field(default=False, description="Auto-reload on changes")
    workers: int = Field(default=1, description="Number of worker processes")
    
    # Security
    api_key: Optional[str] = Field(default=None, description="API key for authentication")
    cors_origins: List[str] = Field(default=["*"], description="CORS allowed origins")
    
    # Rate limiting
    rate_limit_requests: int = Field(default=100, description="Requests per window")
    rate_limit_window: int = Field(default=60, description="Time window in seconds")
    
    # Cache
    cache_enabled: bool = Field(default=True, description="Enable caching")
    cache_ttl: int = Field(default=300, description="Cache TTL in seconds")
    
    # MLflow
    mlflow_tracking_uri: str = Field(default="artifacts/mlruns", description="MLflow tracking URI")
    model_name: str = Field(default="crypto_model", description="Model name in registry")
    model_stage: str = Field(default="Production", description="Model stage")
    
    # WebSocket
    ws_enabled: bool = Field(default=True, description="Enable WebSocket support")
    ws_heartbeat_interval: int = Field(default=30, description="WebSocket heartbeat interval")
    ws_max_connections: int = Field(default=100, description="Maximum WebSocket connections")


class ConfigManager:
    """
    Centralized configuration manager with lazy loading and caching.
    
    This provides a singleton-like interface for managing all configurations
    with memory-efficient loading and automatic caching.
    """
    
    _instance: Optional['ConfigManager'] = None
    _configs: Dict[str, BaseConfig] = {}
    
    def __new__(cls):
        """Ensure singleton pattern."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    @classmethod
    def register(cls, name: str, config: BaseConfig) -> None:
        """Register a configuration."""
        cls._configs[name] = config
    
    @classmethod
    def get(cls, name: str) -> Optional[BaseConfig]:
        """Get a registered configuration."""
        return cls._configs.get(name)
    
    @classmethod
    def list_configs(cls) -> List[str]:
        """List all registered configuration names."""
        return list(cls._configs.keys())
    
    @classmethod
    @lru_cache(maxsize=32)
    def load_from_file(cls, path: Union[str, Path], config_class: Type[T]) -> T:
        """Load configuration from file with caching."""
        return config_class.from_file(path)
    
    @classmethod
    def clear_cache(cls) -> None:
        """Clear all cached configurations."""
        cls._configs.clear()
        cls.load_from_file.cache_clear()


# Convenience functions
def get_config(name: str) -> Optional[BaseConfig]:
    """Get a registered configuration by name."""
    return ConfigManager.get(name)


def load_config(path: Union[str, Path], config_class: Type[T] = BaseConfig) -> T:
    """Load configuration from file."""
    return ConfigManager.load_from_file(path, config_class)


def save_config(config: BaseConfig, path: Union[str, Path]) -> None:
    """Save configuration to file."""
    path = Path(path)
    
    if path.suffix in ['.json', '.jsonc']:
        config.to_json(path)
    elif path.suffix in ['.yaml', '.yml']:
        config.to_yaml(path)
    else:
        raise ValueError(f"Unsupported file type: {path.suffix}")


# Default configurations for quick start
DEFAULT_CONFIGS = {
    'system': SystemConfig(),
    'data': DataConfig(),
    'features': FeatureConfig(),
    'validation': ValidationConfig(),
    'training': TrainingConfig(),
    'xgboost': XGBoostConfig(),
    'lstm': LSTMConfig(),
    'optuna': OptunaConfig(),
    'api': APIConfig()
}

# Register default configs
for name, config in DEFAULT_CONFIGS.items():
    ConfigManager.register(name, config)