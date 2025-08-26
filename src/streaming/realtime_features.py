"""
Real-time feature engineering for streaming data.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Callable, Union, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
from collections import deque, defaultdict
import threading
import time
from abc import ABC, abstractmethod

from .windowing import WindowFrame, WindowManager
from ..features.technical_indicators import TechnicalIndicators
from ..features.microstructure_features import MicrostructureFeatures

logger = logging.getLogger(__name__)


@dataclass
class FeatureConfig:
    """Feature configuration."""
    name: str
    feature_type: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    window_size: Optional[int] = None
    dependencies: List[str] = field(default_factory=list)
    output_columns: List[str] = field(default_factory=list)


class StreamingFeature(ABC):
    """Abstract base class for streaming features."""
    
    def __init__(self, config: FeatureConfig):
        self.config = config
        self.name = config.name
    
    @abstractmethod
    def compute(self, data: Union[pd.DataFrame, Dict[str, Any]], context: Dict[str, Any]) -> Dict[str, Any]:
        """Compute feature values."""
        pass
    
    @abstractmethod
    def reset(self):
        """Reset feature state."""
        pass


class MovingAverageFeature(StreamingFeature):
    """Moving average feature with multiple windows."""
    
    def __init__(self, config: FeatureConfig):
        super().__init__(config)
        self.windows = config.parameters.get('windows', [5, 10, 20])
        self.column = config.parameters.get('column', 'close')
        
        # Circular buffers for each window
        self.buffers = {
            window: deque(maxlen=window) for window in self.windows
        }
    
    def compute(self, data: Union[pd.DataFrame, Dict[str, Any]], context: Dict[str, Any]) -> Dict[str, Any]:
        """Compute moving averages."""
        if isinstance(data, dict):
            value = data.get(self.column)
        else:
            value = data[self.column].iloc[-1] if len(data) > 0 else None
        
        if value is None:
            return {}
        
        results = {}
        
        # Add value to all buffers
        for window in self.windows:
            self.buffers[window].append(value)
            
            # Calculate moving average
            if len(self.buffers[window]) > 0:
                ma = np.mean(self.buffers[window])
                results[f'{self.name}_ma_{window}'] = ma
        
        return results
    
    def reset(self):
        """Reset all buffers."""
        for buffer in self.buffers.values():
            buffer.clear()


class RSIFeature(StreamingFeature):
    """Streaming RSI calculation."""
    
    def __init__(self, config: FeatureConfig):
        super().__init__(config)
        self.window = config.parameters.get('window', 14)
        self.column = config.parameters.get('column', 'close')
        
        # Price history for RSI calculation
        self.prices = deque(maxlen=self.window + 1)
        self.gains = deque(maxlen=self.window)
        self.losses = deque(maxlen=self.window)
        
        # Running averages
        self.avg_gain = None
        self.avg_loss = None
    
    def compute(self, data: Union[pd.DataFrame, Dict[str, Any]], context: Dict[str, Any]) -> Dict[str, Any]:
        """Compute RSI."""
        if isinstance(data, dict):
            value = data.get(self.column)
        else:
            value = data[self.column].iloc[-1] if len(data) > 0 else None
        
        if value is None:
            return {}
        
        self.prices.append(value)
        
        if len(self.prices) < 2:
            return {}
        
        # Calculate price change
        change = self.prices[-1] - self.prices[-2]
        gain = max(0, change)
        loss = max(0, -change)
        
        self.gains.append(gain)
        self.losses.append(loss)
        
        # Calculate RSI once we have enough data
        if len(self.gains) >= self.window:
            if self.avg_gain is None:
                # Initial calculation
                self.avg_gain = np.mean(self.gains)
                self.avg_loss = np.mean(self.losses)
            else:
                # Wilder's smoothing
                self.avg_gain = (self.avg_gain * (self.window - 1) + gain) / self.window
                self.avg_loss = (self.avg_loss * (self.window - 1) + loss) / self.window
            
            if self.avg_loss != 0:
                rs = self.avg_gain / self.avg_loss
                rsi = 100 - (100 / (1 + rs))
                return {f'{self.name}_rsi': rsi}
        
        return {}
    
    def reset(self):
        """Reset RSI state."""
        self.prices.clear()
        self.gains.clear()
        self.losses.clear()
        self.avg_gain = None
        self.avg_loss = None


class VWAPFeature(StreamingFeature):
    """Volume Weighted Average Price."""
    
    def __init__(self, config: FeatureConfig):
        super().__init__(config)
        self.reset_period = config.parameters.get('reset_period', 'daily')
        
        # Cumulative values
        self.cumulative_volume = 0
        self.cumulative_pv = 0
        self.last_reset = datetime.now().date()
    
    def compute(self, data: Union[pd.DataFrame, Dict[str, Any]], context: Dict[str, Any]) -> Dict[str, Any]:
        """Compute VWAP."""
        if isinstance(data, dict):
            price = data.get('close') or data.get('price')
            volume = data.get('volume', 1)
            timestamp = data.get('timestamp', datetime.now())
        else:
            if len(data) == 0:
                return {}
            price = data['close'].iloc[-1] if 'close' in data.columns else data['price'].iloc[-1]
            volume = data['volume'].iloc[-1] if 'volume' in data.columns else 1
            timestamp = data.index[-1] if hasattr(data.index, 'to_pydatetime') else datetime.now()
        
        if price is None:
            return {}
        
        # Check if we need to reset (new day)
        if isinstance(timestamp, str):
            timestamp = pd.to_datetime(timestamp)
        
        current_date = timestamp.date() if hasattr(timestamp, 'date') else datetime.now().date()
        
        if current_date != self.last_reset and self.reset_period == 'daily':
            self.cumulative_volume = 0
            self.cumulative_pv = 0
            self.last_reset = current_date
        
        # Update cumulative values
        self.cumulative_pv += price * volume
        self.cumulative_volume += volume
        
        # Calculate VWAP
        vwap = self.cumulative_pv / self.cumulative_volume if self.cumulative_volume > 0 else price
        
        return {
            f'{self.name}_vwap': vwap,
            f'{self.name}_volume_cum': self.cumulative_volume
        }
    
    def reset(self):
        """Reset VWAP state."""
        self.cumulative_volume = 0
        self.cumulative_pv = 0
        self.last_reset = datetime.now().date()


class BollingerBandsFeature(StreamingFeature):
    """Streaming Bollinger Bands."""
    
    def __init__(self, config: FeatureConfig):
        super().__init__(config)
        self.window = config.parameters.get('window', 20)
        self.num_std = config.parameters.get('num_std', 2)
        self.column = config.parameters.get('column', 'close')
        
        # Price buffer
        self.prices = deque(maxlen=self.window)
    
    def compute(self, data: Union[pd.DataFrame, Dict[str, Any]], context: Dict[str, Any]) -> Dict[str, Any]:
        """Compute Bollinger Bands."""
        if isinstance(data, dict):
            value = data.get(self.column)
        else:
            value = data[self.column].iloc[-1] if len(data) > 0 else None
        
        if value is None:
            return {}
        
        self.prices.append(value)
        
        if len(self.prices) < self.window:
            return {}
        
        # Calculate bands
        prices_array = np.array(self.prices)
        sma = np.mean(prices_array)
        std = np.std(prices_array)
        
        upper_band = sma + (self.num_std * std)
        lower_band = sma - (self.num_std * std)
        
        # Calculate position within bands
        bb_position = (value - lower_band) / (upper_band - lower_band) if upper_band != lower_band else 0.5
        
        return {
            f'{self.name}_bb_upper': upper_band,
            f'{self.name}_bb_lower': lower_band,
            f'{self.name}_bb_middle': sma,
            f'{self.name}_bb_position': bb_position,
            f'{self.name}_bb_width': (upper_band - lower_band) / sma if sma != 0 else 0
        }
    
    def reset(self):
        """Reset Bollinger Bands state."""
        self.prices.clear()


class RealtimeFeatureEngine:
    """
    Real-time feature engineering engine for streaming data.
    
    Features:
    - Multiple feature types (technical indicators, microstructure, custom)
    - Dependency management
    - State management for stateful features
    - Performance optimization
    - Feature versioning
    """
    
    def __init__(self):
        """Initialize feature engine."""
        self.features: Dict[str, StreamingFeature] = {}
        self.feature_configs: Dict[str, FeatureConfig] = {}
        self.dependency_graph: Dict[str, List[str]] = {}
        
        # Feature state
        self.feature_values: Dict[str, Any] = {}
        self.feature_lock = threading.RLock()
        
        # Performance metrics
        self.feature_timings: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self.computation_count = 0
        
        logger.info("RealtimeFeatureEngine initialized")
    
    def register_feature(self, config: FeatureConfig) -> bool:
        """Register a new feature."""
        try:
            # Create feature instance
            feature = self._create_feature(config)
            
            with self.feature_lock:
                self.features[config.name] = feature
                self.feature_configs[config.name] = config
                self.dependency_graph[config.name] = config.dependencies.copy()
            
            logger.info(f"Registered feature: {config.name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to register feature {config.name}: {e}")
            return False
    
    def _create_feature(self, config: FeatureConfig) -> StreamingFeature:
        """Create feature instance based on type."""
        if config.feature_type == "moving_average":
            return MovingAverageFeature(config)
        elif config.feature_type == "rsi":
            return RSIFeature(config)
        elif config.feature_type == "vwap":
            return VWAPFeature(config)
        elif config.feature_type == "bollinger_bands":
            return BollingerBandsFeature(config)
        else:
            raise ValueError(f"Unknown feature type: {config.feature_type}")
    
    def compute_features(
        self,
        data: Union[pd.DataFrame, Dict[str, Any], WindowFrame],
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Compute all features for given data.
        
        Args:
            data: Input data (DataFrame, dict, or WindowFrame)
            context: Additional context information
            
        Returns:
            Dictionary of computed feature values
        """
        start_time = time.time()
        context = context or {}
        
        # Convert WindowFrame to DataFrame if needed
        if isinstance(data, WindowFrame):
            data = data.to_dataframe()
        
        # Resolve feature dependencies and compute in order
        computation_order = self._resolve_dependencies()
        
        all_results = {}
        
        with self.feature_lock:
            for feature_name in computation_order:
                if feature_name not in self.features:
                    continue
                
                try:
                    feature_start = time.time()
                    
                    # Prepare context with previously computed features
                    feature_context = {**context, 'features': all_results}
                    
                    # Compute feature
                    feature_results = self.features[feature_name].compute(data, feature_context)
                    
                    # Store results
                    all_results.update(feature_results)
                    self.feature_values[feature_name] = feature_results
                    
                    # Record timing
                    feature_time = (time.time() - feature_start) * 1000
                    self.feature_timings[feature_name].append(feature_time)
                    
                except Exception as e:
                    logger.error(f"Error computing feature {feature_name}: {e}")
                    continue
        
        self.computation_count += 1
        total_time = (time.time() - start_time) * 1000
        
        # Add metadata
        all_results['_metadata'] = {
            'computation_time_ms': total_time,
            'feature_count': len(computation_order),
            'timestamp': datetime.now().isoformat()
        }
        
        return all_results
    
    def _resolve_dependencies(self) -> List[str]:
        """Resolve feature dependencies using topological sort."""
        # Simple topological sort
        in_degree = {name: 0 for name in self.features.keys()}
        
        # Calculate in-degrees
        for feature_name, deps in self.dependency_graph.items():
            if feature_name in self.features:
                for dep in deps:
                    if dep in in_degree:
                        in_degree[feature_name] += 1
        
        # Find features with no dependencies
        queue = [name for name, degree in in_degree.items() if degree == 0]
        result = []
        
        while queue:
            current = queue.pop(0)
            result.append(current)
            
            # Reduce in-degree for dependent features
            for feature_name, deps in self.dependency_graph.items():
                if current in deps and feature_name in in_degree:
                    in_degree[feature_name] -= 1
                    if in_degree[feature_name] == 0:
                        queue.append(feature_name)
        
        return result
    
    def compute_features_for_window(self, window: WindowFrame) -> Dict[str, Any]:
        """Compute features for a completed window."""
        df = window.to_dataframe()
        
        if df.empty:
            return {}
        
        context = {
            'window_id': window.window_id,
            'window_start': window.start_time,
            'window_end': window.end_time,
            'record_count': len(df)
        }
        
        return self.compute_features(df, context)
    
    def batch_compute_features(
        self,
        data_batch: List[Union[pd.DataFrame, Dict[str, Any]]],
        parallel: bool = True
    ) -> List[Dict[str, Any]]:
        """Compute features for a batch of data."""
        if not parallel:
            # Sequential processing
            return [self.compute_features(data) for data in data_batch]
        
        # Parallel processing (simplified)
        results = []
        for data in data_batch:
            result = self.compute_features(data)
            results.append(result)
        
        return results
    
    def get_feature_values(self, feature_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """Get current feature values."""
        with self.feature_lock:
            if feature_names is None:
                return self.feature_values.copy()
            else:
                return {name: self.feature_values.get(name, {}) for name in feature_names}
    
    def reset_features(self, feature_names: Optional[List[str]] = None):
        """Reset feature states."""
        with self.feature_lock:
            if feature_names is None:
                feature_names = list(self.features.keys())
            
            for feature_name in feature_names:
                if feature_name in self.features:
                    self.features[feature_name].reset()
                    self.feature_values[feature_name] = {}
            
            logger.info(f"Reset {len(feature_names)} features")
    
    def remove_feature(self, feature_name: str) -> bool:
        """Remove a feature."""
        with self.feature_lock:
            if feature_name in self.features:
                del self.features[feature_name]
                del self.feature_configs[feature_name]
                del self.dependency_graph[feature_name]
                if feature_name in self.feature_values:
                    del self.feature_values[feature_name]
                
                logger.info(f"Removed feature: {feature_name}")
                return True
            return False
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        stats = {
            'total_computations': self.computation_count,
            'registered_features': len(self.features),
            'feature_timings': {}
        }
        
        for feature_name, timings in self.feature_timings.items():
            if timings:
                stats['feature_timings'][feature_name] = {
                    'avg_time_ms': sum(timings) / len(timings),
                    'max_time_ms': max(timings),
                    'min_time_ms': min(timings),
                    'count': len(timings)
                }
        
        return stats
    
    def export_features_config(self) -> List[Dict[str, Any]]:
        """Export feature configurations."""
        with self.feature_lock:
            configs = []
            for config in self.feature_configs.values():
                configs.append({
                    'name': config.name,
                    'feature_type': config.feature_type,
                    'parameters': config.parameters,
                    'dependencies': config.dependencies,
                    'output_columns': config.output_columns
                })
            return configs
    
    def import_features_config(self, configs: List[Dict[str, Any]]) -> int:
        """Import feature configurations."""
        successful = 0
        
        for config_dict in configs:
            try:
                config = FeatureConfig(**config_dict)
                if self.register_feature(config):
                    successful += 1
            except Exception as e:
                logger.error(f"Failed to import feature config: {e}")
        
        logger.info(f"Imported {successful}/{len(configs)} feature configurations")
        return successful


# Convenience functions for creating common features
def create_technical_indicators_config() -> List[FeatureConfig]:
    """Create common technical indicators configurations."""
    configs = []
    
    # Moving averages
    configs.append(FeatureConfig(
        name="ma_fast",
        feature_type="moving_average",
        parameters={'windows': [5, 10], 'column': 'close'}
    ))
    
    configs.append(FeatureConfig(
        name="ma_slow",
        feature_type="moving_average",
        parameters={'windows': [20, 50], 'column': 'close'}
    ))
    
    # RSI
    configs.append(FeatureConfig(
        name="rsi_14",
        feature_type="rsi",
        parameters={'window': 14, 'column': 'close'}
    ))
    
    # VWAP
    configs.append(FeatureConfig(
        name="vwap_daily",
        feature_type="vwap",
        parameters={'reset_period': 'daily'}
    ))
    
    # Bollinger Bands
    configs.append(FeatureConfig(
        name="bb_20",
        feature_type="bollinger_bands",
        parameters={'window': 20, 'num_std': 2, 'column': 'close'}
    ))
    
    return configs


def create_feature_engine_with_defaults() -> RealtimeFeatureEngine:
    """Create feature engine with default technical indicators."""
    engine = RealtimeFeatureEngine()
    
    # Add default features
    configs = create_technical_indicators_config()
    for config in configs:
        engine.register_feature(config)
    
    return engine