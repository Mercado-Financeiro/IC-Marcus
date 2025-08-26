"""
Time Series Feature Store with InfluxDB backend.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor
import threading
from queue import Queue

from .influxdb_integration import InfluxDBProvider, InfluxDBConfig

logger = logging.getLogger(__name__)


@dataclass 
class TimeSeriesConfig:
    """Time Series Feature Store configuration."""
    influxdb_config: InfluxDBConfig
    
    # Feature organization
    default_measurement: str = "crypto_features"
    default_tags: Dict[str, str] = field(default_factory=dict)
    
    # Caching
    enable_caching: bool = True
    cache_ttl: int = 300  # seconds
    max_cache_size: int = 1000
    
    # Real-time processing
    enable_real_time: bool = True
    buffer_size: int = 10000
    flush_interval: float = 1.0  # seconds
    
    # Feature validation
    validate_schema: bool = True
    auto_create_measurements: bool = True
    
    # Performance
    max_concurrent_queries: int = 5
    query_timeout: int = 30


class TimeSeriesFeatureStore:
    """
    Time Series Feature Store with advanced capabilities for ML workflows.
    
    Features:
    - High-performance time series ingestion
    - Real-time feature streaming
    - Automatic feature caching
    - Schema validation and evolution
    - Multi-timeframe aggregations
    - Feature lineage tracking
    - Performance monitoring
    """
    
    def __init__(self, config: TimeSeriesConfig):
        """Initialize time series feature store."""
        self.config = config
        self.influxdb = InfluxDBProvider(config.influxdb_config)
        
        # Initialize caches and buffers
        self._feature_cache: Dict[str, Any] = {}
        self._cache_timestamps: Dict[str, datetime] = {}
        self._schema_cache: Dict[str, Dict[str, List[str]]] = {}
        
        # Real-time processing
        self._buffer_queue = Queue(maxsize=config.buffer_size)
        self._flush_thread = None
        self._stop_flushing = threading.Event()
        
        # Performance monitoring
        self._query_stats = {
            "total_queries": 0,
            "total_writes": 0,
            "avg_query_time": 0.0,
            "avg_write_time": 0.0,
            "cache_hits": 0,
            "cache_misses": 0
        }
        
        if config.enable_real_time:
            self._start_real_time_processing()
        
        logger.info("Time Series Feature Store initialized")
    
    def _start_real_time_processing(self):
        """Start real-time processing thread."""
        def flush_buffer():
            while not self._stop_flushing.wait(self.config.flush_interval):
                self._flush_buffer()
        
        self._flush_thread = threading.Thread(target=flush_buffer, daemon=True)
        self._flush_thread.start()
        logger.info("Real-time processing started")
    
    def _flush_buffer(self):
        """Flush buffered features to InfluxDB."""
        if self._buffer_queue.empty():
            return
        
        batch_points = []
        batch_size = min(100, self._buffer_queue.qsize())
        
        try:
            for _ in range(batch_size):
                if not self._buffer_queue.empty():
                    batch_points.append(self._buffer_queue.get_nowait())
            
            if batch_points:
                # Group by measurement for efficient writing
                measurement_groups = {}
                for point in batch_points:
                    measurement = point.get("measurement", self.config.default_measurement)
                    if measurement not in measurement_groups:
                        measurement_groups[measurement] = []
                    measurement_groups[measurement].append(point)
                
                # Write each measurement group
                for measurement, points in measurement_groups.items():
                    df_points = pd.DataFrame([p["features"] for p in points])
                    
                    # Add timestamps
                    if "timestamp" not in df_points.columns:
                        df_points["timestamp"] = [p.get("timestamp", datetime.utcnow()) for p in points]
                    
                    # Get tags from first point (assuming same tags for batch)
                    tags = points[0].get("tags", self.config.default_tags)
                    
                    success = self.influxdb.write_features_batch(
                        df_points, measurement, tags
                    )
                    
                    if success:
                        self._query_stats["total_writes"] += len(points)
                
        except Exception as e:
            logger.error(f"Failed to flush buffer: {e}")
    
    def write_features(
        self,
        features: Union[Dict[str, Any], pd.DataFrame],
        measurement: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
        timestamp: Optional[datetime] = None,
        async_write: bool = True
    ) -> bool:
        """Write features to time series store."""
        try:
            measurement = measurement or self.config.default_measurement
            tags = {**self.config.default_tags, **(tags or {})}
            
            if isinstance(features, dict):
                # Single point write
                point_data = {
                    "features": features,
                    "measurement": measurement,
                    "tags": tags,
                    "timestamp": timestamp or datetime.utcnow()
                }
                
                if async_write and self.config.enable_real_time:
                    # Add to buffer for asynchronous processing
                    try:
                        self._buffer_queue.put(point_data, block=False)
                        return True
                    except:
                        logger.warning("Buffer full, writing synchronously")
                
                # Synchronous write
                return self.influxdb.write_features_streaming(
                    features, measurement, tags, timestamp
                )
            
            elif isinstance(features, pd.DataFrame):
                # Batch write
                if async_write and self.config.enable_real_time:
                    # Split into smaller batches for buffer
                    batch_size = 100
                    for i in range(0, len(features), batch_size):
                        batch_df = features.iloc[i:i+batch_size]
                        for _, row in batch_df.iterrows():
                            point_data = {
                                "features": row.to_dict(),
                                "measurement": measurement,
                                "tags": tags,
                                "timestamp": row.get("timestamp", datetime.utcnow())
                            }
                            
                            try:
                                self._buffer_queue.put(point_data, block=False)
                            except:
                                # Buffer full, write remaining synchronously
                                return self.influxdb.write_features_batch(
                                    features.iloc[i:], measurement, tags
                                )
                    return True
                else:
                    # Synchronous batch write
                    return self.influxdb.write_features_batch(
                        features, measurement, tags
                    )
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to write features: {e}")
            return False
    
    def read_features(
        self,
        measurement: Optional[str] = None,
        start_time: Union[str, datetime] = "-1h", 
        end_time: Optional[Union[str, datetime]] = None,
        fields: Optional[List[str]] = None,
        tags: Optional[Dict[str, str]] = None,
        aggregation: Optional[str] = None,
        window: Optional[str] = None,
        use_cache: bool = True
    ) -> pd.DataFrame:
        """Read features from time series store."""
        try:
            start_query_time = datetime.utcnow()
            measurement = measurement or self.config.default_measurement
            
            # Create cache key
            cache_key = self._create_cache_key(
                measurement, start_time, end_time, fields, tags, aggregation, window
            )
            
            # Check cache
            if use_cache and self.config.enable_caching:
                cached_result = self._get_from_cache(cache_key)
                if cached_result is not None:
                    self._query_stats["cache_hits"] += 1
                    return cached_result
                
                self._query_stats["cache_misses"] += 1
            
            # Query from InfluxDB
            df = self.influxdb.query_features(
                measurement=measurement,
                start_time=start_time,
                end_time=end_time,
                fields=fields,
                tags=tags,
                aggregation=aggregation,
                window=window
            )
            
            # Update cache
            if use_cache and self.config.enable_caching and not df.empty:
                self._store_in_cache(cache_key, df.copy())
            
            # Update stats
            query_time = (datetime.utcnow() - start_query_time).total_seconds()
            self._update_query_stats(query_time)
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to read features: {e}")
            return pd.DataFrame()
    
    def _create_cache_key(self, measurement, start_time, end_time, fields, tags, aggregation, window):
        """Create cache key for feature query."""
        key_parts = [
            str(measurement),
            str(start_time),
            str(end_time),
            str(sorted(fields) if fields else ""),
            str(sorted(tags.items()) if tags else ""),
            str(aggregation),
            str(window)
        ]
        return "|".join(key_parts)
    
    def _get_from_cache(self, cache_key: str) -> Optional[pd.DataFrame]:
        """Get result from cache if valid."""
        if cache_key not in self._feature_cache:
            return None
        
        cache_time = self._cache_timestamps.get(cache_key)
        if not cache_time:
            return None
        
        # Check if cache is still valid
        if (datetime.utcnow() - cache_time).total_seconds() > self.config.cache_ttl:
            self._remove_from_cache(cache_key)
            return None
        
        return self._feature_cache[cache_key].copy()
    
    def _store_in_cache(self, cache_key: str, df: pd.DataFrame):
        """Store result in cache."""
        # Check cache size limit
        if len(self._feature_cache) >= self.config.max_cache_size:
            self._evict_oldest_cache_entries()
        
        self._feature_cache[cache_key] = df
        self._cache_timestamps[cache_key] = datetime.utcnow()
    
    def _remove_from_cache(self, cache_key: str):
        """Remove entry from cache."""
        self._feature_cache.pop(cache_key, None)
        self._cache_timestamps.pop(cache_key, None)
    
    def _evict_oldest_cache_entries(self):
        """Evict oldest cache entries to make space."""
        # Sort by timestamp and remove oldest 20%
        sorted_keys = sorted(
            self._cache_timestamps.keys(),
            key=lambda k: self._cache_timestamps[k]
        )
        
        num_to_remove = max(1, len(sorted_keys) // 5)  # Remove 20%
        
        for key in sorted_keys[:num_to_remove]:
            self._remove_from_cache(key)
    
    def _update_query_stats(self, query_time: float):
        """Update query performance statistics."""
        self._query_stats["total_queries"] += 1
        
        # Update average query time
        total_queries = self._query_stats["total_queries"]
        current_avg = self._query_stats["avg_query_time"]
        
        self._query_stats["avg_query_time"] = (
            (current_avg * (total_queries - 1) + query_time) / total_queries
        )
    
    def read_latest_features(
        self,
        measurement: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
        limit: int = 100,
        use_cache: bool = True
    ) -> pd.DataFrame:
        """Read most recent features."""
        try:
            measurement = measurement or self.config.default_measurement
            
            cache_key = f"latest_{measurement}_{str(tags)}_{limit}"
            
            # Check cache
            if use_cache and self.config.enable_caching:
                cached_result = self._get_from_cache(cache_key)
                if cached_result is not None:
                    return cached_result
            
            # Query latest features
            df = self.influxdb.query_latest_features(measurement, tags, limit)
            
            # Cache result
            if use_cache and self.config.enable_caching and not df.empty:
                self._store_in_cache(cache_key, df.copy())
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to read latest features: {e}")
            return pd.DataFrame()
    
    def create_feature_aggregation(
        self,
        source_measurement: str,
        target_measurement: str,
        aggregation_function: str = "mean",
        window: str = "1m",
        fields: Optional[List[str]] = None
    ) -> bool:
        """Create automated feature aggregation."""
        try:
            query_name = f"agg_{source_measurement}_to_{target_measurement}"
            
            success = self.influxdb.create_continuous_query(
                query_name=query_name,
                source_measurement=source_measurement,
                target_measurement=target_measurement,
                aggregation_window=window,
                aggregation_function=aggregation_function
            )
            
            if success:
                logger.info(f"Created aggregation: {source_measurement} -> {target_measurement}")
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to create feature aggregation: {e}")
            return False
    
    def get_feature_schema(self, measurement: str) -> Dict[str, List[str]]:
        """Get feature schema for measurement."""
        try:
            # Check schema cache
            if measurement in self._schema_cache:
                return self._schema_cache[measurement]
            
            # Get schema from InfluxDB
            schema = self.influxdb.get_measurement_schema(measurement)
            
            # Cache schema
            self._schema_cache[measurement] = schema
            
            return schema
            
        except Exception as e:
            logger.error(f"Failed to get feature schema: {e}")
            return {"tags": [], "fields": []}
    
    def validate_features(
        self,
        features: Dict[str, Any],
        measurement: str,
        strict: bool = False
    ) -> Dict[str, Any]:
        """Validate features against schema."""
        try:
            if not self.config.validate_schema:
                return {"valid": True, "errors": []}
            
            schema = self.get_feature_schema(measurement)
            errors = []
            
            # Check for required fields (if in strict mode)
            if strict and schema.get("fields"):
                for field in schema["fields"]:
                    if field not in features:
                        errors.append(f"Missing required field: {field}")
            
            # Check for unknown fields
            known_fields = set(schema.get("fields", []))
            for field in features:
                if field not in known_fields and strict:
                    errors.append(f"Unknown field: {field}")
            
            validation_result = {
                "valid": len(errors) == 0,
                "errors": errors,
                "schema": schema
            }
            
            return validation_result
            
        except Exception as e:
            logger.error(f"Failed to validate features: {e}")
            return {"valid": False, "errors": [str(e)]}
    
    def delete_features(
        self,
        measurement: str,
        start_time: Union[str, datetime],
        end_time: Union[str, datetime],
        tags: Optional[Dict[str, str]] = None
    ) -> bool:
        """Delete features within time range."""
        try:
            success = self.influxdb.delete_features(
                measurement, start_time, end_time, tags
            )
            
            # Clear related cache entries
            if success:
                self._clear_cache_for_measurement(measurement)
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to delete features: {e}")
            return False
    
    def _clear_cache_for_measurement(self, measurement: str):
        """Clear cache entries related to a measurement."""
        keys_to_remove = [
            key for key in self._feature_cache.keys()
            if key.startswith(f"{measurement}|") or key.startswith(f"latest_{measurement}")
        ]
        
        for key in keys_to_remove:
            self._remove_from_cache(key)
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        stats = self._query_stats.copy()
        
        # Add cache statistics
        stats.update({
            "cache_size": len(self._feature_cache),
            "cache_hit_rate": (
                stats["cache_hits"] / (stats["cache_hits"] + stats["cache_misses"])
                if (stats["cache_hits"] + stats["cache_misses"]) > 0 else 0.0
            ),
            "buffer_size": self._buffer_queue.qsize() if self._buffer_queue else 0,
            "schema_cache_size": len(self._schema_cache)
        })
        
        # Add InfluxDB stats
        try:
            influx_stats = self.influxdb.get_database_stats()
            stats.update({"influxdb": influx_stats})
        except Exception as e:
            logger.warning(f"Failed to get InfluxDB stats: {e}")
        
        return stats
    
    def clear_cache(self):
        """Clear all caches."""
        self._feature_cache.clear()
        self._cache_timestamps.clear()
        self._schema_cache.clear()
        logger.info("All caches cleared")
    
    def close(self):
        """Close time series store."""
        try:
            # Stop real-time processing
            if self._flush_thread and self._flush_thread.is_alive():
                self._stop_flushing.set()
                self._flush_thread.join(timeout=5.0)
            
            # Flush remaining buffer
            if self._buffer_queue:
                self._flush_buffer()
            
            # Close InfluxDB connection
            self.influxdb.close()
            
            logger.info("Time series store closed")
            
        except Exception as e:
            logger.error(f"Error closing time series store: {e}")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()