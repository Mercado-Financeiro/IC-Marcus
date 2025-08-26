"""
Real-time Stream Processor for Time Series Features.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json
import logging
import asyncio
import threading
from queue import Queue, Empty
from concurrent.futures import ThreadPoolExecutor
import time

from .time_series_store import TimeSeriesFeatureStore, TimeSeriesConfig

logger = logging.getLogger(__name__)


@dataclass
class StreamConfig:
    """Real-time stream processing configuration."""
    time_series_config: TimeSeriesConfig
    
    # Stream processing
    max_workers: int = 4
    processing_batch_size: int = 100
    processing_interval: float = 0.1  # seconds
    
    # Feature windows  
    default_windows: List[int] = field(default_factory=lambda: [5, 10, 20, 60])
    max_window_size: int = 300  # seconds
    
    # Alerting
    enable_alerts: bool = True
    alert_thresholds: Dict[str, float] = field(default_factory=dict)
    
    # Performance monitoring
    monitor_latency: bool = True
    latency_alert_threshold: float = 1.0  # seconds
    
    # Error handling
    max_retries: int = 3
    retry_delay: float = 0.5


class RealTimeStreamProcessor:
    """
    Real-time Stream Processor for time series features with advanced capabilities.
    
    Features:
    - Real-time feature ingestion and processing
    - Multi-timeframe windowed features  
    - Custom feature transformation pipelines
    - Anomaly detection and alerting
    - Performance monitoring
    - Fault tolerance and error recovery
    - Backpressure management
    """
    
    def __init__(self, config: StreamConfig):
        """Initialize real-time stream processor."""
        self.config = config
        self.feature_store = TimeSeriesFeatureStore(config.time_series_config)
        
        # Processing pipeline
        self._processing_queue = Queue(maxsize=10000)
        self._result_queue = Queue(maxsize=1000)
        self._workers: List[threading.Thread] = []
        self._stop_processing = threading.Event()
        
        # Feature transformations
        self._transformers: Dict[str, Callable] = {}
        self._windowed_features: Dict[str, List[Dict]] = {}
        
        # Monitoring
        self._processing_stats = {
            "messages_processed": 0,
            "messages_failed": 0,
            "avg_processing_latency": 0.0,
            "avg_feature_computation_time": 0.0,
            "alerts_triggered": 0
        }
        
        # Feature buffers for windowed calculations
        self._feature_buffers: Dict[str, List[Dict]] = {}
        self._buffer_lock = threading.RLock()
        
        self._initialize_workers()
        logger.info("Real-time stream processor initialized")
    
    def _initialize_workers(self):
        """Initialize worker threads for processing."""
        for i in range(self.config.max_workers):
            worker = threading.Thread(
                target=self._worker_loop,
                name=f"StreamWorker-{i}",
                daemon=True
            )
            worker.start()
            self._workers.append(worker)
        
        logger.info(f"Started {len(self._workers)} stream processing workers")
    
    def _worker_loop(self):
        """Main worker loop for processing features."""
        while not self._stop_processing.is_set():
            try:
                # Get batch of messages
                batch = []
                batch_start_time = time.time()
                
                # Collect batch
                for _ in range(self.config.processing_batch_size):
                    try:
                        message = self._processing_queue.get(timeout=self.config.processing_interval)
                        batch.append(message)
                        self._processing_queue.task_done()
                    except Empty:
                        break
                
                if not batch:
                    continue
                
                # Process batch
                self._process_batch(batch, batch_start_time)
                
            except Exception as e:
                logger.error(f"Worker error: {e}")
                self._processing_stats["messages_failed"] += 1
    
    def _process_batch(self, batch: List[Dict], batch_start_time: float):
        """Process a batch of feature messages."""
        try:
            processed_features = []
            
            for message in batch:
                start_time = time.time()
                
                # Extract message data
                features = message.get("features", {})
                measurement = message.get("measurement", "crypto_features")
                tags = message.get("tags", {})
                timestamp = message.get("timestamp", datetime.utcnow())
                
                # Apply transformations
                transformed_features = self._apply_transformations(
                    features, measurement, timestamp
                )
                
                # Compute windowed features
                windowed_features = self._compute_windowed_features(
                    transformed_features, measurement, timestamp
                )
                
                # Combine all features
                all_features = {**transformed_features, **windowed_features}
                
                # Anomaly detection and alerting
                if self.config.enable_alerts:
                    self._check_anomalies(all_features, measurement, tags)
                
                processed_features.append({
                    "features": all_features,
                    "measurement": measurement,
                    "tags": tags,
                    "timestamp": timestamp
                })
                
                # Update processing time stats
                processing_time = time.time() - start_time
                self._update_processing_stats(processing_time)
                
            # Write batch to feature store
            if processed_features:
                self._write_batch_to_store(processed_features)
            
            # Update batch processing stats
            batch_processing_time = time.time() - batch_start_time
            if self.config.monitor_latency and batch_processing_time > self.config.latency_alert_threshold:
                logger.warning(f"High batch processing latency: {batch_processing_time:.2f}s")
            
        except Exception as e:
            logger.error(f"Batch processing error: {e}")
            self._processing_stats["messages_failed"] += len(batch)
    
    def _apply_transformations(
        self,
        features: Dict[str, Any],
        measurement: str,
        timestamp: datetime
    ) -> Dict[str, Any]:
        """Apply registered feature transformations."""
        try:
            transformed = features.copy()
            
            # Apply measurement-specific transformers
            transformer_key = f"{measurement}_transformer"
            if transformer_key in self._transformers:
                transformer_func = self._transformers[transformer_key]
                transformed = transformer_func(transformed, timestamp)
            
            # Apply global transformers
            if "global_transformer" in self._transformers:
                transformer_func = self._transformers["global_transformer"]
                transformed = transformer_func(transformed, timestamp)
            
            return transformed
            
        except Exception as e:
            logger.error(f"Transformation error: {e}")
            return features
    
    def _compute_windowed_features(
        self,
        features: Dict[str, Any],
        measurement: str,
        timestamp: datetime
    ) -> Dict[str, Any]:
        """Compute windowed features (rolling statistics)."""
        try:
            windowed = {}
            
            with self._buffer_lock:
                # Initialize buffer for measurement
                if measurement not in self._feature_buffers:
                    self._feature_buffers[measurement] = []
                
                buffer = self._feature_buffers[measurement]
                
                # Add current features to buffer
                buffer.append({
                    "features": features,
                    "timestamp": timestamp
                })
                
                # Remove old features beyond max window
                cutoff_time = timestamp - timedelta(seconds=self.config.max_window_size)
                buffer[:] = [
                    item for item in buffer 
                    if item["timestamp"] > cutoff_time
                ]
                
                # Compute windowed features for each window size
                for window_seconds in self.config.default_windows:
                    window_start = timestamp - timedelta(seconds=window_seconds)
                    
                    # Filter features within window
                    window_data = [
                        item["features"] for item in buffer
                        if item["timestamp"] >= window_start
                    ]
                    
                    if len(window_data) < 2:  # Need at least 2 data points
                        continue
                    
                    # Convert to DataFrame for easy computation
                    df = pd.DataFrame(window_data)
                    
                    # Compute rolling statistics
                    for column in df.select_dtypes(include=[np.number]).columns:
                        window_key = f"{window_seconds}s"
                        windowed[f"{column}_mean_{window_key}"] = df[column].mean()
                        windowed[f"{column}_std_{window_key}"] = df[column].std()
                        windowed[f"{column}_min_{window_key}"] = df[column].min()
                        windowed[f"{column}_max_{window_key}"] = df[column].max()
                        
                        # Compute additional metrics
                        if len(df) >= 2:
                            windowed[f"{column}_slope_{window_key}"] = self._compute_slope(df[column])
                            windowed[f"{column}_volatility_{window_key}"] = df[column].std() / df[column].mean() if df[column].mean() != 0 else 0
                
            return {k: v for k, v in windowed.items() if pd.notna(v)}
            
        except Exception as e:
            logger.error(f"Windowed features computation error: {e}")
            return {}
    
    def _compute_slope(self, values: pd.Series) -> float:
        """Compute slope of values over time."""
        try:
            if len(values) < 2:
                return 0.0
            
            x = np.arange(len(values))
            y = values.values
            
            # Simple linear regression slope
            n = len(values)
            slope = (n * np.sum(x * y) - np.sum(x) * np.sum(y)) / (n * np.sum(x**2) - (np.sum(x))**2)
            
            return slope
            
        except Exception:
            return 0.0
    
    def _check_anomalies(
        self,
        features: Dict[str, Any],
        measurement: str,
        tags: Dict[str, str]
    ):
        """Check for anomalies and trigger alerts."""
        try:
            alerts = []
            
            # Check configured thresholds
            for feature_name, threshold in self.config.alert_thresholds.items():
                if feature_name in features:
                    value = features[feature_name]
                    
                    if isinstance(value, (int, float)) and abs(value) > threshold:
                        alert = {
                            "type": "threshold_exceeded",
                            "feature": feature_name,
                            "value": value,
                            "threshold": threshold,
                            "measurement": measurement,
                            "tags": tags,
                            "timestamp": datetime.utcnow()
                        }
                        alerts.append(alert)
            
            # Custom anomaly detection (Z-score based)
            for feature_name, value in features.items():
                if isinstance(value, (int, float)) and feature_name.endswith("_std_60s"):
                    # High volatility alert
                    if value > 0.1:  # 10% volatility threshold
                        alert = {
                            "type": "high_volatility",
                            "feature": feature_name,
                            "value": value,
                            "measurement": measurement,
                            "tags": tags,
                            "timestamp": datetime.utcnow()
                        }
                        alerts.append(alert)
            
            # Log and store alerts
            if alerts:
                self._processing_stats["alerts_triggered"] += len(alerts)
                
                for alert in alerts:
                    logger.warning(f"Alert triggered: {alert}")
                    
                    # Store alert in feature store
                    self.feature_store.write_features(
                        features=alert,
                        measurement="alerts",
                        async_write=False
                    )
            
        except Exception as e:
            logger.error(f"Anomaly detection error: {e}")
    
    def _write_batch_to_store(self, batch: List[Dict]):
        """Write processed batch to feature store."""
        try:
            # Group by measurement for efficient writing
            measurement_groups = {}
            
            for item in batch:
                measurement = item["measurement"]
                if measurement not in measurement_groups:
                    measurement_groups[measurement] = {
                        "features": [],
                        "tags": item["tags"],
                        "timestamps": []
                    }
                
                measurement_groups[measurement]["features"].append(item["features"])
                measurement_groups[measurement]["timestamps"].append(item["timestamp"])
            
            # Write each group
            for measurement, group_data in measurement_groups.items():
                df = pd.DataFrame(group_data["features"])
                df["timestamp"] = group_data["timestamps"]
                
                success = self.feature_store.write_features(
                    features=df,
                    measurement=measurement,
                    tags=group_data["tags"],
                    async_write=False  # Already in async context
                )
                
                if not success:
                    logger.error(f"Failed to write batch to measurement: {measurement}")
            
        except Exception as e:
            logger.error(f"Batch write error: {e}")
    
    def _update_processing_stats(self, processing_time: float):
        """Update processing performance statistics."""
        self._processing_stats["messages_processed"] += 1
        
        # Update average processing latency
        total_messages = self._processing_stats["messages_processed"]
        current_avg = self._processing_stats["avg_processing_latency"]
        
        self._processing_stats["avg_processing_latency"] = (
            (current_avg * (total_messages - 1) + processing_time) / total_messages
        )
    
    def register_transformer(
        self,
        name: str,
        transformer_func: Callable[[Dict[str, Any], datetime], Dict[str, Any]]
    ):
        """Register a feature transformation function."""
        self._transformers[name] = transformer_func
        logger.info(f"Registered transformer: {name}")
    
    def unregister_transformer(self, name: str):
        """Unregister a feature transformation function."""
        if name in self._transformers:
            del self._transformers[name]
            logger.info(f"Unregistered transformer: {name}")
    
    def ingest_features(
        self,
        features: Dict[str, Any],
        measurement: str = "crypto_features",
        tags: Optional[Dict[str, str]] = None,
        timestamp: Optional[datetime] = None
    ) -> bool:
        """Ingest features for real-time processing."""
        try:
            message = {
                "features": features,
                "measurement": measurement,
                "tags": tags or {},
                "timestamp": timestamp or datetime.utcnow()
            }
            
            # Add to processing queue
            try:
                self._processing_queue.put(message, block=False)
                return True
            except:
                # Queue full - handle backpressure
                logger.warning("Processing queue full, dropping message")
                return False
            
        except Exception as e:
            logger.error(f"Feature ingestion error: {e}")
            return False
    
    def ingest_features_batch(
        self,
        features_batch: List[Dict[str, Any]],
        measurement: str = "crypto_features",
        tags: Optional[Dict[str, str]] = None
    ) -> int:
        """Ingest batch of features for processing."""
        try:
            successful = 0
            
            for features in features_batch:
                success = self.ingest_features(features, measurement, tags)
                if success:
                    successful += 1
            
            logger.info(f"Ingested {successful}/{len(features_batch)} feature batches")
            return successful
            
        except Exception as e:
            logger.error(f"Batch ingestion error: {e}")
            return 0
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get real-time processing statistics."""
        stats = self._processing_stats.copy()
        
        # Add queue statistics
        stats.update({
            "processing_queue_size": self._processing_queue.qsize(),
            "result_queue_size": self._result_queue.qsize(),
            "active_workers": len([w for w in self._workers if w.is_alive()]),
            "registered_transformers": len(self._transformers),
            "feature_buffer_measurements": len(self._feature_buffers)
        })
        
        # Add buffer sizes
        with self._buffer_lock:
            buffer_sizes = {
                f"buffer_size_{measurement}": len(buffer)
                for measurement, buffer in self._feature_buffers.items()
            }
            stats.update(buffer_sizes)
        
        return stats
    
    def clear_buffers(self):
        """Clear all feature buffers."""
        with self._buffer_lock:
            self._feature_buffers.clear()
            logger.info("Feature buffers cleared")
    
    def set_alert_threshold(self, feature_name: str, threshold: float):
        """Set alert threshold for a feature."""
        self.config.alert_thresholds[feature_name] = threshold
        logger.info(f"Set alert threshold for {feature_name}: {threshold}")
    
    def remove_alert_threshold(self, feature_name: str):
        """Remove alert threshold for a feature."""
        if feature_name in self.config.alert_thresholds:
            del self.config.alert_thresholds[feature_name]
            logger.info(f"Removed alert threshold for {feature_name}")
    
    def get_recent_alerts(self, limit: int = 100) -> pd.DataFrame:
        """Get recent alerts from the system."""
        try:
            return self.feature_store.read_latest_features(
                measurement="alerts",
                limit=limit
            )
        except Exception as e:
            logger.error(f"Failed to get recent alerts: {e}")
            return pd.DataFrame()
    
    def stop(self):
        """Stop the stream processor."""
        try:
            logger.info("Stopping stream processor...")
            
            # Stop processing
            self._stop_processing.set()
            
            # Wait for workers to finish
            for worker in self._workers:
                worker.join(timeout=5.0)
            
            # Close feature store
            self.feature_store.close()
            
            logger.info("Stream processor stopped")
            
        except Exception as e:
            logger.error(f"Error stopping stream processor: {e}")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()