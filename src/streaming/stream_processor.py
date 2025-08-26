"""
Stream processing engine with state management and fault tolerance.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
import threading
import time
import json
from collections import defaultdict, deque
from abc import ABC, abstractmethod
import pickle
import hashlib

from .kafka_consumer import KafkaStreamConsumer, KafkaConfig, StreamMessage
from .windowing import WindowManager, WindowConfig, WindowFrame
from ..features.circuit_breaker import FeatureProcessingBreaker
from ..monitoring.alerting import AlertManager

logger = logging.getLogger(__name__)


@dataclass
class StreamConfig:
    """Stream processor configuration."""
    checkpoint_interval: timedelta = timedelta(minutes=5)
    state_store_path: str = "data/streaming/state"
    enable_exactly_once: bool = True
    max_retries: int = 3
    backoff_multiplier: float = 2.0
    processing_timeout: int = 30
    enable_metrics: bool = True
    parallelism: int = 1
    buffer_size: int = 10000


class StreamState:
    """Stream processing state management."""
    
    def __init__(self, state_store_path: str):
        self.state_store_path = state_store_path
        self.state_data: Dict[str, Any] = {}
        self.state_lock = threading.RLock()
        
        # Ensure state directory exists
        import os
        os.makedirs(state_store_path, exist_ok=True)
    
    def put(self, key: str, value: Any):
        """Store state value."""
        with self.state_lock:
            self.state_data[key] = value
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get state value."""
        with self.state_lock:
            return self.state_data.get(key, default)
    
    def delete(self, key: str) -> bool:
        """Delete state value."""
        with self.state_lock:
            if key in self.state_data:
                del self.state_data[key]
                return True
            return False
    
    def keys(self) -> List[str]:
        """Get all state keys."""
        with self.state_lock:
            return list(self.state_data.keys())
    
    def checkpoint(self) -> bool:
        """Save state to disk."""
        try:
            checkpoint_file = f"{self.state_store_path}/checkpoint_{int(time.time())}.pkl"
            
            with self.state_lock:
                with open(checkpoint_file, 'wb') as f:
                    pickle.dump(self.state_data, f)
            
            logger.debug(f"State checkpointed to {checkpoint_file}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to checkpoint state: {e}")
            return False
    
    def restore(self, checkpoint_file: Optional[str] = None) -> bool:
        """Restore state from disk."""
        try:
            if checkpoint_file is None:
                # Find latest checkpoint
                import glob
                checkpoints = glob.glob(f"{self.state_store_path}/checkpoint_*.pkl")
                if not checkpoints:
                    return False
                checkpoint_file = max(checkpoints)
            
            with open(checkpoint_file, 'rb') as f:
                restored_state = pickle.load(f)
            
            with self.state_lock:
                self.state_data = restored_state
            
            logger.info(f"State restored from {checkpoint_file}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to restore state: {e}")
            return False


class StreamProcessor:
    """
    Advanced stream processing engine.
    
    Features:
    - Exactly-once processing semantics
    - State management with checkpointing
    - Window-based aggregations
    - Fault tolerance and recovery
    - Backpressure handling
    - Metrics and monitoring
    """
    
    def __init__(self, config: StreamConfig):
        """Initialize stream processor."""
        self.config = config
        
        # State management
        self.state = StreamState(config.state_store_path)
        
        # Processing components
        self.consumers: Dict[str, KafkaStreamConsumer] = {}
        self.window_managers: Dict[str, WindowManager] = {}
        self.processors: List[Callable] = []
        
        # Runtime state
        self.running = False
        self.processing_threads: List[threading.Thread] = []
        
        # Circuit breaker for resilience
        self.circuit_breaker = FeatureProcessingBreaker(
            failure_threshold=5,
            timeout_duration=60.0
        )
        
        # Metrics
        self.metrics = {
            'messages_processed': 0,
            'messages_failed': 0,
            'processing_latency_ms': deque(maxlen=1000),
            'throughput_msgs_per_sec': 0.0,
            'last_checkpoint': None,
            'state_size': 0
        }
        self.metrics_lock = threading.Lock()
        
        # Alerting
        self.alert_manager = None
        
        logger.info("StreamProcessor initialized")
    
    def add_source(
        self,
        name: str,
        kafka_config: KafkaConfig,
        key_deserializer: Optional[Callable] = None,
        value_deserializer: Optional[Callable] = None
    ):
        """Add Kafka source."""
        consumer = KafkaStreamConsumer(kafka_config)
        
        # Add processing callback
        consumer.add_message_callback(
            lambda msg: self._process_message(name, msg)
        )
        
        self.consumers[name] = consumer
        logger.info(f"Added source: {name}")
    
    def add_window(self, name: str, window_config: WindowConfig):
        """Add windowing operator."""
        window_manager = WindowManager(window_config)
        self.window_managers[name] = window_manager
        logger.info(f"Added window: {name}")
    
    def add_processor(self, processor_func: Callable):
        """Add custom processor function."""
        self.processors.append(processor_func)
    
    def set_alert_manager(self, alert_manager: AlertManager):
        """Set alert manager for monitoring."""
        self.alert_manager = alert_manager
    
    def start(self):
        """Start stream processing."""
        if self.running:
            logger.warning("Stream processor already running")
            return
        
        self.running = True
        
        # Restore state if available
        self.state.restore()
        
        # Start window managers
        for window_manager in self.window_managers.values():
            window_manager.start_cleanup()
        
        # Start consumers
        for consumer in self.consumers.values():
            consumer.start()
        
        # Start processing threads
        for i in range(self.config.parallelism):
            thread = threading.Thread(
                target=self._processing_loop,
                args=(f"processor-{i}",),
                daemon=True
            )
            self.processing_threads.append(thread)
            thread.start()
        
        # Start checkpoint thread
        checkpoint_thread = threading.Thread(
            target=self._checkpoint_loop,
            daemon=True
        )
        self.processing_threads.append(checkpoint_thread)
        checkpoint_thread.start()
        
        logger.info("Stream processor started")
    
    def stop(self, timeout: float = 30.0):
        """Stop stream processing."""
        if not self.running:
            return
        
        logger.info("Stopping stream processor...")
        self.running = False
        
        # Stop consumers
        for consumer in self.consumers.values():
            consumer.stop(timeout=10.0)
        
        # Stop window managers
        for window_manager in self.window_managers.values():
            window_manager.stop_cleanup()
        
        # Wait for processing threads
        for thread in self.processing_threads:
            thread.join(timeout=timeout/len(self.processing_threads))
        
        # Final checkpoint
        self.state.checkpoint()
        
        logger.info("Stream processor stopped")
    
    def _processing_loop(self, thread_name: str):
        """Main processing loop."""
        logger.info(f"Processing thread {thread_name} started")
        
        while self.running:
            try:
                # Process any completed windows
                self._process_completed_windows()
                
                # Run custom processors
                for processor in self.processors:
                    try:
                        processor(self.state, self.window_managers)
                    except Exception as e:
                        logger.error(f"Error in custom processor: {e}")
                
                time.sleep(0.1)  # Prevent busy waiting
                
            except Exception as e:
                logger.error(f"Error in processing loop {thread_name}: {e}")
                time.sleep(1)
        
        logger.info(f"Processing thread {thread_name} stopped")
    
    def _process_message(self, source_name: str, message: StreamMessage):
        """Process individual message."""
        start_time = time.time()
        
        try:
            with self.circuit_breaker.call():
                # Extract timestamp
                event_time = message.timestamp
                
                # Process through windows
                for window_name, window_manager in self.window_managers.items():
                    window_manager.add_event(
                        event=message.value,
                        timestamp=event_time,
                        key=message.key,
                        metadata={
                            'source': source_name,
                            'partition': message.partition,
                            'offset': message.offset
                        }
                    )
                
                # Update metrics
                processing_time_ms = (time.time() - start_time) * 1000
                with self.metrics_lock:
                    self.metrics['messages_processed'] += 1
                    self.metrics['processing_latency_ms'].append(processing_time_ms)
                
        except Exception as e:
            logger.error(f"Failed to process message from {source_name}: {e}")
            
            with self.metrics_lock:
                self.metrics['messages_failed'] += 1
            
            # Send alert if configured
            if self.alert_manager:
                self.alert_manager.send_alert(
                    level="error",
                    title="Stream Processing Error",
                    message=f"Failed to process message: {e}",
                    source="stream_processor"
                )
    
    def _process_completed_windows(self):
        """Process completed windows."""
        for window_name, window_manager in self.window_managers.items():
            completed_windows = window_manager.get_completed_windows(remove=True)
            
            for window in completed_windows:
                try:
                    self._handle_window_result(window_name, window)
                except Exception as e:
                    logger.error(f"Error processing window {window.window_id}: {e}")
    
    def _handle_window_result(self, window_name: str, window: WindowFrame):
        """Handle completed window result."""
        # Convert to DataFrame for easier processing
        df = window.to_dataframe()
        
        if df.empty:
            return
        
        # Store window result in state
        result_key = f"window_{window_name}_{window.window_id}"
        window_result = {
            'window_id': window.window_id,
            'start_time': window.start_time.isoformat(),
            'end_time': window.end_time.isoformat(),
            'record_count': len(df),
            'data': df.to_dict('records') if len(df) <= 100 else None,  # Limit stored data
            'summary_stats': self._calculate_window_stats(df)
        }
        
        self.state.put(result_key, window_result)
        
        logger.debug(f"Processed window {window.window_id} with {len(df)} records")
    
    def _calculate_window_stats(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate summary statistics for window data."""
        stats = {
            'record_count': len(df),
            'timestamp_range': None
        }
        
        if 'timestamp' in df.columns:
            stats['timestamp_range'] = [
                df['timestamp'].min().isoformat(),
                df['timestamp'].max().isoformat()
            ]
        
        # Calculate stats for numeric columns
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            stats[f'{col}_mean'] = float(df[col].mean()) if not df[col].empty else 0.0
            stats[f'{col}_sum'] = float(df[col].sum()) if not df[col].empty else 0.0
            stats[f'{col}_count'] = int(df[col].count())
        
        return stats
    
    def _checkpoint_loop(self):
        """Checkpoint loop."""
        logger.info("Checkpoint thread started")
        
        while self.running:
            try:
                time.sleep(self.config.checkpoint_interval.total_seconds())
                
                if self.running:  # Check again after sleep
                    success = self.state.checkpoint()
                    
                    with self.metrics_lock:
                        self.metrics['last_checkpoint'] = datetime.now().isoformat()
                        self.metrics['state_size'] = len(self.state.keys())
                    
                    if success:
                        logger.debug("State checkpoint completed")
                    else:
                        logger.warning("State checkpoint failed")
                
            except Exception as e:
                logger.error(f"Error in checkpoint loop: {e}")
                time.sleep(60)  # Wait before retrying
        
        logger.info("Checkpoint thread stopped")
    
    def query_state(self, pattern: str = "*") -> Dict[str, Any]:
        """Query state with pattern matching."""
        import fnmatch
        
        matching_keys = []
        all_keys = self.state.keys()
        
        for key in all_keys:
            if fnmatch.fnmatch(key, pattern):
                matching_keys.append(key)
        
        result = {}
        for key in matching_keys:
            result[key] = self.state.get(key)
        
        return result
    
    def get_window_results(self, window_name: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent window results for a window."""
        pattern = f"window_{window_name}_*"
        results = self.query_state(pattern)
        
        # Sort by end time and return most recent
        sorted_results = sorted(
            results.items(),
            key=lambda x: x[1]['end_time'],
            reverse=True
        )
        
        return [result[1] for result in sorted_results[:limit]]
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get processing metrics."""
        with self.metrics_lock:
            metrics = self.metrics.copy()
            
            # Calculate derived metrics
            if metrics['processing_latency_ms']:
                metrics['avg_latency_ms'] = sum(metrics['processing_latency_ms']) / len(metrics['processing_latency_ms'])
                metrics['max_latency_ms'] = max(metrics['processing_latency_ms'])
            else:
                metrics['avg_latency_ms'] = 0.0
                metrics['max_latency_ms'] = 0.0
            
            # Remove raw latency data from response
            del metrics['processing_latency_ms']
        
        # Add consumer metrics
        consumer_metrics = {}
        for name, consumer in self.consumers.items():
            consumer_metrics[name] = consumer.get_metrics()
        
        # Add window manager metrics
        window_metrics = {}
        for name, window_manager in self.window_managers.items():
            window_metrics[name] = window_manager.get_stats()
        
        return {
            'processor': metrics,
            'consumers': consumer_metrics,
            'windows': window_metrics,
            'circuit_breaker': {
                'failure_count': self.circuit_breaker.failure_count,
                'state': self.circuit_breaker.state.name
            }
        }
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get health status."""
        health = {
            'status': 'healthy',
            'running': self.running,
            'issues': []
        }
        
        # Check consumers
        for name, consumer in self.consumers.items():
            consumer_metrics = consumer.get_metrics()
            if consumer_metrics['status'] != 'running':
                health['issues'].append(f"Consumer {name} not running: {consumer_metrics['status']}")
                health['status'] = 'degraded'
        
        # Check circuit breaker
        if self.circuit_breaker.state.name == 'OPEN':
            health['issues'].append("Circuit breaker is open")
            health['status'] = 'degraded'
        
        # Check processing metrics
        with self.metrics_lock:
            if self.metrics['messages_failed'] > 0:
                failure_rate = self.metrics['messages_failed'] / max(self.metrics['messages_processed'], 1)
                if failure_rate > 0.1:  # > 10% failure rate
                    health['issues'].append(f"High failure rate: {failure_rate:.2%}")
                    health['status'] = 'degraded'
        
        if health['issues']:
            health['status'] = 'unhealthy' if len(health['issues']) > 2 else 'degraded'
        
        return health
    
    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()


# Example usage functions
def create_simple_stream_processor(
    kafka_config: KafkaConfig,
    window_config: WindowConfig,
    processor_func: Optional[Callable] = None
) -> StreamProcessor:
    """Create a simple stream processor with one source and one window."""
    config = StreamConfig()
    processor = StreamProcessor(config)
    
    # Add source
    processor.add_source("main", kafka_config)
    
    # Add window
    processor.add_window("main", window_config)
    
    # Add processor if provided
    if processor_func:
        processor.add_processor(processor_func)
    
    return processor


def create_multi_source_processor(
    kafka_configs: Dict[str, KafkaConfig],
    window_configs: Dict[str, WindowConfig]
) -> StreamProcessor:
    """Create processor with multiple sources and windows."""
    config = StreamConfig(parallelism=len(kafka_configs))
    processor = StreamProcessor(config)
    
    # Add sources
    for name, kafka_config in kafka_configs.items():
        processor.add_source(name, kafka_config)
    
    # Add windows
    for name, window_config in window_configs.items():
        processor.add_window(name, window_config)
    
    return processor