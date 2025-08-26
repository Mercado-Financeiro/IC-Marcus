"""
Kafka consumer for streaming data ingestion.
"""

import json
import time
from typing import Dict, List, Optional, Any, Callable, Iterator
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import logging
import threading
from collections import deque
import pandas as pd
import numpy as np
from enum import Enum

logger = logging.getLogger(__name__)


class StreamStatus(Enum):
    """Stream status enum."""
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    PAUSED = "paused"
    ERROR = "error"


@dataclass
class KafkaConfig:
    """Kafka configuration."""
    bootstrap_servers: List[str]
    topic: str
    group_id: str = "ml_pipeline_consumer"
    auto_offset_reset: str = "latest"
    enable_auto_commit: bool = False
    session_timeout_ms: int = 30000
    heartbeat_interval_ms: int = 10000
    max_poll_records: int = 1000
    max_poll_interval_ms: int = 300000
    security_protocol: str = "PLAINTEXT"
    sasl_mechanism: Optional[str] = None
    sasl_username: Optional[str] = None
    sasl_password: Optional[str] = None
    ssl_cafile: Optional[str] = None
    ssl_certfile: Optional[str] = None
    ssl_keyfile: Optional[str] = None


@dataclass
class StreamMessage:
    """Streaming message container."""
    topic: str
    partition: int
    offset: int
    timestamp: datetime
    key: Optional[str]
    value: Any
    headers: Dict[str, str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'topic': self.topic,
            'partition': self.partition,
            'offset': self.offset,
            'timestamp': self.timestamp.isoformat(),
            'key': self.key,
            'value': self.value,
            'headers': self.headers
        }


class KafkaStreamConsumer:
    """
    High-performance Kafka consumer with backpressure handling.
    
    Features:
    - Automatic reconnection and error recovery
    - Backpressure handling with buffering
    - Batch processing for efficiency
    - Metrics and monitoring
    - Circuit breaker pattern
    - Graceful shutdown
    """
    
    def __init__(self, config: KafkaConfig):
        """Initialize Kafka consumer."""
        self.config = config
        
        # Consumer instance
        self.consumer = None
        self.consumer_lock = threading.Lock()
        
        # State management
        self.status = StreamStatus.STOPPED
        self.consumer_thread = None
        self.running = False
        
        # Message buffering
        self.message_buffer = deque(maxlen=10000)
        self.buffer_lock = threading.Lock()
        
        # Metrics
        self.metrics = {
            'messages_consumed': 0,
            'messages_processed': 0,
            'errors': 0,
            'last_message_time': None,
            'throughput_msgs_per_sec': 0.0,
            'buffer_size': 0,
            'lag_ms': 0.0
        }
        self.metrics_lock = threading.Lock()
        
        # Callbacks
        self.message_callbacks: List[Callable[[StreamMessage], None]] = []
        self.batch_callbacks: List[Callable[[List[StreamMessage]], None]] = []
        self.error_callbacks: List[Callable[[Exception], None]] = []
        
        # Circuit breaker
        self.circuit_breaker = {
            'failure_count': 0,
            'failure_threshold': 5,
            'reset_timeout': 60,
            'last_failure_time': None,
            'state': 'closed'  # closed, open, half_open
        }
        
        logger.info(f"KafkaStreamConsumer initialized for topic: {config.topic}")
    
    def start(self):
        """Start consuming messages."""
        if self.status == StreamStatus.RUNNING:
            logger.warning("Consumer already running")
            return
        
        self.status = StreamStatus.STARTING
        self.running = True
        
        # Start consumer thread
        self.consumer_thread = threading.Thread(target=self._consumer_loop, daemon=True)
        self.consumer_thread.start()
        
        logger.info("Kafka consumer started")
    
    def stop(self, timeout: float = 30.0):
        """Stop consuming messages."""
        if self.status == StreamStatus.STOPPED:
            return
        
        logger.info("Stopping Kafka consumer...")
        self.running = False
        
        # Wait for consumer thread to finish
        if self.consumer_thread:
            self.consumer_thread.join(timeout=timeout)
        
        # Close consumer
        with self.consumer_lock:
            if self.consumer:
                try:
                    self.consumer.close()
                except Exception as e:
                    logger.warning(f"Error closing consumer: {e}")
                finally:
                    self.consumer = None
        
        self.status = StreamStatus.STOPPED
        logger.info("Kafka consumer stopped")
    
    def _consumer_loop(self):
        """Main consumer loop."""
        while self.running:
            try:
                # Check circuit breaker
                if self._is_circuit_open():
                    time.sleep(5)
                    continue
                
                # Initialize consumer if needed
                if not self._ensure_consumer():
                    time.sleep(5)
                    continue
                
                self.status = StreamStatus.RUNNING
                
                # Poll for messages
                message_batch = self._poll_messages()
                
                if message_batch:
                    self._process_message_batch(message_batch)
                    self._update_metrics(len(message_batch))
                
            except Exception as e:
                self._handle_error(e)
                time.sleep(1)
        
        self.status = StreamStatus.STOPPED
    
    def _ensure_consumer(self) -> bool:
        """Ensure consumer is initialized and connected."""
        with self.consumer_lock:
            if self.consumer is not None:
                return True
            
            try:
                # Try to import kafka-python
                try:
                    from kafka import KafkaConsumer
                except ImportError:
                    logger.error("kafka-python not installed. Install with: pip install kafka-python")
                    return False
                
                # Create consumer configuration
                consumer_config = {
                    'bootstrap_servers': self.config.bootstrap_servers,
                    'group_id': self.config.group_id,
                    'auto_offset_reset': self.config.auto_offset_reset,
                    'enable_auto_commit': self.config.enable_auto_commit,
                    'session_timeout_ms': self.config.session_timeout_ms,
                    'heartbeat_interval_ms': self.config.heartbeat_interval_ms,
                    'max_poll_records': self.config.max_poll_records,
                    'max_poll_interval_ms': self.config.max_poll_interval_ms,
                    'security_protocol': self.config.security_protocol,
                    'value_deserializer': lambda m: json.loads(m.decode('utf-8')) if m else None,
                    'key_deserializer': lambda m: m.decode('utf-8') if m else None
                }
                
                # Add authentication if configured
                if self.config.sasl_mechanism:
                    consumer_config['sasl_mechanism'] = self.config.sasl_mechanism
                    consumer_config['sasl_plain_username'] = self.config.sasl_username
                    consumer_config['sasl_plain_password'] = self.config.sasl_password
                
                if self.config.ssl_cafile:
                    consumer_config['ssl_cafile'] = self.config.ssl_cafile
                    consumer_config['ssl_certfile'] = self.config.ssl_certfile
                    consumer_config['ssl_keyfile'] = self.config.ssl_keyfile
                
                # Create consumer
                self.consumer = KafkaConsumer(**consumer_config)
                
                # Subscribe to topic
                self.consumer.subscribe([self.config.topic])
                
                logger.info(f"Connected to Kafka topic: {self.config.topic}")
                self._reset_circuit_breaker()
                return True
                
            except Exception as e:
                logger.error(f"Failed to initialize Kafka consumer: {e}")
                self.consumer = None
                self._record_failure()
                return False
    
    def _poll_messages(self) -> List[StreamMessage]:
        """Poll for messages from Kafka."""
        try:
            with self.consumer_lock:
                if not self.consumer:
                    return []
                
                # Poll with timeout
                raw_messages = self.consumer.poll(timeout_ms=1000, max_records=self.config.max_poll_records)
            
            # Convert to StreamMessage objects
            messages = []
            for topic_partition, msgs in raw_messages.items():
                for msg in msgs:
                    stream_msg = StreamMessage(
                        topic=msg.topic,
                        partition=msg.partition,
                        offset=msg.offset,
                        timestamp=datetime.fromtimestamp(msg.timestamp / 1000),
                        key=msg.key,
                        value=msg.value,
                        headers=dict(msg.headers) if msg.headers else {}
                    )
                    messages.append(stream_msg)
            
            return messages
            
        except Exception as e:
            logger.error(f"Error polling messages: {e}")
            raise
    
    def _process_message_batch(self, messages: List[StreamMessage]):
        """Process batch of messages."""
        try:
            # Buffer messages for downstream processing
            with self.buffer_lock:
                self.message_buffer.extend(messages)
            
            # Call batch callbacks
            for callback in self.batch_callbacks:
                try:
                    callback(messages)
                except Exception as e:
                    logger.error(f"Error in batch callback: {e}")
            
            # Call individual message callbacks
            for message in messages:
                for callback in self.message_callbacks:
                    try:
                        callback(message)
                    except Exception as e:
                        logger.error(f"Error in message callback: {e}")
            
            # Commit offsets if auto-commit is disabled
            if not self.config.enable_auto_commit:
                self._commit_offsets()
            
        except Exception as e:
            logger.error(f"Error processing message batch: {e}")
            raise
    
    def _commit_offsets(self):
        """Manually commit offsets."""
        try:
            with self.consumer_lock:
                if self.consumer:
                    self.consumer.commit()
        except Exception as e:
            logger.warning(f"Failed to commit offsets: {e}")
    
    def _update_metrics(self, message_count: int):
        """Update consumer metrics."""
        with self.metrics_lock:
            self.metrics['messages_consumed'] += message_count
            self.metrics['last_message_time'] = datetime.now()
            self.metrics['buffer_size'] = len(self.message_buffer)
            
            # Calculate throughput (simple moving average)
            now = time.time()
            if not hasattr(self, '_last_metrics_update'):
                self._last_metrics_update = now
                self._throughput_window = deque(maxlen=10)
            
            time_diff = now - self._last_metrics_update
            if time_diff >= 1.0:  # Update every second
                throughput = message_count / time_diff
                self._throughput_window.append(throughput)
                self.metrics['throughput_msgs_per_sec'] = sum(self._throughput_window) / len(self._throughput_window)
                self._last_metrics_update = now
    
    def _handle_error(self, error: Exception):
        """Handle consumer errors."""
        logger.error(f"Consumer error: {error}")
        
        with self.metrics_lock:
            self.metrics['errors'] += 1
        
        # Call error callbacks
        for callback in self.error_callbacks:
            try:
                callback(error)
            except Exception as e:
                logger.error(f"Error in error callback: {e}")
        
        # Record failure for circuit breaker
        self._record_failure()
        
        # Close and recreate consumer on serious errors
        if "Broker not available" in str(error) or "Connection refused" in str(error):
            with self.consumer_lock:
                if self.consumer:
                    try:
                        self.consumer.close()
                    except Exception:
                        pass
                    self.consumer = None
        
        self.status = StreamStatus.ERROR
    
    def _is_circuit_open(self) -> bool:
        """Check if circuit breaker is open."""
        if self.circuit_breaker['state'] == 'closed':
            return False
        
        if self.circuit_breaker['state'] == 'open':
            # Check if reset timeout has elapsed
            if (self.circuit_breaker['last_failure_time'] and
                time.time() - self.circuit_breaker['last_failure_time'] > self.circuit_breaker['reset_timeout']):
                self.circuit_breaker['state'] = 'half_open'
                logger.info("Circuit breaker moving to half-open state")
                return False
            return True
        
        # half_open state - let through but monitor
        return False
    
    def _record_failure(self):
        """Record a failure for circuit breaker."""
        self.circuit_breaker['failure_count'] += 1
        self.circuit_breaker['last_failure_time'] = time.time()
        
        if self.circuit_breaker['failure_count'] >= self.circuit_breaker['failure_threshold']:
            self.circuit_breaker['state'] = 'open'
            logger.warning(f"Circuit breaker opened after {self.circuit_breaker['failure_count']} failures")
    
    def _reset_circuit_breaker(self):
        """Reset circuit breaker after successful operation."""
        if self.circuit_breaker['state'] != 'closed':
            self.circuit_breaker['state'] = 'closed'
            self.circuit_breaker['failure_count'] = 0
            logger.info("Circuit breaker reset to closed state")
    
    def add_message_callback(self, callback: Callable[[StreamMessage], None]):
        """Add callback for individual messages."""
        self.message_callbacks.append(callback)
    
    def add_batch_callback(self, callback: Callable[[List[StreamMessage]], None]):
        """Add callback for message batches."""
        self.batch_callbacks.append(callback)
    
    def add_error_callback(self, callback: Callable[[Exception], None]):
        """Add callback for errors."""
        self.error_callbacks.append(callback)
    
    def get_messages(self, count: int = 100) -> List[StreamMessage]:
        """Get messages from buffer."""
        messages = []
        with self.buffer_lock:
            for _ in range(min(count, len(self.message_buffer))):
                if self.message_buffer:
                    messages.append(self.message_buffer.popleft())
        
        with self.metrics_lock:
            self.metrics['messages_processed'] += len(messages)
        
        return messages
    
    def get_messages_as_dataframe(self, count: int = 100, parse_json: bool = True) -> pd.DataFrame:
        """Get messages as pandas DataFrame."""
        messages = self.get_messages(count)
        
        if not messages:
            return pd.DataFrame()
        
        # Convert to records
        records = []
        for msg in messages:
            record = {
                'timestamp': msg.timestamp,
                'topic': msg.topic,
                'partition': msg.partition,
                'offset': msg.offset,
                'key': msg.key
            }
            
            # Handle value parsing
            if parse_json and isinstance(msg.value, dict):
                record.update(msg.value)
            else:
                record['value'] = msg.value
            
            # Add headers
            for header_key, header_value in msg.headers.items():
                record[f'header_{header_key}'] = header_value
            
            records.append(record)
        
        return pd.DataFrame(records)
    
    def pause(self):
        """Pause message consumption."""
        if self.status == StreamStatus.RUNNING:
            self.status = StreamStatus.PAUSED
            logger.info("Consumer paused")
    
    def resume(self):
        """Resume message consumption."""
        if self.status == StreamStatus.PAUSED:
            self.status = StreamStatus.RUNNING
            logger.info("Consumer resumed")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get consumer metrics."""
        with self.metrics_lock:
            return {
                **self.metrics.copy(),
                'status': self.status.value,
                'circuit_breaker_state': self.circuit_breaker['state'],
                'circuit_breaker_failures': self.circuit_breaker['failure_count'],
                'topic': self.config.topic,
                'group_id': self.config.group_id
            }
    
    def get_lag(self) -> Dict[str, Any]:
        """Get consumer lag information."""
        try:
            with self.consumer_lock:
                if not self.consumer:
                    return {'lag': 'unknown', 'reason': 'consumer_not_initialized'}
                
                # Get partition assignments
                assignments = self.consumer.assignment()
                if not assignments:
                    return {'lag': 0, 'partitions': {}}
                
                # Get high water marks
                high_water_marks = self.consumer.end_offsets(assignments)
                
                # Get current positions
                lag_info = {'total_lag': 0, 'partitions': {}}
                
                for tp in assignments:
                    current_offset = self.consumer.position(tp)
                    high_water_mark = high_water_marks.get(tp, 0)
                    partition_lag = max(0, high_water_mark - current_offset)
                    
                    lag_info['partitions'][f'{tp.topic}_{tp.partition}'] = {
                        'current_offset': current_offset,
                        'high_water_mark': high_water_mark,
                        'lag': partition_lag
                    }
                    lag_info['total_lag'] += partition_lag
                
                return lag_info
                
        except Exception as e:
            logger.warning(f"Failed to get lag information: {e}")
            return {'lag': 'error', 'reason': str(e)}
    
    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()