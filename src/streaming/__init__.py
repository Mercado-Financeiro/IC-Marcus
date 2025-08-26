"""
Real-time streaming data processing module.
"""

from .kafka_consumer import KafkaStreamConsumer, KafkaConfig
from .stream_processor import StreamProcessor, StreamConfig
from .windowing import WindowManager, WindowConfig
from .realtime_features import RealtimeFeatureEngine

__all__ = [
    'KafkaStreamConsumer',
    'KafkaConfig', 
    'StreamProcessor',
    'StreamConfig',
    'WindowManager',
    'WindowConfig',
    'RealtimeFeatureEngine'
]