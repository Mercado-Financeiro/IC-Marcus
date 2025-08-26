"""
Windowing operations for streaming data.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Callable, Iterator, Union, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
import threading
from collections import defaultdict, deque
from enum import Enum
import time

logger = logging.getLogger(__name__)


class WindowType(Enum):
    """Window type enum."""
    TUMBLING = "tumbling"
    SLIDING = "sliding"
    SESSION = "session"
    COUNT = "count"


@dataclass
class WindowConfig:
    """Window configuration."""
    window_type: WindowType
    size: Union[timedelta, int]  # Time duration or count
    slide: Optional[Union[timedelta, int]] = None
    grace_period: Optional[timedelta] = None
    session_timeout: Optional[timedelta] = None
    retention_time: Optional[timedelta] = None


@dataclass
class WindowFrame:
    """Window frame container."""
    window_id: str
    start_time: datetime
    end_time: datetime
    data: List[Any]
    metadata: Dict[str, Any]
    
    def __len__(self) -> int:
        return len(self.data)
    
    def is_complete(self, current_time: datetime, grace_period: Optional[timedelta] = None) -> bool:
        """Check if window is complete (past grace period)."""
        if grace_period is None:
            return current_time >= self.end_time
        return current_time >= (self.end_time + grace_period)
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert window data to DataFrame."""
        if not self.data:
            return pd.DataFrame()
        
        if isinstance(self.data[0], dict):
            return pd.DataFrame(self.data)
        elif hasattr(self.data[0], '__dict__'):
            return pd.DataFrame([item.__dict__ for item in self.data])
        else:
            return pd.DataFrame({'value': self.data})


class WindowManager:
    """
    Advanced windowing manager for streaming data.
    
    Features:
    - Multiple window types (tumbling, sliding, session, count)
    - Grace period handling for late arrivals
    - Window watermarking for progress tracking
    - Memory-efficient window storage
    - Thread-safe operations
    """
    
    def __init__(self, config: WindowConfig):
        """Initialize window manager."""
        self.config = config
        
        # Window storage
        self.windows: Dict[str, WindowFrame] = {}
        self.window_lock = threading.RWLock()
        
        # Watermark tracking
        self.watermark = datetime.min
        self.watermark_lock = threading.Lock()
        
        # Statistics
        self.stats = {
            'windows_created': 0,
            'windows_completed': 0,
            'late_arrivals': 0,
            'dropped_late_arrivals': 0,
            'total_events': 0
        }
        
        # Cleanup thread
        self.cleanup_running = False
        self.cleanup_thread = None
        
        logger.info(f"WindowManager initialized with {config.window_type.value} windows")
    
    def start_cleanup(self, interval: timedelta = timedelta(minutes=1)):
        """Start automatic window cleanup."""
        if self.cleanup_running:
            return
        
        self.cleanup_running = True
        self.cleanup_thread = threading.Thread(
            target=self._cleanup_loop,
            args=(interval,),
            daemon=True
        )
        self.cleanup_thread.start()
        logger.info("Window cleanup started")
    
    def stop_cleanup(self):
        """Stop automatic window cleanup."""
        self.cleanup_running = False
        if self.cleanup_thread:
            self.cleanup_thread.join(timeout=5)
        logger.info("Window cleanup stopped")
    
    def _cleanup_loop(self, interval: timedelta):
        """Cleanup loop for expired windows."""
        while self.cleanup_running:
            try:
                self._cleanup_expired_windows()
                time.sleep(interval.total_seconds())
            except Exception as e:
                logger.error(f"Error in window cleanup: {e}")
                time.sleep(60)  # Wait before retrying
    
    def add_event(
        self,
        event: Any,
        timestamp: datetime,
        key: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[WindowFrame]:
        """
        Add event to appropriate windows.
        
        Args:
            event: Event data
            timestamp: Event timestamp
            key: Optional event key for keyed windows
            metadata: Additional metadata
            
        Returns:
            List of completed windows
        """
        self.stats['total_events'] += 1
        
        # Update watermark
        self._update_watermark(timestamp)
        
        # Determine target windows
        window_ids = self._get_target_windows(timestamp, key)
        
        completed_windows = []
        
        with self.window_lock.gen_wlock():
            for window_id in window_ids:
                # Create window if it doesn't exist
                if window_id not in self.windows:
                    window = self._create_window(window_id, timestamp, key)
                    self.windows[window_id] = window
                    self.stats['windows_created'] += 1
                
                window = self.windows[window_id]
                
                # Check if event is within grace period
                if self._is_event_acceptable(window, timestamp):
                    window.data.append(event)
                    if metadata:
                        window.metadata.update(metadata)
                else:
                    self.stats['late_arrivals'] += 1
                    if not self.config.grace_period or \
                       timestamp < (window.end_time + self.config.grace_period):
                        self.stats['dropped_late_arrivals'] += 1
                        continue
                
                # Check if window is complete
                if window.is_complete(datetime.now(), self.config.grace_period):
                    completed_windows.append(window)
        
        return completed_windows
    
    def _update_watermark(self, timestamp: datetime):
        """Update watermark (highest timestamp seen)."""
        with self.watermark_lock:
            self.watermark = max(self.watermark, timestamp)
    
    def _get_target_windows(self, timestamp: datetime, key: Optional[str]) -> List[str]:
        """Get target window IDs for an event."""
        if self.config.window_type == WindowType.TUMBLING:
            return [self._get_tumbling_window_id(timestamp, key)]
        
        elif self.config.window_type == WindowType.SLIDING:
            return self._get_sliding_window_ids(timestamp, key)
        
        elif self.config.window_type == WindowType.SESSION:
            return self._get_session_window_ids(timestamp, key)
        
        elif self.config.window_type == WindowType.COUNT:
            return [self._get_count_window_id(key)]
        
        else:
            raise ValueError(f"Unsupported window type: {self.config.window_type}")
    
    def _get_tumbling_window_id(self, timestamp: datetime, key: Optional[str]) -> str:
        """Get tumbling window ID."""
        if not isinstance(self.config.size, timedelta):
            raise ValueError("Tumbling windows require timedelta size")
        
        # Align to window boundaries
        epoch = datetime(1970, 1, 1)
        seconds_since_epoch = (timestamp - epoch).total_seconds()
        window_seconds = self.config.size.total_seconds()
        
        window_start_seconds = int(seconds_since_epoch // window_seconds) * window_seconds
        window_start = epoch + timedelta(seconds=window_start_seconds)
        
        prefix = f"key_{key}_" if key else ""
        return f"{prefix}tumbling_{window_start.isoformat()}"
    
    def _get_sliding_window_ids(self, timestamp: datetime, key: Optional[str]) -> List[str]:
        """Get sliding window IDs."""
        if not isinstance(self.config.size, timedelta) or not isinstance(self.config.slide, timedelta):
            raise ValueError("Sliding windows require timedelta size and slide")
        
        window_ids = []
        slide_seconds = self.config.slide.total_seconds()
        size_seconds = self.config.size.total_seconds()
        
        # Calculate how many windows this event belongs to
        epoch = datetime(1970, 1, 1)
        seconds_since_epoch = (timestamp - epoch).total_seconds()
        
        # Find the latest window that starts before or at this timestamp
        latest_window_start_seconds = int(seconds_since_epoch // slide_seconds) * slide_seconds
        
        # Go back to find all windows that contain this timestamp
        current_start_seconds = latest_window_start_seconds
        while current_start_seconds > seconds_since_epoch - size_seconds:
            window_start = epoch + timedelta(seconds=current_start_seconds)
            
            prefix = f"key_{key}_" if key else ""
            window_id = f"{prefix}sliding_{window_start.isoformat()}"
            window_ids.append(window_id)
            
            current_start_seconds -= slide_seconds
        
        return window_ids
    
    def _get_session_window_ids(self, timestamp: datetime, key: Optional[str]) -> List[str]:
        """Get session window IDs (simplified - creates new session for each event)."""
        if not self.config.session_timeout:
            raise ValueError("Session windows require session_timeout")
        
        # For simplicity, create a session window per key
        # In a full implementation, you'd merge nearby sessions
        prefix = f"key_{key}_" if key else ""
        return [f"{prefix}session_{timestamp.isoformat()}"]
    
    def _get_count_window_id(self, key: Optional[str]) -> str:
        """Get count-based window ID."""
        if not isinstance(self.config.size, int):
            raise ValueError("Count windows require integer size")
        
        # Simple count-based windowing (round-robin)
        prefix = f"key_{key}_" if key else ""
        return f"{prefix}count_current"
    
    def _create_window(self, window_id: str, timestamp: datetime, key: Optional[str]) -> WindowFrame:
        """Create a new window frame."""
        if self.config.window_type in [WindowType.TUMBLING, WindowType.SLIDING]:
            # Extract start time from window ID
            parts = window_id.split('_')
            start_time_str = parts[-1]
            start_time = datetime.fromisoformat(start_time_str)
            end_time = start_time + self.config.size
        
        elif self.config.window_type == WindowType.SESSION:
            start_time = timestamp
            end_time = timestamp + self.config.session_timeout
        
        elif self.config.window_type == WindowType.COUNT:
            start_time = datetime.now()
            end_time = datetime.max  # Count windows don't have time bounds
        
        else:
            start_time = timestamp
            end_time = timestamp + self.config.size
        
        return WindowFrame(
            window_id=window_id,
            start_time=start_time,
            end_time=end_time,
            data=[],
            metadata={'key': key, 'created_at': datetime.now()}
        )
    
    def _is_event_acceptable(self, window: WindowFrame, timestamp: datetime) -> bool:
        """Check if event is acceptable for the window."""
        if self.config.window_type == WindowType.COUNT:
            # Count windows accept all events until full
            return len(window.data) < self.config.size
        
        # Time-based windows
        if timestamp < window.start_time:
            return False  # Too early
        
        if timestamp >= window.end_time:
            # Check grace period
            if self.config.grace_period:
                return timestamp < (window.end_time + self.config.grace_period)
            return False
        
        return True
    
    def _cleanup_expired_windows(self):
        """Clean up expired windows."""
        if not self.config.retention_time:
            return
        
        current_time = datetime.now()
        cutoff_time = current_time - self.config.retention_time
        
        expired_windows = []
        
        with self.window_lock.gen_rlock():
            for window_id, window in self.windows.items():
                if window.end_time < cutoff_time:
                    expired_windows.append(window_id)
        
        if expired_windows:
            with self.window_lock.gen_wlock():
                for window_id in expired_windows:
                    if window_id in self.windows:
                        del self.windows[window_id]
            
            logger.debug(f"Cleaned up {len(expired_windows)} expired windows")
    
    def get_completed_windows(self, remove: bool = True) -> List[WindowFrame]:
        """Get and optionally remove completed windows."""
        completed = []
        current_time = datetime.now()
        
        with self.window_lock.gen_wlock():
            to_remove = []
            
            for window_id, window in self.windows.items():
                if window.is_complete(current_time, self.config.grace_period):
                    completed.append(window)
                    if remove:
                        to_remove.append(window_id)
            
            # Remove completed windows if requested
            for window_id in to_remove:
                del self.windows[window_id]
                self.stats['windows_completed'] += 1
        
        return completed
    
    def get_window_by_id(self, window_id: str) -> Optional[WindowFrame]:
        """Get specific window by ID."""
        with self.window_lock.gen_rlock():
            return self.windows.get(window_id)
    
    def get_active_windows(self) -> List[WindowFrame]:
        """Get all active (incomplete) windows."""
        with self.window_lock.gen_rlock():
            return list(self.windows.values())
    
    def get_watermark(self) -> datetime:
        """Get current watermark."""
        with self.watermark_lock:
            return self.watermark
    
    def get_stats(self) -> Dict[str, Any]:
        """Get window manager statistics."""
        with self.window_lock.gen_rlock():
            active_windows = len(self.windows)
            total_events_in_windows = sum(len(w.data) for w in self.windows.values())
        
        with self.watermark_lock:
            watermark = self.watermark
        
        return {
            **self.stats.copy(),
            'active_windows': active_windows,
            'total_events_in_windows': total_events_in_windows,
            'watermark': watermark.isoformat() if watermark != datetime.min else None,
            'window_type': self.config.window_type.value,
            'window_size': str(self.config.size)
        }
    
    def apply_window_function(
        self,
        func: Callable[[WindowFrame], Any],
        window_filter: Optional[Callable[[WindowFrame], bool]] = None,
        remove_processed: bool = True
    ) -> List[Tuple[str, Any]]:
        """
        Apply function to windows and return results.
        
        Args:
            func: Function to apply to each window
            window_filter: Optional filter for windows
            remove_processed: Whether to remove processed windows
            
        Returns:
            List of (window_id, result) tuples
        """
        results = []
        
        with self.window_lock.gen_wlock():
            to_remove = []
            
            for window_id, window in self.windows.items():
                # Apply filter if provided
                if window_filter and not window_filter(window):
                    continue
                
                try:
                    result = func(window)
                    results.append((window_id, result))
                    
                    if remove_processed:
                        to_remove.append(window_id)
                        
                except Exception as e:
                    logger.error(f"Error applying function to window {window_id}: {e}")
            
            # Remove processed windows
            for window_id in to_remove:
                del self.windows[window_id]
        
        return results
    
    def force_complete_all_windows(self) -> List[WindowFrame]:
        """Force completion of all active windows."""
        with self.window_lock.gen_wlock():
            completed = list(self.windows.values())
            self.windows.clear()
            self.stats['windows_completed'] += len(completed)
        
        return completed
    
    def __enter__(self):
        """Context manager entry."""
        self.start_cleanup()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop_cleanup()


# Convenience functions
def create_tumbling_window(size: timedelta, grace_period: Optional[timedelta] = None) -> WindowConfig:
    """Create tumbling window configuration."""
    return WindowConfig(
        window_type=WindowType.TUMBLING,
        size=size,
        grace_period=grace_period
    )


def create_sliding_window(
    size: timedelta,
    slide: timedelta,
    grace_period: Optional[timedelta] = None
) -> WindowConfig:
    """Create sliding window configuration."""
    return WindowConfig(
        window_type=WindowType.SLIDING,
        size=size,
        slide=slide,
        grace_period=grace_period
    )


def create_session_window(
    session_timeout: timedelta,
    grace_period: Optional[timedelta] = None
) -> WindowConfig:
    """Create session window configuration."""
    return WindowConfig(
        window_type=WindowType.SESSION,
        size=session_timeout,  # Used as session timeout
        session_timeout=session_timeout,
        grace_period=grace_period
    )


def create_count_window(size: int) -> WindowConfig:
    """Create count-based window configuration."""
    return WindowConfig(
        window_type=WindowType.COUNT,
        size=size
    )


# Threading RWLock implementation
class RWLock:
    """Reader-Writer lock implementation."""
    
    def __init__(self):
        self._readers = 0
        self._readers_lock = threading.Lock()
        self._writers_lock = threading.Lock()
    
    def gen_rlock(self):
        """Generate reader lock context manager."""
        return self.RLockContext(self)
    
    def gen_wlock(self):
        """Generate writer lock context manager."""
        return self.WLockContext(self)
    
    def acquire_read(self):
        """Acquire read lock."""
        with self._readers_lock:
            self._readers += 1
            if self._readers == 1:
                self._writers_lock.acquire()
    
    def release_read(self):
        """Release read lock."""
        with self._readers_lock:
            self._readers -= 1
            if self._readers == 0:
                self._writers_lock.release()
    
    def acquire_write(self):
        """Acquire write lock."""
        self._writers_lock.acquire()
    
    def release_write(self):
        """Release write lock."""
        self._writers_lock.release()
    
    class RLockContext:
        def __init__(self, lock):
            self.lock = lock
        
        def __enter__(self):
            self.lock.acquire_read()
        
        def __exit__(self, exc_type, exc_val, exc_tb):
            self.lock.release_read()
    
    class WLockContext:
        def __init__(self, lock):
            self.lock = lock
        
        def __enter__(self):
            self.lock.acquire_write()
        
        def __exit__(self, exc_type, exc_val, exc_tb):
            self.lock.release_write()


# Monkey patch the RWLock into threading module
threading.RWLock = RWLock