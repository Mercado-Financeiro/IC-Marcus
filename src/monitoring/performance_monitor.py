"""
Performance monitoring for ML trading pipeline.
"""

import time
import psutil
import GPUtil
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from pathlib import Path
import json
import logging
from dataclasses import dataclass, asdict
import threading
from collections import deque

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class SystemMetrics:
    """System performance metrics."""
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    memory_mb: float
    disk_percent: float
    gpu_percent: Optional[float] = None
    gpu_memory_mb: Optional[float] = None
    network_sent_mb: float = 0
    network_recv_mb: float = 0


@dataclass
class ModelMetrics:
    """Model performance metrics."""
    timestamp: datetime
    model_name: str
    prediction_latency_ms: float
    throughput_qps: float
    error_rate: float
    cache_hit_rate: float
    queue_size: int


@dataclass
class TradingMetrics:
    """Trading performance metrics."""
    timestamp: datetime
    total_positions: int
    open_positions: int
    pnl: float
    win_rate: float
    sharpe_ratio: float
    max_drawdown: float
    turnover: float


class PerformanceMonitor:
    """Monitor system, model, and trading performance."""
    
    def __init__(
        self,
        log_dir: str = "logs/monitoring",
        sampling_interval: int = 10,
        window_size: int = 100
    ):
        """Initialize performance monitor.
        
        Args:
            log_dir: Directory for logs
            sampling_interval: Seconds between samples
            window_size: Number of samples to keep in memory
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.sampling_interval = sampling_interval
        self.window_size = window_size
        
        # Metrics storage
        self.system_metrics = deque(maxlen=window_size)
        self.model_metrics = deque(maxlen=window_size)
        self.trading_metrics = deque(maxlen=window_size)
        
        # Monitoring thread
        self.monitoring = False
        self.monitor_thread = None
        
        # Performance thresholds
        self.thresholds = {
            'cpu_percent': 80,
            'memory_percent': 85,
            'disk_percent': 90,
            'prediction_latency_ms': 100,
            'error_rate': 0.01,
            'max_drawdown': 0.20
        }
        
        # Alert callbacks
        self.alert_callbacks = []
    
    def start(self):
        """Start monitoring."""
        if self.monitoring:
            logger.warning("Monitoring already started")
            return
        
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        logger.info("Performance monitoring started")
    
    def stop(self):
        """Stop monitoring."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        logger.info("Performance monitoring stopped")
    
    def _monitor_loop(self):
        """Main monitoring loop."""
        while self.monitoring:
            try:
                # Collect metrics
                system_metrics = self._collect_system_metrics()
                self.system_metrics.append(system_metrics)
                
                # Check thresholds
                self._check_thresholds(system_metrics)
                
                # Log metrics
                self._log_metrics(system_metrics)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
            
            time.sleep(self.sampling_interval)
    
    def _collect_system_metrics(self) -> SystemMetrics:
        """Collect system performance metrics."""
        # CPU and Memory
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        # Network (bytes)
        net_io = psutil.net_io_counters()
        
        # GPU (if available)
        gpu_percent = None
        gpu_memory_mb = None
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]
                gpu_percent = gpu.load * 100
                gpu_memory_mb = gpu.memoryUsed
        except:
            pass
        
        return SystemMetrics(
            timestamp=datetime.now(),
            cpu_percent=cpu_percent,
            memory_percent=memory.percent,
            memory_mb=memory.used / 1024 / 1024,
            disk_percent=disk.percent,
            gpu_percent=gpu_percent,
            gpu_memory_mb=gpu_memory_mb,
            network_sent_mb=net_io.bytes_sent / 1024 / 1024,
            network_recv_mb=net_io.bytes_recv / 1024 / 1024
        )
    
    def record_model_metrics(
        self,
        model_name: str,
        latency_ms: float,
        success: bool = True,
        cached: bool = False
    ):
        """Record model performance metrics.
        
        Args:
            model_name: Name of the model
            latency_ms: Prediction latency in milliseconds
            success: Whether prediction succeeded
            cached: Whether result was cached
        """
        # Calculate rates from recent history
        recent_model = [m for m in self.model_metrics if m.model_name == model_name][-10:]
        
        if recent_model:
            throughput = len(recent_model) / (
                (datetime.now() - recent_model[0].timestamp).total_seconds()
            )
            error_rate = sum(1 for m in recent_model if m.error_rate > 0) / len(recent_model)
            cache_rate = sum(1 for m in recent_model if m.cache_hit_rate > 0) / len(recent_model)
        else:
            throughput = 1.0
            error_rate = 0.0 if success else 1.0
            cache_rate = 1.0 if cached else 0.0
        
        metrics = ModelMetrics(
            timestamp=datetime.now(),
            model_name=model_name,
            prediction_latency_ms=latency_ms,
            throughput_qps=throughput,
            error_rate=error_rate,
            cache_hit_rate=cache_rate,
            queue_size=0  # TODO: Implement queue monitoring
        )
        
        self.model_metrics.append(metrics)
        self._check_model_thresholds(metrics)
    
    def record_trading_metrics(
        self,
        positions: Dict[str, Any],
        pnl: float,
        returns: pd.Series
    ):
        """Record trading performance metrics.
        
        Args:
            positions: Current positions dictionary
            pnl: Current P&L
            returns: Series of returns
        """
        # Calculate metrics
        total_positions = len(positions)
        open_positions = sum(1 for p in positions.values() if p.get('size', 0) != 0)
        
        win_rate = (returns > 0).mean() if len(returns) > 0 else 0
        sharpe = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
        
        # Maximum drawdown
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_dd = drawdown.min()
        
        # Turnover
        if len(self.trading_metrics) > 0:
            prev_positions = self.trading_metrics[-1].open_positions
            turnover = abs(open_positions - prev_positions) / max(prev_positions, 1)
        else:
            turnover = 0
        
        metrics = TradingMetrics(
            timestamp=datetime.now(),
            total_positions=total_positions,
            open_positions=open_positions,
            pnl=pnl,
            win_rate=win_rate,
            sharpe_ratio=sharpe,
            max_drawdown=max_dd,
            turnover=turnover
        )
        
        self.trading_metrics.append(metrics)
        self._check_trading_thresholds(metrics)
    
    def _check_thresholds(self, metrics: SystemMetrics):
        """Check system metrics against thresholds."""
        alerts = []
        
        if metrics.cpu_percent > self.thresholds['cpu_percent']:
            alerts.append(f"High CPU usage: {metrics.cpu_percent:.1f}%")
        
        if metrics.memory_percent > self.thresholds['memory_percent']:
            alerts.append(f"High memory usage: {metrics.memory_percent:.1f}%")
        
        if metrics.disk_percent > self.thresholds['disk_percent']:
            alerts.append(f"High disk usage: {metrics.disk_percent:.1f}%")
        
        for alert in alerts:
            self._trigger_alert('system', alert, metrics)
    
    def _check_model_thresholds(self, metrics: ModelMetrics):
        """Check model metrics against thresholds."""
        alerts = []
        
        if metrics.prediction_latency_ms > self.thresholds['prediction_latency_ms']:
            alerts.append(
                f"High prediction latency for {metrics.model_name}: "
                f"{metrics.prediction_latency_ms:.1f}ms"
            )
        
        if metrics.error_rate > self.thresholds['error_rate']:
            alerts.append(
                f"High error rate for {metrics.model_name}: "
                f"{metrics.error_rate:.2%}"
            )
        
        for alert in alerts:
            self._trigger_alert('model', alert, metrics)
    
    def _check_trading_thresholds(self, metrics: TradingMetrics):
        """Check trading metrics against thresholds."""
        alerts = []
        
        if abs(metrics.max_drawdown) > self.thresholds['max_drawdown']:
            alerts.append(f"Large drawdown: {metrics.max_drawdown:.2%}")
        
        if metrics.sharpe_ratio < 0:
            alerts.append(f"Negative Sharpe ratio: {metrics.sharpe_ratio:.2f}")
        
        for alert in alerts:
            self._trigger_alert('trading', alert, metrics)
    
    def _trigger_alert(self, category: str, message: str, metrics: Any):
        """Trigger an alert."""
        logger.warning(f"[{category.upper()}] {message}")
        
        # Call registered callbacks
        for callback in self.alert_callbacks:
            try:
                callback(category, message, metrics)
            except Exception as e:
                logger.error(f"Error in alert callback: {e}")
    
    def register_alert_callback(self, callback):
        """Register a callback for alerts.
        
        Args:
            callback: Function(category, message, metrics) to call on alert
        """
        self.alert_callbacks.append(callback)
    
    def _log_metrics(self, metrics: SystemMetrics):
        """Log metrics to file."""
        log_file = self.log_dir / f"system_{datetime.now():%Y%m%d}.jsonl"
        
        with open(log_file, 'a') as f:
            json.dump(asdict(metrics), f, default=str)
            f.write('\n')
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of recent metrics."""
        summary = {}
        
        # System metrics
        if self.system_metrics:
            recent_system = list(self.system_metrics)[-10:]
            summary['system'] = {
                'avg_cpu_percent': np.mean([m.cpu_percent for m in recent_system]),
                'avg_memory_percent': np.mean([m.memory_percent for m in recent_system]),
                'max_cpu_percent': max(m.cpu_percent for m in recent_system),
                'max_memory_percent': max(m.memory_percent for m in recent_system)
            }
        
        # Model metrics
        if self.model_metrics:
            recent_model = list(self.model_metrics)[-10:]
            summary['model'] = {
                'avg_latency_ms': np.mean([m.prediction_latency_ms for m in recent_model]),
                'max_latency_ms': max(m.prediction_latency_ms for m in recent_model),
                'avg_error_rate': np.mean([m.error_rate for m in recent_model]),
                'avg_cache_hit_rate': np.mean([m.cache_hit_rate for m in recent_model])
            }
        
        # Trading metrics
        if self.trading_metrics:
            latest_trading = self.trading_metrics[-1]
            summary['trading'] = {
                'open_positions': latest_trading.open_positions,
                'pnl': latest_trading.pnl,
                'win_rate': latest_trading.win_rate,
                'sharpe_ratio': latest_trading.sharpe_ratio,
                'max_drawdown': latest_trading.max_drawdown
            }
        
        return summary
    
    def export_metrics(self, output_file: str):
        """Export all metrics to file.
        
        Args:
            output_file: Path to output file
        """
        data = {
            'system': [asdict(m) for m in self.system_metrics],
            'model': [asdict(m) for m in self.model_metrics],
            'trading': [asdict(m) for m in self.trading_metrics],
            'summary': self.get_summary()
        }
        
        with open(output_file, 'w') as f:
            json.dump(data, f, default=str, indent=2)
        
        logger.info(f"Metrics exported to {output_file}")


# Context manager for monitoring
class monitor_performance:
    """Context manager for performance monitoring."""
    
    def __init__(self, model_name: str, monitor: PerformanceMonitor):
        self.model_name = model_name
        self.monitor = monitor
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        latency_ms = (time.time() - self.start_time) * 1000
        success = exc_type is None
        self.monitor.record_model_metrics(
            self.model_name,
            latency_ms,
            success=success
        )


if __name__ == "__main__":
    # Example usage
    monitor = PerformanceMonitor()
    
    # Register alert callback
    def alert_handler(category, message, metrics):
        print(f"ALERT [{category}]: {message}")
    
    monitor.register_alert_callback(alert_handler)
    
    # Start monitoring
    monitor.start()
    
    try:
        # Simulate some work
        for i in range(10):
            # Record model metrics
            with monitor_performance("xgboost", monitor):
                time.sleep(0.1)  # Simulate prediction
            
            # Get summary
            if i % 5 == 0:
                summary = monitor.get_summary()
                print(f"Summary: {summary}")
            
            time.sleep(1)
    
    finally:
        # Stop monitoring
        monitor.stop()
        
        # Export metrics
        monitor.export_metrics("performance_metrics.json")