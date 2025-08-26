"""
Resource management for distributed processing.
"""

import psutil
import os
import threading
import time
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
from collections import deque
from concurrent.futures import ThreadPoolExecutor, Future
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ResourceConfig:
    """Resource configuration."""
    max_cpu_percent: float = 80.0
    max_memory_percent: float = 85.0
    max_workers: Optional[int] = None
    adaptive_scaling: bool = True
    monitoring_interval: int = 5
    scale_up_threshold: float = 70.0
    scale_down_threshold: float = 30.0
    min_workers: int = 1
    max_concurrent_jobs: int = 100


@dataclass
class ResourceMetrics:
    """Resource usage metrics."""
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    memory_gb: float
    available_workers: int
    active_jobs: int
    queue_size: int
    throughput_jobs_per_min: float


class ResourceManager:
    """
    Smart resource management with adaptive scaling.
    
    Features:
    - Automatic worker scaling based on load
    - Resource monitoring and limits
    - Job queuing and prioritization
    - Performance optimization
    - Failure recovery
    """
    
    def __init__(self, config: Optional[ResourceConfig] = None):
        """Initialize resource manager."""
        self.config = config or ResourceConfig()
        
        # Auto-detect optimal worker count
        if self.config.max_workers is None:
            self.config.max_workers = min(os.cpu_count() or 4, 16)
        
        # Worker pool
        self.executor = ThreadPoolExecutor(
            max_workers=self.config.min_workers,
            thread_name_prefix="resource_worker"
        )
        
        # Job tracking
        self.active_jobs: Dict[str, Future] = {}
        self.job_queue = deque()
        self.completed_jobs = deque(maxlen=1000)
        
        # Metrics
        self.metrics_history = deque(maxlen=100)
        self.job_completion_times = deque(maxlen=100)
        
        # Monitoring
        self.monitoring = False
        self.monitor_thread = None
        self.current_workers = self.config.min_workers
        
        # Callbacks
        self.resource_callbacks = []
        
        logger.info(f"ResourceManager initialized with {self.current_workers} workers")
    
    def start_monitoring(self):
        """Start resource monitoring."""
        if self.monitoring:
            logger.warning("Resource monitoring already started")
            return
        
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
        logger.info("Resource monitoring started")
    
    def stop_monitoring(self):
        """Stop resource monitoring."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        logger.info("Resource monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.monitoring:
            try:
                # Collect metrics
                metrics = self._collect_metrics()
                self.metrics_history.append(metrics)
                
                # Check resource limits
                self._check_resource_limits(metrics)
                
                # Adaptive scaling
                if self.config.adaptive_scaling:
                    self._adaptive_scaling(metrics)
                
                # Trigger callbacks
                for callback in self.resource_callbacks:
                    try:
                        callback(metrics)
                    except Exception as e:
                        logger.error(f"Error in resource callback: {e}")
                
                time.sleep(self.config.monitoring_interval)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(self.config.monitoring_interval)
    
    def _collect_metrics(self) -> ResourceMetrics:
        """Collect current resource metrics."""
        # System metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        
        # Job metrics
        active_jobs = len(self.active_jobs)
        queue_size = len(self.job_queue)
        
        # Throughput calculation
        now = datetime.now()
        recent_completions = [
            completion_time for completion_time in self.job_completion_times
            if now - completion_time < timedelta(minutes=1)
        ]
        throughput = len(recent_completions)
        
        return ResourceMetrics(
            timestamp=now,
            cpu_percent=cpu_percent,
            memory_percent=memory.percent,
            memory_gb=memory.used / (1024**3),
            available_workers=self.current_workers - active_jobs,
            active_jobs=active_jobs,
            queue_size=queue_size,
            throughput_jobs_per_min=throughput
        )
    
    def _check_resource_limits(self, metrics: ResourceMetrics):
        """Check if resource limits are exceeded."""
        alerts = []
        
        if metrics.cpu_percent > self.config.max_cpu_percent:
            alerts.append(f"High CPU usage: {metrics.cpu_percent:.1f}%")
        
        if metrics.memory_percent > self.config.max_memory_percent:
            alerts.append(f"High memory usage: {metrics.memory_percent:.1f}%")
        
        if alerts:
            logger.warning(f"Resource limits exceeded: {', '.join(alerts)}")
            # Could implement job throttling here
    
    def _adaptive_scaling(self, metrics: ResourceMetrics):
        """Adaptive worker scaling based on load."""
        try:
            # Calculate load indicators
            cpu_load = metrics.cpu_percent
            queue_pressure = min(metrics.queue_size / 10, 100)  # Scale queue size to percentage
            worker_utilization = (metrics.active_jobs / self.current_workers * 100) if self.current_workers > 0 else 0
            
            # Combined load score
            load_score = np.mean([cpu_load, queue_pressure, worker_utilization])
            
            # Scaling decisions
            if load_score > self.config.scale_up_threshold and self.current_workers < self.config.max_workers:
                self._scale_up()
            elif load_score < self.config.scale_down_threshold and self.current_workers > self.config.min_workers:
                self._scale_down()
            
        except Exception as e:
            logger.error(f"Error in adaptive scaling: {e}")
    
    def _scale_up(self):
        """Scale up worker count."""
        try:
            new_workers = min(self.current_workers + 2, self.config.max_workers)
            if new_workers > self.current_workers:
                # Create new executor with more workers
                old_executor = self.executor
                self.executor = ThreadPoolExecutor(
                    max_workers=new_workers,
                    thread_name_prefix="resource_worker"
                )
                
                # Transfer pending jobs
                self._transfer_jobs(old_executor)
                
                self.current_workers = new_workers
                logger.info(f"Scaled up to {new_workers} workers")
                
        except Exception as e:
            logger.error(f"Failed to scale up: {e}")
    
    def _scale_down(self):
        """Scale down worker count."""
        try:
            new_workers = max(self.current_workers - 1, self.config.min_workers)
            if new_workers < self.current_workers:
                # Only scale down if no active jobs would be disrupted
                if len(self.active_jobs) <= new_workers:
                    old_executor = self.executor
                    self.executor = ThreadPoolExecutor(
                        max_workers=new_workers,
                        thread_name_prefix="resource_worker"
                    )
                    
                    # Transfer pending jobs
                    self._transfer_jobs(old_executor)
                    
                    self.current_workers = new_workers
                    logger.info(f"Scaled down to {new_workers} workers")
                
        except Exception as e:
            logger.error(f"Failed to scale down: {e}")
    
    def _transfer_jobs(self, old_executor: ThreadPoolExecutor):
        """Transfer jobs from old executor to new one."""
        # Graceful shutdown of old executor
        old_executor.shutdown(wait=False)
    
    def submit_job(
        self,
        func: Callable,
        *args,
        job_id: Optional[str] = None,
        priority: int = 0,
        timeout: Optional[int] = None,
        **kwargs
    ) -> str:
        """
        Submit job for execution.
        
        Args:
            func: Function to execute
            *args: Function arguments
            job_id: Optional job identifier
            priority: Job priority (higher = more priority)
            timeout: Job timeout in seconds
            **kwargs: Function keyword arguments
            
        Returns:
            Job ID
        """
        if job_id is None:
            job_id = f"job_{datetime.now().timestamp():.6f}"
        
        # Check concurrent job limits
        if len(self.active_jobs) + len(self.job_queue) >= self.config.max_concurrent_jobs:
            raise RuntimeError("Maximum concurrent jobs limit reached")
        
        # Submit job
        try:
            future = self.executor.submit(self._execute_job, func, job_id, timeout, *args, **kwargs)
            self.active_jobs[job_id] = future
            
            # Add completion callback
            future.add_done_callback(lambda f: self._job_completed(job_id, f))
            
            logger.debug(f"Submitted job {job_id}")
            return job_id
            
        except Exception as e:
            logger.error(f"Failed to submit job {job_id}: {e}")
            raise
    
    def _execute_job(self, func: Callable, job_id: str, timeout: Optional[int], *args, **kwargs):
        """Execute job with monitoring."""
        start_time = datetime.now()
        
        try:
            # Execute with optional timeout
            if timeout:
                import signal
                
                def timeout_handler(signum, frame):
                    raise TimeoutError(f"Job {job_id} timed out after {timeout} seconds")
                
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(timeout)
            
            result = func(*args, **kwargs)
            
            if timeout:
                signal.alarm(0)  # Cancel timeout
            
            return result
            
        except Exception as e:
            logger.error(f"Job {job_id} failed: {e}")
            raise
        finally:
            # Record execution time
            execution_time = (datetime.now() - start_time).total_seconds()
            logger.debug(f"Job {job_id} completed in {execution_time:.2f}s")
    
    def _job_completed(self, job_id: str, future: Future):
        """Handle job completion."""
        try:
            # Remove from active jobs
            if job_id in self.active_jobs:
                del self.active_jobs[job_id]
            
            # Record completion time
            self.job_completion_times.append(datetime.now())
            
            # Store result/error info
            job_info = {
                'job_id': job_id,
                'completed_at': datetime.now(),
                'success': not future.exception(),
                'exception': str(future.exception()) if future.exception() else None
            }
            self.completed_jobs.append(job_info)
            
        except Exception as e:
            logger.error(f"Error handling job completion: {e}")
    
    def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get job status."""
        if job_id in self.active_jobs:
            future = self.active_jobs[job_id]
            return {
                'job_id': job_id,
                'status': 'running' if future.running() else 'pending',
                'done': future.done()
            }
        
        # Check completed jobs
        for job_info in self.completed_jobs:
            if job_info['job_id'] == job_id:
                return {
                    'job_id': job_id,
                    'status': 'completed',
                    'success': job_info['success'],
                    'completed_at': job_info['completed_at'].isoformat(),
                    'exception': job_info['exception']
                }
        
        return None
    
    def get_job_result(self, job_id: str, timeout: Optional[float] = None):
        """Get job result."""
        if job_id in self.active_jobs:
            future = self.active_jobs[job_id]
            return future.result(timeout=timeout)
        
        raise ValueError(f"Job {job_id} not found or already completed")
    
    def cancel_job(self, job_id: str) -> bool:
        """Cancel running job."""
        if job_id in self.active_jobs:
            future = self.active_jobs[job_id]
            cancelled = future.cancel()
            
            if cancelled:
                del self.active_jobs[job_id]
                logger.info(f"Cancelled job {job_id}")
            
            return cancelled
        
        return False
    
    def wait_for_jobs(self, job_ids: List[str], timeout: Optional[float] = None):
        """Wait for multiple jobs to complete."""
        futures = []
        for job_id in job_ids:
            if job_id in self.active_jobs:
                futures.append(self.active_jobs[job_id])
        
        if futures:
            from concurrent.futures import wait
            wait(futures, timeout=timeout)
    
    def register_resource_callback(self, callback: Callable[[ResourceMetrics], None]):
        """Register callback for resource metrics."""
        self.resource_callbacks.append(callback)
    
    def get_resource_stats(self) -> Dict[str, Any]:
        """Get current resource statistics."""
        if not self.metrics_history:
            return {'message': 'No metrics available'}
        
        latest = self.metrics_history[-1]
        
        # Calculate averages from recent history
        recent_metrics = list(self.metrics_history)[-10:]
        avg_cpu = np.mean([m.cpu_percent for m in recent_metrics])
        avg_memory = np.mean([m.memory_percent for m in recent_metrics])
        avg_throughput = np.mean([m.throughput_jobs_per_min for m in recent_metrics])
        
        return {
            'current_workers': self.current_workers,
            'active_jobs': latest.active_jobs,
            'queue_size': latest.queue_size,
            'cpu_percent': latest.cpu_percent,
            'memory_percent': latest.memory_percent,
            'memory_gb': latest.memory_gb,
            'throughput_jobs_per_min': latest.throughput_jobs_per_min,
            'averages': {
                'cpu_percent': avg_cpu,
                'memory_percent': avg_memory,
                'throughput_jobs_per_min': avg_throughput
            },
            'config': {
                'max_workers': self.config.max_workers,
                'max_cpu_percent': self.config.max_cpu_percent,
                'max_memory_percent': self.config.max_memory_percent,
                'adaptive_scaling': self.config.adaptive_scaling
            }
        }
    
    def shutdown(self, wait: bool = True):
        """Shutdown resource manager."""
        logger.info("Shutting down resource manager...")
        
        self.stop_monitoring()
        
        if self.executor:
            self.executor.shutdown(wait=wait)
        
        logger.info("Resource manager shutdown complete")
    
    def __enter__(self):
        """Context manager entry."""
        self.start_monitoring()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.shutdown()