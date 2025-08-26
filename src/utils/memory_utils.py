"""
Memory management utilities for ML pipelines.

Provides context managers and utilities for automatic memory cleanup,
tracking, and optimization during model training and optimization.
"""

import gc
import os
import psutil
import warnings
from contextlib import contextmanager
from typing import Optional, Dict, Any, List, Callable
from functools import wraps
import tracemalloc
import logging

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    
try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

# Setup logger
logger = logging.getLogger(__name__)


class MemoryTracker:
    """Track memory usage during execution."""
    
    def __init__(self, enabled: bool = True):
        """
        Initialize memory tracker.
        
        Args:
            enabled: Whether to enable tracking
        """
        self.enabled = enabled
        self.process = psutil.Process(os.getpid())
        self.initial_memory = None
        self.peak_memory = 0
        self.snapshots = []
        
    def start(self):
        """Start memory tracking."""
        if not self.enabled:
            return
            
        # Get initial memory usage
        self.initial_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        self.peak_memory = self.initial_memory
        
        # Start tracemalloc for detailed tracking
        if not tracemalloc.is_tracing():
            tracemalloc.start()
    
    def snapshot(self, label: str = ""):
        """Take a memory snapshot."""
        if not self.enabled:
            return
            
        current_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        self.peak_memory = max(self.peak_memory, current_memory)
        
        snapshot = {
            'label': label,
            'memory_mb': current_memory,
            'delta_mb': current_memory - self.initial_memory if self.initial_memory else 0
        }
        
        # Add GPU memory if available
        if TORCH_AVAILABLE and torch.cuda.is_available():
            snapshot['gpu_memory_mb'] = torch.cuda.memory_allocated() / 1024 / 1024
            snapshot['gpu_reserved_mb'] = torch.cuda.memory_reserved() / 1024 / 1024
        
        self.snapshots.append(snapshot)
        return snapshot
    
    def stop(self) -> Dict[str, Any]:
        """Stop tracking and return summary."""
        if not self.enabled:
            return {}
            
        final_memory = self.process.memory_info().rss / 1024 / 1024
        
        summary = {
            'initial_memory_mb': self.initial_memory,
            'final_memory_mb': final_memory,
            'peak_memory_mb': self.peak_memory,
            'total_delta_mb': final_memory - self.initial_memory if self.initial_memory else 0,
            'snapshots': self.snapshots
        }
        
        # Stop tracemalloc
        if tracemalloc.is_tracing():
            tracemalloc.stop()
            
        return summary


@contextmanager
def memory_cleanup(cleanup_func: Optional[Callable] = None, 
                   force_gc: bool = True,
                   clear_cuda: bool = True,
                   track: bool = False):
    """
    Context manager for automatic memory cleanup.
    
    Args:
        cleanup_func: Optional custom cleanup function
        force_gc: Whether to force garbage collection
        clear_cuda: Whether to clear CUDA cache
        track: Whether to track memory usage
        
    Example:
        with memory_cleanup():
            # Your memory-intensive code here
            model = train_model(data)
    """
    tracker = MemoryTracker(enabled=track) if track else None
    
    if tracker:
        tracker.start()
        
    try:
        yield tracker
    finally:
        # Run custom cleanup if provided
        if cleanup_func:
            try:
                cleanup_func()
            except Exception as e:
                logger.warning(f"Custom cleanup failed: {e}")
        
        # Force garbage collection
        if force_gc:
            gc.collect()
            
        # Clear CUDA cache if available
        if clear_cuda and TORCH_AVAILABLE and torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            
        # Final garbage collection
        if force_gc:
            gc.collect()
            
        if tracker:
            summary = tracker.stop()
            if summary.get('total_delta_mb', 0) > 100:  # Log if > 100MB leaked
                logger.warning(f"Memory delta: {summary['total_delta_mb']:.2f} MB")


@contextmanager
def optuna_memory_manager():
    """
    Context manager specifically for Optuna trials.
    Ensures proper cleanup after each trial.
    """
    try:
        yield
    finally:
        # Clear all Python objects
        gc.collect()
        
        # Clear CUDA if available
        if TORCH_AVAILABLE and torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            
        # Force another collection
        gc.collect()


def cleanup_after_trial(trial_func: Callable) -> Callable:
    """
    Decorator for Optuna objective functions to ensure cleanup after each trial.
    
    Example:
        @cleanup_after_trial
        def objective(trial):
            # Your trial code
            return score
    """
    @wraps(trial_func)
    def wrapper(*args, **kwargs):
        try:
            result = trial_func(*args, **kwargs)
            return result
        finally:
            # Cleanup
            gc.collect()
            if TORCH_AVAILABLE and torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    return wrapper


@contextmanager
def mlflow_run_context(run_name: Optional[str] = None, 
                       experiment_name: Optional[str] = None,
                       auto_end: bool = True):
    """
    Context manager for MLflow runs with automatic cleanup.
    
    Args:
        run_name: Name for the MLflow run
        experiment_name: Name of the experiment
        auto_end: Whether to automatically end the run
        
    Example:
        with mlflow_run_context(run_name="my_experiment"):
            mlflow.log_params(params)
            mlflow.log_metrics(metrics)
    """
    if not MLFLOW_AVAILABLE:
        yield None
        return
        
    run = None
    try:
        if experiment_name:
            mlflow.set_experiment(experiment_name)
            
        run = mlflow.start_run(run_name=run_name)
        yield run
    finally:
        if auto_end and run:
            try:
                mlflow.end_run()
            except Exception as e:
                logger.warning(f"Failed to end MLflow run: {e}")
                
        # Cleanup MLflow objects
        gc.collect()


class DataLoaderCleanup:
    """Context manager for PyTorch DataLoader cleanup."""
    
    def __init__(self, dataloader):
        """
        Initialize with a DataLoader.
        
        Args:
            dataloader: PyTorch DataLoader instance
        """
        self.dataloader = dataloader
        
    def __enter__(self):
        return self.dataloader
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Clean up DataLoader and its dataset."""
        if not TORCH_AVAILABLE:
            return
            
        # Delete dataset reference
        if hasattr(self.dataloader, 'dataset'):
            del self.dataloader.dataset
            
        # Clear any remaining batches
        if hasattr(self.dataloader, '_iterator'):
            del self.dataloader._iterator
            
        # Delete the dataloader itself
        del self.dataloader
        
        # Force cleanup
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def cleanup_model(model: Any, clear_cuda: bool = True):
    """
    Properly cleanup a PyTorch model.
    
    Args:
        model: PyTorch model instance
        clear_cuda: Whether to clear CUDA cache
    """
    if not TORCH_AVAILABLE:
        return
        
    try:
        # Move to CPU first if on CUDA
        if hasattr(model, 'cpu'):
            model.cpu()
            
        # Clear gradients
        if hasattr(model, 'zero_grad'):
            model.zero_grad(set_to_none=True)
            
        # Delete parameters
        if hasattr(model, 'parameters'):
            for param in model.parameters():
                if param.grad is not None:
                    del param.grad
                    
        # Delete the model
        del model
        
        # Clear CUDA cache
        if clear_cuda and torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        # Force garbage collection
        gc.collect()
        
    except Exception as e:
        logger.warning(f"Model cleanup failed: {e}")


def get_memory_info() -> Dict[str, float]:
    """
    Get current memory information.
    
    Returns:
        Dictionary with memory statistics in MB
    """
    process = psutil.Process(os.getpid())
    info = {
        'rss_mb': process.memory_info().rss / 1024 / 1024,
        'vms_mb': process.memory_info().vms / 1024 / 1024,
        'percent': process.memory_percent(),
        'available_mb': psutil.virtual_memory().available / 1024 / 1024
    }
    
    # Add GPU info if available
    if TORCH_AVAILABLE and torch.cuda.is_available():
        info['gpu_allocated_mb'] = torch.cuda.memory_allocated() / 1024 / 1024
        info['gpu_reserved_mb'] = torch.cuda.memory_reserved() / 1024 / 1024
        info['gpu_free_mb'] = (torch.cuda.memory_reserved() - torch.cuda.memory_allocated()) / 1024 / 1024
        
    return info


def log_memory_usage(label: str = "", logger_func: Optional[Callable] = None):
    """
    Log current memory usage.
    
    Args:
        label: Label for the log entry
        logger_func: Custom logging function (defaults to print)
    """
    if logger_func is None:
        logger_func = logger.info if logger else print
        
    info = get_memory_info()
    message = f"[Memory {label}] RSS: {info['rss_mb']:.1f}MB, Available: {info['available_mb']:.1f}MB"
    
    if 'gpu_allocated_mb' in info:
        message += f", GPU: {info['gpu_allocated_mb']:.1f}/{info['gpu_reserved_mb']:.1f}MB"
        
    logger_func(message)


def optimize_memory_settings():
    """
    Apply memory optimization settings for the current process.
    """
    # Python garbage collection optimization
    gc.set_threshold(700, 10, 10)  # More aggressive collection
    
    # PyTorch specific optimizations
    if TORCH_AVAILABLE and torch.cuda.is_available():
        try:
            # Reduce memory fragmentation
            if hasattr(torch.cuda, 'set_per_process_memory_fraction'):
                torch.cuda.set_per_process_memory_fraction(0.8)  # Limit to 80% of GPU
                
            # Enable memory efficient attention if available
            if hasattr(torch.backends, 'cuda'):
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
        except (AssertionError, RuntimeError):
            # CUDA not available or not compiled with CUDA
            pass
            
    # Set process priority to normal (not high)
    try:
        p = psutil.Process(os.getpid())
        p.nice(psutil.NORMAL_PRIORITY_CLASS if os.name == 'nt' else 0)
    except:
        pass


class MemoryMonitor:
    """
    Monitor memory usage and raise warnings if thresholds are exceeded.
    """
    
    def __init__(self, 
                 warn_threshold_mb: float = 10000,  # 10GB
                 error_threshold_mb: float = 15000,  # 15GB
                 check_interval: int = 100):  # Check every N calls
        """
        Initialize memory monitor.
        
        Args:
            warn_threshold_mb: Memory threshold for warnings
            error_threshold_mb: Memory threshold for errors
            check_interval: How often to check memory
        """
        self.warn_threshold_mb = warn_threshold_mb
        self.error_threshold_mb = error_threshold_mb
        self.check_interval = check_interval
        self.call_count = 0
        
    def check(self):
        """Check memory usage and raise warnings/errors if needed."""
        self.call_count += 1
        
        if self.call_count % self.check_interval != 0:
            return
            
        info = get_memory_info()
        current_mb = info['rss_mb']
        
        if current_mb > self.error_threshold_mb:
            raise MemoryError(f"Memory usage ({current_mb:.1f}MB) exceeds error threshold ({self.error_threshold_mb}MB)")
        elif current_mb > self.warn_threshold_mb:
            warnings.warn(f"High memory usage: {current_mb:.1f}MB", ResourceWarning)


# Create global monitor instance
memory_monitor = MemoryMonitor()


def safe_delete(*objects):
    """
    Safely delete objects and clear references.
    
    Args:
        *objects: Objects to delete
    """
    for obj in objects:
        try:
            # Clear PyTorch tensors
            if TORCH_AVAILABLE and isinstance(obj, torch.Tensor):
                if obj.grad is not None:
                    del obj.grad
                del obj
            # Clear DataFrames
            elif hasattr(obj, '__class__') and obj.__class__.__name__ == 'DataFrame':
                obj.drop(obj.index, inplace=True)
                del obj
            # Default deletion
            else:
                del obj
        except:
            pass
    
    gc.collect()