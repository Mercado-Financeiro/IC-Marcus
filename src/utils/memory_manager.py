"""
Centralized memory management service for the entire project.
Provides consistent memory optimization, monitoring, and cleanup.
"""

import gc
import psutil
import pandas as pd
import numpy as np
from typing import Optional, Dict, Any, List, Callable
from contextlib import contextmanager
import warnings
from functools import wraps
import torch

from .logging import log as logger


class MemoryManager:
    """
    Centralized memory management service.
    
    Features:
    - Automatic dataframe memory optimization
    - GPU memory management for PyTorch
    - Memory monitoring and profiling
    - Automatic garbage collection
    - Context managers for memory-intensive operations
    """
    
    def __init__(self):
        self.initial_memory = None
        self.peak_memory = None
        self.memory_history = []
        
    @staticmethod
    def optimize_dataframe(df: pd.DataFrame, 
                          verbose: bool = True,
                          deep: bool = True) -> pd.DataFrame:
        """
        Optimize DataFrame memory usage by downcasting types.
        
        Args:
            df: DataFrame to optimize
            verbose: Print memory savings
            deep: Apply deep optimization (may be slower)
            
        Returns:
            Optimized DataFrame
        """
        if df.empty:
            return df
            
        start_mem = df.memory_usage(deep=True).sum() / 1024**2
        
        # Optimize numeric columns
        for col in df.select_dtypes(include=['int']).columns:
            col_min = df[col].min()
            col_max = df[col].max()
            
            # Downcast integers
            if col_min >= 0:
                if col_max < 255:
                    df[col] = df[col].astype(np.uint8)
                elif col_max < 65535:
                    df[col] = df[col].astype(np.uint16)
                elif col_max < 4294967295:
                    df[col] = df[col].astype(np.uint32)
                else:
                    df[col] = df[col].astype(np.uint64)
            else:
                if col_min > np.iinfo(np.int8).min and col_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif col_min > np.iinfo(np.int16).min and col_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif col_min > np.iinfo(np.int32).min and col_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                else:
                    df[col] = df[col].astype(np.int64)
        
        # Optimize float columns
        for col in df.select_dtypes(include=['float']).columns:
            col_min = df[col].min()
            col_max = df[col].max()
            
            if deep:
                # Check if we can convert to int
                if df[col].dropna().apply(lambda x: x.is_integer()).all():
                    df[col] = df[col].astype(np.int32)
                    continue
            
            # Downcast floats
            if col_min > np.finfo(np.float16).min and col_max < np.finfo(np.float16).max:
                df[col] = df[col].astype(np.float16)
            elif col_min > np.finfo(np.float32).min and col_max < np.finfo(np.float32).max:
                df[col] = df[col].astype(np.float32)
            else:
                df[col] = df[col].astype(np.float64)
        
        # Optimize object columns
        if deep:
            for col in df.select_dtypes(include=['object']).columns:
                num_unique_values = len(df[col].unique())
                num_total_values = len(df[col])
                if num_unique_values / num_total_values < 0.5:
                    df[col] = df[col].astype('category')
        
        end_mem = df.memory_usage(deep=True).sum() / 1024**2
        
        if verbose:
            reduction_pct = 100 * (start_mem - end_mem) / start_mem
            logger.info(
                "dataframe_memory_optimized",
                start_mb=f"{start_mem:.2f}",
                end_mb=f"{end_mem:.2f}",
                reduction_pct=f"{reduction_pct:.1f}%"
            )
        
        return df
    
    @staticmethod
    def clean_memory(force: bool = False):
        """
        Perform garbage collection and memory cleanup.
        
        Args:
            force: Force full collection (slower but more thorough)
        """
        if force:
            for _ in range(3):
                gc.collect()
        else:
            gc.collect()
        
        # Clear PyTorch cache if available
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except:
            pass
    
    @contextmanager
    def memory_context(self, name: str = "operation", threshold_mb: float = 100):
        """
        Context manager for memory-intensive operations.
        
        Args:
            name: Name of the operation
            threshold_mb: Warning threshold for memory increase
            
        Example:
            with memory_manager.memory_context("training"):
                model.fit(X, y)
        """
        # Record initial memory
        process = psutil.Process()
        initial_mem = process.memory_info().rss / 1024**2
        
        logger.info(f"memory_context_start", operation=name, initial_mb=f"{initial_mem:.1f}")
        
        try:
            yield
        finally:
            # Cleanup and measure
            self.clean_memory()
            
            final_mem = process.memory_info().rss / 1024**2
            increase = final_mem - initial_mem
            
            if increase > threshold_mb:
                warnings.warn(
                    f"High memory increase in {name}: {increase:.1f} MB",
                    ResourceWarning
                )
            
            logger.info(
                f"memory_context_end",
                operation=name,
                final_mb=f"{final_mem:.1f}",
                increase_mb=f"{increase:.1f}"
            )
    
    def monitor_memory(self, interval_seconds: int = 1) -> Dict[str, float]:
        """
        Monitor current memory usage.
        
        Returns:
            Dictionary with memory statistics
        """
        process = psutil.Process()
        memory_info = process.memory_info()
        
        stats = {
            'rss_mb': memory_info.rss / 1024**2,
            'vms_mb': memory_info.vms / 1024**2,
            'percent': process.memory_percent(),
            'available_mb': psutil.virtual_memory().available / 1024**2
        }
        
        self.memory_history.append(stats)
        
        # Track peak memory
        if self.peak_memory is None or stats['rss_mb'] > self.peak_memory:
            self.peak_memory = stats['rss_mb']
        
        return stats
    
    @staticmethod
    def estimate_dataframe_memory(shape: tuple, dtype: str = 'float32') -> float:
        """
        Estimate memory requirement for a DataFrame.
        
        Args:
            shape: (rows, columns) tuple
            dtype: Data type of columns
            
        Returns:
            Estimated memory in MB
        """
        dtype_sizes = {
            'float64': 8, 'float32': 4, 'float16': 2,
            'int64': 8, 'int32': 4, 'int16': 2, 'int8': 1,
            'uint64': 8, 'uint32': 4, 'uint16': 2, 'uint8': 1,
            'object': 8, 'category': 1
        }
        
        bytes_per_element = dtype_sizes.get(dtype, 8)
        total_bytes = shape[0] * shape[1] * bytes_per_element
        
        return total_bytes / 1024**2
    
    def batch_process(self, 
                     data: pd.DataFrame,
                     func: Callable,
                     batch_size: Optional[int] = None,
                     **kwargs) -> pd.DataFrame:
        """
        Process large DataFrame in memory-efficient batches.
        
        Args:
            data: Input DataFrame
            func: Function to apply to each batch
            batch_size: Batch size (auto-calculated if None)
            **kwargs: Additional arguments for func
            
        Returns:
            Processed DataFrame
        """
        if batch_size is None:
            # Auto-calculate based on available memory
            available_mb = psutil.virtual_memory().available / 1024**2
            data_mb = data.memory_usage(deep=True).sum() / 1024**2
            
            # Use 25% of available memory per batch
            batch_size = max(
                100,
                int(len(data) * (available_mb * 0.25) / data_mb)
            )
        
        results = []
        n_batches = (len(data) + batch_size - 1) // batch_size
        
        logger.info(
            "batch_processing_started",
            total_rows=len(data),
            batch_size=batch_size,
            n_batches=n_batches
        )
        
        for i in range(0, len(data), batch_size):
            batch = data.iloc[i:i + batch_size]
            
            with self.memory_context(f"batch_{i//batch_size}"):
                result = func(batch, **kwargs)
                results.append(result)
            
            # Clean memory after each batch
            self.clean_memory()
        
        return pd.concat(results, ignore_index=True)
    
    @staticmethod
    def memory_efficient_decorator(threshold_mb: float = 500):
        """
        Decorator to ensure memory-efficient execution.
        
        Args:
            threshold_mb: Memory threshold to trigger cleanup
            
        Example:
            @memory_efficient_decorator(threshold_mb=1000)
            def process_data(df):
                return df.apply(complex_function)
        """
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                # Check memory before execution
                process = psutil.Process()
                initial_mem = process.memory_info().rss / 1024**2
                
                if initial_mem > threshold_mb:
                    logger.warning(
                        "high_memory_before_execution",
                        function=func.__name__,
                        memory_mb=f"{initial_mem:.1f}",
                        threshold_mb=threshold_mb
                    )
                    # Force cleanup
                    gc.collect()
                
                try:
                    result = func(*args, **kwargs)
                finally:
                    # Always cleanup after
                    gc.collect()
                
                return result
            
            return wrapper
        return decorator
    
    def get_memory_report(self) -> Dict[str, Any]:
        """
        Get comprehensive memory usage report.
        
        Returns:
            Dictionary with memory statistics and history
        """
        process = psutil.Process()
        vm = psutil.virtual_memory()
        
        return {
            'current': {
                'process_mb': process.memory_info().rss / 1024**2,
                'process_percent': process.memory_percent(),
                'system_total_mb': vm.total / 1024**2,
                'system_available_mb': vm.available / 1024**2,
                'system_percent': vm.percent
            },
            'peak_mb': self.peak_memory,
            'history': self.memory_history[-100:] if self.memory_history else [],
            'gpu': self._get_gpu_memory() if torch.cuda.is_available() else None
        }
    
    @staticmethod
    def _get_gpu_memory() -> Dict[str, float]:
        """Get GPU memory statistics if available."""
        try:
            return {
                'allocated_mb': torch.cuda.memory_allocated() / 1024**2,
                'reserved_mb': torch.cuda.memory_reserved() / 1024**2,
                'free_mb': (torch.cuda.memory_reserved() - torch.cuda.memory_allocated()) / 1024**2
            }
        except:
            return {}


# Global memory manager instance
memory_manager = MemoryManager()


# Convenience functions
def optimize_memory(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Convenience function to optimize DataFrame memory."""
    return memory_manager.optimize_dataframe(df, verbose)


def clean_memory(force: bool = False):
    """Convenience function to clean memory."""
    memory_manager.clean_memory(force)


def memory_context(name: str = "operation"):
    """Convenience function for memory context manager."""
    return memory_manager.memory_context(name)


# Export main components
__all__ = [
    'MemoryManager',
    'memory_manager',
    'optimize_memory',
    'clean_memory', 
    'memory_context'
]