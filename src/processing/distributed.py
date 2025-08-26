"""
Distributed processing with Ray/Dask backend support.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Callable, Union, Tuple
from dataclasses import dataclass
from pathlib import Path
import logging
import time
from concurrent.futures import Future, as_completed
import threading
from abc import ABC, abstractmethod

from .resource_manager import ResourceManager, ResourceConfig
from .chunked_processor import ChunkedProcessor, ChunkConfig

logger = logging.getLogger(__name__)


@dataclass
class DistributedConfig:
    """Configuration for distributed processing."""
    backend: str = "threading"  # threading, ray, dask
    n_workers: Optional[int] = None
    memory_per_worker_gb: float = 2.0
    ray_address: Optional[str] = None
    dask_scheduler: Optional[str] = None
    fault_tolerance: bool = True
    max_retries: int = 3
    timeout_seconds: int = 3600


class DistributedBackend(ABC):
    """Abstract base class for distributed backends."""
    
    @abstractmethod
    def initialize(self, config: DistributedConfig) -> bool:
        """Initialize the backend."""
        pass
    
    @abstractmethod
    def shutdown(self):
        """Shutdown the backend."""
        pass
    
    @abstractmethod
    def submit(self, func: Callable, *args, **kwargs) -> Any:
        """Submit task for execution."""
        pass
    
    @abstractmethod
    def gather(self, futures: List[Any]) -> List[Any]:
        """Gather results from futures."""
        pass
    
    @abstractmethod
    def get_worker_info(self) -> Dict[str, Any]:
        """Get worker information."""
        pass


class ThreadingBackend(DistributedBackend):
    """Threading-based backend for local parallel processing."""
    
    def __init__(self):
        self.resource_manager = None
    
    def initialize(self, config: DistributedConfig) -> bool:
        """Initialize threading backend."""
        try:
            resource_config = ResourceConfig(
                max_workers=config.n_workers,
                adaptive_scaling=True,
                max_memory_percent=85.0
            )
            self.resource_manager = ResourceManager(resource_config)
            self.resource_manager.start_monitoring()
            
            logger.info(f"Threading backend initialized with {config.n_workers} workers")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize threading backend: {e}")
            return False
    
    def shutdown(self):
        """Shutdown threading backend."""
        if self.resource_manager:
            self.resource_manager.shutdown()
    
    def submit(self, func: Callable, *args, **kwargs) -> str:
        """Submit task to thread pool."""
        return self.resource_manager.submit_job(func, *args, **kwargs)
    
    def gather(self, job_ids: List[str]) -> List[Any]:
        """Gather results from job IDs."""
        results = []
        for job_id in job_ids:
            result = self.resource_manager.get_job_result(job_id)
            results.append(result)
        return results
    
    def get_worker_info(self) -> Dict[str, Any]:
        """Get worker information."""
        if self.resource_manager:
            return self.resource_manager.get_resource_stats()
        return {}


class RayBackend(DistributedBackend):
    """Ray-based distributed backend."""
    
    def __init__(self):
        self.ray = None
        self.initialized = False
    
    def initialize(self, config: DistributedConfig) -> bool:
        """Initialize Ray backend."""
        try:
            import ray
            self.ray = ray
            
            # Initialize Ray
            if config.ray_address:
                ray.init(address=config.ray_address)
            else:
                ray.init(
                    num_cpus=config.n_workers,
                    object_store_memory=int(config.memory_per_worker_gb * 1e9)
                )
            
            self.initialized = True
            logger.info(f"Ray backend initialized")
            return True
            
        except ImportError:
            logger.error("Ray not available. Install with: pip install ray")
            return False
        except Exception as e:
            logger.error(f"Failed to initialize Ray backend: {e}")
            return False
    
    def shutdown(self):
        """Shutdown Ray backend."""
        if self.initialized and self.ray:
            self.ray.shutdown()
            self.initialized = False
    
    def submit(self, func: Callable, *args, **kwargs) -> Any:
        """Submit task to Ray."""
        if not self.initialized:
            raise RuntimeError("Ray backend not initialized")
        
        # Convert function to Ray remote function
        remote_func = self.ray.remote(func)
        return remote_func.remote(*args, **kwargs)
    
    def gather(self, futures: List[Any]) -> List[Any]:
        """Gather results from Ray futures."""
        if not self.initialized:
            raise RuntimeError("Ray backend not initialized")
        
        return self.ray.get(futures)
    
    def get_worker_info(self) -> Dict[str, Any]:
        """Get Ray cluster information."""
        if not self.initialized:
            return {}
        
        try:
            cluster_resources = self.ray.cluster_resources()
            return {
                'backend': 'ray',
                'cluster_resources': cluster_resources,
                'nodes': len(self.ray.nodes())
            }
        except Exception as e:
            logger.warning(f"Failed to get Ray cluster info: {e}")
            return {'backend': 'ray', 'status': 'unknown'}


class DaskBackend(DistributedBackend):
    """Dask-based distributed backend."""
    
    def __init__(self):
        self.client = None
        self.initialized = False
    
    def initialize(self, config: DistributedConfig) -> bool:
        """Initialize Dask backend."""
        try:
            import dask
            from dask.distributed import Client, LocalCluster
            
            if config.dask_scheduler:
                self.client = Client(config.dask_scheduler)
            else:
                # Create local cluster
                cluster = LocalCluster(
                    n_workers=config.n_workers or 4,
                    memory_limit=f"{config.memory_per_worker_gb}GB"
                )
                self.client = Client(cluster)
            
            self.initialized = True
            logger.info(f"Dask backend initialized: {self.client}")
            return True
            
        except ImportError:
            logger.error("Dask not available. Install with: pip install dask[distributed]")
            return False
        except Exception as e:
            logger.error(f"Failed to initialize Dask backend: {e}")
            return False
    
    def shutdown(self):
        """Shutdown Dask backend."""
        if self.client:
            self.client.close()
            self.initialized = False
    
    def submit(self, func: Callable, *args, **kwargs) -> Any:
        """Submit task to Dask."""
        if not self.initialized:
            raise RuntimeError("Dask backend not initialized")
        
        return self.client.submit(func, *args, **kwargs)
    
    def gather(self, futures: List[Any]) -> List[Any]:
        """Gather results from Dask futures."""
        if not self.initialized:
            raise RuntimeError("Dask backend not initialized")
        
        return self.client.gather(futures)
    
    def get_worker_info(self) -> Dict[str, Any]:
        """Get Dask cluster information."""
        if not self.initialized or not self.client:
            return {}
        
        try:
            info = self.client.scheduler_info()
            return {
                'backend': 'dask',
                'workers': len(info['workers']),
                'total_cores': sum(w['ncores'] for w in info['workers'].values()),
                'total_memory_gb': sum(w['memory_limit'] for w in info['workers'].values()) / 1e9
            }
        except Exception as e:
            logger.warning(f"Failed to get Dask cluster info: {e}")
            return {'backend': 'dask', 'status': 'unknown'}


class DistributedProcessor:
    """
    High-level distributed processing orchestrator.
    
    Features:
    - Multiple backend support (Threading, Ray, Dask)
    - Automatic backend selection
    - Fault tolerance and retries
    - Load balancing and work stealing
    - Performance monitoring
    - Graceful degradation
    """
    
    def __init__(self, config: Optional[DistributedConfig] = None):
        """Initialize distributed processor."""
        self.config = config or DistributedConfig()
        self.backend: Optional[DistributedBackend] = None
        self.chunked_processor: Optional[ChunkedProcessor] = None
        
        # Performance tracking
        self.task_history = []
        self.performance_metrics = {}
        
        logger.info(f"DistributedProcessor initialized with backend: {self.config.backend}")
    
    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()
    
    def start(self):
        """Start distributed processing."""
        self.backend = self._create_backend()
        
        if not self.backend.initialize(self.config):
            logger.warning(f"Failed to initialize {self.config.backend} backend, falling back to threading")
            self.config.backend = "threading"
            self.backend = self._create_backend()
            if not self.backend.initialize(self.config):
                raise RuntimeError("Failed to initialize any backend")
        
        # Initialize chunked processor
        chunk_config = ChunkConfig(
            memory_limit_mb=self.config.memory_per_worker_gb * 1000,
            adaptive_sizing=True
        )
        self.chunked_processor = ChunkedProcessor(config=chunk_config)
        
        logger.info("Distributed processing started")
    
    def stop(self):
        """Stop distributed processing."""
        if self.chunked_processor:
            self.chunked_processor.cleanup()
        
        if self.backend:
            self.backend.shutdown()
        
        logger.info("Distributed processing stopped")
    
    def _create_backend(self) -> DistributedBackend:
        """Create appropriate backend."""
        if self.config.backend == "ray":
            return RayBackend()
        elif self.config.backend == "dask":
            return DaskBackend()
        else:
            return ThreadingBackend()
    
    def map(
        self,
        func: Callable,
        data_list: List[Any],
        chunk_size: Optional[int] = None,
        timeout: Optional[int] = None
    ) -> List[Any]:
        """
        Apply function to list of data in parallel.
        
        Args:
            func: Function to apply
            data_list: List of data items
            chunk_size: Chunk size for batching
            timeout: Timeout in seconds
            
        Returns:
            List of results
        """
        if not self.backend:
            raise RuntimeError("Backend not initialized")
        
        start_time = time.time()
        timeout = timeout or self.config.timeout_seconds
        
        # Determine chunking strategy
        if chunk_size is None:
            chunk_size = max(1, len(data_list) // (self.config.n_workers or 4))
        
        # Create chunks
        chunks = [
            data_list[i:i + chunk_size]
            for i in range(0, len(data_list), chunk_size)
        ]
        
        # Submit tasks with retry logic
        futures = []
        for i, chunk in enumerate(chunks):
            future = self._submit_with_retry(
                lambda data_chunk: [func(item) for item in data_chunk],
                chunk,
                task_id=f"map_chunk_{i}"
            )
            futures.append(future)
        
        # Gather results
        try:
            chunk_results = self.backend.gather(futures)
            
            # Flatten results
            results = []
            for chunk_result in chunk_results:
                results.extend(chunk_result)
            
            # Record performance
            duration = time.time() - start_time
            self._record_performance('map', len(data_list), duration)
            
            return results
            
        except Exception as e:
            logger.error(f"Map operation failed: {e}")
            raise
    
    def map_partitions(
        self,
        func: Callable[[pd.DataFrame], pd.DataFrame],
        df: pd.DataFrame,
        n_partitions: Optional[int] = None,
        combine: bool = True
    ) -> Union[pd.DataFrame, List[pd.DataFrame]]:
        """
        Apply function to DataFrame partitions.
        
        Args:
            func: Function to apply to each partition
            df: Input DataFrame
            n_partitions: Number of partitions
            combine: Whether to combine results
            
        Returns:
            Combined DataFrame or list of DataFrames
        """
        if not self.backend:
            raise RuntimeError("Backend not initialized")
        
        start_time = time.time()
        
        # Determine partitions
        if n_partitions is None:
            n_partitions = self.config.n_workers or 4
        
        partition_size = len(df) // n_partitions
        partitions = []
        
        for i in range(n_partitions):
            start_idx = i * partition_size
            if i == n_partitions - 1:
                end_idx = len(df)
            else:
                end_idx = (i + 1) * partition_size
            
            partitions.append(df.iloc[start_idx:end_idx])
        
        # Submit tasks
        futures = []
        for i, partition in enumerate(partitions):
            if len(partition) > 0:  # Skip empty partitions
                future = self._submit_with_retry(
                    func, partition, task_id=f"partition_{i}"
                )
                futures.append(future)
        
        # Gather results
        try:
            results = self.backend.gather(futures)
            
            # Filter out empty results
            results = [r for r in results if not r.empty if isinstance(r, pd.DataFrame) else r is not None]
            
            if combine and results:
                if all(isinstance(r, pd.DataFrame) for r in results):
                    combined_result = pd.concat(results, ignore_index=True)
                else:
                    combined_result = results
            else:
                combined_result = results
            
            # Record performance
            duration = time.time() - start_time
            self._record_performance('map_partitions', len(df), duration)
            
            return combined_result
            
        except Exception as e:
            logger.error(f"Map partitions operation failed: {e}")
            raise
    
    def reduce(
        self,
        func: Callable[[Any, Any], Any],
        data_list: List[Any],
        initial_value: Any = None
    ) -> Any:
        """
        Reduce operation with tree reduction for efficiency.
        
        Args:
            func: Reduction function
            data_list: List of data to reduce
            initial_value: Initial value for reduction
            
        Returns:
            Reduced result
        """
        if not data_list:
            return initial_value
        
        if len(data_list) == 1:
            return data_list[0] if initial_value is None else func(initial_value, data_list[0])
        
        # Tree reduction for better parallelism
        current_level = data_list[:]
        if initial_value is not None:
            current_level.insert(0, initial_value)
        
        while len(current_level) > 1:
            next_level = []
            futures = []
            
            # Pair up items for parallel reduction
            for i in range(0, len(current_level), 2):
                if i + 1 < len(current_level):
                    # Submit pairwise reduction
                    future = self._submit_with_retry(
                        func, current_level[i], current_level[i + 1],
                        task_id=f"reduce_{i//2}"
                    )
                    futures.append(future)
                else:
                    # Odd item carries to next level
                    next_level.append(current_level[i])
            
            # Gather paired results
            if futures:
                paired_results = self.backend.gather(futures)
                next_level.extend(paired_results)
            
            current_level = next_level
        
        return current_level[0]
    
    def process_large_dataset(
        self,
        data_source: Union[str, pd.DataFrame, List[str]],
        process_func: Callable[[pd.DataFrame], pd.DataFrame],
        combine_func: Optional[Callable[[List[pd.DataFrame]], pd.DataFrame]] = None,
        file_pattern: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Process large datasets efficiently.
        
        Args:
            data_source: File path, DataFrame, or list of file paths
            process_func: Function to apply to each chunk
            combine_func: Function to combine results
            file_pattern: Pattern for file matching
            
        Returns:
            Processed DataFrame
        """
        start_time = time.time()
        
        if isinstance(data_source, str):
            # Single file
            return self._process_single_file(data_source, process_func, combine_func)
        
        elif isinstance(data_source, list):
            # Multiple files
            return self._process_multiple_files(data_source, process_func, combine_func)
        
        elif isinstance(data_source, pd.DataFrame):
            # DataFrame partitioning
            result = self.map_partitions(process_func, data_source)
            
            if combine_func and isinstance(result, list):
                result = combine_func(result)
            
            duration = time.time() - start_time
            self._record_performance('large_dataset', len(data_source), duration)
            
            return result
        
        else:
            raise ValueError(f"Unsupported data source type: {type(data_source)}")
    
    def _process_single_file(
        self,
        file_path: str,
        process_func: Callable[[pd.DataFrame], pd.DataFrame],
        combine_func: Optional[Callable[[List[pd.DataFrame]], pd.DataFrame]]
    ) -> pd.DataFrame:
        """Process single large file."""
        if not self.chunked_processor:
            raise RuntimeError("Chunked processor not initialized")
        
        # Use chunked processor with distributed backend
        def distributed_process(chunk):
            future = self._submit_with_retry(process_func, chunk)
            return self.backend.gather([future])[0]
        
        return self.chunked_processor.process_file_chunks(
            file_path=file_path,
            process_func=distributed_process,
            combine_func=combine_func,
            parallel=False  # Already using distributed processing
        )
    
    def _process_multiple_files(
        self,
        file_paths: List[str],
        process_func: Callable[[pd.DataFrame], pd.DataFrame],
        combine_func: Optional[Callable[[List[pd.DataFrame]], pd.DataFrame]]
    ) -> pd.DataFrame:
        """Process multiple files in parallel."""
        # Submit file processing tasks
        futures = []
        for i, file_path in enumerate(file_paths):
            def process_file(path):
                if path.endswith('.parquet'):
                    df = pd.read_parquet(path)
                else:
                    df = pd.read_csv(path)
                return process_func(df)
            
            future = self._submit_with_retry(
                process_file, file_path, task_id=f"file_{i}"
            )
            futures.append(future)
        
        # Gather results
        results = self.backend.gather(futures)
        
        # Combine results
        if combine_func:
            return combine_func(results)
        elif all(isinstance(r, pd.DataFrame) for r in results):
            return pd.concat(results, ignore_index=True)
        else:
            return results[0] if len(results) == 1 else results
    
    def _submit_with_retry(
        self,
        func: Callable,
        *args,
        task_id: Optional[str] = None,
        **kwargs
    ) -> Any:
        """Submit task with retry logic."""
        for attempt in range(self.config.max_retries + 1):
            try:
                future = self.backend.submit(func, *args, **kwargs)
                
                if task_id:
                    self.task_history.append({
                        'task_id': task_id,
                        'submitted_at': time.time(),
                        'attempt': attempt + 1
                    })
                
                return future
                
            except Exception as e:
                if attempt == self.config.max_retries:
                    logger.error(f"Task {task_id} failed after {attempt + 1} attempts: {e}")
                    raise
                else:
                    logger.warning(f"Task {task_id} attempt {attempt + 1} failed: {e}, retrying...")
                    time.sleep(2 ** attempt)  # Exponential backoff
    
    def _record_performance(self, operation: str, data_size: int, duration: float):
        """Record performance metrics."""
        if operation not in self.performance_metrics:
            self.performance_metrics[operation] = []
        
        self.performance_metrics[operation].append({
            'data_size': data_size,
            'duration': duration,
            'throughput': data_size / duration if duration > 0 else 0,
            'timestamp': time.time()
        })
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        stats = {}
        
        for operation, metrics in self.performance_metrics.items():
            if metrics:
                durations = [m['duration'] for m in metrics]
                throughputs = [m['throughput'] for m in metrics]
                
                stats[operation] = {
                    'count': len(metrics),
                    'avg_duration': np.mean(durations),
                    'avg_throughput': np.mean(throughputs),
                    'max_throughput': np.max(throughputs),
                    'total_data_processed': sum(m['data_size'] for m in metrics)
                }
        
        # Add backend info
        if self.backend:
            stats['backend_info'] = self.backend.get_worker_info()
        
        return stats
    
    def get_cluster_info(self) -> Dict[str, Any]:
        """Get cluster information."""
        if self.backend:
            return self.backend.get_worker_info()
        return {'status': 'not_initialized'}


# Convenience functions
def distributed_map(
    func: Callable,
    data_list: List[Any],
    backend: str = "threading",
    n_workers: Optional[int] = None
) -> List[Any]:
    """
    Convenience function for distributed map operation.
    
    Args:
        func: Function to apply
        data_list: List of data
        backend: Backend to use
        n_workers: Number of workers
        
    Returns:
        List of results
    """
    config = DistributedConfig(backend=backend, n_workers=n_workers)
    
    with DistributedProcessor(config) as processor:
        return processor.map(func, data_list)


def distributed_dataframe_apply(
    df: pd.DataFrame,
    func: Callable[[pd.DataFrame], pd.DataFrame],
    backend: str = "threading",
    n_partitions: Optional[int] = None
) -> pd.DataFrame:
    """
    Convenience function for distributed DataFrame processing.
    
    Args:
        df: Input DataFrame
        func: Function to apply
        backend: Backend to use
        n_partitions: Number of partitions
        
    Returns:
        Processed DataFrame
    """
    config = DistributedConfig(backend=backend)
    
    with DistributedProcessor(config) as processor:
        return processor.map_partitions(func, df, n_partitions)