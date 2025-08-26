"""
Intelligent chunked processing for memory-efficient operations.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Callable, Iterator, Union, Tuple
from dataclasses import dataclass
import logging
import psutil
from pathlib import Path
import pickle
import tempfile
import os
from concurrent.futures import as_completed
from functools import partial

from .resource_manager import ResourceManager, ResourceConfig

logger = logging.getLogger(__name__)


@dataclass
class ChunkConfig:
    """Configuration for chunked processing."""
    chunk_size: Optional[int] = None
    memory_limit_mb: float = 1000
    overlap_rows: int = 0
    preserve_order: bool = True
    use_temp_files: bool = True
    compression: str = 'gzip'
    adaptive_sizing: bool = True
    min_chunk_size: int = 1000
    max_chunk_size: int = 1000000


class ChunkedProcessor:
    """
    Intelligent chunked processing with adaptive sizing and memory management.
    
    Features:
    - Adaptive chunk sizing based on memory usage
    - Overlap handling for rolling window operations
    - Parallel processing with work-stealing
    - Temporary file management for large results
    - Progress tracking and error recovery
    """
    
    def __init__(
        self,
        resource_manager: Optional[ResourceManager] = None,
        config: Optional[ChunkConfig] = None
    ):
        """Initialize chunked processor."""
        self.config = config or ChunkConfig()
        
        # Resource management
        if resource_manager is None:
            resource_config = ResourceConfig(
                max_memory_percent=80.0,
                adaptive_scaling=True
            )
            self.resource_manager = ResourceManager(resource_config)
            self._own_resource_manager = True
        else:
            self.resource_manager = resource_manager
            self._own_resource_manager = False
        
        # Temporary files tracking
        self.temp_files = []
        self.temp_dir = None
        
        logger.info("ChunkedProcessor initialized")
    
    def __enter__(self):
        """Context manager entry."""
        if self._own_resource_manager:
            self.resource_manager.start_monitoring()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.cleanup()
        if self._own_resource_manager:
            self.resource_manager.shutdown()
    
    def _estimate_optimal_chunk_size(self, data: pd.DataFrame) -> int:
        """Estimate optimal chunk size based on data and available memory."""
        if not self.config.adaptive_sizing:
            return self.config.chunk_size or self.config.min_chunk_size
        
        # Get available memory
        memory = psutil.virtual_memory()
        available_mb = (memory.available / (1024 ** 2)) * 0.5  # Use 50% of available
        target_mb = min(available_mb, self.config.memory_limit_mb)
        
        # Estimate memory per row
        sample_size = min(1000, len(data))
        sample = data.head(sample_size)
        
        # Calculate memory usage of sample
        sample_memory_mb = sample.memory_usage(deep=True).sum() / (1024 ** 2)
        memory_per_row = sample_memory_mb / sample_size
        
        # Calculate optimal chunk size
        if memory_per_row > 0:
            optimal_size = int(target_mb / memory_per_row)
        else:
            optimal_size = self.config.min_chunk_size
        
        # Apply bounds
        chunk_size = max(
            self.config.min_chunk_size,
            min(optimal_size, self.config.max_chunk_size)
        )
        
        logger.debug(f"Estimated optimal chunk size: {chunk_size} rows")
        return chunk_size
    
    def _create_chunks(
        self,
        data: Union[pd.DataFrame, int],
        chunk_size: int
    ) -> List[Tuple[int, int]]:
        """Create chunk boundaries."""
        if isinstance(data, pd.DataFrame):
            total_rows = len(data)
        else:
            total_rows = data
        
        chunks = []
        overlap = self.config.overlap_rows
        
        start = 0
        while start < total_rows:
            end = min(start + chunk_size, total_rows)
            
            # Add overlap for non-first chunks
            chunk_start = max(0, start - overlap) if start > 0 else start
            chunks.append((chunk_start, end))
            
            start = end
        
        logger.debug(f"Created {len(chunks)} chunks for {total_rows} rows")
        return chunks
    
    def _setup_temp_dir(self):
        """Setup temporary directory for intermediate results."""
        if self.temp_dir is None:
            self.temp_dir = tempfile.mkdtemp(prefix='chunked_proc_')
            logger.debug(f"Created temp directory: {self.temp_dir}")
    
    def _save_chunk_result(self, result: Any, chunk_id: str) -> str:
        """Save chunk result to temporary file."""
        self._setup_temp_dir()
        
        temp_file = os.path.join(self.temp_dir, f"chunk_{chunk_id}.pkl")
        
        try:
            with open(temp_file, 'wb') as f:
                pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            self.temp_files.append(temp_file)
            return temp_file
            
        except Exception as e:
            logger.error(f"Failed to save chunk result: {e}")
            raise
    
    def _load_chunk_result(self, temp_file: str) -> Any:
        """Load chunk result from temporary file."""
        try:
            with open(temp_file, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            logger.error(f"Failed to load chunk result: {e}")
            raise
    
    def process_dataframe_chunks(
        self,
        data: pd.DataFrame,
        process_func: Callable[[pd.DataFrame], Any],
        combine_func: Optional[Callable[[List[Any]], Any]] = None,
        chunk_size: Optional[int] = None,
        parallel: bool = True,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> Any:
        """
        Process DataFrame in chunks with optional parallelization.
        
        Args:
            data: Input DataFrame
            process_func: Function to apply to each chunk
            combine_func: Function to combine chunk results
            chunk_size: Chunk size (auto-calculated if None)
            parallel: Whether to use parallel processing
            progress_callback: Progress callback function(current, total)
            
        Returns:
            Combined result
        """
        logger.info(f"Processing DataFrame with {len(data)} rows")
        
        # Determine chunk size
        if chunk_size is None:
            chunk_size = self._estimate_optimal_chunk_size(data)
        
        # Create chunks
        chunks = self._create_chunks(data, chunk_size)
        
        if not chunks:
            return None
        
        # Process chunks
        if parallel and len(chunks) > 1:
            results = self._process_chunks_parallel(
                data, chunks, process_func, progress_callback
            )
        else:
            results = self._process_chunks_sequential(
                data, chunks, process_func, progress_callback
            )
        
        # Combine results
        if combine_func:
            logger.info("Combining chunk results...")
            final_result = combine_func(results)
        elif len(results) == 1:
            final_result = results[0]
        elif all(isinstance(r, pd.DataFrame) for r in results):
            # Auto-combine DataFrames
            final_result = pd.concat(results, ignore_index=True)
        else:
            final_result = results
        
        logger.info("Chunked processing completed")
        return final_result
    
    def _process_chunks_sequential(
        self,
        data: pd.DataFrame,
        chunks: List[Tuple[int, int]],
        process_func: Callable[[pd.DataFrame], Any],
        progress_callback: Optional[Callable[[int, int], None]]
    ) -> List[Any]:
        """Process chunks sequentially."""
        results = []
        
        for i, (start, end) in enumerate(chunks):
            try:
                chunk_data = data.iloc[start:end]
                result = process_func(chunk_data)
                
                if self.config.use_temp_files and len(chunks) > 1:
                    # Save to temp file for large operations
                    temp_file = self._save_chunk_result(result, f"seq_{i}")
                    results.append(temp_file)
                else:
                    results.append(result)
                
                if progress_callback:
                    progress_callback(i + 1, len(chunks))
                    
            except Exception as e:
                logger.error(f"Error processing chunk {i}: {e}")
                raise
        
        # Load results from temp files if needed
        if self.config.use_temp_files and len(chunks) > 1:
            results = [self._load_chunk_result(f) for f in results]
        
        return results
    
    def _process_chunks_parallel(
        self,
        data: pd.DataFrame,
        chunks: List[Tuple[int, int]],
        process_func: Callable[[pd.DataFrame], Any],
        progress_callback: Optional[Callable[[int, int], None]]
    ) -> List[Any]:
        """Process chunks in parallel."""
        # Submit all chunk jobs
        job_ids = []
        chunk_jobs = {}
        
        for i, (start, end) in enumerate(chunks):
            chunk_data = data.iloc[start:end]
            job_id = self.resource_manager.submit_job(
                process_func,
                chunk_data,
                job_id=f"chunk_{i}"
            )
            job_ids.append(job_id)
            chunk_jobs[job_id] = i
        
        # Collect results as they complete
        results = [None] * len(chunks)
        completed = 0
        
        while completed < len(chunks):
            for job_id in list(chunk_jobs.keys()):
                try:
                    result = self.resource_manager.get_job_result(job_id, timeout=0.1)
                    chunk_idx = chunk_jobs[job_id]
                    
                    if self.config.use_temp_files:
                        temp_file = self._save_chunk_result(result, f"par_{chunk_idx}")
                        results[chunk_idx] = temp_file
                    else:
                        results[chunk_idx] = result
                    
                    del chunk_jobs[job_id]
                    completed += 1
                    
                    if progress_callback:
                        progress_callback(completed, len(chunks))
                        
                except Exception:
                    # Job not ready yet, continue
                    continue
            
            if chunk_jobs:  # Still have pending jobs
                import time
                time.sleep(0.1)
        
        # Load results from temp files if needed
        if self.config.use_temp_files:
            final_results = []
            for i, temp_file in enumerate(results):
                try:
                    result = self._load_chunk_result(temp_file)
                    final_results.append(result)
                except Exception as e:
                    logger.error(f"Failed to load result for chunk {i}: {e}")
                    raise
            results = final_results
        
        return results
    
    def process_file_chunks(
        self,
        file_path: str,
        process_func: Callable[[pd.DataFrame], Any],
        combine_func: Optional[Callable[[List[Any]], Any]] = None,
        chunk_size: Optional[int] = None,
        read_kwargs: Optional[Dict[str, Any]] = None,
        parallel: bool = True,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> Any:
        """
        Process large files in chunks without loading everything into memory.
        
        Args:
            file_path: Path to input file
            process_func: Function to apply to each chunk
            combine_func: Function to combine results
            chunk_size: Chunk size
            read_kwargs: Additional arguments for pd.read_csv/read_parquet
            parallel: Whether to use parallel processing
            progress_callback: Progress callback
            
        Returns:
            Combined result
        """
        file_path = Path(file_path)
        read_kwargs = read_kwargs or {}
        
        logger.info(f"Processing file: {file_path}")
        
        # Determine file reader
        if file_path.suffix.lower() == '.parquet':
            reader = pd.read_parquet
        elif file_path.suffix.lower() in ['.csv', '.txt']:
            reader = partial(pd.read_csv, **read_kwargs)
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")
        
        # For CSV, we can use chunksize parameter
        if file_path.suffix.lower() in ['.csv', '.txt']:
            return self._process_csv_chunks(
                file_path, process_func, combine_func, chunk_size, read_kwargs, parallel, progress_callback
            )
        else:
            # Load and process normally for other formats
            data = reader(file_path)
            return self.process_dataframe_chunks(
                data, process_func, combine_func, chunk_size, parallel, progress_callback
            )
    
    def _process_csv_chunks(
        self,
        file_path: str,
        process_func: Callable[[pd.DataFrame], Any],
        combine_func: Optional[Callable[[List[Any]], Any]],
        chunk_size: Optional[int],
        read_kwargs: Dict[str, Any],
        parallel: bool,
        progress_callback: Optional[Callable[[int, int], None]]
    ) -> Any:
        """Process CSV file in chunks using pandas chunksize."""
        if chunk_size is None:
            chunk_size = self.config.min_chunk_size
        
        # First pass - count total chunks for progress
        total_chunks = 0
        try:
            for _ in pd.read_csv(file_path, chunksize=chunk_size, **read_kwargs):
                total_chunks += 1
        except Exception as e:
            logger.warning(f"Could not count chunks: {e}")
            total_chunks = None
        
        # Process chunks
        results = []
        chunk_reader = pd.read_csv(file_path, chunksize=chunk_size, **read_kwargs)
        
        if parallel:
            # Collect all chunks first for parallel processing
            chunks_data = []
            for i, chunk in enumerate(chunk_reader):
                chunks_data.append(chunk)
            
            # Process in parallel
            job_ids = []
            for i, chunk in enumerate(chunks_data):
                job_id = self.resource_manager.submit_job(
                    process_func, chunk, job_id=f"file_chunk_{i}"
                )
                job_ids.append(job_id)
            
            # Collect results
            for i, job_id in enumerate(job_ids):
                result = self.resource_manager.get_job_result(job_id)
                results.append(result)
                
                if progress_callback and total_chunks:
                    progress_callback(i + 1, total_chunks)
        else:
            # Sequential processing
            for i, chunk in enumerate(chunk_reader):
                try:
                    result = process_func(chunk)
                    results.append(result)
                    
                    if progress_callback and total_chunks:
                        progress_callback(i + 1, total_chunks)
                        
                except Exception as e:
                    logger.error(f"Error processing chunk {i}: {e}")
                    raise
        
        # Combine results
        if combine_func:
            return combine_func(results)
        elif len(results) == 1:
            return results[0]
        elif all(isinstance(r, pd.DataFrame) for r in results):
            return pd.concat(results, ignore_index=True)
        else:
            return results
    
    def apply_rolling_operation(
        self,
        data: pd.DataFrame,
        operation_func: Callable[[pd.DataFrame], pd.DataFrame],
        window_size: int,
        step_size: Optional[int] = None,
        parallel: bool = True
    ) -> pd.DataFrame:
        """
        Apply rolling window operation with proper overlap handling.
        
        Args:
            data: Input DataFrame
            operation_func: Function to apply to each window
            window_size: Size of rolling window
            step_size: Step size (default: window_size // 2)
            parallel: Whether to use parallel processing
            
        Returns:
            Result DataFrame
        """
        if step_size is None:
            step_size = max(1, window_size // 2)
        
        # Set overlap to ensure proper window coverage
        original_overlap = self.config.overlap_rows
        self.config.overlap_rows = window_size - step_size
        
        try:
            # Create rolling windows
            windows = []
            for start in range(0, len(data), step_size):
                end = min(start + window_size, len(data))
                if end - start >= window_size // 2:  # Minimum window size
                    windows.append((start, end))
            
            # Process windows
            def process_window(window_data):
                if len(window_data) < window_size // 2:
                    return pd.DataFrame()  # Skip too small windows
                return operation_func(window_data)
            
            results = []
            if parallel and len(windows) > 1:
                # Parallel processing
                job_ids = []
                for i, (start, end) in enumerate(windows):
                    window_data = data.iloc[start:end]
                    job_id = self.resource_manager.submit_job(
                        process_window, window_data, job_id=f"window_{i}"
                    )
                    job_ids.append(job_id)
                
                for job_id in job_ids:
                    result = self.resource_manager.get_job_result(job_id)
                    if not result.empty:
                        results.append(result)
            else:
                # Sequential processing
                for start, end in windows:
                    window_data = data.iloc[start:end]
                    result = process_window(window_data)
                    if not result.empty:
                        results.append(result)
            
            # Combine results
            if results:
                return pd.concat(results, ignore_index=True)
            else:
                return pd.DataFrame()
                
        finally:
            # Restore original overlap setting
            self.config.overlap_rows = original_overlap
    
    def cleanup(self):
        """Clean up temporary files."""
        for temp_file in self.temp_files:
            try:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            except Exception as e:
                logger.warning(f"Failed to remove temp file {temp_file}: {e}")
        
        if self.temp_dir and os.path.exists(self.temp_dir):
            try:
                os.rmdir(self.temp_dir)
            except Exception as e:
                logger.warning(f"Failed to remove temp directory {self.temp_dir}: {e}")
        
        self.temp_files.clear()
        self.temp_dir = None


# Convenience functions
def process_large_dataframe(
    data: pd.DataFrame,
    process_func: Callable[[pd.DataFrame], Any],
    combine_func: Optional[Callable[[List[Any]], Any]] = None,
    memory_limit_mb: float = 1000,
    parallel: bool = True
) -> Any:
    """
    Convenience function for processing large DataFrames.
    
    Args:
        data: Input DataFrame
        process_func: Function to apply to chunks
        combine_func: Function to combine results
        memory_limit_mb: Memory limit in MB
        parallel: Whether to use parallel processing
        
    Returns:
        Combined result
    """
    config = ChunkConfig(memory_limit_mb=memory_limit_mb)
    
    with ChunkedProcessor(config=config) as processor:
        return processor.process_dataframe_chunks(
            data, process_func, combine_func, parallel=parallel
        )