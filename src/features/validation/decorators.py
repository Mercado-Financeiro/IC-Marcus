"""Decorators for automatic input/output validation."""

import functools
import pandas as pd
import numpy as np
from typing import List, Optional, Dict, Any, Callable, Union
import structlog

from .validator import FeatureValidator
from .exceptions import ValidationError

log = structlog.get_logger()


def validate_inputs(
    required_cols: Optional[List[str]] = None,
    optional_cols: Optional[List[str]] = None,
    dependent_cols: Optional[List[str]] = None,
    min_rows: int = 1,
    validate_ohlcv: bool = False,
    validate_numeric: bool = True,
    allow_nan: bool = True,
    validate_consistency: bool = False,
    strict_consistency: bool = False
) -> Callable:
    """
    Decorator for automatic DataFrame input validation.
    
    Args:
        required_cols: Columns that must exist
        optional_cols: Columns that may exist (just for documentation)
        dependent_cols: Columns that should exist if they were created by prerequisites
        min_rows: Minimum number of rows required
        validate_ohlcv: Whether to validate basic OHLCV structure
        validate_numeric: Whether to validate numeric types
        allow_nan: Whether to allow NaN values in numeric columns
        validate_consistency: Whether to validate OHLC consistency
        strict_consistency: Whether to raise errors on inconsistencies (vs warnings)
        
    Returns:
        Decorated function with automatic validation
        
    Example:
        @validate_inputs(['volume', 'close'], min_rows=20)
        def calculate_volume_ratios(self, df):
            # Function body - validation happens automatically
            return df
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> pd.DataFrame:
            # Extract DataFrame (assume it's first argument after self)
            if len(args) < 2:
                raise ValueError(f"{func.__name__} requires at least 2 arguments (self, df)")
            
            df = args[1]
            if not isinstance(df, pd.DataFrame):
                raise ValueError(f"{func.__name__} second argument must be DataFrame, got {type(df)}")
            
            function_name = f"{args[0].__class__.__name__}.{func.__name__}"
            
            try:
                # Basic validations
                if min_rows > 1:
                    FeatureValidator.validate_sufficient_data(df, min_rows, function_name)
                
                # OHLCV validation
                if validate_ohlcv:
                    FeatureValidator.validate_ohlcv_basic(df, function_name)
                    
                    if validate_consistency:
                        FeatureValidator.validate_ohlc_consistency(df, strict_consistency)
                    
                    if 'volume' in df.columns:
                        FeatureValidator.validate_volume_data(df, strict_consistency)
                
                # Required columns validation
                if required_cols:
                    FeatureValidator.validate_columns_exist(df, required_cols, function_name)
                
                # Dependent columns validation (warn if missing)
                if dependent_cols:
                    missing_deps = [col for col in dependent_cols if col not in df.columns]
                    if missing_deps:
                        log.warning(
                            "dependent_columns_missing",
                            missing=missing_deps,
                            function=function_name,
                            suggestion="Run prerequisite feature calculations first"
                        )
                
                # Numeric validation
                if validate_numeric:
                    cols_to_check = []
                    if required_cols:
                        cols_to_check.extend([col for col in required_cols if col in df.columns])
                    if dependent_cols:
                        cols_to_check.extend([col for col in dependent_cols if col in df.columns])
                    
                    if cols_to_check:
                        FeatureValidator.validate_numeric_columns(df, cols_to_check, allow_nan)
                
                # Special validations for returns data
                if required_cols and 'returns' in required_cols and 'returns' in df.columns:
                    FeatureValidator.validate_returns_data(df, 'returns')
                
                log.debug(
                    "input_validation_passed",
                    function=function_name,
                    shape=df.shape,
                    required_cols=required_cols or [],
                    min_rows=min_rows
                )
                
            except ValidationError as e:
                log.error(
                    "input_validation_failed",
                    function=function_name,
                    error=str(e),
                    error_type=type(e).__name__
                )
                raise
            
            # Call original function
            result = func(*args, **kwargs)
            
            return result
        
        # Add metadata to decorated function
        wrapper._validation_info = {
            'required_cols': required_cols,
            'optional_cols': optional_cols,
            'dependent_cols': dependent_cols,
            'min_rows': min_rows,
            'validate_ohlcv': validate_ohlcv,
            'validate_numeric': validate_numeric
        }
        
        return wrapper
    
    return decorator


def validate_outputs(
    expected_new_cols: Optional[List[str]] = None,
    allow_row_reduction: bool = True,
    check_infinite: bool = True,
    check_all_nan: bool = True,
    max_nan_ratio: Optional[float] = None
) -> Callable:
    """
    Decorator for automatic DataFrame output validation.
    
    Args:
        expected_new_cols: Expected new column names to be created
        allow_row_reduction: Whether to allow fewer rows in output
        check_infinite: Whether to check for infinite values
        check_all_nan: Whether to check for all-NaN columns
        max_nan_ratio: Maximum allowed ratio of NaN values in new columns
        
    Returns:
        Decorated function with automatic output validation
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> pd.DataFrame:
            # Get input DataFrame
            input_df = args[1] if len(args) >= 2 else None
            function_name = f"{args[0].__class__.__name__}.{func.__name__}"
            
            # Call original function
            result = func(*args, **kwargs)
            
            if not isinstance(result, pd.DataFrame):
                log.warning(
                    "non_dataframe_output",
                    function=function_name,
                    output_type=type(result)
                )
                return result
            
            try:
                # Basic output validation
                if input_df is not None:
                    FeatureValidator.validate_feature_output(
                        input_df, result, expected_new_cols, allow_row_reduction
                    )
                
                # Additional output checks
                if input_df is not None:
                    new_cols = set(result.columns) - set(input_df.columns)
                    
                    for col in new_cols:
                        col_data = result[col]
                        
                        # Check for infinite values
                        if check_infinite and np.isinf(col_data).any():
                            inf_count = np.isinf(col_data).sum()
                            log.warning(
                                "infinite_values_in_output",
                                function=function_name,
                                column=col,
                                count=inf_count,
                                total=len(col_data)
                            )
                        
                        # Check for all-NaN columns
                        if check_all_nan and col_data.isna().all():
                            log.warning(
                                "all_nan_output_column",
                                function=function_name,
                                column=col
                            )
                        
                        # Check NaN ratio
                        if max_nan_ratio is not None:
                            nan_ratio = col_data.isna().sum() / len(col_data)
                            if nan_ratio > max_nan_ratio:
                                log.warning(
                                    "high_nan_ratio_output",
                                    function=function_name,
                                    column=col,
                                    nan_ratio=nan_ratio,
                                    threshold=max_nan_ratio
                                )
                
                log.debug(
                    "output_validation_passed",
                    function=function_name,
                    input_shape=input_df.shape if input_df is not None else None,
                    output_shape=result.shape,
                    new_columns=len(set(result.columns) - set(input_df.columns)) if input_df is not None else 0
                )
                
            except ValidationError as e:
                log.error(
                    "output_validation_failed",
                    function=function_name,
                    error=str(e),
                    error_type=type(e).__name__
                )
                # Don't raise - output validation is usually informational
            
            return result
        
        return wrapper
    
    return decorator


def validate_feature_method(
    required_cols: Optional[List[str]] = None,
    min_rows: int = 1,
    expected_new_cols: Optional[List[str]] = None,
    validate_ohlcv: bool = False,
    **kwargs
) -> Callable:
    """
    Convenience decorator combining input and output validation.
    
    Args:
        required_cols: Required input columns
        min_rows: Minimum number of input rows
        expected_new_cols: Expected new columns in output
        validate_ohlcv: Whether to validate OHLCV structure
        **kwargs: Additional arguments for validate_inputs
        
    Returns:
        Decorated function with both input and output validation
    """
    def decorator(func: Callable) -> Callable:
        # Apply both decorators
        func = validate_inputs(
            required_cols=required_cols,
            min_rows=min_rows,
            validate_ohlcv=validate_ohlcv,
            **kwargs
        )(func)
        
        func = validate_outputs(
            expected_new_cols=expected_new_cols
        )(func)
        
        return func
    
    return decorator


def log_execution_time(func: Callable) -> Callable:
    """
    Decorator to log function execution time.
    
    Useful for performance monitoring of feature calculations.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        import time
        
        function_name = f"{args[0].__class__.__name__}.{func.__name__}"
        start_time = time.time()
        
        try:
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            
            log.debug(
                "function_executed",
                function=function_name,
                execution_time_seconds=round(execution_time, 3),
                output_shape=result.shape if hasattr(result, 'shape') else None
            )
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            log.error(
                "function_failed",
                function=function_name,
                execution_time_seconds=round(execution_time, 3),
                error=str(e),
                error_type=type(e).__name__
            )
            raise
    
    return wrapper