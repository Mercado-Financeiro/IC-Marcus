"""
Centralized logging configuration for the entire project.

This module provides a consistent logging setup using structlog,
with proper formatting, context, and level management.
"""

import logging
import sys
from pathlib import Path
from typing import Optional, Dict, Any
import structlog
from structlog.processors import TimeStamper, add_log_level, format_exc_info
from structlog.dev import ConsoleRenderer
from structlog.processors import JSONRenderer
import warnings


class LogConfig:
    """Centralized logging configuration."""
    
    # Log levels
    DEBUG = logging.DEBUG
    INFO = logging.INFO
    WARNING = logging.WARNING
    ERROR = logging.ERROR
    CRITICAL = logging.CRITICAL
    
    def __init__(
        self,
        level: int = logging.INFO,
        format: str = "console",  # "console" or "json"
        log_file: Optional[Path] = None,
        add_context: bool = True
    ):
        """
        Initialize logging configuration.
        
        Args:
            level: Logging level
            format: Output format ("console" or "json")
            log_file: Optional file to write logs to
            add_context: Whether to add execution context
        """
        self.level = level
        self.format = format
        self.log_file = log_file
        self.add_context = add_context
        self._configured = False
        
    def configure(self):
        """Configure structlog with specified settings."""
        if self._configured:
            return
            
        # Configure Python's logging
        logging.basicConfig(
            format="%(message)s",
            stream=sys.stdout,
            level=self.level,
        )
        
        # Processors for structlog
        processors = [
            TimeStamper(fmt="iso", utc=False),
            add_log_level,
            self._add_context_processor if self.add_context else None,
            format_exc_info,
        ]
        
        # Remove None processors
        processors = [p for p in processors if p is not None]
        
        # Add renderer based on format
        if self.format == "json":
            processors.append(JSONRenderer())
        else:
            processors.append(ConsoleRenderer())
        
        # Configure structlog with version-agnostic logger factory
        try:
            # Prefer stdlib factory when available (works across versions)
            from structlog.stdlib import LoggerFactory as _LoggerFactory  # type: ignore
            logger_factory = _LoggerFactory()
        except Exception:
            # Fallbacks for older/newer structlog distributions
            logger_factory = getattr(structlog, 'StandardLoggerFactory', None)
            if callable(logger_factory):
                logger_factory = logger_factory()
            else:
                # Last resort: simple print logger
                from structlog import PrintLoggerFactory
                logger_factory = PrintLoggerFactory()

        structlog.configure(
            processors=processors,
            context_class=dict,
            logger_factory=logger_factory,
            cache_logger_on_first_use=True,
        )
        
        self._configured = True
        
        # Also configure warnings to use logging
        logging.captureWarnings(True)
        
    def _add_context_processor(self, logger, method_name, event_dict):
        """Add execution context to log entries."""
        import inspect
        
        # Get caller information
        frame = inspect.currentframe()
        if frame and frame.f_back and frame.f_back.f_back:
            caller_frame = frame.f_back.f_back
            event_dict['module'] = caller_frame.f_globals.get('__name__', 'unknown')
            event_dict['function'] = caller_frame.f_code.co_name
            event_dict['line'] = caller_frame.f_lineno
            
        return event_dict
    
    @staticmethod
    def get_logger(name: Optional[str] = None) -> structlog.BoundLogger:
        """
        Get a configured logger instance.
        
        Args:
            name: Logger name (usually __name__)
            
        Returns:
            Configured structlog logger
        """
        return structlog.get_logger(name)


# Global configuration instance
_global_config = LogConfig()


def configure_logging(
    level: int = logging.INFO,
    format: str = "console",
    log_file: Optional[Path] = None,
    add_context: bool = True
):
    """
    Configure global logging settings.
    
    This should be called once at application startup.
    
    Args:
        level: Logging level
        format: Output format ("console" or "json")
        log_file: Optional file to write logs to
        add_context: Whether to add execution context
    """
    global _global_config
    _global_config = LogConfig(level, format, log_file, add_context)
    _global_config.configure()


def get_logger(name: Optional[str] = None) -> structlog.BoundLogger:
    """
    Get a logger instance.
    
    Args:
        name: Logger name (usually __name__)
        
    Returns:
        Configured structlog logger
        
    Example:
        >>> log = get_logger(__name__)
        >>> log.info("process_started", symbol="BTCUSDT", timeframe="15m")
        >>> log.warning("low_sample_size", n_samples=100, required=1000)
        >>> log.error("division_by_zero", numerator=10, denominator=0)
    """
    if not _global_config._configured:
        _global_config.configure()
    return _global_config.get_logger(name)


def log_function_call(func):
    """
    Decorator to log function calls with arguments and results.
    
    Example:
        @log_function_call
        def calculate_metrics(data):
            return {"accuracy": 0.95}
    """
    import functools
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        log = get_logger(func.__module__)
        
        # Log function entry
        log.debug(
            "function_called",
            function=func.__name__,
            args_count=len(args),
            kwargs_keys=list(kwargs.keys())
        )
        
        try:
            result = func(*args, **kwargs)
            
            # Log successful execution
            log.debug(
                "function_completed",
                function=func.__name__,
                has_result=result is not None
            )
            
            return result
            
        except Exception as e:
            # Log error
            log.error(
                "function_failed",
                function=func.__name__,
                error=str(e),
                error_type=type(e).__name__
            )
            raise
    
    return wrapper


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """
    Safely divide two numbers with logging of edge cases.
    
    Args:
        numerator: The numerator
        denominator: The denominator
        default: Default value if division is undefined
        
    Returns:
        Result of division or default value
        
    Example:
        >>> ratio = safe_divide(10, 0, default=0)  # Returns 0, logs warning
        >>> ratio = safe_divide(10, 2)  # Returns 5.0
    """
    log = get_logger("utils.math")
    
    if denominator == 0:
        log.warning(
            "division_by_zero",
            numerator=numerator,
            denominator=denominator,
            default_used=default
        )
        return default
    
    if abs(denominator) < 1e-10:
        log.warning(
            "near_zero_denominator",
            numerator=numerator,
            denominator=denominator,
            default_used=default
        )
        return default
        
    result = numerator / denominator
    
    if not np.isfinite(result):
        log.warning(
            "non_finite_result",
            numerator=numerator,
            denominator=denominator,
            result=result,
            default_used=default
        )
        return default
    
    return result


def validate_dataframe(df: 'pd.DataFrame', name: str = "dataframe") -> None:
    """
    Validate a DataFrame and log any issues found.
    
    Args:
        df: DataFrame to validate
        name: Name for logging context
        
    Raises:
        ValueError: If DataFrame has critical issues
        
    Example:
        >>> validate_dataframe(df, "features")
    """
    import pandas as pd
    import numpy as np
    
    log = get_logger("utils.validation")
    
    # Check if empty
    if df.empty:
        log.error("empty_dataframe", name=name)
        raise ValueError(f"DataFrame '{name}' is empty")
    
    # Check for all-null columns
    null_cols = df.columns[df.isnull().all()].tolist()
    if null_cols:
        log.warning(
            "all_null_columns",
            name=name,
            columns=null_cols,
            action="dropping"
        )
        df.drop(columns=null_cols, inplace=True)
    
    # Check for high null percentage
    null_pct = df.isnull().sum() / len(df)
    high_null = null_pct[null_pct > 0.5]
    if not high_null.empty:
        log.warning(
            "high_null_percentage",
            name=name,
            columns=high_null.to_dict()
        )
    
    # Check for infinite values
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if np.isinf(df[col]).any():
            inf_count = np.isinf(df[col]).sum()
            log.warning(
                "infinite_values",
                name=name,
                column=col,
                count=inf_count
            )
            # Replace with NaN
            df[col].replace([np.inf, -np.inf], np.nan, inplace=True)
    
    log.debug(
        "dataframe_validated",
        name=name,
        shape=df.shape,
        dtypes=df.dtypes.value_counts().to_dict()
    )


# Convenience imports
import numpy as np

__all__ = [
    'configure_logging',
    'get_logger',
    'log_function_call',
    'safe_divide',
    'validate_dataframe',
    'LogConfig'
]
