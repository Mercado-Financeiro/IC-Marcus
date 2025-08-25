"""Central feature validator with comprehensive data checks."""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional, Union
import structlog

from .exceptions import (
    ColumnMissingError, DataInconsistencyError, InvalidDataTypeError,
    InsufficientDataError, InvalidRangeError
)

log = structlog.get_logger()


class FeatureValidator:
    """
    Central validator for feature engineering inputs and outputs.
    
    Provides comprehensive validation methods for DataFrame structure,
    data consistency, and feature calculation prerequisites.
    """
    
    @staticmethod
    def validate_columns_exist(
        df: pd.DataFrame, 
        required_cols: List[str],
        function_name: Optional[str] = None
    ) -> None:
        """
        Validate that required columns exist in DataFrame.
        
        Args:
            df: DataFrame to validate
            required_cols: List of required column names
            function_name: Name of calling function for error context
            
        Raises:
            ColumnMissingError: If any required columns are missing
        """
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            log.error(
                "missing_columns",
                missing=missing_cols,
                available=list(df.columns),
                function=function_name
            )
            raise ColumnMissingError(missing_cols, function_name)
    
    @staticmethod
    def validate_numeric_columns(
        df: pd.DataFrame,
        columns: List[str],
        allow_nan: bool = True
    ) -> None:
        """
        Validate that columns contain numeric data.
        
        Args:
            df: DataFrame to validate
            columns: List of columns to check
            allow_nan: Whether to allow NaN values
            
        Raises:
            InvalidDataTypeError: If columns are not numeric
        """
        for col in columns:
            if col not in df.columns:
                continue
                
            # Check if column is numeric
            if not pd.api.types.is_numeric_dtype(df[col]):
                actual_type = str(df[col].dtype)
                log.error(
                    "invalid_data_type",
                    column=col,
                    expected="numeric",
                    actual=actual_type
                )
                raise InvalidDataTypeError(col, "numeric", actual_type)
            
            # Check for infinite values
            if np.isinf(df[col]).any():
                inf_count = np.isinf(df[col]).sum()
                log.warning(
                    "infinite_values",
                    column=col,
                    count=inf_count,
                    total=len(df)
                )
            
            # Check for NaN if not allowed
            if not allow_nan and df[col].isna().any():
                nan_count = df[col].isna().sum()
                log.warning(
                    "nan_values",
                    column=col,
                    count=nan_count,
                    total=len(df)
                )
    
    @staticmethod
    def validate_ohlcv_basic(df: pd.DataFrame, function_name: Optional[str] = None) -> None:
        """
        Validate basic OHLCV structure and presence.
        
        Args:
            df: DataFrame to validate
            function_name: Name of calling function for error context
            
        Raises:
            ColumnMissingError: If OHLCV columns are missing
            InvalidDataTypeError: If columns are not numeric
        """
        required_cols = ["open", "high", "low", "close", "volume"]
        
        # Check column existence
        FeatureValidator.validate_columns_exist(df, required_cols, function_name)
        
        # Check numeric types
        FeatureValidator.validate_numeric_columns(df, required_cols, allow_nan=False)
    
    @staticmethod
    def validate_ohlc_consistency(df: pd.DataFrame, strict: bool = False) -> None:
        """
        Validate OHLC price consistency.
        
        Args:
            df: DataFrame with OHLC data
            strict: If True, raise error on inconsistencies. If False, log warnings.
            
        Raises:
            DataInconsistencyError: If strict=True and inconsistencies found
        """
        FeatureValidator.validate_columns_exist(df, ["open", "high", "low", "close"])
        
        inconsistencies = []
        
        # Check high >= max(open, close)
        high_invalid = df["high"] < df[["open", "close"]].max(axis=1)
        if high_invalid.any():
            count = high_invalid.sum()
            inconsistencies.append(f"high < max(open, close): {count} rows")
        
        # Check low <= min(open, close)  
        low_invalid = df["low"] > df[["open", "close"]].min(axis=1)
        if low_invalid.any():
            count = low_invalid.sum()
            inconsistencies.append(f"low > min(open, close): {count} rows")
        
        # Check positive prices
        negative_prices = (df[["open", "high", "low", "close"]] <= 0).any(axis=1)
        if negative_prices.any():
            count = negative_prices.sum()
            inconsistencies.append(f"non-positive prices: {count} rows")
        
        if inconsistencies:
            message = "OHLC inconsistencies detected: " + "; ".join(inconsistencies)
            
            log.warning(
                "ohlc_inconsistency",
                issues=inconsistencies,
                total_rows=len(df)
            )
            
            if strict:
                invalid_rows = (high_invalid | low_invalid | negative_prices).sum()
                raise DataInconsistencyError(message, invalid_rows)
    
    @staticmethod
    def validate_volume_data(df: pd.DataFrame, strict: bool = False) -> None:
        """
        Validate volume data consistency.
        
        Args:
            df: DataFrame with volume data
            strict: If True, raise error on issues. If False, log warnings.
            
        Raises:
            DataInconsistencyError: If strict=True and issues found
        """
        FeatureValidator.validate_columns_exist(df, ["volume"])
        FeatureValidator.validate_numeric_columns(df, ["volume"])
        
        issues = []
        
        # Check non-negative volume
        negative_vol = df["volume"] < 0
        if negative_vol.any():
            count = negative_vol.sum()
            issues.append(f"negative volume: {count} rows")
        
        # Check for zero volume (warning only)
        zero_vol = df["volume"] == 0
        if zero_vol.any():
            count = zero_vol.sum()
            log.info("zero_volume_detected", count=count, total=len(df))
        
        # Check for extreme outliers (> 10x median)
        if len(df) > 10:
            median_vol = df["volume"].median()
            if median_vol > 0:
                extreme_vol = df["volume"] > (10 * median_vol)
                if extreme_vol.any():
                    count = extreme_vol.sum()
                    log.warning("extreme_volume_outliers", count=count, threshold=f"10x median ({median_vol:.0f})")
        
        if issues and strict:
            message = "Volume data issues: " + "; ".join(issues)
            invalid_rows = negative_vol.sum()
            raise DataInconsistencyError(message, invalid_rows)
    
    @staticmethod
    def validate_sufficient_data(
        df: pd.DataFrame,
        min_rows: int,
        function_name: Optional[str] = None
    ) -> None:
        """
        Validate DataFrame has sufficient rows for calculation.
        
        Args:
            df: DataFrame to validate
            min_rows: Minimum required rows
            function_name: Name of calling function for error context
            
        Raises:
            InsufficientDataError: If insufficient rows
        """
        actual_rows = len(df)
        
        if actual_rows < min_rows:
            log.error(
                "insufficient_data",
                required=min_rows,
                actual=actual_rows,
                function=function_name
            )
            raise InsufficientDataError(min_rows, actual_rows, function_name)
    
    @staticmethod
    def validate_lookback_periods(
        periods: List[int],
        df_length: int,
        function_name: Optional[str] = None
    ) -> List[int]:
        """
        Validate and filter lookback periods based on data length.
        
        Args:
            periods: List of lookback periods
            df_length: Length of DataFrame
            function_name: Name of calling function for context
            
        Returns:
            List of valid periods (filtered)
        """
        valid_periods = [p for p in periods if p <= df_length]
        invalid_periods = [p for p in periods if p > df_length]
        
        if invalid_periods:
            log.warning(
                "invalid_periods_filtered",
                invalid=invalid_periods,
                valid=valid_periods,
                df_length=df_length,
                function=function_name
            )
        
        return valid_periods
    
    @staticmethod
    def validate_returns_data(df: pd.DataFrame, column: str = "returns") -> None:
        """
        Validate returns data for reasonableness.
        
        Args:
            df: DataFrame with returns data
            column: Name of returns column
            
        Raises:
            DataInconsistencyError: If returns data is unreasonable
        """
        FeatureValidator.validate_columns_exist(df, [column])
        FeatureValidator.validate_numeric_columns(df, [column])
        
        returns = df[column].dropna()
        
        if len(returns) == 0:
            log.warning("empty_returns_data", column=column)
            return
        
        # Check for extreme returns (>50% in single period - likely error)
        extreme_threshold = 0.5  # 50%
        extreme_returns = np.abs(returns) > extreme_threshold
        
        if extreme_returns.any():
            count = extreme_returns.sum()
            max_return = np.abs(returns).max()
            
            log.warning(
                "extreme_returns_detected",
                column=column,
                count=count,
                max_absolute_return=max_return,
                threshold=extreme_threshold
            )
            
            # If more than 1% of returns are extreme, flag as suspicious
            if count / len(returns) > 0.01:
                raise DataInconsistencyError(
                    f"Excessive extreme returns in {column}: {count}/{len(returns)} "
                    f"returns exceed {extreme_threshold*100}%"
                )
    
    @staticmethod
    def validate_feature_output(
        input_df: pd.DataFrame,
        output_df: pd.DataFrame,
        expected_new_cols: Optional[List[str]] = None,
        allow_row_reduction: bool = True
    ) -> None:
        """
        Validate feature calculation output.
        
        Args:
            input_df: Original DataFrame
            output_df: DataFrame after feature calculation
            expected_new_cols: Expected new column names
            allow_row_reduction: Whether to allow fewer rows in output
            
        Raises:
            DataInconsistencyError: If output validation fails
        """
        # Check that output has same or fewer rows
        if not allow_row_reduction and len(output_df) != len(input_df):
            raise DataInconsistencyError(
                f"Output has {len(output_df)} rows, expected {len(input_df)}"
            )
        
        if allow_row_reduction and len(output_df) > len(input_df):
            raise DataInconsistencyError(
                f"Output has more rows ({len(output_df)}) than input ({len(input_df)})"
            )
        
        # Check that original columns are preserved
        original_cols = set(input_df.columns)
        output_cols = set(output_df.columns)
        
        if not original_cols.issubset(output_cols):
            missing = original_cols - output_cols
            raise DataInconsistencyError(f"Original columns missing in output: {missing}")
        
        # Check expected new columns if specified
        if expected_new_cols:
            new_cols = output_cols - original_cols
            missing_new = set(expected_new_cols) - new_cols
            
            if missing_new:
                log.warning(
                    "expected_features_missing",
                    missing=list(missing_new),
                    created=list(new_cols)
                )
        
        # Check for infinite or all-NaN new columns
        new_cols = output_cols - original_cols
        for col in new_cols:
            if col in output_df.columns:
                col_data = output_df[col]
                
                # Check for all infinite
                if np.isinf(col_data).all():
                    log.error("all_infinite_feature", column=col)
                
                # Check for all NaN
                if col_data.isna().all():
                    log.warning("all_nan_feature", column=col)