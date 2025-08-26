"""
Great Expectations validation for crypto data quality.
Aligned with PRD section 2 - Data Quality requirements.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import great_expectations as gx
from great_expectations.core.batch import BatchRequest
from great_expectations.checkpoint import Checkpoint
import structlog

log = structlog.get_logger()


class CryptoDataValidator:
    """Validator for OHLCV crypto data following PRD specifications."""
    
    def __init__(self, data_dir: str = "data/raw"):
        """Initialize Great Expectations context."""
        self.data_dir = Path(data_dir)
        self.context = gx.get_context()
        self.suite_name = "crypto_ohlcv_suite"
        
    def create_expectations_suite(self) -> None:
        """Create comprehensive expectations suite for OHLCV data."""
        
        # Create or update suite
        suite = self.context.add_or_update_expectation_suite(
            expectation_suite_name=self.suite_name
        )
        
        # Required columns (PRD Section 5)
        suite.add_expectation(
            gx.expectations.expect_table_columns_to_match_ordered_list(
                column_list=["timestamp", "open", "high", "low", "close", "volume", "trades"]
            )
        )
        
        # No null values in critical columns
        for col in ["timestamp", "open", "high", "low", "close", "volume"]:
            suite.add_expectation(
                gx.expectations.expect_column_values_to_not_be_null(column=col)
            )
        
        # Price sanity checks
        suite.add_expectation(
            gx.expectations.expect_column_values_to_be_between(
                column="open", min_value=0, strict_min=True
            )
        )
        suite.add_expectation(
            gx.expectations.expect_column_values_to_be_between(
                column="high", min_value=0, strict_min=True
            )
        )
        suite.add_expectation(
            gx.expectations.expect_column_values_to_be_between(
                column="low", min_value=0, strict_min=True
            )
        )
        suite.add_expectation(
            gx.expectations.expect_column_values_to_be_between(
                column="close", min_value=0, strict_min=True
            )
        )
        
        # Volume must be non-negative
        suite.add_expectation(
            gx.expectations.expect_column_values_to_be_between(
                column="volume", min_value=0
            )
        )
        
        # High >= Low check
        suite.add_expectation(
            gx.expectations.expect_column_pair_values_A_to_be_greater_than_B(
                column_A="high", column_B="low", or_equal=True
            )
        )
        
        # High >= Open and Close
        suite.add_expectation(
            gx.expectations.expect_column_pair_values_A_to_be_greater_than_B(
                column_A="high", column_B="open", or_equal=True
            )
        )
        suite.add_expectation(
            gx.expectations.expect_column_pair_values_A_to_be_greater_than_B(
                column_A="high", column_B="close", or_equal=True
            )
        )
        
        # Low <= Open and Close
        suite.add_expectation(
            gx.expectations.expect_column_pair_values_A_to_be_less_than_B(
                column_A="low", column_B="open", or_equal=True
            )
        )
        suite.add_expectation(
            gx.expectations.expect_column_pair_values_A_to_be_less_than_B(
                column_A="low", column_B="close", or_equal=True
            )
        )
        
        # Timestamp checks
        suite.add_expectation(
            gx.expectations.expect_column_values_to_be_unique(column="timestamp")
        )
        suite.add_expectation(
            gx.expectations.expect_column_values_to_be_increasing(
                column="timestamp", strictly=True
            )
        )
        
        self.context.save_expectation_suite(suite)
        log.info("Created expectations suite", suite=self.suite_name)
        
    def check_gaps_and_duplicates(self, df: pd.DataFrame, timeframe: str) -> Dict:
        """Check for gaps and duplicates in timeseries."""
        
        # Convert timeframe to timedelta
        timeframe_map = {
            "1m": pd.Timedelta(minutes=1),
            "5m": pd.Timedelta(minutes=5),
            "15m": pd.Timedelta(minutes=15),
            "1h": pd.Timedelta(hours=1),
            "4h": pd.Timedelta(hours=4),
        }
        
        expected_diff = timeframe_map.get(timeframe)
        if not expected_diff:
            raise ValueError(f"Unknown timeframe: {timeframe}")
        
        # Check for duplicates
        duplicates = df.index.duplicated().sum()
        
        # Check for gaps
        time_diffs = pd.Series(df.index).diff()
        gaps = (time_diffs > expected_diff).sum()
        
        # Check monotonicicity
        is_monotonic = df.index.is_monotonic_increasing
        
        results = {
            "duplicates": int(duplicates),
            "gaps": int(gaps),
            "is_monotonic": bool(is_monotonic),
            "total_rows": len(df),
            "date_range": {
                "start": str(df.index.min()),
                "end": str(df.index.max())
            }
        }
        
        # Log warnings
        if duplicates > 0:
            log.warning(f"Found {duplicates} duplicate timestamps")
        if gaps > 0:
            log.warning(f"Found {gaps} gaps in timeseries")
        if not is_monotonic:
            log.error("Timestamps are not monotonically increasing!")
            
        return results
    
    def validate_dataframe(
        self, 
        df: pd.DataFrame, 
        symbol: str, 
        timeframe: str
    ) -> Tuple[bool, Dict]:
        """Validate a dataframe against expectations."""
        
        # Basic structure checks
        quality_report = {
            "symbol": symbol,
            "timeframe": timeframe,
            "structural_checks": self.check_gaps_and_duplicates(df, timeframe),
        }
        
        # Prepare dataframe for GE
        df_validation = df.reset_index()
        df_validation.columns = ["timestamp", "open", "high", "low", "close", "volume", "trades"]
        
        # Create batch
        batch_request = BatchRequest(
            datasource_name="pandas_datasource",
            data_asset_name=f"{symbol}_{timeframe}",
            runtime_parameters={"batch_data": df_validation},
            batch_identifiers={"symbol": symbol, "timeframe": timeframe}
        )
        
        # Run validation
        try:
            validator = self.context.get_validator(
                batch_request=batch_request,
                expectation_suite_name=self.suite_name
            )
            results = validator.validate()
            
            quality_report["validation_results"] = {
                "success": results.success,
                "failed_expectations": len(results.results) - sum(
                    1 for r in results.results if r.success
                ),
                "total_expectations": len(results.results)
            }
            
            # Extract specific failures
            if not results.success:
                failures = []
                for result in results.results:
                    if not result.success:
                        failures.append({
                            "expectation": result.expectation_config.expectation_type,
                            "kwargs": result.expectation_config.kwargs
                        })
                quality_report["failures"] = failures
                
        except Exception as e:
            log.error(f"Validation failed: {e}")
            quality_report["validation_results"] = {
                "success": False,
                "error": str(e)
            }
            
        # Overall success
        is_valid = (
            quality_report.get("validation_results", {}).get("success", False) and
            quality_report["structural_checks"]["duplicates"] == 0 and
            quality_report["structural_checks"]["is_monotonic"]
        )
        
        return is_valid, quality_report
    
    def generate_data_docs(self) -> str:
        """Generate and return path to data docs."""
        self.context.build_data_docs()
        docs_sites = self.context.list_data_docs_sites()
        if docs_sites:
            return docs_sites[0]["site_url"]
        return ""


def validate_crypto_data(
    df: pd.DataFrame,
    symbol: str = "BTCUSDT",
    timeframe: str = "15m"
) -> Tuple[bool, Dict]:
    """Quick validation function for notebooks."""
    
    validator = CryptoDataValidator()
    validator.create_expectations_suite()
    is_valid, report = validator.validate_dataframe(df, symbol, timeframe)
    
    if is_valid:
        log.info(
            "Data validation passed",
            symbol=symbol,
            timeframe=timeframe,
            rows=len(df)
        )
    else:
        log.error(
            "Data validation failed",
            symbol=symbol,
            timeframe=timeframe,
            report=report
        )
    
    return is_valid, report


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python ge_checks.py <parquet_file>")
        sys.exit(1)
        
    file_path = Path(sys.argv[1])
    if not file_path.exists():
        print(f"File not found: {file_path}")
        sys.exit(1)
        
    # Load data
    df = pd.read_parquet(file_path)
    
    # Extract symbol and timeframe from filename
    # Expected format: BTCUSDT_15m_2024.parquet
    parts = file_path.stem.split("_")
    symbol = parts[0] if len(parts) > 0 else "UNKNOWN"
    timeframe = parts[1] if len(parts) > 1 else "15m"
    
    # Validate
    is_valid, report = validate_crypto_data(df, symbol, timeframe)
    
    # Print report
    import json
    print(json.dumps(report, indent=2))
    
    # Exit with appropriate code
    sys.exit(0 if is_valid else 1)