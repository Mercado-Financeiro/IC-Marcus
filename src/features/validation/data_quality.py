"""
Data quality validation using Great Expectations patterns.
Ensures data integrity throughout the ML pipeline.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


@dataclass
class DataQualityReport:
    """Container for data quality validation results."""
    passed: bool
    total_checks: int
    passed_checks: int
    failed_checks: int
    warnings: List[str]
    errors: List[str]
    metrics: Dict[str, Any]
    timestamp: datetime
    
    @property
    def pass_rate(self) -> float:
        """Calculate pass rate percentage."""
        return (self.passed_checks / self.total_checks * 100) if self.total_checks > 0 else 0.0
    
    def to_dict(self) -> Dict:
        """Convert report to dictionary."""
        return {
            'passed': self.passed,
            'pass_rate': self.pass_rate,
            'total_checks': self.total_checks,
            'passed_checks': self.passed_checks,
            'failed_checks': self.failed_checks,
            'warnings': self.warnings,
            'errors': self.errors,
            'metrics': self.metrics,
            'timestamp': self.timestamp.isoformat()
        }


class CryptoDataValidator:
    """
    Validator for cryptocurrency OHLCV data.
    Implements Great Expectations-style validation.
    """
    
    def __init__(self, 
                 symbol: str = "BTCUSDT",
                 timeframe: str = "15m",
                 strict_mode: bool = False):
        """
        Initialize validator.
        
        Args:
            symbol: Trading pair symbol
            timeframe: Data timeframe (1m, 5m, 15m, 1h, etc.)
            strict_mode: If True, warnings become errors
        """
        self.symbol = symbol
        self.timeframe = timeframe
        self.strict_mode = strict_mode
        
        # Parse timeframe to get expected frequency
        self.expected_freq = self._parse_timeframe(timeframe)
        
    def _parse_timeframe(self, timeframe: str) -> timedelta:
        """Parse timeframe string to timedelta."""
        unit = timeframe[-1]
        value = int(timeframe[:-1])
        
        if unit == 'm':
            return timedelta(minutes=value)
        elif unit == 'h':
            return timedelta(hours=value)
        elif unit == 'd':
            return timedelta(days=value)
        else:
            raise ValueError(f"Unknown timeframe unit: {unit}")
    
    def validate(self, df: pd.DataFrame) -> DataQualityReport:
        """
        Run comprehensive data quality checks.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataQualityReport with validation results
        """
        errors = []
        warnings = []
        metrics = {}
        checks_passed = 0
        total_checks = 0
        
        # Check 1: Required columns
        total_checks += 1
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            errors.append(f"Missing required columns: {missing_cols}")
        else:
            checks_passed += 1
            metrics['has_required_columns'] = True
        
        # Check 2: No missing values
        total_checks += 1
        null_counts = df[required_cols].isnull().sum()
        if null_counts.any():
            errors.append(f"Found missing values: {null_counts[null_counts > 0].to_dict()}")
        else:
            checks_passed += 1
            metrics['no_missing_values'] = True
        
        # Check 3: OHLC relationships
        total_checks += 1
        invalid_ohlc = (
            (df['high'] < df['low']) |
            (df['high'] < df['open']) |
            (df['high'] < df['close']) |
            (df['low'] > df['open']) |
            (df['low'] > df['close'])
        )
        if invalid_ohlc.any():
            count = invalid_ohlc.sum()
            errors.append(f"Found {count} rows with invalid OHLC relationships")
            metrics['invalid_ohlc_rows'] = count
        else:
            checks_passed += 1
            metrics['valid_ohlc'] = True
        
        # Check 4: Positive prices and volume
        total_checks += 1
        negative_prices = (df[['open', 'high', 'low', 'close']] <= 0).any(axis=1)
        if negative_prices.any():
            count = negative_prices.sum()
            errors.append(f"Found {count} rows with non-positive prices")
        else:
            checks_passed += 1
            metrics['positive_prices'] = True
        
        # Check 5: Volume check
        total_checks += 1
        zero_volume = (df['volume'] == 0).sum()
        zero_volume_pct = zero_volume / len(df) * 100
        metrics['zero_volume_pct'] = zero_volume_pct
        if zero_volume_pct > 10:
            msg = f"{zero_volume_pct:.1f}% of rows have zero volume"
            if self.strict_mode:
                errors.append(msg)
            else:
                warnings.append(msg)
        else:
            checks_passed += 1
        
        # Check 6: Temporal integrity
        if 'timestamp' in df.columns or df.index.name == 'timestamp':
            total_checks += 1
            ts = df.index if df.index.name == 'timestamp' else df['timestamp']
            
            # Check if timestamps are sorted
            if not ts.is_monotonic_increasing:
                errors.append("Timestamps are not sorted in ascending order")
            else:
                checks_passed += 1
                metrics['timestamps_sorted'] = True
            
            # Check for duplicates
            total_checks += 1
            duplicates = ts.duplicated().sum()
            if duplicates > 0:
                errors.append(f"Found {duplicates} duplicate timestamps")
            else:
                checks_passed += 1
                metrics['no_duplicate_timestamps'] = True
            
            # Check for gaps
            if len(ts) > 1:
                total_checks += 1
                time_diffs = pd.Series(ts).diff()[1:]
                expected_diff = self.expected_freq.total_seconds()
                
                # Convert time_diffs to seconds for comparison
                time_diffs_seconds = time_diffs.dt.total_seconds()
                
                # Allow 10% tolerance for gaps
                gaps = time_diffs_seconds[time_diffs_seconds > expected_diff * 1.1]
                if len(gaps) > 0:
                    max_gap_seconds = gaps.max()
                    max_gap = pd.Timedelta(seconds=max_gap_seconds)
                    msg = f"Found {len(gaps)} time gaps, max gap: {max_gap}"
                    if len(gaps) > len(df) * 0.05:  # More than 5% gaps
                        errors.append(msg)
                    else:
                        warnings.append(msg)
                    metrics['time_gaps'] = len(gaps)
                else:
                    checks_passed += 1
                    metrics['no_time_gaps'] = True
        
        # Check 7: Statistical outliers
        total_checks += 1
        returns = df['close'].pct_change(fill_method=None).dropna()
        extreme_returns = returns.abs() > 0.2  # 20% moves
        if extreme_returns.any():
            count = extreme_returns.sum()
            msg = f"Found {count} extreme price movements (>20%)"
            if count > len(df) * 0.01:  # More than 1%
                warnings.append(msg)
            else:
                checks_passed += 1
            metrics['extreme_returns'] = count
        else:
            checks_passed += 1
        
        # Check 8: Data recency
        if 'timestamp' in df.columns or df.index.name == 'timestamp':
            total_checks += 1
            ts = df.index if df.index.name == 'timestamp' else df['timestamp']
            last_ts = pd.to_datetime(ts.max())
            
            # Handle timezone-aware and naive datetimes
            try:
                # If last_ts is timezone-aware, use UTC for comparison
                if last_ts.tzinfo is not None:
                    from datetime import timezone
                    current_time = datetime.now(timezone.utc)
                else:
                    # If timezone-naive, use naive datetime
                    current_time = datetime.now()
                
                age = current_time - last_ts
                age_days = age.days
            except TypeError:
                # Fallback: try to remove timezone info
                try:
                    last_ts_naive = last_ts.tz_localize(None) if hasattr(last_ts, 'tz_localize') else last_ts.replace(tzinfo=None)
                    age = datetime.now() - last_ts_naive
                    age_days = age.days
                except:
                    # If all else fails, skip this check
                    warnings.append("Could not determine data age due to timezone issues")
                    age_days = -1
            
            if age_days >= 0:
                if age_days > 7:
                    warnings.append(f"Data is {age_days} days old")
                else:
                    checks_passed += 1
                metrics['data_age_days'] = age_days
            else:
                metrics['data_age_days'] = 'unknown'
        
        # Additional metrics
        metrics.update({
            'row_count': len(df),
            'column_count': len(df.columns),
            'price_min': df['close'].min(),
            'price_max': df['close'].max(),
            'price_mean': df['close'].mean(),
            'volume_mean': df['volume'].mean(),
        })
        
        # Create report
        passed = len(errors) == 0
        
        return DataQualityReport(
            passed=passed,
            total_checks=total_checks,
            passed_checks=checks_passed,
            failed_checks=total_checks - checks_passed,
            warnings=warnings,
            errors=errors,
            metrics=metrics,
            timestamp=datetime.now()
        )


class FeatureDataValidator:
    """
    Validator for feature data before model training.
    """
    
    def __init__(self, strict_mode: bool = False):
        """
        Initialize validator.
        
        Args:
            strict_mode: If True, warnings become errors
        """
        self.strict_mode = strict_mode
    
    def validate(self, 
                 X: pd.DataFrame, 
                 y: pd.Series,
                 feature_names: Optional[List[str]] = None) -> DataQualityReport:
        """
        Validate feature data before training.
        
        Args:
            X: Feature DataFrame
            y: Target Series
            feature_names: Expected feature names
            
        Returns:
            DataQualityReport with validation results
        """
        errors = []
        warnings = []
        metrics = {}
        checks_passed = 0
        total_checks = 0
        
        # Check 1: Shape consistency
        total_checks += 1
        if len(X) != len(y):
            errors.append(f"Shape mismatch: X has {len(X)} rows, y has {len(y)} rows")
        else:
            checks_passed += 1
            metrics['shape_consistent'] = True
        
        # Check 2: No NaN in features
        total_checks += 1
        nan_features = X.columns[X.isnull().any()].tolist()
        if nan_features:
            errors.append(f"Features with NaN values: {nan_features[:10]}")  # Show first 10
            metrics['nan_features'] = len(nan_features)
        else:
            checks_passed += 1
            metrics['no_nan_features'] = True
        
        # Check 3: No NaN in target
        total_checks += 1
        nan_target = y.isnull().sum() if isinstance(y, pd.Series) else y.isnull().sum().sum()
        if nan_target > 0:
            errors.append(f"Target has {nan_target} NaN values")
        else:
            checks_passed += 1
            metrics['no_nan_target'] = True
        
        # Check 4: Check for constant features
        total_checks += 1
        constant_features = []
        for col in X.columns:
            if X[col].nunique() <= 1:
                constant_features.append(col)
        
        if constant_features:
            msg = f"Found {len(constant_features)} constant features"
            if self.strict_mode:
                errors.append(msg + f": {constant_features[:10]}")
            else:
                warnings.append(msg)
            metrics['constant_features'] = len(constant_features)
        else:
            checks_passed += 1
            metrics['no_constant_features'] = True
        
        # Check 5: Check for infinite values
        total_checks += 1
        inf_features = []
        for col in X.select_dtypes(include=[np.number]).columns:
            if np.isinf(X[col]).any():
                inf_features.append(col)
        
        if inf_features:
            errors.append(f"Features with infinite values: {inf_features[:10]}")
            metrics['inf_features'] = len(inf_features)
        else:
            checks_passed += 1
            metrics['no_inf_values'] = True
        
        # Check 6: Class balance for classification
        if y.dtype in ['int', 'bool'] or y.nunique() < 10:
            total_checks += 1
            class_counts = y.value_counts()
            min_class = class_counts.min()
            max_class = class_counts.max()
            imbalance_ratio = max_class / min_class if min_class > 0 else float('inf')
            
            metrics['class_balance'] = class_counts.to_dict()
            metrics['imbalance_ratio'] = imbalance_ratio
            
            if imbalance_ratio > 10:
                msg = f"Severe class imbalance: ratio {imbalance_ratio:.1f}"
                if self.strict_mode:
                    errors.append(msg)
                else:
                    warnings.append(msg)
            else:
                checks_passed += 1
        
        # Check 7: Feature scale check
        total_checks += 1
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        scales = {}
        huge_scale_features = []
        
        for col in numeric_cols:
            col_range = X[col].max() - X[col].min()
            scales[col] = col_range
            if col_range > 1e6:  # Very large scale
                huge_scale_features.append(col)
        
        if huge_scale_features:
            msg = f"Found {len(huge_scale_features)} features with very large scale"
            warnings.append(msg)
            metrics['large_scale_features'] = len(huge_scale_features)
        else:
            checks_passed += 1
        
        # Check 8: Expected features check
        if feature_names:
            total_checks += 1
            missing_features = set(feature_names) - set(X.columns)
            extra_features = set(X.columns) - set(feature_names)
            
            if missing_features:
                errors.append(f"Missing expected features: {list(missing_features)[:10]}")
            if extra_features:
                warnings.append(f"Unexpected features: {list(extra_features)[:10]}")
            
            if not missing_features:
                checks_passed += 1
                metrics['all_expected_features'] = True
        
        # Additional metrics
        metrics.update({
            'n_samples': len(X),
            'n_features': len(X.columns),
            'memory_usage_mb': X.memory_usage(deep=True).sum() / 1024 / 1024,
            'target_unique': y.nunique(),
        })
        
        # Create report
        passed = len(errors) == 0
        
        return DataQualityReport(
            passed=passed,
            total_checks=total_checks,
            passed_checks=checks_passed,
            failed_checks=total_checks - checks_passed,
            warnings=warnings,
            errors=errors,
            metrics=metrics,
            timestamp=datetime.now()
        )


def validate_temporal_split(df_train: pd.DataFrame,
                           df_val: pd.DataFrame,
                           df_test: pd.DataFrame,
                           time_col: str = 'timestamp') -> DataQualityReport:
    """
    Validate temporal train/val/test split for data leakage.
    
    Args:
        df_train: Training DataFrame
        df_val: Validation DataFrame
        df_test: Test DataFrame
        time_col: Name of timestamp column
        
    Returns:
        DataQualityReport with validation results
    """
    errors = []
    warnings = []
    metrics = {}
    checks_passed = 0
    total_checks = 0
    
    # Get timestamps - try multiple ways
    train_times = None
    val_times = None
    test_times = None
    
    # Method 1: Column exists
    if time_col in df_train.columns:
        train_times = pd.to_datetime(df_train[time_col])
        val_times = pd.to_datetime(df_val[time_col])
        test_times = pd.to_datetime(df_test[time_col])
    # Method 2: Index name matches
    elif df_train.index.name == time_col:
        train_times = pd.to_datetime(df_train.index)
        val_times = pd.to_datetime(df_val.index)
        test_times = pd.to_datetime(df_test.index)
    # Method 3: Index is already datetime
    elif isinstance(df_train.index, pd.DatetimeIndex):
        train_times = df_train.index
        val_times = df_val.index
        test_times = df_test.index
    # Method 4: Try to parse index as datetime
    else:
        try:
            train_times = pd.to_datetime(df_train.index)
            val_times = pd.to_datetime(df_val.index)
            test_times = pd.to_datetime(df_test.index)
        except:
            warnings.append(f"Could not find or parse timestamps")
            return DataQualityReport(
                passed=False,
                total_checks=1,
                passed_checks=0,
                failed_checks=1,
                warnings=warnings,
                errors=["Cannot validate temporal split without timestamps"],
                metrics={},
                timestamp=datetime.now()
            )
    
    # Check 1: No overlap between train and val
    total_checks += 1
    train_max = train_times.max()
    val_min = val_times.min()
    
    if train_max >= val_min:
        errors.append(f"Train/Val overlap: train ends at {train_max}, val starts at {val_min}")
    else:
        checks_passed += 1
        gap = val_min - train_max
        metrics['train_val_gap'] = str(gap)
    
    # Check 2: No overlap between val and test
    total_checks += 1
    val_max = val_times.max()
    test_min = test_times.min()
    
    if val_max >= test_min:
        errors.append(f"Val/Test overlap: val ends at {val_max}, test starts at {test_min}")
    else:
        checks_passed += 1
        gap = test_min - val_max
        metrics['val_test_gap'] = str(gap)
    
    # Check 3: Check for sufficient embargo
    total_checks += 1
    if 'train_val_gap' in metrics:
        gap = pd.Timedelta(metrics['train_val_gap'])
        if gap < timedelta(hours=1):  # Less than 1 hour embargo
            warnings.append(f"Small train/val embargo: {gap}")
        else:
            checks_passed += 1
    
    total_checks += 1
    if 'val_test_gap' in metrics:
        gap = pd.Timedelta(metrics['val_test_gap'])
        if gap < timedelta(hours=1):  # Less than 1 hour embargo
            warnings.append(f"Small val/test embargo: {gap}")
        else:
            checks_passed += 1
    
    # Check 4: Temporal order within each set
    total_checks += 1
    if not train_times.is_monotonic_increasing:
        errors.append("Training data not in temporal order")
    else:
        checks_passed += 1
    
    total_checks += 1
    if not val_times.is_monotonic_increasing:
        errors.append("Validation data not in temporal order")
    else:
        checks_passed += 1
    
    total_checks += 1
    if not test_times.is_monotonic_increasing:
        errors.append("Test data not in temporal order")
    else:
        checks_passed += 1
    
    # Additional metrics
    metrics.update({
        'train_size': len(df_train),
        'val_size': len(df_val),
        'test_size': len(df_test),
        'train_start': str(train_times.min()),
        'train_end': str(train_times.max()),
        'val_start': str(val_times.min()),
        'val_end': str(val_times.max()),
        'test_start': str(test_times.min()),
        'test_end': str(test_times.max()),
    })
    
    # Create report
    passed = len(errors) == 0
    
    return DataQualityReport(
        passed=passed,
        total_checks=total_checks,
        passed_checks=checks_passed,
        failed_checks=total_checks - checks_passed,
        warnings=warnings,
        errors=errors,
        metrics=metrics,
        timestamp=datetime.now()
    )


# Convenience function for quick validation
def quick_validate(df: pd.DataFrame, 
                  y: Optional[pd.Series] = None,
                  data_type: str = 'ohlcv') -> Tuple[bool, List[str]]:
    """
    Quick validation with simple pass/fail result.
    
    Args:
        df: DataFrame to validate
        y: Optional target Series (for feature validation)
        data_type: Type of data ('ohlcv' or 'features')
        
    Returns:
        Tuple of (passed, list of errors/warnings)
    """
    if data_type == 'ohlcv':
        validator = CryptoDataValidator()
        report = validator.validate(df)
    elif data_type == 'features' and y is not None:
        validator = FeatureDataValidator()
        report = validator.validate(df, y)
    else:
        return False, ["Invalid data_type or missing target for feature validation"]
    
    messages = report.errors + report.warnings
    return report.passed, messages