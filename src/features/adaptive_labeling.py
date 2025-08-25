"""Adaptive labeling system based on volatility for crypto markets."""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.ensemble import RandomForestClassifier
# Use temporal validator with embargo
from src.features.validation.temporal import TemporalValidator, TemporalValidationConfig
from sklearn.metrics import f1_score, average_precision_score
import warnings

from .volatility_features import VolatilityEstimators
from src.utils.logging_config import get_logger, safe_divide, validate_dataframe

warnings.filterwarnings('ignore')

# Initialize module logger
log = get_logger(__name__)


def resolve_funding_minutes(symbol: str, timestamp: pd.Timestamp = None) -> int:
    """
    Resolve funding period dynamically based on symbol and timestamp.
    
    Based on Binance funding rules:
    - Standard: 480 min (8 hours) for most perpetual contracts
    - Conditional: 60 min (1 hour) when funding rate hits cap/floor
    - Special: Some contracts may have 240 min (4 hours)
    
    Args:
        symbol: Contract symbol (e.g., BTCUSDT)
        timestamp: Timestamp to check for special rules
        
    Returns:
        Funding period in minutes
    """
    # Base funding periods by symbol
    # Source: https://www.binance.com/en/support/faq/detail/360033525031
    FUNDING_PERIODS = {
        # Majors - 8 hours standard
        "BTCUSDT": 480,
        "ETHUSDT": 480,
        "BNBUSDT": 480,
        "XRPUSDT": 480,
        "ADAUSDT": 480,
        "SOLUSDT": 480,
        "DOTUSDT": 480,
        "DOGEUSDT": 480,
        
        # Contracts with special periods (examples)
        # Add as needed based on exchange documentation
    }
    
    # Get base period for symbol
    base_period = FUNDING_PERIODS.get(symbol, 480)  # Default 8h
    
    # TODO: Implement conditional logic
    # If funding rate hits cap/floor, may switch to 1h temporarily
    # This requires access to current funding rate from exchange
    
    return base_period


class AdaptiveLabeler:
    """
    Adaptive labeling system based on volatility for 24/7 crypto markets.
    
    Creates labels using dynamic thresholds based on market volatility,
    supporting multiple time horizons aligned with 15-minute timeframes.
    """
    
    def __init__(
        self,
        horizon_bars: int = 4,  # 1h in 15min data
        k: float = 1.0,  # Threshold multiplier
        vol_estimator: str = 'yang_zhang',  # Volatility estimator
        neutral_zone: bool = True,  # Use neutral zone
        funding_period_minutes: int = 480  # 8 hours default
    ):
        """
        Initialize adaptive labeler.
        
        Args:
            horizon_bars: Future window for return calculation
            k: Threshold multiplier (to be optimized)
            vol_estimator: 'atr', 'garman_klass', 'yang_zhang', 'parkinson'
            neutral_zone: If True, creates neutral zone between thresholds
            funding_period_minutes: Funding period in minutes for horizon mapping
        """
        self.horizon_bars = horizon_bars
        self.k = k
        self.vol_estimator = vol_estimator
        self.neutral_zone = neutral_zone
        self.funding_period_minutes = funding_period_minutes
        self.volatility_estimators = VolatilityEstimators()
        
        # Dynamic horizon mapping based on funding period
        funding_horizon_bars = funding_period_minutes // 15
        self.horizon_map = {
            '15m': 1,   # 15 minutes = 1 bar
            '30m': 2,   # 30 minutes = 2 bars
            '60m': 4,   # 60 minutes = 4 bars
            '120m': 8,  # 120 minutes = 8 bars
            '240m': 16, # 240 minutes = 16 bars
            f'{funding_period_minutes}m': funding_horizon_bars  # Dynamic funding cycle
        }
    
    def calculate_volatility(self, df: pd.DataFrame, window: int = 20) -> pd.Series:
        """
        Calculate volatility using selected estimator.
        
        Args:
            df: DataFrame with OHLC data
            window: Rolling window period
            
        Returns:
            Volatility series
        """
        estimator_map = {
            'atr': self.volatility_estimators.atr,
            'garman_klass': self.volatility_estimators.garman_klass,
            'yang_zhang': self.volatility_estimators.yang_zhang,
            'parkinson': self.volatility_estimators.parkinson,
            'realized': self.volatility_estimators.realized_volatility
        }
        
        if self.vol_estimator not in estimator_map:
            raise ValueError(f"Estimator {self.vol_estimator} not supported")
        
        return estimator_map[self.vol_estimator](df, window)
    
    def calculate_adaptive_threshold(self, df: pd.DataFrame, window: int = 20) -> pd.Series:
        """
        Calculate adaptive threshold based on volatility.
        
        Args:
            df: DataFrame with OHLC data
            window: Rolling window for volatility
            
        Returns:
            Adaptive threshold series
        """
        # Validate inputs
        validate_dataframe(df, "price_data")
        
        if window <= 0:
            log.error("invalid_window", window=window)
            raise ValueError(f"Window must be positive, got {window}")
        
        if window > len(df):
            log.warning(
                "window_exceeds_data",
                window=window,
                data_length=len(df),
                action="capping_window"
            )
            window = min(window, len(df))
        
        volatility = self.calculate_volatility(df, window)
        
        # Adjust threshold based on horizon
        # Longer horizons need higher thresholds
        horizon_adjustment = np.sqrt(self.horizon_bars)
        
        # Ensure k is positive
        assert self.k > 0, f"k must be positive, got {self.k}"
        
        threshold = self.k * volatility * horizon_adjustment
        
        # Apply reasonable bounds
        threshold = threshold.clip(lower=0.001, upper=0.10)
        
        # Log threshold statistics
        log.debug(
            "threshold_calculated",
            k=self.k,
            horizon_bars=self.horizon_bars,
            horizon_adjustment=horizon_adjustment,
            threshold_mean=threshold.mean(),
            threshold_std=threshold.std(),
            threshold_min=threshold.min(),
            threshold_max=threshold.max()
        )
        
        return threshold
    
    def create_labels(self, df: pd.DataFrame, window: int = 20) -> pd.Series:
        """
        Create adaptive labels based on volatility thresholds.
        
        Args:
            df: DataFrame with OHLC data
            window: Window for threshold calculation
            
        Returns:
            Labels series: 1 (long), 0 (neutral), -1 (short)
        """
        # Validate inputs
        validate_dataframe(df, "price_data")
        
        if 'close' not in df.columns:
            log.error("missing_close_column", columns=df.columns.tolist())
            raise ValueError("DataFrame must have 'close' column")
        
        # Check for sufficient data
        if len(df) <= self.horizon_bars:
            log.error(
                "insufficient_data_for_horizon",
                data_length=len(df),
                horizon_bars=self.horizon_bars
            )
            raise ValueError(f"Need at least {self.horizon_bars + 1} bars, got {len(df)}")
        
        # Calculate future return with safety check
        close_prices = df['close']
        if (close_prices == 0).any():
            log.warning(
                "zero_prices_found",
                count=(close_prices == 0).sum(),
                action="will_cause_division_issues"
            )
        
        # Use standard division for now, as safe_divide expects scalars
        future_return = (
            df['close'].shift(-self.horizon_bars) / df['close'] - 1
        )
        
        # Calculate adaptive threshold
        threshold = self.calculate_adaptive_threshold(df, window)
        
        # Create labels
        labels = pd.Series(index=df.index, dtype=float)
        
        if self.neutral_zone:
            # With neutral zone: -1, 0, 1
            labels[future_return > threshold] = 1  # Long
            labels[future_return < -threshold] = -1  # Short  
            labels[(future_return >= -threshold) & (future_return <= threshold)] = 0  # Neutral
        else:
            # Without neutral zone: -1, 1
            labels[future_return > 0] = 1  # Long
            labels[future_return <= 0] = -1  # Short
        
        # Log label distribution
        distribution = self.get_label_distribution(labels)
        log.info(
            "labels_created",
            neutral_zone=self.neutral_zone,
            horizon_bars=self.horizon_bars,
            window=window,
            distribution=distribution
        )
        
        return labels
    
    def _create_labels_with_params(
        self,
        df: pd.DataFrame,
        k: float,
        horizon_bars: int,
        window: int = 20
    ) -> pd.Series:
        """
        Create labels with specific parameters without modifying object state.
        
        This is a helper method that allows testing different parameters
        without mutating the object's internal state.
        
        Args:
            df: DataFrame with OHLC data
            k: Threshold multiplier to use
            horizon_bars: Horizon in bars
            window: Window for threshold calculation
            
        Returns:
            Labels series: 1 (long), 0 (neutral), -1 (short)
        """
        # Calculate future return with specified horizon
        future_return = (
            df['close'].shift(-horizon_bars) / df['close'] - 1
        )
        
        # Calculate volatility
        volatility = self.calculate_volatility(df, window)
        
        # Calculate threshold with specified k and horizon
        horizon_adjustment = np.sqrt(horizon_bars)
        threshold = k * volatility * horizon_adjustment
        threshold = threshold.clip(lower=0.001, upper=0.10)
        
        # Create labels
        labels = pd.Series(index=df.index, dtype=float)
        
        if self.neutral_zone:
            # With neutral zone: -1, 0, 1
            labels[future_return > threshold] = 1  # Long
            labels[future_return < -threshold] = -1  # Short  
            labels[(future_return >= -threshold) & (future_return <= threshold)] = 0  # Neutral
        else:
            # Without neutral zone: -1, 1
            labels[future_return > 0] = 1  # Long
            labels[future_return <= 0] = -1  # Short
        
        return labels
    
    def get_label_distribution(self, labels: pd.Series) -> Dict:
        """
        Get distribution statistics for labels.
        
        Args:
            labels: Labels series
            
        Returns:
            Distribution statistics
        """
        # Validate input
        if labels is None or labels.empty:
            log.warning("empty_labels_for_distribution")
            return {
                'counts': {},
                'proportions': {},
                'total': 0,
                'balance_ratio': 0
            }
        
        # Remove NaN values
        labels_clean = labels.dropna()
        if labels_clean.empty:
            log.warning("all_nan_labels")
            return {
                'counts': {},
                'proportions': {},
                'total': 0,
                'balance_ratio': 0
            }
        
        counts = labels_clean.value_counts()
        proportions = labels_clean.value_counts(normalize=True)
        
        # Calculate balance ratio safely
        if len(counts) == 0:
            log.warning("no_classes_found")
            balance_ratio = 0
        elif len(counts) == 1:
            log.warning("single_class_found", class_value=counts.index[0])
            balance_ratio = 0  # Single class has no balance
        else:
            balance_ratio = safe_divide(counts.min(), counts.max(), default=0)
            
            # Log if severely imbalanced
            if balance_ratio < 0.1:
                log.warning(
                    "severe_class_imbalance",
                    balance_ratio=balance_ratio,
                    min_class_count=counts.min(),
                    max_class_count=counts.max()
                )
        
        result = {
            'counts': counts.to_dict(),
            'proportions': proportions.to_dict(),
            'total': len(labels_clean),
            'balance_ratio': balance_ratio
        }
        
        log.debug("label_distribution_calculated", **result)
        
        return result
    
    def optimize_k_for_horizon(
        self,
        df: pd.DataFrame,
        X: pd.DataFrame,
        horizon: str,
        cv_splits: int = 3,
        metric: str = 'pr_auc',
        k_range: Tuple[float, float] = (0.5, 2.0)
    ) -> float:
        """
        Optimize k multiplier for specific horizon.
        
        This method tests different k values WITHOUT modifying the object's state.
        
        Args:
            df: DataFrame with OHLC data
            X: Features for model training
            horizon: Target horizon ('15m', '30m', etc.)
            cv_splits: Number of CV splits
            metric: Optimization metric ('f1', 'pr_auc')
            k_range: Range of k values to test
            
        Returns:
            Optimal k value
        """
        # Input validation
        validate_dataframe(df, "price_data")
        validate_dataframe(X, "features")
        
        if cv_splits < 2:
            log.error("insufficient_cv_splits", cv_splits=cv_splits, required=2)
            raise ValueError(f"cv_splits must be >= 2, got {cv_splits}")
        
        if k_range[0] <= 0 or k_range[1] <= k_range[0]:
            log.error("invalid_k_range", k_range=k_range)
            raise ValueError(f"Invalid k_range {k_range}, must have 0 < k_min < k_max")
        
        if metric not in ['f1', 'pr_auc']:
            log.error("unsupported_metric", metric=metric)
            raise ValueError(f"Metric {metric} not supported. Use 'f1' or 'pr_auc'")
        
        log.info(
            "k_optimization_started",
            horizon=horizon,
            cv_splits=cv_splits,
            metric=metric,
            k_range=k_range,
            data_shape=df.shape,
            features_shape=X.shape
        )
        
        # Save original state
        original_k = self.k
        original_horizon_bars = self.horizon_bars
        
        try:
            # Get horizon value without modifying self
            if horizon not in self.horizon_map:
                log.error("invalid_horizon", horizon=horizon, available=list(self.horizon_map.keys()))
                raise ValueError(f"Horizon {horizon} not supported. Available: {list(self.horizon_map.keys())}")
            horizon_bars = self.horizon_map[horizon]
            
            best_k = original_k  # Start with current k as default
            best_score = -np.inf
            
            # Test different k values
            k_values = np.linspace(k_range[0], k_range[1], 20)
            log.debug("testing_k_values", k_values=k_values.tolist())
            
            for k_idx, k_test in enumerate(k_values):
                log.debug("testing_k", k_test=k_test, iteration=k_idx+1, total=len(k_values))
                
                # Create labels with test parameters (without modifying state)
                labels = self._create_labels_with_params(df, k=k_test, horizon_bars=horizon_bars)
                
                # Remove NaN values
                mask = ~(labels.isna() | X.isna().any(axis=1))
                X_clean = X[mask]
                y_clean = labels[mask]
                
                # Check if we have enough data
                if len(X_clean) < cv_splits * 100:  # Minimum 100 samples per fold
                    log.warning(
                        "insufficient_data_for_cv",
                        k_test=k_test,
                        n_samples=len(X_clean),
                        required=cv_splits * 100
                    )
                    continue
                
                # Convert to binary if needed
                if metric in ['f1', 'pr_auc']:
                    y_clean = (y_clean > 0).astype(int)
                
                # Time series CV with embargo
                val_config = TemporalValidationConfig(n_splits=cv_splits, embargo=10)
                validator = TemporalValidator(val_config)
                scores = []
                
                for fold_idx, (train_idx, val_idx) in enumerate(validator.split(X_clean, y_clean, strategy='purged_kfold')):
                    X_train, X_val = X_clean.iloc[train_idx], X_clean.iloc[val_idx]
                    y_train, y_val = y_clean.iloc[train_idx], y_clean.iloc[val_idx]
                    
                    # Ensure we have enough samples for each class
                    train_classes = np.unique(y_train)
                    val_classes = np.unique(y_val)
                    
                    if len(train_classes) < 2:
                        log.warning(
                            "insufficient_train_classes",
                            k_test=k_test,
                            fold=fold_idx,
                            n_classes=len(train_classes),
                            classes=train_classes.tolist()
                        )
                        continue
                    
                    if len(val_classes) < 2:
                        log.warning(
                            "insufficient_val_classes",
                            k_test=k_test,
                            fold=fold_idx,
                            n_classes=len(val_classes),
                            classes=val_classes.tolist()
                        )
                        continue
                        
                    try:
                        # Simple model for quick evaluation
                        model = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
                        model.fit(X_train, y_train)
                        
                        if metric == 'f1':
                            y_pred = model.predict(X_val)
                            # Ensure prediction length matches validation labels
                            assert len(y_pred) == len(y_val), f"Prediction length mismatch: {len(y_pred)} != {len(y_val)}"
                            score = f1_score(y_val, y_pred, average='weighted')
                        elif metric == 'pr_auc':
                            y_proba = model.predict_proba(X_val)
                            if y_proba.shape[1] > 1:
                                y_proba = y_proba[:, 1]
                            else:
                                y_proba = y_proba[:, 0]
                            assert len(y_proba) == len(y_val), f"Probability length mismatch: {len(y_proba)} != {len(y_val)}"
                            score = average_precision_score(y_val, y_proba)
                        else:
                            raise ValueError(f"Metric {metric} not supported")
                        
                        if not np.isnan(score):
                            scores.append(score)
                            log.debug(
                                "fold_score",
                                k_test=k_test,
                                fold=fold_idx,
                                metric=metric,
                                score=score
                            )
                        else:
                            log.warning(
                                "nan_score",
                                k_test=k_test,
                                fold=fold_idx,
                                metric=metric
                            )
                            
                    except Exception as e:
                        log.warning(
                            "fold_evaluation_failed",
                            k_test=k_test,
                            fold=fold_idx,
                            error=str(e),
                            error_type=type(e).__name__
                        )
                        continue
                
                # Calculate average score with safety checks
                if not scores:
                    log.warning(
                        "no_valid_scores",
                        k_test=k_test,
                        reason="All folds failed or had insufficient data"
                    )
                    avg_score = 0
                else:
                    avg_score = np.mean(scores)
                    log.debug(
                        "k_performance",
                        k_test=k_test,
                        avg_score=avg_score,
                        n_valid_folds=len(scores),
                        scores=scores
                    )
                
                if avg_score > best_score:
                    best_score = avg_score
                    best_k = k_test
                    log.info(
                        "new_best_k",
                        k=best_k,
                        score=best_score,
                        previous_best=original_k
                    )
            
            # Final validation
            if best_score == -np.inf:
                log.error(
                    "optimization_failed",
                    horizon=horizon,
                    reason="No valid scores obtained for any k value"
                )
                log.info("using_default_k", k=original_k)
                best_k = original_k
            else:
                log.info(
                    "optimization_complete",
                    horizon=horizon,
                    optimal_k=best_k,
                    best_score=best_score,
                    improvement=(best_score > 0)
                )
                print(f"✅ Optimal k for {horizon}: {best_k:.3f} (score: {best_score:.4f})")
            
            return best_k
            
        except Exception as e:
            log.error(
                "optimization_error",
                horizon=horizon,
                error=str(e),
                error_type=type(e).__name__
            )
            raise
        finally:
            # Always restore original state
            self.k = original_k
            self.horizon_bars = original_horizon_bars
            log.debug(
                "state_restored",
                k=self.k,
                horizon_bars=self.horizon_bars
            )
    
    def optimize_k_multi_horizon(
        self,
        df: pd.DataFrame,
        X: pd.DataFrame,
        horizons: Optional[List[str]] = None,
        cv_splits: int = 3,
        metric: str = 'pr_auc'
    ) -> Dict:
        """
        Optimize k for multiple horizons.
        
        Args:
            df: DataFrame with OHLC data
            X: Features
            horizons: List of horizons to optimize
            cv_splits: CV splits
            metric: Optimization metric
            
        Returns:
            Dictionary with optimal k for each horizon
        """
        if horizons is None:
            horizons = ['15m', '30m', '60m', '120m']
        
        results = {}
        
        for horizon in horizons:
            print(f"\nOptimizing k for horizon {horizon}...")
            optimal_k = self.optimize_k_for_horizon(
                df, X, horizon, cv_splits, metric
            )
            results[horizon] = optimal_k
        
        return results
    
    def set_horizon(self, horizon: str) -> None:
        """
        Set horizon from string representation.
        
        Args:
            horizon: Horizon string ('15m', '30m', etc.)
        """
        if horizon not in self.horizon_map:
            raise ValueError(f"Horizon {horizon} not supported. Available: {list(self.horizon_map.keys())}")
        
        self.horizon_bars = self.horizon_map[horizon]
    
    def get_threshold_stats(self, df: pd.DataFrame, window: int = 20) -> Dict:
        """
        Get statistics about adaptive thresholds.
        
        Args:
            df: DataFrame with OHLC data
            window: Window for threshold calculation
            
        Returns:
            Threshold statistics
        """
        # Validate inputs
        validate_dataframe(df, "price_data")
        
        threshold = self.calculate_adaptive_threshold(df, window)
        
        # Check for valid threshold values
        if threshold.isna().all():
            log.error("all_nan_thresholds")
            raise ValueError("All threshold values are NaN")
        
        if threshold.isna().any():
            log.warning(
                "nan_thresholds_found",
                count=threshold.isna().sum(),
                percentage=threshold.isna().sum() / len(threshold) * 100
            )
        
        # Calculate statistics safely
        valid_threshold = threshold.dropna()
        
        if len(valid_threshold) == 0:
            log.error("no_valid_thresholds")
            return {
                'mean': np.nan,
                'std': np.nan,
                'min': np.nan,
                'max': np.nan,
                'median': np.nan,
                'q25': np.nan,
                'q75': np.nan
            }
        
        stats = {
            'mean': valid_threshold.mean(),
            'std': valid_threshold.std(),
            'min': valid_threshold.min(),
            'max': valid_threshold.max(),
            'median': valid_threshold.median(),
            'q25': valid_threshold.quantile(0.25),
            'q75': valid_threshold.quantile(0.75)
        }
        
        log.debug("threshold_stats_calculated", **stats)
        
        return stats