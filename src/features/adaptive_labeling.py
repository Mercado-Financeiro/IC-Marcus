"""Adaptive labeling system based on volatility for crypto markets."""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import f1_score, average_precision_score
import warnings

from .volatility_features import VolatilityEstimators

warnings.filterwarnings('ignore')


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
        volatility = self.calculate_volatility(df, window)
        
        # Adjust threshold based on horizon
        # Longer horizons need higher thresholds
        horizon_adjustment = np.sqrt(self.horizon_bars)
        
        threshold = self.k * volatility * horizon_adjustment
        
        # Apply reasonable bounds
        threshold = threshold.clip(lower=0.001, upper=0.10)
        
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
        # Calculate future return
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
        
        return labels
    
    def get_label_distribution(self, labels: pd.Series) -> Dict:
        """
        Get distribution statistics for labels.
        
        Args:
            labels: Labels series
            
        Returns:
            Distribution statistics
        """
        counts = labels.value_counts()
        proportions = labels.value_counts(normalize=True)
        
        return {
            'counts': counts.to_dict(),
            'proportions': proportions.to_dict(),
            'total': len(labels.dropna()),
            'balance_ratio': counts.min() / counts.max() if len(counts) > 0 else 0
        }
    
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
        # Set horizon
        self.horizon_bars = self.horizon_map[horizon]
        
        best_k = self.k
        best_score = -np.inf
        
        # Test different k values
        k_values = np.linspace(k_range[0], k_range[1], 20)
        
        for k in k_values:
            self.k = k
            
            # Create labels with current k
            labels = self.create_labels(df)
            
            # Remove NaN values
            mask = ~(labels.isna() | X.isna().any(axis=1))
            X_clean = X[mask]
            y_clean = labels[mask]
            
            # Convert to binary if needed
            if metric in ['f1', 'pr_auc']:
                y_clean = (y_clean > 0).astype(int)
            
            # Time series CV
            tscv = TimeSeriesSplit(n_splits=cv_splits)
            scores = []
            
            for train_idx, val_idx in tscv.split(X_clean):
                X_train, X_val = X_clean.iloc[train_idx], X_clean.iloc[val_idx]
                y_train, y_val = y_clean.iloc[train_idx], y_clean.iloc[val_idx]
                
                # Ensure we have enough samples for each class
                if len(np.unique(y_train)) < 2 or len(np.unique(y_val)) < 2:
                    continue
                    
                try:
                    # Simple model for quick evaluation
                    model = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
                    model.fit(X_train, y_train)
                    
                    if metric == 'f1':
                        y_pred = model.predict(X_val)
                        # Ensure prediction length matches validation labels
                        if len(y_pred) != len(y_val):
                            continue
                        score = f1_score(y_val, y_pred, average='weighted')
                    elif metric == 'pr_auc':
                        y_proba = model.predict_proba(X_val)
                        if y_proba.shape[1] > 1:
                            y_proba = y_proba[:, 1]
                        else:
                            y_proba = y_proba[:, 0]
                        if len(y_proba) != len(y_val):
                            continue
                        score = average_precision_score(y_val, y_proba)
                    else:
                        raise ValueError(f"Metric {metric} not supported")
                    
                    if not np.isnan(score):
                        scores.append(score)
                        
                except Exception as e:
                    # Skip problematic folds
                    print(f"Warning: Skipping fold due to error: {e}")
                    continue
            
            avg_score = np.mean(scores)
            
            if avg_score > best_score:
                best_score = avg_score
                best_k = k
        
        print(f"✅ Optimal k for {horizon}: {best_k:.3f} (score: {best_score:.4f})")
        
        return best_k
    
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
        threshold = self.calculate_adaptive_threshold(df, window)
        
        return {
            'mean': threshold.mean(),
            'std': threshold.std(),
            'min': threshold.min(),
            'max': threshold.max(),
            'median': threshold.median(),
            'q25': threshold.quantile(0.25),
            'q75': threshold.quantile(0.75)
        }