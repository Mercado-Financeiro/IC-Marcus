"""
AutoML Feature Generator for automatic feature creation and selection.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Callable, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
from abc import ABC, abstractmethod
import itertools
from concurrent.futures import ProcessPoolExecutor
import warnings
warnings.filterwarnings('ignore')

# Scientific computing
from scipy import stats
from scipy.signal import find_peaks
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_classif, f_regression, mutual_info_classif
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer

# Technical analysis
import ta
from ta.utils import dropna

logger = logging.getLogger(__name__)


@dataclass
class FeatureGenerationConfig:
    """Configuration for AutoML feature generation."""
    # Basic features
    enable_basic_stats: bool = True
    enable_rolling_features: bool = True
    enable_technical_indicators: bool = True
    enable_time_features: bool = True
    
    # Advanced features
    enable_interaction_features: bool = True
    enable_polynomial_features: bool = True
    enable_frequency_features: bool = True
    enable_lag_features: bool = True
    
    # Selection and filtering
    max_features: int = 500
    correlation_threshold: float = 0.95
    importance_threshold: float = 0.01
    enable_feature_selection: bool = True
    
    # Rolling windows for time series
    rolling_windows: List[int] = field(default_factory=lambda: [5, 10, 20, 50])
    lag_periods: List[int] = field(default_factory=lambda: [1, 2, 3, 5, 10])
    
    # Performance settings
    n_jobs: int = -1
    chunk_size: int = 10000
    memory_limit_gb: float = 4.0


class FeatureTransformer(ABC):
    """Abstract base class for feature transformers."""
    
    def __init__(self, name: str, config: Dict[str, Any] = None):
        self.name = name
        self.config = config or {}
        self.fitted = False
        self.feature_names = []
    
    @abstractmethod
    def fit(self, data: pd.DataFrame, target: Optional[pd.Series] = None):
        """Fit transformer to data."""
        pass
    
    @abstractmethod
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Transform data to create features."""
        pass
    
    def fit_transform(self, data: pd.DataFrame, target: Optional[pd.Series] = None) -> pd.DataFrame:
        """Fit and transform data."""
        self.fit(data, target)
        return self.transform(data)


class BasicStatsTransformer(FeatureTransformer):
    """Generate basic statistical features."""
    
    def fit(self, data: pd.DataFrame, target: Optional[pd.Series] = None):
        """Fit basic stats transformer."""
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        self.numeric_columns = list(numeric_cols)
        self.fitted = True
        return self
    
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate basic statistical features."""
        if not self.fitted:
            raise ValueError("Transformer not fitted")
        
        features = pd.DataFrame(index=data.index)
        
        for col in self.numeric_columns:
            if col in data.columns:
                # Basic statistics
                features[f'{col}_abs'] = data[col].abs()
                features[f'{col}_squared'] = data[col] ** 2
                features[f'{col}_sqrt'] = np.sqrt(data[col].abs())
                features[f'{col}_log'] = np.log1p(data[col].abs())
                
                # Percentile ranks
                features[f'{col}_rank'] = data[col].rank(pct=True)
                
                # Z-score (standardized)
                features[f'{col}_zscore'] = (data[col] - data[col].mean()) / data[col].std()
                
                # Clipping indicators
                q01 = data[col].quantile(0.01)
                q99 = data[col].quantile(0.99)
                features[f'{col}_outlier_high'] = (data[col] > q99).astype(int)
                features[f'{col}_outlier_low'] = (data[col] < q01).astype(int)
        
        return features


class RollingStatsTransformer(FeatureTransformer):
    """Generate rolling window statistical features."""
    
    def __init__(self, name: str, windows: List[int] = [5, 10, 20]):
        super().__init__(name)
        self.windows = windows
    
    def fit(self, data: pd.DataFrame, target: Optional[pd.Series] = None):
        """Fit rolling stats transformer."""
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        self.numeric_columns = list(numeric_cols)
        self.fitted = True
        return self
    
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate rolling statistical features."""
        if not self.fitted:
            raise ValueError("Transformer not fitted")
        
        features = pd.DataFrame(index=data.index)
        
        for col in self.numeric_columns:
            if col in data.columns:
                for window in self.windows:
                    # Rolling statistics
                    rolling = data[col].rolling(window=window, min_periods=1)
                    
                    features[f'{col}_rolling_mean_{window}'] = rolling.mean()
                    features[f'{col}_rolling_std_{window}'] = rolling.std()
                    features[f'{col}_rolling_min_{window}'] = rolling.min()
                    features[f'{col}_rolling_max_{window}'] = rolling.max()
                    features[f'{col}_rolling_median_{window}'] = rolling.median()
                    features[f'{col}_rolling_skew_{window}'] = rolling.skew()
                    features[f'{col}_rolling_kurt_{window}'] = rolling.kurt()
                    
                    # Rolling ratios
                    features[f'{col}_vs_rolling_mean_{window}'] = data[col] / rolling.mean()
                    features[f'{col}_vs_rolling_std_{window}'] = (data[col] - rolling.mean()) / rolling.std()
                    
                    # Rolling quantiles
                    features[f'{col}_rolling_q25_{window}'] = rolling.quantile(0.25)
                    features[f'{col}_rolling_q75_{window}'] = rolling.quantile(0.75)
        
        return features


class TechnicalIndicatorsTransformer(FeatureTransformer):
    """Generate technical analysis indicators."""
    
    def fit(self, data: pd.DataFrame, target: Optional[pd.Series] = None):
        """Fit technical indicators transformer."""
        # Assume OHLCV data format
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        self.has_ohlcv = all(col in data.columns for col in required_cols)
        
        if not self.has_ohlcv:
            # Look for price-like columns
            price_cols = ['price', 'close', 'value']
            self.price_column = None
            for col in price_cols:
                if col in data.columns:
                    self.price_column = col
                    break
        
        self.fitted = True
        return self
    
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate technical indicator features."""
        if not self.fitted:
            raise ValueError("Transformer not fitted")
        
        features = pd.DataFrame(index=data.index)
        
        if self.has_ohlcv:
            # Full OHLCV technical indicators
            try:
                # Trend indicators
                features['sma_10'] = ta.trend.sma_indicator(data['close'], window=10)
                features['sma_20'] = ta.trend.sma_indicator(data['close'], window=20)
                features['ema_10'] = ta.trend.ema_indicator(data['close'], window=10)
                features['ema_20'] = ta.trend.ema_indicator(data['close'], window=20)
                
                # MACD
                macd_line, macd_signal, macd_histogram = ta.trend.MACD(data['close']).values()
                features['macd_line'] = macd_line
                features['macd_signal'] = macd_signal
                features['macd_histogram'] = macd_histogram
                
                # Bollinger Bands
                bb_high, bb_mid, bb_low = ta.volatility.BollingerBands(data['close']).values()
                features['bb_high'] = bb_high
                features['bb_mid'] = bb_mid
                features['bb_low'] = bb_low
                features['bb_position'] = (data['close'] - bb_low) / (bb_high - bb_low)
                
                # RSI
                features['rsi'] = ta.momentum.rsi(data['close'])
                
                # Stochastic
                features['stoch_k'] = ta.momentum.stoch(data['high'], data['low'], data['close'])
                features['stoch_d'] = ta.momentum.stoch_signal(data['high'], data['low'], data['close'])
                
                # Volume indicators
                features['volume_sma'] = ta.volume.volume_sma(data['close'], data['volume'])
                features['vwap'] = ta.volume.volume_weighted_average_price(
                    data['high'], data['low'], data['close'], data['volume']
                )
                
                # ATR (Average True Range)
                features['atr'] = ta.volatility.average_true_range(data['high'], data['low'], data['close'])
                
                # Williams %R
                features['williams_r'] = ta.momentum.williams_r(data['high'], data['low'], data['close'])
                
            except Exception as e:
                logger.warning(f"Error generating OHLCV indicators: {e}")
        
        elif self.price_column:
            # Basic price-based indicators
            price = data[self.price_column]
            
            # Simple moving averages
            for window in [5, 10, 20, 50]:
                features[f'sma_{window}'] = price.rolling(window).mean()
                features[f'price_vs_sma_{window}'] = price / features[f'sma_{window}']
            
            # Exponential moving averages
            for window in [5, 10, 20]:
                features[f'ema_{window}'] = price.ewm(span=window).mean()
            
            # Price momentum
            for period in [1, 5, 10]:
                features[f'momentum_{period}'] = price / price.shift(period) - 1
            
            # Volatility (rolling standard deviation)
            for window in [10, 20]:
                features[f'volatility_{window}'] = price.rolling(window).std()
                features[f'volatility_ratio_{window}'] = features[f'volatility_{window}'] / price
        
        return features


class InteractionFeaturesTransformer(FeatureTransformer):
    """Generate interaction features between variables."""
    
    def __init__(self, name: str, max_interactions: int = 50):
        super().__init__(name)
        self.max_interactions = max_interactions
    
    def fit(self, data: pd.DataFrame, target: Optional[pd.Series] = None):
        """Fit interaction features transformer."""
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        self.numeric_columns = list(numeric_cols)[:10]  # Limit to avoid explosion
        self.fitted = True
        return self
    
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate interaction features."""
        if not self.fitted:
            raise ValueError("Transformer not fitted")
        
        features = pd.DataFrame(index=data.index)
        interaction_count = 0
        
        # Pairwise interactions
        for i, col1 in enumerate(self.numeric_columns):
            if col1 not in data.columns:
                continue
                
            for col2 in self.numeric_columns[i+1:]:
                if col2 not in data.columns or interaction_count >= self.max_interactions:
                    continue
                
                # Multiplicative interaction
                features[f'{col1}_x_{col2}'] = data[col1] * data[col2]
                
                # Ratio interactions (avoid division by zero)
                features[f'{col1}_div_{col2}'] = data[col1] / (data[col2].abs() + 1e-8)
                
                # Additive interaction
                features[f'{col1}_plus_{col2}'] = data[col1] + data[col2]
                
                # Difference
                features[f'{col1}_minus_{col2}'] = data[col1] - data[col2]
                
                interaction_count += 4
                
                if interaction_count >= self.max_interactions:
                    break
        
        return features


class TimeFeaturesTransformer(FeatureTransformer):
    """Generate time-based features from datetime index or column."""
    
    def fit(self, data: pd.DataFrame, target: Optional[pd.Series] = None):
        """Fit time features transformer."""
        # Check for datetime index
        self.has_datetime_index = isinstance(data.index, pd.DatetimeIndex)
        
        # Check for datetime columns
        self.datetime_columns = []
        for col in data.columns:
            if pd.api.types.is_datetime64_any_dtype(data[col]):
                self.datetime_columns.append(col)
        
        self.fitted = True
        return self
    
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate time-based features."""
        if not self.fitted:
            raise ValueError("Transformer not fitted")
        
        features = pd.DataFrame(index=data.index)
        
        # Features from datetime index
        if self.has_datetime_index:
            dt = data.index
            
            # Basic time features
            features['hour'] = dt.hour
            features['day_of_week'] = dt.dayofweek
            features['day_of_month'] = dt.day
            features['month'] = dt.month
            features['quarter'] = dt.quarter
            features['year'] = dt.year
            
            # Cyclical encoding
            features['hour_sin'] = np.sin(2 * np.pi * dt.hour / 24)
            features['hour_cos'] = np.cos(2 * np.pi * dt.hour / 24)
            features['day_sin'] = np.sin(2 * np.pi * dt.dayofweek / 7)
            features['day_cos'] = np.cos(2 * np.pi * dt.dayofweek / 7)
            features['month_sin'] = np.sin(2 * np.pi * dt.month / 12)
            features['month_cos'] = np.cos(2 * np.pi * dt.month / 12)
            
            # Weekend/weekday
            features['is_weekend'] = (dt.dayofweek >= 5).astype(int)
            
            # Business/market hours (assuming financial data)
            features['is_market_hours'] = ((dt.hour >= 9) & (dt.hour <= 16)).astype(int)
        
        # Features from datetime columns
        for col in self.datetime_columns:
            if col in data.columns:
                dt = pd.to_datetime(data[col])
                
                features[f'{col}_hour'] = dt.dt.hour
                features[f'{col}_day_of_week'] = dt.dt.dayofweek
                features[f'{col}_month'] = dt.dt.month
                features[f'{col}_is_weekend'] = (dt.dt.dayofweek >= 5).astype(int)
        
        return features


class AutoMLFeatureGenerator:
    """
    Automatic feature generation system using multiple transformers.
    
    Features:
    - Multiple feature generation strategies
    - Automatic feature selection
    - Performance optimization
    - Memory management
    - Feature importance ranking
    """
    
    def __init__(self, config: FeatureGenerationConfig = None):
        """Initialize AutoML feature generator."""
        self.config = config or FeatureGenerationConfig()
        
        # Initialize transformers based on config
        self.transformers = []
        self._setup_transformers()
        
        # Feature selection
        self.feature_selector = None
        self.selected_features = []
        self.feature_importances = {}
        
        # Fitted state
        self.fitted = False
        
        logger.info(f"AutoMLFeatureGenerator initialized with {len(self.transformers)} transformers")
    
    def _setup_transformers(self):
        """Setup feature transformers based on configuration."""
        if self.config.enable_basic_stats:
            self.transformers.append(BasicStatsTransformer("basic_stats"))
        
        if self.config.enable_rolling_features:
            self.transformers.append(
                RollingStatsTransformer("rolling_stats", self.config.rolling_windows)
            )
        
        if self.config.enable_technical_indicators:
            self.transformers.append(TechnicalIndicatorsTransformer("technical"))
        
        if self.config.enable_time_features:
            self.transformers.append(TimeFeaturesTransformer("time_features"))
        
        if self.config.enable_interaction_features:
            self.transformers.append(
                InteractionFeaturesTransformer("interactions", max_interactions=50)
            )
    
    def fit(self, data: pd.DataFrame, target: Optional[pd.Series] = None) -> 'AutoMLFeatureGenerator':
        """
        Fit the feature generator to data.
        
        Args:
            data: Input data
            target: Optional target variable for supervised feature selection
            
        Returns:
            Fitted feature generator
        """
        logger.info(f"Fitting AutoML feature generator on {len(data)} samples")
        
        # Fit all transformers
        for transformer in self.transformers:
            try:
                transformer.fit(data, target)
                logger.debug(f"Fitted transformer: {transformer.name}")
            except Exception as e:
                logger.warning(f"Failed to fit transformer {transformer.name}: {e}")
        
        # Generate features for feature selection
        if self.config.enable_feature_selection and target is not None:
            features = self.transform(data)
            self._fit_feature_selector(features, target)
        
        self.fitted = True
        logger.info("AutoML feature generator fitted successfully")
        return self
    
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data to generate features.
        
        Args:
            data: Input data
            
        Returns:
            Generated features DataFrame
        """
        if not self.fitted:
            raise ValueError("Feature generator not fitted. Call fit() first.")
        
        all_features = pd.DataFrame(index=data.index)
        
        # Apply all transformers
        for transformer in self.transformers:
            try:
                transformer_features = transformer.transform(data)
                
                # Add prefix to avoid column name conflicts
                transformer_features.columns = [
                    f"{transformer.name}_{col}" for col in transformer_features.columns
                ]
                
                all_features = pd.concat([all_features, transformer_features], axis=1)
                
            except Exception as e:
                logger.warning(f"Failed to apply transformer {transformer.name}: {e}")
        
        # Handle infinite and missing values
        all_features = all_features.replace([np.inf, -np.inf], np.nan)
        all_features = all_features.fillna(all_features.median())
        
        # Apply feature selection if available
        if self.feature_selector is not None and len(self.selected_features) > 0:
            available_features = [f for f in self.selected_features if f in all_features.columns]
            all_features = all_features[available_features]
        
        # Limit number of features
        if len(all_features.columns) > self.config.max_features:
            # Select top features by importance or variance
            if self.feature_importances:
                # Sort by importance
                sorted_features = sorted(
                    self.feature_importances.items(),
                    key=lambda x: x[1],
                    reverse=True
                )
                top_features = [f[0] for f in sorted_features[:self.config.max_features]]
                top_features = [f for f in top_features if f in all_features.columns]
                all_features = all_features[top_features]
            else:
                # Select by variance
                feature_vars = all_features.var().sort_values(ascending=False)
                top_features = feature_vars.head(self.config.max_features).index
                all_features = all_features[top_features]
        
        logger.info(f"Generated {len(all_features.columns)} features")
        return all_features
    
    def fit_transform(self, data: pd.DataFrame, target: Optional[pd.Series] = None) -> pd.DataFrame:
        """Fit and transform data."""
        return self.fit(data, target).transform(data)
    
    def _fit_feature_selector(self, features: pd.DataFrame, target: pd.Series):
        """Fit feature selector for automatic feature selection."""
        logger.info("Fitting feature selector...")
        
        # Remove highly correlated features
        features_clean = self._remove_correlated_features(features)
        
        # Determine problem type
        is_classification = len(target.unique()) < 20  # Simple heuristic
        
        if is_classification:
            # Use RandomForest for feature importance
            selector = RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                n_jobs=min(4, self.config.n_jobs)
            )
        else:
            selector = RandomForestRegressor(
                n_estimators=100,
                random_state=42,
                n_jobs=min(4, self.config.n_jobs)
            )
        
        try:
            # Fit selector
            selector.fit(features_clean, target)
            
            # Get feature importances
            importances = dict(zip(features_clean.columns, selector.feature_importances_))
            
            # Filter by importance threshold
            self.selected_features = [
                feature for feature, importance in importances.items()
                if importance >= self.config.importance_threshold
            ]
            
            self.feature_importances = importances
            self.feature_selector = selector
            
            logger.info(f"Selected {len(self.selected_features)} features based on importance")
            
        except Exception as e:
            logger.warning(f"Feature selection failed: {e}")
            self.selected_features = list(features_clean.columns)
    
    def _remove_correlated_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """Remove highly correlated features."""
        if len(features.columns) <= 1:
            return features
        
        try:
            # Calculate correlation matrix
            corr_matrix = features.corr().abs()
            
            # Find highly correlated pairs
            upper_triangle = corr_matrix.where(
                np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
            )
            
            # Find features to remove
            to_remove = [
                column for column in upper_triangle.columns
                if any(upper_triangle[column] > self.config.correlation_threshold)
            ]
            
            # Remove correlated features
            features_clean = features.drop(columns=to_remove)
            
            logger.info(f"Removed {len(to_remove)} correlated features")
            return features_clean
            
        except Exception as e:
            logger.warning(f"Correlation filtering failed: {e}")
            return features
    
    def get_feature_importance(self, top_n: int = 20) -> Dict[str, float]:
        """Get top N most important features."""
        if not self.feature_importances:
            return {}
        
        sorted_features = sorted(
            self.feature_importances.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        return dict(sorted_features[:top_n])
    
    def generate_feature_report(self) -> Dict[str, Any]:
        """Generate comprehensive feature generation report."""
        report = {
            'total_transformers': len(self.transformers),
            'transformers': [t.name for t in self.transformers],
            'total_features_generated': len(self.selected_features) if self.selected_features else 0,
            'feature_selection_enabled': self.config.enable_feature_selection,
            'top_features': self.get_feature_importance(10),
            'config': {
                'max_features': self.config.max_features,
                'correlation_threshold': self.config.correlation_threshold,
                'importance_threshold': self.config.importance_threshold,
                'rolling_windows': self.config.rolling_windows
            }
        }
        
        return report
    
    def save_features(self, features: pd.DataFrame, filepath: str):
        """Save generated features to file."""
        if filepath.endswith('.parquet'):
            features.to_parquet(filepath, compression='snappy')
        else:
            features.to_csv(filepath, index=True)
        
        logger.info(f"Features saved to {filepath}")
    
    def export_transformers(self) -> List[Dict[str, Any]]:
        """Export transformer configurations."""
        transformers_config = []
        
        for transformer in self.transformers:
            config = {
                'name': transformer.name,
                'type': type(transformer).__name__,
                'fitted': transformer.fitted,
                'config': getattr(transformer, 'config', {})
            }
            transformers_config.append(config)
        
        return transformers_config


# Convenience functions
def create_automl_features(
    data: pd.DataFrame,
    target: Optional[pd.Series] = None,
    config: Optional[FeatureGenerationConfig] = None
) -> Tuple[pd.DataFrame, AutoMLFeatureGenerator]:
    """
    Create features automatically using AutoML.
    
    Args:
        data: Input data
        target: Optional target variable
        config: Feature generation configuration
        
    Returns:
        Tuple of (features, fitted_generator)
    """
    generator = AutoMLFeatureGenerator(config)
    features = generator.fit_transform(data, target)
    return features, generator


def quick_feature_engineering(
    data: pd.DataFrame,
    target_column: Optional[str] = None,
    max_features: int = 100
) -> pd.DataFrame:
    """
    Quick feature engineering with sensible defaults.
    
    Args:
        data: Input data
        target_column: Name of target column
        max_features: Maximum number of features to generate
        
    Returns:
        Generated features DataFrame
    """
    # Prepare target
    target = data[target_column] if target_column else None
    
    # Quick configuration
    config = FeatureGenerationConfig(
        max_features=max_features,
        enable_polynomial_features=False,  # Disable for speed
        enable_frequency_features=False,
        rolling_windows=[5, 10, 20],
        n_jobs=2
    )
    
    # Generate features
    generator = AutoMLFeatureGenerator(config)
    features = generator.fit_transform(data, target)
    
    return features