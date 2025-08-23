"""Feature engineering orchestrator with no temporal leakage."""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from sklearn.preprocessing import StandardScaler, RobustScaler
import structlog
import warnings

from src.features.price_features import PriceFeatures
from src.features.volatility_features import VolatilityFeatures
from src.features.technical_indicators import TechnicalIndicators
from src.features.microstructure_features import MicrostructureFeatures
from src.features.calendar_features import CalendarFeatures

warnings.filterwarnings("ignore")
log = structlog.get_logger()


class FeatureEngineer:
    """Orchestrates feature engineering with temporal safety guarantees."""

    def __init__(
        self,
        lookback_periods: List[int] = None,
        technical_indicators: List[str] = None,
        scaler_type: str = "robust",
        include_microstructure: bool = True,
        include_advanced: bool = False,
    ):
        """
        Initialize the feature engineering orchestrator.
        
        Args:
            lookback_periods: Periods for rolling window features
            technical_indicators: List of technical indicators to calculate
            scaler_type: Type of scaler (standard, robust, none)
            include_microstructure: Include microstructure features
            include_advanced: Include advanced microstructure and derivatives
        """
        self.lookback_periods = lookback_periods or [5, 10, 20, 50, 100, 200]
        self.technical_indicators_list = technical_indicators or [
            "rsi", "macd", "bbands", "atr", "obv", "adx", "cci", "stoch"
        ]
        self.scaler_type = scaler_type
        self.scaler = None
        self.feature_names = []
        self.include_microstructure = include_microstructure
        self.include_advanced = include_advanced
        
        # Initialize feature calculators
        self.price_features = PriceFeatures(lookback_periods)
        self.volatility_features = VolatilityFeatures(lookback_periods)
        self.technical_indicators = TechnicalIndicators(technical_indicators)
        self.microstructure_features = MicrostructureFeatures()
        self.calendar_features = CalendarFeatures()
        
        # Configure scaler
        if scaler_type == "standard":
            self.scaler = StandardScaler()
        elif scaler_type == "robust":
            self.scaler = RobustScaler()

    def create_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Complete feature creation pipeline.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with all features
        """
        log.info("creating_features", initial_shape=df.shape)
        
        # Validate required columns
        required_cols = ["open", "high", "low", "close", "volume"]
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Required column missing: {col}")
        
        # Create copy to avoid modifying original
        df = df.copy()
        
        # 1. Price and return features
        df = self.price_features.calculate_all(df)
        
        # 2. Volatility features (requires returns from price features)
        df = self.volatility_features.calculate_all(df)
        
        # 3. Technical indicators
        df = self.technical_indicators.calculate_all(df)
        
        # 4. Microstructure features (if enabled)
        if self.include_microstructure:
            df = self.microstructure_features.calculate_all(df)
        
        # 5. Advanced features (if enabled)
        if self.include_advanced:
            df = self._add_advanced_features(df)
        
        # 6. Calendar features
        df = self.calendar_features.calculate_all(df)
        
        # Remove initial NaN rows (due to rolling windows)
        initial_rows = len(df)
        df = df.dropna()
        rows_dropped = initial_rows - len(df)
        
        # Save feature names
        self.feature_names = [
            col for col in df.columns
            if col not in ["open", "high", "low", "close", "volume"]
        ]
        
        log.info(
            "features_created",
            final_shape=df.shape,
            rows_dropped=rows_dropped,
            n_features=len(self.feature_names),
        )
        
        return df
    
    def _add_advanced_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add advanced features if available.
        
        Args:
            df: DataFrame with basic features
            
        Returns:
            DataFrame with advanced features added
        """
        try:
            # Try to import advanced microstructure module
            from src.features.microstructure import calculate_microstructure_features
            micro_features = calculate_microstructure_features(
                df, 
                lookback_periods=[10, 30, 60],
                include_order_book=False
            )
            df = pd.concat([df, micro_features], axis=1)
            log.info("advanced_microstructure_added")
        except ImportError:
            log.warning("advanced_microstructure_not_available")
        
        try:
            # Try to import derivatives module
            from src.features.derivatives import calculate_derivatives_features
            deriv_features = calculate_derivatives_features(
                df,
                symbol="BTCUSDT",
                fetch_live_data=False,
                lookback_periods=[24, 96, 288]
            )
            df = pd.concat([df, deriv_features], axis=1)
            log.info("derivatives_features_added")
        except ImportError:
            log.warning("derivatives_features_not_available")
        
        return df

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fit scaler and transform features.
        
        Use only on training data!
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with scaled features
        """
        # Create features
        df = self.create_all_features(df)
        
        # Fit and transform scaler if configured
        if self.scaler is not None:
            feature_cols = self.feature_names
            df[feature_cols] = self.scaler.fit_transform(df[feature_cols])
            log.info("scaler_fitted", scaler_type=self.scaler_type)
        
        return df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform features using fitted scaler.
        
        Use on validation/test data!
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with scaled features
        """
        # Create features
        df = self.create_all_features(df)
        
        # Transform with fitted scaler
        if self.scaler is not None:
            if not hasattr(self.scaler, "mean_") and not hasattr(self.scaler, "center_"):
                raise ValueError("Scaler not fitted! Use fit_transform first.")
            
            feature_cols = self.feature_names
            df[feature_cols] = self.scaler.transform(df[feature_cols])
        
        return df

    def get_feature_importance(self, model, feature_names: List[str] = None) -> pd.DataFrame:
        """
        Extract feature importance from model.
        
        Args:
            model: Trained model (XGBoost, RandomForest, etc)
            feature_names: Feature names (uses self.feature_names if None)
            
        Returns:
            DataFrame with importances sorted
        """
        if feature_names is None:
            feature_names = self.feature_names
        
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
        elif hasattr(model, "coef_"):
            importances = np.abs(model.coef_).flatten()
        else:
            raise ValueError("Model has no importance attribute")
        
        importance_df = pd.DataFrame({
            "feature": feature_names,
            "importance": importances
        }).sort_values("importance", ascending=False)
        
        # Normalize to percentage
        importance_df["importance_pct"] = (
            importance_df["importance"] / importance_df["importance"].sum() * 100
        )
        
        return importance_df

    def get_feature_groups(self) -> Dict[str, List[str]]:
        """
        Get features organized by group.
        
        Returns:
            Dictionary mapping group names to feature lists
        """
        groups = {
            "price": [],
            "volatility": [],
            "technical": [],
            "microstructure": [],
            "calendar": [],
            "other": []
        }
        
        for feature in self.feature_names:
            if any(x in feature for x in ["returns", "momentum", "sma", "ema", "zscore"]):
                groups["price"].append(feature)
            elif any(x in feature for x in ["volatility", "vol_", "parkinson", "gk_"]):
                groups["volatility"].append(feature)
            elif any(x in feature for x in ["rsi", "macd", "bb_", "atr", "obv", "adx", "cci", "stoch"]):
                groups["technical"].append(feature)
            elif any(x in feature for x in ["volume", "vwap", "spread", "liquidity", "kyle", "amihud"]):
                groups["microstructure"].append(feature)
            elif any(x in feature for x in ["hour", "day", "month", "session", "weekend"]):
                groups["calendar"].append(feature)
            else:
                groups["other"].append(feature)
        
        # Remove empty groups
        groups = {k: v for k, v in groups.items() if v}
        
        return groups