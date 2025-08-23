"""Spread-based microstructure features."""

import pandas as pd
import numpy as np
from typing import List

from ..validation import validate_inputs, validate_outputs, log_execution_time


class SpreadFeatures:
    """
    Calculate spread-based microstructure features.
    
    Focuses on bid-ask spread proxies and intrabar price spreads,
    key indicators of market liquidity and trading costs.
    """
    
    def __init__(self, lookback_periods: List[int] = None):
        """
        Initialize spread features calculator.
        
        Args:
            lookback_periods: Periods for rolling calculations
        """
        self.lookback_periods = lookback_periods or [20, 50]
    
    @validate_inputs(['open', 'high', 'low', 'close'], min_rows=1, validate_ohlcv=True, validate_consistency=True)
    @validate_outputs(['hl_spread'])
    def calculate_hl_spread(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate high-low spread features.
        
        Args:
            df: DataFrame with OHLC data
            
        Returns:
            DataFrame with high-low spread features
        """
        df = df.copy()
        
        # High-Low spread as proportion of price
        df["hl_spread"] = (df["high"] - df["low"]) / (df["close"] + 1e-10)
        
        # Rolling averages of spread
        for period in self.lookback_periods:
            if period <= len(df):
                df[f"hl_spread_{period}"] = df["hl_spread"].rolling(period).mean()
        
        return df
    
    def calculate_oc_spread(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate open-close spread features.
        
        Args:
            df: DataFrame with OHLC data
            
        Returns:
            DataFrame with open-close spread features
        """
        df = df.copy()
        
        # Open-Close spread (intrabar price movement)
        df["oc_spread"] = np.abs(df["open"] - df["close"]) / (df["close"] + 1e-10)
        
        # Rolling average of OC spread
        df["oc_spread_20"] = df["oc_spread"].rolling(20).mean()
        
        return df
    
    def calculate_price_position(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate price position within the bar.
        
        Args:
            df: DataFrame with OHLC data
            
        Returns:
            DataFrame with price position features
        """
        df = df.copy()
        
        # Close position in bar (0 = at low, 1 = at high)
        df["close_position"] = (
            (df["close"] - df["low"]) / 
            (df["high"] - df["low"] + 1e-10)
        )
        
        # Open position in bar
        df["open_position"] = (
            (df["open"] - df["low"]) / 
            (df["high"] - df["low"] + 1e-10)
        )
        
        return df
    
    def calculate_bar_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate bar type classifications.
        
        Args:
            df: DataFrame with OHLC data
            
        Returns:
            DataFrame with bar type features
        """
        df = df.copy()
        
        # Bar direction indicators
        df["bullish_bar"] = (df["close"] > df["open"]).astype(int)
        df["bearish_bar"] = (df["close"] < df["open"]).astype(int)
        df["neutral_bar"] = (df["close"] == df["open"]).astype(int)
        
        # Bar body size (open-close distance)
        df["bar_body_size"] = np.abs(df["close"] - df["open"]) / (df["close"] + 1e-10)
        
        # Upper and lower shadows
        df["upper_shadow"] = (df["high"] - np.maximum(df["open"], df["close"])) / (df["close"] + 1e-10)
        df["lower_shadow"] = (np.minimum(df["open"], df["close"]) - df["low"]) / (df["close"] + 1e-10)
        
        return df
    
    def calculate_spread_volatility(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate spread volatility features.
        
        Args:
            df: DataFrame with spread features
            
        Returns:
            DataFrame with spread volatility features
        """
        df = df.copy()
        
        # HL spread volatility
        if "hl_spread" in df.columns:
            df["hl_spread_vol_20"] = df["hl_spread"].rolling(20).std()
        
        # OC spread volatility
        if "oc_spread" in df.columns:
            df["oc_spread_vol_20"] = df["oc_spread"].rolling(20).std()
        
        return df
    
    @validate_inputs(['open', 'high', 'low', 'close'], min_rows=5, validate_ohlcv=True, validate_consistency=True)
    @log_execution_time
    def calculate_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all spread-based features.
        
        Args:
            df: DataFrame with OHLC data
            
        Returns:
            DataFrame with all spread features
        """
        df = df.copy()
        
        # Calculate features in logical order
        df = self.calculate_hl_spread(df)
        df = self.calculate_oc_spread(df)
        df = self.calculate_price_position(df)
        df = self.calculate_bar_types(df)
        df = self.calculate_spread_volatility(df)  # Depends on spreads
        
        return df
    
    def get_feature_names(self) -> List[str]:
        """
        Get list of feature names this class generates.
        
        Returns:
            List of feature column names
        """
        base_features = [
            "hl_spread", "oc_spread", "oc_spread_20",
            "close_position", "open_position",
            "bullish_bar", "bearish_bar", "neutral_bar",
            "bar_body_size", "upper_shadow", "lower_shadow",
            "hl_spread_vol_20", "oc_spread_vol_20"
        ]
        
        # Add period-based features
        period_features = []
        for period in self.lookback_periods:
            period_features.append(f"hl_spread_{period}")
        
        return base_features + period_features