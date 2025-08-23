"""Price and return-based feature calculations."""

import pandas as pd
import numpy as np
from typing import List


class PriceFeatures:
    """Calculate price and return-based features."""
    
    def __init__(self, lookback_periods: List[int] = None):
        """
        Initialize price features calculator.
        
        Args:
            lookback_periods: Periods for rolling calculations
        """
        self.lookback_periods = lookback_periods or [5, 10, 20, 50, 100, 200]
    
    def calculate_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate various return metrics.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with return features
        """
        # Logarithmic returns with protection
        price_ratio = df["close"] / df["close"].shift(1)
        df["returns"] = np.where(
            price_ratio > 0,
            np.log(price_ratio),
            0
        )
        
        # Returns for multiple periods
        for period in self.lookback_periods:
            # Cumulative returns
            df[f"returns_{period}"] = df["returns"].rolling(period).sum()
            
            # Momentum
            df[f"momentum_{period}"] = df["close"] / df["close"].shift(period) - 1
        
        return df
    
    def calculate_moving_averages(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate moving averages and related features.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with MA features
        """
        for period in self.lookback_periods:
            # Simple moving average
            df[f"sma_{period}"] = df["close"].rolling(period).mean()
            
            # Exponential moving average
            df[f"ema_{period}"] = df["close"].ewm(span=period, adjust=False).mean()
            
            # Distance from SMA (with protection)
            sma_col = df[f"sma_{period}"]
            df[f"price_to_sma_{period}"] = np.where(
                sma_col != 0,
                (df["close"] - sma_col) / sma_col,
                0
            )
        
        return df
    
    def calculate_zscore(self, df: pd.DataFrame, periods: List[int] = None) -> pd.DataFrame:
        """
        Calculate price z-scores.
        
        Args:
            df: DataFrame with OHLCV data
            periods: Periods for z-score calculation
            
        Returns:
            DataFrame with z-score features
        """
        periods = periods or [20, 50, 100]
        
        for period in periods:
            mean = df["close"].rolling(period).mean()
            std = df["close"].rolling(period).std()
            df[f"zscore_{period}"] = (df["close"] - mean) / (std + 1e-10)
        
        return df
    
    def calculate_crossovers(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate moving average crossovers.
        
        Args:
            df: DataFrame with MA features
            
        Returns:
            DataFrame with crossover signals
        """
        # MA crossovers (only if columns exist)
        if "sma_20" in df.columns and "sma_50" in df.columns:
            df["sma_cross_20_50"] = (
                (df["sma_20"] > df["sma_50"]).astype(int) -
                (df["sma_20"].shift(1) > df["sma_50"].shift(1)).astype(int)
            )
        
        if "sma_50" in df.columns and "sma_100" in df.columns:
            df["sma_cross_50_100"] = (
                (df["sma_50"] > df["sma_100"]).astype(int) -
                (df["sma_50"].shift(1) > df["sma_100"].shift(1)).astype(int)
            )
        
        if "ema_12" in df.columns and "ema_26" in df.columns:
            df["ema_cross_12_26"] = (
                (df["ema_12"] > df["ema_26"]).astype(int) -
                (df["ema_12"].shift(1) > df["ema_26"].shift(1)).astype(int)
            )
        
        return df
    
    def calculate_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all price features.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with all price features
        """
        df = df.copy()
        
        df = self.calculate_returns(df)
        df = self.calculate_moving_averages(df)
        df = self.calculate_zscore(df)
        df = self.calculate_crossovers(df)
        
        return df