"""Volume-based microstructure features."""

import pandas as pd
import numpy as np
from typing import List

from ..validation import validate_inputs, validate_outputs, log_execution_time


class VolumeFeatures:
    """
    Calculate volume-based microstructure features.
    
    Specialized class focusing solely on volume analysis,
    separated from the monolithic MicrostructureFeatures.
    """
    
    def __init__(self, lookback_periods: List[int] = None):
        """
        Initialize volume features calculator.
        
        Args:
            lookback_periods: Periods for rolling calculations
        """
        self.lookback_periods = lookback_periods or [10, 20, 30, 50, 60]
    
    @validate_inputs(['volume'], min_rows=1, validate_numeric=True)
    @validate_outputs(['volume_sma_10', 'volume_sma_20'], allow_row_reduction=False)
    def calculate_volume_averages(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate volume moving averages.
        
        Args:
            df: DataFrame with volume data
            
        Returns:
            DataFrame with volume average features
        """
        df = df.copy()
        
        # Volume moving averages for configured periods
        for period in self.lookback_periods:
            if period <= len(df):
                df[f"volume_sma_{period}"] = df["volume"].rolling(period).mean()
        
        return df
    
    @validate_inputs(['volume'], dependent_cols=['volume_sma_20', 'volume_sma_10', 'volume_sma_50'], min_rows=1)
    @validate_outputs(['volume_ratio', 'volume_trend'], allow_row_reduction=False)
    def calculate_volume_ratios(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate volume ratio features.
        
        Args:
            df: DataFrame with volume and volume averages
            
        Returns:
            DataFrame with volume ratio features
        """
        df = df.copy()
        
        # Current volume vs short-term average
        if "volume_sma_20" in df.columns:
            df["volume_ratio"] = df["volume"] / (df["volume_sma_20"] + 1e-10)
        
        # Short vs long term volume trend
        if "volume_sma_10" in df.columns and "volume_sma_50" in df.columns:
            df["volume_trend"] = (
                df["volume_sma_10"] / (df["volume_sma_50"] + 1e-10)
            )
        
        return df
    
    @validate_inputs(['close', 'volume'], min_rows=1, validate_numeric=True)
    @validate_outputs(['dollar_volume', 'dollar_volume_20', 'dollar_volume_50'])
    def calculate_dollar_volume(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate dollar volume features.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with dollar volume features
        """
        df = df.copy()
        
        # Dollar volume (price * volume)
        df["dollar_volume"] = df["close"] * df["volume"]
        
        # Dollar volume averages
        for period in [20, 50]:
            if period <= len(df):
                df[f"dollar_volume_{period}"] = (
                    df["dollar_volume"].rolling(period).mean()
                )
        
        return df
    
    def calculate_volume_momentum(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate volume momentum features.
        
        Args:
            df: DataFrame with volume data
            
        Returns:
            DataFrame with volume momentum features
        """
        df = df.copy()
        
        # Volume momentum (rate of change)
        for period in [5, 20]:
            if period < len(df):
                df[f"volume_momentum_{period}"] = df["volume"].pct_change(period, fill_method=None)
        
        return df
    
    def calculate_volume_volatility(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate volume volatility features.
        
        Args:
            df: DataFrame with volume data
            
        Returns:
            DataFrame with volume volatility features
        """
        df = df.copy()
        
        # Volume volatility (standard deviation)
        df["volume_volatility_20"] = df["volume"].rolling(20).std()
        
        # Coefficient of variation (normalized volatility)
        if "volume_sma_20" in df.columns:
            df["volume_cv_20"] = (
                df["volume_volatility_20"] / (df["volume_sma_20"] + 1e-10)
            )
        
        return df
    
    @validate_inputs(['close', 'volume'], min_rows=10, validate_numeric=True)
    @log_execution_time
    def calculate_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all volume-based features.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with all volume features
        """
        df = df.copy()
        
        # Calculate features in logical order
        df = self.calculate_volume_averages(df)
        df = self.calculate_volume_ratios(df)  # Depends on averages
        df = self.calculate_dollar_volume(df)
        df = self.calculate_volume_momentum(df)
        df = self.calculate_volume_volatility(df)  # Depends on averages
        
        return df
    
    def get_feature_names(self) -> List[str]:
        """
        Get list of feature names this class generates.
        
        Returns:
            List of feature column names
        """
        base_features = [
            "dollar_volume", "volume_ratio", "volume_trend",
            "volume_volatility_20", "volume_cv_20"
        ]
        
        # Add period-based features
        period_features = []
        for period in self.lookback_periods:
            period_features.extend([
                f"volume_sma_{period}",
            ])
        
        # Add fixed period features
        period_features.extend([
            "dollar_volume_20", "dollar_volume_50",
            "volume_momentum_5", "volume_momentum_20"
        ])
        
        return base_features + period_features