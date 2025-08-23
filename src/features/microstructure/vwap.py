"""VWAP (Volume Weighted Average Price) microstructure features."""

import pandas as pd
import numpy as np
from typing import List


class VWAPFeatures:
    """
    Calculate VWAP-based microstructure features.
    
    VWAP is a key benchmark used by institutional traders.
    Price behavior relative to VWAP provides insights into
    institutional order flow and market microstructure.
    """
    
    def __init__(self, lookback_periods: List[int] = None):
        """
        Initialize VWAP features calculator.
        
        Args:
            lookback_periods: Periods for VWAP calculations
        """
        self.lookback_periods = lookback_periods or [20, 50, 100]
    
    def calculate_vwap(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate VWAP for multiple periods.
        
        VWAP = Sum(Price * Volume) / Sum(Volume)
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with VWAP features
        """
        df = df.copy()
        
        # Use typical price (HLC/3) for VWAP calculation
        df["typical_price"] = (df["high"] + df["low"] + df["close"]) / 3
        df["price_volume"] = df["typical_price"] * df["volume"]
        
        # Calculate VWAP for different periods
        for period in self.lookback_periods:
            if period <= len(df):
                df[f"vwap_{period}"] = (
                    df["price_volume"].rolling(period).sum() /
                    df["volume"].rolling(period).sum()
                )
        
        return df
    
    def calculate_vwap_distance(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate distance from current price to VWAP.
        
        Args:
            df: DataFrame with price and VWAP data
            
        Returns:
            DataFrame with VWAP distance features
        """
        df = df.copy()
        
        # Price distance from VWAP (percentage)
        for period in self.lookback_periods:
            vwap_col = f"vwap_{period}"
            if vwap_col in df.columns:
                df[f"vwap_distance_{period}"] = (
                    (df["close"] - df[vwap_col]) / (df[vwap_col] + 1e-10)
                )
                
                # Absolute distance
                df[f"vwap_abs_distance_{period}"] = np.abs(df[f"vwap_distance_{period}"])
        
        return df
    
    def calculate_vwap_slope(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate VWAP slope (trend direction).
        
        Args:
            df: DataFrame with VWAP data
            
        Returns:
            DataFrame with VWAP slope features
        """
        df = df.copy()
        
        # VWAP slope (rate of change)
        for period in self.lookback_periods:
            vwap_col = f"vwap_{period}"
            if vwap_col in df.columns:
                df[f"vwap_slope_{period}"] = (
                    df[vwap_col].pct_change(5, fill_method=None)  # 5-period slope
                )
        
        return df
    
    def calculate_vwap_crossover(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate VWAP crossover signals.
        
        Args:
            df: DataFrame with price and VWAP data
            
        Returns:
            DataFrame with VWAP crossover features
        """
        df = df.copy()
        
        # Price above/below VWAP signals
        for period in [20, 50]:  # Use common periods
            vwap_col = f"vwap_{period}"
            if vwap_col in df.columns:
                # Above VWAP indicator
                df[f"above_vwap_{period}"] = (df["close"] > df[vwap_col]).astype(int)
                
                # Crossover signals (1 = bullish cross, -1 = bearish cross, 0 = no cross)
                prev_above = (df["close"].shift(1) > df[vwap_col].shift(1)).astype(int)
                curr_above = (df["close"] > df[vwap_col]).astype(int)
                df[f"vwap_crossover_{period}"] = curr_above - prev_above
        
        return df
    
    def calculate_vwap_bands(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate VWAP bands (similar to Bollinger Bands).
        
        Args:
            df: DataFrame with VWAP and price data
            
        Returns:
            DataFrame with VWAP band features
        """
        df = df.copy()
        
        # Calculate VWAP bands using standard deviation of price from VWAP
        for period in [20, 50]:
            vwap_col = f"vwap_{period}"
            distance_col = f"vwap_distance_{period}"
            
            if vwap_col in df.columns and distance_col in df.columns:
                # Standard deviation of distance from VWAP
                distance_std = df[distance_col].rolling(period).std()
                
                # Upper and lower bands (2 standard deviations)
                df[f"vwap_upper_band_{period}"] = df[vwap_col] * (1 + 2 * distance_std)
                df[f"vwap_lower_band_{period}"] = df[vwap_col] * (1 - 2 * distance_std)
                
                # Band width
                df[f"vwap_band_width_{period}"] = (
                    (df[f"vwap_upper_band_{period}"] - df[f"vwap_lower_band_{period}"]) /
                    df[vwap_col]
                )
                
                # Price position within bands
                df[f"vwap_band_position_{period}"] = (
                    (df["close"] - df[f"vwap_lower_band_{period}"]) /
                    (df[f"vwap_upper_band_{period}"] - df[f"vwap_lower_band_{period}"] + 1e-10)
                )
        
        return df
    
    def calculate_vwap_momentum(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate VWAP momentum features.
        
        Args:
            df: DataFrame with VWAP features
            
        Returns:
            DataFrame with VWAP momentum features
        """
        df = df.copy()
        
        # VWAP distance momentum
        for period in [20, 50]:
            distance_col = f"vwap_distance_{period}"
            if distance_col in df.columns:
                df[f"vwap_distance_momentum_{period}"] = (
                    df[distance_col].pct_change(5, fill_method=None)
                )
        
        return df
    
    def calculate_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all VWAP-based features.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with all VWAP features
        """
        df = df.copy()
        
        # Calculate features in logical order
        df = self.calculate_vwap(df)              # Base VWAP calculation
        df = self.calculate_vwap_distance(df)     # Depends on VWAP
        df = self.calculate_vwap_slope(df)        # Depends on VWAP
        df = self.calculate_vwap_crossover(df)    # Depends on VWAP
        df = self.calculate_vwap_bands(df)        # Depends on VWAP and distance
        df = self.calculate_vwap_momentum(df)     # Depends on distance
        
        # Clean up intermediate columns
        df = df.drop(columns=["typical_price", "price_volume"], errors='ignore')
        
        return df
    
    def get_feature_names(self) -> List[str]:
        """
        Get list of feature names this class generates.
        
        Returns:
            List of feature column names
        """
        base_features = []
        
        # Add period-based features
        for period in self.lookback_periods:
            base_features.extend([
                f"vwap_{period}",
                f"vwap_distance_{period}",
                f"vwap_abs_distance_{period}",
                f"vwap_slope_{period}"
            ])
        
        # Add fixed period features (20, 50)
        for period in [20, 50]:
            if period in self.lookback_periods or period <= max(self.lookback_periods):
                base_features.extend([
                    f"above_vwap_{period}",
                    f"vwap_crossover_{period}",
                    f"vwap_upper_band_{period}",
                    f"vwap_lower_band_{period}",
                    f"vwap_band_width_{period}",
                    f"vwap_band_position_{period}",
                    f"vwap_distance_momentum_{period}"
                ])
        
        return base_features