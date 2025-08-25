"""Liquidity-based microstructure features."""

import pandas as pd
import numpy as np
from typing import List


class LiquidityFeatures:
    """
    Calculate liquidity-based microstructure features.
    
    Focuses on market liquidity measures including Amihud illiquidity,
    Kyle's lambda, and Roll's measure - key indicators of market depth
    and transaction costs.
    """
    
    def __init__(self, lookback_periods: List[int] = None):
        """
        Initialize liquidity features calculator.
        
        Args:
            lookback_periods: Periods for rolling calculations
        """
        self.lookback_periods = lookback_periods or [20, 50]
    
    def calculate_amihud_illiquidity(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Amihud illiquidity measure.
        
        The Amihud measure captures the price impact of trading:
        |Return| / Dollar Volume
        
        Higher values indicate less liquid markets.
        
        Args:
            df: DataFrame with returns and dollar_volume data
            
        Returns:
            DataFrame with Amihud illiquidity features
        """
        df = df.copy()
        
        # Validate required columns
        if "returns" not in df.columns:
            raise ValueError("DataFrame must contain 'returns' column")
        if "dollar_volume" not in df.columns:
            raise ValueError("DataFrame must contain 'dollar_volume' column")
        
        # Amihud illiquidity measure
        df["amihud_illiq"] = (
            np.abs(df["returns"]) / (df["dollar_volume"] + 1e-10)
        )
        
        # Rolling averages
        for period in self.lookback_periods:
            if period <= len(df):
                df[f"amihud_illiq_{period}"] = (
                    df["amihud_illiq"].rolling(period).mean()
                )
        
        return df
    
    def calculate_kyle_lambda(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Kyle's lambda (simplified version).
        
        Kyle's lambda measures the price impact of order flow:
        Price Change Volatility / Volume Volatility
        
        Args:
            df: DataFrame with returns and volume data
            
        Returns:
            DataFrame with Kyle's lambda features
        """
        df = df.copy()
        
        # Validate required columns
        if "returns" not in df.columns:
            raise ValueError("DataFrame must contain 'returns' column")
        
        # Kyle's lambda (simplified)
        for period in self.lookback_periods:
            if period <= len(df):
                df[f"kyle_lambda_{period}"] = (
                    df["returns"].rolling(period).std() /
                    (df["volume"].rolling(period).std() + 1e-10)
                )
        
        return df
    
    def calculate_roll_measure(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Roll's measure of bid-ask spread.
        
        Roll's measure estimates the effective spread from
        return serial correlation: 2 * sqrt(|cov(r_t, r_t-1)|)
        
        Args:
            df: DataFrame with returns data
            
        Returns:
            DataFrame with Roll's measure features
        """
        df = df.copy()
        
        # Validate required columns
        if "returns" not in df.columns:
            raise ValueError("DataFrame must contain 'returns' column")
        
        # Roll's measure (rolling window)
        df["roll_measure"] = 2 * np.sqrt(np.abs(
            df["returns"].rolling(20).cov(df["returns"].shift(1))
        ))
        
        # Alternative Roll measure with longer window
        df["roll_measure_50"] = 2 * np.sqrt(np.abs(
            df["returns"].rolling(50).cov(df["returns"].shift(1))
        ))
        
        return df
    
    def calculate_price_impact(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate price impact measures.
        
        Args:
            df: DataFrame with volume and price data
            
        Returns:
            DataFrame with price impact features
        """
        df = df.copy()
        
        # Simple price impact: price change per unit volume
        if "returns" in df.columns:
            df["price_impact"] = (
                np.abs(df["returns"]) / (df["volume"] + 1e-10)
            )
            
            # Smoothed price impact
            df["price_impact_20"] = df["price_impact"].rolling(20).mean()
        
        return df
    
    def calculate_liquidity_ratios(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate liquidity-related ratios.
        
        Args:
            df: DataFrame with liquidity features
            
        Returns:
            DataFrame with liquidity ratio features
        """
        df = df.copy()
        
        # Short vs long term liquidity (Amihud ratio)
        if "amihud_illiq_20" in df.columns and "amihud_illiq_50" in df.columns:
            df["liquidity_ratio_20_50"] = (
                df["amihud_illiq_20"] / (df["amihud_illiq_50"] + 1e-10)
            )
        
        # Kyle lambda ratio
        if "kyle_lambda_20" in df.columns and "kyle_lambda_50" in df.columns:
            df["kyle_ratio_20_50"] = (
                df["kyle_lambda_20"] / (df["kyle_lambda_50"] + 1e-10)
            )
        
        return df
    
    def calculate_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all liquidity-based features.
        
        Args:
            df: DataFrame with OHLCV, returns, and dollar_volume data
            
        Returns:
            DataFrame with all liquidity features
        """
        df = df.copy()
        
        try:
            # Calculate features in logical order
            df = self.calculate_amihud_illiquidity(df)
            df = self.calculate_kyle_lambda(df)
            df = self.calculate_roll_measure(df)
            df = self.calculate_price_impact(df)
            df = self.calculate_liquidity_ratios(df)  # Depends on other measures
        except ValueError as e:
            # If required columns are missing, skip liquidity features
            print(f"Warning: Skipping liquidity features due to missing data: {e}")
        
        return df
    
    def get_feature_names(self) -> List[str]:
        """
        Get list of feature names this class generates.
        
        Returns:
            List of feature column names
        """
        base_features = [
            "amihud_illiq", "roll_measure", "roll_measure_50",
            "price_impact", "price_impact_20",
            "liquidity_ratio_20_50", "kyle_ratio_20_50"
        ]
        
        # Add period-based features
        period_features = []
        for period in self.lookback_periods:
            period_features.extend([
                f"amihud_illiq_{period}",
                f"kyle_lambda_{period}"
            ])
        
        return base_features + period_features