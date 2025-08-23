"""Market regime microstructure features."""

import pandas as pd
import numpy as np
from typing import List
from scipy import stats


class RegimeFeatures:
    """
    Calculate market regime features.
    
    Identifies different market phases and regimes using price and volume
    trend analysis. Helps categorize market states such as accumulation,
    distribution, markup, and markdown phases.
    """
    
    def __init__(self, lookback_periods: List[int] = None):
        """
        Initialize regime features calculator.
        
        Args:
            lookback_periods: Periods for trend analysis
        """
        self.lookback_periods = lookback_periods or [30, 50, 100]
    
    def calculate_price_trends(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate price trend features using linear regression.
        
        Args:
            df: DataFrame with price data
            
        Returns:
            DataFrame with price trend features
        """
        df = df.copy()
        
        # Price trend using linear regression for multiple periods
        for period in self.lookback_periods:
            if period <= len(df):
                df[f"price_trend_{period}"] = df["close"].rolling(period).apply(
                    lambda x: stats.linregress(range(len(x)), x)[0]
                    if len(x) == period else np.nan,
                    raw=False
                )
                
                # Trend R-squared (strength of trend)
                df[f"price_trend_r2_{period}"] = df["close"].rolling(period).apply(
                    lambda x: stats.linregress(range(len(x)), x)[2]**2
                    if len(x) == period else np.nan,
                    raw=False
                )
        
        return df
    
    def calculate_volume_trends(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate volume trend features using linear regression.
        
        Args:
            df: DataFrame with volume data
            
        Returns:
            DataFrame with volume trend features
        """
        df = df.copy()
        
        # Volume trend using linear regression
        for period in self.lookback_periods:
            if period <= len(df):
                df[f"volume_trend_{period}"] = df["volume"].rolling(period).apply(
                    lambda x: stats.linregress(range(len(x)), x)[0]
                    if len(x) == period else np.nan,
                    raw=False
                )
        
        return df
    
    def calculate_market_phases(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Wyckoff-style market phases.
        
        Uses price and volume trends to identify:
        - Accumulation: Price down, Volume up
        - Markup: Price up, Volume up  
        - Distribution: Price up, Volume down
        - Markdown: Price down, Volume down
        
        Args:
            df: DataFrame with price and volume trend features
            
        Returns:
            DataFrame with market phase features
        """
        df = df.copy()
        
        # Use medium-term trends for phase classification
        trend_period = 50
        price_trend_col = f"price_trend_{trend_period}"
        volume_trend_col = f"volume_trend_{trend_period}"
        
        if price_trend_col in df.columns and volume_trend_col in df.columns:
            # Market phases based on price/volume trend combinations
            df["accumulation_phase"] = (
                (df[price_trend_col] < 0) & (df[volume_trend_col] > 0)
            ).astype(int)
            
            df["markup_phase"] = (
                (df[price_trend_col] > 0) & (df[volume_trend_col] > 0)
            ).astype(int)
            
            df["distribution_phase"] = (
                (df[price_trend_col] > 0) & (df[volume_trend_col] < 0)
            ).astype(int)
            
            df["markdown_phase"] = (
                (df[price_trend_col] < 0) & (df[volume_trend_col] < 0)
            ).astype(int)
        
        return df
    
    def calculate_momentum_regimes(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate momentum-based regime features.
        
        Args:
            df: DataFrame with returns data
            
        Returns:
            DataFrame with momentum regime features
        """
        df = df.copy()
        
        if "returns" in df.columns:
            # Rolling momentum (cumulative returns)
            for period in [20, 50]:
                if period <= len(df):
                    df[f"momentum_{period}"] = df["returns"].rolling(period).sum()
            
            # Momentum percentile ranking
            if "momentum_20" in df.columns:
                df["momentum_percentile"] = (
                    df["momentum_20"].rolling(252).rank(pct=True)
                )
                
                # Momentum regime indicators
                df["high_momentum"] = (df["momentum_percentile"] > 0.8).astype(int)
                df["low_momentum"] = (df["momentum_percentile"] < 0.2).astype(int)
                df["neutral_momentum"] = (
                    (df["momentum_percentile"] >= 0.2) & 
                    (df["momentum_percentile"] <= 0.8)
                ).astype(int)
        
        return df
    
    def calculate_volatility_regimes(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate volatility-based regime features.
        
        Args:
            df: DataFrame with returns data
            
        Returns:
            DataFrame with volatility regime features
        """
        df = df.copy()
        
        if "returns" in df.columns:
            # Rolling volatility
            df["volatility_20"] = df["returns"].rolling(20).std()
            
            # Volatility percentile ranking
            df["volatility_percentile"] = (
                df["volatility_20"].rolling(252).rank(pct=True)
            )
            
            # Volatility regime indicators
            df["high_volatility"] = (df["volatility_percentile"] > 0.8).astype(int)
            df["low_volatility"] = (df["volatility_percentile"] < 0.2).astype(int)
            df["normal_volatility"] = (
                (df["volatility_percentile"] >= 0.2) & 
                (df["volatility_percentile"] <= 0.8)
            ).astype(int)
        
        return df
    
    def calculate_trend_strength(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate trend strength indicators.
        
        Args:
            df: DataFrame with trend features
            
        Returns:
            DataFrame with trend strength features
        """
        df = df.copy()
        
        # Strong trend indicators based on R-squared
        for period in [30, 50]:
            r2_col = f"price_trend_r2_{period}"
            if r2_col in df.columns:
                df[f"strong_trend_{period}"] = (df[r2_col] > 0.5).astype(int)
                df[f"weak_trend_{period}"] = (df[r2_col] < 0.2).astype(int)
        
        return df
    
    def calculate_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all regime-based features.
        
        Args:
            df: DataFrame with OHLCV and returns data
            
        Returns:
            DataFrame with all regime features
        """
        df = df.copy()
        
        # Calculate features in logical order
        df = self.calculate_price_trends(df)
        df = self.calculate_volume_trends(df)
        df = self.calculate_market_phases(df)       # Depends on trends
        df = self.calculate_momentum_regimes(df)
        df = self.calculate_volatility_regimes(df)
        df = self.calculate_trend_strength(df)      # Depends on trend R-squared
        
        return df
    
    def get_feature_names(self) -> List[str]:
        """
        Get list of feature names this class generates.
        
        Returns:
            List of feature column names
        """
        base_features = [
            "accumulation_phase", "markup_phase", "distribution_phase", "markdown_phase",
            "momentum_percentile", "high_momentum", "low_momentum", "neutral_momentum",
            "volatility_20", "volatility_percentile", "high_volatility", "low_volatility", "normal_volatility"
        ]
        
        # Add period-based features
        period_features = []
        
        # Price and volume trends
        for period in self.lookback_periods:
            period_features.extend([
                f"price_trend_{period}",
                f"price_trend_r2_{period}",
                f"volume_trend_{period}"
            ])
        
        # Momentum features
        for period in [20, 50]:
            period_features.append(f"momentum_{period}")
        
        # Trend strength features
        for period in [30, 50]:
            period_features.extend([
                f"strong_trend_{period}",
                f"weak_trend_{period}"
            ])
        
        return base_features + period_features