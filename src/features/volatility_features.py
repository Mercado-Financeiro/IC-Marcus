"""Volatility-based feature calculations with advanced estimators."""

import pandas as pd
import numpy as np
from typing import List


class VolatilityEstimators:
    """
    Advanced volatility estimators for 24/7 markets.
    
    Based on Sinclair (2008) - Volatility Trading
    Optimized for crypto markets with no gaps but high frequency data.
    """
    
    @staticmethod
    def atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
        """
        Average True Range - robust for gaps and market structure.
        
        Args:
            df: DataFrame with OHLC data
            window: Rolling window period
            
        Returns:
            ATR normalized by close price (as return fraction)
        """
        high = df['high']
        low = df['low']
        close = df['close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=window).mean()
        
        # Normalize as proportion of price (return fraction)
        return atr / close
    
    @staticmethod
    def garman_klass(df: pd.DataFrame, window: int = 14) -> pd.Series:
        """
        Garman-Klass estimator (1980).
        
        Uses OHLC data, ~8x more efficient than close-to-close.
        Optimal for markets with no overnight gaps (like crypto).
        
        Args:
            df: DataFrame with OHLC data
            window: Rolling window period
            
        Returns:
            GK volatility (return fraction scale)
        """
        log_hl = np.log(df['high'] / df['low'])
        log_co = np.log(df['close'] / df['open'])
        
        gk = np.sqrt(
            0.5 * log_hl**2 - 
            (2 * np.log(2) - 1) * log_co**2
        )
        
        # Rolling mean for stability
        gk_mean = gk.rolling(window=window).mean()
        return gk_mean.clip(lower=1e-8)  # Avoid division by zero
    
    @staticmethod  
    def yang_zhang(df: pd.DataFrame, window: int = 14) -> pd.Series:
        """
        Yang-Zhang estimator (2000).
        
        Best estimator for handling drift and gaps.
        Particularly good for crypto due to continuous trading.
        
        Args:
            df: DataFrame with OHLC data
            window: Rolling window period
            
        Returns:
            YZ volatility (return fraction scale)
        """
        log_ho = np.log(df['high'] / df['open'])
        log_lo = np.log(df['low'] / df['open'])
        log_co = np.log(df['close'] / df['open'])
        
        log_oc = np.log(df['open'] / df['close'].shift())
        log_oc_mean = log_oc.rolling(window=window).mean()
        
        log_cc = np.log(df['close'] / df['close'].shift())
        log_cc_mean = log_cc.rolling(window=window).mean()
        
        # Overnight volatility
        vol_overnight = (log_oc - log_oc_mean)**2
        vol_overnight = vol_overnight.rolling(window=window).mean()
        
        # Close-to-close volatility
        vol_cc = (log_cc - log_cc_mean)**2
        vol_cc = vol_cc.rolling(window=window).mean()
        
        # Rogers-Satchell volatility
        rs = log_ho * (log_ho - log_co) + log_lo * (log_lo - log_co)
        vol_rs = rs.rolling(window=window).mean()
        
        # Combine with optimal weights
        k = 0.34 / (1.34 + (window + 1) / (window - 1))
        yz = np.sqrt(vol_overnight + k * vol_cc + (1 - k) * vol_rs)
        
        return yz.clip(lower=1e-8)
    
    @staticmethod
    def parkinson(df: pd.DataFrame, window: int = 14) -> pd.Series:
        """
        Parkinson estimator (1980).
        
        Uses high-low range, ~5x more efficient than close-to-close.
        Good baseline estimator for continuous markets.
        
        Args:
            df: DataFrame with OHLC data
            window: Rolling window period
            
        Returns:
            Parkinson volatility (return fraction scale)
        """
        log_hl = np.log(df['high'] / df['low'])
        park = log_hl / (2 * np.sqrt(np.log(2)))
        
        park_mean = park.rolling(window=window).mean()
        return park_mean.clip(lower=1e-8)
    
    @staticmethod
    def realized_volatility(df: pd.DataFrame, window: int = 14) -> pd.Series:
        """
        Classical realized volatility from close-to-close returns.
        
        Args:
            df: DataFrame with close prices
            window: Rolling window period
            
        Returns:
            Realized volatility
        """
        returns = np.log(df['close'] / df['close'].shift())
        return returns.rolling(window=window).std()


class VolatilityFeatures:
    """Enhanced volatility features with advanced estimators."""
    
    def __init__(self, lookback_periods: List[int] = None):
        """
        Initialize enhanced volatility features calculator.
        
        Args:
            lookback_periods: Periods for rolling calculations
        """
        self.lookback_periods = lookback_periods or [5, 10, 20, 50, 100, 200]
        self.vol_estimators = VolatilityEstimators()
    
    def calculate_advanced_volatility(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate advanced volatility estimators.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with advanced volatility features
        """
        for period in self.lookback_periods:
            # ATR volatility
            df[f"atr_vol_{period}"] = self.vol_estimators.atr(df, period)
            
            # Garman-Klass volatility
            df[f"gk_vol_{period}"] = self.vol_estimators.garman_klass(df, period)
            
            # Yang-Zhang volatility (best estimator)
            df[f"yz_vol_{period}"] = self.vol_estimators.yang_zhang(df, period)
            
            # Parkinson volatility
            df[f"park_vol_{period}"] = self.vol_estimators.parkinson(df, period)
        
        return df
    
    def calculate_historical_volatility(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate historical volatility metrics.
        
        Args:
            df: DataFrame with returns
            
        Returns:
            DataFrame with volatility features
        """
        if "returns" not in df.columns:
            raise ValueError("DataFrame must contain 'returns' column")
        
        for period in self.lookback_periods:
            # Standard deviation of returns
            df[f"volatility_{period}"] = df["returns"].rolling(period).std()
            
            # Annualized volatility (assuming 15-min bars)
            df[f"volatility_ann_{period}"] = (
                df[f"volatility_{period}"] * np.sqrt(365 * 24 * 4)
            )
        
        return df
    
    def calculate_parkinson_volatility(self, df: pd.DataFrame, 
                                      periods: List[int] = None) -> pd.DataFrame:
        """
        Calculate Parkinson volatility using centralized estimator.
        
        Args:
            df: DataFrame with OHLC data
            periods: Periods for calculation
            
        Returns:
            DataFrame with Parkinson volatility
        """
        periods = periods or [10, 20, 50]
        
        for period in periods:
            df[f"parkinson_vol_{period}"] = self.vol_estimators.parkinson(df, period)
        
        return df
    
    def calculate_garman_klass_volatility(self, df: pd.DataFrame,
                                         periods: List[int] = None) -> pd.DataFrame:
        """
        Calculate Garman-Klass volatility using centralized estimator.
        
        Args:
            df: DataFrame with OHLC data
            periods: Periods for calculation
            
        Returns:
            DataFrame with GK volatility
        """
        periods = periods or [10, 20, 50]
        
        for period in periods:
            df[f"gk_vol_{period}"] = self.vol_estimators.garman_klass(df, period)
        
        return df
    
    def calculate_volatility_ratios(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate volatility ratios (short/long term).
        
        Args:
            df: DataFrame with volatility features
            
        Returns:
            DataFrame with volatility ratios
        """
        # Short/long term volatility ratios
        if "volatility_10" in df.columns and "volatility_50" in df.columns:
            df["vol_ratio_10_50"] = df["volatility_10"] / (df["volatility_50"] + 1e-10)
        
        if "volatility_20" in df.columns and "volatility_100" in df.columns:
            df["vol_ratio_20_100"] = df["volatility_20"] / (df["volatility_100"] + 1e-10)
        
        if "volatility_5" in df.columns and "volatility_20" in df.columns:
            df["vol_ratio_5_20"] = df["volatility_5"] / (df["volatility_20"] + 1e-10)
        
        # Volatility momentum
        if "volatility_20" in df.columns:
            df["vol_momentum_5"] = df["volatility_20"].pct_change(5, fill_method=None)
            df["vol_momentum_20"] = df["volatility_20"].pct_change(20, fill_method=None)
        
        return df
    
    def calculate_volatility_percentiles(self, df: pd.DataFrame,
                                        lookback: int = 252) -> pd.DataFrame:
        """
        Calculate volatility percentiles.
        
        Args:
            df: DataFrame with volatility features
            lookback: Lookback period for percentile calculation
            
        Returns:
            DataFrame with volatility percentiles
        """
        if "volatility_20" in df.columns:
            df[f"vol_percentile_{lookback}"] = (
                df["volatility_20"].rolling(lookback).rank(pct=True)
            )
            
            # Volatility regime indicators
            df["high_vol_regime"] = (df[f"vol_percentile_{lookback}"] > 0.8).astype(int)
            df["low_vol_regime"] = (df[f"vol_percentile_{lookback}"] < 0.2).astype(int)
            df["normal_vol_regime"] = (
                (df[f"vol_percentile_{lookback}"] >= 0.2) & 
                (df[f"vol_percentile_{lookback}"] <= 0.8)
            ).astype(int)
        
        return df
    
    def calculate_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all volatility features including advanced estimators.
        
        Args:
            df: DataFrame with OHLC and returns data
            
        Returns:
            DataFrame with all volatility features
        """
        df = df.copy()
        
        # Enhanced with advanced estimators
        df = self.calculate_advanced_volatility(df)
        df = self.calculate_historical_volatility(df)
        df = self.calculate_parkinson_volatility(df)
        df = self.calculate_garman_klass_volatility(df)
        df = self.calculate_volatility_ratios(df)
        df = self.calculate_volatility_percentiles(df)
        
        return df