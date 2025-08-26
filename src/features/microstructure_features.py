"""Market microstructure feature calculations."""

import pandas as pd
import numpy as np
from typing import List, Optional
from scipy import stats


class MicrostructureFeatures:
    """Calculate market microstructure features."""
    
    def __init__(self, lookback_periods: List[int] = None):
        """
        Initialize microstructure features calculator.
        
        Args:
            lookback_periods: Periods for rolling calculations
        """
        self.lookback_periods = lookback_periods or [10, 20, 30, 50, 60]
    
    def calculate_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate volume-based features.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with volume features
        """
        # Volume moving averages
        df["volume_sma_20"] = df["volume"].rolling(20).mean()
        df["volume_sma_50"] = df["volume"].rolling(50).mean()
        
        # Volume ratios
        df["volume_ratio"] = df["volume"] / (df["volume_sma_20"] + 1e-10)
        df["volume_trend"] = (
            df["volume"].rolling(10).mean() /
            (df["volume"].rolling(50).mean() + 1e-10)
        )
        
        # Dollar volume
        df["dollar_volume"] = df["close"] * df["volume"]
        df["dollar_volume_20"] = df["dollar_volume"].rolling(20).mean()
        df["dollar_volume_50"] = df["dollar_volume"].rolling(50).mean()
        
        # Volume momentum
        df["volume_momentum_5"] = df["volume"].pct_change(5)
        df["volume_momentum_20"] = df["volume"].pct_change(20)
        
        # Volume volatility
        df["volume_volatility_20"] = df["volume"].rolling(20).std()
        df["volume_cv_20"] = (
            df["volume_volatility_20"] / (df["volume_sma_20"] + 1e-10)
        )
        
        return df
    
    def calculate_vwap(self, df: pd.DataFrame, periods: List[int] = None) -> pd.DataFrame:
        """
        Calculate Volume Weighted Average Price.
        
        Args:
            df: DataFrame with OHLCV data
            periods: VWAP periods
            
        Returns:
            DataFrame with VWAP features
        """
        periods = periods or [20, 50, 100]
        
        for period in periods:
            df[f"vwap_{period}"] = (
                (df["close"] * df["volume"]).rolling(period).sum() /
                df["volume"].rolling(period).sum()
            )
            df[f"vwap_distance_{period}"] = (
                (df["close"] - df[f"vwap_{period}"]) / 
                (df[f"vwap_{period}"] + 1e-10)
            )
        
        return df
    
    def calculate_spread_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate spread-based features.
        
        Args:
            df: DataFrame with OHLC data
            
        Returns:
            DataFrame with spread features
        """
        # High-Low spread
        df["hl_spread"] = (df["high"] - df["low"]) / (df["close"] + 1e-10)
        df["hl_spread_20"] = df["hl_spread"].rolling(20).mean()
        df["hl_spread_50"] = df["hl_spread"].rolling(50).mean()
        
        # Open-Close spread
        df["oc_spread"] = np.abs(df["open"] - df["close"]) / (df["close"] + 1e-10)
        df["oc_spread_20"] = df["oc_spread"].rolling(20).mean()
        
        # Close position in bar
        df["close_position"] = (
            (df["close"] - df["low"]) / 
            (df["high"] - df["low"] + 1e-10)
        )
        
        # Bar type (bullish/bearish/neutral)
        df["bullish_bar"] = (df["close"] > df["open"]).astype(int)
        df["bearish_bar"] = (df["close"] < df["open"]).astype(int)
        df["neutral_bar"] = (df["close"] == df["open"]).astype(int)
        
        return df
    
    def calculate_liquidity_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate liquidity-based features.
        
        Args:
            df: DataFrame with OHLCV and returns data
            
        Returns:
            DataFrame with liquidity features
        """
        if "returns" not in df.columns:
            raise ValueError("DataFrame must contain 'returns' column")
        
        # Amihud illiquidity measure
        df["amihud_illiq"] = (
            np.abs(df["returns"]) / (df["dollar_volume"] + 1e-10)
        )
        df["amihud_illiq_20"] = df["amihud_illiq"].rolling(20).mean()
        df["amihud_illiq_50"] = df["amihud_illiq"].rolling(50).mean()
        
        # Kyle's lambda (simplified version)
        for period in [20, 50]:
            df[f"kyle_lambda_{period}"] = (
                df["returns"].rolling(period).std() /
                (df["volume"].rolling(period).std() + 1e-10)
            )
        
        # Roll's measure (simplified)
        df["roll_measure"] = 2 * np.sqrt(np.abs(
            df["returns"].rolling(20).cov(df["returns"].shift(1))
        ))
        
        return df
    
    def calculate_order_flow_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate order flow imbalance features.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with order flow features
        """
        # Buy/Sell pressure proxy (using close position)
        df["buy_pressure"] = df["close_position"]
        df["sell_pressure"] = 1 - df["close_position"]
        
        # Order flow imbalance
        df["order_flow_imbalance"] = df["buy_pressure"] - df["sell_pressure"]
        df["ofi_20"] = df["order_flow_imbalance"].rolling(20).mean()
        
        # Volume-weighted order flow
        df["volume_weighted_ofi"] = (
            df["order_flow_imbalance"] * df["volume"]
        )
        df["vw_ofi_20"] = df["volume_weighted_ofi"].rolling(20).sum()
        
        # Cumulative order flow
        df["cumulative_ofi"] = df["order_flow_imbalance"].cumsum()
        
        return df
    
    def calculate_regime_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate market regime features.
        
        Args:
            df: DataFrame with price and volume data
            
        Returns:
            DataFrame with regime features
        """
        # Price trend using linear regression
        price_trend = df["close"].rolling(50).apply(
            lambda x: stats.linregress(range(len(x)), x)[0]
            if len(x) == 50 else np.nan
        )
        
        # Volume trend
        volume_trend = df["volume"].rolling(50).apply(
            lambda x: stats.linregress(range(len(x)), x)[0]
            if len(x) == 50 else np.nan
        )
        
        # Market phases
        df["accumulation_phase"] = (
            (price_trend < 0) & (volume_trend > 0)
        ).astype(int)
        df["markup_phase"] = (
            (price_trend > 0) & (volume_trend > 0)
        ).astype(int)
        df["distribution_phase"] = (
            (price_trend > 0) & (volume_trend < 0)
        ).astype(int)
        df["markdown_phase"] = (
            (price_trend < 0) & (volume_trend < 0)
        ).astype(int)
        
        # Momentum regime
        if "returns" in df.columns:
            mom_20 = df["returns"].rolling(20).sum()
            df["momentum_percentile"] = mom_20.rolling(252).rank(pct=True)
            df["high_momentum"] = (df["momentum_percentile"] > 0.8).astype(int)
            df["low_momentum"] = (df["momentum_percentile"] < 0.2).astype(int)
        
        return df
    
    def calculate_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all microstructure features.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with all microstructure features
        """
        df = df.copy()
        
        df = self.calculate_volume_features(df)
        df = self.calculate_vwap(df)
        df = self.calculate_spread_features(df)
        
        # Only calculate if we have dollar_volume
        if "dollar_volume" in df.columns:
            df = self.calculate_liquidity_features(df)
        
        # Only calculate if we have close_position
        if "close_position" in df.columns:
            df = self.calculate_order_flow_features(df)
        
        df = self.calculate_regime_features(df)
        
        return df