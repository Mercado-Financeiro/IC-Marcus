"""Technical indicators feature engineering."""

import pandas as pd
import numpy as np
import ta
from typing import List
import structlog

log = structlog.get_logger()


class TechnicalIndicators:
    """Calculate technical indicators for feature engineering."""
    
    def __init__(self, indicators: List[str] = None):
        """
        Initialize technical indicators calculator.
        
        Args:
            indicators: List of indicators to calculate
        """
        self.indicators = indicators or [
            "rsi", "macd", "bbands", "atr", "obv", "adx", "cci", "stoch"
        ]
    
    def calculate_rsi(self, df: pd.DataFrame, periods: List[int] = None) -> pd.DataFrame:
        """
        Calculate RSI indicators.
        
        Args:
            df: DataFrame with OHLCV data
            periods: RSI periods to calculate
            
        Returns:
            DataFrame with RSI features
        """
        periods = periods or [7, 14, 21]
        
        for period in periods:
            df[f"rsi_{period}"] = ta.momentum.RSIIndicator(
                df["close"], window=period
            ).rsi()
            
            # RSI overbought/oversold signals
            df[f"rsi_{period}_overbought"] = (df[f"rsi_{period}"] > 70).astype(int)
            df[f"rsi_{period}_oversold"] = (df[f"rsi_{period}"] < 30).astype(int)
        
        return df
    
    def calculate_macd(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate MACD indicators.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with MACD features
        """
        macd = ta.trend.MACD(df["close"])
        df["macd"] = macd.macd()
        df["macd_signal"] = macd.macd_signal()
        df["macd_diff"] = macd.macd_diff()
        
        # MACD crossover signal
        df["macd_crossover"] = (
            (df["macd"] > df["macd_signal"]).astype(int) -
            (df["macd"].shift(1) > df["macd_signal"].shift(1)).astype(int)
        )
        
        return df
    
    def calculate_bollinger_bands(self, df: pd.DataFrame, periods: List[int] = None) -> pd.DataFrame:
        """
        Calculate Bollinger Bands indicators.
        
        Args:
            df: DataFrame with OHLCV data
            periods: BB periods to calculate
            
        Returns:
            DataFrame with Bollinger Bands features
        """
        periods = periods or [20, 50]
        
        for period in periods:
            bb = ta.volatility.BollingerBands(
                df["close"], window=period, window_dev=2
            )
            df[f"bb_high_{period}"] = bb.bollinger_hband()
            df[f"bb_low_{period}"] = bb.bollinger_lband()
            df[f"bb_mid_{period}"] = bb.bollinger_mavg()
            
            # Band width and position
            df[f"bb_width_{period}"] = (
                df[f"bb_high_{period}"] - df[f"bb_low_{period}"]
            )
            df[f"bb_position_{period}"] = (
                (df["close"] - df[f"bb_low_{period}"]) /
                (df[f"bb_width_{period}"] + 1e-10)
            )
            
            # Bollinger squeeze indicator
            df[f"bb_squeeze_{period}"] = (
                df[f"bb_width_{period}"] /
                df[f"bb_width_{period}"].rolling(100).mean()
            )
        
        return df
    
    def calculate_atr(self, df: pd.DataFrame, periods: List[int] = None) -> pd.DataFrame:
        """
        Calculate ATR (Average True Range).
        
        Args:
            df: DataFrame with OHLCV data
            periods: ATR periods to calculate
            
        Returns:
            DataFrame with ATR features
        """
        periods = periods or [7, 14, 21]
        
        for period in periods:
            df[f"atr_{period}"] = ta.volatility.AverageTrueRange(
                df["high"], df["low"], df["close"], window=period
            ).average_true_range()
            
            # ATR as percentage of price
            df[f"atr_pct_{period}"] = df[f"atr_{period}"] / df["close"]
        
        return df
    
    def calculate_obv(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate On Balance Volume indicators.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with OBV features
        """
        df["obv"] = ta.volume.OnBalanceVolumeIndicator(
            df["close"], df["volume"]
        ).on_balance_volume()
        
        # OBV momentum
        for period in [10, 20]:
            df[f"obv_momentum_{period}"] = df["obv"].pct_change(period)
        
        return df
    
    def calculate_adx(self, df: pd.DataFrame, periods: List[int] = None) -> pd.DataFrame:
        """
        Calculate ADX (Average Directional Index).
        
        Args:
            df: DataFrame with OHLCV data
            periods: ADX periods to calculate
            
        Returns:
            DataFrame with ADX features
        """
        periods = periods or [14, 21]
        
        for period in periods:
            adx = ta.trend.ADXIndicator(
                df["high"], df["low"], df["close"], window=period
            )
            df[f"adx_{period}"] = adx.adx()
            df[f"adx_pos_{period}"] = adx.adx_pos()
            df[f"adx_neg_{period}"] = adx.adx_neg()
            
            # Trend strength
            df[f"trend_strength_{period}"] = (df[f"adx_{period}"] > 25).astype(int)
        
        return df
    
    def calculate_cci(self, df: pd.DataFrame, periods: List[int] = None) -> pd.DataFrame:
        """
        Calculate CCI (Commodity Channel Index).
        
        Args:
            df: DataFrame with OHLCV data
            periods: CCI periods to calculate
            
        Returns:
            DataFrame with CCI features
        """
        periods = periods or [14, 20]
        
        for period in periods:
            df[f"cci_{period}"] = ta.trend.CCIIndicator(
                df["high"], df["low"], df["close"], window=period
            ).cci()
            
            # CCI overbought/oversold
            df[f"cci_{period}_overbought"] = (df[f"cci_{period}"] > 100).astype(int)
            df[f"cci_{period}_oversold"] = (df[f"cci_{period}"] < -100).astype(int)
        
        return df
    
    def calculate_stochastic(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Stochastic Oscillator.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with Stochastic features
        """
        stoch = ta.momentum.StochasticOscillator(
            df["high"], df["low"], df["close"]
        )
        df["stoch_k"] = stoch.stoch()
        df["stoch_d"] = stoch.stoch_signal()
        
        # Stochastic crossover
        df["stoch_crossover"] = (
            (df["stoch_k"] > df["stoch_d"]).astype(int) -
            (df["stoch_k"].shift(1) > df["stoch_d"].shift(1)).astype(int)
        )
        
        # Overbought/oversold
        df["stoch_overbought"] = (df["stoch_k"] > 80).astype(int)
        df["stoch_oversold"] = (df["stoch_k"] < 20).astype(int)
        
        return df
    
    def calculate_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all selected technical indicators.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with all technical indicator features
        """
        df = df.copy()
        
        if "rsi" in self.indicators:
            df = self.calculate_rsi(df)
        
        if "macd" in self.indicators:
            df = self.calculate_macd(df)
        
        if "bbands" in self.indicators:
            df = self.calculate_bollinger_bands(df)
        
        if "atr" in self.indicators:
            df = self.calculate_atr(df)
        
        if "obv" in self.indicators:
            df = self.calculate_obv(df)
        
        if "adx" in self.indicators:
            df = self.calculate_adx(df)
        
        if "cci" in self.indicators:
            df = self.calculate_cci(df)
        
        if "stoch" in self.indicators:
            df = self.calculate_stochastic(df)
        
        return df