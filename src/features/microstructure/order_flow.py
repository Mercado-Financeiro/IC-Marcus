"""Order flow microstructure features."""

import pandas as pd
import numpy as np
from typing import List


class OrderFlowFeatures:
    """
    Calculate order flow imbalance features.
    
    Focuses on buy/sell pressure indicators and order flow imbalance
    measures, providing insights into market microstructure dynamics
    and short-term price pressure.
    """
    
    def __init__(self, lookback_periods: List[int] = None):
        """
        Initialize order flow features calculator.
        
        Args:
            lookback_periods: Periods for rolling calculations
        """
        self.lookback_periods = lookback_periods or [10, 20, 50]
    
    def calculate_buy_sell_pressure(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate buy and sell pressure proxies.
        
        Uses close position within the bar as a proxy for buy/sell pressure:
        - Close near high = buying pressure
        - Close near low = selling pressure
        
        Args:
            df: DataFrame with close_position feature
            
        Returns:
            DataFrame with buy/sell pressure features
        """
        df = df.copy()
        
        # Validate required columns
        if "close_position" not in df.columns:
            raise ValueError("DataFrame must contain 'close_position' column")
        
        # Buy/Sell pressure proxy (using close position)
        df["buy_pressure"] = df["close_position"]
        df["sell_pressure"] = 1 - df["close_position"]
        
        return df
    
    def calculate_order_flow_imbalance(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate order flow imbalance measures.
        
        Order flow imbalance = Buy Pressure - Sell Pressure
        Range: [-1, 1] where 1 = all buying, -1 = all selling
        
        Args:
            df: DataFrame with buy_pressure and sell_pressure
            
        Returns:
            DataFrame with order flow imbalance features
        """
        df = df.copy()
        
        # Validate required columns
        if "buy_pressure" not in df.columns or "sell_pressure" not in df.columns:
            raise ValueError("DataFrame must contain buy_pressure and sell_pressure columns")
        
        # Order flow imbalance
        df["order_flow_imbalance"] = df["buy_pressure"] - df["sell_pressure"]
        
        # Rolling averages of OFI
        for period in self.lookback_periods:
            if period <= len(df):
                df[f"ofi_{period}"] = df["order_flow_imbalance"].rolling(period).mean()
        
        return df
    
    def calculate_volume_weighted_ofi(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate volume-weighted order flow features.
        
        Volume-weighted OFI gives more weight to periods with higher volume,
        providing a better measure of actual order flow impact.
        
        Args:
            df: DataFrame with order_flow_imbalance and volume
            
        Returns:
            DataFrame with volume-weighted OFI features
        """
        df = df.copy()
        
        # Validate required columns
        if "order_flow_imbalance" not in df.columns:
            raise ValueError("DataFrame must contain 'order_flow_imbalance' column")
        if "volume" not in df.columns:
            raise ValueError("DataFrame must contain 'volume' column")
        
        # Volume-weighted order flow
        df["volume_weighted_ofi"] = (
            df["order_flow_imbalance"] * df["volume"]
        )
        
        # Rolling sums of volume-weighted OFI
        for period in [20, 50]:
            if period <= len(df):
                df[f"vw_ofi_{period}"] = df["volume_weighted_ofi"].rolling(period).sum()
        
        return df
    
    def calculate_cumulative_ofi(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate cumulative order flow measures.
        
        Cumulative OFI tracks the persistent buy/sell pressure over time,
        helping identify sustained directional pressure.
        
        Args:
            df: DataFrame with order_flow_imbalance
            
        Returns:
            DataFrame with cumulative OFI features
        """
        df = df.copy()
        
        # Validate required columns
        if "order_flow_imbalance" not in df.columns:
            raise ValueError("DataFrame must contain 'order_flow_imbalance' column")
        
        # Cumulative order flow
        df["cumulative_ofi"] = df["order_flow_imbalance"].cumsum()
        
        # Rolling cumulative OFI (reset periodically)
        for period in [100, 200]:
            if period <= len(df):
                df[f"cumulative_ofi_{period}"] = (
                    df["order_flow_imbalance"]
                    .rolling(period, min_periods=1)
                    .sum()
                )
        
        return df
    
    def calculate_ofi_momentum(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate order flow momentum features.
        
        Args:
            df: DataFrame with order flow features
            
        Returns:
            DataFrame with OFI momentum features
        """
        df = df.copy()
        
        # OFI momentum (rate of change)
        if "order_flow_imbalance" in df.columns:
            for period in [5, 10, 20]:
                if period < len(df):
                    df[f"ofi_momentum_{period}"] = (
                        df["order_flow_imbalance"].pct_change(period, fill_method=None)
                    )
        
        # Cumulative OFI momentum
        if "cumulative_ofi" in df.columns:
            df["cumulative_ofi_momentum"] = df["cumulative_ofi"].pct_change(10, fill_method=None)
        
        return df
    
    def calculate_ofi_volatility(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate order flow volatility measures.
        
        Args:
            df: DataFrame with order flow features
            
        Returns:
            DataFrame with OFI volatility features
        """
        df = df.copy()
        
        # OFI volatility
        if "order_flow_imbalance" in df.columns:
            df["ofi_volatility_20"] = df["order_flow_imbalance"].rolling(20).std()
            df["ofi_volatility_50"] = df["order_flow_imbalance"].rolling(50).std()
        
        return df
    
    def calculate_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all order flow features.
        
        Args:
            df: DataFrame with OHLCV and close_position data
            
        Returns:
            DataFrame with all order flow features
        """
        df = df.copy()
        
        try:
            # Calculate features in logical order
            df = self.calculate_buy_sell_pressure(df)
            df = self.calculate_order_flow_imbalance(df)  # Depends on buy/sell pressure
            df = self.calculate_volume_weighted_ofi(df)  # Depends on OFI
            df = self.calculate_cumulative_ofi(df)       # Depends on OFI
            df = self.calculate_ofi_momentum(df)         # Depends on OFI
            df = self.calculate_ofi_volatility(df)       # Depends on OFI
        except ValueError as e:
            # If required columns are missing, skip order flow features
            print(f"Warning: Skipping order flow features due to missing data: {e}")
        
        return df
    
    def get_feature_names(self) -> List[str]:
        """
        Get list of feature names this class generates.
        
        Returns:
            List of feature column names
        """
        base_features = [
            "buy_pressure", "sell_pressure", "order_flow_imbalance",
            "volume_weighted_ofi", "cumulative_ofi",
            "cumulative_ofi_momentum", "ofi_volatility_20", "ofi_volatility_50"
        ]
        
        # Add period-based features
        period_features = []
        
        # OFI rolling averages
        for period in self.lookback_periods:
            period_features.append(f"ofi_{period}")
        
        # Volume-weighted OFI
        for period in [20, 50]:
            period_features.append(f"vw_ofi_{period}")
        
        # Cumulative OFI
        for period in [100, 200]:
            period_features.append(f"cumulative_ofi_{period}")
        
        # OFI momentum
        for period in [5, 10, 20]:
            period_features.append(f"ofi_momentum_{period}")
        
        return base_features + period_features