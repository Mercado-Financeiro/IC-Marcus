"""Unified microstructure features interface with backward compatibility."""

import pandas as pd
import numpy as np
from typing import List
import warnings

from .volume import VolumeFeatures
from .spread import SpreadFeatures
from .liquidity import LiquidityFeatures
from .order_flow import OrderFlowFeatures
from .regime import RegimeFeatures
from .vwap import VWAPFeatures


class MicrostructureFeatures:
    """
    Unified microstructure features calculator.
    
    Maintains backward compatibility with the original monolithic class
    while using the new modular architecture internally. This allows
    existing code to work without changes while benefiting from the
    improved structure.
    """
    
    def __init__(self, lookback_periods: List[int] = None):
        """
        Initialize microstructure features calculator.
        
        Args:
            lookback_periods: Periods for rolling calculations
        """
        self.lookback_periods = lookback_periods or [10, 20, 30, 50, 60]
        
        # Initialize all specialized feature calculators
        self.volume_features = VolumeFeatures(lookback_periods)
        self.spread_features = SpreadFeatures(lookback_periods)
        self.liquidity_features = LiquidityFeatures(lookback_periods)
        self.order_flow_features = OrderFlowFeatures(lookback_periods)
        self.regime_features = RegimeFeatures(lookback_periods)
        self.vwap_features = VWAPFeatures(lookback_periods)
    
    def calculate_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate volume-based features (backward compatibility).
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with volume features
        """
        return self.volume_features.calculate_all(df)
    
    def calculate_vwap(self, df: pd.DataFrame, periods: List[int] = None) -> pd.DataFrame:
        """
        Calculate Volume Weighted Average Price (backward compatibility).
        
        Args:
            df: DataFrame with OHLCV data
            periods: VWAP periods
            
        Returns:
            DataFrame with VWAP features
        """
        if periods is not None:
            # Temporarily override periods for backward compatibility
            original_periods = self.vwap_features.lookback_periods
            self.vwap_features.lookback_periods = periods
            result = self.vwap_features.calculate_vwap(df)
            self.vwap_features.lookback_periods = original_periods
            return result
        else:
            return self.vwap_features.calculate_vwap(df)
    
    def calculate_spread_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate spread-based features (backward compatibility).
        
        Args:
            df: DataFrame with OHLC data
            
        Returns:
            DataFrame with spread features
        """
        return self.spread_features.calculate_all(df)
    
    def calculate_liquidity_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate liquidity-based features (backward compatibility).
        
        Args:
            df: DataFrame with OHLCV and returns data
            
        Returns:
            DataFrame with liquidity features
        """
        return self.liquidity_features.calculate_all(df)
    
    def calculate_order_flow_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate order flow imbalance features (backward compatibility).
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with order flow features
        """
        return self.order_flow_features.calculate_all(df)
    
    def calculate_regime_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate market regime features (backward compatibility).
        
        Args:
            df: DataFrame with price and volume data
            
        Returns:
            DataFrame with regime features
        """
        return self.regime_features.calculate_all(df)
    
    def calculate_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all microstructure features.
        
        This method maintains the exact same interface as the original
        monolithic class while using the new modular architecture.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with all microstructure features
        """
        df = df.copy()
        
        # Calculate all feature types using specialized modules
        df = self.volume_features.calculate_all(df)
        df = self.vwap_features.calculate_all(df)
        df = self.spread_features.calculate_all(df)
        
        # Conditional features (only if prerequisites are met)
        if "dollar_volume" in df.columns and "returns" in df.columns:
            df = self.liquidity_features.calculate_all(df)
        
        if "close_position" in df.columns:
            df = self.order_flow_features.calculate_all(df)
        
        df = self.regime_features.calculate_all(df)
        
        return df
    
    def get_feature_names(self) -> List[str]:
        """
        Get all feature names from all modules.
        
        Returns:
            List of all feature column names
        """
        all_features = []
        
        # Collect feature names from all modules
        all_features.extend(self.volume_features.get_feature_names())
        all_features.extend(self.spread_features.get_feature_names())
        all_features.extend(self.liquidity_features.get_feature_names())
        all_features.extend(self.order_flow_features.get_feature_names())
        all_features.extend(self.regime_features.get_feature_names())
        all_features.extend(self.vwap_features.get_feature_names())
        
        # Remove duplicates while preserving order
        seen = set()
        unique_features = []
        for feature in all_features:
            if feature not in seen:
                unique_features.append(feature)
                seen.add(feature)
        
        return unique_features
    
    def get_feature_groups(self) -> dict:
        """
        Get features organized by functional groups.
        
        Returns:
            Dictionary mapping group names to feature lists
        """
        return {
            'volume': self.volume_features.get_feature_names(),
            'spread': self.spread_features.get_feature_names(),
            'liquidity': self.liquidity_features.get_feature_names(),
            'order_flow': self.order_flow_features.get_feature_names(),
            'regime': self.regime_features.get_feature_names(),
            'vwap': self.vwap_features.get_feature_names()
        }
    
    # Selective feature calculation methods for performance
    def calculate_volume_only(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate only volume features."""
        return self.volume_features.calculate_all(df.copy())
    
    def calculate_spread_only(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate only spread features."""
        return self.spread_features.calculate_all(df.copy())
    
    def calculate_liquidity_only(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate only liquidity features."""
        return self.liquidity_features.calculate_all(df.copy())
    
    def calculate_order_flow_only(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate only order flow features."""
        return self.order_flow_features.calculate_all(df.copy())
    
    def calculate_regime_only(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate only regime features."""
        return self.regime_features.calculate_all(df.copy())
    
    def calculate_vwap_only(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate only VWAP features."""
        return self.vwap_features.calculate_all(df.copy())


# Deprecated: Issue warning for direct import of old location
def __getattr__(name):
    """Handle deprecated imports with warnings."""
    if name == "MicrostructureFeatures":
        warnings.warn(
            "Importing MicrostructureFeatures from this location is deprecated. "
            "Use 'from src.features.microstructure import MicrostructureFeatures' instead.",
            DeprecationWarning,
            stacklevel=2
        )
        return MicrostructureFeatures
    raise AttributeError(f"module has no attribute '{name}'")