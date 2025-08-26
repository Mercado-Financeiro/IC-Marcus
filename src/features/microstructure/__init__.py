"""Market microstructure features - modular architecture.

This module provides both specialized feature classes for focused analysis
and a unified interface for backward compatibility.

Specialized Classes (New Modular Approach):
    - VolumeFeatures: Volume-based microstructure analysis
    - SpreadFeatures: Bid-ask spread and intrabar price analysis
    - LiquidityFeatures: Market liquidity measures (Amihud, Kyle's lambda, Roll)
    - OrderFlowFeatures: Order flow imbalance and buy/sell pressure
    - RegimeFeatures: Market regime identification and trend analysis
    - VWAPFeatures: VWAP-based institutional flow analysis

Unified Interface (Backward Compatibility):
    - MicrostructureFeatures: Original monolithic interface using modular backend

Example Usage:

    # New modular approach (recommended)
    from src.features.microstructure import VolumeFeatures, LiquidityFeatures
    volume_calc = VolumeFeatures([10, 20, 50])
    df = volume_calc.calculate_all(df)
    
    # Backward compatible approach
    from src.features.microstructure import MicrostructureFeatures
    micro = MicrostructureFeatures()
    df = micro.calculate_all(df)
"""

# Import all specialized feature classes
from .volume import VolumeFeatures
from .spread import SpreadFeatures  
from .liquidity import LiquidityFeatures
from .order_flow import OrderFlowFeatures
from .regime import RegimeFeatures
from .vwap import VWAPFeatures

# Backward compatibility - maintain original interface
from .microstructure_unified import MicrostructureFeatures

__all__ = [
    'VolumeFeatures',
    'SpreadFeatures', 
    'LiquidityFeatures',
    'OrderFlowFeatures',
    'RegimeFeatures',
    'VWAPFeatures',
    'MicrostructureFeatures'  # Backward compatibility
]

# Version information
__version__ = '2.0.0'
__author__ = 'ML Finance Team'