"""
Microstructure features for cryptocurrency trading.

Implements order book imbalance, VPIN, Kyle's Lambda and other
microstructure indicators for better price discovery and informed trading detection.
"""

import numpy as np
import pandas as pd
from typing import Optional, Tuple, Dict, Any
import warnings

# Optional numba for performance
try:
    from numba import jit
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    # Dummy decorator if numba not available
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator


class MicrostructureFeatures:
    """Calculate microstructure features from order book and trade data."""
    
    def __init__(self, lookback_periods: list = [10, 30, 60]):
        """
        Initialize microstructure feature calculator.
        
        Args:
            lookback_periods: List of lookback periods for rolling calculations
        """
        self.lookback_periods = lookback_periods
    
    def order_book_imbalance(
        self,
        bid_volume: pd.Series,
        ask_volume: pd.Series,
        levels: int = 5
    ) -> pd.DataFrame:
        """
        Calculate order book imbalance at different levels.
        
        Order Book Imbalance (OBI) = (Bid Volume - Ask Volume) / (Bid Volume + Ask Volume)
        
        Args:
            bid_volume: Total bid volume up to level
            ask_volume: Total ask volume up to level
            levels: Number of order book levels to consider
            
        Returns:
            DataFrame with OBI features
        """
        features = pd.DataFrame(index=bid_volume.index)
        
        # Basic imbalance
        total_volume = bid_volume + ask_volume + 1e-10  # Avoid division by zero
        obi = (bid_volume - ask_volume) / total_volume
        features['obi'] = obi
        
        # Imbalance momentum
        features['obi_momentum'] = obi - obi.shift(1)
        
        # Rolling statistics
        for period in self.lookback_periods:
            features[f'obi_mean_{period}'] = obi.rolling(period).mean()
            features[f'obi_std_{period}'] = obi.rolling(period).std()
            features[f'obi_skew_{period}'] = obi.rolling(period).skew()
            
            # Imbalance persistence
            features[f'obi_autocorr_{period}'] = obi.rolling(period).apply(
                lambda x: x.autocorr() if len(x) > 1 else 0
            )
        
        # Weighted imbalance (more weight to top levels)
        weights = np.exp(-np.arange(levels) / 2)  # Exponential decay
        if len(bid_volume.shape) > 1:  # Multiple levels
            weighted_bid = (bid_volume * weights).sum(axis=1)
            weighted_ask = (ask_volume * weights).sum(axis=1)
            weighted_total = weighted_bid + weighted_ask + 1e-10
            features['obi_weighted'] = (weighted_bid - weighted_ask) / weighted_total
        
        return features
    
    def vpin(
        self,
        price: pd.Series,
        volume: pd.Series,
        bucket_size: Optional[float] = None,
        n_buckets: int = 50
    ) -> pd.Series:
        """
        Calculate Volume-Synchronized Probability of Informed Trading (VPIN).
        
        VPIN measures the probability of informed trading based on volume imbalance.
        
        Args:
            price: Price series
            volume: Volume series
            bucket_size: Size of each volume bucket (auto-calculated if None)
            n_buckets: Number of buckets for VPIN calculation
            
        Returns:
            VPIN series
        """
        # Calculate returns
        returns = price.pct_change()
        
        # Determine bucket size
        if bucket_size is None:
            total_volume = volume.sum()
            bucket_size = total_volume / (len(volume) / n_buckets)
        
        # Initialize arrays
        buy_volume = []
        sell_volume = []
        current_bucket_volume = 0
        current_buy = 0
        current_sell = 0
        
        # Classify volume as buy or sell using tick rule
        for i in range(1, len(price)):
            vol = volume.iloc[i]
            
            # Classify based on price movement
            if returns.iloc[i] > 0:
                current_buy += vol
            elif returns.iloc[i] < 0:
                current_sell += vol
            else:
                # Split equally if no price change
                current_buy += vol / 2
                current_sell += vol / 2
            
            current_bucket_volume += vol
            
            # Check if bucket is full
            if current_bucket_volume >= bucket_size:
                buy_volume.append(current_buy)
                sell_volume.append(current_sell)
                current_bucket_volume = 0
                current_buy = 0
                current_sell = 0
        
        # Calculate VPIN
        if len(buy_volume) < n_buckets:
            # Not enough data, return NaN series
            return pd.Series(np.nan, index=price.index)
        
        buy_volume = np.array(buy_volume)
        sell_volume = np.array(sell_volume)
        
        vpin_values = []
        for i in range(n_buckets, len(buy_volume) + 1):
            bucket_buy = buy_volume[i-n_buckets:i]
            bucket_sell = sell_volume[i-n_buckets:i]
            
            total_volume = bucket_buy.sum() + bucket_sell.sum() + 1e-10
            vpin = np.abs(bucket_buy.sum() - bucket_sell.sum()) / total_volume
            vpin_values.append(vpin)
        
        # Align with original index
        vpin_series = pd.Series(np.nan, index=price.index)
        if vpin_values:
            # Distribute VPIN values across the original index
            step = len(price) // len(vpin_values)
            for i, vpin_val in enumerate(vpin_values):
                start_idx = i * step
                end_idx = min((i + 1) * step, len(price))
                vpin_series.iloc[start_idx:end_idx] = vpin_val
        
        # Forward fill to handle any gaps
        vpin_series = vpin_series.ffill()
        
        return vpin_series
    
    def kyles_lambda(
        self,
        price: pd.Series,
        volume: pd.Series,
        window: int = 60
    ) -> pd.Series:
        """
        Calculate Kyle's Lambda - price impact coefficient.
        
        Measures the price impact of trades, indicating market depth and liquidity.
        
        Args:
            price: Price series
            volume: Volume series  
            window: Rolling window for regression
            
        Returns:
            Kyle's Lambda series
        """
        # Calculate returns and signed volume
        returns = price.pct_change()
        signed_volume = volume * np.sign(returns)
        
        # Rolling regression to estimate lambda
        lambda_series = pd.Series(np.nan, index=price.index)
        
        for i in range(window, len(price)):
            window_returns = returns.iloc[i-window:i].values
            window_volume = signed_volume.iloc[i-window:i].values
            
            # Remove NaN values
            mask = ~(np.isnan(window_returns) | np.isnan(window_volume))
            clean_returns = window_returns[mask]
            clean_volume = window_volume[mask]
            
            if len(clean_returns) > window // 2:  # Need sufficient data
                # Simple linear regression: returns = lambda * signed_volume
                if clean_volume.std() > 0:
                    lambda_est = np.cov(clean_returns, clean_volume)[0, 1] / np.var(clean_volume)
                    lambda_series.iloc[i] = abs(lambda_est)  # Use absolute value
        
        return lambda_series
    
    def roll_measure(
        self,
        price: pd.Series,
        window: int = 60
    ) -> pd.Series:
        """
        Calculate Roll's implied spread measure.
        
        Estimates the bid-ask spread from price changes.
        
        Args:
            price: Price series
            window: Rolling window
            
        Returns:
            Roll measure series
        """
        # Calculate price changes
        price_changes = price.diff()
        
        # Rolling calculation
        roll_series = pd.Series(np.nan, index=price.index)
        
        for i in range(window, len(price)):
            window_changes = price_changes.iloc[i-window:i]
            
            # Calculate autocovariance
            cov = window_changes.cov(window_changes.shift(1))
            
            if cov < 0:  # Valid Roll measure
                roll_series.iloc[i] = 2 * np.sqrt(-cov)
            else:
                # Use 0 when covariance is positive (no spread detected)
                roll_series.iloc[i] = 0
        
        return roll_series
    
    def amihud_illiquidity(
        self,
        returns: pd.Series,
        volume: pd.Series,
        dollar_volume: pd.Series,
        window: int = 30
    ) -> pd.Series:
        """
        Calculate Amihud illiquidity measure.
        
        Measures price impact per unit of trading volume.
        
        Args:
            returns: Return series
            volume: Volume series
            dollar_volume: Dollar volume series
            window: Rolling window
            
        Returns:
            Amihud illiquidity series
        """
        # Calculate absolute returns over dollar volume
        illiquidity = np.abs(returns) / (dollar_volume + 1e-10)
        
        # Rolling average
        amihud = illiquidity.rolling(window).mean()
        
        return amihud
    
    def calculate_all_features(
        self,
        df: pd.DataFrame,
        bid_volume_col: Optional[str] = None,
        ask_volume_col: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Calculate all microstructure features.
        
        Args:
            df: DataFrame with OHLCV data
            bid_volume_col: Column name for bid volume (optional)
            ask_volume_col: Column name for ask volume (optional)
            
        Returns:
            DataFrame with all microstructure features
        """
        features = pd.DataFrame(index=df.index)
        
        # Order book imbalance (if bid/ask volumes available)
        if bid_volume_col and ask_volume_col and bid_volume_col in df.columns:
            obi_features = self.order_book_imbalance(
                df[bid_volume_col],
                df[ask_volume_col]
            )
            features = pd.concat([features, obi_features], axis=1)
        
        # VPIN
        if 'close' in df.columns and 'volume' in df.columns:
            features['vpin'] = self.vpin(df['close'], df['volume'])
            
            # VPIN momentum
            features['vpin_momentum'] = features['vpin'].diff()
            
            # VPIN percentile
            for period in self.lookback_periods:
                features[f'vpin_pct_{period}'] = features['vpin'].rolling(period).rank(pct=True)
        
        # Kyle's Lambda
        if 'close' in df.columns and 'volume' in df.columns:
            for period in self.lookback_periods:
                features[f'kyles_lambda_{period}'] = self.kyles_lambda(
                    df['close'], 
                    df['volume'],
                    window=period
                )
        
        # Roll measure
        if 'close' in df.columns:
            for period in self.lookback_periods:
                features[f'roll_spread_{period}'] = self.roll_measure(
                    df['close'],
                    window=period
                )
        
        # Amihud illiquidity
        if 'close' in df.columns and 'volume' in df.columns:
            returns = df['close'].pct_change()
            dollar_volume = df['close'] * df['volume']
            
            for period in self.lookback_periods:
                features[f'amihud_{period}'] = self.amihud_illiquidity(
                    returns,
                    df['volume'],
                    dollar_volume,
                    window=period
                )
        
        # Realized volatility at different frequencies
        if 'close' in df.columns:
            returns = df['close'].pct_change()
            for period in self.lookback_periods:
                features[f'realized_vol_{period}'] = returns.rolling(period).std()
                features[f'realized_skew_{period}'] = returns.rolling(period).skew()
                features[f'realized_kurt_{period}'] = returns.rolling(period).kurt()
        
        # Volume features
        if 'volume' in df.columns:
            for period in self.lookback_periods:
                # Volume momentum
                features[f'volume_momentum_{period}'] = (
                    df['volume'].rolling(period).mean() / 
                    df['volume'].rolling(period * 2).mean()
                )
                
                # Volume concentration (Herfindahl index proxy)
                features[f'volume_concentration_{period}'] = (
                    df['volume'].rolling(period).apply(
                        lambda x: np.sum((x / x.sum()) ** 2) if x.sum() > 0 else 0
                    )
                )
        
        # Trade intensity
        if 'volume' in df.columns and 'close' in df.columns:
            # Trades per price movement
            price_changes = df['close'].pct_change().abs()
            features['trade_intensity'] = df['volume'] / (price_changes + 1e-10)
            
            for period in self.lookback_periods:
                features[f'trade_intensity_mean_{period}'] = (
                    features['trade_intensity'].rolling(period).mean()
                )
        
        return features


@jit(nopython=True)
def fast_order_book_imbalance(bid_volumes: np.ndarray, ask_volumes: np.ndarray) -> np.ndarray:
    """
    Fast numba implementation of order book imbalance calculation.
    
    Args:
        bid_volumes: Array of bid volumes
        ask_volumes: Array of ask volumes
        
    Returns:
        Array of order book imbalances
    """
    total = bid_volumes + ask_volumes + 1e-10
    return (bid_volumes - ask_volumes) / total


def calculate_microstructure_features(
    df: pd.DataFrame,
    lookback_periods: list = [10, 30, 60],
    include_order_book: bool = False
) -> pd.DataFrame:
    """
    Convenience function to calculate all microstructure features.
    
    Args:
        df: DataFrame with OHLCV data
        lookback_periods: List of lookback periods
        include_order_book: Whether to include order book features
        
    Returns:
        DataFrame with microstructure features
    """
    calculator = MicrostructureFeatures(lookback_periods)
    
    # Determine bid/ask columns if available
    bid_col = 'bid_volume' if include_order_book and 'bid_volume' in df.columns else None
    ask_col = 'ask_volume' if include_order_book and 'ask_volume' in df.columns else None
    
    features = calculator.calculate_all_features(df, bid_col, ask_col)
    
    # Add prefix to all features
    features.columns = [f'micro_{col}' for col in features.columns]
    
    return features