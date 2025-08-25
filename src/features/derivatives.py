"""
Derivatives and funding rate features for cryptocurrency trading.

Implements features from perpetual futures markets including funding rates,
open interest, basis, and term structure.
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, List, Tuple
import requests
from datetime import datetime, timedelta
import json


class DerivativesFeatures:
    """Calculate features from derivatives markets."""
    
    def __init__(self, symbol: str = "BTCUSDT"):
        """
        Initialize derivatives feature calculator.
        
        Args:
            symbol: Trading symbol
        """
        self.symbol = symbol
        self.base_url = "https://fapi.binance.com"  # Futures API
        
    def fetch_funding_rate(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 1000
    ) -> pd.DataFrame:
        """
        Fetch historical funding rates from Binance.
        
        Args:
            start_time: Start time for data
            end_time: End time for data
            limit: Maximum number of records
            
        Returns:
            DataFrame with funding rates
        """
        endpoint = f"{self.base_url}/fapi/v1/fundingRate"
        
        params = {
            "symbol": self.symbol,
            "limit": limit
        }
        
        if start_time:
            params["startTime"] = int(start_time.timestamp() * 1000)
        if end_time:
            params["endTime"] = int(end_time.timestamp() * 1000)
        
        try:
            response = requests.get(endpoint, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if data:
                df = pd.DataFrame(data)
                df['fundingTime'] = pd.to_datetime(df['fundingTime'], unit='ms')
                df['fundingRate'] = df['fundingRate'].astype(float)
                df.set_index('fundingTime', inplace=True)
                return df[['fundingRate']]
            
        except Exception as e:
            print(f"Error fetching funding rate: {e}")
        
        return pd.DataFrame()
    
    def fetch_open_interest(
        self,
        interval: str = "5m",
        limit: int = 500
    ) -> pd.DataFrame:
        """
        Fetch open interest history from Binance.
        
        Args:
            interval: Time interval (5m, 15m, 30m, 1h, etc.)
            limit: Maximum number of records
            
        Returns:
            DataFrame with open interest
        """
        endpoint = f"{self.base_url}/futures/data/openInterestHist"
        
        params = {
            "symbol": self.symbol,
            "period": interval,
            "limit": limit
        }
        
        try:
            response = requests.get(endpoint, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if data:
                df = pd.DataFrame(data)
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df['sumOpenInterest'] = df['sumOpenInterest'].astype(float)
                df['sumOpenInterestValue'] = df['sumOpenInterestValue'].astype(float)
                df.set_index('timestamp', inplace=True)
                return df[['sumOpenInterest', 'sumOpenInterestValue']]
            
        except Exception as e:
            print(f"Error fetching open interest: {e}")
        
        return pd.DataFrame()
    
    def calculate_funding_features(
        self,
        funding_rates: pd.Series,
        lookback_periods: List[int] = [8, 24, 72]
    ) -> pd.DataFrame:
        """
        Calculate features from funding rates.
        
        Args:
            funding_rates: Series of funding rates
            lookback_periods: Periods for rolling calculations (in funding periods, usually 8h)
            
        Returns:
            DataFrame with funding rate features
        """
        features = pd.DataFrame(index=funding_rates.index)
        
        # Current funding rate
        features['funding_rate'] = funding_rates
        
        # Cumulative funding over periods
        for period in lookback_periods:
            features[f'funding_cumsum_{period}'] = funding_rates.rolling(period).sum()
            features[f'funding_mean_{period}'] = funding_rates.rolling(period).mean()
            features[f'funding_std_{period}'] = funding_rates.rolling(period).std()
            
            # Funding momentum
            features[f'funding_momentum_{period}'] = (
                funding_rates.rolling(period).mean() - 
                funding_rates.rolling(period * 2).mean()
            )
            
            # Funding regime (consistently positive/negative)
            features[f'funding_positive_pct_{period}'] = (
                (funding_rates > 0).rolling(period).mean()
            )
        
        # Funding rate changes
        features['funding_change'] = funding_rates.diff()
        features['funding_change_abs'] = funding_rates.diff().abs()
        
        # Extreme funding indicators
        features['funding_extreme_high'] = (
            funding_rates > funding_rates.rolling(72).quantile(0.95)
        ).astype(int)
        
        features['funding_extreme_low'] = (
            funding_rates < funding_rates.rolling(72).quantile(0.05)
        ).astype(int)
        
        # APR equivalent (funding rate * 3 * 365 for 8h funding periods)
        features['funding_apr'] = funding_rates * 3 * 365
        
        return features
    
    def calculate_open_interest_features(
        self,
        open_interest: pd.Series,
        price: pd.Series,
        volume: pd.Series,
        lookback_periods: List[int] = [24, 96, 288]
    ) -> pd.DataFrame:
        """
        Calculate features from open interest.
        
        Args:
            open_interest: Series of open interest
            price: Price series
            volume: Volume series
            lookback_periods: Periods for rolling calculations
            
        Returns:
            DataFrame with open interest features
        """
        features = pd.DataFrame(index=open_interest.index)
        
        # Basic OI features
        features['oi'] = open_interest
        features['oi_change'] = open_interest.diff()
        features['oi_pct_change'] = open_interest.pct_change()
        
        # OI to volume ratio
        features['oi_volume_ratio'] = open_interest / (volume + 1e-10)
        
        # OI weighted by price changes
        price_change = price.pct_change()
        features['oi_price_weighted'] = open_interest * price_change
        
        for period in lookback_periods:
            # OI statistics
            features[f'oi_mean_{period}'] = open_interest.rolling(period).mean()
            features[f'oi_std_{period}'] = open_interest.rolling(period).std()
            
            # OI momentum
            features[f'oi_momentum_{period}'] = (
                open_interest.rolling(period).mean() / 
                open_interest.rolling(period * 2).mean() - 1
            )
            
            # OI percentile
            features[f'oi_percentile_{period}'] = (
                open_interest.rolling(period).rank(pct=True)
            )
            
            # OI-price divergence
            oi_change_sum = open_interest.diff().rolling(period).sum()
            price_change_sum = price.pct_change().rolling(period).sum()
            features[f'oi_price_divergence_{period}'] = (
                oi_change_sum * np.sign(price_change_sum) * -1
            )
        
        # Long/Short ratio proxy (using OI changes and price movements)
        oi_increase = (open_interest.diff() > 0).astype(int)
        price_increase = (price.diff() > 0).astype(int)
        
        # Approximate long/short positioning
        features['long_positioning'] = oi_increase * price_increase
        features['short_positioning'] = oi_increase * (1 - price_increase)
        
        # Liquidation risk indicator (high OI + high volatility)
        volatility = price.pct_change().rolling(24).std()
        features['liquidation_risk'] = (
            (open_interest > open_interest.rolling(96).quantile(0.8)).astype(int) *
            (volatility > volatility.rolling(96).quantile(0.8)).astype(int)
        )
        
        return features
    
    def calculate_basis_features(
        self,
        spot_price: pd.Series,
        futures_price: pd.Series,
        time_to_expiry: Optional[float] = None
    ) -> pd.DataFrame:
        """
        Calculate basis and term structure features.
        
        Args:
            spot_price: Spot price series
            futures_price: Futures price series
            time_to_expiry: Time to expiry in days (for futures, not perpetuals)
            
        Returns:
            DataFrame with basis features
        """
        features = pd.DataFrame(index=spot_price.index)
        
        # Raw basis
        basis = futures_price - spot_price
        features['basis'] = basis
        features['basis_pct'] = basis / spot_price
        
        # Annualized basis (for perpetuals, use 365 days)
        if time_to_expiry:
            features['basis_annualized'] = features['basis_pct'] * (365 / time_to_expiry)
        else:
            # For perpetuals, use simple annualization
            features['basis_annualized'] = features['basis_pct'] * 365
        
        # Basis momentum
        features['basis_change'] = basis.diff()
        features['basis_momentum'] = basis.rolling(24).mean() - basis.rolling(96).mean()
        
        # Basis volatility
        features['basis_volatility'] = basis.rolling(96).std()
        
        # Contango/Backwardation indicator
        features['contango'] = (basis > 0).astype(int)
        features['backwardation'] = (basis < 0).astype(int)
        
        # Basis mean reversion
        basis_mean = basis.rolling(288).mean()
        basis_std = basis.rolling(288).std()
        features['basis_zscore'] = (basis - basis_mean) / (basis_std + 1e-10)
        
        # Basis extremes
        features['basis_extreme_high'] = (
            basis > basis.rolling(288).quantile(0.95)
        ).astype(int)
        
        features['basis_extreme_low'] = (
            basis < basis.rolling(288).quantile(0.05)
        ).astype(int)
        
        return features
    
    def calculate_all_features(
        self,
        df: pd.DataFrame,
        fetch_live_data: bool = False,
        lookback_periods: List[int] = [24, 96, 288]
    ) -> pd.DataFrame:
        """
        Calculate all derivatives features.
        
        Args:
            df: DataFrame with OHLCV data
            fetch_live_data: Whether to fetch live funding/OI data
            lookback_periods: Periods for rolling calculations
            
        Returns:
            DataFrame with all derivatives features
        """
        features = pd.DataFrame(index=df.index)
        
        if fetch_live_data:
            # Fetch funding rates
            start_time = df.index[0] if not df.empty else None
            end_time = df.index[-1] if not df.empty else None
            
            funding_df = self.fetch_funding_rate(start_time, end_time)
            if not funding_df.empty:
                # Resample to match main dataframe frequency
                funding_resampled = funding_df.resample(
                    pd.infer_freq(df.index) or '1H'
                ).last().ffill()
                
                # Align with main dataframe
                funding_aligned = funding_resampled.reindex(df.index).ffill()
                
                if 'fundingRate' in funding_aligned.columns:
                    funding_features = self.calculate_funding_features(
                        funding_aligned['fundingRate'],
                        lookback_periods=[p//3 for p in lookback_periods]  # Adjust for 8h periods
                    )
                    features = pd.concat([features, funding_features], axis=1)
            
            # Fetch open interest
            oi_df = self.fetch_open_interest()
            if not oi_df.empty and 'close' in df.columns and 'volume' in df.columns:
                # Resample and align
                oi_resampled = oi_df.resample(
                    pd.infer_freq(df.index) or '1H'
                ).last().ffill()
                
                oi_aligned = oi_resampled.reindex(df.index).ffill()
                
                if 'sumOpenInterest' in oi_aligned.columns:
                    oi_features = self.calculate_open_interest_features(
                        oi_aligned['sumOpenInterest'],
                        df['close'],
                        df['volume'],
                        lookback_periods
                    )
                    features = pd.concat([features, oi_features], axis=1)
        
        # Calculate synthetic features if no live data
        if features.empty and 'close' in df.columns and 'volume' in df.columns:
            # Create synthetic funding rate based on price momentum
            returns = df['close'].pct_change()
            synthetic_funding = returns.rolling(24).mean() * 0.01  # Scale down
            
            funding_features = self.calculate_funding_features(
                synthetic_funding,
                lookback_periods=[p//3 for p in lookback_periods]
            )
            features = pd.concat([features, funding_features], axis=1)
            
            # Create synthetic OI based on volume
            synthetic_oi = df['volume'].rolling(96).sum()
            
            oi_features = self.calculate_open_interest_features(
                synthetic_oi,
                df['close'],
                df['volume'],
                lookback_periods
            )
            features = pd.concat([features, oi_features], axis=1)
        
        # Add prefix to all features
        features.columns = [f'deriv_{col}' for col in features.columns]
        
        return features


def calculate_derivatives_features(
    df: pd.DataFrame,
    symbol: str = "BTCUSDT",
    fetch_live_data: bool = False,
    lookback_periods: List[int] = [24, 96, 288]
) -> pd.DataFrame:
    """
    Convenience function to calculate all derivatives features.
    
    Args:
        df: DataFrame with OHLCV data
        symbol: Trading symbol
        fetch_live_data: Whether to fetch live data
        lookback_periods: Periods for rolling calculations
        
    Returns:
        DataFrame with derivatives features
    """
    calculator = DerivativesFeatures(symbol)
    return calculator.calculate_all_features(df, fetch_live_data, lookback_periods)