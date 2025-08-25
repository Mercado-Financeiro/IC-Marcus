"""Calendar and time-based feature calculations for 24/7 crypto markets."""

import pandas as pd
import numpy as np
from typing import Optional, Dict


class Crypto24x7Features:
    """
    Features specific for 24/7 crypto markets.
    
    Includes calendar, regional trading sessions, and funding cycles.
    Optimized for perpetual futures with funding every 8 hours.
    """
    
    @staticmethod
    def create_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Create comprehensive calendar features for 24/7 markets.
        
        Crypto doesn't have market closures, but has patterns:
        - High volume periods (regional market overlaps)
        - Day of week effects
        - End-of-month rebalancing
        
        Args:
            df: DataFrame with datetime index
            
        Returns:
            DataFrame with calendar features
        """
        features = pd.DataFrame(index=df.index)
        
        # Extract temporal components
        features['hour'] = df.index.hour
        features['day_of_week'] = df.index.dayofweek
        features['day_of_month'] = df.index.day
        features['week_of_year'] = df.index.isocalendar().week
        features['month'] = df.index.month
        features['quarter'] = df.index.quarter
        
        # Cyclical encoding (preserves periodicity)
        features['hour_sin'] = np.sin(2 * np.pi * features['hour'] / 24)
        features['hour_cos'] = np.cos(2 * np.pi * features['hour'] / 24)
        features['dow_sin'] = np.sin(2 * np.pi * features['day_of_week'] / 7)
        features['dow_cos'] = np.cos(2 * np.pi * features['day_of_week'] / 7)
        features['month_sin'] = np.sin(2 * np.pi * features['month'] / 12)
        features['month_cos'] = np.cos(2 * np.pi * features['month'] / 12)
        
        # Special periods
        features['is_weekend'] = (features['day_of_week'] >= 5).astype(int)
        features['is_month_end'] = (df.index.day >= 28).astype(int)
        features['is_quarter_end'] = ((features['month'] % 3 == 0) & 
                                      (features['is_month_end'] == 1)).astype(int)
        
        # Combined hour of week (0-167)
        features['hour_of_week'] = features['day_of_week'] * 24 + features['hour']
        
        return features
    
    @staticmethod
    def create_session_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Identify regional trading sessions and overlaps.
        
        Main sessions (UTC):
        - Asia: 00:00 - 09:00
        - Europe: 07:00 - 16:00  
        - Americas: 13:00 - 22:00
        
        Args:
            df: DataFrame with datetime index
            
        Returns:
            DataFrame with session features
        """
        features = pd.DataFrame(index=df.index)
        hour = df.index.hour
        
        # Major trading sessions
        features['session_asia'] = ((hour >= 0) & (hour < 9)).astype(int)
        features['session_europe'] = ((hour >= 7) & (hour < 16)).astype(int)
        features['session_americas'] = ((hour >= 13) & (hour < 22)).astype(int)
        
        # Session overlaps (higher volume/volatility)
        features['overlap_asia_europe'] = ((hour >= 7) & (hour < 9)).astype(int)
        features['overlap_europe_americas'] = ((hour >= 13) & (hour < 16)).astype(int)
        
        # Count active sessions
        features['active_sessions'] = (
            features['session_asia'] + 
            features['session_europe'] + 
            features['session_americas']
        )
        
        # Low activity period (no major sessions)
        features['low_activity'] = (features['active_sessions'] == 0).astype(int)
        
        return features
    
    @staticmethod
    def create_funding_features(
        df: pd.DataFrame, 
        features: Optional[pd.DataFrame] = None,
        funding_period_minutes: int = 480
    ) -> pd.DataFrame:
        """
        Create funding-related features for perpetual futures.
        
        Default: 480 minutes (8 hours) - standard for most perpetual contracts
        Some contracts use 60 minutes (1 hour) - adjust by symbol
        
        Args:
            df: DataFrame with datetime index
            features: Existing features to append to
            funding_period_minutes: Funding period in minutes
            
        Returns:
            DataFrame with funding features
        """
        if features is None:
            features = pd.DataFrame(index=df.index)
        else:
            features = features.copy()
        
        # Convert funding period to bars (15min each)
        funding_period_bars = funding_period_minutes // 15
        
        # Extract time components
        hour = df.index.hour
        minute = df.index.minute
        
        # Minutes from start of day
        minutes_in_day = hour * 60 + minute
        
        # Generate funding times dynamically
        funding_times = list(range(0, 1440, funding_period_minutes))
        
        # Calculate minutes to next funding
        features['minutes_to_funding'] = [
            min(((ft - m) % 1440) for ft in funding_times) 
            for m in minutes_in_day
        ]
        
        features['bars_to_funding'] = features['minutes_to_funding'] / 15
        
        # Funding proximity (exponential decay)
        features['funding_proximity'] = np.exp(-features['bars_to_funding'] / 10)
        
        # Is funding time?
        features['is_funding_time'] = (features['minutes_to_funding'] == 0).astype(int)
        
        # Pre-funding window (12.5% of period before funding)
        pre_funding_minutes = min(60, funding_period_minutes // 8)
        features['pre_funding_window'] = (
            features['minutes_to_funding'] <= pre_funding_minutes
        ).astype(int)
        
        # Funding cycle number (which period in day)
        features['funding_cycle'] = (minutes_in_day // funding_period_minutes).astype(int)
        
        # Cyclical features for funding
        features['funding_cycle_sin'] = np.sin(
            2 * np.pi * features['bars_to_funding'] / funding_period_bars
        )
        features['funding_cycle_cos'] = np.cos(
            2 * np.pi * features['bars_to_funding'] / funding_period_bars
        )
        
        return features


class CalendarFeatures:
    """Enhanced calendar features with crypto 24/7 support."""
    
    def __init__(self, include_crypto_features: bool = True):
        """
        Initialize calendar features calculator.
        
        Args:
            include_crypto_features: Include 24/7 crypto-specific features
        """
        self.include_crypto_features = include_crypto_features
        self.crypto_features = Crypto24x7Features()
    
    def calculate_time_components(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract basic time components using centralized method.
        
        Args:
            df: DataFrame with datetime index
            
        Returns:
            DataFrame with time component features
        """
        # Use the crypto calendar method to avoid duplication
        crypto_calendar = self.crypto_features.create_calendar_features(df)
        
        # Add extra components not in crypto method
        crypto_calendar["year"] = df.index.year
        crypto_calendar["day_of_year"] = df.index.dayofyear
        
        return crypto_calendar
    
    def calculate_cyclical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate additional cyclical features not covered by crypto method.
        
        Args:
            df: DataFrame with time components
            
        Returns:
            DataFrame with cyclical features
        """
        # Add additional cyclical features not in crypto method
        # (crypto method already has hour_sin/cos, dow_sin/cos, month_sin/cos)
        
        # Day of month cyclical encoding
        if "day_of_month" in df.columns:
            df["day_of_month_sin"] = np.sin(2 * np.pi * df["day_of_month"] / 31)
            df["day_of_month_cos"] = np.cos(2 * np.pi * df["day_of_month"] / 31)
        
        return df
    
    def calculate_trading_sessions(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate additional trading session indicators using centralized method.
        
        Args:
            df: DataFrame with hour feature
            
        Returns:
            DataFrame with trading session features
        """
        if "hour" not in df.columns:
            raise ValueError("DataFrame must contain 'hour' column")
        
        # Use crypto session method to avoid duplication
        crypto_sessions = self.crypto_features.create_session_features(df)
        
        # Add any additional session features not in crypto method
        # (crypto method already covers main sessions and overlaps)
        
        # Active trading hours (high volume periods) - additional feature
        crypto_sessions["active_hours"] = (
            ((df["hour"] >= 8) & (df["hour"] <= 10)) |  # Europe open
            ((df["hour"] >= 14) & (df["hour"] <= 16))   # US open
        ).astype(int)
        
        return crypto_sessions
    
    def calculate_special_periods(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate additional special time period indicators.
        
        Args:
            df: DataFrame with time components
            
        Returns:
            DataFrame with special period features
        """
        # Additional weekday indicators (is_weekend already in crypto method)
        if "day_of_week" in df.columns:
            df["is_weekday"] = (df["day_of_week"] < 5).astype(int)
            df["is_monday"] = (df["day_of_week"] == 0).astype(int)
            df["is_friday"] = (df["day_of_week"] == 4).astype(int)
        
        # Month boundaries
        if "day_of_month" in df.columns:
            df["month_start"] = (df["day_of_month"] <= 5).astype(int)
            df["month_end"] = (df["day_of_month"] >= 25).astype(int)
            df["month_middle"] = (
                (df["day_of_month"] > 10) & (df["day_of_month"] < 20)
            ).astype(int)
        
        # Quarter boundaries
        if "month" in df.columns:
            df["quarter_start"] = (df["month"] % 3 == 1).astype(int)
            df["quarter_end"] = (df["month"] % 3 == 0).astype(int)
        
        # Year boundaries
        if "month" in df.columns:
            df["year_start"] = (df["month"] == 1).astype(int)
            df["year_end"] = (df["month"] == 12).astype(int)
        
        return df
    
    def calculate_time_since_events(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate time since specific events.
        
        Args:
            df: DataFrame with time components
            
        Returns:
            DataFrame with time-since features
        """
        # Time since last weekend
        if "is_weekend" in df.columns:
            weekend_mask = df["is_weekend"] == 1
            df["bars_since_weekend"] = (~weekend_mask).groupby(
                weekend_mask.cumsum()
            ).cumsum()
        
        # Time since month start
        if "month_start" in df.columns:
            month_start_mask = df["month_start"] == 1
            df["bars_since_month_start"] = (~month_start_mask).groupby(
                month_start_mask.cumsum()
            ).cumsum()
        
        # Time of day progress
        if "hour" in df.columns:
            df["day_progress"] = df["hour"] / 24
        
        # Month progress
        if "day_of_month" in df.columns:
            df["month_progress"] = df["day_of_month"] / 31
        
        return df
    
    def calculate_all(self, df: pd.DataFrame, funding_period_minutes: int = 480) -> pd.DataFrame:
        """
        Calculate all calendar features including crypto 24/7 features.
        
        Args:
            df: DataFrame with datetime index
            funding_period_minutes: Funding period for crypto features
            
        Returns:
            DataFrame with all calendar features
        """
        df = df.copy()
        
        if self.include_crypto_features:
            # Use centralized crypto methods directly
            crypto_calendar = self.crypto_features.create_calendar_features(df)
            crypto_sessions = self.crypto_features.create_session_features(df)
            crypto_funding = self.crypto_features.create_funding_features(
                df, funding_period_minutes=funding_period_minutes
            )
            
            # Start with crypto features as base
            for features_df in [crypto_calendar, crypto_sessions, crypto_funding]:
                existing_cols = set(df.columns)
                new_cols = [col for col in features_df.columns if col not in existing_cols]
                if new_cols:
                    df = pd.concat([df, features_df[new_cols]], axis=1)
            
            # Add any additional features not covered
            df = self.calculate_cyclical_features(df)  # Will only add day_of_month_sin/cos
            df = self.calculate_special_periods(df)    # Will add extra weekday flags
            df = self.calculate_time_since_events(df)
            
            # Add year and day_of_year not in crypto
            df["year"] = df.index.year
            df["day_of_year"] = df.index.dayofyear
        else:
            # Fallback to standard methods
            df = self.calculate_time_components(df)
            df = self.calculate_cyclical_features(df)
            df = self.calculate_trading_sessions(df)
            df = self.calculate_special_periods(df)
            df = self.calculate_time_since_events(df)
        
        return df