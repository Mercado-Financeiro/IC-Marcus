"""Tests for feature engineering module."""

import pytest
import numpy as np
import pandas as pd
import sys
from pathlib import Path
from sklearn.preprocessing import StandardScaler, RobustScaler

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.features.engineering import FeatureEngineer


class TestFeatureEngineer:
    """Test suite for feature engineering."""
    
    def test_create_all_features(self, sample_ohlcv_data):
        """Test complete feature creation pipeline."""
        engineer = FeatureEngineer()
        
        df = engineer.create_all_features(sample_ohlcv_data.copy())
        
        # Check that features were created
        assert len(df.columns) > len(sample_ohlcv_data.columns), \
            "No new features created"
        
        # Check no NaN in result (should be dropped)
        assert df.isna().sum().sum() == 0, "NaN values present in features"
        
        # Check feature names were saved
        assert len(engineer.feature_names) > 0, "Feature names not saved"
        
        # Check specific feature categories exist
        feature_cols = df.columns.tolist()
        
        # Price features
        assert any('returns' in col for col in feature_cols), "No returns features"
        assert any('sma_' in col for col in feature_cols), "No SMA features"
        assert any('zscore_' in col for col in feature_cols), "No z-score features"
        
        # Volatility features  
        assert any('volatility_' in col for col in feature_cols), "No volatility features"
        
        # Technical indicators
        assert any('rsi_' in col for col in feature_cols), "No RSI features"
        
        # Microstructure
        assert any('volume' in col for col in feature_cols), "No volume features"
        
        # Calendar features
        assert 'hour' in feature_cols, "No hour feature"
        assert 'day_of_week' in feature_cols, "No day of week feature"
    
    def test_no_look_ahead_bias(self, sample_ohlcv_data):
        """Test that features don't use future information."""
        engineer = FeatureEngineer(lookback_periods=[5, 10])
        
        df = engineer.create_all_features(sample_ohlcv_data.copy())
        
        # For rolling features, check that they use only past data
        # This is implicitly tested by pandas rolling behavior
        # but we can verify by checking that early rows have NaN
        
        # The first few rows should have been dropped due to NaN
        assert len(df) < len(sample_ohlcv_data), \
            "No rows dropped - possible look-ahead bias"
    
    def test_price_features(self, sample_ohlcv_data):
        """Test price-based features."""
        engineer = FeatureEngineer(lookback_periods=[5, 10, 20])
        
        df = engineer.create_price_features(sample_ohlcv_data.copy())
        
        # Check returns
        assert 'returns' in df.columns
        assert df['returns'].iloc[1:].notna().all(), "Returns have NaN after first"
        
        # Check cumulative returns
        for period in [5, 10, 20]:
            assert f'returns_{period}' in df.columns
            assert f'sma_{period}' in df.columns
            assert f'momentum_{period}' in df.columns
        
        # Check z-scores
        assert 'zscore_20' in df.columns
        assert 'zscore_50' in df.columns
        
        # Check crossovers
        assert 'sma_cross_20_50' in df.columns
    
    def test_volatility_features(self, sample_ohlcv_data):
        """Test volatility features."""
        engineer = FeatureEngineer(lookback_periods=[10, 20])
        
        # Need returns first
        df = sample_ohlcv_data.copy()
        df['returns'] = df['close'].pct_change()
        
        df = engineer.create_volatility_features(df)
        
        # Check historical volatility
        assert 'volatility_10' in df.columns
        assert 'volatility_20' in df.columns
        
        # Check annualized volatility
        assert 'volatility_ann_10' in df.columns
        
        # Check Parkinson volatility
        assert 'parkinson_vol_10' in df.columns
        
        # Check Garman-Klass volatility
        assert 'gk_vol_10' in df.columns
        
        # Check volatility ratios
        assert 'vol_ratio_10_50' in df.columns
    
    def test_technical_indicators(self, sample_ohlcv_data):
        """Test technical indicators."""
        engineer = FeatureEngineer(
            technical_indicators=['rsi', 'macd', 'bbands', 'atr']
        )
        
        df = engineer.create_technical_indicators(sample_ohlcv_data.copy())
        
        # RSI
        assert 'rsi_14' in df.columns
        assert 'rsi_14_overbought' in df.columns
        assert 'rsi_14_oversold' in df.columns
        
        # MACD
        assert 'macd' in df.columns
        assert 'macd_signal' in df.columns
        assert 'macd_diff' in df.columns
        assert 'macd_crossover' in df.columns
        
        # Bollinger Bands
        assert 'bb_high_20' in df.columns
        assert 'bb_low_20' in df.columns
        assert 'bb_position_20' in df.columns
        
        # ATR
        assert 'atr_14' in df.columns
        assert 'atr_pct_14' in df.columns
    
    def test_microstructure_features(self, sample_ohlcv_data):
        """Test microstructure features."""
        engineer = FeatureEngineer()
        
        df = engineer.create_microstructure_features(sample_ohlcv_data.copy())
        
        # Volume features
        assert 'volume_ratio' in df.columns
        assert 'volume_trend' in df.columns
        assert 'dollar_volume' in df.columns
        
        # VWAP
        assert 'vwap_20' in df.columns
        assert 'vwap_distance_20' in df.columns
        
        # Spread
        assert 'hl_spread' in df.columns
        assert 'close_position' in df.columns
        
        # Liquidity measures
        assert 'amihud_illiq' in df.columns
        assert 'kyle_lambda_20' in df.columns
    
    def test_regime_features(self, sample_ohlcv_data):
        """Test regime features."""
        engineer = FeatureEngineer()
        
        # Need returns for regime features
        df = sample_ohlcv_data.copy()
        df['returns'] = df['close'].pct_change()
        
        df = engineer.create_regime_features(df)
        
        # Volatility regime
        assert 'vol_percentile_252' in df.columns
        assert 'high_vol_regime' in df.columns
        assert 'low_vol_regime' in df.columns
        
        # Momentum regime
        assert 'momentum_percentile' in df.columns
        assert 'high_momentum' in df.columns
        
        # Market phases
        assert 'accumulation_phase' in df.columns
        assert 'markup_phase' in df.columns
        assert 'distribution_phase' in df.columns
        assert 'markdown_phase' in df.columns
    
    def test_calendar_features(self, sample_ohlcv_data):
        """Test calendar features."""
        engineer = FeatureEngineer()
        
        df = engineer.create_calendar_features(sample_ohlcv_data.copy())
        
        # Time components
        assert 'hour' in df.columns
        assert 'day_of_week' in df.columns
        assert 'month' in df.columns
        assert 'quarter' in df.columns
        
        # Cyclical features
        assert 'hour_sin' in df.columns
        assert 'hour_cos' in df.columns
        assert 'day_sin' in df.columns
        assert 'day_cos' in df.columns
        
        # Trading sessions
        assert 'asian_session' in df.columns
        assert 'european_session' in df.columns
        assert 'american_session' in df.columns
        assert 'session_overlap' in df.columns
        
        # Special periods
        assert 'is_weekend' in df.columns
        assert 'month_start' in df.columns
        assert 'month_end' in df.columns
        
        # Verify cyclical encoding
        assert df['hour_sin'].min() >= -1.01
        assert df['hour_sin'].max() <= 1.01
        assert df['hour_cos'].min() >= -1.01
        assert df['hour_cos'].max() <= 1.01
    
    def test_fit_transform_with_scaler(self, sample_ohlcv_data):
        """Test fit_transform with different scalers."""
        # Standard scaler
        engineer_std = FeatureEngineer(scaler_type='standard')
        df_std = engineer_std.fit_transform(sample_ohlcv_data.copy())
        
        # Check scaler was fitted
        assert engineer_std.scaler is not None
        assert hasattr(engineer_std.scaler, 'mean_')
        
        # Check features are scaled (approximately mean=0, std=1)
        feature_cols = engineer_std.feature_names
        if len(feature_cols) > 0:
            means = df_std[feature_cols].mean()
            stds = df_std[feature_cols].std()
            
            # Most features should be close to mean=0, std=1
            assert abs(means.mean()) < 0.1, "Scaled features not centered"
            assert abs(stds.mean() - 1.0) < 0.2, "Scaled features not normalized"
        
        # Robust scaler
        engineer_rob = FeatureEngineer(scaler_type='robust')
        df_rob = engineer_rob.fit_transform(sample_ohlcv_data.copy())
        
        assert engineer_rob.scaler is not None
        assert hasattr(engineer_rob.scaler, 'center_')
    
    def test_transform_without_fit(self, sample_ohlcv_data):
        """Test that transform fails without fit."""
        engineer = FeatureEngineer(scaler_type='standard')
        
        with pytest.raises(ValueError, match="Scaler"):
            engineer.transform(sample_ohlcv_data.copy())
    
    def test_transform_after_fit(self, sample_ohlcv_data):
        """Test transform after fit."""
        engineer = FeatureEngineer(scaler_type='standard')
        
        # Split data
        split_idx = len(sample_ohlcv_data) // 2
        train_data = sample_ohlcv_data.iloc[:split_idx].copy()
        test_data = sample_ohlcv_data.iloc[split_idx:].copy()
        
        # Fit on train
        train_features = engineer.fit_transform(train_data)
        
        # Transform test
        test_features = engineer.transform(test_data)
        
        # Check same features created
        assert set(train_features.columns) == set(test_features.columns)
        
        # Check scaler was applied (not refitted)
        # Test data might have different distribution but scaler params unchanged
        assert hasattr(engineer.scaler, 'mean_')
        original_mean = engineer.scaler.mean_.copy()
        
        # Transform again - scaler params should be same
        test_features2 = engineer.transform(test_data)
        np.testing.assert_array_equal(engineer.scaler.mean_, original_mean)
    
    def test_feature_importance_extraction(self):
        """Test feature importance extraction."""
        engineer = FeatureEngineer()
        
        # Mock model with feature importances
        class MockModel:
            def __init__(self):
                self.feature_importances_ = np.array([0.3, 0.2, 0.5])
        
        model = MockModel()
        feature_names = ['feature1', 'feature2', 'feature3']
        
        importance_df = engineer.get_feature_importance(model, feature_names)
        
        # Check structure
        assert 'feature' in importance_df.columns
        assert 'importance' in importance_df.columns
        assert 'importance_pct' in importance_df.columns
        
        # Check sorting
        assert importance_df['importance'].iloc[0] >= importance_df['importance'].iloc[1]
        
        # Check percentages sum to 100
        assert np.isclose(importance_df['importance_pct'].sum(), 100)
    
    def test_reproducibility(self, sample_ohlcv_data):
        """Test that feature creation is reproducible."""
        engineer1 = FeatureEngineer(lookback_periods=[5, 10])
        engineer2 = FeatureEngineer(lookback_periods=[5, 10])
        
        df1 = engineer1.create_all_features(sample_ohlcv_data.copy())
        df2 = engineer2.create_all_features(sample_ohlcv_data.copy())
        
        # Check same columns
        assert set(df1.columns) == set(df2.columns)
        
        # Check same values
        pd.testing.assert_frame_equal(df1, df2)


class TestFeatureEngineerEdgeCases:
    """Test edge cases for feature engineering."""
    
    def test_insufficient_data(self):
        """Test with insufficient data for features."""
        dates = pd.date_range('2023-01-01', periods=5, freq='15min', tz='UTC')
        
        df = pd.DataFrame({
            'open': [100.0] * 5,
            'high': [101.0] * 5,
            'low': [99.0] * 5,
            'close': [100.0] * 5,
            'volume': [1000] * 5
        }, index=dates)
        
        engineer = FeatureEngineer(lookback_periods=[10, 20])  # Longer than data
        
        # Should handle gracefully
        result = engineer.create_all_features(df)
        
        # Might be empty or very few rows
        assert len(result) <= len(df)
    
    def test_missing_columns(self):
        """Test with missing required columns."""
        df = pd.DataFrame({
            'close': [100.0] * 10,
            'volume': [1000] * 10
        })
        
        engineer = FeatureEngineer()
        
        with pytest.raises(ValueError, match="obrigatória"):
            engineer.create_all_features(df)
    
    def test_constant_prices(self):
        """Test with constant prices (no variation)."""
        dates = pd.date_range('2023-01-01', periods=100, freq='15min', tz='UTC')
        
        df = pd.DataFrame({
            'open': [100.0] * 100,
            'high': [100.0] * 100,
            'low': [100.0] * 100,
            'close': [100.0] * 100,
            'volume': [1000] * 100
        }, index=dates)
        
        engineer = FeatureEngineer()
        
        result = engineer.create_all_features(df)
        
        # Should handle division by zero in z-scores, volatility, etc.
        assert not result.empty
        
        # Check for inf or extreme values
        assert not np.isinf(result.select_dtypes(include=[np.number])).any().any()