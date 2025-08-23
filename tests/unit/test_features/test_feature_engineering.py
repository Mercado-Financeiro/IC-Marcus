"""Unit tests for main feature engineering orchestrator."""

import pytest
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler

from src.features.engineering import FeatureEngineer


@pytest.fixture
def sample_ohlcv_data():
    """Create sample OHLCV data for testing."""
    dates = pd.date_range(start="2024-01-01", periods=500, freq="15min")
    np.random.seed(42)
    
    close_prices = 50000 + np.cumsum(np.random.randn(500) * 100)
    
    df = pd.DataFrame({
        "open": close_prices + np.random.randn(500) * 50,
        "high": close_prices + np.abs(np.random.randn(500) * 100),
        "low": close_prices - np.abs(np.random.randn(500) * 100),
        "close": close_prices,
        "volume": np.abs(np.random.randn(500) * 1000000) + 100000
    }, index=dates)
    
    return df


class TestFeatureEngineer:
    """Test feature engineering orchestrator."""
    
    def test_initialization(self):
        """Test FeatureEngineer initialization."""
        # Default initialization
        fe = FeatureEngineer()
        assert fe.lookback_periods == [5, 10, 20, 50, 100, 200]
        assert fe.technical_indicators_list == [
            "rsi", "macd", "bbands", "atr", "obv", "adx", "cci", "stoch"
        ]
        assert fe.scaler_type == "robust"
        assert isinstance(fe.scaler, RobustScaler)
        
        # Custom initialization
        fe = FeatureEngineer(
            lookback_periods=[10, 20],
            technical_indicators=["rsi", "macd"],
            scaler_type="standard",
            include_microstructure=False
        )
        assert fe.lookback_periods == [10, 20]
        assert fe.technical_indicators_list == ["rsi", "macd"]
        assert isinstance(fe.scaler, StandardScaler)
        assert not fe.include_microstructure
        
        # No scaler
        fe = FeatureEngineer(scaler_type="none")
        assert fe.scaler is None
    
    def test_create_all_features(self, sample_ohlcv_data):
        """Test complete feature creation."""
        fe = FeatureEngineer(
            lookback_periods=[10, 20, 50],  # Reduced from default [5, 10, 20, 50, 100, 200]
            technical_indicators=["rsi", "macd"],
            include_microstructure=True
        )
        
        df = fe.create_all_features(sample_ohlcv_data)
        
        # Check that features were added
        assert len(df.columns) > len(sample_ohlcv_data.columns)
        assert len(fe.feature_names) > 0
        
        # Check specific feature groups
        feature_names = set(df.columns)
        
        # Price features
        assert "returns" in feature_names
        assert "sma_10" in feature_names
        assert "momentum_10" in feature_names
        
        # Volatility features
        assert "volatility_10" in feature_names
        
        # Technical indicators
        assert "rsi_14" in feature_names
        assert "macd" in feature_names
        
        # Calendar features
        assert "hour" in feature_names
        assert "day_of_week" in feature_names
        
        # Microstructure features (if enabled)
        assert "volume_sma_20" in feature_names
        assert "vwap_20" in feature_names
        
        # Check no NaN values after dropna
        assert not df.isna().any().any()
    
    def test_fit_transform(self, sample_ohlcv_data):
        """Test fit_transform method."""
        fe = FeatureEngineer(
            lookback_periods=[10, 20],  # Smaller periods
            technical_indicators=["rsi"],
            scaler_type="standard"
        )
        
        df = fe.fit_transform(sample_ohlcv_data)
        
        # Check scaler was fitted
        assert hasattr(fe.scaler, "mean_")
        assert hasattr(fe.scaler, "scale_")
        
        # Check features were scaled
        # Standard scaler should produce mean ~0 and std ~1
        feature_cols = [col for col in df.columns 
                       if col not in ["open", "high", "low", "close", "volume"]]
        
        if len(feature_cols) > 0 and len(df) > 0:
            feature_values = df[feature_cols].values.flatten()
            feature_values = feature_values[~np.isnan(feature_values)]
            if len(feature_values) > 0:
                assert abs(np.mean(feature_values)) < 0.5  # Mean close to 0
                assert abs(np.std(feature_values) - 1) < 0.5  # Std close to 1
    
    def test_transform(self, sample_ohlcv_data):
        """Test transform method."""
        fe = FeatureEngineer(
            lookback_periods=[10, 20],  # Smaller periods
            technical_indicators=["rsi"],
            scaler_type="robust"
        )
        
        # Use full data for both train and test to ensure enough data after dropna
        # In real scenario, you'd have much more data
        train_data = sample_ohlcv_data
        test_data = sample_ohlcv_data
        
        # Fit on train
        train_df = fe.fit_transform(train_data)
        
        # Transform test (same data in this test, but uses transform not fit_transform)
        test_df = fe.transform(test_data)
        
        # Check same features
        assert set(train_df.columns) == set(test_df.columns)
        assert len(test_df) > 0  # Should have some data
        
        # Check transform without fit raises error
        fe_new = FeatureEngineer(scaler_type="standard")
        with pytest.raises(ValueError, match="not fitted"):
            fe_new.transform(test_data)
    
    def test_get_feature_groups(self, sample_ohlcv_data):
        """Test feature grouping."""
        fe = FeatureEngineer(
            lookback_periods=[10, 20],
            technical_indicators=["rsi", "macd", "bbands"],
            include_microstructure=True
        )
        
        df = fe.create_all_features(sample_ohlcv_data)
        groups = fe.get_feature_groups()
        
        # Check groups exist
        assert "price" in groups
        assert "volatility" in groups
        assert "technical" in groups
        assert "microstructure" in groups
        assert "calendar" in groups
        
        # Check features are properly grouped
        assert any("returns" in f for f in groups["price"])
        assert any("volatility" in f for f in groups["volatility"])
        assert any("rsi" in f for f in groups["technical"])
        assert any("volume" in f or "vwap" in f for f in groups["microstructure"])
        assert any("hour" in f or "day" in f for f in groups["calendar"])
        
        # Check all features are accounted for
        all_grouped = []
        for group_features in groups.values():
            all_grouped.extend(group_features)
        assert set(all_grouped) == set(fe.feature_names)
    
    def test_get_feature_importance(self, sample_ohlcv_data):
        """Test feature importance extraction."""
        from sklearn.ensemble import RandomForestRegressor
        
        fe = FeatureEngineer(
            lookback_periods=[10, 20],  # Smaller periods
            technical_indicators=["rsi"],
            scaler_type="none"
        )
        
        # Create features
        df = fe.create_all_features(sample_ohlcv_data)
        
        # Create simple target
        y = df["close"].shift(-1).ffill()
        
        # Train simple model
        feature_cols = fe.feature_names
        X = df[feature_cols].fillna(0)
        
        model = RandomForestRegressor(n_estimators=10, random_state=42)
        model.fit(X, y)
        
        # Get importance
        importance_df = fe.get_feature_importance(model)
        
        # Check structure
        assert "feature" in importance_df.columns
        assert "importance" in importance_df.columns
        assert "importance_pct" in importance_df.columns
        
        # Check sorted
        assert importance_df["importance"].is_monotonic_decreasing
        
        # Check percentages sum to 100
        assert abs(importance_df["importance_pct"].sum() - 100) < 0.01
    
    def test_missing_columns_error(self):
        """Test error when required columns are missing."""
        fe = FeatureEngineer()
        
        # Missing close column
        df = pd.DataFrame({
            "open": [100, 101],
            "high": [102, 103],
            "low": [99, 100],
            "volume": [1000, 1100]
        })
        
        with pytest.raises(ValueError, match="Required column missing"):
            fe.create_all_features(df)
    
    def test_advanced_features(self, sample_ohlcv_data):
        """Test advanced features handling."""
        fe = FeatureEngineer(
            lookback_periods=[10, 20],  # Smaller periods
            include_advanced=True  # Enable advanced features
        )
        
        # Should not raise error even if modules don't exist
        df = fe.create_all_features(sample_ohlcv_data)
        # May have fewer rows due to NaN dropping, but should have some data
        assert len(df.columns) > len(sample_ohlcv_data.columns)
    
    def test_no_data_modification(self, sample_ohlcv_data):
        """Test that original data is not modified."""
        fe = FeatureEngineer()
        
        original_copy = sample_ohlcv_data.copy()
        df = fe.create_all_features(sample_ohlcv_data)
        
        # Original should be unchanged
        pd.testing.assert_frame_equal(sample_ohlcv_data, original_copy)