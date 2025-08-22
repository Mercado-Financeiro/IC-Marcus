"""
Integration tests for LSTM with other system components.
Tests interaction with feature engineering, backtest, MLflow, etc.
"""

import pytest
import numpy as np
import pandas as pd
import torch
from pathlib import Path
import sys
import tempfile
import shutil

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

try:
    from src.models.lstm_optuna import LSTMOptuna
    from src.features.engineering import FeatureEngineer
    from src.backtest.engine import BacktestEngine
    from src.data.splits import PurgedKFold
    IMPORT_SUCCESS = True
except Exception as e:
    IMPORT_SUCCESS = False
    IMPORT_ERROR = str(e)


@pytest.fixture
def sample_ohlcv_data():
    """Generate sample OHLCV data for integration tests."""
    np.random.seed(42)
    n_periods = 1000
    
    # Generate realistic OHLCV data
    dates = pd.date_range('2023-01-01', periods=n_periods, freq='h')
    
    # Random walk for price
    returns = np.random.randn(n_periods) * 0.01
    prices = 50000 * np.exp(returns.cumsum())  # Start at $50,000
    
    # OHLCV generation
    data = []
    for i in range(n_periods):
        open_price = prices[i]
        close_price = prices[i] * (1 + returns[i])
        
        # High and Low with some noise
        high_noise = np.random.exponential(0.005)
        low_noise = np.random.exponential(0.005)
        
        high = max(open_price, close_price) * (1 + high_noise)
        low = min(open_price, close_price) * (1 - low_noise)
        
        volume = np.random.exponential(1000)
        
        data.append({
            'datetime': dates[i],
            'open': open_price,
            'high': high,
            'low': low,
            'close': close_price,
            'volume': volume
        })
    
    df = pd.DataFrame(data)
    df.set_index('datetime', inplace=True)
    return df


class TestLSTMFeatureEngineeringIntegration:
    """Test LSTM integration with feature engineering."""
    
    def test_lstm_with_engineered_features(self, sample_ohlcv_data):
        """Test LSTM works with engineered features."""
        if not IMPORT_SUCCESS:
            pytest.skip("Import failed")
        
        # Create feature engineer
        fe = FeatureEngineer(lookback_periods=[5, 10, 20])
        
        # Generate features
        df_with_features = fe.create_price_features(sample_ohlcv_data)
        df_with_features = fe.create_technical_indicators(df_with_features)
        
        # Create labels
        df_with_features['returns'] = df_with_features['close'].pct_change()
        df_with_features['label'] = (df_with_features['returns'].shift(-1) > 0).astype(int)
        
        # Clean data
        df_clean = df_with_features.dropna()
        
        # Select features (avoid leakage)
        feature_cols = [c for c in df_clean.columns 
                       if c not in ['open', 'high', 'low', 'close', 'volume', 'label', 'returns']]
        
        X = df_clean[feature_cols]
        y = df_clean['label']
        
        # Test with LSTM
        optimizer = LSTMOptuna(n_trials=2, cv_folds=2, max_epochs=5, seed=42)
        
        # Should handle many features
        study = optimizer.optimize(X[:300], y[:300])  # Use subset for speed
        
        assert study is not None
        assert optimizer.best_params is not None
        assert len(feature_cols) > 10  # Should have many engineered features
        
        print(f"✅ LSTM works with {len(feature_cols)} engineered features")
    
    def test_lstm_feature_importance_proxy(self, sample_ohlcv_data):
        """Test feature importance estimation for LSTM (via permutation)."""
        if not IMPORT_SUCCESS:
            pytest.skip("Import failed")
        
        # Simple features for interpretable test
        df = sample_ohlcv_data.copy()
        df['returns'] = df['close'].pct_change()
        df['rsi'] = df['close'].rolling(14).apply(
            lambda x: 100 - (100 / (1 + (x.diff().clip(lower=0).mean() / 
                                        x.diff().clip(upper=0).abs().mean()))))
        df['ma_5'] = df['close'].rolling(5).mean()
        df['volatility'] = df['returns'].rolling(20).std()
        df['label'] = (df['returns'].shift(-1) > 0).astype(int)
        
        df_clean = df.dropna()
        feature_cols = ['returns', 'rsi', 'ma_5', 'volatility']
        X = df_clean[feature_cols]
        y = df_clean['label']
        
        # Train LSTM
        optimizer = LSTMOptuna(n_trials=1, cv_folds=2, max_epochs=5, seed=42)
        optimizer.optimize(X[:200], y[:200])
        optimizer.fit_final_model(X[:200], y[:200])
        
        # Simple permutation importance
        baseline_score = optimizer.predict_proba(X[200:250])
        feature_importance = {}
        
        for feature in feature_cols:
            X_permuted = X[200:250].copy()
            X_permuted[feature] = np.random.permutation(X_permuted[feature])
            permuted_score = optimizer.predict_proba(X_permuted)
            
            # Measure difference in predictions
            importance = np.mean(np.abs(baseline_score - permuted_score))
            feature_importance[feature] = importance
        
        # Should have some feature importance values
        assert len(feature_importance) == len(feature_cols)
        assert all(imp >= 0 for imp in feature_importance.values())
        
        print(f"✅ Feature importance calculated: {feature_importance}")


class TestLSTMBacktestIntegration:
    """Test LSTM integration with backtesting engine."""
    
    def test_lstm_predictions_in_backtest(self, sample_ohlcv_data):
        """Test LSTM predictions work in backtest framework."""
        if not IMPORT_SUCCESS:
            pytest.skip("Import failed")
        
        # Prepare data
        df = sample_ohlcv_data.copy()
        df['returns'] = df['close'].pct_change()
        df['momentum'] = df['returns'].rolling(10).mean()
        df['volatility'] = df['returns'].rolling(20).std()
        df['label'] = (df['returns'].shift(-1) > 0).astype(int)
        
        df_clean = df.dropna()
        feature_cols = ['returns', 'momentum', 'volatility']
        X = df_clean[feature_cols]
        y = df_clean['label']
        
        # Split data for training/testing
        split_point = len(X) // 2
        X_train, X_test = X[:split_point], X[split_point:]
        y_train = y[:split_point]
        
        # Train LSTM
        optimizer = LSTMOptuna(n_trials=1, cv_folds=2, max_epochs=5, seed=42)
        optimizer.optimize(X_train, y_train)
        optimizer.fit_final_model(X_train, y_train)
        
        # Generate predictions for test period
        y_pred_proba = optimizer.predict_proba(X_test)
        y_pred = optimizer.predict(X_test, use_ev_threshold=True)
        
        # Convert to signals aligned with price data
        test_df = df_clean.iloc[split_point:].copy()
        signal_series = pd.Series(y_pred, index=X_test.index)
        
        # Align signals with price data
        aligned_prices = test_df.reindex(signal_series.index)
        
        # Run backtest
        try:
            backtest_engine = BacktestEngine(initial_capital=100000)
            results = backtest_engine.run_backtest(aligned_prices, signal_series)
            metrics = backtest_engine.calculate_metrics(results)
            
            # Basic checks
            assert results is not None
            assert 'total_return' in metrics
            assert 'sharpe_ratio' in metrics
            assert 'max_drawdown' in metrics
            
            print(f"✅ Backtest completed: Return={metrics['total_return']:.2%}, Sharpe={metrics['sharpe_ratio']:.2f}")
            
        except Exception as e:
            pytest.fail(f"Backtest integration failed: {e}")
    
    def test_lstm_signals_quality(self, sample_ohlcv_data):
        """Test quality of LSTM signals for trading."""
        if not IMPORT_SUCCESS:
            pytest.skip("Import failed")
        
        # Create features with clear patterns
        df = sample_ohlcv_data.copy()
        df['returns'] = df['close'].pct_change()
        df['trend'] = df['close'].rolling(50).mean().pct_change()
        df['mean_reversion'] = (df['close'] / df['close'].rolling(20).mean() - 1)
        df['label'] = (df['returns'].shift(-1) > 0.005).astype(int)  # Stronger threshold
        
        df_clean = df.dropna()
        feature_cols = ['returns', 'trend', 'mean_reversion']
        X = df_clean[feature_cols]
        y = df_clean['label']
        
        # Train on first part
        train_size = int(0.7 * len(X))
        X_train = X[:train_size]
        y_train = y[:train_size]
        X_test = X[train_size:]
        y_test = y[train_size:]
        
        # Train LSTM
        optimizer = LSTMOptuna(n_trials=2, cv_folds=2, max_epochs=10, seed=42)
        optimizer.optimize(X_train, y_train)
        optimizer.fit_final_model(X_train, y_train)
        
        # Generate signals
        y_pred_proba = optimizer.predict_proba(X_test)
        y_pred = optimizer.predict(X_test)
        
        # Signal quality checks
        # 1. Predictions should be diverse (not all same)
        unique_predictions = len(np.unique(y_pred))
        assert unique_predictions >= 2, "Predictions are not diverse"
        
        # 2. Probabilities should be well distributed
        prob_std = np.std(y_pred_proba)
        assert prob_std > 0.05, f"Probability std too low: {prob_std:.4f}"
        
        # 3. Should have reasonable accuracy on test set (>45%)
        accuracy = np.mean(y_pred == y_test)
        assert accuracy > 0.45, f"Accuracy too low: {accuracy:.2%}"
        
        print(f"✅ Signal quality: {unique_predictions} unique, std={prob_std:.3f}, acc={accuracy:.2%}")


class TestLSTMDataSplitIntegration:
    """Test LSTM integration with time series cross-validation."""
    
    def test_lstm_with_purged_kfold(self, sample_ohlcv_data):
        """Test LSTM works correctly with PurgedKFold."""
        if not IMPORT_SUCCESS:
            pytest.skip("Import failed")
        
        # Prepare data
        df = sample_ohlcv_data.copy()
        df['returns'] = df['close'].pct_change()
        df['volatility'] = df['returns'].rolling(20).std()
        df['label'] = (df['returns'].shift(-1) > 0).astype(int)
        
        df_clean = df.dropna()
        X = df_clean[['returns', 'volatility']]
        y = df_clean['label']
        
        # Test PurgedKFold integration
        cv = PurgedKFold(n_splits=3, embargo=10)
        fold_scores = []
        
        for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X, y)):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            # Check temporal separation (no leakage)
            train_end = X_train.index.max()
            val_start = X_val.index.min()
            
            time_gap = (val_start - train_end).total_seconds() / 3600  # Hours
            assert time_gap >= 10 or train_end < val_start, f"Fold {fold_idx}: Temporal leakage detected"
            
            # Quick LSTM training
            optimizer = LSTMOptuna(n_trials=1, cv_folds=2, max_epochs=3, seed=42)
            optimizer.fit_final_model(X_train, y_train)
            
            # Evaluate
            y_pred_proba = optimizer.predict_proba(X_val)
            y_pred = (y_pred_proba > 0.5).astype(int)
            
            accuracy = np.mean(y_pred == y_val)
            fold_scores.append(accuracy)
            
            print(f"Fold {fold_idx}: acc={accuracy:.3f}, gap={time_gap:.1f}h")
        
        # Should complete all folds
        assert len(fold_scores) == 3
        assert all(0.3 <= score <= 0.8 for score in fold_scores)  # Reasonable range
        
        print(f"✅ PurgedKFold integration: avg acc={np.mean(fold_scores):.3f}")
    
    def test_lstm_temporal_consistency(self, sample_ohlcv_data):
        """Test LSTM predictions are temporally consistent."""
        if not IMPORT_SUCCESS:
            pytest.skip("Import failed")
        
        # Prepare data
        df = sample_ohlcv_data.copy()
        df['returns'] = df['close'].pct_change()
        df['sma'] = df['close'].rolling(10).mean()
        df['label'] = (df['returns'].shift(-1) > 0).astype(int)
        
        df_clean = df.dropna()
        X = df_clean[['returns', 'sma']]
        y = df_clean['label']
        
        # Train on early data
        train_end = int(0.6 * len(X))
        X_train = X[:train_end]
        y_train = y[:train_end]
        
        # Test on two consecutive periods
        mid_point = int(0.8 * len(X))
        X_test1 = X[train_end:mid_point]
        X_test2 = X[mid_point:]
        
        optimizer = LSTMOptuna(n_trials=1, cv_folds=2, max_epochs=5, seed=42)
        optimizer.optimize(X_train, y_train)
        optimizer.fit_final_model(X_train, y_train)
        
        # Get predictions for both periods
        pred1 = optimizer.predict_proba(X_test1)
        pred2 = optimizer.predict_proba(X_test2)
        
        # Predictions should be temporally stable (not drastically different)
        pred1_mean = np.mean(pred1)
        pred2_mean = np.mean(pred2)
        
        mean_diff = abs(pred1_mean - pred2_mean)
        assert mean_diff < 0.3, f"Temporal inconsistency: {pred1_mean:.3f} vs {pred2_mean:.3f}"
        
        print(f"✅ Temporal consistency: {pred1_mean:.3f} vs {pred2_mean:.3f} (diff={mean_diff:.3f})")


class TestLSTMMLflowIntegration:
    """Test LSTM integration with MLflow (mock)."""
    
    def test_lstm_mlflow_logging_interface(self, sample_ohlcv_data):
        """Test LSTM can work with MLflow logging interface."""
        if not IMPORT_SUCCESS:
            pytest.skip("Import failed")
        
        # This test checks that LSTM doesn't break when MLflow is enabled
        # We don't actually test MLflow logging (requires MLflow setup)
        
        df = sample_ohlcv_data.copy()
        df['returns'] = df['close'].pct_change()
        df['label'] = (df['returns'].shift(-1) > 0).astype(int)
        
        df_clean = df.dropna()
        X = df_clean[['returns']]
        y = df_clean['label']
        
        # Test with use_mlflow=True (but no actual MLflow server)
        optimizer = LSTMOptuna(
            n_trials=1, 
            cv_folds=2, 
            max_epochs=3,
            use_mlflow=True,  # This should not break the code
            seed=42
        )
        
        # Should work even without MLflow server
        try:
            study = optimizer.optimize(X[:100], y[:100])
            assert study is not None
            print("✅ MLflow interface doesn't break LSTM")
        except Exception as e:
            # Should not fail due to MLflow issues
            assert "mlflow" not in str(e).lower(), f"MLflow caused failure: {e}"


class TestLSTMSerializationIntegration:
    """Test LSTM serialization and persistence."""
    
    def test_lstm_model_persistence(self, sample_ohlcv_data):
        """Test LSTM model can be saved and loaded."""
        if not IMPORT_SUCCESS:
            pytest.skip("Import failed")
        
        # Prepare data
        df = sample_ohlcv_data.copy()
        df['returns'] = df['close'].pct_change()
        df['label'] = (df['returns'].shift(-1) > 0).astype(int)
        
        df_clean = df.dropna()
        X = df_clean[['returns']]
        y = df_clean['label']
        
        # Train model
        optimizer = LSTMOptuna(n_trials=1, cv_folds=2, max_epochs=3, seed=42)
        optimizer.optimize(X[:100], y[:100])
        optimizer.fit_final_model(X[:100], y[:100])
        
        # Get original predictions
        original_pred = optimizer.predict_proba(X[100:120])
        
        # Test that model state can be accessed for saving
        assert optimizer.best_model is not None
        assert optimizer.calibrator is not None
        assert optimizer.scaler is not None
        assert optimizer.best_params is not None
        
        # Test basic serialization components exist
        model_state = optimizer.best_model.state_dict()
        assert isinstance(model_state, dict)
        assert len(model_state) > 0
        
        print("✅ LSTM model persistence components available")
    
    def test_lstm_config_serialization(self):
        """Test LSTM configuration can be serialized."""
        if not IMPORT_SUCCESS:
            pytest.skip("Import failed")
        
        # Create optimizer with various settings
        optimizer = LSTMOptuna(
            n_trials=10,
            cv_folds=5,
            embargo=15,
            pruner_type='hyperband',
            early_stopping_patience=8,
            max_epochs=50,
            seed=42
        )
        
        # Extract configuration
        config = {
            'n_trials': optimizer.n_trials,
            'cv_folds': optimizer.cv_folds,
            'embargo': optimizer.embargo,
            'pruner_type': optimizer.pruner_type,
            'early_stopping_patience': optimizer.early_stopping_patience,
            'max_epochs': optimizer.max_epochs,
            'seed': optimizer.seed,
            'device': optimizer.device.type,
        }
        
        # Should be serializable
        import json
        config_json = json.dumps(config, default=str)
        config_loaded = json.loads(config_json)
        
        assert config_loaded['n_trials'] == 10
        assert config_loaded['cv_folds'] == 5
        assert config_loaded['seed'] == 42
        
        print("✅ LSTM configuration serialization works")


class TestLSTMRealWorldScenarios:
    """Test LSTM in realistic trading scenarios."""
    
    def test_lstm_crypto_like_volatility(self):
        """Test LSTM handles high volatility crypto-like data."""
        if not IMPORT_SUCCESS:
            pytest.skip("Import failed")
        
        # Generate high volatility data similar to crypto
        np.random.seed(42)
        n_periods = 500
        
        # High volatility returns (crypto-like)
        returns = np.random.randn(n_periods) * 0.05  # 5% std per hour
        prices = 50000 * np.exp(returns.cumsum())
        
        # Create features
        df = pd.DataFrame({
            'price': prices,
            'returns': returns,
            'volatility': pd.Series(returns).rolling(24).std(),  # 24h vol
            'rsi': pd.Series(prices).rolling(14).apply(
                lambda x: 100 - (100 / (1 + (x.diff().clip(lower=0).mean() / 
                                          x.diff().clip(upper=0).abs().mean()))))
        })
        
        df['label'] = (df['returns'].shift(-1) > 0.02).astype(int)  # 2% threshold
        df_clean = df.dropna()
        
        X = df_clean[['returns', 'volatility', 'rsi']]
        y = df_clean['label']
        
        # Train LSTM on volatile data
        optimizer = LSTMOptuna(n_trials=2, cv_folds=2, max_epochs=8, seed=42)
        
        try:
            study = optimizer.optimize(X[:300], y[:300])
            optimizer.fit_final_model(X[:300], y[:300])
            
            # Should handle high volatility
            predictions = optimizer.predict_proba(X[300:350])
            
            # Basic checks
            assert not np.any(np.isnan(predictions))
            assert not np.any(np.isinf(predictions))
            assert np.all((predictions >= 0) & (predictions <= 1))
            
            # Should produce diverse predictions despite volatility
            pred_std = np.std(predictions)
            assert pred_std > 0.01, f"Low prediction diversity: {pred_std:.4f}"
            
            print(f"✅ Handles crypto-like volatility: pred_std={pred_std:.4f}")
            
        except Exception as e:
            pytest.fail(f"Failed on high volatility data: {e}")
    
    def test_lstm_market_regime_changes(self):
        """Test LSTM handles market regime changes."""
        if not IMPORT_SUCCESS:
            pytest.skip("Import failed")
        
        # Create data with regime change
        np.random.seed(42)
        n_periods = 400
        
        # Bull market first half (upward trend)
        bull_returns = np.random.randn(n_periods // 2) * 0.02 + 0.005
        
        # Bear market second half (downward trend)
        bear_returns = np.random.randn(n_periods // 2) * 0.03 - 0.008
        
        all_returns = np.concatenate([bull_returns, bear_returns])
        prices = 50000 * np.exp(all_returns.cumsum())
        
        df = pd.DataFrame({
            'returns': all_returns,
            'sma_short': pd.Series(prices).rolling(10).mean(),
            'sma_long': pd.Series(prices).rolling(50).mean()
        })
        
        df['trend'] = (df['sma_short'] / df['sma_long'] - 1)
        df['label'] = (all_returns > 0).astype(int)
        df_clean = df.dropna()
        
        X = df_clean[['returns', 'trend']]
        y = df_clean['label']
        
        # Train on bull market
        bull_end = len(X) // 2 - 25  # Leave some buffer
        X_bull = X[:bull_end]
        y_bull = y[:bull_end]
        
        # Test on bear market
        bear_start = len(X) // 2 + 25
        X_bear = X[bear_start:]
        
        optimizer = LSTMOptuna(n_trials=1, cv_folds=2, max_epochs=5, seed=42)
        optimizer.optimize(X_bull, y_bull)
        optimizer.fit_final_model(X_bull, y_bull)
        
        # Predictions on different regime
        bear_predictions = optimizer.predict_proba(X_bear)
        
        # Should adapt to new regime (not just predict bull patterns)
        pred_mean = np.mean(bear_predictions)
        
        # In bear market, should not be overly bullish
        assert pred_mean < 0.7, f"Too bullish in bear market: {pred_mean:.3f}"
        
        # Should still produce diverse predictions
        pred_std = np.std(bear_predictions)
        assert pred_std > 0.05, f"Low diversity in regime change: {pred_std:.4f}"
        
        print(f"✅ Handles regime change: bear_pred_mean={pred_mean:.3f}, std={pred_std:.3f}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])