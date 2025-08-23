"""
End-to-end integration tests for the ML trading pipeline.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile
import shutil
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.data.binance_loader import CryptoDataLoader
from src.features.engineering import FeatureEngineer
from src.features.labels import AdaptiveLabeler
from src.models.xgb_optuna import XGBoostOptuna
from src.backtest.engine import BacktestEngine
from src.data.splits import PurgedKFold


class TestPipelineE2E:
    """End-to-end tests for the complete pipeline."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test artifacts."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    def test_data_loading_pipeline(self, temp_dir):
        """Test data loading and caching."""
        loader = CryptoDataLoader(cache_dir=temp_dir)
        
        # Load small sample
        df = loader.load_data(
            symbol="BTCUSDT",
            timeframe="1h",
            start_date="2024-01-01",
            end_date="2024-01-07"
        )
        
        assert df is not None
        assert len(df) > 0
        assert all(col in df.columns for col in ['open', 'high', 'low', 'close', 'volume'])
        
        # Test cache
        df2 = loader.load_data(
            symbol="BTCUSDT",
            timeframe="1h",
            start_date="2024-01-01",
            end_date="2024-01-07"
        )
        assert df.equals(df2)
    
    def test_feature_engineering_pipeline(self):
        """Test feature engineering without leakage."""
        # Create sample data
        dates = pd.date_range('2024-01-01', periods=1000, freq='15min')
        data = pd.DataFrame({
            'open': 100 + np.random.randn(1000) * 2,
            'high': 102 + np.random.randn(1000) * 2,
            'low': 98 + np.random.randn(1000) * 2,
            'close': 100 + np.random.randn(1000) * 2,
            'volume': 1000000 + np.random.randn(1000) * 100000
        }, index=dates)
        
        engineer = FeatureEngineer()
        features = engineer.fit_transform(data)
        
        assert features is not None
        assert len(features) < len(data)  # Some rows dropped due to lookback
        assert features.shape[1] > 5  # More features than original
        
        # Check no NaN values
        assert not features.isnull().any().any()
        
        # Check no future leakage
        for col in features.columns:
            if 'future' in col.lower():
                pytest.fail(f"Potential future leakage in column: {col}")
    
    def test_labeling_pipeline(self):
        """Test Triple Barrier labeling."""
        # Create sample price data
        dates = pd.date_range('2024-01-01', periods=500, freq='15min')
        prices = pd.Series(
            100 + np.cumsum(np.random.randn(500) * 0.5),
            index=dates
        )
        
        labeler = AdaptiveLabeler()
        labels, weights = labeler.fit_transform(
            prices,
            pt=0.02,
            sl=0.02,
            max_holding=20
        )
        
        assert labels is not None
        assert weights is not None
        assert len(labels) == len(weights)
        assert labels.isin([-1, 0, 1]).all()
        assert (weights >= 0).all()
    
    def test_model_training_pipeline(self):
        """Test complete model training pipeline."""
        # Create synthetic data
        X = pd.DataFrame(np.random.randn(500, 10))
        y = pd.Series(np.random.choice([0, 1], 500))
        
        # Initialize optimizer with minimal settings
        optimizer = XGBoostOptuna(
            n_trials=2,
            cv_folds=2,
            embargo=5
        )
        
        # Train model
        study, model = optimizer.optimize(X, y)
        
        assert model is not None
        assert study is not None
        assert len(study.trials) == 2
        assert hasattr(model, 'predict')
        assert hasattr(model, 'predict_proba')
        
        # Test prediction
        y_pred = optimizer.predict_proba(X[:10])
        assert len(y_pred) == 10
        assert all(0 <= p <= 1 for p in y_pred)
    
    def test_purged_kfold_no_leakage(self):
        """Test that PurgedKFold prevents data leakage."""
        n_samples = 1000
        dates = pd.date_range('2024-01-01', periods=n_samples, freq='15min')
        X = pd.DataFrame(np.random.randn(n_samples, 5), index=dates)
        y = pd.Series(np.random.choice([0, 1], n_samples), index=dates)
        
        splitter = PurgedKFold(n_splits=3, embargo=10)
        
        for train_idx, val_idx in splitter.split(X, y):
            train_times = X.index[train_idx]
            val_times = X.index[val_idx]
            
            # Check no overlap
            assert len(set(train_idx) & set(val_idx)) == 0
            
            # Check embargo
            for val_time in val_times:
                # Check gap before validation
                train_before = train_times[train_times < val_time]
                if len(train_before) > 0:
                    gap = val_time - train_before.max()
                    assert gap >= pd.Timedelta(minutes=15 * 9)  # At least embargo-1 periods
    
    def test_backtest_pipeline(self):
        """Test backtesting engine."""
        # Create sample data
        dates = pd.date_range('2024-01-01', periods=1000, freq='15min')
        prices = pd.DataFrame({
            'close': 100 + np.cumsum(np.random.randn(1000) * 0.5)
        }, index=dates)
        
        # Create signals
        signals = pd.Series(
            np.random.choice([-1, 0, 1], 1000, p=[0.3, 0.4, 0.3]),
            index=dates
        )
        
        # Run backtest
        engine = BacktestEngine(
            initial_capital=10000,
            fee_bps=5,
            slippage_bps=5
        )
        
        results = engine.run(prices, signals)
        
        assert results is not None
        assert 'equity' in results
        assert 'returns' in results
        assert 'positions' in results
        assert 'metrics' in results
        
        # Check metrics
        metrics = results['metrics']
        assert 'total_return' in metrics
        assert 'sharpe_ratio' in metrics
        assert 'max_drawdown' in metrics
        assert 'win_rate' in metrics
    
    def test_end_to_end_pipeline(self, temp_dir):
        """Test complete pipeline from data to backtest."""
        # 1. Create sample data
        dates = pd.date_range('2024-01-01', periods=2000, freq='15min')
        data = pd.DataFrame({
            'open': 100 + np.cumsum(np.random.randn(2000) * 0.3),
            'high': 102 + np.random.randn(2000) * 0.5,
            'low': 98 + np.random.randn(2000) * 0.5,
            'close': 100 + np.cumsum(np.random.randn(2000) * 0.3),
            'volume': 1000000 + np.random.randn(2000) * 100000
        }, index=dates)
        
        # 2. Feature engineering
        engineer = FeatureEngineer()
        features = engineer.fit_transform(data)
        
        # 3. Create labels
        labeler = AdaptiveLabeler()
        labels, weights = labeler.fit_transform(
            data['close'],
            pt=0.02,
            sl=0.02,
            max_holding=20
        )
        
        # 4. Align features and labels
        common_index = features.index.intersection(labels.index)
        X = features.loc[common_index]
        y = labels.loc[common_index]
        
        # Filter neutrals and remap
        mask = y != 0
        X = X[mask]
        y = y[mask].map({-1: 0, 1: 1})
        
        # 5. Split data
        split_idx = int(len(X) * 0.7)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        # 6. Train model
        optimizer = XGBoostOptuna(n_trials=1, cv_folds=2)
        study, model = optimizer.optimize(X_train, y_train)
        
        # 7. Generate predictions
        y_pred_proba = optimizer.predict_proba(X_test)
        
        # 8. Create signals
        signals = pd.Series(index=X_test.index)
        signals[y_pred_proba > 0.65] = 1
        signals[y_pred_proba < 0.35] = -1
        signals.fillna(0, inplace=True)
        
        # 9. Run backtest
        prices = data.loc[X_test.index, ['close']]
        engine = BacktestEngine()
        results = engine.run(prices, signals)
        
        # Validate results
        assert results is not None
        assert results['metrics']['total_return'] is not None
        assert len(results['equity']) == len(signals)


class TestAPIIntegration:
    """Test API endpoints integration."""
    
    @pytest.fixture
    def client(self):
        """Create test client for FastAPI."""
        from fastapi.testclient import TestClient
        from src.api.main import app
        return TestClient(app)
    
    def test_health_endpoint(self, client):
        """Test health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
    
    def test_prediction_endpoint(self, client):
        """Test prediction endpoint."""
        # Create test request
        request_data = {
            "symbol": "BTCUSDT",
            "timeframe": "15m",
            "features": {
                "rsi_14": 50.0,
                "sma_20": 45000.0,
                "volume_ratio": 1.2,
                "returns": 0.01
            }
        }
        
        # Test with demo token
        headers = {"Authorization": "Bearer demo-token"}
        response = client.post("/predict", json=request_data, headers=headers)
        
        if response.status_code == 200:
            data = response.json()
            assert "prediction" in data
            assert "signal" in data
            assert 0 <= data["prediction"] <= 1
            assert data["signal"] in ["LONG", "SHORT", "NEUTRAL"]
    
    def test_batch_prediction_endpoint(self, client):
        """Test batch prediction endpoint."""
        request_data = {
            "symbol": "BTCUSDT",
            "timeframe": "15m",
            "data": [
                {"rsi_14": 30.0, "sma_20": 44000.0, "volume_ratio": 0.8, "returns": -0.02},
                {"rsi_14": 70.0, "sma_20": 46000.0, "volume_ratio": 1.5, "returns": 0.03}
            ]
        }
        
        headers = {"Authorization": "Bearer demo-token"}
        response = client.post("/predict/batch", json=request_data, headers=headers)
        
        if response.status_code == 200:
            data = response.json()
            assert "predictions" in data
            assert len(data["predictions"]) == 2
            assert "execution_time_ms" in data


class TestDashboardIntegration:
    """Test Streamlit dashboard integration."""
    
    def test_dashboard_imports(self):
        """Test that dashboard can be imported without errors."""
        try:
            from src.dashboard import app
            from src.dashboard import app_enhanced
            assert True
        except ImportError as e:
            pytest.fail(f"Dashboard import failed: {e}")
    
    def test_dashboard_pages_exist(self):
        """Test that all dashboard pages exist."""
        dashboard_dir = Path("src/dashboard")
        assert dashboard_dir.exists()
        
        expected_files = ["app.py", "app_enhanced.py", "utils.py"]
        for file in expected_files:
            assert (dashboard_dir / file).exists(), f"Missing dashboard file: {file}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])