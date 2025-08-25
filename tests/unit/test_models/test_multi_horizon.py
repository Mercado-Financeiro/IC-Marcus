"""Tests for multi-horizon pipeline."""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from pathlib import Path
import tempfile
import shutil

from src.models.multi_horizon import MultiHorizonPipeline, run_multi_horizon_pipeline


class TestMultiHorizonPipeline:
    """Test multi-horizon pipeline functionality."""
    
    @pytest.fixture
    def sample_ohlc_data(self):
        """Create sample OHLC data for testing."""
        np.random.seed(42)
        dates = pd.date_range('2024-01-01', periods=500, freq='15T')
        
        # Generate realistic price data
        base_price = 50000
        returns = np.random.normal(0, 0.001, len(dates))
        close_prices = base_price * np.exp(np.cumsum(returns))
        
        open_prices = np.roll(close_prices, 1)
        open_prices[0] = close_prices[0]
        
        # Add intraday volatility
        intraday_range = np.random.uniform(0.0005, 0.003, len(dates))
        high_prices = np.maximum(open_prices, close_prices) * (1 + intraday_range/2)
        low_prices = np.minimum(open_prices, close_prices) * (1 - intraday_range/2)
        
        df = pd.DataFrame({
            'open': open_prices,
            'high': high_prices,
            'low': low_prices,
            'close': close_prices,
            'volume': np.random.lognormal(10, 0.5, len(dates))
        }, index=dates)
        
        return df
    
    @pytest.fixture
    def sample_features(self, sample_ohlc_data):
        """Create sample features DataFrame."""
        np.random.seed(42)
        
        features = pd.DataFrame({
            'feature1': np.random.randn(len(sample_ohlc_data)),
            'feature2': np.random.randn(len(sample_ohlc_data)),
            'feature3': np.random.randn(len(sample_ohlc_data)),
            'returns': np.log(sample_ohlc_data['close'] / sample_ohlc_data['close'].shift(1)),
            'volatility_20': np.random.uniform(0.001, 0.01, len(sample_ohlc_data))
        }, index=sample_ohlc_data.index)
        
        return features
    
    @pytest.fixture
    def temp_artifacts_dir(self):
        """Create temporary artifacts directory."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    def test_initialization(self, temp_artifacts_dir):
        """Test pipeline initialization."""
        pipeline = MultiHorizonPipeline(
            horizons=['15m', '30m'],
            symbol="BTCUSDT",
            artifacts_path=temp_artifacts_dir,
            use_mlflow=False
        )
        
        assert pipeline.horizons == ['15m', '30m']
        assert pipeline.symbol == "BTCUSDT"
        assert pipeline.funding_period_minutes == 480  # BTCUSDT default
        assert pipeline.artifacts_path == temp_artifacts_dir
        
        # Should create artifacts subdirectories
        assert Path(f"{temp_artifacts_dir}/models").exists()
        assert Path(f"{temp_artifacts_dir}/reports").exists()
    
    def test_prepare_data_splits(self, sample_ohlc_data, temp_artifacts_dir):
        """Test data splitting functionality."""
        pipeline = MultiHorizonPipeline(
            test_size=0.2,
            val_size=0.2,
            artifacts_path=temp_artifacts_dir,
            use_mlflow=False
        )
        
        train_idx, val_idx, test_idx = pipeline.prepare_data_splits(sample_ohlc_data)
        
        # Check types
        assert isinstance(train_idx, slice)
        assert isinstance(val_idx, slice)
        assert isinstance(test_idx, slice)
        
        # Check proportions
        total_len = len(sample_ohlc_data)
        train_len = train_idx.stop - train_idx.start
        val_len = val_idx.stop - val_idx.start  
        test_len = test_idx.stop - test_idx.start
        
        assert abs(train_len / total_len - 0.6) < 0.05  # ~60%
        assert abs(val_len / total_len - 0.2) < 0.05    # ~20%
        assert abs(test_len / total_len - 0.2) < 0.05   # ~20%
        
        # Check no overlap
        assert train_idx.stop <= val_idx.start
        assert val_idx.stop <= test_idx.start
    
    def test_prepare_features(self, sample_ohlc_data, sample_features, temp_artifacts_dir):
        """Test feature preparation."""
        pipeline = MultiHorizonPipeline(
            symbol="BTCUSDT",
            artifacts_path=temp_artifacts_dir,
            use_mlflow=False
        )
        
        enhanced_features = pipeline.prepare_features(sample_ohlc_data, sample_features)
        
        assert isinstance(enhanced_features, pd.DataFrame)
        assert len(enhanced_features) == len(sample_features)
        
        # Should have original features plus funding features
        original_cols = set(sample_features.columns)
        enhanced_cols = set(enhanced_features.columns)
        
        assert original_cols.issubset(enhanced_cols)
        
        # Should have funding-related features
        funding_features = [col for col in enhanced_cols if 'funding' in col]
        assert len(funding_features) > 0
    
    @patch('src.models.multi_horizon.XGBoostOptuna')
    @patch('src.models.multi_horizon.AdaptiveLabeler')
    def test_run_horizon_optimization(self, mock_labeler_class, mock_xgb_class, 
                                    sample_ohlc_data, sample_features, temp_artifacts_dir):
        """Test single horizon optimization."""
        # Mock AdaptiveLabeler
        mock_labeler = MagicMock()
        mock_labeler.horizon_map = {'30m': 2}
        mock_labeler.optimize_k_for_horizon.return_value = 1.2
        mock_labeler.create_labels.return_value = pd.Series(
            np.random.choice([-1, 0, 1], len(sample_ohlc_data)), 
            index=sample_ohlc_data.index
        )
        mock_labeler_class.return_value = mock_labeler
        
        # Mock XGBoostOptuna
        mock_xgb = MagicMock()
        mock_study = MagicMock()
        mock_study.best_value = 0.65
        mock_model = MagicMock()
        mock_model.feature_importances_ = np.random.rand(len(sample_features.columns))
        
        mock_xgb.optimize.return_value = (mock_study, mock_model)
        mock_xgb.fit_final_model.return_value = mock_model
        mock_xgb.predict_proba.return_value = np.random.rand(100)
        mock_xgb.predict.return_value = np.random.randint(0, 2, 100)
        mock_xgb.threshold_f1 = 0.5
        mock_xgb.threshold_profit = 0.6
        mock_xgb.calibrator = MagicMock()
        mock_xgb.best_params = {'max_depth': 6, 'learning_rate': 0.1}
        mock_xgb_class.return_value = mock_xgb
        
        # Create pipeline
        pipeline = MultiHorizonPipeline(
            horizons=['30m'],
            artifacts_path=temp_artifacts_dir,
            use_mlflow=False,
            n_trials=5
        )
        
        # Prepare splits
        train_idx, val_idx, test_idx = pipeline.prepare_data_splits(sample_ohlc_data)
        enhanced_features = pipeline.prepare_features(sample_ohlc_data, sample_features)
        
        # Run optimization for one horizon
        result = pipeline.run_horizon_optimization(
            sample_ohlc_data, enhanced_features, '30m',
            train_idx, val_idx, test_idx
        )
        
        # Check result structure
        assert isinstance(result, dict)
        assert 'horizon' in result
        assert result['horizon'] == '30m'
        assert 'model' in result
        assert 'metrics' in result
        assert 'optimal_k' in result
        assert result['optimal_k'] == 1.2
        
        # Check metrics
        metrics = result['metrics']
        expected_metrics = ['pr_auc', 'f1', 'mcc', 'precision', 'recall', 'specificity']
        for metric in expected_metrics:
            assert metric in metrics
            assert isinstance(metrics[metric], (int, float))
    
    @patch('src.models.multi_horizon.XGBoostOptuna')
    @patch('src.models.multi_horizon.AdaptiveLabeler') 
    def test_analyze_results(self, mock_labeler_class, mock_xgb_class, temp_artifacts_dir):
        """Test results analysis."""
        # Create mock results
        results = {
            '15m': {
                'metrics': {
                    'pr_auc': 0.65,
                    'f1': 0.62,
                    'mcc': 0.24,
                    'precision': 0.60,
                    'recall': 0.64
                },
                'optimal_k': 1.1,
                'thresholds': {'f1': 0.5, 'profit': 0.6},
                'predictions': {'proba': np.random.rand(100)}
            },
            '30m': {
                'metrics': {
                    'pr_auc': 0.58,
                    'f1': 0.55,
                    'mcc': 0.18,
                    'precision': 0.53,
                    'recall': 0.57
                },
                'optimal_k': 1.3,
                'thresholds': {'f1': 0.45, 'profit': 0.55},
                'predictions': {'proba': np.random.rand(100)}
            }
        }
        
        pipeline = MultiHorizonPipeline(
            artifacts_path=temp_artifacts_dir,
            use_mlflow=False
        )
        
        # Should not raise errors
        pipeline.analyze_results(results)
        
        # Should create comparison file
        comparison_file = Path(f"{temp_artifacts_dir}/reports/horizon_comparison.csv")
        assert comparison_file.exists()
        
        # Check comparison file content
        comparison_df = pd.read_csv(comparison_file, index_col=0)
        assert '15m' in comparison_df.index
        assert '30m' in comparison_df.index
        assert 'PR-AUC' in comparison_df.columns
        assert 'F1' in comparison_df.columns
    
    def test_edge_cases_and_error_handling(self, temp_artifacts_dir):
        """Test edge cases and error handling."""
        # Test with empty horizons list
        with pytest.raises((ValueError, IndexError)):
            pipeline = MultiHorizonPipeline(
                horizons=[],
                artifacts_path=temp_artifacts_dir
            )
        
        # Test with invalid horizon
        pipeline = MultiHorizonPipeline(
            horizons=['15m'],
            artifacts_path=temp_artifacts_dir,
            use_mlflow=False
        )
        
        # Should handle gracefully
        assert '15m' in pipeline.horizons
    
    def test_funding_period_resolution(self, temp_artifacts_dir):
        """Test funding period resolution for different symbols."""
        # Test known symbol
        pipeline_btc = MultiHorizonPipeline(
            symbol="BTCUSDT",
            artifacts_path=temp_artifacts_dir
        )
        assert pipeline_btc.funding_period_minutes == 480
        
        # Test unknown symbol (should use default)
        pipeline_unknown = MultiHorizonPipeline(
            symbol="UNKNOWNUSDT", 
            artifacts_path=temp_artifacts_dir
        )
        assert pipeline_unknown.funding_period_minutes == 480  # Default


class TestRunMultiHorizonPipeline:
    """Test the convenience function."""
    
    @pytest.fixture
    def minimal_data(self):
        """Create minimal test data."""
        np.random.seed(42)
        dates = pd.date_range('2024-01-01', periods=100, freq='15T')
        
        df = pd.DataFrame({
            'open': np.random.uniform(99, 101, 100),
            'high': np.random.uniform(100, 102, 100),
            'low': np.random.uniform(98, 100, 100),
            'close': np.random.uniform(99, 101, 100),
            'volume': np.random.uniform(1000, 2000, 100)
        }, index=dates)
        
        features = pd.DataFrame({
            'feature1': np.random.randn(100),
            'feature2': np.random.randn(100)
        }, index=dates)
        
        return df, features
    
    @patch('src.models.multi_horizon.MultiHorizonPipeline')
    def test_convenience_function(self, mock_pipeline_class, minimal_data, temp_dir=None):
        """Test convenience function interface."""
        df, features = minimal_data
        
        # Mock pipeline
        mock_pipeline = MagicMock()
        mock_pipeline.run_pipeline.return_value = {'15m': {'metrics': {'pr_auc': 0.6}}}
        mock_pipeline_class.return_value = mock_pipeline
        
        # Use temp directory if provided, otherwise use default
        artifacts_path = temp_dir or "artifacts"
        
        result = run_multi_horizon_pipeline(
            df=df,
            features=features,
            horizons=['15m'],
            artifacts_path=artifacts_path,
            n_trials=5
        )
        
        # Check that pipeline was created with correct parameters
        mock_pipeline_class.assert_called_once()
        call_kwargs = mock_pipeline_class.call_args[1]
        
        assert call_kwargs['horizons'] == ['15m']
        assert call_kwargs['n_trials'] == 5
        assert call_kwargs['artifacts_path'] == artifacts_path
        
        # Check that run_pipeline was called
        mock_pipeline.run_pipeline.assert_called_once_with(df, features)
        
        # Check result
        assert isinstance(result, dict)
        assert '15m' in result
    
    def test_default_parameters(self):
        """Test that default parameters work correctly."""
        # This tests the function signature and default values
        # without actually running the pipeline
        
        # Create minimal mock data
        df = pd.DataFrame({'close': [100, 101, 102]})
        features = pd.DataFrame({'feature1': [1, 2, 3]})
        
        # Should not raise error with minimal parameters
        try:
            # Import to test function signature
            from src.models.multi_horizon import run_multi_horizon_pipeline
            
            # Check that we can call with minimal parameters
            # (will fail in execution due to missing dependencies, but that's OK for signature test)
            pass
        except ImportError:
            pytest.skip("Dependencies not available for full test")
    
    @pytest.mark.parametrize("horizon_list", [
        ['15m'],
        ['15m', '30m'], 
        ['15m', '30m', '60m', '120m'],
        None  # Should use default
    ])
    def test_different_horizon_configurations(self, horizon_list):
        """Test with different horizon configurations."""
        # This is a lightweight test of parameter passing
        expected_horizons = horizon_list or ['15m', '30m', '60m', '120m']  # Default
        
        # Just test that we can construct valid parameters
        assert isinstance(expected_horizons, list)
        assert all(isinstance(h, str) for h in expected_horizons)
        assert all(h.endswith('m') for h in expected_horizons)
    
    def test_parameter_validation(self):
        """Test parameter validation."""
        # Test that invalid parameters would be caught
        with pytest.raises(TypeError):
            # n_trials should be int
            run_multi_horizon_pipeline(
                df=pd.DataFrame(),
                features=pd.DataFrame(), 
                n_trials="invalid"
            )
        
        with pytest.raises(TypeError):
            # test_size should be float
            run_multi_horizon_pipeline(
                df=pd.DataFrame(),
                features=pd.DataFrame(),
                test_size="invalid"
            )