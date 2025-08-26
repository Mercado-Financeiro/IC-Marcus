"""Tests for deterministic behavior across the project."""

import pytest
import numpy as np
import pandas as pd
import random
import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.utils.determinism import set_deterministic_environment, verify_determinism


class TestDeterminism:
    """Test suite for deterministic behavior."""
    
    @pytest.mark.determinism
    def test_set_deterministic_environment(self):
        """Test that deterministic environment is properly set."""
        set_deterministic_environment(seed=42)
        
        # Check environment variables
        assert os.environ.get('PYTHONHASHSEED') == '42'
        assert os.environ.get('CUBLAS_WORKSPACE_CONFIG') == ':4096:8'
        assert os.environ.get('CUDA_LAUNCH_BLOCKING') == '1'
        
        # Check Python random
        random1 = [random.random() for _ in range(10)]
        
        set_deterministic_environment(seed=42)
        random2 = [random.random() for _ in range(10)]
        
        assert random1 == random2, "Python random not deterministic"
        
        # Check NumPy random
        set_deterministic_environment(seed=42)
        np1 = np.random.randn(10)
        
        set_deterministic_environment(seed=42)
        np2 = np.random.randn(10)
        
        np.testing.assert_array_equal(np1, np2, "NumPy random not deterministic")
    
    @pytest.mark.determinism
    def test_numpy_operations_deterministic(self):
        """Test that NumPy operations are deterministic."""
        set_deterministic_environment(seed=123)
        
        # Random array generation
        arr1 = np.random.randn(100, 50)
        
        set_deterministic_environment(seed=123)
        arr2 = np.random.randn(100, 50)
        
        np.testing.assert_array_equal(arr1, arr2)
        
        # Complex operations
        result1 = np.linalg.svd(arr1)[1]  # Singular values
        result2 = np.linalg.svd(arr2)[1]
        
        np.testing.assert_array_almost_equal(result1, result2)
    
    @pytest.mark.determinism
    def test_pandas_operations_deterministic(self):
        """Test that Pandas operations are deterministic."""
        set_deterministic_environment(seed=456)
        
        # Create random DataFrame
        df1 = pd.DataFrame(
            np.random.randn(100, 5),
            columns=['A', 'B', 'C', 'D', 'E']
        )
        
        set_deterministic_environment(seed=456)
        df2 = pd.DataFrame(
            np.random.randn(100, 5),
            columns=['A', 'B', 'C', 'D', 'E']
        )
        
        pd.testing.assert_frame_equal(df1, df2)
        
        # Sampling operations
        set_deterministic_environment(seed=789)
        sample1 = df1.sample(n=10)
        
        set_deterministic_environment(seed=789)
        sample2 = df1.sample(n=10)
        
        pd.testing.assert_frame_equal(sample1, sample2)
    
    @pytest.mark.determinism
    @pytest.mark.skipif(
        not pytest.importorskip("torch", reason="PyTorch not installed"),
        reason="PyTorch required"
    )
    def test_torch_deterministic(self):
        """Test that PyTorch operations are deterministic."""
        import torch
        
        set_deterministic_environment(seed=111)
        
        # Check torch configuration
        assert torch.backends.cudnn.deterministic == True
        assert torch.backends.cudnn.benchmark == False
        
        # Random tensor generation
        tensor1 = torch.randn(10, 5)
        
        set_deterministic_environment(seed=111)
        tensor2 = torch.randn(10, 5)
        
        assert torch.allclose(tensor1, tensor2), "Torch tensors not deterministic"
        
        # Neural network operations
        set_deterministic_environment(seed=222)
        
        model = torch.nn.Sequential(
            torch.nn.Linear(10, 5),
            torch.nn.ReLU(),
            torch.nn.Linear(5, 1)
        )
        
        input_tensor = torch.randn(32, 10)
        output1 = model(input_tensor)
        
        # Reset and recreate
        set_deterministic_environment(seed=222)
        
        model2 = torch.nn.Sequential(
            torch.nn.Linear(10, 5),
            torch.nn.ReLU(),
            torch.nn.Linear(5, 1)
        )
        
        output2 = model2(input_tensor)
        
        assert torch.allclose(output1, output2), "Neural network not deterministic"
    
    @pytest.mark.determinism
    def test_verify_determinism_function(self):
        """Test the verify_determinism utility function."""
        
        def deterministic_function(seed):
            """A function that should be deterministic."""
            set_deterministic_environment(seed)
            return np.random.randn(10)
        
        def non_deterministic_function(seed):
            """A function that is not deterministic."""
            # Not setting seed properly
            return np.random.randn(10)
        
        # Test deterministic function
        is_det, results = verify_determinism(
            lambda: deterministic_function(42),
            n_runs=3
        )
        assert is_det, "Deterministic function not detected as deterministic"
        
        # All results should be identical
        for i in range(1, len(results)):
            np.testing.assert_array_equal(results[0], results[i])
        
        # Test non-deterministic function
        is_det, results = verify_determinism(
            lambda: non_deterministic_function(42),
            n_runs=3
        )
        assert not is_det, "Non-deterministic function detected as deterministic"
    
    @pytest.mark.determinism
    def test_sklearn_deterministic(self):
        """Test that scikit-learn operations are deterministic."""
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import train_test_split
        
        set_deterministic_environment(seed=333)
        
        # Generate data
        X = np.random.randn(100, 10)
        y = np.random.randint(0, 2, 100)
        
        # Split data
        X_train1, X_test1, y_train1, y_test1 = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        X_train2, X_test2, y_train2, y_test2 = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        np.testing.assert_array_equal(X_train1, X_train2)
        np.testing.assert_array_equal(y_train1, y_train2)
        
        # Train model
        set_deterministic_environment(seed=444)
        
        model1 = RandomForestClassifier(n_estimators=10, random_state=42)
        model1.fit(X_train1, y_train1)
        pred1 = model1.predict_proba(X_test1)
        
        set_deterministic_environment(seed=444)
        
        model2 = RandomForestClassifier(n_estimators=10, random_state=42)
        model2.fit(X_train2, y_train2)
        pred2 = model2.predict_proba(X_test2)
        
        np.testing.assert_array_almost_equal(pred1, pred2)
    
    @pytest.mark.determinism
    def test_xgboost_deterministic(self):
        """Test that XGBoost is deterministic."""
        import xgboost as xgb
        
        set_deterministic_environment(seed=555)
        
        # Generate data
        X = np.random.randn(100, 10)
        y = np.random.randint(0, 2, 100)
        
        # Train model with deterministic settings
        params = {
            'objective': 'binary:logistic',
            'max_depth': 3,
            'seed': 42,
            'random_state': 42,
            'tree_method': 'exact',  # More deterministic than 'hist'
            'predictor': 'cpu_predictor'  # CPU is more deterministic
        }
        
        dtrain = xgb.DMatrix(X, label=y)
        
        model1 = xgb.train(params, dtrain, num_boost_round=10)
        pred1 = model1.predict(dtrain)
        
        model2 = xgb.train(params, dtrain, num_boost_round=10)
        pred2 = model2.predict(dtrain)
        
        np.testing.assert_array_almost_equal(pred1, pred2)
    
    @pytest.mark.determinism
    def test_hash_consistency(self):
        """Test that hash values are consistent."""
        set_deterministic_environment(seed=666)
        
        # Test string hashing
        test_strings = ["test1", "test2", "test3"]
        
        hashes1 = [hash(s) for s in test_strings]
        hashes2 = [hash(s) for s in test_strings]
        
        assert hashes1 == hashes2, "String hashes not consistent"
        
        # Test dictionary ordering (Python 3.7+)
        dict1 = {f"key{i}": i for i in range(100)}
        dict2 = {f"key{i}": i for i in range(100)}
        
        assert list(dict1.keys()) == list(dict2.keys()), "Dictionary ordering not consistent"
    
    @pytest.mark.determinism
    @pytest.mark.parametrize("seed", [0, 42, 123, 999])
    def test_different_seeds_produce_different_results(self, seed):
        """Test that different seeds produce different results."""
        set_deterministic_environment(seed=seed)
        result1 = np.random.randn(10)
        
        set_deterministic_environment(seed=seed + 1)
        result2 = np.random.randn(10)
        
        # Different seeds should produce different results
        assert not np.allclose(result1, result2), \
            f"Different seeds {seed} and {seed+1} produced same results"
        
        # Same seed should produce same results
        set_deterministic_environment(seed=seed)
        result3 = np.random.randn(10)
        
        np.testing.assert_array_equal(result1, result3)


class TestDeterminismInPipeline:
    """Test determinism in the ML pipeline."""
    
    @pytest.mark.determinism
    def test_feature_engineering_deterministic(self, sample_ohlcv_data):
        """Test that feature engineering is deterministic."""
        from src.features.engineering import FeatureEngineer
        
        set_deterministic_environment(seed=777)
        
        engineer1 = FeatureEngineer()
        features1 = engineer1.create_all_features(sample_ohlcv_data.copy())
        
        set_deterministic_environment(seed=777)
        
        engineer2 = FeatureEngineer()
        features2 = engineer2.create_all_features(sample_ohlcv_data.copy())
        
        pd.testing.assert_frame_equal(features1, features2)
    
    @pytest.mark.determinism
    def test_triple_barrier_deterministic(self, sample_ohlcv_data):
        """Test that triple barrier labeling is deterministic."""
        from src.features.labels import AdaptiveLabeler
        
        set_deterministic_environment(seed=888)
        
        labeler1 = AdaptiveLabeler()
        df1, info1 = labeler1.apply_triple_barrier(sample_ohlcv_data.copy())
        
        set_deterministic_environment(seed=888)
        
        labeler2 = AdaptiveLabeler()
        df2, info2 = labeler2.apply_triple_barrier(sample_ohlcv_data.copy())
        
        pd.testing.assert_series_equal(df1['label'], df2['label'])
        
        # Check barrier info
        for i in range(min(10, len(info1))):
            assert info1[i]['label'] == info2[i]['label']
            assert info1[i]['exit_reason'] == info2[i]['exit_reason']
    
    @pytest.mark.determinism
    def test_purged_kfold_deterministic(self, sample_features_data):
        """Test that Purged K-Fold splits are deterministic."""
        from src.data.splits import PurgedKFold
        
        X = sample_features_data.drop('label', axis=1)
        y = sample_features_data['label']
        
        cv1 = PurgedKFold(n_splits=3, embargo=5)
        splits1 = list(cv1.split(X, y))
        
        cv2 = PurgedKFold(n_splits=3, embargo=5)
        splits2 = list(cv2.split(X, y))
        
        assert len(splits1) == len(splits2)
        
        for (train1, val1), (train2, val2) in zip(splits1, splits2):
            np.testing.assert_array_equal(train1, train2)
            np.testing.assert_array_equal(val1, val2)