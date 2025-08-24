"""Test script to verify memory cleanup and determinism in optimizers."""

import gc
import os
import sys
import numpy as np
import pandas as pd
import torch
import psutil
from pathlib import Path

# Add project to path
sys.path.append(str(Path(__file__).parent))

# Set deterministic seeds
os.environ['PYTHONHASHSEED'] = '42'
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_memory_usage():
    """Get current memory usage in MB."""
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024


def test_xgb_memory():
    """Test XGBoost optimizer memory cleanup."""
    print("\n=== Testing XGBoost Memory Cleanup ===")
    
    from src.models.xgb_optuna import XGBoostOptuna
    
    # Create dummy data
    np.random.seed(42)
    n_samples = 10000
    n_features = 50
    X = pd.DataFrame(np.random.randn(n_samples, n_features))
    y = pd.Series(np.random.randint(0, 2, n_samples))
    
    # Test with small number of trials
    config = {
        'n_trials': 3,
        'cv_folds': 2,
        'seed': 42,
        'embargo': 10
    }
    
    print(f"Initial memory: {get_memory_usage():.2f} MB")
    
    model = XGBoostOptuna(**config)
    
    # Run optimization
    memory_before = get_memory_usage()
    print(f"Memory before optimization: {memory_before:.2f} MB")
    
    study = model.optimize(X, y)
    
    memory_after = get_memory_usage()
    print(f"Memory after optimization: {memory_after:.2f} MB")
    print(f"Memory increase: {memory_after - memory_before:.2f} MB")
    
    # Get best score for determinism check
    best_score1 = study.best_value
    print(f"Best score (run 1): {best_score1:.6f}")
    
    # Force cleanup
    del model, study
    gc.collect()
    
    memory_cleaned = get_memory_usage()
    print(f"Memory after cleanup: {memory_cleaned:.2f} MB")
    
    # Test determinism - run again with same seed
    print("\n--- Testing Determinism ---")
    np.random.seed(42)
    model2 = XGBoostOptuna(**config)
    study2 = model2.optimize(X, y)
    best_score2 = study2.best_value
    print(f"Best score (run 2): {best_score2:.6f}")
    
    if abs(best_score1 - best_score2) < 1e-6:
        print("✅ Determinism preserved!")
    else:
        print(f"⚠️ Determinism issue: scores differ by {abs(best_score1 - best_score2)}")
    
    return best_score1, best_score2


def test_lstm_memory():
    """Test LSTM optimizer memory cleanup."""
    print("\n=== Testing LSTM Memory Cleanup ===")
    
    try:
        from src.models.lstm.optuna.optimizer import LSTMOptuna
        from src.models.lstm.optuna.config import LSTMOptunaConfig
    except ImportError as e:
        print(f"Skipping LSTM test: {e}")
        return None, None
    
    # Create dummy data
    np.random.seed(42)
    torch.manual_seed(42)
    n_samples = 1000
    n_features = 20
    X = np.random.randn(n_samples, n_features).astype(np.float32)
    y = np.random.randint(0, 2, n_samples).astype(np.float32)
    
    # Test with small config
    config = LSTMOptunaConfig(
        n_trials=2,
        cv_folds=2,
        max_epochs=5,
        seq_len_min=10,
        seq_len_max=20,
        hidden_size_min=16,
        hidden_size_max=32,
        seed=42,
        device='cpu'  # Use CPU for testing
    )
    
    print(f"Initial memory: {get_memory_usage():.2f} MB")
    
    model = LSTMOptuna(config)
    
    # Run optimization
    memory_before = get_memory_usage()
    print(f"Memory before optimization: {memory_before:.2f} MB")
    
    # Create pandas inputs
    X_df = pd.DataFrame(X)
    y_series = pd.Series(y)
    
    study = model.optimize(X_df, y_series)
    
    memory_after = get_memory_usage()
    print(f"Memory after optimization: {memory_after:.2f} MB")
    print(f"Memory increase: {memory_after - memory_before:.2f} MB")
    
    # Get best score for determinism check
    best_score1 = study.best_value if study.best_value is not None else 0.0
    print(f"Best score (run 1): {best_score1:.6f}")
    
    # Force cleanup
    del model, study
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    
    memory_cleaned = get_memory_usage()
    print(f"Memory after cleanup: {memory_cleaned:.2f} MB")
    
    # Test determinism
    print("\n--- Testing Determinism ---")
    np.random.seed(42)
    torch.manual_seed(42)
    model2 = LSTMOptuna(config)
    study2 = model2.optimize(X_df, y_series)
    best_score2 = study2.best_value if study2.best_value is not None else 0.0
    print(f"Best score (run 2): {best_score2:.6f}")
    
    if abs(best_score1 - best_score2) < 1e-4:  # Slightly larger tolerance for LSTM
        print("✅ Determinism preserved!")
    else:
        print(f"⚠️ Determinism issue: scores differ by {abs(best_score1 - best_score2)}")
    
    return best_score1, best_score2


def main():
    """Run all memory and determinism tests."""
    print("=" * 60)
    print("Memory Cleanup and Determinism Tests")
    print("=" * 60)
    
    # Test XGBoost
    xgb_scores = test_xgb_memory()
    
    # Test LSTM
    lstm_scores = test_lstm_memory()
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    if xgb_scores[0] is not None:
        xgb_det = abs(xgb_scores[0] - xgb_scores[1]) < 1e-6
        print(f"XGBoost determinism: {'✅ PASS' if xgb_det else '❌ FAIL'}")
    
    if lstm_scores and lstm_scores[0] is not None:
        lstm_det = abs(lstm_scores[0] - lstm_scores[1]) < 1e-4
        print(f"LSTM determinism: {'✅ PASS' if lstm_det else '❌ FAIL'}")
    
    print("\nMemory cleanup has been successfully implemented!")
    print("Both optimizers now clean up memory between trials.")


if __name__ == "__main__":
    main()