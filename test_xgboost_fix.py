#!/usr/bin/env python3
"""
Simple test script to verify XGBoost early stopping fix.
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

print("Testing XGBoost with EarlyStopping callback...")
print(f"XGBoost version: {xgb.__version__}")

# Create dummy data
np.random.seed(42)
n_samples = 1000
n_features = 20

X = np.random.randn(n_samples, n_features)
y = np.random.randint(0, 2, n_samples)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create model parameters - early_stopping_rounds is a model parameter in XGBoost 2.1.4
params = {
    'n_estimators': 500,
    'max_depth': 5,
    'learning_rate': 0.1,
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'random_state': 42,
    'early_stopping_rounds': 50  # Set as model parameter
}

print("\nTraining XGBoost with early stopping...")

# Create model with early stopping parameter
model = xgb.XGBClassifier(**params)

# Fit model
model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=False
)

# Get predictions
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"[OK] Training completed successfully!")
print(f"   Best iteration: {model.best_iteration}")
print(f"   Test accuracy: {accuracy:.4f}")

# Test with the modular optimizer
print("\n" + "="*50)
print("Testing modular XGBoost optimizer...")

try:
    from src.models.xgb.optuna.optimizer import XGBOptuna
    from src.models.xgb.optuna.config import XGBOptunaConfig
    
    # Create configuration
    config = XGBOptunaConfig(
        n_trials=2,  # Just 2 trials for testing
        cv_folds=2,
        embargo=5,
        pruner_type='median',
        seed=42
    )
    
    # Create optimizer
    optimizer = XGBOptuna(config=config)
    
    # Convert to DataFrame
    X_df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(n_features)])
    y_series = pd.Series(y)
    
    print("Running optimization with 2 trials...")
    
    # Run optimization
    optimizer.optimize(X_df, y_series)
    
    print(f"[OK] Optimization completed successfully!")
    print(f"   Best score: {optimizer.best_score:.4f}")
    print(f"   Best parameters found!")
    
except Exception as e:
    print(f"[ERROR] Error during optimization: {str(e)}")
    import traceback
    traceback.print_exc()

print("\n" + "="*50)
print("All tests completed!")