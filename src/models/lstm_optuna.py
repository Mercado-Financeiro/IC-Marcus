"""
LSTM with Bayesian optimization using Optuna - Backward compatibility wrapper.

This file maintains backward compatibility with existing code.
The actual implementation is now in src.models.lstm.optuna package.
"""

from src.models.lstm.optuna import (
    LSTMOptuna,
    LSTMModel,
    LSTMWrapper,
    LSTMOptunaConfig,
    set_lstm_deterministic,
    check_constant_predictions
)

# Export all for backward compatibility
__all__ = [
    'LSTMOptuna',
    'LSTMModel', 
    'LSTMWrapper',
    'LSTMOptunaConfig',
    'set_lstm_deterministic',
    'check_constant_predictions'
]

# For direct imports (backward compatibility)
def main():
    """Example usage."""
    import pandas as pd
    import numpy as np
    
    # Generate sample data
    n_samples = 1000
    n_features = 10
    
    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f'feature_{i}' for i in range(n_features)]
    )
    y = pd.Series(np.random.randint(0, 2, n_samples))
    
    # Initialize optimizer
    config = LSTMOptunaConfig(
        n_trials=10,
        cv_folds=3,
        max_epochs=50
    )
    optimizer = LSTMOptuna(config)
    
    # Optimize
    study = optimizer.optimize(X, y)
    
    # Fit final model
    optimizer.fit_final_model(X, y)
    
    # Predict
    predictions = optimizer.predict(X)
    probabilities = optimizer.predict_proba(X)
    
    print(f"Best score: {optimizer.best_score:.4f}")
    print(f"Best parameters: {optimizer.best_params}")


if __name__ == "__main__":
    main()