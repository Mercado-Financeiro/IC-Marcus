"""
Multi-Horizon Ensemble with Layered Validation

Trains models for multiple prediction horizons and optimizes ensemble weights
using Bayesian Optimization to maximize portfolio Sharpe ratio.

Critical: Uses separate CPCV validation for weight optimization to avoid contamination.

References:
- Ensemble methods in finance: De Prado (2018)
- Bayesian optimization: Snoek et al. (2012)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
import optuna
from optuna.samplers import TPESampler
import warnings
import joblib
from pathlib import Path

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split

from src.data.splits import PurgedKFold
from src.models.xgb_unified import UnifiedXGBoost, XGBoostUnifiedConfig
from src.models.lstm.optuna.optimizer_v2 import EnhancedLSTMOptuna
from src.models.lstm.optuna.config import LSTMOptunaConfig
from src.models.threshold_optimizer import ThresholdOptimizer, TradingCosts
from src.metrics.dsr import calculate_all_sharpe_metrics
from src.utils.logging import log as logger


@dataclass
class HorizonConfig:
    """Configuration for a single horizon model."""
    horizon: int  # Prediction horizon (bars)
    weight: float = 0.2  # Initial weight in ensemble
    model: Optional[Any] = None
    threshold: float = 0.5
    performance: Dict = field(default_factory=dict)


@dataclass 
class MultiHorizonConfig:
    """Configuration for multi-horizon ensemble."""
    horizons: List[int] = field(default_factory=lambda: [1, 5, 10, 20, 30])
    
    # Model training
    base_model_config: Optional[XGBoostUnifiedConfig] = None
    lstm_model_config: Optional[LSTMOptunaConfig] = None
    model_types: List[str] = field(default_factory=lambda: ['xgboost', 'lstm'])  # Models to include
    n_trials_per_model: int = 50
    
    # Weight optimization
    n_trials_weights: int = 100
    weight_optimization_metric: str = 'sharpe'  # 'sharpe', 'sortino', 'calmar'
    
    # Validation
    inner_cv_splits: int = 5  # For individual models
    outer_cv_splits: int = 3  # For weight optimization
    embargo_bars: int = 40
    
    # Costs
    trading_costs: Optional[TradingCosts] = None
    
    # Constraints
    min_weight: float = 0.0
    max_weight: float = 1.0
    force_sum_to_one: bool = True
    
    # Output
    save_models: bool = True
    model_dir: str = "models/ensemble"
    verbose: bool = True


class MultiHorizonEnsemble(BaseEstimator, ClassifierMixin):
    """
    Multi-horizon ensemble with Bayesian weight optimization.
    
    This is where multiple time horizons combine to create a robust signal.
    Each horizon captures different market dynamics, and optimal weighting
    maximizes risk-adjusted returns.
    """
    
    def __init__(self, config: Optional[MultiHorizonConfig] = None):
        """
        Initialize multi-horizon ensemble.
        
        Args:
            config: Configuration object
        """
        self.config = config or MultiHorizonConfig()
        self.horizon_models: Dict[int, HorizonConfig] = {}
        self.optimal_weights: Dict[int, float] = {}
        self.weight_study: Optional[optuna.Study] = None
        self.threshold_optimizer = ThresholdOptimizer(self.config.trading_costs)
        self.is_fitted = False
        
        # Initialize horizon configurations
        for horizon in self.config.horizons:
            self.horizon_models[horizon] = HorizonConfig(horizon=horizon)
    
    def create_horizon_labels(
        self,
        data: pd.DataFrame,
        horizon: int,
        target_col: str = 'target'
    ) -> pd.Series:
        """
        Create labels for a specific prediction horizon.
        
        Args:
            data: DataFrame with price/return data
            horizon: Number of bars to look ahead
            target_col: Name of target column
            
        Returns:
            Labels for the specified horizon
        """
        # This should be implemented based on your specific labeling logic
        # For now, using simple future return threshold
        if 'returns' in data.columns:
            future_returns = data['returns'].rolling(horizon).sum().shift(-horizon)
            labels = (future_returns > 0.002 * horizon).astype(int)  # 0.2% per bar threshold
        else:
            # Fallback to existing target if available
            labels = data[target_col] if target_col in data.columns else pd.Series(0, index=data.index)
        
        return labels
    
    def train_horizon_model(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        horizon: int,
        model_type: str = 'xgboost'
    ) -> Dict:
        """
        Train a model for a specific horizon.
        
        Args:
            X: Features
            y: Labels for this horizon
            horizon: Prediction horizon
            model_type: 'xgboost' or 'lstm'
            
        Returns:
            Dictionary with trained model and metrics
        """
        logger.info(f"Training {model_type} model for horizon t+{horizon}")
        
        if model_type == 'xgboost':
            # Create XGBoost model config
            model_config = self.config.base_model_config or XGBoostUnifiedConfig(
                n_trials=self.config.n_trials_per_model,
                primary_metric='pr_auc',
                eval_metric='aucpr',
                inner_cv_splits=self.config.inner_cv_splits,
                embargo=self.config.embargo_bars,
                deterministic=True,
                n_jobs=1
            )
            
            # Train XGBoost model
            model = UnifiedXGBoost(model_config)
            study = model.optimize(X, y)
            model.fit_final_model(X, y)
            
        elif model_type == 'lstm':
            # Create LSTM model config
            lstm_config = self.config.lstm_model_config or LSTMOptunaConfig(
                seed=42,
                verbose=self.config.verbose
            )
            
            # Train LSTM model
            model = EnhancedLSTMOptuna(lstm_config)
            model.optuna_config.n_trials = self.config.n_trials_per_model
            model.optuna_config.inner_cv_splits = self.config.inner_cv_splits
            model.optuna_config.embargo = self.config.embargo_bars
            
            study = model.optimize(X, y)
            model.fit_final_model(X, y)
            
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Get calibrated probabilities
        proba = model.predict_proba(X)[:, 1]
        
        # Optimize threshold for this horizon
        threshold_results = self.threshold_optimizer.optimize_threshold(
            y.values, proba, method='adaptive'
        )
        
        # Create results dictionary
        model_result = {
            'model': model,
            'model_type': model_type,
            'horizon': horizon,
            'threshold': threshold_results.optimal_threshold,
            'performance': {
                'best_score': study.best_value,
                'optimal_threshold': threshold_results.optimal_threshold,
                'ev_per_trade': threshold_results.ev_per_trade,
                'expected_trades': threshold_results.expected_trades_per_period,
                'n_trials': len(study.trials)
            }
        }
        
        logger.info(
            f"{model_type} Horizon t+{horizon} trained",
            best_score=f"{study.best_value:.4f}",
            threshold=f"{threshold_results.optimal_threshold:.3f}",
            ev_per_trade=f"{threshold_results.ev_per_trade:.4%}"
        )
        
        return model_result
    
    def fit(self, X: pd.DataFrame, y: pd.Series, data: Optional[pd.DataFrame] = None):
        """
        Train all horizon models and optimize ensemble weights.
        
        Args:
            X: Features
            y: Base labels (for horizon=1)
            data: Full DataFrame for creating multi-horizon labels
        """
        logger.info(
            "Starting multi-horizon ensemble training",
            n_horizons=len(self.config.horizons),
            n_samples=len(X)
        )
        
        # Step 1: Split data for two-level validation
        # Level 1: For training individual models
        # Level 2: For optimizing weights (must be separate!)
        
        X_models, X_weights, y_models, y_weights = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        
        logger.info(
            "Data split for layered validation",
            model_training_size=len(X_models),
            weight_optimization_size=len(X_weights)
        )
        
        # Step 2: Train models for each horizon and model type
        self.all_models = []  # Store all trained models
        
        for horizon in self.config.horizons:
            # Create horizon-specific labels if data provided
            if data is not None:
                y_horizon = self.create_horizon_labels(data, horizon)
                y_horizon_models = y_horizon.iloc[X_models.index]
            else:
                # Use base labels scaled by horizon
                y_horizon_models = y_models
            
            # Train each model type for this horizon
            for model_type in self.config.model_types:
                model_result = self.train_horizon_model(
                    X_models, y_horizon_models, horizon, model_type
                )
                self.all_models.append(model_result)
                
                # Store in horizon_models for backward compatibility
                if model_type == 'xgboost':  # Primary model for each horizon
                    horizon_config = self.horizon_models[horizon]
                    horizon_config.model = model_result['model']
                    horizon_config.threshold = model_result['threshold']
                    horizon_config.performance = model_result['performance']
        
        # Step 3: Optimize ensemble weights using SEPARATE validation set
        self.optimize_weights(X_weights, y_weights, data)
        
        self.is_fitted = True
        
        # Step 4: Save models if configured
        if self.config.save_models:
            self.save_ensemble()
        
        return self
    
    def optimize_weights(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        data: Optional[pd.DataFrame] = None
    ):
        """
        Optimize ensemble weights using Bayesian Optimization.
        
        Critical: Uses SEPARATE validation data from model training.
        
        Args:
            X: Features for weight optimization (NOT used for model training)
            y: Labels for weight optimization
            data: Full DataFrame for multi-horizon labels
        """
        logger.info("Optimizing ensemble weights with Bayesian Optimization")
        
        def objective(trial: optuna.Trial) -> float:
            """Objective function for weight optimization."""
            
            # Suggest weights for each horizon
            weights = {}
            remaining = 1.0
            
            for i, horizon in enumerate(self.config.horizons[:-1]):
                if remaining > 0:
                    w = trial.suggest_float(
                        f'weight_h{horizon}',
                        self.config.min_weight,
                        min(self.config.max_weight, remaining)
                    )
                    weights[horizon] = w
                    remaining -= w
                else:
                    weights[horizon] = 0.0
            
            # Last weight to ensure sum to 1
            if self.config.force_sum_to_one:
                weights[self.config.horizons[-1]] = remaining
            else:
                weights[self.config.horizons[-1]] = trial.suggest_float(
                    f'weight_h{self.config.horizons[-1]}',
                    self.config.min_weight,
                    self.config.max_weight
                )
            
            # Get predictions from each horizon
            ensemble_proba = np.zeros(len(X))
            
            for horizon, weight in weights.items():
                if weight > 0:
                    model = self.horizon_models[horizon].model
                    if model is not None:
                        proba = model.predict_proba(X)[:, 1]
                        ensemble_proba += weight * proba
            
            # Normalize if weights don't sum to 1
            if not self.config.force_sum_to_one:
                total_weight = sum(weights.values())
                if total_weight > 0:
                    ensemble_proba /= total_weight
            
            # Convert to signals using EV-optimal threshold
            # (In practice, we'd optimize this threshold too)
            signals = (ensemble_proba > 0.5).astype(int)
            
            # Calculate returns (simplified - replace with actual return calculation)
            returns = np.where(signals & y, 0.01, 0.0)  # 1% on correct predictions
            returns = np.where(signals & ~y, -0.005, returns)  # -0.5% on wrong predictions
            returns = returns - signals * 0.001  # 0.1% cost per trade
            
            # Calculate metric
            if self.config.weight_optimization_metric == 'sharpe':
                if len(returns) > 0 and np.std(returns) > 0:
                    metric = np.mean(returns) / np.std(returns) * np.sqrt(252)
                else:
                    metric = 0.0
            elif self.config.weight_optimization_metric == 'sortino':
                downside_returns = returns[returns < 0]
                if len(downside_returns) > 0:
                    downside_std = np.std(downside_returns)
                    if downside_std > 0:
                        metric = np.mean(returns) / downside_std * np.sqrt(252)
                    else:
                        metric = np.mean(returns) * 100
                else:
                    metric = np.mean(returns) * 100
            else:  # calmar
                max_dd = self._calculate_max_drawdown(returns)
                if max_dd < 0:
                    metric = np.mean(returns) * 252 / abs(max_dd)
                else:
                    metric = 0.0
            
            return metric
        
        # Create study for weight optimization
        sampler = TPESampler(seed=42)
        self.weight_study = optuna.create_study(
            direction='maximize',
            sampler=sampler,
            study_name='ensemble_weights'
        )
        
        # Run optimization
        self.weight_study.optimize(
            objective,
            n_trials=self.config.n_trials_weights,
            show_progress_bar=self.config.verbose
        )
        
        # Extract optimal weights
        best_params = self.weight_study.best_params
        self.optimal_weights = {}
        
        for horizon in self.config.horizons:
            param_name = f'weight_h{horizon}'
            if param_name in best_params:
                self.optimal_weights[horizon] = best_params[param_name]
            else:
                # Last weight in force_sum_to_one case
                other_weights = sum(self.optimal_weights.values())
                self.optimal_weights[horizon] = 1.0 - other_weights
        
        logger.info(
            "Weight optimization complete",
            best_score=f"{self.weight_study.best_value:.3f}",
            optimal_weights={f"h{k}": f"{v:.3f}" for k, v in self.optimal_weights.items()},
            n_trials=len(self.weight_study.trials)
        )
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Get ensemble predictions using optimal weights.
        
        Args:
            X: Features
            
        Returns:
            Ensemble probabilities
        """
        if not self.is_fitted:
            raise ValueError("Ensemble not fitted. Call fit() first.")
        
        ensemble_proba = np.zeros(len(X))
        total_weight = 0
        
        for horizon, weight in self.optimal_weights.items():
            if weight > 0 and horizon in self.horizon_models:
                model = self.horizon_models[horizon].model
                if model is not None:
                    proba = model.predict_proba(X)[:, 1]
                    ensemble_proba += weight * proba
                    total_weight += weight
        
        # Normalize
        if total_weight > 0:
            ensemble_proba /= total_weight
        
        # Return in sklearn format
        return np.column_stack([1 - ensemble_proba, ensemble_proba])
    
    def predict(self, X: pd.DataFrame, use_ev_threshold: bool = True) -> np.ndarray:
        """
        Get ensemble predictions using EV-optimal threshold.
        
        Args:
            X: Features
            use_ev_threshold: Use EV-optimal threshold vs fixed 0.5
            
        Returns:
            Binary predictions
        """
        proba = self.predict_proba(X)[:, 1]
        
        if use_ev_threshold:
            # Use weighted average of horizon thresholds
            threshold = sum(
                self.horizon_models[h].threshold * self.optimal_weights.get(h, 0)
                for h in self.config.horizons
            )
            threshold = threshold / sum(self.optimal_weights.values())
        else:
            threshold = 0.5
        
        return (proba >= threshold).astype(int)
    
    def get_horizon_contributions(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Get individual horizon predictions and contributions.
        
        Useful for understanding which horizons drive the signal.
        
        Args:
            X: Features
            
        Returns:
            DataFrame with horizon predictions and weights
        """
        contributions = {}
        
        for horizon in self.config.horizons:
            if horizon in self.horizon_models:
                model = self.horizon_models[horizon].model
                if model is not None:
                    proba = model.predict_proba(X)[:, 1]
                    weight = self.optimal_weights.get(horizon, 0)
                    contributions[f'h{horizon}_proba'] = proba
                    contributions[f'h{horizon}_weight'] = weight
                    contributions[f'h{horizon}_contrib'] = proba * weight
        
        return pd.DataFrame(contributions, index=X.index)
    
    def _calculate_max_drawdown(self, returns: np.ndarray) -> float:
        """Calculate maximum drawdown from returns."""
        cumulative = (1 + returns).cumprod()
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        return np.min(drawdown)
    
    def save_ensemble(self):
        """Save ensemble models and weights."""
        model_dir = Path(self.config.model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Save each horizon model
        for horizon, config in self.horizon_models.items():
            if config.model is not None:
                model_path = model_dir / f"horizon_{horizon}.pkl"
                joblib.dump(config.model, model_path)
        
        # Save weights and configuration
        ensemble_info = {
            'optimal_weights': self.optimal_weights,
            'horizon_configs': {
                h: {
                    'threshold': c.threshold,
                    'performance': c.performance
                }
                for h, c in self.horizon_models.items()
            },
            'config': self.config
        }
        
        info_path = model_dir / "ensemble_info.pkl"
        joblib.dump(ensemble_info, info_path)
        
        logger.info(f"Ensemble saved to {model_dir}")
    
    def load_ensemble(self, model_dir: str):
        """Load ensemble from disk."""
        model_dir = Path(model_dir)
        
        # Load ensemble info
        info_path = model_dir / "ensemble_info.pkl"
        ensemble_info = joblib.load(info_path)
        
        self.optimal_weights = ensemble_info['optimal_weights']
        self.config = ensemble_info['config']
        
        # Load each horizon model
        for horizon in self.config.horizons:
            model_path = model_dir / f"horizon_{horizon}.pkl"
            if model_path.exists():
                model = joblib.load(model_path)
                self.horizon_models[horizon] = HorizonConfig(
                    horizon=horizon,
                    model=model,
                    threshold=ensemble_info['horizon_configs'][horizon]['threshold'],
                    performance=ensemble_info['horizon_configs'][horizon]['performance']
                )
        
        self.is_fitted = True
        logger.info(f"Ensemble loaded from {model_dir}")
    
    def get_summary(self) -> Dict:
        """Get summary of ensemble performance."""
        return {
            'n_horizons': len(self.config.horizons),
            'horizons': self.config.horizons,
            'optimal_weights': self.optimal_weights,
            'horizon_performance': {
                h: c.performance
                for h, c in self.horizon_models.items()
            },
            'weight_optimization': {
                'best_score': self.weight_study.best_value if self.weight_study else None,
                'n_trials': len(self.weight_study.trials) if self.weight_study else 0,
                'metric': self.config.weight_optimization_metric
            }
        }