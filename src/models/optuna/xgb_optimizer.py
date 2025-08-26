"""
XGBoost implementation of the unified optimizer.
This replaces the duplicate XGBoost optimizer implementations.
"""

from typing import Any, Dict, Optional, Tuple
import numpy as np
import optuna
from xgboost import XGBClassifier

from .unified_optimizer import UnifiedModelOptimizer, UnifiedOptunaConfig


class XGBoostUnifiedOptimizer(UnifiedModelOptimizer):
    """
    XGBoost implementation of unified optimizer.
    
    This class replaces:
    - src/models/xgb/optuna/optimizer.py
    - src/models/xgb/optuna/optimizer_enhanced.py
    - src/models/xgb_optuna.py (optimization parts)
    
    Example:
        >>> config = UnifiedOptunaConfig(
        ...     model_type='xgb',
        ...     n_trials=50,
        ...     pruner_type='hyperband',
        ...     primary_metric='pr_auc'
        ... )
        >>> optimizer = XGBoostUnifiedOptimizer(config)
        >>> study = optimizer.optimize(X_train, y_train)
        >>> print(f"Best score: {optimizer.best_score:.4f}")
    """
    
    def _create_search_space(self, trial: optuna.Trial) -> Dict[str, Any]:
        """
        Create XGBoost-specific hyperparameter search space.
        
        This is a comprehensive search space that covers all important
        XGBoost hyperparameters with sensible ranges.
        """
        return {
            # Tree parameters
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            
            # Learning parameters
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'gamma': trial.suggest_float('gamma', 0, 5),
            
            # Sampling parameters
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.5, 1.0),
            'colsample_bynode': trial.suggest_float('colsample_bynode', 0.5, 1.0),
            
            # Regularization parameters
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
            'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
            
            # Advanced parameters (optional)
            'max_delta_step': trial.suggest_float('max_delta_step', 0, 10),
            'scale_pos_weight': trial.suggest_float('scale_pos_weight', 0.5, 5),
        }
    
    def _create_model(self, params: Dict[str, Any]) -> XGBClassifier:
        """
        Create XGBoost model with given parameters.
        
        Adds fixed parameters for consistency and determinism.
        """
        # Add fixed parameters
        model_params = params.copy()
        model_params.update({
            'objective': 'binary:logistic',
            'eval_metric': 'aucpr',  # Use PR-AUC for imbalanced data
            'use_label_encoder': False,
            'random_state': self.config.seed,
            'n_jobs': 1,  # For determinism
            'tree_method': 'hist',  # Fast and deterministic
            'enable_categorical': False,
            'verbosity': 0,
        })
        
        return XGBClassifier(**model_params)
    
    def _train_model(
        self,
        model: XGBClassifier,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        trial: Optional[optuna.Trial] = None
    ) -> Tuple[XGBClassifier, Dict[str, float]]:
        """
        Train XGBoost model with early stopping and pruning.
        
        Args:
            model: XGBoost model instance
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            trial: Optuna trial for pruning (optional)
            
        Returns:
            Tuple of (trained_model, validation_metrics)
        """
        # Prepare evaluation set
        eval_set = [(X_train, y_train)]
        if X_val is not None and y_val is not None:
            eval_set.append((X_val, y_val))
        
        # Prepare callbacks
        callbacks = []
        if trial and X_val is not None:
            # Add pruning callback for early stopping of bad trials
            from optuna.integration import XGBoostPruningCallback
            callbacks.append(
                XGBoostPruningCallback(trial, 'validation_1-aucpr')
            )
        
        # Train model
        model.fit(
            X_train, y_train,
            eval_set=eval_set,
            early_stopping_rounds=self.config.early_stopping_patience,
            verbose=False,
            callbacks=callbacks if callbacks else None
        )
        
        # Calculate metrics
        metrics = {}
        if X_val is not None and y_val is not None:
            y_pred = model.predict(X_val)
            y_proba = model.predict_proba(X_val)[:, 1]
            metrics = self._calculate_metrics(y_val, y_pred, y_proba)
            
            # Add XGBoost-specific metrics
            metrics['best_iteration'] = model.best_iteration
            metrics['n_estimators_used'] = model.best_ntree_limit
        
        return model, metrics
    
    def get_feature_importance(self, importance_type: str = 'gain') -> pd.DataFrame:
        """
        Get feature importance from best XGBoost model.
        
        Args:
            importance_type: Type of importance ('gain', 'weight', 'cover')
            
        Returns:
            DataFrame with feature importance
        """
        if self.best_model is None:
            raise ValueError("No model available. Run optimize() first.")
        
        import pandas as pd
        
        # Get importance based on type
        if importance_type == 'gain':
            importance = self.best_model.feature_importances_
        else:
            importance = self.best_model.get_booster().get_score(
                importance_type=importance_type
            )
            # Convert to array format
            importance_array = np.zeros(self.X.shape[1])
            for feat_idx, score in importance.items():
                idx = int(feat_idx[1:])  # Remove 'f' prefix
                importance_array[idx] = score
            importance = importance_array
        
        # Create DataFrame
        importance_df = pd.DataFrame({
            'feature': [f'feature_{i}' for i in range(len(importance))],
            'importance': importance,
            'importance_type': importance_type
        })
        
        return importance_df.sort_values('importance', ascending=False)
    
    def plot_optimization_history(self):
        """Plot optimization history using Optuna's built-in visualization."""
        if self.study is None:
            raise ValueError("No study available. Run optimize() first.")
        
        try:
            import optuna.visualization as vis
            import plotly.io as pio
            
            # Create optimization history plot
            fig = vis.plot_optimization_history(self.study)
            fig.show()
            
            # Create parameter importance plot
            fig_importance = vis.plot_param_importances(self.study)
            fig_importance.show()
            
        except ImportError:
            print("Plotly not installed. Install with: pip install plotly")
    
    def get_best_params_for_production(self) -> Dict[str, Any]:
        """
        Get best parameters formatted for production use.
        
        Returns:
            Dictionary of parameters ready for XGBClassifier
        """
        if self.best_params is None:
            raise ValueError("No best parameters available. Run optimize() first.")
        
        params = self.best_params.copy()
        params.update({
            'objective': 'binary:logistic',
            'eval_metric': 'aucpr',
            'use_label_encoder': False,
            'random_state': self.config.seed,
            'n_jobs': -1,  # Use all cores in production
            'tree_method': 'hist',
            'verbosity': 1,
        })
        
        return params


# Convenience function for quick optimization
def optimize_xgboost(
    X_train,
    y_train,
    n_trials: int = 50,
    validation_strategy: str = 'walkforward',
    primary_metric: str = 'pr_auc',
    **kwargs
) -> XGBoostUnifiedOptimizer:
    """
    Convenience function for quick XGBoost optimization.
    
    Args:
        X_train: Training features
        y_train: Training labels
        n_trials: Number of optimization trials
        validation_strategy: Validation strategy ('walkforward', 'purged_kfold')
        primary_metric: Primary metric to optimize
        **kwargs: Additional configuration parameters
        
    Returns:
        Trained optimizer with best model
        
    Example:
        >>> optimizer = optimize_xgboost(X_train, y_train, n_trials=30)
        >>> best_model = optimizer.best_model
        >>> predictions = best_model.predict(X_test)
    """
    config = UnifiedOptunaConfig(
        model_type='xgb',
        n_trials=n_trials,
        validation_strategy=validation_strategy,
        primary_metric=primary_metric,
        **kwargs
    )
    
    optimizer = XGBoostUnifiedOptimizer(config)
    optimizer.optimize(X_train, y_train)
    
    return optimizer