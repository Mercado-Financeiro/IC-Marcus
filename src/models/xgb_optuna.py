"""XGBoost with Bayesian optimization using Optuna."""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, Any
import xgboost as xgb
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    f1_score, precision_recall_curve, auc, roc_auc_score,
    brier_score_loss, matthews_corrcoef
)
import optuna
from optuna.pruners import MedianPruner, HyperbandPruner, SuccessiveHalvingPruner
from optuna.samplers import TPESampler
try:
    import mlflow
    import mlflow.xgboost
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
# Logging will be initialized per instance to avoid issues

import warnings
from pathlib import Path
import sys

# Add parent to path (notebook-safe)
try:
    base_path = Path(__file__).parent.parent
except NameError:
    # In notebook, __file__ doesn't exist
    base_path = Path.cwd()
sys.path.append(str(base_path))

from data.splits import PurgedKFold
from utils.determinism import set_deterministic_environment

warnings.filterwarnings('ignore')


class XGBoostOptuna:
    """XGBoost classifier with Optuna optimization and mandatory calibration."""
    
    def __init__(
        self,
        n_trials: int = 100,
        cv_folds: int = 5,
        embargo: int = 10,
        pruner_type: str = 'hyperband',
        use_mlflow: bool = True,
        seed: int = 42
    ):
        """Initialize XGBoost optimizer.
        
        Args:
            n_trials: Number of Optuna trials
            cv_folds: Number of CV folds
            embargo: Embargo period for Purged K-Fold
            pruner_type: Type of pruner (median, hyperband, successive_halving)
            use_mlflow: Whether to log to MLflow
            seed: Random seed
        """
        self.n_trials = n_trials
        self.cv_folds = cv_folds
        self.embargo = embargo
        self.pruner_type = pruner_type
        self.use_mlflow = use_mlflow
        self.seed = seed
        
        self.best_model = None
        self.best_params = None
        self.best_score = None
        self.calibrator = None
        self.threshold_f1 = 0.5
        self.threshold_ev = 0.5
        
        # Initialize logging (instance-level)
        self.log = self._setup_logging()
        
        # Set deterministic environment
        set_deterministic_environment(seed)
        
        self.log.info(
            "xgboost_optuna_initialized",
            n_trials=n_trials,
            cv_folds=cv_folds,
            embargo=embargo,
            pruner=pruner_type
        )
    
    def _setup_logging(self):
        """Setup logging with fallback for missing structlog."""
        try:
            import structlog
            return structlog.get_logger()
        except ImportError:
            import logging
            logging.basicConfig(level=logging.INFO)
            _pylog = logging.getLogger(__name__)
            
            class LogShim:
                """Shim to make logging work like structlog."""
                def info(self, event, **kw):
                    extras = ", ".join(f"{k}={v}" for k, v in kw.items())
                    _pylog.info(f"{event} | {extras}" if extras else event)
                
                def warning(self, event, **kw):
                    extras = ", ".join(f"{k}={v}" for k, v in kw.items())
                    _pylog.warning(f"{event} | {extras}" if extras else event)
                
                def error(self, event, **kw):
                    extras = ", ".join(f"{k}={v}" for k, v in kw.items())
                    _pylog.error(f"{event} | {extras}" if extras else event)
            
            return LogShim()
    
    def _calculate_scale_pos_weight(self, y: pd.Series) -> float:
        """Calculate scale_pos_weight for class imbalance."""
        n_neg = (y == 0).sum()
        n_pos = (y == 1).sum()
        
        if n_pos == 0:
            return 1.0
        
        weight = n_neg / n_pos
        # Clip to reasonable range to avoid extreme values
        return float(np.clip(weight, 0.1, 10.0))
    
    def _create_search_space(self, trial: optuna.Trial) -> Dict:
        """Create hyperparameter search space.
        
        Args:
            trial: Optuna trial
            
        Returns:
            Dictionary of hyperparameters
        """
        params = {
            'max_depth': trial.suggest_int('max_depth', 3, 8),
            'min_child_weight': trial.suggest_float('min_child_weight', 0.5, 3.0),  # Menos restritivo
            'gamma': trial.suggest_float('gamma', 0.0, 0.5),  # Muito menos agressivo
            'subsample': trial.suggest_float('subsample', 0.7, 1.0),  # Range menor, valores maiores
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 1.0),  # Range menor
            'learning_rate': trial.suggest_float('learning_rate', 0.05, 0.3, log=True),  # LR mínimo maior
            'reg_lambda': trial.suggest_float('reg_lambda', 0.01, 1.0),  # Muito menos agressivo
            'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 0.5),  # Menos agressivo
            'max_delta_step': trial.suggest_float('max_delta_step', 0, 3),  # Para class imbalance
            'max_bin': 256,
            'n_estimators': 2000,  # Large number for early stopping
            
            # Fixed parameters
            'tree_method': 'hist',
            'objective': 'binary:logistic',
            'eval_metric': 'aucpr',  # Better for imbalanced data
            'random_state': self.seed,
            'verbosity': 0,  # Reduce noise
            'n_jobs': -1,
            'early_stopping_rounds': 50  # Menos paciência para detectar não-aprendizado
        }
        
        return params
    
    def _get_pruner(self) -> optuna.pruners.BasePruner:
        """Get Optuna pruner based on type.
        
        Returns:
            Optuna pruner
        """
        if self.pruner_type == 'median':
            return MedianPruner(n_startup_trials=5, n_warmup_steps=5)
        elif self.pruner_type == 'hyperband':
            return HyperbandPruner(min_resource=1, max_resource=self.cv_folds)
        elif self.pruner_type == 'successive_halving':
            return SuccessiveHalvingPruner()
        else:
            return MedianPruner()
    
    def _optimize_threshold_f1(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
        """Optimize threshold for F1 score.
        
        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            
        Returns:
            Optimal threshold
        """
        precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
        
        # Handle edge cases
        f1_scores = np.nan_to_num(f1_scores)
        
        if len(f1_scores) > 0 and len(thresholds) > 0:
            best_idx = np.argmax(f1_scores[:-1])  # Last value is for threshold=1
            return float(thresholds[best_idx])
        
        return 0.5
    
    def _optimize_threshold_ev(
        self, y_true: np.ndarray, y_pred_proba: np.ndarray, costs: Dict
    ) -> float:
        """Optimize threshold for Expected Value.
        
        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            costs: Dictionary with fee_bps and slippage_bps
            
        Returns:
            Optimal threshold for EV
        """
        thresholds = np.linspace(0.05, 0.95, 181)
        best_ev = -np.inf
        best_threshold = 0.5
        
        total_cost_bps = costs.get('fee_bps', 5) + costs.get('slippage_bps', 5)
        
        for threshold in thresholds:
            # Generate signals
            signals = (y_pred_proba >= threshold).astype(int)
            
            # Calculate returns (simplified)
            # Assume 1% return for correct predictions
            returns = np.where(signals == y_true, 0.01, -0.01)
            
            # Apply costs
            trades = np.diff(np.concatenate([[0], signals]))
            n_trades = np.abs(trades).sum()
            
            if len(signals) > 0:
                avg_return = returns.mean()
                trade_cost = (n_trades / len(signals)) * total_cost_bps / 10000
                ev = avg_return - trade_cost
                
                if ev > best_ev:
                    best_ev = ev
                    best_threshold = threshold
        
        return float(best_threshold)
    
    def _create_objective(
        self, X: pd.DataFrame, y: pd.Series, sample_weights: Optional[np.ndarray] = None
    ):
        """Create objective function for Optuna.
        
        Args:
            X: Features
            y: Labels
            sample_weights: Optional sample weights
            
        Returns:
            Objective function
        """
        # Calculate scale_pos_weight consistently
        pos_weight = self._calculate_scale_pos_weight(y)
        
        def objective(trial):
            # Get hyperparameters
            params = self._create_search_space(trial)
            # Always add scale_pos_weight for consistency
            params['scale_pos_weight'] = pos_weight
            
            # Debug: Print first trial params
            if trial.number == 0:
                self.log.info("first_trial_params", params=params)
            
            # Create cross-validator
            cv = PurgedKFold(n_splits=self.cv_folds, embargo=self.embargo)
            
            scores = []
            
            for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X, y)):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                
                # Sample weights
                if sample_weights is not None:
                    w_train = sample_weights[train_idx]
                else:
                    w_train = None
                
                # Create model with early stopping (XGBoost 3.x API)
                model = xgb.XGBClassifier(**params)
                
                # Fit model with eval_set (early stopping handled by constructor)
                model.fit(
                    X_train, y_train,
                    sample_weight=w_train,
                    eval_set=[(X_val, y_val)],
                    verbose=False
                )
                
                # Skip calibration during optimization (only calibrate final model)
                # This avoids overfitting on validation set
                y_pred_proba = model.predict_proba(X_val)[:, 1]
                
                # Check if predictions are constant (indicates model not learning)
                pred_std = y_pred_proba.std()
                pred_mean = y_pred_proba.mean()
                unique_preds = len(np.unique(np.round(y_pred_proba, 3)))
                pred_range = y_pred_proba.max() - y_pred_proba.min()
                
                # Critérios mais flexíveis: múltiplas verificações
                is_constant = (pred_std < 0.001 or 
                              unique_preds < 5 or 
                              pred_range < 0.01 or
                              np.allclose(y_pred_proba, pred_mean, atol=0.001))
                
                if is_constant:
                    self.log.warning("constant_predictions_detected", 
                                   trial=trial.number, pred_std=pred_std, 
                                   pred_mean=pred_mean, unique_count=unique_preds,
                                   pred_range=pred_range)
                    self.log.warning("sample_predictions", samples=y_pred_proba[:10].tolist())
                    return -1.0  # Penalize trials with constant predictions
                
                # Optimize threshold for F1
                threshold = self._optimize_threshold_f1(y_val, y_pred_proba)
                y_pred = (y_pred_proba >= threshold).astype(int)
                
                # Calculate metrics
                f1 = f1_score(y_val, y_pred, zero_division=0)
                
                # PR-AUC
                precision, recall, _ = precision_recall_curve(y_val, y_pred_proba)
                pr_auc = auc(recall, precision)
                
                # Brier score (lower is better)
                brier = brier_score_loss(y_val, y_pred_proba)
                
                # Matthews correlation coefficient
                mcc = matthews_corrcoef(y_val, y_pred)
                
                # Composite score (maximize)
                score = 0.4 * f1 + 0.3 * pr_auc + 0.2 * mcc - 0.1 * brier
                scores.append(score)
                
                # Log trial metrics for debugging
                if fold_idx == 0:  # Log only first fold to avoid spam
                    self.log.info(
                        f"Trial {trial.number} Fold {fold_idx}: "
                        f"f1={f1:.4f}, pr_auc={pr_auc:.4f}, "
                        f"mcc={mcc:.4f}, brier={brier:.4f}, "
                        f"score={score:.4f}"
                    )
                
                # Report for pruning
                trial.report(score, fold_idx)
                
                # Prune if needed
                if trial.should_prune():
                    raise optuna.TrialPruned()
            
            return np.mean(scores)
        
        return objective
    
    def optimize(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        sample_weights: Optional[np.ndarray] = None
    ) -> Tuple[optuna.Study, Any]:
        """Run Bayesian optimization.
        
        Args:
            X: Features
            y: Labels
            sample_weights: Optional sample weights
            
        Returns:
            Tuple of (study, best_model)
        """
        # Validate input
        if len(X) == 0 or len(y) == 0:
            raise ValueError("Empty data provided")
        
        # Check for NaN/Inf in features
        if X.isna().any().any() or np.isinf(X.values).any():
            n_nan = X.isna().sum().sum()
            n_inf = np.isinf(X.values).sum()
            raise ValueError(f"Input data contains NaN ({n_nan}) or Inf ({n_inf}) values. Clean data first.")
        
        # Check for NaN in labels
        if y.isna().any():
            raise ValueError("Labels contain NaN values")
        
        if len(np.unique(y)) < 2:
            raise ValueError("Only single class in labels")
        
        if len(X) < self.cv_folds * 2:
            raise ValueError(f"Insufficient data for {self.cv_folds}-fold CV")
        
        self.log.info(
            "starting_optimization",
            n_samples=len(X),
            n_features=X.shape[1],
            class_balance=y.value_counts().to_dict()
        )
        
        # Create study
        study = optuna.create_study(
            direction='maximize',
            sampler=TPESampler(seed=self.seed),
            pruner=self._get_pruner(),
            study_name='xgboost_optimization'
        )
        
        # Create objective
        objective = self._create_objective(X, y, sample_weights)
        
        # Optimize
        study.optimize(
            objective,
            n_trials=self.n_trials,
            show_progress_bar=True,
            n_jobs=1  # Parallel inside XGBoost
        )
        
        # Save best parameters
        self.best_params = study.best_params
        self.best_score = study.best_value
        
        self.log.info(
            "optimization_complete",
            best_score=self.best_score,
            best_params=self.best_params,
            n_trials=len(study.trials),
            n_pruned=len(study.get_trials(states=[optuna.trial.TrialState.PRUNED]))
        )
        
        # Fit final model with best parameters
        self.best_model = self.fit_final_model(X, y, sample_weights)
        
        # MLflow logging
        if self.use_mlflow and MLFLOW_AVAILABLE and mlflow.active_run():
            self._log_to_mlflow(study, X, y)
        
        return study, self.best_model
    
    def fit_final_model(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        sample_weights: Optional[np.ndarray] = None
    ) -> Any:
        """Fit final model with best parameters.
        
        Args:
            X: Features
            y: Labels
            sample_weights: Optional sample weights
            
        Returns:
            Fitted model
        """
        if self.best_params is None:
            raise ValueError("No best parameters found. Run optimize first.")
        
        # Prepare parameters (keep same API as optimization)
        params = self.best_params.copy()
        params.update({
            'tree_method': 'hist',
            'objective': 'binary:logistic',
            'eval_metric': 'aucpr',  # Consistent with optimization
            'random_state': self.seed,
            'verbosity': 0,
            'n_jobs': -1,
            'early_stopping_rounds': 100  # Same early stopping as optimization
        })
        
        # Use consistent scale_pos_weight calculation
        params['scale_pos_weight'] = self._calculate_scale_pos_weight(y)
        
        # Train model with validation split for early stopping
        from sklearn.model_selection import train_test_split
        if len(X) > 100:  # Only split if we have enough data
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=0.2, shuffle=False, stratify=None
            )
            model = xgb.XGBClassifier(**params)
            model.fit(
                X_train, y_train, 
                sample_weight=sample_weights[:len(X_train)] if sample_weights is not None else None,
                eval_set=[(X_val, y_val)],
                verbose=False
            )
        else:
            # Small dataset, train on all data without early stopping
            params_no_early_stop = {k: v for k, v in params.items() if k != 'early_stopping_rounds'}
            model = xgb.XGBClassifier(**params_no_early_stop)
            model.fit(X, y, sample_weight=sample_weights, verbose=False)
        
        # Calibrate (mandatory) - use base model without early stopping for calibration
        base_model_params = {k: v for k, v in params.items() if k != 'early_stopping_rounds'}
        base_model = xgb.XGBClassifier(**base_model_params)
        base_model.fit(X, y, sample_weight=sample_weights, verbose=False)
        
        self.calibrator = CalibratedClassifierCV(
            base_model, method='isotonic', cv=3
        )
        self.calibrator.fit(X, y)
        
        # Calculate thresholds
        y_pred_proba = self.calibrator.predict_proba(X)[:, 1]
        self.threshold_f1 = self._optimize_threshold_f1(y, y_pred_proba)
        
        costs = {'fee_bps': 5, 'slippage_bps': 5}  # Default costs
        self.threshold_ev = self._optimize_threshold_ev(y, y_pred_proba, costs)
        
        self.log.info(
            "final_model_fitted",
            threshold_f1=self.threshold_f1,
            threshold_ev=self.threshold_ev
        )
        
        return model
    
    def train_final_model(self, X: pd.DataFrame, y: pd.Series, sample_weights: Optional[np.ndarray] = None) -> Any:
        """Alias for fit_final_model to match expected interface."""
        return self.fit_final_model(X, y, sample_weights)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict probabilities with calibrated model.
        
        Args:
            X: Features
            
        Returns:
            Calibrated probabilities
        """
        if self.calibrator is None:
            raise ValueError("Model not fitted. Run optimize first.")
        
        return self.calibrator.predict_proba(X)[:, 1]
    
    def predict(self, X: pd.DataFrame, use_ev_threshold: bool = False) -> np.ndarray:
        """Predict classes using optimized threshold.
        
        Args:
            X: Features
            use_ev_threshold: Whether to use EV threshold instead of F1
            
        Returns:
            Predicted classes
        """
        proba = self.predict_proba(X)
        threshold = self.threshold_ev if use_ev_threshold else self.threshold_f1
        return (proba >= threshold).astype(int)
    
    def _log_to_mlflow(self, study: optuna.Study, X: pd.DataFrame, y: pd.Series):
        """Log results to MLflow.
        
        Args:
            study: Optuna study
            X: Features
            y: Labels
        """
        try:
            # Log parameters
            for key, value in self.best_params.items():
                mlflow.log_param(f"xgb_{key}", value)
            
            # Log metrics
            mlflow.log_metric("best_score", self.best_score)
            mlflow.log_metric("n_trials", len(study.trials))
            mlflow.log_metric("threshold_f1", self.threshold_f1)
            mlflow.log_metric("threshold_ev", self.threshold_ev)
            
            # Calculate and log final metrics
            y_pred_proba = self.predict_proba(X)
            y_pred = self.predict(X)
            
            mlflow.log_metric("train_f1", f1_score(y, y_pred))
            mlflow.log_metric("train_brier", brier_score_loss(y, y_pred_proba))
            
            # Log model
            mlflow.xgboost.log_model(self.best_model, "xgboost_model")
            
            self.log.info("mlflow_logging_complete")
            
        except Exception as e:
            self.log.error("mlflow_logging_failed", error=str(e))