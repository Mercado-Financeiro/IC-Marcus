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
try:
    import structlog
    log = structlog.get_logger()
except ImportError:
    import logging
    logging.basicConfig(level=logging.INFO)
    log = logging.getLogger(__name__)

import warnings
from pathlib import Path
import sys

# Add parent to path
sys.path.append(str(Path(__file__).parent.parent))

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
        
        # Set deterministic environment
        set_deterministic_environment(seed)
        
        log.info(
            "xgboost_optuna_initialized",
            n_trials=n_trials,
            cv_folds=cv_folds,
            embargo=embargo,
            pruner=pruner_type
        )
    
    def _create_search_space(self, trial: optuna.Trial) -> Dict:
        """Create hyperparameter search space.
        
        Args:
            trial: Optuna trial
            
        Returns:
            Dictionary of hyperparameters
        """
        params = {
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'gamma': trial.suggest_float('gamma', 0, 5),
            'subsample': trial.suggest_float('subsample', 0.5, 0.95),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 0.95),
            'eta': trial.suggest_float('eta', 0.01, 0.3, log=True),
            'lambda': trial.suggest_float('lambda', 0.5, 5),
            'alpha': trial.suggest_float('alpha', 0, 3),
            'max_bin': trial.suggest_categorical('max_bin', [256, 512, 1024]),
            'n_estimators': trial.suggest_int('n_estimators', 100, 2000),
            
            # Fixed parameters
            'tree_method': 'hist',
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'random_state': self.seed + trial.number,  # Different seed per trial for diversity
            'verbosity': 0,
            'n_jobs': -1
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
        # Adjust for class imbalance
        pos_weight = (y == 0).sum() / (y == 1).sum() if (y == 1).sum() > 0 else 1.0
        
        def objective(trial):
            # Get hyperparameters
            params = self._create_search_space(trial)
            params['scale_pos_weight'] = pos_weight
            
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
                
                # Train model with early stopping in params
                params_with_early_stopping = params.copy()
                params_with_early_stopping['early_stopping_rounds'] = 50
                
                model = xgb.XGBClassifier(**params_with_early_stopping)
                
                # Fit model with proper eval_set
                model.fit(
                    X_train, y_train,
                    sample_weight=w_train,
                    eval_set=[(X_val, y_val)],
                    verbose=False
                )
                
                # Skip calibration during optimization (only calibrate final model)
                # This avoids overfitting on validation set
                y_pred_proba = model.predict_proba(X_val)[:, 1]
                
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
                    log.info(
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
        
        if len(np.unique(y)) < 2:
            raise ValueError("Only single class in labels")
        
        if len(X) < self.cv_folds * 2:
            raise ValueError(f"Insufficient data for {self.cv_folds}-fold CV")
        
        log.info(
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
        
        log.info(
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
        
        # Prepare parameters
        params = self.best_params.copy()
        params.update({
            'tree_method': 'hist',
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'random_state': self.seed,  # Fixed seed for final model
            'verbosity': 0,
            'n_jobs': -1
        })
        
        # Adjust for class imbalance
        pos_weight = (y == 0).sum() / (y == 1).sum() if (y == 1).sum() > 0 else 1.0
        params['scale_pos_weight'] = pos_weight
        
        # Train model
        model = xgb.XGBClassifier(**params)
        model.fit(X, y, sample_weight=sample_weights)
        
        # Calibrate (mandatory)
        self.calibrator = CalibratedClassifierCV(
            model, method='isotonic', cv=3
        )
        self.calibrator.fit(X, y)
        
        # Calculate thresholds
        y_pred_proba = self.calibrator.predict_proba(X)[:, 1]
        self.threshold_f1 = self._optimize_threshold_f1(y, y_pred_proba)
        
        costs = {'fee_bps': 5, 'slippage_bps': 5}  # Default costs
        self.threshold_ev = self._optimize_threshold_ev(y, y_pred_proba, costs)
        
        log.info(
            "final_model_fitted",
            threshold_f1=self.threshold_f1,
            threshold_ev=self.threshold_ev
        )
        
        return model
    
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
            
            log.info("mlflow_logging_complete")
            
        except Exception as e:
            log.error("mlflow_logging_failed", error=str(e))