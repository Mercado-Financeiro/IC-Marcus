"""XGBoost with Bayesian optimization using Optuna."""

import gc
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, Any
import xgboost as xgb
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import (
    f1_score, precision_recall_curve, auc, roc_auc_score,
    brier_score_loss, matthews_corrcoef, confusion_matrix
)
# Import our new modules
try:
    from .metrics.quality_gates import QualityGates, calculate_comprehensive_metrics
    from .metrics.pr_auc import calculate_pr_auc_normalized
    from .calibration.beta import BetaCalibration, compare_calibration_methods
    CUSTOM_MODULES = True
except ImportError:
    CUSTOM_MODULES = False
    import sys
    from pathlib import Path
    # Add parent paths for imports
    base_path = Path(__file__).parent
    sys.path.insert(0, str(base_path))
    try:
        from metrics.quality_gates import QualityGates, calculate_comprehensive_metrics
        from metrics.pr_auc import calculate_pr_auc_normalized
        from calibration.beta import BetaCalibration, compare_calibration_methods
        CUSTOM_MODULES = True
    except ImportError:
        CUSTOM_MODULES = False
from scipy.stats.mstats import winsorize
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
    from optuna.integration import XGBoostPruningCallback
    OPTUNA_XGB_INTEGRATION = True
except ImportError:
    OPTUNA_XGB_INTEGRATION = False
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
from utils.memory_utils import memory_cleanup, cleanup_after_trial, safe_delete, log_memory_usage

warnings.filterwarnings('ignore')


class XGBoostOptuna:
    """XGBoost classifier with Optuna optimization and mandatory calibration."""

    def __init__(
        self,
        n_trials: int = 100,
        cv_folds: int = 5,
        embargo: int = 30,  # Increased embargo to avoid temporal leakage
        pruner_type: str = 'hyperband',
        use_mlflow: bool = True,
        seed: int = 42,
        study_name: Optional[str] = None,
        storage: Optional[str] = None,
        reset_study: bool = False,
        selection_method: str = 'none',
        selection_params: Optional[Dict] = None,
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
        # Generate study name with timestamp if not provided
        from datetime import datetime
        self.study_name = study_name or f'xgb_{datetime.now():%Y%m%d_%H%M%S}'
        self.reset_study = reset_study
        # Default persistent storage (Optuna RDB) if not provided
        self.storage = storage or str((Path('artifacts') / 'optuna' / 'xgb_study.db').resolve())
        # Feature selection config (applied per-fold to avoid leakage)
        self.selection_method = selection_method
        self.selection_params = selection_params or {}

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
            pruner=pruner_type,
            storage=self.storage,
            study_name=self.study_name
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

    def _preprocess_features(
        self, 
        X_train: pd.DataFrame, 
        X_val: Optional[pd.DataFrame] = None,
        winsorize_limits: Tuple[float, float] = (0.01, 0.01)
    ) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        """Preprocess features with RobustScaler and winsorization.
        
        Args:
            X_train: Training features
            X_val: Validation features (optional)
            winsorize_limits: Lower and upper percentile limits for winsorization
            
        Returns:
            Preprocessed training and validation features
        """
        with memory_cleanup():
            # Copy to avoid modifying original data
            X_train_proc = X_train.copy()
            
            # Winsorize outliers in training data (1st and 99th percentile by default)
            for col in X_train_proc.columns:
                if X_train_proc[col].dtype in ['float64', 'float32', 'int64', 'int32']:
                    X_train_proc[col] = winsorize(X_train_proc[col], limits=winsorize_limits)
            
            # Fit RobustScaler on winsorized training data
            self.scaler = RobustScaler()
            X_train_scaled = pd.DataFrame(
                self.scaler.fit_transform(X_train_proc),
                index=X_train.index,
                columns=X_train.columns
            )
            
            # Clean up temporary data
            del X_train_proc
            
            # Apply same transformation to validation if provided
            X_val_scaled = None
            if X_val is not None:
                X_val_proc = X_val.copy()
                # Apply same winsorization limits from training
                for col in X_val_proc.columns:
                    if X_val_proc[col].dtype in ['float64', 'float32', 'int64', 'int32']:
                        # Get limits from training data
                        lower = np.percentile(X_train[col], winsorize_limits[0] * 100)
                        upper = np.percentile(X_train[col], (1 - winsorize_limits[1]) * 100)
                        X_val_proc[col] = np.clip(X_val_proc[col], lower, upper)
                
                # Apply scaler fitted on training data
                X_val_scaled = pd.DataFrame(
                    self.scaler.transform(X_val_proc),
                    index=X_val.index,
                    columns=X_val.columns
                )
                
                # Clean up temporary data
                del X_val_proc
                
        return X_train_scaled, X_val_scaled
    
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
            'n_estimators': 500,  # Reduced since no early stopping

            # Fixed parameters
            'tree_method': 'hist',
            'objective': 'binary:logistic',
            'eval_metric': 'aucpr',  # Better for imbalanced data
            'random_state': self.seed,
            'verbosity': 0,  # Reduce noise
            'n_jobs': -1
            # early_stopping_rounds removed - will be passed in fit method
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

    def _optimize_threshold_profit(
        self,
        y: pd.Series,
        y_pred_proba: np.ndarray,
        cost_per_trade: float = 0.002,
        win_return: float = 0.015,
        min_threshold: float = 0.1,
        max_threshold: float = 0.9,
        n_thresholds: int = 100
    ) -> float:
        """
        Encontra o threshold que maximiza o lucro esperado considerando custos de transação.

        Args:
            y: Rótulos verdadeiros (1 ou 0)
            y_pred_proba: Probabilidades preditas pelo modelo
            cost_per_trade: Custo total por trade (fee + slippage) em decimal. Ex: 0.002 = 0.2%
            win_return: Retorno médio esperado de um trade correto (em decimal). Ex: 0.015 = 1.5%
            min_threshold: Threshold mínimo para testar
            max_threshold: Threshold máximo para testar
            n_thresholds: Número de thresholds para testar

        Returns:
            Threshold ótimo que maximiza o lucro esperado
        """
        thresholds = np.linspace(min_threshold, max_threshold, n_thresholds)
        expected_profits = []

        for thresh in thresholds:
            signals = (y_pred_proba >= thresh).astype(int)  # Previsões com o threshold
            n_trades = signals.sum()  # Número total de trades sinalizados

            if n_trades == 0:
                expected_profits.append(0)
                continue

            # Calcula o número de trades corretos (true positives)
            correct_trades = np.sum((signals == 1) & (signals == y))

            # Lucro Bruto: (Trades Corretos * Retorno do Win)
            gross_profit = correct_trades * win_return

            # Custo Total: (Número de Trades * Custo por Trade)
            total_cost = n_trades * cost_per_trade

            # Lucro Líquido
            net_profit = gross_profit - total_cost

            expected_profits.append(net_profit)

        optimal_idx = np.argmax(expected_profits)
        optimal_threshold = thresholds[optimal_idx]
        max_profit = expected_profits[optimal_idx]

        self.log.info(
            "profit_threshold_optimized",
            optimal_threshold=optimal_threshold,
            max_profit=max_profit,
            cost_per_trade=cost_per_trade,
            win_return=win_return,
            n_trades_at_optimal=((y_pred_proba >= optimal_threshold).astype(int)).sum()
        )

        return optimal_threshold

    def _optimize_threshold_ev_with_turnover(
        self, 
        y: pd.Series, 
        y_pred_proba: np.ndarray, 
        costs: Dict,
        turnover_penalty: float = 0.001
    ) -> float:
        """Optimize threshold for expected value with turnover penalty.
        
        Args:
            y: True labels
            y_pred_proba: Predicted probabilities
            costs: Dictionary with fee_bps and slippage_bps
            turnover_penalty: Cost per position flip (default 0.1%)
            
        Returns:
            Optimal threshold maximizing net EV
        """
        thresholds = np.linspace(0.1, 0.9, 100)
        ev_scores = []
        
        fee_bps = costs.get('fee_bps', 5)
        slippage_bps = costs.get('slippage_bps', 5)
        total_cost_bps = fee_bps + slippage_bps
        
        for thresh in thresholds:
            signals = (y_pred_proba >= thresh).astype(int)
            
            # Calculate turnover (position changes)
            if len(signals) > 1:
                turnover = np.sum(np.abs(np.diff(signals))) / len(signals)
            else:
                turnover = 0
                
            # Calculate accuracy metrics
            correct = (signals == y).mean()
            action_rate = signals.mean()
            
            # Skip if no trades
            if action_rate == 0:
                ev_scores.append(-turnover_penalty)  # Only penalty
                continue
                
            # Expected return (assuming 1.5% avg move on correct predictions)
            gross_return = correct * action_rate * 0.015
            
            # Transaction costs
            tx_cost = action_rate * (total_cost_bps / 10000)
            
            # Turnover cost
            turnover_cost = turnover * turnover_penalty
            
            # Net EV
            net_ev = gross_return - tx_cost - turnover_cost
            ev_scores.append(net_ev)
            
        optimal_idx = np.argmax(ev_scores)
        return thresholds[optimal_idx]
    
    def _optimize_threshold_ev(self, y: pd.Series, y_pred_proba: np.ndarray, costs: Dict) -> float:
        """Optimize threshold for expected value (legacy method - kept for compatibility)."""
        # Redirect to new method with turnover
        return self._optimize_threshold_ev_with_turnover(y, y_pred_proba, costs)

    def _create_objective(
        self, X: pd.DataFrame, y: pd.Series, sample_weights: Optional[np.ndarray] = None
    ):
        """Create objective function for Optuna with PR-AUC optimization.

        Args:
            X: Features
            y: Labels
            sample_weights: Optional sample weights

        Returns:
            Objective function optimizing PR-AUC
        """
        # Calculate scale_pos_weight consistently
        pos_weight = self._calculate_scale_pos_weight(y)
        
        # Initialize quality gates if available
        gates = QualityGates() if CUSTOM_MODULES else None

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

                # Optional: per-fold feature selection (leak-safe)
                if self.selection_method and self.selection_method != 'none':
                    try:
                        if self.selection_method == 'select_kbest':
                            from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
                            k = int(self.selection_params.get('k', min(100, X_train.shape[1])))
                            sf = self.selection_params.get('score_func', 'f_classif')
                            scorer = mutual_info_classif if sf == 'mutual_info' else f_classif
                            skb = SelectKBest(score_func=scorer, k=min(k, X_train.shape[1]))
                            X_train = pd.DataFrame(skb.fit_transform(X_train, y_train), index=X_train.index)
                            X_val = pd.DataFrame(skb.transform(X_val), index=X_val.index)
                        elif self.selection_method == 'pca':
                            from sklearn.decomposition import PCA
                            n_components = int(self.selection_params.get('n_components', min(64, X_train.shape[1])))
                            pca = PCA(n_components=n_components, random_state=self.seed)
                            X_train = pd.DataFrame(pca.fit_transform(X_train), index=X_train.index)
                            X_val = pd.DataFrame(pca.transform(X_val), index=X_val.index)
                        if fold_idx == 0:
                            self.log.info("per_fold_feature_selection", method=self.selection_method)
                    except Exception as e:
                        self.log.warning("feature_selection_failed", error=str(e))

                # Sample weights - ensure proper alignment
                if sample_weights is not None:
                    if isinstance(sample_weights, pd.Series):
                        w_train = sample_weights.iloc[train_idx].values
                    else:
                        w_train = sample_weights[train_idx]
                else:
                    w_train = None

                # Setup callbacks for pruning - disabled for now due to version compatibility
                callbacks = []
                # Note: XGBoostPruningCallback requires specific XGBoost version
                # For now, we'll use manual pruning via trial.report() instead
                
                # Create model with early stopping
                params_with_es = params.copy()
                params_with_es['early_stopping_rounds'] = 50
                model = xgb.XGBClassifier(**params_with_es)

                # Fit model with PR-AUC as primary metric
                model.set_params(eval_metric='aucpr')
                model.fit(
                    X_train, y_train,
                    sample_weight=w_train,
                    eval_set=[(X_val, y_val)],
                    verbose=False
                )

                # Get predictions
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

                # Calculate PR-AUC as primary metric
                precision, recall, _ = precision_recall_curve(y_val, y_pred_proba)
                pr_auc = auc(recall, precision)
                
                # Calculate normalized PR-AUC
                prevalence = y_val.mean()
                if CUSTOM_MODULES:
                    pr_auc, pr_auc_norm, baseline = calculate_pr_auc_normalized(y_val, y_pred_proba)
                else:
                    pr_auc_norm = (pr_auc - prevalence) / (1 - prevalence) if prevalence < 1 else 0
                    baseline = prevalence
                
                # Quality gate check if available
                if CUSTOM_MODULES and gates:
                    pr_gate = gates.check_pr_auc_gate(y_val, y_pred_proba, return_details=False)
                    if not pr_gate['passed']:
                        # Penalize trials that don't meet minimum PR-AUC requirement
                        self.log.warning("trial_failed_pr_gate", trial=trial.number, pr_auc=pr_auc, required=baseline*1.2)
                        return -1.0  # Heavy penalty
                
                # Calculate additional metrics for composite score
                brier = brier_score_loss(y_val, y_pred_proba)
                
                # Optimize threshold for F1 (secondary metric)
                threshold = self._optimize_threshold_f1(y_val, y_pred_proba)
                y_pred = (y_pred_proba >= threshold).astype(int)
                f1 = f1_score(y_val, y_pred, zero_division=0)
                mcc = matthews_corrcoef(y_val, y_pred)
                
                # Primary optimization: 80% PR-AUC, 20% calibration (lower Brier is better)
                # This focuses on PR-AUC while ensuring good calibration
                score = 0.8 * pr_auc_norm + 0.2 * (1 - min(brier, 1.0))
                scores.append(score)

                # Log trial metrics for debugging
                if fold_idx == 0:  # Log only first fold to avoid spam
                    self.log.info(
                        f"Trial {trial.number} Fold {fold_idx}: "
                        f"PR-AUC={pr_auc:.4f} (norm={pr_auc_norm:.4f}, baseline={baseline:.3f}), "
                        f"Brier={brier:.4f}, F1={f1:.4f}, MCC={mcc:.4f}, "
                        f"Score={score:.4f}"
                    )

                # Clean up model and data after fold
                safe_delete(model, X_train, X_val, y_train, y_val)
                if w_train is not None:
                    del w_train
                gc.collect()
                
                # Report for pruning
                trial.report(score, fold_idx)

                # Prune if needed
                if trial.should_prune():
                    # Additional cleanup before pruning
                    gc.collect()
                    raise optuna.TrialPruned()

            # Final cleanup after all folds
            gc.collect()
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

        # Sanitize and validate features: keep only numeric columns
        X_numeric = X.select_dtypes(include=[np.number])
        if X_numeric.shape[1] != X.shape[1]:
            try:
                non_numeric = list(set(X.columns) - set(X_numeric.columns))
                self.log.warning("non_numeric_features_dropped", count=len(non_numeric), features=non_numeric[:10])
            except Exception:
                pass

        # Replace inf with NaN, then check
        X_checked = X_numeric.replace([np.inf, -np.inf], np.nan)
        if X_checked.isna().any().any():
            n_nan = int(X_checked.isna().sum().sum())
            raise ValueError(f"Input data contains NaN ({n_nan}) values. Clean data first.")

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
        # Ensure storage directory exists if using SQLite path
        try:
            storage_path = Path(self.storage)
            if storage_path.suffix == '.db':
                storage_path.parent.mkdir(parents=True, exist_ok=True)
                storage_url = f"sqlite:///{storage_path}"
            else:
                storage_url = self.storage
        except Exception:
            storage_url = None

        # Reset study if requested
        if self.reset_study and storage_url:
            try:
                optuna.delete_study(study_name=self.study_name, storage=storage_url)
                self.log.info(f"Deleted existing study: {self.study_name}")
            except Exception:
                pass  # Study might not exist
        
        study = optuna.create_study(
            direction='maximize',
            sampler=TPESampler(seed=self.seed),
            pruner=self._get_pruner(),
            study_name=self.study_name,
            storage=storage_url,
            load_if_exists=not self.reset_study  # Don't load if resetting
        )

        # Create objective
        objective = self._create_objective(X, y, sample_weights)

        # Optimize with enhanced memory cleanup callback
        def callback(study, trial):
            # Force garbage collection after each trial
            gc.collect()
            # Log memory usage every 10 trials
            if trial.number % 10 == 0:
                log_memory_usage(f"Trial {trial.number}")
        
        study.optimize(
            objective,
            n_trials=self.n_trials,
            show_progress_bar=True,
            n_jobs=1,  # Parallel inside XGBoost
            callbacks=[callback]
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

        # Persist trials to CSV for later analysis
        try:
            import pandas as pd
            trials_data = []
            for t in study.trials:
                row = {
                    'number': t.number,
                    'value': t.value,
                    'state': str(t.state),
                }
                row.update({f"param_{k}": v for k, v in t.params.items()})
                trials_data.append(row)
            trials_df = pd.DataFrame(trials_data)
            out_dir = Path('artifacts') / 'optuna'
            out_dir.mkdir(parents=True, exist_ok=True)
            out_csv = out_dir / f"{self.study_name}_trials.csv"
            trials_df.to_csv(out_csv, index=False)

            if self.use_mlflow and MLFLOW_AVAILABLE and mlflow.active_run():
                mlflow.log_artifact(str(out_csv))
        except Exception as e:
            self.log.warning("trials_export_failed", error=str(e))

        # Fit final model with best parameters using temporal validation split
        val_size = max(int(len(X_checked) * 0.2), 1)
        X_train, X_val = X_checked.iloc[:-val_size], X_checked.iloc[-val_size:]
        y_train, y_val = y.iloc[:-val_size], y.iloc[-val_size:]

        sw_train = None
        if sample_weights is not None:
            import pandas as pd
            if isinstance(sample_weights, (pd.Series, pd.DataFrame)):
                sw_train = sample_weights.iloc[:-val_size]
            else:
                # assume numpy array-like
                sw_train = np.asarray(sample_weights)[: len(X_train)]

        self.best_model = self.fit_final_model(X_train, y_train, X_val, y_val, sw_train)

        # MLflow logging
        if self.use_mlflow and MLFLOW_AVAILABLE and mlflow.active_run():
            self._log_to_mlflow(study, X, y)

        return study, self.best_model

    def fit_final_model(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        sample_weights: Optional[np.ndarray] = None
    ) -> Any:
        """Fit final model with best parameters using PROPER validation split.
        
        CRITICAL: Calibration and threshold optimization done on VALIDATION set,
        never on training set to prevent data leakage.

        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features  
            y_val: Validation labels
            sample_weights: Optional sample weights for training

        Returns:
            Fitted model (uncalibrated base model)
        """
        if self.best_params is None:
            raise ValueError("No best parameters found. Run optimize first.")

        # Prepare parameters
        params = self.best_params.copy()
        params.update({
            'tree_method': 'hist',
            'objective': 'binary:logistic',
            'eval_metric': 'aucpr',
            'random_state': self.seed,
            'verbosity': 0,
            'n_jobs': -1
        })

        # Use consistent scale_pos_weight calculation on TRAINING data
        params['scale_pos_weight'] = self._calculate_scale_pos_weight(y_train)

        # Train model on TRAINING data only
        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train, sample_weight=sample_weights, verbose=False)

        # CRITICAL: Compare calibration methods including Beta calibration
        from sklearn.metrics import brier_score_loss
        
        # Get uncalibrated predictions
        y_val_pred_uncal = model.predict_proba(X_val)[:, 1]
        brier_uncal = brier_score_loss(y_val, y_val_pred_uncal)
        
        calibration_results = {}
        
        # Try isotonic calibration
        calibrator_iso = CalibratedClassifierCV(
            model, method='isotonic', cv='prefit'
        )
        calibrator_iso.fit(X_val, y_val)
        y_val_pred_iso = calibrator_iso.predict_proba(X_val)[:, 1]
        brier_iso = brier_score_loss(y_val, y_val_pred_iso)
        calibration_results['isotonic'] = (calibrator_iso, brier_iso)
        
        # Try Platt calibration
        calibrator_platt = CalibratedClassifierCV(
            model, method='sigmoid', cv='prefit'
        )
        calibrator_platt.fit(X_val, y_val)
        y_val_pred_platt = calibrator_platt.predict_proba(X_val)[:, 1]
        brier_platt = brier_score_loss(y_val, y_val_pred_platt)
        calibration_results['platt'] = (calibrator_platt, brier_platt)
        
        # Try Beta calibration if available
        if CUSTOM_MODULES:
            try:
                beta_cal = BetaCalibration(method='brier')
                beta_cal.fit(y_val_pred_uncal, y_val)
                y_val_pred_beta = beta_cal.transform(y_val_pred_uncal)
                brier_beta = brier_score_loss(y_val, y_val_pred_beta)
                
                # Create wrapper for consistency
                class BetaWrapper:
                    def __init__(self, model, beta_cal):
                        self.model = model
                        self.beta_cal = beta_cal
                    def predict_proba(self, X):
                        proba = self.model.predict_proba(X)[:, 1]
                        calibrated = self.beta_cal.transform(proba)
                        return np.column_stack([1 - calibrated, calibrated])
                
                calibrator_beta = BetaWrapper(model, beta_cal)
                calibration_results['beta'] = (calibrator_beta, brier_beta)
                self.log.info(f"Beta calibration: a={beta_cal.a_:.3f}, b={beta_cal.b_:.3f}, Brier={brier_beta:.4f}")
            except Exception as e:
                self.log.warning("beta_calibration_failed", error=str(e))
        
        # Choose best calibration method
        best_method = min(calibration_results.keys(), key=lambda k: calibration_results[k][1])
        self.calibrator, best_brier = calibration_results[best_method]
        self.calibration_method = best_method
        
        self.log.info(
            f"Calibration comparison - Uncalibrated: {brier_uncal:.4f}, "
            f"Isotonic: {brier_iso:.4f}, Platt: {brier_platt:.4f}"
            + (f", Beta: {calibration_results['beta'][1]:.4f}" if 'beta' in calibration_results else "")
            + f" -> Using {best_method} (Brier: {best_brier:.4f})"
        )
        
        # Clean up unused calibrators
        for method, (cal, _) in calibration_results.items():
            if method != best_method:
                del cal
        del y_val_pred_iso, y_val_pred_platt, y_val_pred_uncal
        if 'beta' in calibration_results:
            del y_val_pred_beta
        gc.collect()

        # CRITICAL: Calculate thresholds using VALIDATION data only
        y_val_pred_proba = self.calibrator.predict_proba(X_val)[:, 1]
        
        self.threshold_f1 = self._optimize_threshold_f1(y_val, y_val_pred_proba)

        # Configurações de custos para criptomoedas (realistas)
        cost_per_trade = 0.002  # 0.2% (0.1% fee + 0.1% slippage)
        win_return = 0.015      # 1.5% retorno médio esperado por trade correto

        # Novo threshold baseado em lucro (mais realista)
        self.threshold_profit = self._optimize_threshold_profit(
            y_val, y_val_pred_proba,
            cost_per_trade=cost_per_trade,
            win_return=win_return
        )

        # Threshold EV usando validation
        costs = {'fee_bps': 5, 'slippage_bps': 5}
        self.threshold_ev = self._optimize_threshold_ev(y_val, y_val_pred_proba, costs)

        self.log.info(
            "final_model_fitted_PROPERLY",
            threshold_f1=self.threshold_f1,
            threshold_profit=self.threshold_profit, 
            threshold_ev=self.threshold_ev,
            calibration_on="validation_set",  # CRITICAL LOG
            threshold_on="validation_set",    # CRITICAL LOG
            cost_per_trade=cost_per_trade,
            win_return=win_return
        )

        return model

    def train_final_model(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series, 
        X_val: pd.DataFrame,
        y_val: pd.Series,
        sample_weights: Optional[np.ndarray] = None
    ) -> Any:
        """Alias for fit_final_model to match expected interface.
        
        DEPRECATED: This method signature is being updated to enforce proper
        train/validation split for calibration and threshold optimization.
        """
        return self.fit_final_model(X_train, y_train, X_val, y_val, sample_weights)

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

    def predict(self, X: pd.DataFrame, threshold_type: str = 'f1') -> np.ndarray:
        """Predict classes using optimized threshold.

        Args:
            X: Features
            threshold_type: Which threshold to use ('f1', 'profit', 'ev')

        Returns:
            Predicted classes
        """
        proba = self.predict_proba(X)

        if threshold_type == 'profit':
            threshold = self.threshold_profit
        elif threshold_type == 'ev':
            threshold = self.threshold_ev
        else:  # default to f1
            threshold = self.threshold_f1

        return (proba >= threshold).astype(int)

    def calculate_trading_metrics(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        cost_per_trade: float = 0.002,
        win_return: float = 0.015
    ) -> Dict[str, float]:
        """
        Calcula métricas de trading usando o threshold baseado em lucro.

        Args:
            X: Features
            y: True labels
            cost_per_trade: Custo por trade (fee + slippage)
            win_return: Retorno esperado por trade correto

        Returns:
            Dictionary com métricas de trading
        """
        y_pred_proba = self.predict_proba(X)
        y_pred_profit = self.predict(X, threshold_type='profit')

        # Confusion matrix
        tn, fp, fn, tp = confusion_matrix(y, y_pred_profit).ravel()

        # Métricas básicas
        total_trades = tp + fp
        correct_trades = tp
        incorrect_trades = fp

        # Taxas
        win_rate = tp / total_trades if total_trades > 0 else 0
        accuracy = (tp + tn) / len(y) if len(y) > 0 else 0

        # Lucro esperado
        gross_profit = correct_trades * win_return
        total_cost = total_trades * cost_per_trade
        net_profit = gross_profit - total_cost

        # ROI
        roi = net_profit / total_cost if total_cost > 0 else 0

        # Profit factor
        profit_factor = gross_profit / total_cost if total_cost > 0 else 0

        # Expected value per trade
        ev_per_trade = net_profit / total_trades if total_trades > 0 else 0

        metrics = {
            'total_trades': total_trades,
            'correct_trades': correct_trades,
            'incorrect_trades': incorrect_trades,
            'win_rate': win_rate,
            'accuracy': accuracy,
            'gross_profit': gross_profit,
            'total_cost': total_cost,
            'net_profit': net_profit,
            'roi': roi,
            'profit_factor': profit_factor,
            'ev_per_trade': ev_per_trade,
            'threshold_profit': self.threshold_profit,
            'cost_per_trade': cost_per_trade,
            'win_return': win_return
        }

        return metrics

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
