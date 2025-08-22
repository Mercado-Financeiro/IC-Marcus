"""
Model Validator for MLOps pipeline.
Ensures models meet quality standards before deployment.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, List, Any, Optional
from sklearn.metrics import (
    f1_score, precision_recall_curve, auc, roc_auc_score,
    brier_score_loss, confusion_matrix, classification_report,
    mean_absolute_error, mean_squared_error
)
from sklearn.calibration import calibration_curve
import warnings
from dataclasses import dataclass
import json
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Container for validation results."""
    passed: bool
    score: float  # 0-100
    checks: Dict[str, bool]
    metrics: Dict[str, float]
    warnings: List[str]
    errors: List[str]
    timestamp: str
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            'passed': self.passed,
            'score': self.score,
            'checks': self.checks,
            'metrics': self.metrics,
            'warnings': self.warnings,
            'errors': self.errors,
            'timestamp': self.timestamp
        }
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)


class ModelValidator:
    """
    Comprehensive model validation for production deployment.
    
    Performs various checks:
    - Metric thresholds
    - Overfitting detection
    - Calibration quality
    - Data leakage detection
    - Prediction distribution
    - Model stability
    """
    
    def __init__(self, strict_mode: bool = False):
        """
        Initialize validator.
        
        Args:
            strict_mode: If True, all checks must pass. If False, warnings allowed.
        """
        self.strict_mode = strict_mode
        
        # Default thresholds (can be customized)
        self.metric_thresholds = {
            'f1_score': {'min': 0.5, 'warning': 0.55},
            'pr_auc': {'min': 0.5, 'warning': 0.55},
            'roc_auc': {'min': 0.55, 'warning': 0.6},
            'brier_score': {'max': 0.3, 'warning': 0.25},
            'calibration_error': {'max': 0.1, 'warning': 0.08}
        }
        
        # Overfitting thresholds
        self.overfit_thresholds = {
            'max_gap': 0.15,  # Max train-val gap
            'warning_gap': 0.10
        }
        
        # Trading metrics thresholds
        self.trading_thresholds = {
            'sharpe_ratio': {'min': 0.5, 'warning': 1.0},
            'max_drawdown': {'max': 0.3, 'warning': 0.2},
            'win_rate': {'min': 0.45, 'warning': 0.5}
        }
    
    def validate_model(
        self,
        model: Any,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        X_test: Optional[pd.DataFrame] = None,
        y_test: Optional[pd.Series] = None,
        model_metadata: Optional[Dict] = None
    ) -> ValidationResult:
        """
        Perform comprehensive model validation.
        
        Args:
            model: Trained model with predict_proba method
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            X_test: Optional test features
            y_test: Optional test labels
            model_metadata: Optional metadata (training history, params, etc.)
            
        Returns:
            ValidationResult object with all checks and metrics
        """
        checks = {}
        metrics = {}
        warnings = []
        errors = []
        
        try:
            # 1. Basic Predictions Check
            logger.info("Validating predictions...")
            pred_check, pred_metrics = self._validate_predictions(
                model, X_val, y_val
            )
            checks.update(pred_check)
            metrics.update(pred_metrics)
            
            # 2. Overfitting Check
            logger.info("Checking for overfitting...")
            overfit_check, overfit_metrics = self._check_overfitting(
                model, X_train, y_train, X_val, y_val
            )
            checks.update(overfit_check)
            metrics.update(overfit_metrics)
            
            # 3. Calibration Check
            logger.info("Checking calibration...")
            calib_check, calib_metrics = self._check_calibration(
                model, X_val, y_val
            )
            checks.update(calib_check)
            metrics.update(calib_metrics)
            
            # 4. Data Leakage Check
            logger.info("Checking for data leakage...")
            leak_check = self._check_data_leakage(
                X_train, X_val, X_test
            )
            checks.update(leak_check)
            
            # 5. Model Stability Check
            logger.info("Checking model stability...")
            stability_check = self._check_model_stability(
                model, X_val
            )
            checks.update(stability_check)
            
            # 6. Prediction Distribution Check
            logger.info("Checking prediction distribution...")
            dist_check = self._check_prediction_distribution(
                model, X_val
            )
            checks.update(dist_check)
            
            # 7. Feature Importance Sanity (if available)
            if hasattr(model, 'feature_importances_'):
                logger.info("Checking feature importances...")
                feat_check = self._check_feature_importance(
                    model, X_train.columns
                )
                checks.update(feat_check)
            
            # Calculate overall score
            score = self._calculate_validation_score(checks, metrics)
            
            # Determine if passed
            passed = self._determine_pass_status(checks, warnings, errors)
            
        except Exception as e:
            logger.error(f"Validation error: {str(e)}")
            errors.append(f"Validation failed: {str(e)}")
            passed = False
            score = 0.0
        
        return ValidationResult(
            passed=passed,
            score=score,
            checks=checks,
            metrics=metrics,
            warnings=warnings,
            errors=errors,
            timestamp=datetime.now().isoformat()
        )
    
    def _validate_predictions(
        self, model: Any, X: pd.DataFrame, y: pd.Series
    ) -> Tuple[Dict[str, bool], Dict[str, float]]:
        """Validate model predictions."""
        checks = {}
        metrics = {}
        
        # Get predictions
        y_pred_proba = model.predict_proba(X)[:, 1]
        y_pred = (y_pred_proba >= 0.5).astype(int)
        
        # Calculate metrics
        metrics['f1_score'] = f1_score(y, y_pred)
        metrics['roc_auc'] = roc_auc_score(y, y_pred_proba)
        
        precision, recall, _ = precision_recall_curve(y, y_pred_proba)
        metrics['pr_auc'] = auc(recall, precision)
        metrics['brier_score'] = brier_score_loss(y, y_pred_proba)
        
        # Check thresholds
        checks['f1_above_threshold'] = metrics['f1_score'] >= self.metric_thresholds['f1_score']['min']
        checks['pr_auc_above_threshold'] = metrics['pr_auc'] >= self.metric_thresholds['pr_auc']['min']
        checks['roc_auc_above_threshold'] = metrics['roc_auc'] >= self.metric_thresholds['roc_auc']['min']
        checks['brier_below_threshold'] = metrics['brier_score'] <= self.metric_thresholds['brier_score']['max']
        
        # Check predictions are valid
        checks['predictions_valid'] = (
            not np.any(np.isnan(y_pred_proba)) and
            not np.any(np.isinf(y_pred_proba)) and
            np.all(y_pred_proba >= 0) and
            np.all(y_pred_proba <= 1)
        )
        
        return checks, metrics
    
    def _check_overfitting(
        self, model: Any,
        X_train: pd.DataFrame, y_train: pd.Series,
        X_val: pd.DataFrame, y_val: pd.Series
    ) -> Tuple[Dict[str, bool], Dict[str, float]]:
        """Check for overfitting."""
        checks = {}
        metrics = {}
        
        # Get train and val predictions
        y_train_pred = model.predict_proba(X_train)[:, 1]
        y_val_pred = model.predict_proba(X_val)[:, 1]
        
        # Calculate scores
        train_f1 = f1_score(y_train, (y_train_pred >= 0.5).astype(int))
        val_f1 = f1_score(y_val, (y_val_pred >= 0.5).astype(int))
        
        gap = train_f1 - val_f1
        
        metrics['train_f1'] = train_f1
        metrics['val_f1'] = val_f1
        metrics['train_val_gap'] = gap
        
        checks['no_severe_overfitting'] = gap <= self.overfit_thresholds['max_gap']
        checks['no_moderate_overfitting'] = gap <= self.overfit_thresholds['warning_gap']
        
        return checks, metrics
    
    def _check_calibration(
        self, model: Any, X: pd.DataFrame, y: pd.Series
    ) -> Tuple[Dict[str, bool], Dict[str, float]]:
        """Check model calibration."""
        checks = {}
        metrics = {}
        
        y_pred_proba = model.predict_proba(X)[:, 1]
        
        # Calculate calibration curve
        fraction_pos, mean_pred = calibration_curve(y, y_pred_proba, n_bins=10)
        
        # Expected Calibration Error (ECE)
        ece = np.mean(np.abs(fraction_pos - mean_pred))
        metrics['calibration_error'] = ece
        
        # Maximum Calibration Error (MCE)
        mce = np.max(np.abs(fraction_pos - mean_pred))
        metrics['max_calibration_error'] = mce
        
        checks['well_calibrated'] = ece <= self.metric_thresholds['calibration_error']['max']
        checks['calibration_acceptable'] = mce <= 0.2
        
        return checks, metrics
    
    def _check_data_leakage(
        self,
        X_train: pd.DataFrame,
        X_val: pd.DataFrame,
        X_test: Optional[pd.DataFrame] = None
    ) -> Dict[str, bool]:
        """Check for potential data leakage."""
        checks = {}
        
        # Check for overlapping indices
        train_val_overlap = len(set(X_train.index) & set(X_val.index))
        checks['no_train_val_overlap'] = train_val_overlap == 0
        
        if X_test is not None:
            train_test_overlap = len(set(X_train.index) & set(X_test.index))
            val_test_overlap = len(set(X_val.index) & set(X_test.index))
            checks['no_train_test_overlap'] = train_test_overlap == 0
            checks['no_val_test_overlap'] = val_test_overlap == 0
        
        # Check temporal ordering if index is datetime
        if isinstance(X_train.index, pd.DatetimeIndex):
            checks['temporal_order_maintained'] = (
                X_train.index.max() < X_val.index.min()
            )
            if X_test is not None and isinstance(X_test.index, pd.DatetimeIndex):
                checks['test_is_future'] = X_val.index.max() < X_test.index.min()
        
        return checks
    
    def _check_model_stability(
        self, model: Any, X: pd.DataFrame
    ) -> Dict[str, bool]:
        """Check model stability with edge cases."""
        checks = {}
        
        try:
            # Test with edge cases
            n_features = X.shape[1]
            edge_cases = {
                'zeros': np.zeros((5, n_features)),
                'ones': np.ones((5, n_features)),
                'large': np.ones((5, n_features)) * 1e6,
                'small': np.ones((5, n_features)) * 1e-6
            }
            
            all_stable = True
            for case_name, X_edge in edge_cases.items():
                X_edge_df = pd.DataFrame(X_edge, columns=X.columns)
                try:
                    preds = model.predict_proba(X_edge_df)[:, 1]
                    if np.any(np.isnan(preds)) or np.any(np.isinf(preds)):
                        all_stable = False
                        break
                except:
                    all_stable = False
                    break
            
            checks['stable_with_edge_cases'] = all_stable
            
        except Exception as e:
            logger.warning(f"Stability check failed: {str(e)}")
            checks['stable_with_edge_cases'] = False
        
        return checks
    
    def _check_prediction_distribution(
        self, model: Any, X: pd.DataFrame
    ) -> Dict[str, bool]:
        """Check prediction distribution is reasonable."""
        checks = {}
        
        y_pred_proba = model.predict_proba(X)[:, 1]
        
        # Check not constant
        unique_preds = np.unique(np.round(y_pred_proba, 4))
        checks['predictions_not_constant'] = len(unique_preds) > 10
        
        # Check reasonable variance
        pred_std = np.std(y_pred_proba)
        checks['predictions_have_variance'] = pred_std > 0.05
        
        # Check not all extreme
        extreme_low = np.sum(y_pred_proba < 0.1) / len(y_pred_proba)
        extreme_high = np.sum(y_pred_proba > 0.9) / len(y_pred_proba)
        checks['predictions_not_extreme'] = (extreme_low + extreme_high) < 0.9
        
        return checks
    
    def _check_feature_importance(
        self, model: Any, feature_names: List[str]
    ) -> Dict[str, bool]:
        """Check feature importance sanity."""
        checks = {}
        
        importances = model.feature_importances_
        
        # Check not all zero
        checks['features_have_importance'] = not np.allclose(importances, 0)
        
        # Check not single feature dominates
        max_importance = np.max(importances)
        checks['no_single_feature_dominance'] = max_importance < 0.9
        
        # Check reasonable number of important features
        important_features = np.sum(importances > 0.01)
        checks['multiple_features_used'] = important_features >= min(5, len(feature_names))
        
        return checks
    
    def _calculate_validation_score(
        self, checks: Dict[str, bool], metrics: Dict[str, float]
    ) -> float:
        """Calculate overall validation score (0-100)."""
        # Weight different components
        weights = {
            'metrics': 0.4,
            'checks': 0.3,
            'overfitting': 0.2,
            'calibration': 0.1
        }
        
        scores = {}
        
        # Metrics score
        metric_scores = []
        if 'f1_score' in metrics:
            metric_scores.append(min(metrics['f1_score'] / 0.7, 1.0))
        if 'pr_auc' in metrics:
            metric_scores.append(min(metrics['pr_auc'] / 0.7, 1.0))
        if 'roc_auc' in metrics:
            metric_scores.append(min(metrics['roc_auc'] / 0.8, 1.0))
        if 'brier_score' in metrics:
            metric_scores.append(max(0, 1 - metrics['brier_score'] / 0.3))
        
        scores['metrics'] = np.mean(metric_scores) if metric_scores else 0.5
        
        # Checks score
        check_results = [v for k, v in checks.items() if not k.startswith('no_')]
        scores['checks'] = np.mean(check_results) if check_results else 0.5
        
        # Overfitting score
        if 'train_val_gap' in metrics:
            scores['overfitting'] = max(0, 1 - metrics['train_val_gap'] / 0.15)
        else:
            scores['overfitting'] = 0.5
        
        # Calibration score
        if 'calibration_error' in metrics:
            scores['calibration'] = max(0, 1 - metrics['calibration_error'] / 0.1)
        else:
            scores['calibration'] = 0.5
        
        # Calculate weighted score
        total_score = sum(scores[k] * weights[k] for k in weights.keys())
        
        return round(total_score * 100, 1)
    
    def _determine_pass_status(
        self, checks: Dict[str, bool], warnings: List[str], errors: List[str]
    ) -> bool:
        """Determine if validation passes."""
        if errors:
            return False
        
        if self.strict_mode:
            return all(checks.values()) and len(warnings) == 0
        else:
            # Must pass critical checks
            critical_checks = [
                'predictions_valid',
                'no_severe_overfitting',
                'predictions_not_constant',
                'no_train_val_overlap'
            ]
            
            for check in critical_checks:
                if check in checks and not checks[check]:
                    return False
            
            return True
    
    def validate_before_production(
        self,
        model: Any,
        test_data: Tuple[pd.DataFrame, pd.Series],
        production_thresholds: Optional[Dict] = None
    ) -> bool:
        """
        Final validation before production deployment.
        
        Args:
            model: Model to deploy
            test_data: (X_test, y_test) tuple
            production_thresholds: Optional custom thresholds
            
        Returns:
            True if model is ready for production
        """
        X_test, y_test = test_data
        
        # Update thresholds if provided
        if production_thresholds:
            self.metric_thresholds.update(production_thresholds)
        
        # Strict validation for production
        self.strict_mode = True
        
        # Create dummy train/val for overfitting check
        # In practice, these would be loaded from MLflow
        result = self.validate_model(
            model=model,
            X_train=X_test[:100],  # Dummy
            y_train=y_test[:100],  # Dummy
            X_val=X_test[100:],
            y_val=y_test[100:]
        )
        
        logger.info(f"Production validation score: {result.score}/100")
        logger.info(f"Passed: {result.passed}")
        
        if not result.passed:
            logger.error(f"Failed checks: {[k for k, v in result.checks.items() if not v]}")
            logger.error(f"Errors: {result.errors}")
            logger.warning(f"Warnings: {result.warnings}")
        
        return result.passed


if __name__ == "__main__":
    # Example usage
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.datasets import make_classification
    
    # Create synthetic data
    X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
    X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(20)])
    y = pd.Series(y)
    
    # Split data
    train_size = 600
    val_size = 200
    
    X_train = X[:train_size]
    y_train = y[:train_size]
    X_val = X[train_size:train_size+val_size]
    y_val = y[train_size:train_size+val_size]
    X_test = X[train_size+val_size:]
    y_test = y[train_size+val_size:]
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Validate
    validator = ModelValidator(strict_mode=False)
    result = validator.validate_model(
        model=model,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test
    )
    
    print(f"Validation passed: {result.passed}")
    print(f"Validation score: {result.score}/100")
    print(f"Checks: {json.dumps(result.checks, indent=2)}")
    print(f"Metrics: {json.dumps(result.metrics, indent=2)}")