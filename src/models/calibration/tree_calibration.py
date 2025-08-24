"""
Tree-specific calibration methods for XGBoost and other tree-based models.

Implements Platt scaling, Isotonic regression, and Beta calibration specifically
optimized for boosted tree predictions.

References:
- "Probabilistic Outputs for Support Vector Machines" (Platt, 1999)
- "Predicting Good Probabilities With Supervised Learning" (Niculescu-Mizil & Caruana, 2005)
- "Beta calibration: a well-founded and easily implemented improvement" (Kull et al., 2017)
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, Tuple, Union
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss
import warnings
from scipy.optimize import minimize_scalar
from scipy.special import expit, logit
import matplotlib.pyplot as plt


class BetaCalibration(BaseEstimator):
    """
    Beta calibration for tree-based models.
    
    Based on "Beta calibration: a well-founded and easily implemented 
    improvement on Platt scaling" (Kull et al., 2017).
    
    Beta calibration fits a Beta distribution to calibrate probabilities,
    which is particularly effective for boosted tree models.
    """
    
    def __init__(self, parameters: str = "abm"):
        """
        Initialize Beta calibration.
        
        Args:
            parameters: Which parameters to fit ("ab", "abm", or "am")
                - "ab": fit shape parameters a and b
                - "abm": fit a, b, and location parameter m
                - "am": fit a and location m (b=1)
        """
        self.parameters = parameters
        self.a_ = None
        self.b_ = None
        self.m_ = None
        self.map_ = None  # MAP estimates for regularization
        
    def fit(self, probabilities: np.ndarray, y_true: np.ndarray) -> 'BetaCalibration':
        """
        Fit Beta calibration parameters.
        
        Args:
            probabilities: Uncalibrated probabilities
            y_true: True binary labels
            
        Returns:
            Self
        """
        # Convert to numpy arrays
        probabilities = np.asarray(probabilities).flatten()
        y_true = np.asarray(y_true).flatten()
        
        # Clip probabilities to avoid numerical issues
        eps = 1e-15
        probabilities = np.clip(probabilities, eps, 1 - eps)
        
        # Initial parameter estimates using method of moments
        p_mean = probabilities.mean()
        p_var = probabilities.var()
        
        if p_var == 0:
            p_var = 1e-6
            
        # Method of moments estimates for Beta parameters
        if p_mean * (1 - p_mean) <= p_var:
            # Degenerate case, fall back to simple estimates
            alpha_init = 1.0
            beta_init = 1.0
        else:
            alpha_init = p_mean * (p_mean * (1 - p_mean) / p_var - 1)
            beta_init = (1 - p_mean) * (p_mean * (1 - p_mean) / p_var - 1)
            
        alpha_init = max(alpha_init, 0.1)
        beta_init = max(beta_init, 0.1)
        
        # Negative log-likelihood for Beta calibration
        def neg_log_likelihood(params):
            if self.parameters == "ab":
                a, b = params
                m = 0.0
            elif self.parameters == "abm":
                a, b, m = params
            else:  # "am"
                a, m = params
                b = 1.0
                
            if a <= 0 or b <= 0:
                return np.inf
                
            # Beta calibration formula
            calibrated_probs = self._beta_calibration(probabilities, a, b, m)
            
            # Clip to avoid log(0)
            calibrated_probs = np.clip(calibrated_probs, eps, 1 - eps)
            
            # Negative log-likelihood
            nll = -np.mean(y_true * np.log(calibrated_probs) + 
                          (1 - y_true) * np.log(1 - calibrated_probs))
            
            return nll
        
        # Optimize parameters
        if self.parameters == "ab":
            from scipy.optimize import minimize
            result = minimize(
                neg_log_likelihood,
                x0=[alpha_init, beta_init],
                bounds=[(0.01, 100), (0.01, 100)],
                method='L-BFGS-B'
            )
            self.a_, self.b_ = result.x
            self.m_ = 0.0
            
        elif self.parameters == "abm":
            from scipy.optimize import minimize
            result = minimize(
                neg_log_likelihood,
                x0=[alpha_init, beta_init, 0.0],
                bounds=[(0.01, 100), (0.01, 100), (-1, 1)],
                method='L-BFGS-B'
            )
            self.a_, self.b_, self.m_ = result.x
            
        else:  # "am"
            from scipy.optimize import minimize
            result = minimize(
                neg_log_likelihood,
                x0=[alpha_init, 0.0],
                bounds=[(0.01, 100), (-1, 1)],
                method='L-BFGS-B'
            )
            self.a_, self.m_ = result.x
            self.b_ = 1.0
        
        return self
    
    def _beta_calibration(self, probabilities: np.ndarray, a: float, b: float, m: float = 0.0) -> np.ndarray:
        """Apply beta calibration transformation."""
        # Clip probabilities
        eps = 1e-15
        probabilities = np.clip(probabilities, eps, 1 - eps)
        
        # Apply location parameter if specified
        if abs(m) > eps:
            probabilities = probabilities * (1 - 2 * abs(m)) + abs(m)
            probabilities = np.clip(probabilities, eps, 1 - eps)
        
        # Beta calibration formula
        calibrated = probabilities ** a / (probabilities ** a + (1 - probabilities) ** b)
        
        return np.clip(calibrated, eps, 1 - eps)
    
    def predict_proba(self, probabilities: np.ndarray) -> np.ndarray:
        """
        Apply beta calibration to probabilities.
        
        Args:
            probabilities: Uncalibrated probabilities
            
        Returns:
            Calibrated probabilities as 2D array for sklearn compatibility
        """
        if self.a_ is None:
            raise ValueError("Beta calibration not fitted yet")
        
        probabilities = np.asarray(probabilities).flatten()
        calibrated = self._beta_calibration(probabilities, self.a_, self.b_, self.m_ or 0.0)
        
        # Return as 2D array for sklearn compatibility
        return np.column_stack([1 - calibrated, calibrated])
    
    def transform(self, probabilities: np.ndarray) -> np.ndarray:
        """Transform probabilities (alias for predict_proba)."""
        return self.predict_proba(probabilities)[:, 1]


class TreeCalibrationSelector:
    """
    Automatic calibration method selector for tree-based models.
    
    Compares Platt scaling, Isotonic regression, and Beta calibration,
    selecting the best method based on Brier score or ECE.
    """
    
    def __init__(
        self,
        methods: list = ['platt', 'isotonic', 'beta'],
        selection_metric: str = 'brier',
        cv_folds: int = 3,
        verbose: bool = False
    ):
        """
        Initialize calibration selector.
        
        Args:
            methods: List of calibration methods to try
            selection_metric: Metric for method selection ('brier' or 'ece')
            cv_folds: Cross-validation folds for method selection
            verbose: Whether to print selection process
        """
        self.methods = methods
        self.selection_metric = selection_metric
        self.cv_folds = cv_folds
        self.verbose = verbose
        self.best_method_ = None
        self.best_calibrator_ = None
        self.method_scores_ = {}
        
    def fit(self, probabilities: np.ndarray, y_true: np.ndarray) -> 'TreeCalibrationSelector':
        """
        Fit and select best calibration method.
        
        Args:
            probabilities: Uncalibrated probabilities from tree model
            y_true: True binary labels
            
        Returns:
            Self
        """
        probabilities = np.asarray(probabilities).flatten()
        y_true = np.asarray(y_true).flatten()
        
        # Evaluate each calibration method
        method_scores = {}
        fitted_calibrators = {}
        
        for method in self.methods:
            try:
                if method == 'platt':
                    # Platt scaling (sigmoid)
                    calibrator = LogisticRegression()
                    calibrator.fit(probabilities.reshape(-1, 1), y_true)
                    cal_probs = calibrator.predict_proba(probabilities.reshape(-1, 1))[:, 1]
                    
                elif method == 'isotonic':
                    # Isotonic regression
                    calibrator = IsotonicRegression(out_of_bounds='clip')
                    calibrator.fit(probabilities, y_true)
                    cal_probs = calibrator.transform(probabilities)
                    
                elif method == 'beta':
                    # Beta calibration
                    calibrator = BetaCalibration(parameters="abm")
                    calibrator.fit(probabilities, y_true)
                    cal_probs = calibrator.transform(probabilities)
                    
                else:
                    continue
                
                # Calculate selection metric
                if self.selection_metric == 'brier':
                    score = brier_score_loss(y_true, cal_probs)
                    # Lower is better for Brier
                    
                elif self.selection_metric == 'ece':
                    # Import ECE calculation
                    from ..metrics.calibration import expected_calibration_error
                    score, _ = expected_calibration_error(y_true, cal_probs)
                    # Lower is better for ECE
                    
                else:
                    # Fallback to log loss
                    cal_probs_clipped = np.clip(cal_probs, 1e-15, 1 - 1e-15)
                    score = log_loss(y_true, cal_probs_clipped)
                
                method_scores[method] = score
                fitted_calibrators[method] = calibrator
                
                if self.verbose:
                    print(f"{method.capitalize()}: {self.selection_metric} = {score:.4f}")
                    
            except Exception as e:
                if self.verbose:
                    print(f"Failed to fit {method}: {e}")
                continue
        
        if not method_scores:
            raise ValueError("No calibration method could be fitted successfully")
        
        # Select best method (lowest score)
        self.best_method_ = min(method_scores.keys(), key=lambda x: method_scores[x])
        self.best_calibrator_ = fitted_calibrators[self.best_method_]
        self.method_scores_ = method_scores
        
        if self.verbose:
            print(f"Selected method: {self.best_method_} ({self.selection_metric} = {method_scores[self.best_method_]:.4f})")
        
        return self
    
    def predict_proba(self, probabilities: np.ndarray) -> np.ndarray:
        """Apply best calibration method."""
        if self.best_calibrator_ is None:
            raise ValueError("Calibrator not fitted yet")
        
        probabilities = np.asarray(probabilities).flatten()
        
        if self.best_method_ == 'platt':
            cal_probs = self.best_calibrator_.predict_proba(probabilities.reshape(-1, 1))
            return cal_probs
        elif self.best_method_ == 'isotonic':
            cal_probs = self.best_calibrator_.transform(probabilities)
            return np.column_stack([1 - cal_probs, cal_probs])
        elif self.best_method_ == 'beta':
            return self.best_calibrator_.predict_proba(probabilities)
        else:
            raise ValueError(f"Unknown method: {self.best_method_}")
    
    def transform(self, probabilities: np.ndarray) -> np.ndarray:
        """Transform probabilities using best method."""
        return self.predict_proba(probabilities)[:, 1]


def compare_tree_calibration_methods(
    probabilities: np.ndarray,
    y_true: np.ndarray,
    methods: list = ['uncalibrated', 'platt', 'isotonic', 'beta'],
    plot: bool = False
) -> Dict[str, Dict]:
    """
    Compare different calibration methods for tree models.
    
    Args:
        probabilities: Uncalibrated probabilities
        y_true: True labels
        methods: Methods to compare
        plot: Whether to plot reliability diagrams
        
    Returns:
        Dictionary with results for each method
    """
    from ..metrics.calibration import (
        expected_calibration_error, comprehensive_calibration_metrics
    )
    
    probabilities = np.asarray(probabilities).flatten()
    y_true = np.asarray(y_true).flatten()
    
    results = {}
    
    for method in methods:
        try:
            if method == 'uncalibrated':
                cal_probs = probabilities
                
            elif method == 'platt':
                calibrator = LogisticRegression()
                calibrator.fit(probabilities.reshape(-1, 1), y_true)
                cal_probs = calibrator.predict_proba(probabilities.reshape(-1, 1))[:, 1]
                
            elif method == 'isotonic':
                calibrator = IsotonicRegression(out_of_bounds='clip')
                calibrator.fit(probabilities, y_true)
                cal_probs = calibrator.transform(probabilities)
                
            elif method == 'beta':
                calibrator = BetaCalibration()
                calibrator.fit(probabilities, y_true)
                cal_probs = calibrator.transform(probabilities)
                
            else:
                continue
            
            # Calculate comprehensive metrics
            metrics = comprehensive_calibration_metrics(y_true, cal_probs)
            
            results[method] = {
                'calibrated_probs': cal_probs,
                'metrics': metrics
            }
            
        except Exception as e:
            print(f"Failed to evaluate {method}: {e}")
            continue
    
    if plot and len(results) > 1:
        plot_calibration_comparison(y_true, results)
    
    return results


def plot_calibration_comparison(
    y_true: np.ndarray,
    results: Dict[str, Dict],
    figsize: Tuple[int, int] = (15, 5)
):
    """Plot reliability diagrams for calibration method comparison."""
    from ..metrics.calibration import plot_reliability_diagram
    
    methods = list(results.keys())
    n_methods = len(methods)
    
    fig, axes = plt.subplots(1, n_methods, figsize=figsize)
    if n_methods == 1:
        axes = [axes]
    
    for i, method in enumerate(methods):
        cal_probs = results[method]['calibrated_probs']
        metrics = results[method]['metrics']
        
        plot_reliability_diagram(
            y_true, cal_probs, 
            title=f'{method.capitalize()}\nECE: {metrics["ece_uniform"]:.3f}',
            ax=axes[i]
        )
    
    plt.tight_layout()
    plt.show()


class XGBoostCalibrator:
    """
    Specialized calibrator for XGBoost models.
    
    Automatically selects the best calibration method and provides
    easy integration with XGBoost pipelines.
    """
    
    def __init__(
        self,
        method: str = 'auto',
        selection_metric: str = 'brier',
        verbose: bool = False
    ):
        """
        Initialize XGBoost calibrator.
        
        Args:
            method: Calibration method ('auto', 'platt', 'isotonic', 'beta')
            selection_metric: Metric for automatic selection
            verbose: Whether to print calibration info
        """
        self.method = method
        self.selection_metric = selection_metric
        self.verbose = verbose
        self.calibrator_ = None
        
    def fit(self, xgb_model, X_val: np.ndarray, y_val: np.ndarray) -> 'XGBoostCalibrator':
        """
        Fit calibrator using XGBoost model predictions.
        
        Args:
            xgb_model: Fitted XGBoost model
            X_val: Validation features
            y_val: Validation labels
            
        Returns:
            Self
        """
        # Get uncalibrated probabilities
        probabilities = xgb_model.predict_proba(X_val)[:, 1]
        
        if self.method == 'auto':
            # Automatic method selection
            self.calibrator_ = TreeCalibrationSelector(
                selection_metric=self.selection_metric,
                verbose=self.verbose
            )
        elif self.method == 'platt':
            self.calibrator_ = LogisticRegression()
        elif self.method == 'isotonic':
            self.calibrator_ = IsotonicRegression(out_of_bounds='clip')
        elif self.method == 'beta':
            self.calibrator_ = BetaCalibration()
        else:
            raise ValueError(f"Unknown calibration method: {self.method}")
        
        # Fit calibrator
        if self.method == 'platt':
            self.calibrator_.fit(probabilities.reshape(-1, 1), y_val)
        else:
            self.calibrator_.fit(probabilities, y_val)
        
        return self
    
    def predict_proba(self, xgb_model, X: np.ndarray) -> np.ndarray:
        """
        Get calibrated probabilities.
        
        Args:
            xgb_model: XGBoost model
            X: Features
            
        Returns:
            Calibrated probabilities
        """
        if self.calibrator_ is None:
            raise ValueError("Calibrator not fitted yet")
        
        # Get uncalibrated probabilities
        probabilities = xgb_model.predict_proba(X)[:, 1]
        
        # Apply calibration
        if self.method == 'platt':
            return self.calibrator_.predict_proba(probabilities.reshape(-1, 1))
        else:
            return self.calibrator_.predict_proba(probabilities)
    
    def get_calibration_info(self) -> Dict:
        """Get information about the fitted calibrator."""
        info = {'method': self.method}
        
        if hasattr(self.calibrator_, 'best_method_'):
            info['selected_method'] = self.calibrator_.best_method_
            info['method_scores'] = self.calibrator_.method_scores_
        
        if hasattr(self.calibrator_, 'a_'):
            info['beta_parameters'] = {
                'a': self.calibrator_.a_,
                'b': self.calibrator_.b_,
                'm': getattr(self.calibrator_, 'm_', None)
            }
        
        return info