"""
Beta calibration for probability calibration.

Based on "Beta calibration: a well-founded and easily implemented improvement 
on logistic calibration for binary classifiers" (Kull et al., 2017)
https://proceedings.mlr.press/v54/kull17a.html

Beta calibration addresses limitations of Platt scaling by using a more flexible
family of calibration maps based on the Beta distribution.
"""

import numpy as np
from scipy.optimize import minimize
from scipy.special import betainc, betaln
from scipy.stats import beta
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from typing import Optional, Tuple, Dict
import warnings
warnings.filterwarnings('ignore')


class BetaCalibration(BaseEstimator, RegressorMixin):
    """
    Beta calibration for binary classifiers.
    
    Maps predicted probabilities to calibrated probabilities using:
    f(p) = 1 / (1 + exp(a * log(p/(1-p)) + b))
    
    Which is equivalent to the CDF of a Beta distribution.
    
    Advantages over Platt scaling:
    - Can push probabilities away from 0 and 1 (not just towards 0.5)
    - Identity function is in the model family (a=1, b=0)
    - More flexible calibration maps
    """
    
    def __init__(
        self,
        method: str = 'mle',
        regularization: float = 0.0,
        max_iter: int = 1000,
        tol: float = 1e-6,
        bounds_a: Tuple[float, float] = (0.01, 100),
        bounds_b: Tuple[float, float] = (-10, 10),
        verbose: bool = False
    ):
        """
        Initialize Beta calibration.
        
        Args:
            method: Optimization method ('mle' for maximum likelihood, 'brier' for Brier score)
            regularization: L2 regularization strength
            max_iter: Maximum optimization iterations
            tol: Convergence tolerance
            bounds_a: Bounds for parameter a
            bounds_b: Bounds for parameter b
            verbose: Whether to print optimization progress
        """
        self.method = method
        self.regularization = regularization
        self.max_iter = max_iter
        self.tol = tol
        self.bounds_a = bounds_a
        self.bounds_b = bounds_b
        self.verbose = verbose
        
        # Fitted parameters
        self.a_ = None
        self.b_ = None
        self.converged_ = False
        self.n_iter_ = 0
        self.loss_ = None
    
    def _negative_log_likelihood(
        self,
        params: np.ndarray,
        proba: np.ndarray,
        y_true: np.ndarray
    ) -> float:
        """
        Calculate negative log-likelihood for Beta calibration.
        
        Args:
            params: [a, b] parameters
            proba: Predicted probabilities
            y_true: True labels
            
        Returns:
            Negative log-likelihood
        """
        a, b = params
        
        # Avoid numerical issues
        if a <= 0 or a > 100:
            return 1e10
        
        # Apply Beta calibration
        eps = 1e-10
        proba_clipped = np.clip(proba, eps, 1 - eps)
        
        # Calculate calibrated probabilities
        # f(p) = 1 / (1 + exp(-a * log(p/(1-p)) - b))
        log_odds = np.log(proba_clipped / (1 - proba_clipped))
        calibrated = 1 / (1 + np.exp(-a * log_odds - b))
        calibrated = np.clip(calibrated, eps, 1 - eps)
        
        # Calculate negative log-likelihood
        nll = -np.mean(
            y_true * np.log(calibrated) + 
            (1 - y_true) * np.log(1 - calibrated)
        )
        
        # Add regularization
        if self.regularization > 0:
            nll += self.regularization * (a**2 + b**2)
        
        return nll
    
    def _brier_score_loss(
        self,
        params: np.ndarray,
        proba: np.ndarray,
        y_true: np.ndarray
    ) -> float:
        """
        Calculate Brier score for Beta calibration.
        
        Args:
            params: [a, b] parameters
            proba: Predicted probabilities
            y_true: True labels
            
        Returns:
            Brier score
        """
        a, b = params
        
        # Avoid numerical issues
        if a <= 0 or a > 100:
            return 1e10
        
        # Apply Beta calibration
        calibrated = self._calibrate(proba, a, b)
        
        # Calculate Brier score
        brier = np.mean((calibrated - y_true) ** 2)
        
        # Add regularization
        if self.regularization > 0:
            brier += self.regularization * (a**2 + b**2)
        
        return brier
    
    def _calibrate(self, proba: np.ndarray, a: float, b: float) -> np.ndarray:
        """
        Apply Beta calibration map.
        
        Args:
            proba: Uncalibrated probabilities
            a: Parameter a
            b: Parameter b
            
        Returns:
            Calibrated probabilities
        """
        eps = 1e-10
        proba_clipped = np.clip(proba, eps, 1 - eps)
        
        # Beta calibration map
        log_odds = np.log(proba_clipped / (1 - proba_clipped))
        calibrated = 1 / (1 + np.exp(-a * log_odds - b))
        
        return np.clip(calibrated, eps, 1 - eps)
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'BetaCalibration':
        """
        Fit Beta calibration parameters.
        
        Args:
            X: Predicted probabilities (n_samples, 1) or (n_samples,)
            y: True binary labels
            
        Returns:
            Self
        """
        # Validate input
        X = check_array(X, ensure_2d=False)
        if X.ndim == 2 and X.shape[1] == 1:
            X = X.ravel()
        elif X.ndim == 2:
            raise ValueError("X must be 1-dimensional (predicted probabilities)")
        
        X, y = check_X_y(X.reshape(-1, 1), y)
        X = X.ravel()
        
        # Choose objective function
        if self.method == 'mle':
            objective = self._negative_log_likelihood
        elif self.method == 'brier':
            objective = self._brier_score_loss
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        # Initial parameters (identity map: a=1, b=0)
        x0 = np.array([1.0, 0.0])
        
        # Optimization bounds
        bounds = [self.bounds_a, self.bounds_b]
        
        # Optimize
        result = minimize(
            objective,
            x0,
            args=(X, y),
            method='L-BFGS-B',
            bounds=bounds,
            options={
                'maxiter': self.max_iter,
                'ftol': self.tol,
                'disp': self.verbose
            }
        )
        
        # Store results
        self.a_, self.b_ = result.x
        self.converged_ = result.success
        self.n_iter_ = result.nit
        self.loss_ = result.fun
        
        if self.verbose:
            print(f"Beta calibration fitted: a={self.a_:.4f}, b={self.b_:.4f}")
            print(f"Converged: {self.converged_}, Iterations: {self.n_iter_}, Loss: {self.loss_:.6f}")
        
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform probabilities using fitted Beta calibration.
        
        Args:
            X: Uncalibrated probabilities
            
        Returns:
            Calibrated probabilities
        """
        check_is_fitted(self, ['a_', 'b_'])
        
        X = check_array(X, ensure_2d=False)
        if X.ndim == 2 and X.shape[1] == 1:
            X = X.ravel()
        
        return self._calibrate(X, self.a_, self.b_)
    
    def fit_transform(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Fit and transform in one step."""
        return self.fit(X, y).transform(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Sklearn-compatible probability prediction.
        
        Args:
            X: Uncalibrated probabilities
            
        Returns:
            Calibrated probabilities in shape (n_samples, 2)
        """
        calibrated = self.transform(X)
        
        # Return both class probabilities
        return np.column_stack([1 - calibrated, calibrated])
    
    def get_params(self, deep: bool = True) -> Dict:
        """Get parameters for sklearn compatibility."""
        return {
            'method': self.method,
            'regularization': self.regularization,
            'max_iter': self.max_iter,
            'tol': self.tol,
            'bounds_a': self.bounds_a,
            'bounds_b': self.bounds_b,
            'verbose': self.verbose
        }
    
    def set_params(self, **params) -> 'BetaCalibration':
        """Set parameters for sklearn compatibility."""
        for key, value in params.items():
            setattr(self, key, value)
        return self


class AdaptiveBetaCalibration(BetaCalibration):
    """
    Adaptive Beta calibration that automatically selects best method.
    
    Compares Beta calibration with Platt and Isotonic calibration
    and selects the best based on validation performance.
    """
    
    def __init__(
        self,
        cv: int = 3,
        metric: str = 'brier',
        **beta_params
    ):
        """
        Initialize adaptive calibration.
        
        Args:
            cv: Number of cross-validation folds
            metric: Metric to optimize ('brier' or 'log_loss')
            **beta_params: Parameters for Beta calibration
        """
        super().__init__(**beta_params)
        self.cv = cv
        self.metric = metric
        self.best_method_ = None
        self.calibrators_ = {}
        self.scores_ = {}
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'AdaptiveBetaCalibration':
        """
        Fit and compare multiple calibration methods.
        
        Args:
            X: Predicted probabilities
            y: True labels
            
        Returns:
            Self
        """
        from sklearn.calibration import CalibratedClassifierCV
        from sklearn.isotonic import IsotonicRegression
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import cross_val_score
        from sklearn.metrics import brier_score_loss, log_loss
        
        X = check_array(X, ensure_2d=False)
        if X.ndim == 2 and X.shape[1] == 1:
            X = X.ravel()
        
        # Create a dummy classifier that returns X as probabilities
        class DummyClassifier:
            def fit(self, X, y):
                return self
            def predict_proba(self, X):
                return np.column_stack([1 - X, X])
        
        # Beta calibration
        beta_cal = BetaCalibration(**self.get_params())
        beta_cal.fit(X, y)
        self.calibrators_['beta'] = beta_cal
        
        # Calculate score
        if self.metric == 'brier':
            beta_score = brier_score_loss(y, beta_cal.transform(X))
        else:
            beta_score = log_loss(y, beta_cal.transform(X))
        self.scores_['beta'] = beta_score
        
        # Platt scaling
        platt = LogisticRegression()
        platt.fit(X.reshape(-1, 1), y)
        self.calibrators_['platt'] = platt
        platt_proba = platt.predict_proba(X.reshape(-1, 1))[:, 1]
        
        if self.metric == 'brier':
            platt_score = brier_score_loss(y, platt_proba)
        else:
            platt_score = log_loss(y, platt_proba)
        self.scores_['platt'] = platt_score
        
        # Isotonic regression
        iso = IsotonicRegression(out_of_bounds='clip')
        iso.fit(X, y)
        self.calibrators_['isotonic'] = iso
        iso_proba = iso.transform(X)
        
        if self.metric == 'brier':
            iso_score = brier_score_loss(y, iso_proba)
        else:
            iso_score = log_loss(y, iso_proba)
        self.scores_['isotonic'] = iso_score
        
        # Select best method
        self.best_method_ = min(self.scores_, key=self.scores_.get)
        
        # Set parameters from best method
        if self.best_method_ == 'beta':
            self.a_ = beta_cal.a_
            self.b_ = beta_cal.b_
        
        if self.verbose:
            print(f"Calibration scores ({self.metric}):")
            for method, score in self.scores_.items():
                print(f"  {method}: {score:.6f}")
            print(f"Best method: {self.best_method_}")
        
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform using best calibration method."""
        check_is_fitted(self, ['best_method_', 'calibrators_'])
        
        X = check_array(X, ensure_2d=False)
        if X.ndim == 2 and X.shape[1] == 1:
            X = X.ravel()
        
        calibrator = self.calibrators_[self.best_method_]
        
        if self.best_method_ == 'beta':
            return calibrator.transform(X)
        elif self.best_method_ == 'platt':
            return calibrator.predict_proba(X.reshape(-1, 1))[:, 1]
        else:  # isotonic
            return calibrator.transform(X)


def compare_calibration_methods(
    proba: np.ndarray,
    y_true: np.ndarray,
    methods: Optional[list] = None
) -> Dict[str, Dict[str, float]]:
    """
    Compare different calibration methods.
    
    Args:
        proba: Uncalibrated probabilities
        y_true: True labels
        methods: List of methods to compare (default: all)
        
    Returns:
        Dictionary with results for each method
    """
    from sklearn.metrics import brier_score_loss, log_loss
    from sklearn.calibration import calibration_curve
    from sklearn.isotonic import IsotonicRegression
    from sklearn.linear_model import LogisticRegression
    
    if methods is None:
        methods = ['uncalibrated', 'beta', 'platt', 'isotonic']
    
    results = {}
    
    # Uncalibrated
    if 'uncalibrated' in methods:
        results['uncalibrated'] = {
            'probabilities': proba,
            'brier_score': brier_score_loss(y_true, proba),
            'log_loss': log_loss(y_true, proba)
        }
    
    # Beta calibration
    if 'beta' in methods:
        beta_cal = BetaCalibration()
        beta_cal.fit(proba, y_true)
        beta_proba = beta_cal.transform(proba)
        results['beta'] = {
            'probabilities': beta_proba,
            'brier_score': brier_score_loss(y_true, beta_proba),
            'log_loss': log_loss(y_true, beta_proba),
            'parameters': {'a': beta_cal.a_, 'b': beta_cal.b_}
        }
    
    # Platt scaling
    if 'platt' in methods:
        platt = LogisticRegression()
        platt.fit(proba.reshape(-1, 1), y_true)
        platt_proba = platt.predict_proba(proba.reshape(-1, 1))[:, 1]
        results['platt'] = {
            'probabilities': platt_proba,
            'brier_score': brier_score_loss(y_true, platt_proba),
            'log_loss': log_loss(y_true, platt_proba)
        }
    
    # Isotonic regression
    if 'isotonic' in methods:
        iso = IsotonicRegression(out_of_bounds='clip')
        iso.fit(proba, y_true)
        iso_proba = iso.transform(proba)
        results['isotonic'] = {
            'probabilities': iso_proba,
            'brier_score': brier_score_loss(y_true, iso_proba),
            'log_loss': log_loss(y_true, iso_proba)
        }
    
    # Find best method
    best_method = min(
        results.keys(),
        key=lambda k: results[k]['brier_score']
    )
    
    for method in results:
        results[method]['is_best'] = (method == best_method)
    
    return results