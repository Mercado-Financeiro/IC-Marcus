"""
Data Drift Detection for ML Pipeline.
Detects distribution changes in features and target variables.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from scipy import stats
from scipy.stats import ks_2samp, chi2_contingency, wasserstein_distance
import logging
from datetime import datetime
from dataclasses import dataclass
from enum import Enum
import json
from pathlib import Path

logger = logging.getLogger(__name__)


class DriftType(Enum):
    """Types of drift detection."""
    NO_DRIFT = "no_drift"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class DriftResult:
    """Result of drift detection for a feature."""
    feature_name: str
    drift_type: DriftType
    drift_score: float
    p_value: Optional[float]
    test_used: str
    baseline_stats: Dict[str, float]
    current_stats: Dict[str, float]
    timestamp: datetime
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'feature_name': self.feature_name,
            'drift_type': self.drift_type.value,
            'drift_score': self.drift_score,
            'p_value': self.p_value,
            'test_used': self.test_used,
            'baseline_stats': self.baseline_stats,
            'current_stats': self.current_stats,
            'timestamp': self.timestamp.isoformat()
        }


class DriftDetector:
    """
    Comprehensive drift detection for features and targets.
    
    Supports multiple statistical tests:
    - Kolmogorov-Smirnov test for numerical features
    - Chi-square test for categorical features
    - Population Stability Index (PSI)
    - Wasserstein distance
    - Jensen-Shannon divergence
    """
    
    def __init__(self,
                 warning_threshold: float = 0.1,
                 critical_threshold: float = 0.3,
                 psi_threshold: float = 0.2,
                 p_value_threshold: float = 0.05,
                 min_samples: int = 100):
        """
        Initialize drift detector.
        
        Args:
            warning_threshold: Threshold for warning level drift
            critical_threshold: Threshold for critical drift
            psi_threshold: PSI threshold for drift detection
            p_value_threshold: P-value threshold for statistical tests
            min_samples: Minimum samples required for detection
        """
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold
        self.psi_threshold = psi_threshold
        self.p_value_threshold = p_value_threshold
        self.min_samples = min_samples
        
        # Store baseline distributions
        self.baseline_data = None
        self.baseline_stats = {}
        self.feature_types = {}
        
        # Drift history
        self.drift_history = []
        
    def fit_baseline(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """
        Fit baseline distributions for drift detection.
        
        Args:
            X: Baseline feature DataFrame
            y: Baseline target Series
        """
        logger.info(f"Fitting baseline with {len(X)} samples and {len(X.columns)} features")
        
        self.baseline_data = X.copy()
        
        # Determine feature types
        for col in X.columns:
            if X[col].dtype in ['int64', 'float64']:
                self.feature_types[col] = 'numerical'
            else:
                self.feature_types[col] = 'categorical'
        
        # Calculate baseline statistics
        self.baseline_stats = self._calculate_statistics(X)
        
        # Store target baseline if provided
        if y is not None:
            self.baseline_target = y.copy()
            self.target_stats = self._calculate_target_statistics(y)
        
        logger.info(f"Baseline fitted: {len(self.feature_types)} features analyzed")
    
    def detect_drift(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> Dict[str, Any]:
        """
        Detect drift in new data compared to baseline.
        
        Args:
            X: Current feature DataFrame
            y: Current target Series
            
        Returns:
            Dictionary with drift detection results
        """
        if self.baseline_data is None:
            raise ValueError("Baseline not fitted. Call fit_baseline first.")
        
        if len(X) < self.min_samples:
            logger.warning(f"Insufficient samples for drift detection: {len(X)} < {self.min_samples}")
            return {'status': 'insufficient_data', 'samples': len(X)}
        
        results = {
            'timestamp': datetime.now(),
            'feature_drift': {},
            'target_drift': None,
            'summary': {
                'total_features': len(X.columns),
                'drifted_features': 0,
                'warning_features': 0,
                'critical_features': 0
            }
        }
        
        # Detect feature drift
        for col in X.columns:
            if col not in self.baseline_data.columns:
                logger.warning(f"Feature {col} not in baseline, skipping")
                continue
            
            drift_result = self._detect_feature_drift(col, X[col])
            results['feature_drift'][col] = drift_result
            
            # Update summary
            if drift_result.drift_type == DriftType.WARNING:
                results['summary']['warning_features'] += 1
                results['summary']['drifted_features'] += 1
            elif drift_result.drift_type == DriftType.CRITICAL:
                results['summary']['critical_features'] += 1
                results['summary']['drifted_features'] += 1
        
        # Detect target drift
        if y is not None and hasattr(self, 'baseline_target'):
            results['target_drift'] = self._detect_target_drift(y)
        
        # Calculate overall drift score
        results['overall_drift_score'] = self._calculate_overall_drift_score(results)
        
        # Store in history
        self.drift_history.append(results)
        
        return results
    
    def _detect_feature_drift(self, feature_name: str, current_data: pd.Series) -> DriftResult:
        """Detect drift for a single feature."""
        baseline_data = self.baseline_data[feature_name]
        
        if self.feature_types[feature_name] == 'numerical':
            return self._detect_numerical_drift(feature_name, baseline_data, current_data)
        else:
            return self._detect_categorical_drift(feature_name, baseline_data, current_data)
    
    def _detect_numerical_drift(self, 
                                feature_name: str,
                                baseline: pd.Series,
                                current: pd.Series) -> DriftResult:
        """Detect drift in numerical features using multiple tests."""
        # Remove NaN values
        baseline_clean = baseline.dropna()
        current_clean = current.dropna()
        
        if len(baseline_clean) == 0 or len(current_clean) == 0:
            return DriftResult(
                feature_name=feature_name,
                drift_type=DriftType.NO_DRIFT,
                drift_score=0,
                p_value=1.0,
                test_used='insufficient_data',
                baseline_stats={},
                current_stats={},
                timestamp=datetime.now()
            )
        
        # Kolmogorov-Smirnov test
        ks_stat, ks_pvalue = ks_2samp(baseline_clean, current_clean)
        
        # Wasserstein distance (Earth Mover's Distance)
        wasserstein = wasserstein_distance(baseline_clean, current_clean)
        
        # Population Stability Index (PSI)
        psi = self._calculate_psi(baseline_clean, current_clean)
        
        # Jensen-Shannon divergence
        js_divergence = self._calculate_js_divergence(baseline_clean, current_clean)
        
        # Combine scores (weighted average)
        drift_score = (
            ks_stat * 0.3 +
            min(wasserstein / (baseline_clean.std() + 1e-7), 1.0) * 0.3 +
            psi * 0.2 +
            js_divergence * 0.2
        )
        
        # Determine drift type
        if drift_score >= self.critical_threshold or ks_pvalue < self.p_value_threshold / 10:
            drift_type = DriftType.CRITICAL
        elif drift_score >= self.warning_threshold or ks_pvalue < self.p_value_threshold:
            drift_type = DriftType.WARNING
        else:
            drift_type = DriftType.NO_DRIFT
        
        # Calculate statistics
        baseline_stats = {
            'mean': float(baseline_clean.mean()),
            'std': float(baseline_clean.std()),
            'median': float(baseline_clean.median()),
            'q25': float(baseline_clean.quantile(0.25)),
            'q75': float(baseline_clean.quantile(0.75))
        }
        
        current_stats = {
            'mean': float(current_clean.mean()),
            'std': float(current_clean.std()),
            'median': float(current_clean.median()),
            'q25': float(current_clean.quantile(0.25)),
            'q75': float(current_clean.quantile(0.75))
        }
        
        return DriftResult(
            feature_name=feature_name,
            drift_type=drift_type,
            drift_score=float(drift_score),
            p_value=float(ks_pvalue),
            test_used='combined_numerical',
            baseline_stats=baseline_stats,
            current_stats=current_stats,
            timestamp=datetime.now()
        )
    
    def _detect_categorical_drift(self,
                                 feature_name: str,
                                 baseline: pd.Series,
                                 current: pd.Series) -> DriftResult:
        """Detect drift in categorical features."""
        # Get value counts
        baseline_counts = baseline.value_counts()
        current_counts = current.value_counts()
        
        # Align categories
        all_categories = set(baseline_counts.index) | set(current_counts.index)
        baseline_aligned = pd.Series(
            [baseline_counts.get(cat, 0) for cat in all_categories],
            index=list(all_categories)
        )
        current_aligned = pd.Series(
            [current_counts.get(cat, 0) for cat in all_categories],
            index=list(all_categories)
        )
        
        # Chi-square test
        chi2, p_value = self._chi_square_test(baseline_aligned, current_aligned)
        
        # Population Stability Index
        psi = self._calculate_psi_categorical(baseline_aligned, current_aligned)
        
        # Total variation distance
        tvd = self._calculate_tvd(baseline_aligned, current_aligned)
        
        # Combine scores
        drift_score = (chi2 / (len(all_categories) + 1e-7)) * 0.4 + psi * 0.3 + tvd * 0.3
        
        # Determine drift type
        if drift_score >= self.critical_threshold or p_value < self.p_value_threshold / 10:
            drift_type = DriftType.CRITICAL
        elif drift_score >= self.warning_threshold or p_value < self.p_value_threshold:
            drift_type = DriftType.WARNING
        else:
            drift_type = DriftType.NO_DRIFT
        
        # Calculate statistics
        baseline_stats = {
            'n_categories': len(baseline_counts),
            'mode': str(baseline.mode()[0]) if len(baseline.mode()) > 0 else 'N/A',
            'entropy': float(stats.entropy(baseline_aligned / baseline_aligned.sum()))
        }
        
        current_stats = {
            'n_categories': len(current_counts),
            'mode': str(current.mode()[0]) if len(current.mode()) > 0 else 'N/A',
            'entropy': float(stats.entropy(current_aligned / current_aligned.sum()))
        }
        
        return DriftResult(
            feature_name=feature_name,
            drift_type=drift_type,
            drift_score=float(drift_score),
            p_value=float(p_value),
            test_used='combined_categorical',
            baseline_stats=baseline_stats,
            current_stats=current_stats,
            timestamp=datetime.now()
        )
    
    def _detect_target_drift(self, current_target: pd.Series) -> DriftResult:
        """Detect drift in target variable."""
        if current_target.dtype in ['int64', 'float64']:
            if len(current_target.unique()) <= 10:
                # Treat as categorical if few unique values
                return self._detect_categorical_drift('target', self.baseline_target, current_target)
            else:
                return self._detect_numerical_drift('target', self.baseline_target, current_target)
        else:
            return self._detect_categorical_drift('target', self.baseline_target, current_target)
    
    def _calculate_psi(self, baseline: pd.Series, current: pd.Series, n_bins: int = 10) -> float:
        """Calculate Population Stability Index for numerical data."""
        # Create bins based on baseline
        _, bin_edges = pd.qcut(baseline, q=n_bins, retbins=True, duplicates='drop')
        
        # Calculate frequencies
        baseline_freq = pd.cut(baseline, bins=bin_edges, include_lowest=True).value_counts(normalize=True).sort_index()
        current_freq = pd.cut(current, bins=bin_edges, include_lowest=True).value_counts(normalize=True).sort_index()
        
        # Add small epsilon to avoid log(0)
        epsilon = 1e-10
        baseline_freq = baseline_freq + epsilon
        current_freq = current_freq + epsilon
        
        # Calculate PSI
        psi = np.sum((current_freq - baseline_freq) * np.log(current_freq / baseline_freq))
        
        return float(psi)
    
    def _calculate_psi_categorical(self, baseline: pd.Series, current: pd.Series) -> float:
        """Calculate PSI for categorical data."""
        # Normalize to probabilities
        baseline_prob = baseline / baseline.sum()
        current_prob = current / current.sum()
        
        # Add small epsilon
        epsilon = 1e-10
        baseline_prob = baseline_prob + epsilon
        current_prob = current_prob + epsilon
        
        # Calculate PSI
        psi = np.sum((current_prob - baseline_prob) * np.log(current_prob / baseline_prob))
        
        return float(psi)
    
    def _calculate_js_divergence(self, baseline: pd.Series, current: pd.Series, n_bins: int = 50) -> float:
        """Calculate Jensen-Shannon divergence."""
        # Create histogram bins
        min_val = min(baseline.min(), current.min())
        max_val = max(baseline.max(), current.max())
        bins = np.linspace(min_val, max_val, n_bins)
        
        # Calculate histograms
        baseline_hist, _ = np.histogram(baseline, bins=bins)
        current_hist, _ = np.histogram(current, bins=bins)
        
        # Normalize
        baseline_prob = baseline_hist / baseline_hist.sum()
        current_prob = current_hist / current_hist.sum()
        
        # Calculate average distribution
        avg_prob = (baseline_prob + current_prob) / 2
        
        # Add epsilon
        epsilon = 1e-10
        baseline_prob = baseline_prob + epsilon
        current_prob = current_prob + epsilon
        avg_prob = avg_prob + epsilon
        
        # Calculate JS divergence
        js_div = 0.5 * stats.entropy(baseline_prob, avg_prob) + 0.5 * stats.entropy(current_prob, avg_prob)
        
        return float(js_div)
    
    def _chi_square_test(self, baseline: pd.Series, current: pd.Series) -> Tuple[float, float]:
        """Perform chi-square test for categorical data."""
        # Create contingency table
        contingency = pd.DataFrame({
            'baseline': baseline,
            'current': current
        })
        
        # Perform chi-square test
        chi2, p_value, _, _ = chi2_contingency(contingency.T)
        
        return float(chi2), float(p_value)
    
    def _calculate_tvd(self, baseline: pd.Series, current: pd.Series) -> float:
        """Calculate Total Variation Distance."""
        # Normalize to probabilities
        baseline_prob = baseline / baseline.sum()
        current_prob = current / current.sum()
        
        # Calculate TVD
        tvd = 0.5 * np.sum(np.abs(baseline_prob - current_prob))
        
        return float(tvd)
    
    def _calculate_statistics(self, X: pd.DataFrame) -> Dict[str, Dict]:
        """Calculate baseline statistics for all features."""
        stats = {}
        
        for col in X.columns:
            if self.feature_types[col] == 'numerical':
                stats[col] = {
                    'mean': float(X[col].mean()),
                    'std': float(X[col].std()),
                    'median': float(X[col].median()),
                    'min': float(X[col].min()),
                    'max': float(X[col].max())
                }
            else:
                value_counts = X[col].value_counts()
                stats[col] = {
                    'n_unique': len(value_counts),
                    'mode': str(X[col].mode()[0]) if len(X[col].mode()) > 0 else 'N/A',
                    'distribution': value_counts.to_dict()
                }
        
        return stats
    
    def _calculate_target_statistics(self, y: pd.Series) -> Dict:
        """Calculate baseline statistics for target."""
        if y.dtype in ['int64', 'float64'] and len(y.unique()) > 10:
            return {
                'mean': float(y.mean()),
                'std': float(y.std()),
                'median': float(y.median())
            }
        else:
            value_counts = y.value_counts()
            return {
                'n_classes': len(value_counts),
                'distribution': value_counts.to_dict(),
                'balance': float(value_counts.min() / value_counts.max())
            }
    
    def _calculate_overall_drift_score(self, results: Dict) -> float:
        """Calculate overall drift score from individual results."""
        if not results['feature_drift']:
            return 0.0
        
        scores = []
        for drift_result in results['feature_drift'].values():
            scores.append(drift_result.drift_score)
        
        # Weighted average with higher weight for critical drifts
        weighted_score = np.mean(scores)
        
        # Adjust based on number of drifted features
        drift_ratio = results['summary']['drifted_features'] / results['summary']['total_features']
        
        return float(weighted_score * (1 + drift_ratio))
    
    def get_drift_report(self) -> Dict[str, Any]:
        """Generate comprehensive drift report."""
        if not self.drift_history:
            return {'status': 'no_history'}
        
        latest = self.drift_history[-1]
        
        report = {
            'timestamp': latest['timestamp'].isoformat(),
            'overall_drift_score': latest['overall_drift_score'],
            'summary': latest['summary'],
            'critical_features': [],
            'warning_features': [],
            'stable_features': []
        }
        
        # Categorize features
        for feature, result in latest['feature_drift'].items():
            feature_info = {
                'name': feature,
                'drift_score': result.drift_score,
                'p_value': result.p_value
            }
            
            if result.drift_type == DriftType.CRITICAL:
                report['critical_features'].append(feature_info)
            elif result.drift_type == DriftType.WARNING:
                report['warning_features'].append(feature_info)
            else:
                report['stable_features'].append(feature_info)
        
        # Sort by drift score
        report['critical_features'].sort(key=lambda x: x['drift_score'], reverse=True)
        report['warning_features'].sort(key=lambda x: x['drift_score'], reverse=True)
        
        return report
    
    def save_drift_history(self, filepath: Path):
        """Save drift history to file."""
        history_data = []
        
        for entry in self.drift_history:
            serialized_entry = {
                'timestamp': entry['timestamp'].isoformat(),
                'summary': entry['summary'],
                'overall_drift_score': entry['overall_drift_score']
            }
            
            # Serialize feature drift results
            serialized_entry['feature_drift'] = {
                feature: result.to_dict()
                for feature, result in entry['feature_drift'].items()
            }
            
            history_data.append(serialized_entry)
        
        with open(filepath, 'w') as f:
            json.dump(history_data, f, indent=2)
        
        logger.info(f"Drift history saved to {filepath}")


class AutomatedDriftMonitor:
    """Automated monitoring with alerts and actions."""
    
    def __init__(self,
                 detector: DriftDetector,
                 alert_callback: Optional[callable] = None,
                 action_callback: Optional[callable] = None):
        """
        Initialize automated monitor.
        
        Args:
            detector: DriftDetector instance
            alert_callback: Function to call for alerts
            action_callback: Function to call for remediation
        """
        self.detector = detector
        self.alert_callback = alert_callback or self._default_alert
        self.action_callback = action_callback or self._default_action
        
    def monitor(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> Dict:
        """Monitor data and trigger alerts/actions."""
        # Detect drift
        results = self.detector.detect_drift(X, y)
        
        # Check if action needed
        if results['summary']['critical_features'] > 0:
            self._handle_critical_drift(results)
        elif results['summary']['warning_features'] > 0:
            self._handle_warning_drift(results)
        
        return results
    
    def _handle_critical_drift(self, results: Dict):
        """Handle critical drift detection."""
        message = f"CRITICAL DRIFT: {results['summary']['critical_features']} features with critical drift"
        
        # Send alert
        self.alert_callback('critical', message, results)
        
        # Trigger action
        self.action_callback('critical', results)
    
    def _handle_warning_drift(self, results: Dict):
        """Handle warning level drift."""
        message = f"WARNING DRIFT: {results['summary']['warning_features']} features with warning drift"
        
        # Send alert
        self.alert_callback('warning', message, results)
    
    def _default_alert(self, severity: str, message: str, results: Dict):
        """Default alert handler."""
        logger.warning(f"[{severity.upper()}] {message}")
        logger.info(f"Drift summary: {results['summary']}")
    
    def _default_action(self, severity: str, results: Dict):
        """Default action handler."""
        logger.info(f"Action triggered for {severity} drift")
        # In production, this could trigger model retraining,
        # feature recalculation, or other remediation