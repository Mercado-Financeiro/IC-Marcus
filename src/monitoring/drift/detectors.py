"""Drift detection algorithms for different types of drift."""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from scipy.stats import ks_2samp
import structlog

from src.monitoring.drift.config import DriftConfig
from src.monitoring.drift.metrics import DriftMetrics

log = structlog.get_logger()


class FeatureDriftDetector:
    """Detect drift in individual features."""
    
    def __init__(self, config: DriftConfig, metrics: DriftMetrics):
        """
        Initialize feature drift detector.
        
        Args:
            config: Drift configuration
            metrics: Drift metrics calculator
        """
        self.config = config
        self.metrics = metrics
    
    def detect(
        self,
        reference_data: pd.DataFrame,
        current_data: pd.DataFrame,
        features: Optional[List[str]] = None
    ) -> Dict[str, Dict]:
        """
        Detect drift in specific features.
        
        Args:
            reference_data: Reference DataFrame
            current_data: Current DataFrame
            features: List of features to monitor (None = all)
            
        Returns:
            Dictionary with drift results per feature
        """
        if features is None:
            features = [col for col in current_data.columns 
                       if col in reference_data.columns]
        
        drift_results = {}
        
        for feature in features:
            if feature not in reference_data.columns:
                log.warning(f"Feature {feature} not in reference data")
                continue
            
            ref_data = reference_data[feature].dropna()
            curr_data = current_data[feature].dropna()
            
            if len(curr_data) < self.config.min_samples_for_detection:
                log.warning(f"Insufficient samples for {feature}")
                continue
            
            # Calculate drift metrics
            psi = self.metrics.calculate_psi(ref_data, curr_data, self.config.n_bins_psi)
            kl = self.metrics.calculate_kl_divergence(ref_data, curr_data, self.config.n_bins_psi)
            js = self.metrics.calculate_js_divergence(ref_data, curr_data, self.config.n_bins_psi)
            wasserstein = self.metrics.calculate_wasserstein(ref_data, curr_data)
            ks_stat, ks_pvalue = self.metrics.calculate_ks_statistic(ref_data, curr_data)
            
            # Determine severity
            severity = self._determine_severity(psi, kl, wasserstein)
            
            drift_results[feature] = {
                'psi': psi,
                'kl_divergence': kl,
                'js_divergence': js,
                'wasserstein_distance': wasserstein,
                'ks_statistic': ks_stat,
                'ks_pvalue': ks_pvalue,
                'severity': severity,
                'has_drift': severity in ['warning', 'critical'],
                'timestamp': datetime.now().isoformat()
            }
            
            # Add comparative statistics
            ref_stats = self.metrics.calculate_statistics(pd.DataFrame({feature: ref_data}))
            curr_stats = self.metrics.calculate_statistics(pd.DataFrame({feature: curr_data}))
            
            if feature in ref_stats and feature in curr_stats:
                ref_stat = ref_stats[feature]
                curr_stat = curr_stats[feature]
                
                drift_results[feature]['stats_comparison'] = {
                    'mean_shift': abs(curr_stat['mean'] - ref_stat['mean']) / (ref_stat['std'] + 1e-10),
                    'std_ratio': curr_stat['std'] / (ref_stat['std'] + 1e-10),
                    'range_shift': abs((curr_stat['max'] - curr_stat['min']) - 
                                     (ref_stat['max'] - ref_stat['min']))
                }
        
        return drift_results
    
    def _determine_severity(self, psi: float, kl: float, wasserstein: float) -> str:
        """
        Determine drift severity.
        
        Args:
            psi: PSI value
            kl: KL divergence
            wasserstein: Wasserstein distance
            
        Returns:
            Severity level: 'normal', 'warning', 'critical'
        """
        if (psi >= self.config.psi_threshold_critical or
            kl >= self.config.kl_threshold_critical or
            wasserstein >= self.config.wasserstein_threshold_critical):
            return 'critical'
        elif (psi >= self.config.psi_threshold_warning or
              kl >= self.config.kl_threshold_warning or
              wasserstein >= self.config.wasserstein_threshold_warning):
            return 'warning'
        else:
            return 'normal'


class PredictionDriftDetector:
    """Detect drift in model predictions."""
    
    def __init__(self, config: DriftConfig, metrics: DriftMetrics):
        """
        Initialize prediction drift detector.
        
        Args:
            config: Drift configuration
            metrics: Drift metrics calculator
        """
        self.config = config
        self.metrics = metrics
    
    def detect(
        self,
        reference_predictions: pd.Series,
        current_predictions: pd.Series
    ) -> Dict:
        """
        Detect drift in model predictions.
        
        Args:
            reference_predictions: Reference predictions
            current_predictions: Current predictions
            
        Returns:
            Dictionary with prediction drift analysis
        """
        # Calculate drift metrics
        psi = self.metrics.calculate_psi(
            reference_predictions, 
            current_predictions,
            self.config.n_bins_psi
        )
        kl = self.metrics.calculate_kl_divergence(
            reference_predictions,
            current_predictions,
            self.config.n_bins_psi
        )
        wasserstein = self.metrics.calculate_wasserstein(
            reference_predictions,
            current_predictions
        )
        
        # Calculate statistics
        ref_stats = {
            'mean': float(reference_predictions.mean()),
            'std': float(reference_predictions.std()),
            'min': float(reference_predictions.min()),
            'max': float(reference_predictions.max())
        }
        
        curr_stats = {
            'mean': float(current_predictions.mean()),
            'std': float(current_predictions.std()),
            'min': float(current_predictions.min()),
            'max': float(current_predictions.max())
        }
        
        # Check class distribution shift (for classification)
        class_shift = None
        if reference_predictions.nunique() <= 10:  # Assume classification
            ref_dist = reference_predictions.value_counts(normalize=True)
            curr_dist = current_predictions.value_counts(normalize=True)
            
            # Ensure same index
            all_classes = set(ref_dist.index) | set(curr_dist.index)
            ref_dist = ref_dist.reindex(all_classes, fill_value=0)
            curr_dist = curr_dist.reindex(all_classes, fill_value=0)
            
            class_shift = float(abs(ref_dist - curr_dist).sum() / 2)
        
        severity = self._determine_severity(psi, kl, wasserstein)
        
        return {
            'psi': psi,
            'kl_divergence': kl,
            'wasserstein_distance': wasserstein,
            'reference_stats': ref_stats,
            'current_stats': curr_stats,
            'class_distribution_shift': class_shift,
            'severity': severity,
            'has_drift': severity in ['warning', 'critical'],
            'timestamp': datetime.now().isoformat()
        }
    
    def _determine_severity(self, psi: float, kl: float, wasserstein: float) -> str:
        """Determine drift severity."""
        if (psi >= self.config.psi_threshold_critical or
            kl >= self.config.kl_threshold_critical or
            wasserstein >= self.config.wasserstein_threshold_critical):
            return 'critical'
        elif (psi >= self.config.psi_threshold_warning or
              kl >= self.config.kl_threshold_warning or
              wasserstein >= self.config.wasserstein_threshold_warning):
            return 'warning'
        else:
            return 'normal'


class ConceptDriftDetector:
    """Detect concept drift (change in P(y|X))."""
    
    def __init__(self, config: DriftConfig):
        """
        Initialize concept drift detector.
        
        Args:
            config: Drift configuration
        """
        self.config = config
    
    def detect(
        self,
        reference_features: pd.DataFrame,
        reference_targets: pd.Series,
        current_features: pd.DataFrame,
        current_targets: pd.Series
    ) -> Dict:
        """
        Detect concept drift.
        
        Args:
            reference_features: Reference features
            reference_targets: Reference targets
            current_features: Current features
            current_targets: Current targets
            
        Returns:
            Dictionary with concept drift analysis
        """
        # Check feature-target correlations
        ref_correlations = {}
        curr_correlations = {}
        
        for col in reference_features.columns:
            if pd.api.types.is_numeric_dtype(reference_features[col]):
                ref_correlations[col] = float(reference_features[col].corr(reference_targets))
                if col in current_features.columns:
                    curr_correlations[col] = float(current_features[col].corr(current_targets))
        
        # Calculate correlation changes
        correlation_changes = {}
        for col in ref_correlations:
            if col in curr_correlations:
                change = abs(curr_correlations[col] - ref_correlations[col])
                correlation_changes[col] = change
        
        # Average correlation change
        avg_correlation_change = np.mean(list(correlation_changes.values())) if correlation_changes else 0
        
        # Test conditional homogeneity
        concept_drift_score = self._calculate_concept_drift_score(
            reference_features,
            reference_targets,
            current_features,
            current_targets
        )
        
        has_concept_drift = (
            avg_correlation_change > self.config.correlation_change_threshold or
            concept_drift_score > self.config.concept_drift_score_threshold
        )
        
        return {
            'correlation_changes': correlation_changes,
            'avg_correlation_change': float(avg_correlation_change),
            'concept_drift_score': float(concept_drift_score),
            'has_concept_drift': has_concept_drift,
            'severity': 'critical' if has_concept_drift else 'normal',
            'timestamp': datetime.now().isoformat()
        }
    
    def _calculate_concept_drift_score(
        self,
        ref_features: pd.DataFrame,
        ref_targets: pd.Series,
        curr_features: pd.DataFrame,
        curr_targets: pd.Series
    ) -> float:
        """
        Calculate concept drift score using conditional distribution tests.
        
        Returns:
            Concept drift score
        """
        n_bins = self.config.n_bins_concept
        concept_drift_score = 0
        n_tests = 0
        
        for col in ref_features.columns:
            if not pd.api.types.is_numeric_dtype(ref_features[col]):
                continue
            
            try:
                # Create bins based on reference
                _, bins = pd.qcut(ref_features[col], q=n_bins, retbins=True, duplicates='drop')
                
                # For each bin, compare y distribution
                for i in range(len(bins)-1):
                    ref_mask = (ref_features[col] >= bins[i]) & (ref_features[col] < bins[i+1])
                    curr_mask = (curr_features[col] >= bins[i]) & (curr_features[col] < bins[i+1])
                    
                    if ref_mask.sum() > 10 and curr_mask.sum() > 10:
                        ref_y = ref_targets[ref_mask]
                        curr_y = curr_targets[curr_mask]
                        
                        # KS test for conditional y distribution
                        ks_stat, _ = ks_2samp(ref_y, curr_y)
                        concept_drift_score += ks_stat
                        n_tests += 1
            except Exception:
                continue
        
        # Normalize score
        if n_tests > 0:
            concept_drift_score /= n_tests
        
        return concept_drift_score