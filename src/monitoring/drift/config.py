"""Configuration for drift monitoring."""

from dataclasses import dataclass


@dataclass
class DriftConfig:
    """Configuration for drift monitoring system."""
    
    # PSI thresholds
    psi_threshold_warning: float = 0.1
    psi_threshold_critical: float = 0.2
    
    # KL divergence thresholds
    kl_threshold_warning: float = 0.1
    kl_threshold_critical: float = 0.2
    
    # Wasserstein distance thresholds
    wasserstein_threshold_warning: float = 0.05
    wasserstein_threshold_critical: float = 0.1
    
    # Feature monitoring thresholds
    feature_importance_shift_threshold: float = 0.3
    performance_degradation_threshold: float = 0.2
    
    # Monitoring parameters
    min_samples_for_detection: int = 100
    n_bins_psi: int = 10
    window_size_days: int = 7
    
    # Concept drift parameters
    n_bins_concept: int = 5
    correlation_change_threshold: float = 0.3
    concept_drift_score_threshold: float = 0.2