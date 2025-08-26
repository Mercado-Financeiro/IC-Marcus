"""Drift detection metrics and calculations."""

import numpy as np
import pandas as pd
from typing import Optional, Tuple
from scipy.spatial.distance import jensenshannon
from scipy.stats import wasserstein_distance, ks_2samp
import structlog

log = structlog.get_logger()


class DriftMetrics:
    """Calculate various drift detection metrics."""
    
    @staticmethod
    def calculate_psi(
        reference: pd.Series,
        current: pd.Series,
        n_bins: int = 10
    ) -> float:
        """
        Calculate Population Stability Index (PSI).
        
        PSI < 0.1: No significant drift
        0.1 <= PSI < 0.2: Moderate drift
        PSI >= 0.2: Significant drift
        
        Args:
            reference: Reference series
            current: Current series
            n_bins: Number of bins for discretization
            
        Returns:
            PSI value
        """
        # Create bins based on reference data
        try:
            _, bins = pd.qcut(reference, q=n_bins, retbins=True, duplicates='drop')
        except Exception:
            # Fallback to uniform bins if qcut fails
            _, bins = pd.cut(reference, bins=n_bins, retbins=True)
        
        # Add small epsilon to avoid division by zero
        eps = 1e-10
        
        # Calculate distributions
        ref_counts = pd.cut(reference, bins=bins, include_lowest=True).value_counts()
        curr_counts = pd.cut(current, bins=bins, include_lowest=True).value_counts()
        
        # Normalize to probabilities
        ref_probs = (ref_counts + eps) / (len(reference) + n_bins * eps)
        curr_probs = (curr_counts + eps) / (len(current) + n_bins * eps)
        
        # Ensure same index
        ref_probs = ref_probs.reindex(ref_counts.index, fill_value=eps)
        curr_probs = curr_probs.reindex(ref_counts.index, fill_value=eps)
        
        # Calculate PSI
        psi = np.sum((curr_probs - ref_probs) * np.log(curr_probs / ref_probs))
        
        return float(psi)
    
    @staticmethod
    def calculate_kl_divergence(
        reference: pd.Series,
        current: pd.Series,
        n_bins: int = 10
    ) -> float:
        """
        Calculate Kullback-Leibler Divergence.
        
        Args:
            reference: Reference series
            current: Current series
            n_bins: Number of bins for discretization
            
        Returns:
            KL divergence value
        """
        # Create histograms
        min_val = min(reference.min(), current.min())
        max_val = max(reference.max(), current.max())
        bins = np.linspace(min_val, max_val, n_bins + 1)
        
        ref_hist, _ = np.histogram(reference, bins=bins)
        curr_hist, _ = np.histogram(current, bins=bins)
        
        # Add epsilon and normalize
        eps = 1e-10
        ref_probs = (ref_hist + eps) / (ref_hist.sum() + n_bins * eps)
        curr_probs = (curr_hist + eps) / (curr_hist.sum() + n_bins * eps)
        
        # KL divergence
        kl = np.sum(ref_probs * np.log(ref_probs / curr_probs))
        
        return float(kl)
    
    @staticmethod
    def calculate_js_divergence(
        reference: pd.Series,
        current: pd.Series,
        n_bins: int = 10
    ) -> float:
        """
        Calculate Jensen-Shannon Divergence.
        
        JS divergence is symmetric and bounded [0, 1].
        
        Args:
            reference: Reference series
            current: Current series
            n_bins: Number of bins
            
        Returns:
            JS divergence value
        """
        # Create histograms
        min_val = min(reference.min(), current.min())
        max_val = max(reference.max(), current.max())
        bins = np.linspace(min_val, max_val, n_bins + 1)
        
        ref_hist, _ = np.histogram(reference, bins=bins)
        curr_hist, _ = np.histogram(current, bins=bins)
        
        # Normalize
        ref_probs = ref_hist / ref_hist.sum()
        curr_probs = curr_hist / curr_hist.sum()
        
        # JS divergence
        js = jensenshannon(ref_probs, curr_probs)
        
        return float(js**2)  # Squared for JS distance
    
    @staticmethod
    def calculate_wasserstein(
        reference: pd.Series,
        current: pd.Series
    ) -> float:
        """
        Calculate Wasserstein Distance (Earth Mover's Distance).
        
        Args:
            reference: Reference series
            current: Current series
            
        Returns:
            Wasserstein distance
        """
        # Normalize to [0, 1] for comparability
        min_val = min(reference.min(), current.min())
        max_val = max(reference.max(), current.max())
        
        if max_val - min_val > 0:
            ref_norm = (reference - min_val) / (max_val - min_val)
            curr_norm = (current - min_val) / (max_val - min_val)
        else:
            ref_norm = reference
            curr_norm = current
        
        distance = wasserstein_distance(ref_norm, curr_norm)
        
        return float(distance)
    
    @staticmethod
    def calculate_ks_statistic(
        reference: pd.Series,
        current: pd.Series
    ) -> Tuple[float, float]:
        """
        Calculate Kolmogorov-Smirnov statistic.
        
        Args:
            reference: Reference series
            current: Current series
            
        Returns:
            Tuple (ks_statistic, p_value)
        """
        ks_stat, p_value = ks_2samp(reference, current)
        
        return float(ks_stat), float(p_value)
    
    @staticmethod
    def calculate_statistics(data: pd.DataFrame) -> dict:
        """
        Calculate comprehensive statistics for data.
        
        Args:
            data: DataFrame with data
            
        Returns:
            Dictionary with statistics per column
        """
        stats_dict = {}
        
        for col in data.columns:
            if pd.api.types.is_numeric_dtype(data[col]):
                stats_dict[col] = {
                    'mean': float(data[col].mean()),
                    'std': float(data[col].std()),
                    'min': float(data[col].min()),
                    'max': float(data[col].max()),
                    'q25': float(data[col].quantile(0.25)),
                    'q50': float(data[col].quantile(0.50)),
                    'q75': float(data[col].quantile(0.75)),
                    'skew': float(data[col].skew()),
                    'kurt': float(data[col].kurtosis())
                }
        
        return stats_dict