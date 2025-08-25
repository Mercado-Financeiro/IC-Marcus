"""Unit tests for drift metrics calculations."""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch

from src.monitoring.drift.metrics import DriftMetrics


class TestDriftMetrics:
    """Test cases for DriftMetrics."""
    
    @pytest.fixture
    def metrics(self):
        """Create DriftMetrics instance."""
        return DriftMetrics()
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        np.random.seed(42)
        reference = pd.Series(np.random.normal(0, 1, 1000))
        current_no_drift = pd.Series(np.random.normal(0, 1, 1000))
        current_with_drift = pd.Series(np.random.normal(2, 1.5, 1000))
        return reference, current_no_drift, current_with_drift
    
    def test_calculate_psi_no_drift(self, metrics, sample_data):
        """Test PSI calculation with no drift."""
        reference, current_no_drift, _ = sample_data
        
        psi = metrics.calculate_psi(reference, current_no_drift, n_bins=10)
        
        assert isinstance(psi, float)
        assert 0 <= psi < 0.1  # No significant drift
    
    def test_calculate_psi_with_drift(self, metrics, sample_data):
        """Test PSI calculation with drift."""
        reference, _, current_with_drift = sample_data
        
        psi = metrics.calculate_psi(reference, current_with_drift, n_bins=10)
        
        assert isinstance(psi, float)
        assert psi > 0.2  # Significant drift
    
    def test_calculate_psi_identical_distributions(self, metrics):
        """Test PSI with identical distributions."""
        data = pd.Series(np.random.normal(0, 1, 500))
        
        psi = metrics.calculate_psi(data, data, n_bins=10)
        
        assert psi < 0.01  # Very small PSI for identical data
    
    def test_calculate_kl_divergence_no_drift(self, metrics, sample_data):
        """Test KL divergence with no drift."""
        reference, current_no_drift, _ = sample_data
        
        kl = metrics.calculate_kl_divergence(reference, current_no_drift, n_bins=10)
        
        assert isinstance(kl, float)
        assert kl >= 0
        assert kl < 0.1  # Small KL for similar distributions
    
    def test_calculate_kl_divergence_with_drift(self, metrics, sample_data):
        """Test KL divergence with drift."""
        reference, _, current_with_drift = sample_data
        
        kl = metrics.calculate_kl_divergence(reference, current_with_drift, n_bins=10)
        
        assert isinstance(kl, float)
        assert kl > 0.2  # Larger KL for different distributions
    
    def test_calculate_js_divergence(self, metrics, sample_data):
        """Test Jensen-Shannon divergence."""
        reference, current_no_drift, current_with_drift = sample_data
        
        js_no_drift = metrics.calculate_js_divergence(reference, current_no_drift)
        js_with_drift = metrics.calculate_js_divergence(reference, current_with_drift)
        
        assert 0 <= js_no_drift <= 1  # JS is bounded
        assert 0 <= js_with_drift <= 1
        assert js_with_drift > js_no_drift  # More divergence with drift
    
    def test_calculate_wasserstein_no_drift(self, metrics, sample_data):
        """Test Wasserstein distance with no drift."""
        reference, current_no_drift, _ = sample_data
        
        distance = metrics.calculate_wasserstein(reference, current_no_drift)
        
        assert isinstance(distance, float)
        assert distance >= 0
        assert distance < 0.1  # Small distance for similar distributions
    
    def test_calculate_wasserstein_with_drift(self, metrics, sample_data):
        """Test Wasserstein distance with drift."""
        reference, _, current_with_drift = sample_data
        
        distance = metrics.calculate_wasserstein(reference, current_with_drift)
        
        assert isinstance(distance, float)
        assert distance > 0.1  # Larger distance with drift
    
    def test_calculate_ks_statistic(self, metrics, sample_data):
        """Test Kolmogorov-Smirnov test."""
        reference, current_no_drift, current_with_drift = sample_data
        
        # Test with no drift
        ks_stat_no, p_value_no = metrics.calculate_ks_statistic(reference, current_no_drift)
        assert 0 <= ks_stat_no <= 1
        assert 0 <= p_value_no <= 1
        assert p_value_no > 0.05  # No significant difference
        
        # Test with drift
        ks_stat_drift, p_value_drift = metrics.calculate_ks_statistic(reference, current_with_drift)
        assert 0 <= ks_stat_drift <= 1
        assert 0 <= p_value_drift <= 1
        assert p_value_drift < 0.05  # Significant difference
        assert ks_stat_drift > ks_stat_no
    
    def test_calculate_statistics(self, metrics):
        """Test statistics calculation."""
        data = pd.DataFrame({
            'feature1': np.random.normal(0, 1, 100),
            'feature2': np.random.exponential(2, 100),
            'categorical': ['A', 'B', 'C'] * 33 + ['A']  # Non-numeric
        })
        
        stats = metrics.calculate_statistics(data)
        
        assert 'feature1' in stats
        assert 'feature2' in stats
        assert 'categorical' not in stats  # Non-numeric excluded
        
        # Check statistics for feature1
        f1_stats = stats['feature1']
        assert 'mean' in f1_stats
        assert 'std' in f1_stats
        assert 'min' in f1_stats
        assert 'max' in f1_stats
        assert 'q25' in f1_stats
        assert 'q50' in f1_stats
        assert 'q75' in f1_stats
        assert 'skew' in f1_stats
        assert 'kurt' in f1_stats
        
        # Check values are floats
        for key, value in f1_stats.items():
            assert isinstance(value, float)
    
    def test_edge_case_constant_values(self, metrics):
        """Test with constant values."""
        reference = pd.Series([5.0] * 100)
        current = pd.Series([5.0] * 100)
        
        # PSI should handle constant values
        psi = metrics.calculate_psi(reference, current, n_bins=5)
        assert psi < 0.01
        
        # Wasserstein should be 0 for identical constants
        distance = metrics.calculate_wasserstein(reference, current)
        assert distance < 0.01
    
    def test_edge_case_small_sample(self, metrics):
        """Test with small sample sizes."""
        reference = pd.Series([1, 2, 3, 4, 5])
        current = pd.Series([2, 3, 4, 5, 6])
        
        # Should handle small samples
        psi = metrics.calculate_psi(reference, current, n_bins=3)
        assert isinstance(psi, float)
        
        kl = metrics.calculate_kl_divergence(reference, current, n_bins=3)
        assert isinstance(kl, float)
    
    @pytest.mark.parametrize("n_bins", [5, 10, 20])
    def test_different_bin_sizes(self, metrics, n_bins):
        """Test metrics with different bin sizes."""
        np.random.seed(42)
        reference = pd.Series(np.random.normal(0, 1, 500))
        current = pd.Series(np.random.normal(0.5, 1, 500))
        
        psi = metrics.calculate_psi(reference, current, n_bins=n_bins)
        kl = metrics.calculate_kl_divergence(reference, current, n_bins=n_bins)
        
        assert isinstance(psi, float)
        assert isinstance(kl, float)
        assert psi > 0
        assert kl > 0