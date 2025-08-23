"""Unit tests for drift detectors."""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, MagicMock
from datetime import datetime

from src.monitoring.drift.config import DriftConfig
from src.monitoring.drift.metrics import DriftMetrics
from src.monitoring.drift.detectors import (
    FeatureDriftDetector,
    PredictionDriftDetector,
    ConceptDriftDetector
)


class TestFeatureDriftDetector:
    """Test cases for FeatureDriftDetector."""
    
    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return DriftConfig(
            psi_threshold_warning=0.1,
            psi_threshold_critical=0.2,
            min_samples_for_detection=50
        )
    
    @pytest.fixture
    def detector(self, config):
        """Create feature drift detector."""
        metrics = DriftMetrics()
        return FeatureDriftDetector(config, metrics)
    
    @pytest.fixture
    def sample_data(self):
        """Create sample feature data."""
        np.random.seed(42)
        reference = pd.DataFrame({
            'feature1': np.random.normal(0, 1, 200),
            'feature2': np.random.exponential(2, 200),
            'feature3': np.random.uniform(0, 1, 200)
        })
        
        # No drift
        current_no_drift = pd.DataFrame({
            'feature1': np.random.normal(0, 1, 200),
            'feature2': np.random.exponential(2, 200),
            'feature3': np.random.uniform(0, 1, 200)
        })
        
        # With drift
        current_with_drift = pd.DataFrame({
            'feature1': np.random.normal(2, 1.5, 200),  # Drift
            'feature2': np.random.exponential(3, 200),  # Drift
            'feature3': np.random.uniform(0, 1, 200)    # No drift
        })
        
        return reference, current_no_drift, current_with_drift
    
    def test_detect_no_drift(self, detector, sample_data):
        """Test detection with no drift."""
        reference, current_no_drift, _ = sample_data
        
        results = detector.detect(reference, current_no_drift)
        
        assert len(results) == 3
        
        # Count features with drift
        features_with_drift = sum(1 for r in results.values() if r['has_drift'])
        
        # Most features should not have drift (allow for some random variation)
        assert features_with_drift <= 1  # At most 1 feature with warning
        
        for feature, result in results.items():
            assert result['severity'] in ['normal', 'warning']  # No critical drift
            assert 'psi' in result
            assert 'kl_divergence' in result
            assert 'wasserstein_distance' in result
    
    def test_detect_with_drift(self, detector, sample_data):
        """Test detection with drift."""
        reference, _, current_with_drift = sample_data
        
        results = detector.detect(reference, current_with_drift)
        
        # feature1 and feature2 should have drift
        assert results['feature1']['has_drift']
        assert results['feature2']['has_drift']
        assert not results['feature3']['has_drift']
        
        assert results['feature1']['severity'] in ['warning', 'critical']
        assert results['feature2']['severity'] in ['warning', 'critical']
    
    def test_detect_specific_features(self, detector, sample_data):
        """Test detection on specific features only."""
        reference, current_no_drift, _ = sample_data
        
        results = detector.detect(
            reference,
            current_no_drift,
            features=['feature1', 'feature3']
        )
        
        assert len(results) == 2
        assert 'feature1' in results
        assert 'feature3' in results
        assert 'feature2' not in results
    
    def test_detect_insufficient_samples(self, detector):
        """Test with insufficient samples."""
        reference = pd.DataFrame({'feature1': np.random.normal(0, 1, 100)})
        current = pd.DataFrame({'feature1': np.random.normal(0, 1, 30)})  # Below threshold
        
        results = detector.detect(reference, current)
        
        assert len(results) == 0  # No results due to insufficient samples
    
    def test_stats_comparison(self, detector, sample_data):
        """Test statistics comparison in results."""
        reference, _, current_with_drift = sample_data
        
        results = detector.detect(reference, current_with_drift)
        
        for feature, result in results.items():
            assert 'stats_comparison' in result
            stats = result['stats_comparison']
            assert 'mean_shift' in stats
            assert 'std_ratio' in stats
            assert 'range_shift' in stats


class TestPredictionDriftDetector:
    """Test cases for PredictionDriftDetector."""
    
    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return DriftConfig()
    
    @pytest.fixture
    def detector(self, config):
        """Create prediction drift detector."""
        metrics = DriftMetrics()
        return PredictionDriftDetector(config, metrics)
    
    def test_detect_regression_no_drift(self, detector):
        """Test regression predictions with no drift."""
        np.random.seed(42)
        reference = pd.Series(np.random.normal(0.5, 0.1, 500))
        current = pd.Series(np.random.normal(0.5, 0.1, 500))
        
        result = detector.detect(reference, current)
        
        assert not result['has_drift']
        assert result['severity'] == 'normal'
        assert result['psi'] < 0.1
        assert result['class_distribution_shift'] is None  # Regression
    
    def test_detect_regression_with_drift(self, detector):
        """Test regression predictions with drift."""
        np.random.seed(42)
        reference = pd.Series(np.random.normal(0.5, 0.1, 500))
        current = pd.Series(np.random.normal(0.7, 0.15, 500))
        
        result = detector.detect(reference, current)
        
        assert result['has_drift']
        assert result['severity'] in ['warning', 'critical']
        assert result['psi'] > 0.1
    
    def test_detect_classification_no_drift(self, detector):
        """Test classification predictions with no drift."""
        np.random.seed(42)
        # Use the same probabilities but different seeds for some variation
        reference = pd.Series(np.random.choice([0, 1], 500, p=[0.6, 0.4]))
        np.random.seed(43)  # Different seed but same distribution
        current = pd.Series(np.random.choice([0, 1], 500, p=[0.6, 0.4]))
        
        result = detector.detect(reference, current)
        
        # Allow for some variation in random sampling
        if result['has_drift']:
            # If drift detected, it should be minor (warning at most)
            assert result['severity'] == 'warning'
            assert result['class_distribution_shift'] < 0.15
        else:
            assert result['class_distribution_shift'] < 0.1
    
    def test_detect_classification_with_drift(self, detector):
        """Test classification predictions with drift."""
        np.random.seed(42)
        reference = pd.Series(np.random.choice([0, 1], 500, p=[0.7, 0.3]))
        current = pd.Series(np.random.choice([0, 1], 500, p=[0.3, 0.7]))
        
        result = detector.detect(reference, current)
        
        assert result['has_drift']
        assert result['class_distribution_shift'] > 0.2
    
    def test_statistics_in_result(self, detector):
        """Test that statistics are included in result."""
        reference = pd.Series(np.random.normal(0, 1, 100))
        current = pd.Series(np.random.normal(1, 1, 100))
        
        result = detector.detect(reference, current)
        
        assert 'reference_stats' in result
        assert 'current_stats' in result
        
        for stats in [result['reference_stats'], result['current_stats']]:
            assert 'mean' in stats
            assert 'std' in stats
            assert 'min' in stats
            assert 'max' in stats


class TestConceptDriftDetector:
    """Test cases for ConceptDriftDetector."""
    
    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return DriftConfig(
            correlation_change_threshold=0.3,
            concept_drift_score_threshold=0.2
        )
    
    @pytest.fixture
    def detector(self, config):
        """Create concept drift detector."""
        return ConceptDriftDetector(config)
    
    def test_detect_no_concept_drift(self, detector):
        """Test with no concept drift."""
        np.random.seed(42)
        n_samples = 500
        
        # Create correlated features and targets
        X_ref = pd.DataFrame({
            'feature1': np.random.normal(0, 1, n_samples),
            'feature2': np.random.normal(0, 1, n_samples)
        })
        y_ref = pd.Series(X_ref['feature1'] * 0.5 + np.random.normal(0, 0.1, n_samples))
        
        X_curr = pd.DataFrame({
            'feature1': np.random.normal(0, 1, n_samples),
            'feature2': np.random.normal(0, 1, n_samples)
        })
        y_curr = pd.Series(X_curr['feature1'] * 0.5 + np.random.normal(0, 0.1, n_samples))
        
        result = detector.detect(X_ref, y_ref, X_curr, y_curr)
        
        assert not result['has_concept_drift']
        assert result['severity'] == 'normal'
        assert result['avg_correlation_change'] < 0.3
    
    def test_detect_with_concept_drift(self, detector):
        """Test with concept drift."""
        np.random.seed(42)
        n_samples = 500
        
        # Reference: feature1 correlated with target
        X_ref = pd.DataFrame({
            'feature1': np.random.normal(0, 1, n_samples),
            'feature2': np.random.normal(0, 1, n_samples)
        })
        y_ref = pd.Series(X_ref['feature1'] * 0.8 + np.random.normal(0, 0.1, n_samples))
        
        # Current: feature2 correlated with target (concept change)
        X_curr = pd.DataFrame({
            'feature1': np.random.normal(0, 1, n_samples),
            'feature2': np.random.normal(0, 1, n_samples)
        })
        y_curr = pd.Series(X_curr['feature2'] * 0.8 + np.random.normal(0, 0.1, n_samples))
        
        result = detector.detect(X_ref, y_ref, X_curr, y_curr)
        
        assert result['has_concept_drift']
        assert result['severity'] == 'critical'
        assert result['avg_correlation_change'] > 0.3
    
    def test_correlation_changes(self, detector):
        """Test correlation change detection."""
        np.random.seed(42)
        n_samples = 200
        
        X_ref = pd.DataFrame({
            'feature1': np.random.normal(0, 1, n_samples),
            'feature2': np.random.normal(0, 1, n_samples),
            'categorical': ['A', 'B'] * (n_samples // 2)  # Non-numeric
        })
        y_ref = pd.Series(np.random.normal(0, 1, n_samples))
        
        X_curr = X_ref.copy()
        y_curr = y_ref.copy()
        
        result = detector.detect(X_ref, y_ref, X_curr, y_curr)
        
        assert 'correlation_changes' in result
        assert 'feature1' in result['correlation_changes']
        assert 'feature2' in result['correlation_changes']
        assert 'categorical' not in result['correlation_changes']  # Non-numeric excluded
    
    def test_concept_drift_score_calculation(self, detector):
        """Test concept drift score calculation."""
        np.random.seed(42)
        n_samples = 300
        
        X_ref = pd.DataFrame({
            'feature1': np.random.uniform(0, 10, n_samples)
        })
        y_ref = pd.Series(np.where(X_ref['feature1'] > 5, 1, 0))
        
        X_curr = pd.DataFrame({
            'feature1': np.random.uniform(0, 10, n_samples)
        })
        # Changed relationship
        y_curr = pd.Series(np.where(X_curr['feature1'] > 7, 1, 0))
        
        result = detector.detect(X_ref, y_ref, X_curr, y_curr)
        
        assert 'concept_drift_score' in result
        assert result['concept_drift_score'] > 0
        assert result['has_concept_drift']