"""Unit tests for the main drift monitor."""

import pytest
import numpy as np
import pandas as pd
import json
import tempfile
from unittest.mock import Mock, patch
from datetime import datetime, timedelta

from src.monitoring.drift.config import DriftConfig
from src.monitoring.drift.monitor import DriftMonitor


class TestDriftMonitor:
    """Test cases for DriftMonitor."""
    
    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return DriftConfig(
            min_samples_for_detection=50,
            window_size_days=7
        )
    
    @pytest.fixture
    def monitor(self, config):
        """Create drift monitor."""
        return DriftMonitor(config)
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        np.random.seed(42)
        n_samples = 200
        
        reference_data = pd.DataFrame({
            'feature1': np.random.normal(0, 1, n_samples),
            'feature2': np.random.exponential(2, n_samples),
            'feature3': np.random.uniform(0, 1, n_samples)
        })
        
        reference_predictions = pd.Series(np.random.choice([0, 1], n_samples, p=[0.6, 0.4]))
        reference_targets = pd.Series(np.random.choice([0, 1], n_samples, p=[0.5, 0.5]))
        
        # Current data with some drift
        current_data = pd.DataFrame({
            'feature1': np.random.normal(0.5, 1.2, n_samples),  # Slight drift
            'feature2': np.random.exponential(2, n_samples),     # No drift
            'feature3': np.random.uniform(0, 1, n_samples)       # No drift
        })
        
        current_predictions = pd.Series(np.random.choice([0, 1], n_samples, p=[0.5, 0.5]))
        current_targets = pd.Series(np.random.choice([0, 1], n_samples, p=[0.5, 0.5]))
        
        return {
            'reference_data': reference_data,
            'reference_predictions': reference_predictions,
            'reference_targets': reference_targets,
            'current_data': current_data,
            'current_predictions': current_predictions,
            'current_targets': current_targets
        }
    
    def test_initialization(self, monitor):
        """Test monitor initialization."""
        assert monitor.reference_data is None
        assert monitor.reference_stats is None
        assert monitor.reference_predictions is None
        assert monitor.reference_targets is None
        assert monitor.drift_history == []
    
    def test_set_reference(self, monitor, sample_data):
        """Test setting reference data."""
        monitor.set_reference(
            sample_data['reference_data'],
            sample_data['reference_predictions'],
            sample_data['reference_targets']
        )
        
        assert monitor.reference_data is not None
        assert len(monitor.reference_data) == len(sample_data['reference_data'])
        assert monitor.reference_predictions is not None
        assert monitor.reference_targets is not None
        assert monitor.reference_stats is not None
    
    def test_monitor_without_reference_raises(self, monitor, sample_data):
        """Test that monitoring without reference raises error."""
        with pytest.raises(ValueError, match="Reference data not set"):
            monitor.monitor(sample_data['current_data'])
    
    def test_monitor_features_only(self, monitor, sample_data):
        """Test monitoring with features only."""
        monitor.set_reference(sample_data['reference_data'])
        
        report = monitor.monitor(sample_data['current_data'])
        
        assert 'features' in report
        assert len(report['features']) == 3
        assert report['predictions'] is None
        assert report['concept'] is None
        assert 'timestamp' in report
        assert 'overall_status' in report
    
    def test_monitor_with_predictions(self, monitor, sample_data):
        """Test monitoring with predictions."""
        monitor.set_reference(
            sample_data['reference_data'],
            sample_data['reference_predictions']
        )
        
        report = monitor.monitor(
            sample_data['current_data'],
            sample_data['current_predictions']
        )
        
        assert report['predictions'] is not None
        assert 'psi' in report['predictions']
        assert 'has_drift' in report['predictions']
    
    def test_monitor_with_concept(self, monitor, sample_data):
        """Test monitoring with concept drift."""
        monitor.set_reference(
            sample_data['reference_data'],
            targets=sample_data['reference_targets']
        )
        
        report = monitor.monitor(
            sample_data['current_data'],
            current_targets=sample_data['current_targets']
        )
        
        assert report['concept'] is not None
        assert 'has_concept_drift' in report['concept']
        assert 'correlation_changes' in report['concept']
    
    def test_monitor_complete(self, monitor, sample_data):
        """Test complete monitoring with all components."""
        monitor.set_reference(
            sample_data['reference_data'],
            sample_data['reference_predictions'],
            sample_data['reference_targets']
        )
        
        report = monitor.monitor(
            sample_data['current_data'],
            sample_data['current_predictions'],
            sample_data['current_targets']
        )
        
        assert report['features'] is not None
        assert report['predictions'] is not None
        assert report['concept'] is not None
        assert len(report['alerts']) >= 0
        assert report['overall_status'] in ['normal', 'warning', 'critical']
    
    def test_determine_overall_status(self, monitor, sample_data):
        """Test overall status determination."""
        monitor.set_reference(sample_data['reference_data'])
        
        # Create report with different severity levels
        report = {
            'features': {
                'feature1': {'severity': 'critical', 'has_drift': True},
                'feature2': {'severity': 'warning', 'has_drift': True},
                'feature3': {'severity': 'normal', 'has_drift': False}
            },
            'predictions': None,
            'concept': None
        }
        
        status = monitor._determine_overall_status(report)
        assert status == 'critical'  # Critical feature present
        
        # Test with warnings only
        report['features']['feature1']['severity'] = 'warning'
        status = monitor._determine_overall_status(report)
        assert status == 'warning'
        
        # Test with no drift
        report['features'] = {
            'feature1': {'severity': 'normal', 'has_drift': False}
        }
        status = monitor._determine_overall_status(report)
        assert status == 'normal'
    
    def test_drift_history_management(self, monitor, sample_data):
        """Test drift history tracking."""
        monitor.set_reference(sample_data['reference_data'])
        
        # Run monitoring multiple times
        for _ in range(3):
            report = monitor.monitor(sample_data['current_data'])
        
        assert len(monitor.drift_history) == 3
        
        # Check history window enforcement
        old_report = monitor.drift_history[0].copy()
        old_report['timestamp'] = (
            datetime.now() - timedelta(days=10)
        ).isoformat()
        monitor.drift_history.insert(0, old_report)
        
        monitor.monitor(sample_data['current_data'])
        
        # Old report should be removed
        assert len(monitor.drift_history) == 4  # 3 recent + 1 new
    
    def test_get_drift_summary_empty(self, monitor):
        """Test drift summary with no history."""
        summary = monitor.get_drift_summary()
        
        assert summary['message'] == 'No drift history available'
    
    def test_get_drift_summary_with_history(self, monitor, sample_data):
        """Test drift summary with history."""
        monitor.set_reference(sample_data['reference_data'])
        
        # Manually create history with controlled statuses
        monitor.drift_history = []
        
        # Add reports with specific statuses
        for i in range(5):
            report = {
                'timestamp': datetime.now().isoformat(),
                'n_samples': 200,
                'features': {},
                'predictions': None,
                'concept': None,
                'overall_status': 'normal',
                'alerts': []
            }
            
            if i == 0:
                report['overall_status'] = 'critical'
            elif i < 3:
                report['overall_status'] = 'warning'
            else:
                report['overall_status'] = 'normal'
            
            monitor.drift_history.append(report)
        
        summary = monitor.get_drift_summary()
        
        assert summary['total_checks'] == 5
        assert summary['critical_incidents'] == 1
        assert summary['warning_incidents'] == 2
        assert summary['normal_percentage'] == 40.0  # 2 normal out of 5
        assert 'top_drifting_features' in summary
        assert 'last_check' in summary
        assert summary['current_status'] == 'normal'  # Last report is normal
    
    def test_export_report(self, monitor, sample_data):
        """Test report export."""
        monitor.set_reference(sample_data['reference_data'])
        
        # Run monitoring
        monitor.monitor(sample_data['current_data'])
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            filepath = f.name
        
        monitor.export_report(filepath)
        
        # Load and verify exported report
        with open(filepath, 'r') as f:
            exported = json.load(f)
        
        assert 'summary' in exported
        assert 'history' in exported
        assert 'config' in exported
        
        # Clean up
        import os
        os.unlink(filepath)
    
    def test_alerts_generation(self, monitor, sample_data):
        """Test alert generation in reports."""
        # Create data with significant drift
        np.random.seed(42)
        n_samples = 200
        
        reference = pd.DataFrame({
            'feature1': np.random.normal(0, 1, n_samples),
            'feature2': np.random.normal(0, 1, n_samples)
        })
        
        current = pd.DataFrame({
            'feature1': np.random.normal(3, 1, n_samples),  # Large drift
            'feature2': np.random.normal(3, 1, n_samples)   # Large drift
        })
        
        monitor.set_reference(reference)
        report = monitor.monitor(current)
        
        assert len(report['alerts']) > 0
        assert any('drift detected' in alert for alert in report['alerts'])