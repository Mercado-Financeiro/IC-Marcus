"""Main drift monitoring orchestrator."""

import pandas as pd
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import json
import structlog

from src.monitoring.drift.config import DriftConfig
from src.monitoring.drift.metrics import DriftMetrics
from src.monitoring.drift.detectors import (
    FeatureDriftDetector,
    PredictionDriftDetector,
    ConceptDriftDetector
)

log = structlog.get_logger()


class DriftMonitor:
    """Main drift monitoring system."""
    
    def __init__(self, config: Optional[DriftConfig] = None):
        """
        Initialize drift monitor.
        
        Args:
            config: Drift configuration
        """
        self.config = config or DriftConfig()
        self.metrics = DriftMetrics()
        
        # Initialize detectors
        self.feature_detector = FeatureDriftDetector(self.config, self.metrics)
        self.prediction_detector = PredictionDriftDetector(self.config, self.metrics)
        self.concept_detector = ConceptDriftDetector(self.config)
        
        # Reference data
        self.reference_data = None
        self.reference_stats = None
        self.reference_predictions = None
        self.reference_targets = None
        
        # History tracking
        self.drift_history = []
    
    def set_reference(
        self,
        data: pd.DataFrame,
        predictions: Optional[pd.Series] = None,
        targets: Optional[pd.Series] = None
    ) -> None:
        """
        Set reference data for comparison.
        
        Args:
            data: Reference feature DataFrame
            predictions: Reference predictions (optional)
            targets: Reference targets (optional)
        """
        log.info("setting_reference_data", shape=data.shape)
        
        self.reference_data = data.copy()
        self.reference_stats = self.metrics.calculate_statistics(data)
        
        if predictions is not None:
            self.reference_predictions = predictions.copy()
        
        if targets is not None:
            self.reference_targets = targets.copy()
    
    def monitor(
        self,
        current_data: pd.DataFrame,
        current_predictions: Optional[pd.Series] = None,
        current_targets: Optional[pd.Series] = None
    ) -> Dict:
        """
        Execute complete drift monitoring.
        
        Args:
            current_data: Current feature DataFrame
            current_predictions: Current predictions (optional)
            current_targets: Current targets (optional)
            
        Returns:
            Complete drift monitoring report
        """
        if self.reference_data is None:
            raise ValueError("Reference data not set. Call set_reference() first.")
        
        log.info("running_drift_monitoring", shape=current_data.shape)
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'n_samples': len(current_data),
            'features': {},
            'predictions': None,
            'concept': None,
            'overall_status': 'normal',
            'alerts': []
        }
        
        # Feature drift detection
        feature_drift = self.feature_detector.detect(
            self.reference_data,
            current_data
        )
        report['features'] = feature_drift
        
        # Count features with drift
        features_with_drift = sum(
            1 for f in feature_drift.values() 
            if f['has_drift']
        )
        if features_with_drift > 0:
            report['alerts'].append(
                f"{features_with_drift} features with drift detected"
            )
        
        # Prediction drift detection
        if current_predictions is not None and self.reference_predictions is not None:
            pred_drift = self.prediction_detector.detect(
                self.reference_predictions,
                current_predictions
            )
            report['predictions'] = pred_drift
            
            if pred_drift['has_drift']:
                report['alerts'].append("Prediction distribution drift detected")
        
        # Concept drift detection
        if (current_targets is not None and 
            self.reference_targets is not None):
            concept_drift = self.concept_detector.detect(
                self.reference_data,
                self.reference_targets,
                current_data,
                current_targets
            )
            report['concept'] = concept_drift
            
            if concept_drift['has_concept_drift']:
                report['alerts'].append("Concept drift detected")
        
        # Determine overall status
        report['overall_status'] = self._determine_overall_status(report)
        
        # Add to history
        self._update_history(report)
        
        log.info(
            "drift_monitoring_complete",
            overall_status=report['overall_status'],
            n_alerts=len(report['alerts'])
        )
        
        return report
    
    def _determine_overall_status(self, report: Dict) -> str:
        """
        Determine overall drift status from report.
        
        Args:
            report: Drift report
            
        Returns:
            Overall status: 'normal', 'warning', or 'critical'
        """
        # Count critical and warning features
        critical_features = sum(
            1 for f in report['features'].values() 
            if f.get('severity') == 'critical'
        )
        warning_features = sum(
            1 for f in report['features'].values() 
            if f.get('severity') == 'warning'
        )
        
        # Check for critical conditions
        if (critical_features > 0 or 
            (report['predictions'] and report['predictions'].get('severity') == 'critical') or
            (report['concept'] and report['concept'].get('has_concept_drift'))):
            return 'critical'
        
        # Check for warning conditions
        elif (warning_features > 0 or 
              (report['predictions'] and report['predictions'].get('severity') == 'warning')):
            return 'warning'
        
        return 'normal'
    
    def _update_history(self, report: Dict) -> None:
        """
        Update drift history with new report.
        
        Args:
            report: New drift report
        """
        self.drift_history.append(report)
        
        # Keep only recent history
        cutoff_date = datetime.now() - timedelta(days=self.config.window_size_days)
        self.drift_history = [
            r for r in self.drift_history 
            if datetime.fromisoformat(r['timestamp']) > cutoff_date
        ]
    
    def get_drift_summary(self) -> Dict:
        """
        Get summary of drift history.
        
        Returns:
            Dictionary with drift statistics
        """
        if not self.drift_history:
            return {'message': 'No drift history available'}
        
        # Aggregate statistics
        total_checks = len(self.drift_history)
        critical_count = sum(
            1 for r in self.drift_history 
            if r['overall_status'] == 'critical'
        )
        warning_count = sum(
            1 for r in self.drift_history 
            if r['overall_status'] == 'warning'
        )
        
        # Find features with most drift
        feature_drift_counts = {}
        for report in self.drift_history:
            for feature, drift_info in report.get('features', {}).items():
                if drift_info.get('has_drift'):
                    feature_drift_counts[feature] = feature_drift_counts.get(feature, 0) + 1
        
        top_drifting_features = sorted(
            feature_drift_counts.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:5]
        
        normal_count = total_checks - critical_count - warning_count
        
        return {
            'total_checks': total_checks,
            'critical_incidents': critical_count,
            'warning_incidents': warning_count,
            'normal_percentage': (normal_count / total_checks * 100) if total_checks > 0 else 0,
            'top_drifting_features': top_drifting_features,
            'last_check': self.drift_history[-1]['timestamp'] if self.drift_history else None,
            'current_status': self.drift_history[-1]['overall_status'] if self.drift_history else 'unknown'
        }
    
    def export_report(self, filepath: str) -> None:
        """
        Export drift report to file.
        
        Args:
            filepath: Path to save report
        """
        report = {
            'summary': self.get_drift_summary(),
            'history': self.drift_history[-10:],  # Last 10 reports
            'config': {
                'psi_threshold_warning': self.config.psi_threshold_warning,
                'psi_threshold_critical': self.config.psi_threshold_critical,
                'kl_threshold_warning': self.config.kl_threshold_warning,
                'kl_threshold_critical': self.config.kl_threshold_critical,
                'wasserstein_threshold_warning': self.config.wasserstein_threshold_warning,
                'wasserstein_threshold_critical': self.config.wasserstein_threshold_critical
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        log.info("drift_report_exported", filepath=filepath)