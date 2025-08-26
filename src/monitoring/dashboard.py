"""
Real-time Data Quality Monitoring Dashboard.
Provides live visibility into data quality metrics and issues.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import time
import json
from pathlib import Path
from typing import Dict, List, Optional, Any
import logging
from collections import deque
import threading
import asyncio
import websocket

from src.features.circuit_breaker import FeatureProcessingBreaker
from src.data.quality_pipeline import DataQualityPipeline

logger = logging.getLogger(__name__)


class DataQualityMonitor:
    """Real-time monitoring of data quality metrics."""
    
    def __init__(self, 
                 history_size: int = 1000,
                 update_interval: int = 5):
        """
        Initialize the monitor.
        
        Args:
            history_size: Number of historical points to keep
            update_interval: Update interval in seconds
        """
        self.history_size = history_size
        self.update_interval = update_interval
        
        # Metrics storage with sliding window
        self.metrics = {
            'validation_pass_rate': deque(maxlen=history_size),
            'feature_count': deque(maxlen=history_size),
            'sample_count': deque(maxlen=history_size),
            'processing_time': deque(maxlen=history_size),
            'memory_usage': deque(maxlen=history_size),
            'circuit_breaker_status': deque(maxlen=history_size),
            'timestamps': deque(maxlen=history_size)
        }
        
        # Drift metrics
        self.drift_metrics = {
            'feature_drift': {},
            'target_drift': [],
            'data_quality_score': []
        }
        
        # Alerts
        self.alerts = deque(maxlen=100)
        
        # Circuit breaker status
        self.breaker = FeatureProcessingBreaker()
        
    def update_metrics(self, pipeline_result: Dict[str, Any]):
        """Update metrics from pipeline results."""
        timestamp = datetime.now()
        
        # Extract metrics
        metrics = pipeline_result.get('metrics', {})
        
        # Update time series
        self.metrics['timestamps'].append(timestamp)
        self.metrics['validation_pass_rate'].append(
            metrics.get('validation_pass_rate', 0) * 100
        )
        self.metrics['feature_count'].append(
            metrics.get('filtered_features', 0)
        )
        self.metrics['sample_count'].append(
            metrics.get('train_samples', 0)
        )
        self.metrics['processing_time'].append(
            metrics.get('processing_time', 0)
        )
        self.metrics['memory_usage'].append(
            metrics.get('memory_usage_mb', 0)
        )
        
        # Circuit breaker status
        cb_stats = self.breaker.get_all_stats()
        open_count = sum(1 for b in cb_stats.values() 
                        if b.get('current_state') == 'open')
        self.metrics['circuit_breaker_status'].append(open_count)
        
        # Check for alerts
        self._check_alerts(metrics)
    
    def _check_alerts(self, metrics: Dict[str, Any]):
        """Check metrics against thresholds and generate alerts."""
        timestamp = datetime.now()
        
        # Validation pass rate alert
        pass_rate = metrics.get('validation_pass_rate', 0)
        if pass_rate < 0.9:
            self.alerts.append({
                'timestamp': timestamp,
                'severity': 'high' if pass_rate < 0.7 else 'medium',
                'type': 'validation_failure',
                'message': f'Validation pass rate low: {pass_rate:.1%}',
                'value': pass_rate
            })
        
        # Feature reduction alert
        reduction = metrics.get('feature_reduction_pct', 0)
        if reduction > 80:
            self.alerts.append({
                'timestamp': timestamp,
                'severity': 'medium',
                'type': 'feature_reduction',
                'message': f'High feature reduction: {reduction:.1f}%',
                'value': reduction
            })
        
        # Memory usage alert
        memory = metrics.get('memory_usage_mb', 0)
        if memory > 2048:
            self.alerts.append({
                'timestamp': timestamp,
                'severity': 'high' if memory > 4096 else 'medium',
                'type': 'memory_usage',
                'message': f'High memory usage: {memory:.0f} MB',
                'value': memory
            })
    
    def get_latest_metrics(self) -> Dict[str, Any]:
        """Get latest metric values."""
        if not self.metrics['timestamps']:
            return {}
        
        return {
            'timestamp': self.metrics['timestamps'][-1],
            'validation_pass_rate': self.metrics['validation_pass_rate'][-1] if self.metrics['validation_pass_rate'] else 0,
            'feature_count': self.metrics['feature_count'][-1] if self.metrics['feature_count'] else 0,
            'sample_count': self.metrics['sample_count'][-1] if self.metrics['sample_count'] else 0,
            'processing_time': self.metrics['processing_time'][-1] if self.metrics['processing_time'] else 0,
            'memory_usage': self.metrics['memory_usage'][-1] if self.metrics['memory_usage'] else 0,
            'circuit_breakers_open': self.metrics['circuit_breaker_status'][-1] if self.metrics['circuit_breaker_status'] else 0
        }
    
    def get_recent_alerts(self, n: int = 10) -> List[Dict]:
        """Get recent alerts."""
        return list(self.alerts)[-n:]


def create_dashboard():
    """Create Streamlit dashboard for data quality monitoring."""
    
    st.set_page_config(
        page_title="Data Quality Monitor",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS
    st.markdown("""
    <style>
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
    }
    .alert-high {
        background-color: #ff4b4b;
        color: white;
        padding: 10px;
        border-radius: 5px;
        margin: 5px 0;
    }
    .alert-medium {
        background-color: #ffa500;
        color: white;
        padding: 10px;
        border-radius: 5px;
        margin: 5px 0;
    }
    .alert-low {
        background-color: #00cc88;
        color: white;
        padding: 10px;
        border-radius: 5px;
        margin: 5px 0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Initialize monitor in session state
    if 'monitor' not in st.session_state:
        st.session_state.monitor = DataQualityMonitor()
        st.session_state.last_update = datetime.now()
    
    monitor = st.session_state.monitor
    
    # Header
    st.title("🔍 Data Quality Monitoring Dashboard")
    st.markdown("Real-time monitoring of data quality metrics and issues")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        auto_refresh = st.checkbox("Auto Refresh", value=True)
        refresh_interval = st.slider("Refresh Interval (seconds)", 1, 60, 5)
        
        st.divider()
        
        # Load sample data button
        if st.button("Load Sample Data"):
            load_sample_data(monitor)
            st.success("Sample data loaded!")
        
        st.divider()
        
        # Export metrics
        if st.button("Export Metrics"):
            export_metrics(monitor)
            st.success("Metrics exported!")
    
    # Main dashboard
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Overview", 
        "📈 Trends", 
        "🚨 Alerts", 
        "🔧 Circuit Breakers",
        "📋 Reports"
    ])
    
    with tab1:
        render_overview(monitor)
    
    with tab2:
        render_trends(monitor)
    
    with tab3:
        render_alerts(monitor)
    
    with tab4:
        render_circuit_breakers(monitor)
    
    with tab5:
        render_reports(monitor)
    
    # Auto refresh
    if auto_refresh:
        time.sleep(refresh_interval)
        st.rerun()


def render_overview(monitor: DataQualityMonitor):
    """Render overview metrics."""
    st.header("Overview")
    
    latest = monitor.get_latest_metrics()
    
    if not latest:
        st.info("No data available yet. Load sample data to get started.")
        return
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Validation Pass Rate",
            f"{latest.get('validation_pass_rate', 0):.1f}%",
            delta=calculate_delta(monitor.metrics['validation_pass_rate'])
        )
    
    with col2:
        st.metric(
            "Active Features",
            latest.get('feature_count', 0),
            delta=calculate_delta(monitor.metrics['feature_count'])
        )
    
    with col3:
        st.metric(
            "Processing Time",
            f"{latest.get('processing_time', 0):.2f}s",
            delta=calculate_delta(monitor.metrics['processing_time'], inverse=True)
        )
    
    with col4:
        st.metric(
            "Memory Usage",
            f"{latest.get('memory_usage', 0):.0f} MB",
            delta=calculate_delta(monitor.metrics['memory_usage'], inverse=True)
        )
    
    # Data quality score gauge
    st.subheader("Data Quality Score")
    quality_score = calculate_quality_score(latest)
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=quality_score,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Overall Quality"},
        gauge={
            'axis': {'range': [None, 100]},
            'bar': {'color': get_color_for_score(quality_score)},
            'steps': [
                {'range': [0, 50], 'color': "lightgray"},
                {'range': [50, 80], 'color': "gray"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    fig.update_layout(height=300)
    st.plotly_chart(fig, use_container_width=True)


def render_trends(monitor: DataQualityMonitor):
    """Render trend charts."""
    st.header("Trends")
    
    if not monitor.metrics['timestamps']:
        st.info("No trend data available yet.")
        return
    
    # Prepare data
    df = pd.DataFrame({
        'timestamp': list(monitor.metrics['timestamps']),
        'validation_pass_rate': list(monitor.metrics['validation_pass_rate']),
        'feature_count': list(monitor.metrics['feature_count']),
        'processing_time': list(monitor.metrics['processing_time']),
        'memory_usage': list(monitor.metrics['memory_usage'])
    })
    
    # Validation pass rate trend
    fig1 = px.line(df, x='timestamp', y='validation_pass_rate',
                   title='Validation Pass Rate Over Time')
    fig1.add_hline(y=90, line_dash="dash", line_color="red",
                   annotation_text="Target: 90%")
    st.plotly_chart(fig1, use_container_width=True)
    
    # Feature count and memory usage
    col1, col2 = st.columns(2)
    
    with col1:
        fig2 = px.line(df, x='timestamp', y='feature_count',
                       title='Active Features Over Time')
        st.plotly_chart(fig2, use_container_width=True)
    
    with col2:
        fig3 = px.line(df, x='timestamp', y='memory_usage',
                       title='Memory Usage Over Time')
        fig3.add_hline(y=2048, line_dash="dash", line_color="orange",
                       annotation_text="Warning: 2GB")
        st.plotly_chart(fig3, use_container_width=True)


def render_alerts(monitor: DataQualityMonitor):
    """Render alerts section."""
    st.header("🚨 Alerts")
    
    alerts = monitor.get_recent_alerts(20)
    
    if not alerts:
        st.success("No active alerts")
        return
    
    # Alert summary
    high_count = sum(1 for a in alerts if a['severity'] == 'high')
    medium_count = sum(1 for a in alerts if a['severity'] == 'medium')
    low_count = sum(1 for a in alerts if a['severity'] == 'low')
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("🔴 High Severity", high_count)
    with col2:
        st.metric("🟡 Medium Severity", medium_count)
    with col3:
        st.metric("🟢 Low Severity", low_count)
    
    # Alert list
    st.subheader("Recent Alerts")
    
    for alert in reversed(alerts):
        severity_class = f"alert-{alert['severity']}"
        icon = "🔴" if alert['severity'] == 'high' else "🟡" if alert['severity'] == 'medium' else "🟢"
        
        st.markdown(f"""
        <div class="{severity_class}">
            {icon} <strong>{alert['type'].upper()}</strong><br>
            {alert['message']}<br>
            <small>{alert['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}</small>
        </div>
        """, unsafe_allow_html=True)


def render_circuit_breakers(monitor: DataQualityMonitor):
    """Render circuit breaker status."""
    st.header("🔧 Circuit Breakers")
    
    breaker_stats = monitor.breaker.get_all_stats()
    
    if not breaker_stats:
        st.info("No circuit breaker data available")
        return
    
    # Circuit breaker status grid
    cols = st.columns(len(breaker_stats))
    
    for idx, (name, stats) in enumerate(breaker_stats.items()):
        with cols[idx]:
            state = stats.get('current_state', 'unknown')
            color = {
                'closed': 'green',
                'open': 'red',
                'half_open': 'orange'
            }.get(state, 'gray')
            
            st.markdown(f"""
            <div style="background-color: {color}; color: white; padding: 20px; 
                        border-radius: 10px; text-align: center;">
                <h3>{name.upper()}</h3>
                <p>State: {state.upper()}</p>
                <p>Failures: {stats.get('failure_count', 0)}</p>
                <p>Success Rate: {calculate_success_rate(stats):.1f}%</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Show recent state changes
            if stats.get('state_changes'):
                st.caption("Recent State Changes:")
                for change in stats['state_changes'][-3:]:
                    st.caption(f"• {change['from']} → {change['to']}")


def render_reports(monitor: DataQualityMonitor):
    """Render reports section."""
    st.header("📋 Reports")
    
    # Load recent reports
    reports_dir = Path("artifacts/data_quality")
    
    if not reports_dir.exists():
        st.info("No reports available yet")
        return
    
    # List recent reports
    report_files = sorted(reports_dir.glob("pipeline_summary_*.json"), 
                         reverse=True)[:10]
    
    if not report_files:
        st.info("No reports found")
        return
    
    st.subheader("Recent Reports")
    
    selected_report = st.selectbox(
        "Select a report to view",
        report_files,
        format_func=lambda x: x.stem.replace("pipeline_summary_", "")
    )
    
    if selected_report:
        with open(selected_report) as f:
            report_data = json.load(f)
        
        # Display report metrics
        col1, col2 = st.columns(2)
        
        with col1:
            st.json(report_data.get('metrics', {}))
        
        with col2:
            st.json(report_data.get('reports_summary', {}))


# Helper functions
def calculate_delta(values: deque, inverse: bool = False) -> Optional[str]:
    """Calculate delta for metric display."""
    if len(values) < 2:
        return None
    
    current = values[-1]
    previous = values[-2]
    
    if previous == 0:
        return None
    
    delta = ((current - previous) / previous) * 100
    
    if inverse:
        delta = -delta
    
    return f"{delta:+.1f}%"


def calculate_quality_score(metrics: Dict[str, Any]) -> float:
    """Calculate overall data quality score."""
    if not metrics:
        return 0
    
    # Weighted scoring
    scores = {
        'validation': metrics.get('validation_pass_rate', 0) * 0.4,
        'features': min(100, metrics.get('feature_count', 0) / 1.5) * 0.2,
        'processing': max(0, 100 - metrics.get('processing_time', 0) * 10) * 0.2,
        'memory': max(0, 100 - (metrics.get('memory_usage', 0) / 40.96)) * 0.2
    }
    
    return sum(scores.values())


def get_color_for_score(score: float) -> str:
    """Get color based on score."""
    if score >= 90:
        return "green"
    elif score >= 70:
        return "orange"
    else:
        return "red"


def calculate_success_rate(stats: Dict) -> float:
    """Calculate success rate from circuit breaker stats."""
    total = stats.get('total_calls', 0)
    successful = stats.get('successful_calls', 0)
    
    if total == 0:
        return 100.0
    
    return (successful / total) * 100


def load_sample_data(monitor: DataQualityMonitor):
    """Load sample data for demonstration."""
    # Generate sample metrics
    for i in range(50):
        sample_metrics = {
            'validation_pass_rate': 0.85 + np.random.random() * 0.15,
            'filtered_features': int(100 + np.random.randint(-20, 20)),
            'train_samples': int(10000 + np.random.randint(-1000, 1000)),
            'processing_time': 2 + np.random.random() * 3,
            'memory_usage_mb': 500 + np.random.random() * 1500,
            'feature_reduction_pct': 60 + np.random.random() * 30
        }
        
        monitor.update_metrics({'metrics': sample_metrics})
        time.sleep(0.01)  # Small delay to create time series


def export_metrics(monitor: DataQualityMonitor):
    """Export metrics to JSON file."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    export_data = {
        'timestamp': timestamp,
        'metrics': {
            key: list(values) if isinstance(values, deque) else values
            for key, values in monitor.metrics.items()
        },
        'alerts': list(monitor.alerts),
        'drift_metrics': monitor.drift_metrics
    }
    
    # Convert datetime objects to strings
    if 'timestamps' in export_data['metrics']:
        export_data['metrics']['timestamps'] = [
            ts.isoformat() for ts in export_data['metrics']['timestamps']
        ]
    
    export_path = Path(f"data_quality_metrics_{timestamp}.json")
    
    with open(export_path, 'w') as f:
        json.dump(export_data, f, indent=2, default=str)
    
    return export_path


if __name__ == "__main__":
    create_dashboard()