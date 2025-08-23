"""Configuration for the dashboard application."""

from pathlib import Path
from dataclasses import dataclass


@dataclass
class DashboardConfig:
    """Dashboard configuration settings."""
    
    # Page settings
    page_title: str = "Crypto ML Dashboard"
    page_icon: str = "📊"
    layout: str = "wide"
    
    # MLflow settings
    mlflow_tracking_uri: str = "../../artifacts/mlruns"
    experiment_name: str = "crypto_ml_pipeline"
    
    # Display settings
    max_runs_display: int = 20
    date_range_days: int = 30
    
    # Chart settings
    chart_height_default: int = 400
    chart_height_large: int = 600
    
    # Metrics thresholds
    good_sharpe_threshold: float = 1.5
    warning_drawdown_threshold: float = -0.15
    good_win_rate_threshold: float = 0.55
    
    # Navigation pages
    pages = [
        "Visão Geral",
        "MLflow Runs",
        "Backtest",
        "Threshold Tuning",
        "Feature Importance",
        "Comparação de Modelos",
        "Análise de Regime"
    ]
    
    # Paths
    @property
    def artifacts_path(self) -> Path:
        return Path("../../artifacts")
    
    @property
    def mlruns_path(self) -> Path:
        return self.artifacts_path / "mlruns"