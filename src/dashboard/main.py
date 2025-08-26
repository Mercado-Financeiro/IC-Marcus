"""Main dashboard application."""

import streamlit as st
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.dashboard.config import DashboardConfig
from src.dashboard.data_loader import DataLoader
from src.dashboard.pages import (
    render_overview_page,
    render_mlflow_page,
    render_backtest_page,
    render_threshold_page,
    render_feature_importance_page,
    render_model_comparison_page,
    render_regime_analysis_page
)


def initialize_dashboard():
    """Initialize dashboard configuration and state."""
    config = DashboardConfig()
    
    # Configure page
    st.set_page_config(
        page_title=config.page_title,
        page_icon=config.page_icon,
        layout=config.layout,
        initial_sidebar_state="expanded"
    )
    
    return config


def render_sidebar(config):
    """Render sidebar navigation."""
    st.sidebar.title("📋 Navegação")
    
    page = st.sidebar.radio(
        "Selecione a página:",
        config.pages
    )
    
    # Add additional sidebar info
    st.sidebar.markdown("---")
    st.sidebar.info("""
    **Dashboard v2.0.0**
    
    🔒 Execução t+1  
    🎯 Calibração Obrigatória  
    📊 Zero Vazamento Temporal
    """)
    
    return page


def render_header():
    """Render main header."""
    st.title("🚀 Crypto ML Trading Dashboard")
    st.markdown("---")


def render_footer():
    """Render footer."""
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center'>
            <p>Crypto ML Trading Dashboard v2.0.0 | 
            🔒 Execução t+1 | 
            🎯 Calibração Obrigatória | 
            📊 Zero Vazamento Temporal</p>
        </div>
        """,
        unsafe_allow_html=True
    )


def main():
    """Main dashboard application."""
    # Initialize
    config = initialize_dashboard()
    data_loader = DataLoader(config)
    
    # Render header
    render_header()
    
    # Get selected page
    page = render_sidebar(config)
    
    # Route to appropriate page
    page_routes = {
        "Visão Geral": render_overview_page,
        "MLflow Runs": render_mlflow_page,
        "Backtest": render_backtest_page,
        "Threshold Tuning": render_threshold_page,
        "Feature Importance": render_feature_importance_page,
        "Comparação de Modelos": render_model_comparison_page,
        "Análise de Regime": render_regime_analysis_page
    }
    
    # Render selected page
    if page in page_routes:
        page_routes[page](config, data_loader)
    
    # Render footer
    render_footer()


if __name__ == "__main__":
    main()