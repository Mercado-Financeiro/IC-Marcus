"""Overview page for dashboard."""

import streamlit as st
import plotly.graph_objects as go
import pandas as pd
from src.dashboard.data_loader import DataLoader


def render_overview_page(config, data_loader: DataLoader):
    """
    Render the overview page.
    
    Args:
        config: Dashboard configuration
        data_loader: Data loader instance
    """
    st.header("📈 Visão Geral do Sistema")
    
    # Key metrics
    render_key_metrics()
    
    # Equity curve
    render_equity_curve(data_loader)
    
    # Detailed statistics
    render_statistics_table()


def render_key_metrics():
    """Render key performance metrics."""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Sharpe Ratio",
            value="1.85",
            delta="+0.12"
        )
    
    with col2:
        st.metric(
            label="Total Return",
            value="45.2%",
            delta="+5.3%"
        )
    
    with col3:
        st.metric(
            label="Max Drawdown",
            value="-12.3%",
            delta="-2.1%"
        )
    
    with col4:
        st.metric(
            label="Win Rate",
            value="58.7%",
            delta="+3.2%"
        )


def render_equity_curve(data_loader: DataLoader):
    """Render equity curve chart."""
    st.subheader("💰 Equity Curve")
    
    # Load or generate data
    equity_data = data_loader.generate_synthetic_data("equity_curve")
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=equity_data["date"],
        y=equity_data["equity"],
        mode="lines",
        name="Equity",
        line=dict(color="blue", width=2)
    ))
    
    fig.update_layout(
        title="Evolução do Capital",
        xaxis_title="Data",
        yaxis_title="Equity ($)",
        hovermode="x unified",
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)


def render_statistics_table():
    """Render detailed statistics table."""
    st.subheader("📄 Estatísticas Detalhadas")
    
    stats_data = {
        "Métrica": [
            "Retorno Total", "Retorno Anualizado", "Volatilidade",
            "Sharpe Ratio", "Sortino Ratio", "Calmar Ratio",
            "Max Drawdown", "Duração Max DD", "Recovery Time",
            "Win Rate", "Profit Factor", "Expectancy"
        ],
        "Valor": [
            "45.2%", "38.7%", "18.3%",
            "1.85", "2.31", "3.15",
            "-12.3%", "45 dias", "23 dias",
            "58.7%", "1.67", "$125"
        ]
    }
    
    stats_df = pd.DataFrame(stats_data)
    st.dataframe(stats_df, use_container_width=True)