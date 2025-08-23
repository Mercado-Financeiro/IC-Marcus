"""Backtest results page for dashboard."""

import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from src.dashboard.data_loader import DataLoader


def render_backtest_page(config, data_loader: DataLoader):
    """
    Render the backtest results page.
    
    Args:
        config: Dashboard configuration
        data_loader: Data loader instance
    """
    st.header("💰 Resultados de Backtest")
    
    # Select run
    runs_df = data_loader.load_mlflow_runs()
    
    if not runs_df.empty:
        selected_run = st.selectbox(
            "Selecione uma run:",
            runs_df["run_id"].head(config.max_runs_display).tolist()
        )
        
        if selected_run:
            render_backtest_results(selected_run, data_loader)
    else:
        st.warning("⚠️ Nenhuma run disponível")


def render_backtest_results(run_id: str, data_loader: DataLoader):
    """Render backtest results for a specific run."""
    backtest_data = data_loader.load_backtest_results(run_id)
    
    if backtest_data:
        # Display metrics
        render_backtest_metrics(backtest_data)
        
        # Display charts
        render_backtest_charts(backtest_data)
    else:
        st.info("ℹ️ Dados de backtest não disponíveis para esta run")


def render_backtest_metrics(backtest_data):
    """Render backtest metrics."""
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Total Return", 
            f"{backtest_data.get('total_return', 0):.2%}"
        )
    
    with col2:
        st.metric(
            "Sharpe Ratio", 
            f"{backtest_data.get('sharpe_ratio', 0):.2f}"
        )
    
    with col3:
        st.metric(
            "Max Drawdown", 
            f"{backtest_data.get('max_drawdown', 0):.2%}"
        )


def render_backtest_charts(backtest_data):
    """Render backtest charts."""
    if "equity" not in backtest_data:
        return
    
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=("Equity Curve", "Drawdown"),
        row_heights=[0.7, 0.3]
    )
    
    # Equity curve
    fig.add_trace(
        go.Scatter(
            y=backtest_data["equity"],
            mode="lines",
            name="Equity",
            line=dict(color="blue")
        ),
        row=1, col=1
    )
    
    # Calculate drawdown
    returns = pd.Series(backtest_data["equity"]).pct_change()
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max * 100
    
    # Drawdown chart
    fig.add_trace(
        go.Scatter(
            y=drawdown,
            mode="lines",
            name="Drawdown",
            fill="tozeroy",
            line=dict(color="red")
        ),
        row=2, col=1
    )
    
    fig.update_layout(height=600, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)