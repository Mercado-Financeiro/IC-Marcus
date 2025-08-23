"""MLflow runs page for dashboard."""

import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from src.dashboard.data_loader import DataLoader


def render_mlflow_page(config, data_loader: DataLoader):
    """
    Render the MLflow runs page.
    
    Args:
        config: Dashboard configuration
        data_loader: Data loader instance
    """
    st.header("🎯 MLflow Experiment Runs")
    
    runs_df = data_loader.load_mlflow_runs()
    
    if not runs_df.empty:
        # Filters
        filtered_runs = apply_filters(runs_df)
        
        # Runs table
        render_runs_table(filtered_runs, config)
        
        # Metrics evolution
        render_metrics_evolution(filtered_runs)
    else:
        st.warning("⚠️ Nenhuma run encontrada no MLflow")


def apply_filters(runs_df):
    """Apply filters to runs dataframe."""
    col1, col2 = st.columns(2)
    
    with col1:
        model_types = ["Todos"]
        if "tags.model_type" in runs_df.columns:
            model_types.extend(runs_df["tags.model_type"].dropna().unique())
        
        selected_model = st.selectbox(
            "Filtrar por modelo:",
            model_types
        )
    
    with col2:
        date_range = st.date_input(
            "Período:",
            value=(datetime.now() - timedelta(days=30), datetime.now()),
            max_value=datetime.now()
        )
    
    # Apply filters
    filtered_runs = runs_df.copy()
    
    if selected_model != "Todos" and "tags.model_type" in filtered_runs.columns:
        filtered_runs = filtered_runs[
            filtered_runs["tags.model_type"] == selected_model
        ]
    
    return filtered_runs


def render_runs_table(filtered_runs, config):
    """Render runs table."""
    st.subheader("📃 Runs Recentes")
    
    display_cols = [
        "run_id", "tags.model_type", "metrics.f1_score",
        "metrics.pr_auc", "metrics.sharpe_ratio", "start_time"
    ]
    
    display_cols = [col for col in display_cols if col in filtered_runs.columns]
    
    if display_cols:
        st.dataframe(
            filtered_runs[display_cols].head(config.max_runs_display),
            use_container_width=True
        )


def render_metrics_evolution(filtered_runs):
    """Render metrics evolution charts."""
    metrics_cols = [
        ("metrics.f1_score", "F1 Score"),
        ("metrics.pr_auc", "PR-AUC"),
        ("metrics.sharpe_ratio", "Sharpe Ratio"),
        ("metrics.brier_score", "Brier Score")
    ]
    
    # Check which metrics are available
    available_metrics = [
        (col, name) for col, name in metrics_cols 
        if col in filtered_runs.columns
    ]
    
    if available_metrics:
        st.subheader("📊 Evolução das Métricas")
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[name for _, name in available_metrics[:4]]
        )
        
        positions = [(1, 1), (1, 2), (2, 1), (2, 2)]
        
        for (col, name), (row, col_pos) in zip(available_metrics[:4], positions):
            fig.add_trace(
                go.Scatter(
                    x=filtered_runs["start_time"],
                    y=filtered_runs[col],
                    mode="lines+markers",
                    name=name
                ),
                row=row, col=col_pos
            )
        
        fig.update_layout(height=600, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)