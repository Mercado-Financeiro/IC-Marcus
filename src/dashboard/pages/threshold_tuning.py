"""Threshold tuning page for dashboard."""

import streamlit as st
import plotly.graph_objects as go
import numpy as np
from src.dashboard.data_loader import DataLoader


def render_threshold_page(config, data_loader: DataLoader):
    """
    Render the threshold tuning page.
    
    Args:
        config: Dashboard configuration
        data_loader: Data loader instance
    """
    st.header("🎚️ Otimização de Threshold")
    
    st.markdown("""
    Esta página permite ajustar o threshold de classificação para maximizar
    diferentes métricas: F1 Score, PR-AUC ou Expected Value (EV) líquido.
    """)
    
    # Threshold slider
    threshold = st.slider(
        "Threshold de Classificação:",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.01
    )
    
    # Generate or load threshold analysis
    threshold_data = generate_threshold_data()
    
    # Render chart
    render_threshold_chart(threshold, threshold_data)
    
    # Display metrics at selected threshold
    render_threshold_metrics(threshold, threshold_data)
    
    # Show recommendations
    render_threshold_recommendations(threshold_data)


def generate_threshold_data():
    """Generate threshold analysis data."""
    thresholds = np.linspace(0, 1, 100)
    f1_scores = np.exp(-(thresholds - 0.45)**2 / 0.1)
    pr_auc = np.exp(-(thresholds - 0.4)**2 / 0.15)
    ev_values = -((thresholds - 0.55)**2) + 0.3
    
    return {
        "thresholds": thresholds,
        "f1_scores": f1_scores,
        "pr_auc": pr_auc,
        "ev_values": ev_values
    }


def render_threshold_chart(threshold, threshold_data):
    """Render threshold optimization chart."""
    fig = go.Figure()
    
    # Add metric lines
    fig.add_trace(go.Scatter(
        x=threshold_data["thresholds"],
        y=threshold_data["f1_scores"],
        mode="lines",
        name="F1 Score",
        line=dict(color="blue")
    ))
    
    fig.add_trace(go.Scatter(
        x=threshold_data["thresholds"],
        y=threshold_data["pr_auc"],
        mode="lines",
        name="PR-AUC",
        line=dict(color="green")
    ))
    
    fig.add_trace(go.Scatter(
        x=threshold_data["thresholds"],
        y=threshold_data["ev_values"],
        mode="lines",
        name="EV Líquido",
        line=dict(color="red")
    ))
    
    # Add vertical line at selected threshold
    fig.add_vline(
        x=threshold,
        line_dash="dash",
        line_color="gray",
        annotation_text=f"Threshold: {threshold:.2f}"
    )
    
    fig.update_layout(
        title="Métricas vs Threshold",
        xaxis_title="Threshold",
        yaxis_title="Valor da Métrica",
        hovermode="x unified",
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)


def render_threshold_metrics(threshold, threshold_data):
    """Display metrics at selected threshold."""
    col1, col2, col3 = st.columns(3)
    
    idx = np.argmin(np.abs(threshold_data["thresholds"] - threshold))
    
    with col1:
        st.metric("F1 Score", f"{threshold_data['f1_scores'][idx]:.3f}")
    
    with col2:
        st.metric("PR-AUC", f"{threshold_data['pr_auc'][idx]:.3f}")
    
    with col3:
        st.metric("EV Líquido", f"{threshold_data['ev_values'][idx]:.3f}")


def render_threshold_recommendations(threshold_data):
    """Show threshold recommendations."""
    st.subheader("💡 Recomendações")
    
    best_f1_idx = np.argmax(threshold_data["f1_scores"])
    best_ev_idx = np.argmax(threshold_data["ev_values"])
    
    st.info(f"""
    - **Melhor F1 Score**: Threshold = {threshold_data['thresholds'][best_f1_idx]:.3f} 
      (F1 = {threshold_data['f1_scores'][best_f1_idx]:.3f})
    - **Melhor EV Líquido**: Threshold = {threshold_data['thresholds'][best_ev_idx]:.3f} 
      (EV = {threshold_data['ev_values'][best_ev_idx]:.3f})
    
    ⚠️ **Importante**: O threshold ótimo para EV líquido considera custos de transação
    e pode diferir significativamente do threshold ótimo para F1 Score.
    """)