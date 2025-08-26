"""Model comparison page for dashboard."""

import streamlit as st
import plotly.graph_objects as go
from src.dashboard.data_loader import DataLoader


def render_model_comparison_page(config, data_loader: DataLoader):
    """
    Render the model comparison page.
    
    Args:
        config: Dashboard configuration
        data_loader: Data loader instance
    """
    st.header("⚖️ Comparação de Modelos")
    
    # Load comparison data
    models_df = data_loader.generate_synthetic_data("model_comparison")
    
    # Render comparison table
    render_comparison_table(models_df)
    
    # Render radar chart
    render_radar_chart(models_df)
    
    # Show recommendations
    render_model_recommendations(models_df, config)


def render_comparison_table(models_df):
    """Render model comparison table."""
    st.subheader("📋 Tabela Comparativa")
    
    # Format numeric columns
    display_df = models_df.copy()
    display_df["f1_score"] = display_df["f1_score"].apply(lambda x: f"{x:.3f}")
    display_df["pr_auc"] = display_df["pr_auc"].apply(lambda x: f"{x:.3f}")
    display_df["sharpe_ratio"] = display_df["sharpe_ratio"].apply(lambda x: f"{x:.2f}")
    display_df["max_drawdown"] = display_df["max_drawdown"].apply(lambda x: f"{x:.1%}")
    display_df["win_rate"] = display_df["win_rate"].apply(lambda x: f"{x:.1%}")
    
    st.dataframe(display_df, use_container_width=True)


def render_radar_chart(models_df):
    """Render radar chart for model comparison."""
    st.subheader("🕸️ Comparação Visual")
    
    categories = ["f1_score", "pr_auc", "sharpe_ratio", "win_rate"]
    
    fig = go.Figure()
    
    for _, row in models_df.iterrows():
        # Normalize win_rate to 0-1 scale if needed
        values = [row[cat] if cat != "win_rate" else row[cat] for cat in categories]
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=["F1 Score", "PR-AUC", "Sharpe Ratio", "Win Rate"],
            fill="toself",
            name=row["model"]
        ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1] if max(models_df["sharpe_ratio"]) <= 1 else [0, 2.5]
            )
        ),
        showlegend=True,
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)


def render_model_recommendations(models_df, config):
    """Show model recommendations."""
    st.subheader("🏆 Modelo Recomendado")
    
    # Find best model based on Sharpe ratio
    best_model = models_df.loc[models_df["sharpe_ratio"].idxmax()]
    
    # Check if meets thresholds
    if best_model["sharpe_ratio"] >= config.good_sharpe_threshold:
        status = "success"
    else:
        status = "warning"
    
    if status == "success":
        st.success(f"""
        **{best_model['model']}** apresenta o melhor desempenho geral com:
        - Maior Sharpe Ratio ({best_model['sharpe_ratio']:.2f})
        - F1 Score: {best_model['f1_score']:.2f}
        - Max Drawdown: {best_model['max_drawdown']:.1%}
        
        Recomenda-se usar {best_model['model']} como modelo principal com calibração isotônica.
        """)
    else:
        st.warning(f"""
        **{best_model['model']}** é o melhor modelo disponível, mas com ressalvas:
        - Sharpe Ratio: {best_model['sharpe_ratio']:.2f} (abaixo do ideal)
        - Considere ensemble ou otimização adicional
        - Recomenda-se backtesting extensivo antes de produção
        """)