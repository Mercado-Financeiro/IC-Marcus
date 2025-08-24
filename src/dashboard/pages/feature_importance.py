"""Feature importance page for dashboard."""

import streamlit as st
import plotly.express as px
from src.dashboard.data_loader import DataLoader


def render_feature_importance_page(config, data_loader: DataLoader):
    """
    Render the feature importance page.
    
    Args:
        config: Dashboard configuration
        data_loader: Data loader instance
    """
    st.header("🔍 Importância das Features")
    
    # Select MLflow run
    runs_df = data_loader.load_mlflow_runs()
    importance_df = None
    selected_run = None
    if runs_df is not None and not runs_df.empty:
        st.subheader("Selecionar Run do MLflow")
        # Build options mapping label -> run_id
        def _label(row):
            name = row.get('tags.mlflow.runName', '') or row.get('run_id', '')
            start = row.get('start_time')
            return f"{name} | {row['run_id'][:8]}"
        options = { _label(row): row['run_id'] for _, row in runs_df.head(50).iterrows() }
        choice = st.selectbox("Run:", list(options.keys()))
        selected_run = options.get(choice)
        if selected_run:
            st.caption(f"Run selecionada: {selected_run}")
            importance_df = data_loader.load_feature_importance(selected_run)

    if importance_df is None:
        st.info("Sem artifact de importância encontrado. Exibindo dados sintéticos para demonstração.")
        importance_df = data_loader.generate_synthetic_data("feature_importance")
    
    # Render bar chart
    render_importance_chart(importance_df)
    
    # Render detailed table
    render_importance_table(importance_df)
    
    # SHAP values section
    render_shap_section()


def render_importance_chart(importance_df):
    """Render feature importance bar chart."""
    # Sort and limit to top 15
    top_features = importance_df.head(15).sort_values("importance", ascending=True)
    
    fig = px.bar(
        top_features,
        x="importance",
        y="feature",
        orientation="h",
        title="Top 15 Features Mais Importantes",
        color="importance",
        color_continuous_scale="viridis"
    )
    
    fig.update_layout(height=600)
    st.plotly_chart(fig, use_container_width=True)


def render_importance_table(importance_df):
    """Render feature importance table."""
    st.subheader("📋 Valores Detalhados")
    
    # Format importance values
    display_df = importance_df.copy()
    display_df["importance"] = display_df["importance"].apply(lambda x: f"{x:.4f}")
    
    st.dataframe(
        display_df,
        use_container_width=True
    )


def render_shap_section():
    """Render SHAP values section."""
    st.subheader("🎨 SHAP Values")
    
    st.info("""
    SHAP (SHapley Additive exPlanations) fornece uma explicação unificada
    da saída do modelo, mostrando como cada feature contribui para a predição.
    """)
    
    # Placeholder for SHAP visualization
    with st.expander("Ver análise SHAP detalhada"):
        st.markdown("""
        ### Interpretação dos SHAP Values
        
        - **Valores positivos**: Feature aumenta a probabilidade de classificação positiva
        - **Valores negativos**: Feature diminui a probabilidade de classificação positiva
        - **Magnitude**: Indica o impacto da feature na predição
        
        ### Features com maior impacto médio:
        1. **volatility_20**: Alto impacto em condições de mercado volátil
        2. **rsi_14**: Importante para detectar condições de sobrecompra/sobrevenda
        3. **macd_diff**: Crucial para identificar mudanças de tendência
        """)
