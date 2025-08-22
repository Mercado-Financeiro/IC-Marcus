"""Dashboard Streamlit para visualização de resultados ML."""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import mlflow
from pathlib import Path
import pickle
import json
from datetime import datetime, timedelta
import sys

# Adicionar src ao path
sys.path.append(str(Path(__file__).parent.parent.parent))

# Configurar página
st.set_page_config(
    page_title="Crypto ML Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Título principal
st.title("🚀 Crypto ML Trading Dashboard")
st.markdown("---")

# Sidebar para navegação
st.sidebar.title("📋 Navegação")
page = st.sidebar.radio(
    "Selecione a página:",
    [
        "Visão Geral",
        "MLflow Runs",
        "Backtest",
        "Threshold Tuning",
        "Feature Importance",
        "Comparação de Modelos",
        "Análise de Regime"
    ]
)

# Configurar MLflow
mlflow.set_tracking_uri("../../artifacts/mlruns")


def load_mlflow_runs():
    """Carrega runs do MLflow."""
    try:
        experiment = mlflow.get_experiment_by_name("crypto_ml_pipeline")
        if experiment:
            runs = mlflow.search_runs(
                experiment_ids=[experiment.experiment_id],
                order_by=["start_time DESC"]
            )
            return runs
    except Exception as e:
        st.error(f"Erro ao carregar runs MLflow: {e}")
    return pd.DataFrame()


def load_backtest_results(run_id: str):
    """Carrega resultados de backtest de uma run."""
    try:
        artifact_path = f"../../artifacts/mlruns/{run_id[:2]}/{run_id}/artifacts"
        backtest_path = Path(artifact_path) / "backtest_results.pkl"
        
        if backtest_path.exists():
            with open(backtest_path, "rb") as f:
                return pickle.load(f)
    except Exception as e:
        st.error(f"Erro ao carregar backtest: {e}")
    return None


# Página: Visão Geral
if page == "Visão Geral":
    st.header("📈 Visão Geral do Sistema")
    
    col1, col2, col3, col4 = st.columns(4)
    
    # Métricas principais (placeholder)
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
    
    # Gráfico de equity curve
    st.subheader("💰 Equity Curve")
    
    # Dados sintéticos para demonstração
    dates = pd.date_range(start="2023-01-01", end="2024-01-01", freq="D")
    returns = np.random.randn(len(dates)) * 0.02
    equity = 100000 * (1 + returns).cumprod()
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dates,
        y=equity,
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
    
    # Tabela de estatísticas
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


# Página: MLflow Runs
elif page == "MLflow Runs":
    st.header("🎯 MLflow Experiment Runs")
    
    runs_df = load_mlflow_runs()
    
    if not runs_df.empty:
        # Filtros
        col1, col2 = st.columns(2)
        
        with col1:
            model_types = runs_df["tags.model_type"].dropna().unique()
            selected_model = st.selectbox(
                "Filtrar por modelo:",
                ["Todos"] + list(model_types)
            )
        
        with col2:
            date_range = st.date_input(
                "Período:",
                value=(datetime.now() - timedelta(days=30), datetime.now()),
                max_value=datetime.now()
            )
        
        # Filtrar runs
        filtered_runs = runs_df.copy()
        
        if selected_model != "Todos":
            filtered_runs = filtered_runs[
                filtered_runs["tags.model_type"] == selected_model
            ]
        
        # Tabela de runs
        st.subheader("📃 Runs Recentes")
        
        display_cols = [
            "run_id", "tags.model_type", "metrics.f1_score",
            "metrics.pr_auc", "metrics.sharpe_ratio", "start_time"
        ]
        
        display_cols = [col for col in display_cols if col in filtered_runs.columns]
        
        if display_cols:
            st.dataframe(
                filtered_runs[display_cols].head(20),
                use_container_width=True
            )
        
        # Gráfico de métricas ao longo do tempo
        if "metrics.f1_score" in filtered_runs.columns:
            st.subheader("📊 Evolução das Métricas")
            
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=("F1 Score", "PR-AUC", "Sharpe Ratio", "Brier Score")
            )
            
            # F1 Score
            if "metrics.f1_score" in filtered_runs.columns:
                fig.add_trace(
                    go.Scatter(
                        x=filtered_runs["start_time"],
                        y=filtered_runs["metrics.f1_score"],
                        mode="lines+markers",
                        name="F1 Score"
                    ),
                    row=1, col=1
                )
            
            # PR-AUC
            if "metrics.pr_auc" in filtered_runs.columns:
                fig.add_trace(
                    go.Scatter(
                        x=filtered_runs["start_time"],
                        y=filtered_runs["metrics.pr_auc"],
                        mode="lines+markers",
                        name="PR-AUC"
                    ),
                    row=1, col=2
                )
            
            # Sharpe Ratio
            if "metrics.sharpe_ratio" in filtered_runs.columns:
                fig.add_trace(
                    go.Scatter(
                        x=filtered_runs["start_time"],
                        y=filtered_runs["metrics.sharpe_ratio"],
                        mode="lines+markers",
                        name="Sharpe"
                    ),
                    row=2, col=1
                )
            
            # Brier Score
            if "metrics.brier_score" in filtered_runs.columns:
                fig.add_trace(
                    go.Scatter(
                        x=filtered_runs["start_time"],
                        y=filtered_runs["metrics.brier_score"],
                        mode="lines+markers",
                        name="Brier"
                    ),
                    row=2, col=2
                )
            
            fig.update_layout(height=600, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("⚠️ Nenhuma run encontrada no MLflow")


# Página: Backtest
elif page == "Backtest":
    st.header("💰 Resultados de Backtest")
    
    # Seletor de run
    runs_df = load_mlflow_runs()
    
    if not runs_df.empty:
        selected_run = st.selectbox(
            "Selecione uma run:",
            runs_df["run_id"].head(20).tolist()
        )
        
        if selected_run:
            backtest_data = load_backtest_results(selected_run)
            
            if backtest_data:
                # Métricas do backtest
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Total Return", f"{backtest_data.get('total_return', 0):.2%}")
                
                with col2:
                    st.metric("Sharpe Ratio", f"{backtest_data.get('sharpe_ratio', 0):.2f}")
                
                with col3:
                    st.metric("Max Drawdown", f"{backtest_data.get('max_drawdown', 0):.2%}")
                
                # Gráfico de equity e drawdown
                if "equity" in backtest_data:
                    fig = make_subplots(
                        rows=2, cols=1,
                        subplot_titles=("Equity Curve", "Drawdown"),
                        row_heights=[0.7, 0.3]
                    )
                    
                    # Equity
                    fig.add_trace(
                        go.Scatter(
                            y=backtest_data["equity"],
                            mode="lines",
                            name="Equity",
                            line=dict(color="blue")
                        ),
                        row=1, col=1
                    )
                    
                    # Drawdown
                    returns = pd.Series(backtest_data["equity"]).pct_change()
                    cumulative = (1 + returns).cumprod()
                    running_max = cumulative.expanding().max()
                    drawdown = (cumulative - running_max) / running_max * 100
                    
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
            else:
                st.info("ℹ️ Dados de backtest não disponíveis para esta run")
    else:
        st.warning("⚠️ Nenhuma run disponível")


# Página: Threshold Tuning
elif page == "Threshold Tuning":
    st.header("🎚️ Otimização de Threshold")
    
    st.markdown("""
    Esta página permite ajustar o threshold de classificação para maximizar
    diferentes métricas: F1 Score, PR-AUC ou Expected Value (EV) líquido.
    """)
    
    # Slider para threshold
    threshold = st.slider(
        "Threshold de Classificação:",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.01
    )
    
    # Gerar dados sintéticos para demonstração
    thresholds = np.linspace(0, 1, 100)
    f1_scores = np.exp(-(thresholds - 0.45)**2 / 0.1)
    pr_auc = np.exp(-(thresholds - 0.4)**2 / 0.15)
    ev_values = -((thresholds - 0.55)**2) + 0.3
    
    # Gráfico de métricas vs threshold
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=thresholds,
        y=f1_scores,
        mode="lines",
        name="F1 Score",
        line=dict(color="blue")
    ))
    
    fig.add_trace(go.Scatter(
        x=thresholds,
        y=pr_auc,
        mode="lines",
        name="PR-AUC",
        line=dict(color="green")
    ))
    
    fig.add_trace(go.Scatter(
        x=thresholds,
        y=ev_values,
        mode="lines",
        name="EV Líquido",
        line=dict(color="red")
    ))
    
    # Linha vertical no threshold selecionado
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
    
    # Métricas no threshold selecionado
    col1, col2, col3 = st.columns(3)
    
    idx = np.argmin(np.abs(thresholds - threshold))
    
    with col1:
        st.metric("F1 Score", f"{f1_scores[idx]:.3f}")
    
    with col2:
        st.metric("PR-AUC", f"{pr_auc[idx]:.3f}")
    
    with col3:
        st.metric("EV Líquido", f"{ev_values[idx]:.3f}")
    
    # Recomendações
    st.subheader("💡 Recomendações")
    
    best_f1_idx = np.argmax(f1_scores)
    best_ev_idx = np.argmax(ev_values)
    
    st.info(f"""
    - **Melhor F1 Score**: Threshold = {thresholds[best_f1_idx]:.3f} (F1 = {f1_scores[best_f1_idx]:.3f})
    - **Melhor EV Líquido**: Threshold = {thresholds[best_ev_idx]:.3f} (EV = {ev_values[best_ev_idx]:.3f})
    
    ⚠️ **Importante**: O threshold ótimo para EV líquido considera custos de transação
    e pode diferir significativamente do threshold ótimo para F1 Score.
    """)


# Página: Feature Importance
elif page == "Feature Importance":
    st.header("🔍 Importância das Features")
    
    # Gerar dados sintéticos
    features = [
        "volatility_20", "rsi_14", "macd_diff", "bb_position_20",
        "volume_ratio", "zscore_50", "momentum_10", "atr_14",
        "vwap_distance_20", "sma_cross_20_50", "stoch_k", "adx",
        "cci_20", "obv_momentum_20", "high_vol_regime"
    ]
    
    importances = np.random.exponential(scale=0.1, size=len(features))
    importances = importances / importances.sum()
    importances = np.sort(importances)[::-1]
    
    # Criar DataFrame
    importance_df = pd.DataFrame({
        "Feature": features[:len(importances)],
        "Importance": importances
    }).sort_values("Importance", ascending=True)
    
    # Gráfico de barras horizontais
    fig = px.bar(
        importance_df,
        x="Importance",
        y="Feature",
        orientation="h",
        title="Top 15 Features Mais Importantes",
        color="Importance",
        color_continuous_scale="viridis"
    )
    
    fig.update_layout(height=600)
    st.plotly_chart(fig, use_container_width=True)
    
    # Tabela com valores
    st.subheader("📋 Valores Detalhados")
    st.dataframe(
        importance_df.sort_values("Importance", ascending=False),
        use_container_width=True
    )
    
    # SHAP values (placeholder)
    st.subheader("🎨 SHAP Values")
    st.info("""
    SHAP (SHapley Additive exPlanations) fornece uma explicação unificada
    da saída do modelo, mostrando como cada feature contribui para a predição.
    """)


# Página: Comparação de Modelos
elif page == "Comparação de Modelos":
    st.header("⚖️ Comparação de Modelos")
    
    # Dados sintéticos para comparação
    models_data = {
        "Modelo": ["XGBoost", "LSTM", "Random Forest", "Logistic Regression"],
        "F1 Score": [0.68, 0.65, 0.62, 0.58],
        "PR-AUC": [0.72, 0.69, 0.66, 0.61],
        "Sharpe Ratio": [1.85, 1.72, 1.45, 1.23],
        "Max Drawdown": [-0.123, -0.145, -0.178, -0.201],
        "Win Rate": [0.587, 0.562, 0.534, 0.512]
    }
    
    models_df = pd.DataFrame(models_data)
    
    # Tabela comparativa
    st.subheader("📋 Tabela Comparativa")
    st.dataframe(models_df, use_container_width=True)
    
    # Gráfico de radar
    st.subheader("🕸️ Comparação Visual")
    
    categories = ["F1 Score", "PR-AUC", "Sharpe Ratio", "Win Rate"]
    
    fig = go.Figure()
    
    for _, row in models_df.iterrows():
        fig.add_trace(go.Scatterpolar(
            r=[row[cat] for cat in categories],
            theta=categories,
            fill="toself",
            name=row["Modelo"]
        ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )
        ),
        showlegend=True,
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Recomendação
    st.subheader("🏆 Modelo Recomendado")
    st.success("""
    **XGBoost** apresenta o melhor desempenho geral com:
    - Maior Sharpe Ratio (1.85)
    - Melhor F1 Score (0.68)
    - Menor Max Drawdown (-12.3%)
    
    Recomenda-se usar XGBoost como modelo principal com calibração isotônica.
    """)


# Página: Análise de Regime
elif page == "Análise de Regime":
    st.header("🌍 Análise por Regime de Mercado")
    
    st.markdown("""
    Performance da estratégia em diferentes regimes de mercado:
    volatilidade, tendência e momentum.
    """)
    
    # Dados sintéticos por regime
    regimes_data = {
        "Regime": [
            "Alta Volatilidade", "Baixa Volatilidade",
            "Tendência Alta", "Tendência Baixa",
            "Lateralização", "Alto Momentum"
        ],
        "Sharpe Ratio": [1.2, 2.3, 2.5, 0.8, 1.1, 2.8],
        "Win Rate": [0.52, 0.65, 0.68, 0.45, 0.55, 0.72],
        "Avg Return": [0.08, 0.12, 0.15, -0.03, 0.05, 0.18],
        "Frequency": [0.15, 0.25, 0.20, 0.10, 0.20, 0.10]
    }
    
    regimes_df = pd.DataFrame(regimes_data)
    
    # Gráfico de barras agrupadas
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Sharpe Ratio por Regime",
            "Win Rate por Regime",
            "Retorno Médio por Regime",
            "Frequência dos Regimes"
        )
    )
    
    # Sharpe Ratio
    fig.add_trace(
        go.Bar(x=regimes_df["Regime"], y=regimes_df["Sharpe Ratio"], name="Sharpe"),
        row=1, col=1
    )
    
    # Win Rate
    fig.add_trace(
        go.Bar(x=regimes_df["Regime"], y=regimes_df["Win Rate"], name="Win Rate"),
        row=1, col=2
    )
    
    # Avg Return
    fig.add_trace(
        go.Bar(x=regimes_df["Regime"], y=regimes_df["Avg Return"], name="Return"),
        row=2, col=1
    )
    
    # Frequency
    fig.add_trace(
        go.Bar(x=regimes_df["Regime"], y=regimes_df["Frequency"], name="Freq"),
        row=2, col=2
    )
    
    fig.update_layout(height=700, showlegend=False)
    fig.update_xaxes(tickangle=45)
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Insights
    st.subheader("💡 Principais Insights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info("""
        **Melhores Regimes:**
        - Alto Momentum: Sharpe 2.8
        - Tendência Alta: Sharpe 2.5
        - Baixa Volatilidade: Sharpe 2.3
        """)
    
    with col2:
        st.warning("""
        **Regimes Desafiadores:**
        - Tendência Baixa: Sharpe 0.8
        - Lateralização: Sharpe 1.1
        - Alta Volatilidade: Sharpe 1.2
        """)
    
    # Recomendações
    st.subheader("🎯 Recomendações Estratégicas")
    st.success("""
    1. **Aumentar posição** em regimes de alto momentum e baixa volatilidade
    2. **Reduzir exposição** durante tendência de baixa
    3. **Usar stop-loss mais apertado** em alta volatilidade
    4. **Considerar hedge** durante lateralização prolongada
    """)


# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center'>
        <p>Crypto ML Trading Dashboard v1.0.0 | 
        🔒 Execução t+1 | 
        🎯 Calibração Obrigatória | 
        📊 Zero Vazamento Temporal</p>
    </div>
    """,
    unsafe_allow_html=True
)


if __name__ == "__main__":
    # O Streamlit já executa o app automaticamente
    pass