"""Regime analysis page for dashboard."""

import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from src.dashboard.data_loader import DataLoader


def render_regime_analysis_page(config, data_loader: DataLoader):
    """
    Render the regime analysis page.
    
    Args:
        config: Dashboard configuration
        data_loader: Data loader instance
    """
    st.header("🌍 Análise por Regime de Mercado")
    
    st.markdown("""
    Performance da estratégia em diferentes regimes de mercado:
    volatilidade, tendência e momentum.
    """)
    
    # Load regime data
    regimes_df = data_loader.generate_synthetic_data("regime_analysis")
    
    # Render regime charts
    render_regime_charts(regimes_df)
    
    # Show insights
    render_regime_insights(regimes_df, config)
    
    # Strategic recommendations
    render_strategic_recommendations()


def render_regime_charts(regimes_df):
    """Render regime analysis charts."""
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
        go.Bar(
            x=regimes_df["regime"],
            y=regimes_df["sharpe_ratio"],
            name="Sharpe",
            marker_color="blue"
        ),
        row=1, col=1
    )
    
    # Win Rate
    fig.add_trace(
        go.Bar(
            x=regimes_df["regime"],
            y=regimes_df["win_rate"],
            name="Win Rate",
            marker_color="green"
        ),
        row=1, col=2
    )
    
    # Average Return
    colors = ['red' if x < 0 else 'lightgreen' for x in regimes_df["avg_return"]]
    fig.add_trace(
        go.Bar(
            x=regimes_df["regime"],
            y=regimes_df["avg_return"],
            name="Return",
            marker_color=colors
        ),
        row=2, col=1
    )
    
    # Frequency
    fig.add_trace(
        go.Bar(
            x=regimes_df["regime"],
            y=regimes_df["frequency"],
            name="Freq",
            marker_color="purple"
        ),
        row=2, col=2
    )
    
    fig.update_layout(height=700, showlegend=False)
    fig.update_xaxes(tickangle=45)
    
    st.plotly_chart(fig, use_container_width=True)


def render_regime_insights(regimes_df, config):
    """Show regime analysis insights."""
    st.subheader("💡 Principais Insights")
    
    col1, col2 = st.columns(2)
    
    # Find best and worst regimes
    best_regimes = regimes_df.nlargest(3, "sharpe_ratio")
    worst_regimes = regimes_df.nsmallest(3, "sharpe_ratio")
    
    with col1:
        st.info(f"""
        **Melhores Regimes:**
        {format_regime_list(best_regimes)}
        """)
    
    with col2:
        st.warning(f"""
        **Regimes Desafiadores:**
        {format_regime_list(worst_regimes)}
        """)


def format_regime_list(regimes_df):
    """Format regime list for display."""
    lines = []
    for _, row in regimes_df.iterrows():
        lines.append(f"- {row['regime']}: Sharpe {row['sharpe_ratio']:.1f}")
    return "\n".join(lines)


def render_strategic_recommendations():
    """Show strategic recommendations."""
    st.subheader("🎯 Recomendações Estratégicas")
    
    recommendations = [
        ("increase", "Aumentar posição em regimes de alto momentum e baixa volatilidade"),
        ("reduce", "Reduzir exposição durante tendência de baixa"),
        ("stop_loss", "Usar stop-loss mais apertado em alta volatilidade"),
        ("hedge", "Considerar hedge durante lateralização prolongada")
    ]
    
    cols = st.columns(len(recommendations))
    
    icons = {"increase": "📈", "reduce": "📉", "stop_loss": "🛑", "hedge": "🛡️"}
    
    for col, (rec_type, text) in zip(cols, recommendations):
        with col:
            st.info(f"{icons[rec_type]} {text}")