"""
Metric components for ML Trading Dashboard.
KPI cards, statistics displays, and performance metrics.
"""

import streamlit as st
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime, timedelta

class MetricCards:
    """Professional metric card components."""
    
    @staticmethod
    def render_kpi_card(
        title: str,
        value: Union[str, float],
        delta: Optional[Union[str, float]] = None,
        delta_color: str = "normal",
        icon: Optional[str] = None,
        subtitle: Optional[str] = None
    ):
        """
        Render a KPI metric card with custom styling.
        
        Args:
            title: Metric title
            value: Main value to display
            delta: Change value
            delta_color: Color for delta ('normal', 'inverse', 'off')
            icon: Optional emoji icon
            subtitle: Optional subtitle text
        """
        with st.container():
            st.markdown(
                f"""
                <div class="dashboard-card">
                    <div style="display: flex; align-items: center; justify-content: space-between;">
                        <div>
                            {f'<span style="font-size: 1.5rem; margin-right: 0.5rem;">{icon}</span>' if icon else ''}
                            <span style="color: var(--text-muted); font-size: 0.875rem; font-weight: 500;">
                                {title}
                            </span>
                        </div>
                    </div>
                    <div style="margin-top: 0.5rem;">
                        <span style="font-size: 1.875rem; font-weight: 700; color: var(--text-primary);">
                            {value}
                        </span>
                    </div>
                    {f'<div style="margin-top: 0.25rem;"><span style="color: var(--text-muted); font-size: 0.75rem;">{subtitle}</span></div>' if subtitle else ''}
                </div>
                """,
                unsafe_allow_html=True
            )
            
            if delta is not None:
                st.metric("", "", delta=delta, delta_color=delta_color, label_visibility="collapsed")
    
    @staticmethod
    def render_metric_grid(metrics: List[Dict], cols: int = 4):
        """
        Render a grid of metric cards.
        
        Args:
            metrics: List of metric dictionaries with keys: title, value, delta, icon, subtitle
            cols: Number of columns
        """
        columns = st.columns(cols)
        
        for idx, metric in enumerate(metrics):
            with columns[idx % cols]:
                MetricCards.render_kpi_card(
                    title=metric.get("title", ""),
                    value=metric.get("value", ""),
                    delta=metric.get("delta"),
                    delta_color=metric.get("delta_color", "normal"),
                    icon=metric.get("icon"),
                    subtitle=metric.get("subtitle")
                )
    
    @staticmethod
    def render_trading_metrics(performance_data: Dict):
        """
        Render comprehensive trading performance metrics.
        
        Args:
            performance_data: Dictionary with performance metrics
        """
        # Main performance metrics
        st.markdown("### 📊 Performance Metrics")
        
        main_metrics = [
            {
                "title": "Total Return",
                "value": f"{performance_data.get('total_return', 0):.2%}",
                "delta": f"{performance_data.get('return_delta', 0):.2%}",
                "icon": "💰",
                "subtitle": "Since inception"
            },
            {
                "title": "Sharpe Ratio",
                "value": f"{performance_data.get('sharpe_ratio', 0):.2f}",
                "delta": f"{performance_data.get('sharpe_delta', 0):+.2f}",
                "icon": "📈",
                "subtitle": "Risk-adjusted return"
            },
            {
                "title": "Max Drawdown",
                "value": f"{performance_data.get('max_drawdown', 0):.2%}",
                "delta": f"{performance_data.get('dd_delta', 0):.2%}",
                "delta_color": "inverse",
                "icon": "📉",
                "subtitle": "Maximum loss"
            },
            {
                "title": "Win Rate",
                "value": f"{performance_data.get('win_rate', 0):.1%}",
                "delta": f"{performance_data.get('wr_delta', 0):+.1%}",
                "icon": "🎯",
                "subtitle": "Profitable trades"
            }
        ]
        
        MetricCards.render_metric_grid(main_metrics, cols=4)
        
        # Risk metrics
        st.markdown("### ⚠️ Risk Metrics")
        
        risk_metrics = [
            {
                "title": "Value at Risk (95%)",
                "value": f"${performance_data.get('var_95', 0):,.0f}",
                "icon": "🛡️",
                "subtitle": "1-day VaR"
            },
            {
                "title": "Sortino Ratio",
                "value": f"{performance_data.get('sortino_ratio', 0):.2f}",
                "icon": "📊",
                "subtitle": "Downside risk"
            },
            {
                "title": "Calmar Ratio",
                "value": f"{performance_data.get('calmar_ratio', 0):.2f}",
                "icon": "⚖️",
                "subtitle": "Return / Max DD"
            },
            {
                "title": "Recovery Factor",
                "value": f"{performance_data.get('recovery_factor', 0):.2f}",
                "icon": "🔄",
                "subtitle": "Profit / Max DD"
            }
        ]
        
        MetricCards.render_metric_grid(risk_metrics, cols=4)
    
    @staticmethod
    def render_position_summary(positions: pd.DataFrame):
        """
        Render current position summary.
        
        Args:
            positions: DataFrame with position data
        """
        if positions.empty:
            st.info("No open positions")
            return
        
        # Summary metrics
        total_value = positions['value'].sum()
        total_pnl = positions['pnl'].sum()
        total_pnl_pct = positions['pnl_pct'].mean()
        
        summary_metrics = [
            {
                "title": "Open Positions",
                "value": len(positions),
                "icon": "📋"
            },
            {
                "title": "Total Value",
                "value": f"${total_value:,.2f}",
                "icon": "💵"
            },
            {
                "title": "Unrealized P&L",
                "value": f"${total_pnl:,.2f}",
                "delta": f"{total_pnl_pct:.2%}",
                "delta_color": "normal" if total_pnl >= 0 else "inverse",
                "icon": "💹"
            },
            {
                "title": "Avg Position Size",
                "value": f"${total_value/len(positions):,.2f}",
                "icon": "📊"
            }
        ]
        
        MetricCards.render_metric_grid(summary_metrics, cols=4)
    
    @staticmethod
    def render_model_metrics(model_data: Dict):
        """
        Render model performance metrics.
        
        Args:
            model_data: Dictionary with model metrics
        """
        st.markdown("### 🤖 Model Performance")
        
        model_metrics = [
            {
                "title": "F1 Score",
                "value": f"{model_data.get('f1_score', 0):.3f}",
                "delta": f"{model_data.get('f1_delta', 0):+.3f}",
                "icon": "🎯",
                "subtitle": "Classification quality"
            },
            {
                "title": "Precision",
                "value": f"{model_data.get('precision', 0):.3f}",
                "delta": f"{model_data.get('precision_delta', 0):+.3f}",
                "icon": "✅",
                "subtitle": "True positive rate"
            },
            {
                "title": "Recall",
                "value": f"{model_data.get('recall', 0):.3f}",
                "delta": f"{model_data.get('recall_delta', 0):+.3f}",
                "icon": "🔍",
                "subtitle": "Sensitivity"
            },
            {
                "title": "ROC-AUC",
                "value": f"{model_data.get('roc_auc', 0):.3f}",
                "delta": f"{model_data.get('auc_delta', 0):+.3f}",
                "icon": "📈",
                "subtitle": "Discrimination ability"
            }
        ]
        
        MetricCards.render_metric_grid(model_metrics, cols=4)
    
    @staticmethod
    def render_live_metrics(streaming_data: Dict):
        """
        Render live streaming metrics with real-time updates.
        
        Args:
            streaming_data: Dictionary with live data
        """
        # Create placeholder for live updates
        placeholder = st.empty()
        
        with placeholder.container():
            live_metrics = [
                {
                    "title": "BTC Price",
                    "value": f"${streaming_data.get('btc_price', 0):,.2f}",
                    "delta": f"{streaming_data.get('btc_change', 0):.2%}",
                    "icon": "₿",
                    "subtitle": "Live"
                },
                {
                    "title": "24h Volume",
                    "value": f"${streaming_data.get('volume_24h', 0)/1e9:.2f}B",
                    "delta": f"{streaming_data.get('volume_change', 0):.1%}",
                    "icon": "📊",
                    "subtitle": "Trading volume"
                },
                {
                    "title": "Signal",
                    "value": streaming_data.get('signal', 'NEUTRAL'),
                    "icon": "🚦",
                    "subtitle": f"Confidence: {streaming_data.get('confidence', 0):.1%}"
                },
                {
                    "title": "Next Update",
                    "value": streaming_data.get('next_update', '00:00'),
                    "icon": "⏱️",
                    "subtitle": "Time remaining"
                }
            ]
            
            MetricCards.render_metric_grid(live_metrics, cols=4)
        
        return placeholder
    
    @staticmethod
    def render_statistics_table(stats: pd.DataFrame, title: str = "Statistics"):
        """
        Render a formatted statistics table.
        
        Args:
            stats: DataFrame with statistics
            title: Table title
        """
        st.markdown(f"### {title}")
        
        # Apply custom styling to dataframe
        styled_df = stats.style.format("{:.4f}").set_properties(**{
            'background-color': 'var(--bg-card)',
            'color': 'var(--text-primary)',
            'border': '1px solid var(--border)'
        })
        
        st.dataframe(styled_df, use_container_width=True)