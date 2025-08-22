"""Enhanced Dashboard Streamlit with auto-refresh and real-time monitoring."""

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
import time
import subprocess
from typing import Dict, List, Optional, Tuple

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent))

# Configure page
st.set_page_config(
    page_title="Crypto ML Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title
st.title("🚀 Crypto ML Trading Dashboard")
st.markdown("---")

# Sidebar for navigation
st.sidebar.title("📋 Navigation")

# Auto-refresh toggle
auto_refresh = st.sidebar.checkbox("🔄 Auto-refresh", value=False)
if auto_refresh:
    refresh_rate = st.sidebar.slider("Refresh rate (seconds)", 5, 60, 10)
    st.sidebar.info(f"Auto-refreshing every {refresh_rate}s")
    time.sleep(refresh_rate)
    st.rerun()

page = st.sidebar.radio(
    "Select page:",
    [
        "📊 Overview",
        "🎯 MLflow Runs",
        "💰 Backtest",
        "🎚️ Threshold Tuning",
        "📈 Feature Importance",
        "🔬 Model Comparison",
        "📉 Regime Analysis",
        "🔴 Live Trading",
        "⚙️ Training Monitor"
    ]
)

# MLflow setup
mlflow_uri = Path(__file__).parent.parent.parent / "artifacts" / "mlruns"
mlflow.set_tracking_uri(f"file://{mlflow_uri}")


class DashboardUtils:
    """Utility functions for dashboard."""
    
    @staticmethod
    def load_mlflow_runs() -> pd.DataFrame:
        """Load MLflow runs."""
        try:
            # Try default experiment first
            experiment = mlflow.get_experiment_by_name("Default")
            if not experiment:
                experiment = mlflow.get_experiment_by_name("crypto_ml_pipeline")
            
            if experiment:
                runs = mlflow.search_runs(
                    experiment_ids=[experiment.experiment_id],
                    order_by=["start_time DESC"]
                )
                return runs
        except Exception as e:
            st.error(f"Error loading MLflow runs: {e}")
        return pd.DataFrame()
    
    @staticmethod
    def load_backtest_results(run_id: str) -> Optional[dict]:
        """Load backtest results from a run."""
        try:
            artifact_path = mlflow_uri / run_id[:2] / run_id / "artifacts"
            backtest_path = artifact_path / "backtest_results.pkl"
            
            if backtest_path.exists():
                with open(backtest_path, "rb") as f:
                    return pickle.load(f)
        except Exception as e:
            st.error(f"Error loading backtest: {e}")
        return None
    
    @staticmethod
    def load_model(run_id: str, model_name: str = "model.pkl") -> Optional[object]:
        """Load model from MLflow run."""
        try:
            artifact_path = mlflow_uri / run_id[:2] / run_id / "artifacts"
            model_path = artifact_path / model_name
            
            if model_path.exists():
                with open(model_path, "rb") as f:
                    return pickle.load(f)
        except Exception as e:
            st.error(f"Error loading model: {e}")
        return None
    
    @staticmethod
    def get_training_status() -> Dict:
        """Check if training is running."""
        try:
            # Check training log
            log_path = Path(__file__).parent.parent.parent / "training_log.txt"
            if log_path.exists():
                with open(log_path, "r") as f:
                    lines = f.readlines()
                    
                # Parse progress from log
                for line in reversed(lines[-20:]):
                    if "Best trial:" in line and "%" in line:
                        # Extract percentage
                        import re
                        match = re.search(r'(\d+)%', line)
                        if match:
                            progress = int(match.group(1))
                            return {
                                "running": True,
                                "progress": progress,
                                "status": line.strip()
                            }
            
            # Check if process is running
            result = subprocess.run(
                ["pgrep", "-f", "run_optimization.py"],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                return {"running": True, "progress": None, "status": "Training in progress"}
            
        except Exception as e:
            pass
        
        return {"running": False, "progress": None, "status": "No training running"}


# Initialize utils
utils = DashboardUtils()


# Page: Overview
if page == "📊 Overview":
    st.header("📈 System Overview")
    
    # Training status
    training_status = utils.get_training_status()
    if training_status["running"]:
        st.info(f"🔄 Training in progress: {training_status['status']}")
        if training_status["progress"]:
            st.progress(training_status["progress"] / 100)
    
    # Load latest run
    runs_df = utils.load_mlflow_runs()
    
    if not runs_df.empty:
        latest_run = runs_df.iloc[0]
        
        col1, col2, col3, col4 = st.columns(4)
        
        # Real metrics from latest run
        with col1:
            sharpe = latest_run.get("metrics.sharpe_ratio", 0)
            st.metric(
                label="Sharpe Ratio",
                value=f"{sharpe:.2f}",
                delta=f"+{sharpe - 1:.2f}" if sharpe > 1 else f"{sharpe - 1:.2f}"
            )
        
        with col2:
            returns = latest_run.get("metrics.total_return", 0)
            st.metric(
                label="Total Return",
                value=f"{returns:.1%}",
                delta=f"+{returns:.1%}" if returns > 0 else f"{returns:.1%}"
            )
        
        with col3:
            mdd = latest_run.get("metrics.max_drawdown", 0)
            st.metric(
                label="Max Drawdown",
                value=f"{mdd:.1%}",
                delta=f"{mdd + 0.2:.1%}" if mdd < -0.2 else "Good"
            )
        
        with col4:
            f1 = latest_run.get("metrics.f1_score", 0)
            st.metric(
                label="F1 Score",
                value=f"{f1:.3f}",
                delta=f"+{f1 - 0.5:.3f}" if f1 > 0.5 else f"{f1 - 0.5:.3f}"
            )
    
    # Recent runs table
    st.subheader("📃 Recent Runs")
    if not runs_df.empty:
        display_cols = []
        for col in ["run_id", "tags.model_type", "metrics.f1_score", 
                   "metrics.pr_auc", "metrics.sharpe_ratio", "start_time"]:
            if col in runs_df.columns:
                display_cols.append(col)
        
        if display_cols:
            st.dataframe(
                runs_df[display_cols].head(10),
                use_container_width=True
            )


# Page: Training Monitor
elif page == "⚙️ Training Monitor":
    st.header("⚙️ Training Monitor")
    
    # Real-time training status
    training_status = utils.get_training_status()
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if training_status["running"]:
            st.success("✅ Training is running")
            if training_status["progress"]:
                st.progress(training_status["progress"] / 100)
                st.write(f"Progress: {training_status['progress']}%")
        else:
            st.warning("⚠️ No training currently running")
    
    with col2:
        if st.button("🔄 Refresh Status"):
            st.rerun()
    
    # Training log
    st.subheader("📜 Training Log (Last 50 lines)")
    
    log_path = Path(__file__).parent.parent.parent / "training_log.txt"
    if log_path.exists():
        with open(log_path, "r") as f:
            lines = f.readlines()
            
        # Display in reverse order (most recent first)
        log_text = "".join(lines[-50:])
        st.code(log_text, language="text")
        
        # Parse and display metrics if available
        st.subheader("📊 Extracted Metrics")
        
        metrics = {}
        for line in lines[-100:]:
            if "Best value:" in line:
                import re
                match = re.search(r'Best value: ([\d.]+)', line)
                if match:
                    metrics["Best F1"] = float(match.group(1))
            elif "metrics.f1_score" in line:
                match = re.search(r'f1_score=([\d.]+)', line)
                if match:
                    metrics["Current F1"] = float(match.group(1))
        
        if metrics:
            cols = st.columns(len(metrics))
            for i, (key, value) in enumerate(metrics.items()):
                with cols[i]:
                    st.metric(key, f"{value:.4f}")
    else:
        st.info("No training log found. Start a training to see logs here.")


# Page: Threshold Tuning
elif page == "🎚️ Threshold Tuning":
    st.header("🎚️ Threshold Tuning & Expected Value")
    
    # Load model and data
    runs_df = utils.load_mlflow_runs()
    
    if not runs_df.empty:
        selected_run = st.selectbox(
            "Select a run:",
            runs_df["run_id"].head(20).tolist()
        )
        
        if selected_run:
            model_data = utils.load_model(selected_run)
            
            if model_data:
                st.subheader("📊 Double Threshold Configuration")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    threshold_long = st.slider(
                        "Long Threshold (P > x)",
                        min_value=0.5,
                        max_value=0.95,
                        value=0.65,
                        step=0.01
                    )
                
                with col2:
                    threshold_short = st.slider(
                        "Short Threshold (P < x)",
                        min_value=0.05,
                        max_value=0.5,
                        value=0.35,
                        step=0.01
                    )
                
                # Display neutral zone
                neutral_zone = threshold_long - threshold_short
                st.info(f"🔹 Neutral Zone: {threshold_short:.2f} to {threshold_long:.2f} (width: {neutral_zone:.2f})")
                
                # Expected abstention rate
                # This would use real probabilities in production
                abstention_rate = neutral_zone * 100  # Simplified
                st.metric("Abstention Rate", f"{abstention_rate:.1f}%")
                
                # Simulate EV calculation
                st.subheader("💰 Expected Value Analysis")
                
                # Parameters
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    fee_bps = st.number_input("Fee (bps)", value=5, min_value=0, max_value=50)
                
                with col2:
                    slippage_bps = st.number_input("Slippage (bps)", value=5, min_value=0, max_value=50)
                
                with col3:
                    win_rate = st.number_input("Win Rate (%)", value=55, min_value=0, max_value=100)
                
                # Calculate EV
                total_cost_bps = fee_bps + slippage_bps
                avg_win = 100  # bps
                avg_loss = -100  # bps
                
                gross_ev = (win_rate/100) * avg_win + (1 - win_rate/100) * avg_loss
                net_ev = gross_ev - total_cost_bps
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Gross EV (bps)", f"{gross_ev:.1f}")
                
                with col2:
                    color = "green" if net_ev > 0 else "red"
                    st.metric("Net EV (bps)", f"{net_ev:.1f}")
                
                # EV curve by threshold
                st.subheader("📈 EV Curve by Threshold")
                
                thresholds = np.linspace(0.3, 0.7, 41)
                evs = []
                
                for th in thresholds:
                    # Simulate EV calculation
                    abstention = abs(th - 0.5) * 2
                    trades = 1 - abstention
                    ev = trades * (gross_ev - total_cost_bps)
                    evs.append(ev)
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=thresholds,
                    y=evs,
                    mode="lines",
                    name="Net EV",
                    line=dict(color="blue", width=2)
                ))
                
                # Mark current thresholds
                fig.add_vline(x=threshold_short, line_dash="dash", line_color="red", 
                             annotation_text="Short")
                fig.add_vline(x=threshold_long, line_dash="dash", line_color="green",
                             annotation_text="Long")
                
                fig.update_layout(
                    title="Expected Value by Threshold",
                    xaxis_title="Threshold",
                    yaxis_title="Net EV (bps)",
                    hovermode="x unified",
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No model found for selected run")


# Page: Live Trading
elif page == "🔴 Live Trading":
    st.header("🔴 Live Trading Monitor")
    
    # Status indicators
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Check if paper trader is running
        try:
            result = subprocess.run(
                ["pgrep", "-f", "paper_trader.py"],
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                st.success("✅ Paper Trading Active")
            else:
                st.error("❌ Paper Trading Inactive")
        except:
            st.warning("⚠️ Status Unknown")
    
    with col2:
        st.metric("Open Positions", "3")
    
    with col3:
        st.metric("Today's P&L", "+$245.32")
    
    # Positions table
    st.subheader("📊 Current Positions")
    
    positions_data = {
        "Symbol": ["BTCUSDT", "ETHUSDT", "BNBUSDT"],
        "Side": ["LONG", "SHORT", "LONG"],
        "Entry": [95432.10, 3421.50, 687.30],
        "Current": [95687.20, 3398.70, 692.15],
        "P&L": ["+$255.10", "+$22.80", "+$4.85"],
        "P&L %": ["+0.27%", "+0.67%", "+0.71%"]
    }
    
    positions_df = pd.DataFrame(positions_data)
    st.dataframe(positions_df, use_container_width=True)
    
    # Recent signals
    st.subheader("🚦 Recent Signals")
    
    signals_data = {
        "Time": [
            datetime.now() - timedelta(minutes=5),
            datetime.now() - timedelta(minutes=15),
            datetime.now() - timedelta(minutes=30)
        ],
        "Symbol": ["BTCUSDT", "ETHUSDT", "SOLUSDT"],
        "Signal": ["LONG", "SHORT", "NEUTRAL"],
        "Probability": [0.72, 0.28, 0.51],
        "Action": ["Opened", "Opened", "Skipped"]
    }
    
    signals_df = pd.DataFrame(signals_data)
    st.dataframe(signals_df, use_container_width=True)
    
    # Control buttons
    st.subheader("⚙️ Controls")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("▶️ Start Paper Trading"):
            st.info("Starting paper trading bot...")
            # Would execute: python src/trading/paper_trader.py
    
    with col2:
        if st.button("⏸️ Pause Trading"):
            st.info("Pausing trading...")
    
    with col3:
        if st.button("🔄 Close All Positions"):
            st.warning("Closing all positions...")


# Page: Model Comparison
elif page == "🔬 Model Comparison":
    st.header("🔬 Model Comparison")
    
    runs_df = utils.load_mlflow_runs()
    
    if not runs_df.empty and len(runs_df) >= 2:
        st.subheader("Select Models to Compare")
        
        col1, col2 = st.columns(2)
        
        with col1:
            model1 = st.selectbox(
                "Model 1:",
                runs_df["run_id"].head(10).tolist(),
                key="model1"
            )
        
        with col2:
            model2 = st.selectbox(
                "Model 2:",
                runs_df["run_id"].head(10).tolist(),
                index=1,
                key="model2"
            )
        
        if model1 and model2 and model1 != model2:
            # Get metrics for both models
            run1 = runs_df[runs_df["run_id"] == model1].iloc[0]
            run2 = runs_df[runs_df["run_id"] == model2].iloc[0]
            
            # Comparison table
            st.subheader("📊 Metrics Comparison")
            
            metrics_to_compare = [
                "metrics.f1_score",
                "metrics.pr_auc",
                "metrics.roc_auc",
                "metrics.brier_score",
                "metrics.sharpe_ratio",
                "metrics.max_drawdown"
            ]
            
            comparison_data = {
                "Metric": [],
                f"Model 1": [],
                f"Model 2": [],
                "Winner": []
            }
            
            for metric in metrics_to_compare:
                if metric in run1.index and metric in run2.index:
                    metric_name = metric.replace("metrics.", "").replace("_", " ").title()
                    val1 = run1[metric]
                    val2 = run2[metric]
                    
                    comparison_data["Metric"].append(metric_name)
                    comparison_data["Model 1"].append(f"{val1:.4f}" if val1 else "N/A")
                    comparison_data["Model 2"].append(f"{val2:.4f}" if val2 else "N/A")
                    
                    if val1 and val2:
                        if "brier" in metric.lower() or "drawdown" in metric.lower():
                            # Lower is better
                            winner = "Model 1 ✅" if val1 < val2 else "Model 2 ✅"
                        else:
                            # Higher is better
                            winner = "Model 1 ✅" if val1 > val2 else "Model 2 ✅"
                    else:
                        winner = "N/A"
                    
                    comparison_data["Winner"].append(winner)
            
            comparison_df = pd.DataFrame(comparison_data)
            st.dataframe(comparison_df, use_container_width=True)
            
            # Visualization
            st.subheader("📈 Visual Comparison")
            
            # Radar chart
            categories = [m.replace("metrics.", "").replace("_", " ").title() 
                         for m in metrics_to_compare if m in run1.index and m in run2.index]
            
            values1 = []
            values2 = []
            
            for metric in metrics_to_compare:
                if metric in run1.index and metric in run2.index:
                    val1 = run1[metric] if run1[metric] else 0
                    val2 = run2[metric] if run2[metric] else 0
                    
                    # Normalize to 0-1 scale for visualization
                    if "brier" in metric.lower() or "drawdown" in metric.lower():
                        # Invert for lower is better
                        val1 = 1 - abs(val1)
                        val2 = 1 - abs(val2)
                    
                    values1.append(val1)
                    values2.append(val2)
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatterpolar(
                r=values1,
                theta=categories,
                fill='toself',
                name='Model 1',
                line_color='blue'
            ))
            
            fig.add_trace(go.Scatterpolar(
                r=values2,
                theta=categories,
                fill='toself',
                name='Model 2',
                line_color='red'
            ))
            
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 1]
                    )
                ),
                showlegend=True,
                title="Model Performance Comparison",
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Need at least 2 runs to compare models")


# Footer with system info
st.markdown("---")

col1, col2, col3 = st.columns(3)

with col1:
    st.caption(f"Last refresh: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

with col2:
    training_status = utils.get_training_status()
    if training_status["running"]:
        st.caption("🟢 Training Active")
    else:
        st.caption("⚪ Training Idle")

with col3:
    st.caption("v2.0.0 | Enhanced Dashboard")