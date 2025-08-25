"""
Utility functions for the Streamlit dashboard.

Handles model loading, data caching, and report generation.
"""

import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import streamlit as st
import mlflow
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


@st.cache_resource
def load_model(model_path: str) -> Dict:
    """
    Load a saved model from disk with caching.
    
    Args:
        model_path: Path to the saved model pickle file
        
    Returns:
        Dictionary containing model and metadata
    """
    try:
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        log.info(f"Model loaded from {model_path}")
        return model_data
    except FileNotFoundError:
        st.error(f"Model file not found: {model_path}")
        return None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None


@st.cache_data(ttl=300)  # Cache for 5 minutes
def load_mlflow_runs(experiment_name: str = "crypto_ml_production") -> pd.DataFrame:
    """
    Load MLflow runs with caching.
    
    Args:
        experiment_name: Name of the MLflow experiment
        
    Returns:
        DataFrame with run information
    """
    try:
        mlflow.set_tracking_uri("artifacts/mlruns")
        experiment = mlflow.get_experiment_by_name(experiment_name)
        
        if experiment:
            runs = mlflow.search_runs(
                experiment_ids=[experiment.experiment_id],
                order_by=["start_time DESC"],
                max_results=100
            )
            
            # Process runs for better display
            if not runs.empty:
                # Extract key metrics
                metric_cols = [col for col in runs.columns if col.startswith('metrics.')]
                param_cols = [col for col in runs.columns if col.startswith('params.')]
                tag_cols = [col for col in runs.columns if col.startswith('tags.')]
                
                # Clean column names
                for col in metric_cols:
                    new_col = col.replace('metrics.', '')
                    runs[new_col] = runs[col]
                
                for col in param_cols:
                    new_col = col.replace('params.', '')
                    runs[new_col] = runs[col]
                
                for col in tag_cols:
                    new_col = col.replace('tags.', '')
                    runs[new_col] = runs[col]
                
                # Add derived metrics
                if 'f1_score' in runs.columns:
                    runs['f1_score'] = runs['f1_score'].astype(float)
                
                if 'sharpe_ratio' in runs.columns:
                    runs['sharpe_ratio'] = runs['sharpe_ratio'].astype(float)
                
                log.info(f"Loaded {len(runs)} MLflow runs")
            
            return runs
        else:
            log.warning(f"Experiment '{experiment_name}' not found")
            return pd.DataFrame()
            
    except Exception as e:
        log.error(f"Error loading MLflow runs: {e}")
        return pd.DataFrame()


@st.cache_data
def load_backtest_results(run_id: str) -> Optional[Dict]:
    """
    Load backtest results for a specific MLflow run.
    
    Args:
        run_id: MLflow run ID
        
    Returns:
        Dictionary with backtest results or None
    """
    try:
        client = mlflow.tracking.MlflowClient()
        
        # Try to download backtest artifact
        try:
            artifact_path = client.download_artifacts(run_id, "backtest_results.pkl")
            with open(artifact_path, 'rb') as f:
                results = pickle.load(f)
            return results
        except:
            # Fallback to local path
            local_path = Path(f"artifacts/mlruns/{run_id[:2]}/{run_id}/artifacts/backtest_results.pkl")
            if local_path.exists():
                with open(local_path, 'rb') as f:
                    return pickle.load(f)
        
        return None
        
    except Exception as e:
        log.error(f"Error loading backtest results: {e}")
        return None


def calculate_drawdown(equity: pd.Series) -> pd.Series:
    """
    Calculate drawdown from equity curve.
    
    Args:
        equity: Series with equity values
        
    Returns:
        Series with drawdown percentages
    """
    running_max = equity.expanding().max()
    drawdown = (equity - running_max) / running_max * 100
    return drawdown


def create_performance_metrics(backtest_results: Dict) -> Dict:
    """
    Calculate comprehensive performance metrics from backtest results.
    
    Args:
        backtest_results: Dictionary with backtest data
        
    Returns:
        Dictionary with calculated metrics
    """
    metrics = {}
    
    if 'equity' in backtest_results:
        equity = pd.Series(backtest_results['equity'])
        returns = equity.pct_change().dropna()
        
        # Basic metrics
        metrics['total_return'] = (equity.iloc[-1] / equity.iloc[0] - 1) * 100
        metrics['annualized_return'] = metrics['total_return'] * (252 / len(returns))
        metrics['volatility'] = returns.std() * np.sqrt(252) * 100
        
        # Risk metrics
        metrics['sharpe_ratio'] = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0
        metrics['sortino_ratio'] = (returns.mean() / returns[returns < 0].std()) * np.sqrt(252) if len(returns[returns < 0]) > 0 else 0
        
        # Drawdown metrics
        drawdown = calculate_drawdown(equity)
        metrics['max_drawdown'] = drawdown.min()
        metrics['avg_drawdown'] = drawdown[drawdown < 0].mean() if len(drawdown[drawdown < 0]) > 0 else 0
        
        # Win rate
        winning_days = (returns > 0).sum()
        total_days = len(returns)
        metrics['win_rate'] = (winning_days / total_days) * 100 if total_days > 0 else 0
        
        # Profit factor
        gains = returns[returns > 0].sum()
        losses = abs(returns[returns < 0].sum())
        metrics['profit_factor'] = gains / losses if losses > 0 else float('inf')
        
    return metrics


def create_equity_chart(backtest_results: Dict) -> go.Figure:
    """
    Create interactive equity curve chart.
    
    Args:
        backtest_results: Dictionary with backtest data
        
    Returns:
        Plotly figure with equity curve and drawdown
    """
    if 'equity' not in backtest_results:
        return go.Figure()
    
    equity = pd.Series(backtest_results['equity'])
    drawdown = calculate_drawdown(equity)
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.7, 0.3],
        subplot_titles=("Equity Curve", "Drawdown %"),
        vertical_spacing=0.05
    )
    
    # Equity curve
    fig.add_trace(
        go.Scatter(
            y=equity,
            mode='lines',
            name='Equity',
            line=dict(color='blue', width=2),
            hovertemplate='Equity: $%{y:,.0f}<extra></extra>'
        ),
        row=1, col=1
    )
    
    # Add buy/hold reference if available
    if 'buy_hold_equity' in backtest_results:
        fig.add_trace(
            go.Scatter(
                y=backtest_results['buy_hold_equity'],
                mode='lines',
                name='Buy & Hold',
                line=dict(color='gray', width=1, dash='dash'),
                hovertemplate='Buy & Hold: $%{y:,.0f}<extra></extra>'
            ),
            row=1, col=1
        )
    
    # Drawdown
    fig.add_trace(
        go.Scatter(
            y=drawdown,
            mode='lines',
            name='Drawdown',
            fill='tozeroy',
            line=dict(color='red', width=1),
            hovertemplate='Drawdown: %{y:.2f}%<extra></extra>'
        ),
        row=2, col=1
    )
    
    # Update layout
    fig.update_xaxes(title_text="Time", row=2, col=1)
    fig.update_yaxes(title_text="Equity ($)", row=1, col=1)
    fig.update_yaxes(title_text="Drawdown (%)", row=2, col=1)
    
    fig.update_layout(
        height=600,
        hovermode='x unified',
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )
    
    return fig


def create_returns_distribution(backtest_results: Dict) -> go.Figure:
    """
    Create returns distribution histogram.
    
    Args:
        backtest_results: Dictionary with backtest data
        
    Returns:
        Plotly figure with returns distribution
    """
    if 'equity' not in backtest_results:
        return go.Figure()
    
    equity = pd.Series(backtest_results['equity'])
    returns = equity.pct_change().dropna() * 100  # Convert to percentage
    
    fig = go.Figure()
    
    # Histogram
    fig.add_trace(go.Histogram(
        x=returns,
        nbinsx=50,
        name='Daily Returns',
        marker_color='blue',
        opacity=0.7
    ))
    
    # Add normal distribution overlay
    mean_return = returns.mean()
    std_return = returns.std()
    x_range = np.linspace(returns.min(), returns.max(), 100)
    normal_dist = np.exp(-(x_range - mean_return)**2 / (2 * std_return**2))
    normal_dist = normal_dist / normal_dist.max() * returns.value_counts().max()
    
    fig.add_trace(go.Scatter(
        x=x_range,
        y=normal_dist,
        mode='lines',
        name='Normal Distribution',
        line=dict(color='red', width=2)
    ))
    
    # Add vertical lines for mean and median
    fig.add_vline(x=mean_return, line_dash="dash", line_color="green", 
                  annotation_text=f"Mean: {mean_return:.2f}%")
    fig.add_vline(x=returns.median(), line_dash="dash", line_color="orange",
                  annotation_text=f"Median: {returns.median():.2f}%")
    
    fig.update_layout(
        title="Daily Returns Distribution",
        xaxis_title="Daily Return (%)",
        yaxis_title="Frequency",
        showlegend=True,
        height=400
    )
    
    return fig


def create_monthly_returns_heatmap(backtest_results: Dict) -> go.Figure:
    """
    Create monthly returns heatmap.
    
    Args:
        backtest_results: Dictionary with backtest data
        
    Returns:
        Plotly figure with monthly returns heatmap
    """
    if 'equity' not in backtest_results or 'dates' not in backtest_results:
        return go.Figure()
    
    # Create DataFrame with dates and equity
    df = pd.DataFrame({
        'date': pd.to_datetime(backtest_results['dates']),
        'equity': backtest_results['equity']
    })
    df.set_index('date', inplace=True)
    
    # Calculate monthly returns
    monthly_returns = df['equity'].resample('M').last().pct_change() * 100
    
    # Create pivot table for heatmap
    monthly_returns_df = pd.DataFrame(monthly_returns)
    monthly_returns_df['year'] = monthly_returns_df.index.year
    monthly_returns_df['month'] = monthly_returns_df.index.month
    
    pivot_table = monthly_returns_df.pivot_table(
        values='equity', 
        index='year', 
        columns='month'
    )
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=pivot_table.values,
        x=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
           'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
        y=pivot_table.index,
        colorscale='RdYlGn',
        zmid=0,
        text=pivot_table.values.round(2),
        texttemplate='%{text}%',
        textfont={"size": 10},
        colorbar=dict(title="Return %")
    ))
    
    fig.update_layout(
        title="Monthly Returns Heatmap",
        xaxis_title="Month",
        yaxis_title="Year",
        height=400
    )
    
    return fig


def generate_performance_report(backtest_results: Dict, model_info: Dict) -> str:
    """
    Generate a comprehensive performance report.
    
    Args:
        backtest_results: Dictionary with backtest data
        model_info: Dictionary with model information
        
    Returns:
        HTML string with formatted report
    """
    metrics = calculate_performance_metrics(backtest_results)
    
    report = f"""
    <h2>Performance Report</h2>
    <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    
    <h3>Model Information</h3>
    <ul>
        <li><strong>Model Type:</strong> {model_info.get('model_type', 'Unknown')}</li>
        <li><strong>Training Period:</strong> {model_info.get('train_period', 'Unknown')}</li>
        <li><strong>Test Period:</strong> {model_info.get('test_period', 'Unknown')}</li>
    </ul>
    
    <h3>Performance Metrics</h3>
    <table style="width:100%">
        <tr>
            <td><strong>Total Return:</strong></td>
            <td>{metrics.get('total_return', 0):.2f}%</td>
            <td><strong>Sharpe Ratio:</strong></td>
            <td>{metrics.get('sharpe_ratio', 0):.2f}</td>
        </tr>
        <tr>
            <td><strong>Annualized Return:</strong></td>
            <td>{metrics.get('annualized_return', 0):.2f}%</td>
            <td><strong>Sortino Ratio:</strong></td>
            <td>{metrics.get('sortino_ratio', 0):.2f}</td>
        </tr>
        <tr>
            <td><strong>Volatility:</strong></td>
            <td>{metrics.get('volatility', 0):.2f}%</td>
            <td><strong>Max Drawdown:</strong></td>
            <td>{metrics.get('max_drawdown', 0):.2f}%</td>
        </tr>
        <tr>
            <td><strong>Win Rate:</strong></td>
            <td>{metrics.get('win_rate', 0):.2f}%</td>
            <td><strong>Profit Factor:</strong></td>
            <td>{metrics.get('profit_factor', 0):.2f}</td>
        </tr>
    </table>
    
    <h3>Risk Analysis</h3>
    <p>
        The strategy shows a Sharpe ratio of {metrics.get('sharpe_ratio', 0):.2f}, 
        indicating {'good' if metrics.get('sharpe_ratio', 0) > 1 else 'poor'} risk-adjusted returns.
        Maximum drawdown of {abs(metrics.get('max_drawdown', 0)):.2f}% suggests 
        {'moderate' if abs(metrics.get('max_drawdown', 0)) < 20 else 'high'} risk exposure.
    </p>
    """
    
    return report


def load_available_models() -> Dict[str, str]:
    """
    Find all available saved models.
    
    Returns:
        Dictionary mapping model names to file paths
    """
    models_dir = Path("artifacts/models")
    models = {}
    
    if models_dir.exists():
        for model_file in models_dir.glob("*.pkl"):
            model_name = model_file.stem.replace('_', ' ').title()
            models[model_name] = str(model_file)
    
    # Also check for ensemble models
    ensemble_dir = Path("artifacts/ensemble")
    if ensemble_dir.exists():
        for model_file in ensemble_dir.glob("*.pkl"):
            model_name = f"Ensemble - {model_file.stem.replace('_', ' ').title()}"
            models[model_name] = str(model_file)
    
    return models


def compare_model_performance(models: List[Dict]) -> pd.DataFrame:
    """
    Create comparison DataFrame for multiple models.
    
    Args:
        models: List of dictionaries with model results
        
    Returns:
        DataFrame with comparison metrics
    """
    comparison_data = []
    
    for model in models:
        row = {
            'Model': model.get('name', 'Unknown'),
            'F1 Score': model.get('f1_score', 0),
            'PR-AUC': model.get('pr_auc', 0),
            'ROC-AUC': model.get('roc_auc', 0),
            'Brier Score': model.get('brier_score', 1),
            'Sharpe Ratio': model.get('sharpe_ratio', 0),
            'Total Return': model.get('total_return', 0),
            'Max Drawdown': model.get('max_drawdown', 0),
            'Win Rate': model.get('win_rate', 0)
        }
        comparison_data.append(row)
    
    return pd.DataFrame(comparison_data)