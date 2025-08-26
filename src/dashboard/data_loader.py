"""Data loading utilities for dashboard."""

import pandas as pd
import pickle
import mlflow
from pathlib import Path
from typing import Optional, Dict, Any
import streamlit as st


class DataLoader:
    """Handle data loading for dashboard."""
    
    def __init__(self, config):
        """
        Initialize data loader.
        
        Args:
            config: Dashboard configuration
        """
        self.config = config
        mlflow.set_tracking_uri(config.mlflow_tracking_uri)
    
    @st.cache_data(ttl=60)  # Cache for 60 seconds
    def load_mlflow_runs(_self) -> pd.DataFrame:
        """
        Load MLflow experiment runs.
        
        Returns:
            DataFrame with run information
        """
        try:
            exp = mlflow.get_experiment_by_name(_self.config.experiment_name)
            if exp is None:
                # Fallback: tentar experimento padrão usado pelo XGBoost
                exp = mlflow.get_experiment_by_name('xgboost_optimization')
            if exp is not None:
                return mlflow.search_runs(
                    experiment_ids=[exp.experiment_id],
                    order_by=["start_time DESC"]
                )
            # Fallback final: juntar todos os experimentos (limite simples)
            runs_all = []
            for e in mlflow.search_experiments():
                df = mlflow.search_runs(experiment_ids=[e.experiment_id], order_by=["start_time DESC"], max_results=100)
                if df is not None and not df.empty:
                    df['experiment'] = e.name
                    runs_all.append(df)
            if runs_all:
                return pd.concat(runs_all, ignore_index=True)
        except Exception as e:
            st.error(f"Erro ao carregar runs MLflow: {e}")
        return pd.DataFrame()
    
    @st.cache_data(ttl=300)  # Cache for 5 minutes
    def load_backtest_results(_self, run_id: str) -> Optional[Dict[str, Any]]:
        """
        Load backtest results for a specific run.
        
        Args:
            run_id: MLflow run ID
            
        Returns:
            Dictionary with backtest results or None
        """
        try:
            artifact_path = _self.config.mlruns_path / run_id[:2] / run_id / "artifacts"
            backtest_path = artifact_path / "backtest_results.pkl"
            
            if backtest_path.exists():
                with open(backtest_path, "rb") as f:
                    return pickle.load(f)
        except Exception as e:
            st.error(f"Erro ao carregar backtest: {e}")
        return None
    
    @st.cache_data(ttl=300)
    def load_feature_importance(_self, run_id: str) -> Optional[pd.DataFrame]:
        """
        Load feature importance for a specific run.
        
        Args:
            run_id: MLflow run ID
            
        Returns:
            DataFrame with feature importance or None
        """
        try:
            artifact_path = _self.config.mlruns_path / run_id[:2] / run_id / "artifacts"
            importance_path = artifact_path / "feature_importance.csv"
            
            if importance_path.exists():
                return pd.read_csv(importance_path)
        except Exception as e:
            st.error(f"Erro ao carregar feature importance: {e}")
        return None
    
    @st.cache_data(ttl=300)
    def load_threshold_analysis(_self, run_id: str) -> Optional[pd.DataFrame]:
        """
        Load threshold analysis for a specific run.
        
        Args:
            run_id: MLflow run ID
            
        Returns:
            DataFrame with threshold analysis or None
        """
        try:
            artifact_path = _self.config.mlruns_path / run_id[:2] / run_id / "artifacts"
            threshold_path = artifact_path / "threshold_analysis.csv"
            
            if threshold_path.exists():
                return pd.read_csv(threshold_path)
        except Exception as e:
            st.error(f"Erro ao carregar análise de threshold: {e}")
        return None
    
    @staticmethod
    def generate_synthetic_data(data_type: str) -> pd.DataFrame:
        """
        Generate synthetic data for demonstration.
        
        Args:
            data_type: Type of data to generate
            
        Returns:
            DataFrame with synthetic data
        """
        import numpy as np
        from datetime import datetime, timedelta
        
        if data_type == "equity_curve":
            dates = pd.date_range(start="2023-01-01", end="2024-01-01", freq="D")
            returns = np.random.randn(len(dates)) * 0.02
            equity = 100000 * (1 + returns).cumprod()
            return pd.DataFrame({"date": dates, "equity": equity, "returns": returns})
        
        elif data_type == "feature_importance":
            features = [
                "volatility_20", "rsi_14", "macd_diff", "bb_position_20",
                "volume_ratio", "zscore_50", "momentum_10", "atr_14",
                "vwap_distance_20", "sma_cross_20_50", "stoch_k", "adx",
                "cci_20", "obv_momentum_20", "high_vol_regime"
            ]
            importances = np.random.exponential(scale=0.1, size=len(features))
            importances = importances / importances.sum()
            return pd.DataFrame({
                "feature": features,
                "importance": importances
            }).sort_values("importance", ascending=False)
        
        elif data_type == "model_comparison":
            return pd.DataFrame({
                "model": ["XGBoost", "LSTM", "Random Forest", "Logistic Regression"],
                "f1_score": [0.68, 0.65, 0.62, 0.58],
                "pr_auc": [0.72, 0.69, 0.66, 0.61],
                "sharpe_ratio": [1.85, 1.72, 1.45, 1.23],
                "max_drawdown": [-0.123, -0.145, -0.178, -0.201],
                "win_rate": [0.587, 0.562, 0.534, 0.512]
            })
        
        elif data_type == "regime_analysis":
            return pd.DataFrame({
                "regime": [
                    "Alta Volatilidade", "Baixa Volatilidade",
                    "Tendência Alta", "Tendência Baixa",
                    "Lateralização", "Alto Momentum"
                ],
                "sharpe_ratio": [1.2, 2.3, 2.5, 0.8, 1.1, 2.8],
                "win_rate": [0.52, 0.65, 0.68, 0.45, 0.55, 0.72],
                "avg_return": [0.08, 0.12, 0.15, -0.03, 0.05, 0.18],
                "frequency": [0.15, 0.25, 0.20, 0.10, 0.20, 0.10]
            })
        
        return pd.DataFrame()
