# %% [markdown]
# # IC Crypto Complete - Pipeline ML para Trading de Criptomoedas
#
# Pipeline completo de Machine Learning para trading de criptomoedas com:
# - Labeling adaptativo baseado em volatilidade
# - Múltiplos horizontes de predição (15m, 30m, 60m, 120m)
# - Features específicas para mercado 24/7
# - Backtest realista com custos e execução t+1
# - Otimização Bayesiana com Optuna
# - Calibração de probabilidades
# - MLflow tracking

# %% [markdown]
# ## 1. Setup Completo - Imports e Configurações

# %%
# ========================== IMPORTS ORGANIZADOS ==========================

# Standard Library
import os
import sys
import gc
import json
import pickle
import random
import hashlib
import warnings
import logging
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Union, Any

# Configurar warnings
warnings.filterwarnings('ignore')

# Data Science
import numpy as np
import pandas as pd
from scipy import stats

# Machine Learning - Scikit-learn
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, matthews_corrcoef,
    confusion_matrix, classification_report, roc_curve,
    precision_recall_curve, brier_score_loss, balanced_accuracy_score
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

# XGBoost
try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    print("⚠️ XGBoost não disponível")

# Optuna
OPTUNA_AVAILABLE = False
OPTUNA_XGBOOST_INTEGRATION = False

try:
    import optuna
    from optuna.pruners import MedianPruner, SuccessiveHalvingPruner, HyperbandPruner
    from optuna.samplers import TPESampler
    OPTUNA_AVAILABLE = True
    print("✅ Optuna core disponível")
    
    # Tentar carregar integração XGBoost separadamente
    try:
        from optuna.integration import XGBoostPruningCallback
        OPTUNA_XGBOOST_INTEGRATION = True
        print("✅ Optuna XGBoost integration disponível")
    except ImportError as e:
        print(f"⚠️ Optuna XGBoost integration não disponível: {e}")
        print("   Pipeline continuará com otimização básica")
        
except ImportError as e:
    print(f"⚠️ Optuna não disponível: {e}")
    print("   Pipeline continuará com hiperparâmetros padrão")

# Deep Learning - PyTorch
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
    
    # Configurar determinismo do PyTorch
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
except ImportError:
    TORCH_AVAILABLE = False
    print("⚠️ PyTorch não disponível")

# MLflow
try:
    import mlflow
    import mlflow.xgboost
    import mlflow.pytorch
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    print("⚠️ MLflow não disponível")

# SHAP
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("⚠️ SHAP não disponível")

# Data APIs
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False
    print("⚠️ yfinance não disponível")

try:
    import ccxt
    CCXT_AVAILABLE = True
except ImportError:
    CCXT_AVAILABLE = False
    print("⚠️ CCXT não disponível")

# Technical Analysis
try:
    import ta
    TA_AVAILABLE = True
except ImportError:
    TA_AVAILABLE = False
    print("⚠️ TA-Lib não disponível")

# Data Validation
try:
    import pandera as pa
    from pandera import Column, DataFrameSchema, Check
    PANDERA_AVAILABLE = True
except ImportError:
    PANDERA_AVAILABLE = False
    print("⚠️ Pandera não disponível")

# Visualização
import matplotlib.pyplot as plt
import seaborn as sns
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("⚠️ Plotly não disponível")

# Imports locais do projeto (quando disponíveis)
try:
    from src.data.binance_loader import CryptoDataLoader
    from src.data.splits import PurgedKFold as ImportedPurgedKFold
    from src.features.engineering import FeatureEngineer as BaseFeatureEngineer
    from src.models.xgb_optuna import XGBoostOptuna as ImportedXGBoostOptuna
    from src.backtest.engine import BacktestEngine as ImportedBacktestEngine, BacktestConfig
    LOCAL_IMPORTS_AVAILABLE = True
except ImportError:
    LOCAL_IMPORTS_AVAILABLE = False
    print("⚠️ Imports locais não disponíveis - usando implementações do notebook")

# Configurações de visualização
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)
pd.set_option('display.float_format', '{:.6f}'.format)

print("✅ Imports concluídos com sucesso!")

# %% [markdown]
# ## 2. Fallbacks para Bibliotecas Opcionais

# %%
class FallbackPruningCallback:
    """Fallback para XGBoostPruningCallback quando optuna-integration não disponível"""
    def __init__(self, trial, metric_name):
        self.trial = trial
        self.metric_name = metric_name
        
    def __call__(self, env):
        """Callback compatível com XGBoost"""
        # Extrair métrica de validação
        if hasattr(env, 'evaluation_result_list') and env.evaluation_result_list:
            # Formato: [('validation_0', 'aucpr', value)]
            for eval_name, metric, value in env.evaluation_result_list:
                if metric in self.metric_name or self.metric_name in metric:
                    self.trial.report(value, env.iteration)
                    if self.trial.should_prune():
                        raise optuna.TrialPruned()
        return False

def create_xgb_pruning_callback(trial, metric_name):
    """Factory function para criar callback apropriado"""
    if OPTUNA_XGBOOST_INTEGRATION:
        from optuna.integration import XGBoostPruningCallback
        return XGBoostPruningCallback(trial, metric_name)
    elif OPTUNA_AVAILABLE:
        return FallbackPruningCallback(trial, metric_name)
    else:
        # Retorna callback dummy que não faz nada
        return lambda env: False

def safe_optuna_study(direction='maximize', sampler=None, pruner=None):
    """Criar study do Optuna com fallbacks"""
    if OPTUNA_AVAILABLE:
        return optuna.create_study(
            direction=direction,
            sampler=sampler or optuna.samplers.TPESampler(seed=42),
            pruner=pruner or optuna.pruners.MedianPruner()
        )
    else:
        # Fallback: usar grid search simples
        print("⚠️ Usando grid search básico no lugar do Optuna")
        return None

def safe_mlflow_log(func_name, *args, **kwargs):
    """Helper para fazer log no MLflow de forma segura"""
    if MLFLOW_AVAILABLE:
        import mlflow
        func = getattr(mlflow, func_name)
        return func(*args, **kwargs)
    return None

def safe_mlflow_start_run(**kwargs):
    """Helper para iniciar run do MLflow de forma segura"""
    if MLFLOW_AVAILABLE:
        import mlflow
        return mlflow.start_run(**kwargs)
    return None

def safe_mlflow_end_run():
    """Helper para finalizar run do MLflow de forma segura"""
    if MLFLOW_AVAILABLE:
        import mlflow
        return mlflow.end_run()
    return None

print("✅ Fallbacks configurados!")

# %% [markdown]
# ## 3. Configuração Determinística do Ambiente

# %%
def setup_deterministic_environment(seed: int = 42):
    """
    Configura ambiente para reprodutibilidade total
    """
    # Python built-in
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # NumPy
    np.random.seed(seed)
    
    # Scikit-learn (se aplicável)
    os.environ['SKLEARN_SEED'] = str(seed)
    
    # PyTorch (se disponível)
    if TORCH_AVAILABLE:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            # Para operações determinísticas em GPU
            os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
            os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    
    # TensorFlow (se disponível)
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
        os.environ['TF_DETERMINISTIC_OPS'] = '1'
        os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    except ImportError:
        pass
    
    print(f"✅ Ambiente configurado para determinismo com seed={seed}")
    print(f"   PYTHONHASHSEED={os.environ.get('PYTHONHASHSEED', 'not set')}")
    if TORCH_AVAILABLE and torch.cuda.is_available():
        print(f"   CUDA determinístico: {torch.backends.cudnn.deterministic}")
        print(f"   CUBLAS_WORKSPACE_CONFIG={os.environ.get('CUBLAS_WORKSPACE_CONFIG', 'not set')}")
    
    return seed

# Aplicar configuração determinística
GLOBAL_SEED = setup_deterministic_environment(42)

# %% [markdown]
# ## 3. Configurações Globais do Projeto

# %%
from dataclasses import dataclass, field

@dataclass
class ProjectConfig:
    """Configurações globais do projeto"""
    
    # Paths
    data_path: str = "data"
    artifacts_path: str = "artifacts"
    models_path: str = "artifacts/models"
    reports_path: str = "artifacts/reports"
    mlflow_tracking_uri: str = "artifacts/mlruns"
    
    # Data
    symbol: str = "BTCUSDT"
    timeframe: str = "15m"
    start_date: str = "2023-01-01"
    end_date: str = "2024-01-01"
    
    # Funding periods por símbolo (em minutos)
    # Fonte: https://www.binance.com/en/support/faq/detail/360033525031
    # Padrão: 480 min (8h) para maioria dos contratos perpétuos
    # Alguns contratos específicos usam 60 min (1h) ou 240 min (4h)
    FUNDING_MIN_BY_SYMBOL: Dict[str, int] = field(default_factory=lambda: {
        "BTCUSDT": 480,    # 8 horas
        "ETHUSDT": 480,    # 8 horas  
        "BNBUSDT": 480,    # 8 horas
        "SOLUSDT": 480,    # 8 horas
        "XRPUSDT": 480,    # 8 horas
        "ADAUSDT": 480,    # 8 horas
        # Adicionar outros símbolos conforme necessário
        # Alguns contratos exóticos podem ter 60 ou 240 minutos
    })
    
    # Horizontes de predição (em barras de 15min)
    horizons: Dict[str, int] = None
    
    # Features
    feature_windows: List[int] = None
    volatility_estimators: List[str] = None
    
    # Model
    test_size: float = 0.2
    val_size: float = 0.2
    cv_splits: int = 5
    embargo_bars: int = 10
    
    # Trading
    initial_capital: float = 100000
    fee_bps: float = 5  # basis points
    slippage_bps: float = 10
    max_leverage: float = 1.0
    funding_period_minutes: int = 480  # período de funding em minutos (8 horas por padrão)
    
    # Optimization
    n_trials_optuna: int = 50
    optuna_timeout: int = 3600  # seconds
    
    # MLflow
    experiment_name: str = "crypto_ml_trading"
    
    def __post_init__(self):
        """Inicializar valores padrão para campos mutáveis"""
        if self.horizons is None:
            # Calcular horizonte de funding dinamicamente
            funding_horizon_bars = self.funding_period_minutes // 15  # converter para barras de 15min
            self.horizons = {
                '15m': 1,   # 15 minutos
                '30m': 2,   # 30 minutos  
                '60m': 4,   # 1 hora
                '120m': 8,  # 2 horas
                '240m': 16, # 4 horas
                f'{self.funding_period_minutes}m': funding_horizon_bars  # funding cycle dinâmico
            }
        
        if self.feature_windows is None:
            self.feature_windows = [5, 10, 20, 50, 100, 200]
        
        if self.volatility_estimators is None:
            self.volatility_estimators = ['atr', 'garman_klass', 'yang_zhang', 'parkinson']
    
    def create_directories(self):
        """Criar estrutura de diretórios"""
        for path in [self.data_path, self.artifacts_path, self.models_path, self.reports_path]:
            Path(path).mkdir(parents=True, exist_ok=True)
        print("✅ Diretórios criados")

# Instanciar configuração global
config = ProjectConfig()
config.create_directories()

# Configurar MLflow
if MLFLOW_AVAILABLE:
    import mlflow
    mlflow.set_tracking_uri(config.mlflow_tracking_uri)
    mlflow.set_experiment(config.experiment_name)
    print(f"✅ MLflow configurado: {config.mlflow_tracking_uri}")

# %% [markdown]
# ## 4. Classes de Estimadores de Volatilidade

# %%
class VolatilityEstimators:
    """
    Implementação de diversos estimadores de volatilidade para mercados 24/7
    Referência: Sinclair (2008) - Volatility Trading
    """
    
    @staticmethod
    def atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
        """Average True Range - robusto para gaps"""
        high = df['high']
        low = df['low']
        close = df['close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=window).mean()
        
        # Normalizar como proporção do preço (retorno implícito)
        return atr / close
    
    @staticmethod
    def garman_klass(df: pd.DataFrame, window: int = 14) -> pd.Series:
        """
        Garman-Klass estimator (1980)
        Usa OHLC, ~8x mais eficiente que close-to-close
        """
        log_hl = np.log(df['high'] / df['low'])
        log_co = np.log(df['close'] / df['open'])
        
        gk = np.sqrt(
            0.5 * log_hl**2 - 
            (2 * np.log(2) - 1) * log_co**2
        )
        
        # Normalizar para escala comparável ao ATR (retorno fracionário)
        gk_mean = gk.rolling(window=window).mean()
        return gk_mean.clip(lower=1e-8)  # Evitar divisão por zero
    
    @staticmethod  
    def yang_zhang(df: pd.DataFrame, window: int = 14) -> pd.Series:
        """
        Yang-Zhang estimator (2000)
        Melhor estimador para drift e gaps
        """
        log_ho = np.log(df['high'] / df['open'])
        log_lo = np.log(df['low'] / df['open'])
        log_co = np.log(df['close'] / df['open'])
        
        log_oc = np.log(df['open'] / df['close'].shift())
        log_oc_mean = log_oc.rolling(window=window).mean()
        
        log_cc = np.log(df['close'] / df['close'].shift())
        log_cc_mean = log_cc.rolling(window=window).mean()
        
        # Volatilidade overnight
        vol_overnight = (log_oc - log_oc_mean)**2
        vol_overnight = vol_overnight.rolling(window=window).mean()
        
        # Volatilidade close-to-close
        vol_cc = (log_cc - log_cc_mean)**2
        vol_cc = vol_cc.rolling(window=window).mean()
        
        # Volatilidade Rogers-Satchell
        rs = log_ho * (log_ho - log_co) + log_lo * (log_lo - log_co)
        vol_rs = rs.rolling(window=window).mean()
        
        # Combinar com pesos ótimos
        k = 0.34 / (1.34 + (window + 1) / (window - 1))
        yz = np.sqrt(vol_overnight + k * vol_cc + (1 - k) * vol_rs)
        
        # Normalizar para escala comparável (retorno fracionário)
        # Yang-Zhang já está em escala de log-retorno, clipar para evitar zero
        return yz.clip(lower=1e-8)
    
    @staticmethod
    def parkinson(df: pd.DataFrame, window: int = 14) -> pd.Series:
        """
        Parkinson estimator (1980)
        Usa high-low, ~5x mais eficiente que close-to-close
        """
        log_hl = np.log(df['high'] / df['low'])
        park = log_hl / (2 * np.sqrt(np.log(2)))
        
        # Normalizar para escala comparável (retorno fracionário)
        park_mean = park.rolling(window=window).mean()
        return park_mean.clip(lower=1e-8)  # Evitar divisão por zero
    
    @staticmethod
    def realized_volatility(df: pd.DataFrame, window: int = 14) -> pd.Series:
        """Volatilidade realizada clássica"""
        returns = np.log(df['close'] / df['close'].shift())
        return returns.rolling(window=window).std()

print("✅ Classe VolatilityEstimators definida")

# %% [markdown]
# ## 5. Sistema de Labeling Adaptativo

# %%
class AdaptiveLabeler:
    """
    Sistema de rotulagem adaptativo baseado em volatilidade
    Sistema mais robusto e interpretável para mercados 24/7
    Suporta múltiplos horizontes alinhados com timeframe de 15m
    """
    
    def __init__(self, 
                 horizon_bars: int = 4,  # 1h em dados de 15min
                 k: float = 1.0,  # Multiplicador do threshold
                 vol_estimator: str = 'atr',  # Estimador de volatilidade
                 neutral_zone: bool = True):  # Usar zona neutra
        """
        Args:
            horizon_bars: Janela futura para calcular retorno
            k: Multiplicador do threshold (hiperparâmetro a otimizar)
            vol_estimator: 'atr', 'garman_klass', 'yang_zhang', 'parkinson'
            neutral_zone: Se True, cria zona morta entre thresholds
        """
        self.horizon_bars = horizon_bars
        self.k = k
        self.vol_estimator = vol_estimator
        self.neutral_zone = neutral_zone
        self.volatility_estimators = VolatilityEstimators()
        
        # Mapeamento de horizontes em minutos para bars de 15m
        # Calcular horizonte de funding dinamicamente
        funding_period_minutes = getattr(self, 'funding_period_minutes', 480)
        funding_horizon_bars = funding_period_minutes // 15  # converter para barras de 15min
        
        self.horizon_map = {
            '15m': 1,   # 15 minutos = 1 bar
            '30m': 2,   # 30 minutos = 2 bars
            '60m': 4,   # 60 minutos = 4 bars
            '120m': 8,  # 120 minutos = 8 bars
            '240m': 16, # 240 minutos = 16 bars
            f'{funding_period_minutes}m': funding_horizon_bars  # funding cycle dinâmico
        }
    
    def calculate_volatility(self, df: pd.DataFrame, window: int = 20) -> pd.Series:
        """Calcula volatilidade usando estimador selecionado"""
        estimator_map = {
            'atr': self.volatility_estimators.atr,
            'garman_klass': self.volatility_estimators.garman_klass,
            'yang_zhang': self.volatility_estimators.yang_zhang,
            'parkinson': self.volatility_estimators.parkinson,
            'realized': self.volatility_estimators.realized_volatility
        }
        
        if self.vol_estimator not in estimator_map:
            raise ValueError(f"Estimador {self.vol_estimator} não suportado")
        
        return estimator_map[self.vol_estimator](df, window)
    
    def calculate_adaptive_threshold(self, df: pd.DataFrame, 
                                    window: int = 20) -> pd.Series:
        """
        Calcula threshold adaptativo baseado em volatilidade
        
        Returns:
            Series com threshold adaptativo para cada barra
        """
        volatility = self.calculate_volatility(df, window)
        
        # Ajustar threshold baseado na volatilidade e horizonte
        # Horizonte maior = threshold maior
        horizon_adjustment = np.sqrt(self.horizon_bars)
        
        threshold = self.k * volatility * horizon_adjustment
        
        # Aplicar limite mínimo e máximo
        threshold = threshold.clip(lower=0.001, upper=0.10)
        
        return threshold
    
    def create_labels(self, df: pd.DataFrame, window: int = 20) -> pd.Series:
        """
        Cria labels baseados em threshold adaptativo
        
        Returns:
            Series com labels: 1 (long), 0 (neutral), -1 (short)
        """
        # Calcular retorno futuro
        future_return = (
            df['close'].shift(-self.horizon_bars) / df['close'] - 1
        )
        
        # Calcular threshold adaptativo
        threshold = self.calculate_adaptive_threshold(df, window)
        
        # Criar labels
        labels = pd.Series(index=df.index, dtype=float)
        
        if self.neutral_zone:
            # Com zona neutra: -1, 0, 1
            labels[future_return > threshold] = 1  # Long
            labels[future_return < -threshold] = -1  # Short  
            labels[(future_return >= -threshold) & (future_return <= threshold)] = 0  # Neutral
        else:
            # Sem zona neutra: -1, 1
            labels[future_return > 0] = 1  # Long
            labels[future_return <= 0] = -1  # Short
        
        return labels
    
    def get_label_distribution(self, labels: pd.Series) -> Dict:
        """Retorna distribuição dos labels"""
        counts = labels.value_counts()
        proportions = labels.value_counts(normalize=True)
        
        return {
            'counts': counts.to_dict(),
            'proportions': proportions.to_dict(),
            'total': len(labels.dropna()),
            'balance_ratio': counts.min() / counts.max() if len(counts) > 0 else 0
        }
    
    def optimize_k_for_horizon(self, df: pd.DataFrame, X: pd.DataFrame,
                               horizon: str, cv_splits: int = 3,
                               metric: str = 'f1',
                               k_range: Tuple[float, float] = (0.5, 2.0)) -> float:
        """
        Otimiza o multiplicador k para um horizonte específico
        
        Args:
            df: DataFrame com OHLC
            X: Features
            horizon: Horizonte alvo ('15m', '30m', etc)
            cv_splits: Número de splits para validação
            metric: Métrica para otimização ('f1', 'pr_auc')
            k_range: Range de valores de k para testar
            
        Returns:
            float: k ótimo para o horizonte
        """
        # Imports já feitos no início do arquivo, não precisam ser repetidos
        
        # Configurar horizonte
        self.horizon_bars = self.horizon_map[horizon]
        
        best_k = self.k
        best_score = -np.inf
        
        # Testar diferentes valores de k
        k_values = np.linspace(k_range[0], k_range[1], 20)
        
        for k in k_values:
            self.k = k
            
            # Criar labels com k atual
            labels = self.create_labels(df)
            
            # Remover NaN
            mask = ~(labels.isna() | X.isna().any(axis=1))
            X_clean = X[mask]
            y_clean = labels[mask]
            
            # Converter para binário se necessário
            if metric in ['f1', 'pr_auc']:
                y_clean = (y_clean > 0).astype(int)
            
            # Validação cruzada temporal
            tscv = TimeSeriesSplit(n_splits=cv_splits)
            scores = []
            
            for train_idx, val_idx in tscv.split(X_clean):
                X_train, X_val = X_clean.iloc[train_idx], X_clean.iloc[val_idx]
                y_train, y_val = y_clean.iloc[train_idx], y_clean.iloc[val_idx]
                
                # Modelo simples para avaliação rápida
                model = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
                model.fit(X_train, y_train)
                
                if metric == 'f1':
                    y_pred = model.predict(X_val)
                    score = f1_score(y_val, y_pred, average='weighted')
                elif metric == 'pr_auc':
                    y_proba = model.predict_proba(X_val)[:, 1]
                    score = average_precision_score(y_val, y_proba)
                else:
                    raise ValueError(f"Métrica {metric} não suportada")
                
                scores.append(score)
            
            avg_score = np.mean(scores)
            
            if avg_score > best_score:
                best_score = avg_score
                best_k = k
            
            print(f"k={k:.2f}: {metric}={avg_score:.4f}")
        
        print(f"✅ k ótimo para {horizon}: {best_k:.3f}")
        
        return best_k
    
    def optimize_k_multi_horizon(self, df: pd.DataFrame, X: pd.DataFrame,
                                 horizons: List[str] = None,
                                 cv_splits: int = 3,
                                 metric: str = 'pr_auc') -> Dict:
        """
        Otimiza k para múltiplos horizontes simultaneamente
        
        Args:
            df: DataFrame com OHLC
            X: Features
            horizons: Lista de horizontes para otimizar
            cv_splits: Número de splits para CV
            metric: Métrica para otimização ('f1', 'pr_auc')
            
        Returns:
            Dict com k ótimo para cada horizonte
        """
        if horizons is None:
            horizons = ['15m', '30m', '60m', '120m']
        
        results = {}
        
        for horizon in horizons:
            print(f"\nOtimizando k para horizonte {horizon}...")
            optimal_k = self.optimize_k_for_horizon(
                df, X, horizon, cv_splits, metric
            )
            results[horizon] = optimal_k
        
        return results
    
    def optimize_k(self, df: pd.DataFrame, X: pd.DataFrame, 
                   cv_splits: int = 5, metric: str = 'f1') -> float:
        """
        Otimiza o multiplicador k usando validação cruzada temporal
        
        Args:
            df: DataFrame com OHLC
            X: Features para treino
            cv_splits: Número de splits temporais
            metric: 'f1' ou 'balanced_accuracy'
            
        Returns:
            float: k ótimo
        """
        # Imports já feitos no início do arquivo, não precisam ser repetidos
        
        best_k = self.k
        best_score = -np.inf
        
        # Range de k para testar
        k_values = np.linspace(0.5, 2.0, 20)
        
        for k in k_values:
            self.k = k
            
            # Criar labels com k atual
            labels = self.create_labels(df)
            
            # Remover NaN
            mask = ~(labels.isna() | X.isna().any(axis=1))
            X_clean = X[mask]
            y_clean = labels[mask]
            
            # Converter para binário (up/down)
            y_binary = (y_clean > 0).astype(int)
            
            # Time Series CV
            tscv = TimeSeriesSplit(n_splits=cv_splits)
            scores = []
            
            for train_idx, val_idx in tscv.split(X_clean):
                X_train, X_val = X_clean.iloc[train_idx], X_clean.iloc[val_idx]
                y_train, y_val = y_binary.iloc[train_idx], y_binary.iloc[val_idx]
                
                # Modelo simples para teste rápido
                clf = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_val)
                
                if metric == 'f1':
                    score = f1_score(y_val, y_pred, average='weighted')
                else:
                    score = balanced_accuracy_score(y_val, y_pred)
                
                scores.append(score)
            
            avg_score = np.mean(scores)
            
            if avg_score > best_score:
                best_score = avg_score
                best_k = k
            
            print(f"k={k:.2f}: {metric}={avg_score:.4f}")
        
        self.k = best_k
        return best_k

print("✅ Classe AdaptiveLabeler definida")

# %% [markdown]
# ## 6. Features para Mercado Cripto 24/7

# %%
# Resolver dinâmico de funding period
def resolve_funding_minutes(symbol: str, timestamp: pd.Timestamp = None) -> int:
    """
    Resolve o período de funding dinamicamente baseado no símbolo e data.
    
    Regras (fonte: Binance Support):
    - Padrão: 480 min (8 horas) para maioria dos contratos perpétuos
    - Condicional: 60 min (1 hora) quando funding rate atinge cap/floor
    - Especial: Alguns contratos podem ter 240 min (4 horas)
    
    Args:
        symbol: Símbolo do contrato (ex: BTCUSDT)
        timestamp: Data/hora para verificar regras específicas do período
        
    Returns:
        Período de funding em minutos
    """
    # Dicionário base de funding por símbolo
    # Fonte: https://www.binance.com/en/support/faq/detail/360033525031
    FUNDING_PERIODS = {
        # Majors - 8 horas padrão
        "BTCUSDT": 480,
        "ETHUSDT": 480,
        "BNBUSDT": 480,
        "XRPUSDT": 480,
        "ADAUSDT": 480,
        "SOLUSDT": 480,
        "DOTUSDT": 480,
        "DOGEUSDT": 480,
        
        # Contratos com período especial (exemplos)
        # Adicionar conforme documentação da exchange
    }
    
    # Obter período base do símbolo
    base_period = FUNDING_PERIODS.get(symbol, 480)  # Default 8h
    
    # TODO: Implementar lógica condicional
    # Se funding rate atingir cap/floor, pode mudar para 1h temporariamente
    # Isso requer acesso ao funding rate atual da exchange
    
    # Log da decisão
    if timestamp:
        print(f"📊 Funding period para {symbol} em {timestamp}: {base_period} min ({base_period/60:.1f}h)")
    
    return base_period

class Crypto24x7Features:
    """
    Features específicas para mercado cripto 24/7
    Inclui calendário, sessões regionais e funding
    """
    
    @staticmethod
    def create_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Cria features de calendário 24/7
        
        Crypto não tem fechamento, mas tem padrões:
        - Horários de maior volume (overlaps de mercados)
        - Dias da semana
        - Fim de mês (rebalanceamento de portfolios)
        """
        features = pd.DataFrame(index=df.index)
        
        # Extrair componentes temporais
        features['hour'] = df.index.hour
        features['day_of_week'] = df.index.dayofweek
        features['day_of_month'] = df.index.day
        features['week_of_year'] = df.index.isocalendar().week
        features['month'] = df.index.month
        features['quarter'] = df.index.quarter
        
        # Features cíclicas (encoding circular)
        features['hour_sin'] = np.sin(2 * np.pi * features['hour'] / 24)
        features['hour_cos'] = np.cos(2 * np.pi * features['hour'] / 24)
        features['dow_sin'] = np.sin(2 * np.pi * features['day_of_week'] / 7)
        features['dow_cos'] = np.cos(2 * np.pi * features['day_of_week'] / 7)
        features['month_sin'] = np.sin(2 * np.pi * features['month'] / 12)
        features['month_cos'] = np.cos(2 * np.pi * features['month'] / 12)
        
        # Períodos especiais
        features['is_weekend'] = (features['day_of_week'] >= 5).astype(int)
        features['is_month_end'] = (df.index.day >= 28).astype(int)
        features['is_quarter_end'] = ((features['month'] % 3 == 0) & 
                                      (features['is_month_end'] == 1)).astype(int)
        
        # Horário combinado (0-167 para hora da semana)
        features['hour_of_week'] = features['day_of_week'] * 24 + features['hour']
        
        return features
    
    @staticmethod
    def create_session_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Identifica sessões de trading regionais
        
        Principais sessões (UTC):
        - Asia: 00:00 - 09:00
        - Europe: 07:00 - 16:00  
        - Americas: 13:00 - 22:00
        """
        features = pd.DataFrame(index=df.index)
        hour = df.index.hour
        
        # Sessões principais
        features['session_asia'] = ((hour >= 0) & (hour < 9)).astype(int)
        features['session_europe'] = ((hour >= 7) & (hour < 16)).astype(int)
        features['session_americas'] = ((hour >= 13) & (hour < 22)).astype(int)
        
        # Overlaps (maior volume/volatilidade)
        features['overlap_asia_europe'] = ((hour >= 7) & (hour < 9)).astype(int)
        features['overlap_europe_americas'] = ((hour >= 13) & (hour < 16)).astype(int)
        
        # Contagem de sessões ativas
        features['active_sessions'] = (
            features['session_asia'] + 
            features['session_europe'] + 
            features['session_americas']
        )
        
        # Período de baixa atividade
        features['low_activity'] = (features['active_sessions'] == 0).astype(int)
        
        return features
    
    @staticmethod
    def create_funding_features(df: pd.DataFrame, 
                               features: pd.DataFrame = None,
                               funding_period_minutes: int = 480) -> pd.DataFrame:
        """
        Features relacionadas ao funding rate (perpetual futures)
        
        Default é 480 minutos (8 horas) - padrão da maioria dos contratos perpetuais
        Alguns contratos específicos usam 60 minutos (1 hora) - ajustar por símbolo
        """
        if features is None:
            features = pd.DataFrame(index=df.index)
        else:
            features = features.copy()
        
        # Converter período de funding para barras (15min cada)
        funding_period_bars = funding_period_minutes // 15  # minutos / 15 = barras
        
        # Identificar proximidade ao funding
        hour = df.index.hour
        minute = df.index.minute
        
        # Minutos até próximo funding
        minutes_in_day = hour * 60 + minute
        
        # Gerar funding times dinamicamente baseado no período
        funding_times = list(range(0, 1440, funding_period_minutes))
        
        # Calcular minutos até próximo funding
        features['minutes_to_funding'] = [
            min(((ft - m) % 1440) for ft in funding_times) 
            for m in minutes_in_day
        ]
        
        features['bars_to_funding'] = features['minutes_to_funding'] / 15
        
        # Proximidade ao funding (decai exponencialmente)
        features['funding_proximity'] = np.exp(-features['bars_to_funding'] / 10)
        
        # É hora de funding?
        features['is_funding_time'] = (features['minutes_to_funding'] == 0).astype(int)
        
        # Janela pré-funding (proporção do período - 12.5% antes do funding)
        pre_funding_minutes = min(60, funding_period_minutes // 8)  # Máximo 1 hora, ou 1/8 do período
        features['pre_funding_window'] = (features['minutes_to_funding'] <= pre_funding_minutes).astype(int)
        
        # Ciclo de funding (qual período estamos)
        features['funding_cycle'] = (minutes_in_day // funding_period_minutes).astype(int)
        
        # Features cíclicas para funding
        features['funding_cycle_sin'] = np.sin(2 * np.pi * features['bars_to_funding'] / funding_period_bars)
        features['funding_cycle_cos'] = np.cos(2 * np.pi * features['bars_to_funding'] / funding_period_bars)
        
        return features

print("✅ Classe Crypto24x7Features definida")# %% [markdown]
# ## 7. Pipeline Multi-Horizonte de Treinamento

# %%
def run_multi_horizon_pipeline(df: pd.DataFrame, 
                              features: pd.DataFrame,
                              horizons: List[str] = ['15m', '30m', '60m', '120m'],
                              test_size: float = 0.2,
                              val_size: float = 0.2,
                              n_trials: int = 20,  # Reduzido para otimização de memória
                              k_range: Tuple[float, float] = (0.5, 2.0)) -> Dict:
    """
    Pipeline completo para treinar e avaliar modelos em múltiplos horizontes
    
    Args:
        df: DataFrame com OHLC
        features: Features preparadas
        horizons: Lista de horizontes para avaliar
        test_size: Proporção para teste
        val_size: Proporção para validação  
        n_trials: Número de trials Optuna
        k_range: Range para otimização do k
        
    Returns:
        Dict com resultados para cada horizonte
    """
    print("="*80)
    print("🚀 INICIANDO PIPELINE MULTI-HORIZONTE")
    print("="*80)
    
    # Verificar disponibilidade de bibliotecas
    if not XGB_AVAILABLE:
        raise ImportError("XGBoost não está disponível")
    
    # Informar sobre status das bibliotecas opcionais
    if not OPTUNA_AVAILABLE:
        print("⚠️ Optuna não disponível - usando hiperparâmetros padrão")
    elif not OPTUNA_XGBOOST_INTEGRATION:
        print("⚠️ Optuna XGBoost integration não disponível - usando otimização básica")
    
    if not MLFLOW_AVAILABLE:
        print("⚠️ MLflow não disponível - resultados não serão tracked")
    
    if not SHAP_AVAILABLE:
        print("⚠️ SHAP não disponível - interpretabilidade limitada")
    
    # Estrutura para armazenar resultados
    results = {}
    
    # Configurar MLflow
    if MLFLOW_AVAILABLE:
        import mlflow
        experiment_name = f"multi_horizon_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"
        mlflow.set_experiment(experiment_name)
    
    # Split temporal dos dados
    n_samples = len(df)
    test_start = int(n_samples * (1 - test_size))
    val_start = int(n_samples * (1 - test_size - val_size))
    
    train_idx = slice(0, val_start)
    val_idx = slice(val_start, test_start)
    test_idx = slice(test_start, n_samples)
    
    print(f"\n📊 Split dos dados:")
    print(f"  Train: {val_start} samples ({val_start/n_samples:.1%})")
    print(f"  Val:   {test_start - val_start} samples ({val_size:.1%})")
    print(f"  Test:  {n_samples - test_start} samples ({test_size:.1%})")
    
    # Adicionar features de funding cycle parametrizado por símbolo
    crypto_features = Crypto24x7Features()
    
    # Resolver período de funding dinamicamente
    symbol = config.symbol if 'config' in locals() else "BTCUSDT"
    last_timestamp = df.index[-1] if not df.empty else None
    funding_period_minutes = resolve_funding_minutes(symbol, last_timestamp)
    
    # Log adicional se MLflow disponível  
    if MLFLOW_AVAILABLE:
        import mlflow
        mlflow.log_param("funding_period_minutes", funding_period_minutes)
        mlflow.log_param("funding_symbol", symbol)
    
    features_with_funding = crypto_features.create_funding_features(
        df, features, funding_period_minutes=funding_period_minutes
    )
    
    # Processar cada horizonte
    for horizon in horizons:
        print(f"\n{'='*60}")
        print(f"⏱️ Processando horizonte: {horizon}")
        print(f"{'='*60}")
        
        run_context = None
        if MLFLOW_AVAILABLE:
            import mlflow
            try:
                run_context = mlflow.start_run(run_name=f"horizon_{horizon}", nested=True)
            except Exception:
                # Se der erro com nested, tentar sem ou finalizar run anterior
                try:
                    mlflow.end_run()
                    run_context = mlflow.start_run(run_name=f"horizon_{horizon}")
                except Exception:
                    print("⚠️ Erro ao inicializar MLflow run - continuando sem tracking")
                    run_context = None
        
        try:
            # Log do horizonte
            if MLFLOW_AVAILABLE:
                import mlflow
                mlflow.log_param("horizon", horizon)
                mlflow.log_param("n_trials", n_trials)
                mlflow.log_param("k_range", k_range)
            
            # 1. Criar labels para este horizonte
            labeler = AdaptiveLabeler(vol_estimator='yang_zhang')
            horizon_bars = labeler.horizon_map[horizon]
            
            # Otimizar k para este horizonte
            print(f"\n🔍 Otimizando k para horizonte {horizon}...")
            optimal_k = labeler.optimize_k_for_horizon(
                df[train_idx], 
                features_with_funding[train_idx],
                horizon=horizon,
                cv_splits=3,
                metric='pr_auc'
            )
            
            if MLFLOW_AVAILABLE:
                import mlflow
                mlflow.log_metric(f"optimal_k_{horizon}", optimal_k)
            
            # Criar labels com k otimizado
            labeler.k = optimal_k
            labeler.horizon_bars = horizon_bars
            labels = labeler.create_labels(df)
            
            # 2. Preparar dados
            X_train = features_with_funding[train_idx]
            y_train = labels[train_idx]
            X_val = features_with_funding[val_idx]
            y_val = labels[val_idx]
            X_test = features_with_funding[test_idx]
            y_test = labels[test_idx]
            
            # Remover NaN
            mask_train = ~(X_train.isna().any(axis=1) | y_train.isna())
            mask_val = ~(X_val.isna().any(axis=1) | y_val.isna())
            mask_test = ~(X_test.isna().any(axis=1) | y_test.isna())
            
            X_train = X_train[mask_train]
            y_train = y_train[mask_train]
            X_val = X_val[mask_val]
            y_val = y_val[mask_val]
            X_test = X_test[mask_test]
            y_test = y_test[mask_test]
            
            # Converter labels para binário (1: up, 0: down/neutral)
            y_train_binary = (y_train > 0).astype(int)
            y_val_binary = (y_val > 0).astype(int)
            y_test_binary = (y_test > 0).astype(int)
            
            # Log distribuição das classes
            train_pos_pct = y_train_binary.mean()
            val_pos_pct = y_val_binary.mean()
            test_pos_pct = y_test_binary.mean()
            
            print(f"\n📈 Distribuição das classes:")
            print(f"  Train: {train_pos_pct:.2%} positivos")
            print(f"  Val:   {val_pos_pct:.2%} positivos")
            print(f"  Test:  {test_pos_pct:.2%} positivos")
            
            if MLFLOW_AVAILABLE:
                mlflow.log_metric("train_positive_pct", train_pos_pct)
                mlflow.log_metric("val_positive_pct", val_pos_pct)
            
            # 3. XGBoost não precisa de normalização (trees são invariantes à escala)
            # Manter dados originais para melhor interpretabilidade
            X_train_scaled = X_train
            X_val_scaled = X_val
            X_test_scaled = X_test
            scaler = None  # XGBoost não precisa
            
            # 4. Otimização com Optuna
            print(f"\n🎯 Otimizando XGBoost com Optuna...")
            
            def objective(trial):
                params = {
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                    'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                    'gamma': trial.suggest_float('gamma', 0.0, 0.5),
                    'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
                    'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 1.0),
                    'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                    'scale_pos_weight': ((1 - train_pos_pct) / train_pos_pct) if train_pos_pct > 0 else 1.0,
                    'objective': 'binary:logistic',
                    'eval_metric': 'aucpr',
                    'tree_method': 'hist',
                    'random_state': 42
                }
                
                # Treinar com early stopping e pruning
                model = xgb.XGBClassifier(**params)
                
                # Usar callback apropriado baseado na disponibilidade
                callbacks = []
                if OPTUNA_AVAILABLE:
                    callbacks.append(create_xgb_pruning_callback(trial, "validation_0-aucpr"))
                
                model.fit(
                    X_train_scaled, y_train_binary,
                    eval_set=[(X_val_scaled, y_val_binary)],
                    verbose=False,
                    early_stopping_rounds=200,
                    callbacks=callbacks if callbacks else None
                )
                
                # Salvar melhor iteração no trial
                if hasattr(model, 'best_iteration'):
                    trial.set_user_attr('best_iteration', model.best_iteration)
                
                # Avaliar com PR-AUC
                y_pred_proba = model.predict_proba(X_val_scaled)[:, 1]
                pr_auc = average_precision_score(y_val_binary, y_pred_proba)
                
                return pr_auc
            
            # Executar otimização
            if OPTUNA_AVAILABLE:
                study = safe_optuna_study(
                    direction='maximize',
                    sampler=optuna.samplers.TPESampler(seed=42),
                    pruner=optuna.pruners.MedianPruner()
                )
                
                study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
                
                # Melhores parâmetros
                best_params = study.best_params
                best_score = study.best_value
            else:
                # Fallback: usar hiperparâmetros padrão otimizados
                print("⚠️ Usando hiperparâmetros padrão (Optuna não disponível)")
                best_params = {
                    'max_depth': 6,
                    'learning_rate': 0.05,
                    'n_estimators': 200,
                    'subsample': 0.8,
                    'colsample_bytree': 0.8,
                    'gamma': 0.1,
                    'reg_alpha': 0.1,
                    'reg_lambda': 1.0,
                    'min_child_weight': 1,
                    'scale_pos_weight': ((1 - train_pos_pct) / train_pos_pct) if train_pos_pct > 0 else 1.0,
                    'objective': 'binary:logistic',
                    'eval_metric': 'aucpr',
                    'tree_method': 'hist',
                    'random_state': 42
                }
                
                # Avaliar hiperparâmetros padrão
                temp_model = xgb.XGBClassifier(**best_params)
                temp_model.fit(X_train_scaled, y_train_binary, eval_set=[(X_val_scaled, y_val_binary)], verbose=False, early_stopping_rounds=200)
                y_pred_proba = temp_model.predict_proba(X_val_scaled)[:, 1]
                best_score = average_precision_score(y_val_binary, y_pred_proba)
            
            # Recuperar melhor iteração se disponível (só quando Optuna usado)
            if OPTUNA_AVAILABLE and 'study' in locals():
                best_iteration = study.best_trial.user_attrs.get('best_iteration')
                if best_iteration is not None:
                    best_params['n_estimators'] = int(best_iteration)
                    print(f"📊 Usando melhor iteração do early stopping: {best_iteration}")
            
            best_params.update({
                'scale_pos_weight': ((1 - train_pos_pct) / train_pos_pct) if train_pos_pct > 0 else 1.0,
                'objective': 'binary:logistic',
                'eval_metric': 'aucpr',
                'tree_method': 'hist',
                'random_state': 42
            })
            
            print(f"\n✅ Melhor PR-AUC em validação: {best_score:.4f}")
            
            if MLFLOW_AVAILABLE:
                mlflow.log_metric(f"best_pr_auc_val_{horizon}", best_score)
                mlflow.log_params({f"xgb_{k}_{horizon}": v for k, v in best_params.items()})
            
            # 5. Treinar modelo final
            print(f"\n🏋️ Treinando modelo final...")
            final_model = xgb.XGBClassifier(**best_params)
            final_model.fit(
                X_train_scaled, y_train_binary,
                eval_set=[(X_val_scaled, y_val_binary)],
                verbose=False,
                early_stopping_rounds=200  # Manter early stopping no modelo final
            )
            
            # 6. Calibração de probabilidades
            print(f"\n📐 Calibrando probabilidades...")
            calibrator = CalibratedClassifierCV(
                final_model, 
                method='isotonic',
                cv='prefit'
            )
            calibrator.fit(X_val, y_val_binary)  # Usar dados originais
            
            # 7. Otimizar threshold no VALIDATION (não no teste!)
            print(f"\n🔍 Otimizando threshold no conjunto de validação...")
            
            # Predições calibradas no validation para escolher threshold
            y_val_pred_cal = calibrator.predict_proba(X_val)[:, 1]
            
            # Otimizar threshold baseado em F1 no VALIDATION
            precision, recall, thresholds = precision_recall_curve(
                y_val_binary, y_val_pred_cal
            )
            f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
            best_threshold_idx = np.argmax(f1_scores)
            best_threshold = thresholds[best_threshold_idx] if best_threshold_idx < len(thresholds) else 0.5
            
            print(f"  Threshold ótimo (do validation): {best_threshold:.4f}")
            
            # 8. Avaliação em teste com threshold fixo
            print(f"\n📊 Avaliando em conjunto de teste com threshold fixo...")
            
            # Predições não calibradas
            y_test_pred_raw = final_model.predict_proba(X_test)[:, 1]
            
            # Predições calibradas
            y_test_pred_cal = calibrator.predict_proba(X_test)[:, 1]
            
            # Aplicar threshold
            y_test_pred_binary = (y_test_pred_cal >= best_threshold).astype(int)
            
            # Métricas finais
            test_pr_auc = average_precision_score(y_test_binary, y_test_pred_cal)
            test_f1 = f1_score(y_test_binary, y_test_pred_binary)
            test_mcc = matthews_corrcoef(y_test_binary, y_test_pred_binary)
            
            # Matriz de confusão (com labels explícitos para evitar erros)
            cm = confusion_matrix(y_test_binary, y_test_pred_binary, labels=[0, 1])
            tn, fp, fn, tp = cm.ravel()
            
            # Métricas adicionais
            precision_score_val = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall_score_val = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            
            print(f"\n📈 Métricas em teste para {horizon}:")
            print(f"  PR-AUC:      {test_pr_auc:.4f}")
            print(f"  F1 Score:    {test_f1:.4f}")
            print(f"  MCC:         {test_mcc:.4f}")
            print(f"  Precision:   {precision_score_val:.4f}")
            print(f"  Recall:      {recall_score_val:.4f}")
            print(f"  Specificity: {specificity:.4f}")
            print(f"  Threshold:   {best_threshold:.4f}")
            
            # Log métricas no MLflow
            if MLFLOW_AVAILABLE:
                import mlflow
                mlflow.log_metrics({
                    f"test_pr_auc_{horizon}": test_pr_auc,
                    f"test_f1_{horizon}": test_f1,
                    f"test_mcc_{horizon}": test_mcc,
                    f"test_precision_{horizon}": precision_score_val,
                    f"test_recall_{horizon}": recall_score_val,
                    f"test_specificity_{horizon}": specificity,
                    f"best_threshold_{horizon}": best_threshold
                })
            
            # 8. Análise de importância de features
            feature_importance = pd.DataFrame({
                'feature': X_train.columns,
                'importance': final_model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            print(f"\n🔝 Top 10 features mais importantes:")
            for idx, row in feature_importance.head(10).iterrows():
                print(f"  {row['feature']:30s}: {row['importance']:.4f}")
            
            # Salvar resultados (incluindo índices para alinhamento no backtest)
            results[horizon] = {
                'model': final_model,
                'calibrator': calibrator,
                'scaler': scaler,
                'labeler': labeler,
                'threshold': best_threshold,
                'metrics': {
                    'pr_auc': test_pr_auc,
                    'f1': test_f1,
                    'mcc': test_mcc,
                    'precision': precision_score_val,
                    'recall': recall_score_val,
                    'specificity': specificity
                },
                'confusion_matrix': cm,
                'feature_importance': feature_importance,
                'predictions': {
                    'raw': y_test_pred_raw,
                    'calibrated': y_test_pred_cal,
                    'binary': y_test_pred_binary
                },
                'labels': y_test_binary,
                'optimal_k': optimal_k,
                'test_indices': X_test.index  # Salvar índices para backtest
            }
            
            # Limpeza de memória após cada horizonte
            del X_train, X_val, X_test, y_train, y_val, y_test
            del X_train_scaled, X_val_scaled, X_test_scaled
            del y_train_binary, y_val_binary, y_test_binary
            del y_test_pred_raw, y_test_pred_cal, y_test_pred_binary
            if 'temp_model' in locals():
                del temp_model
            gc.collect()
            print(f"🧹 Memória limpa após horizonte {horizon}")
            
            # Salvar modelo
            import joblib
            model_path = f"{config.models_path}/xgb_{horizon}_{experiment_name if MLFLOW_AVAILABLE else 'local'}.pkl"
            os.makedirs(config.models_path, exist_ok=True)
            joblib.dump({
                'model': final_model,
                'calibrator': calibrator,
                'scaler': scaler,
                'threshold': best_threshold
            }, model_path)
            
            if MLFLOW_AVAILABLE:
                import mlflow
                mlflow.log_artifact(model_path)
        
        finally:
            if MLFLOW_AVAILABLE and run_context:
                import mlflow
                mlflow.end_run()
    
    # 9. Análise comparativa entre horizontes
    print(f"\n{'='*80}")
    print("📊 ANÁLISE COMPARATIVA ENTRE HORIZONTES")
    print(f"{'='*80}")
    
    comparison_df = pd.DataFrame({
        horizon: {
            'PR-AUC': results[horizon]['metrics']['pr_auc'],
            'F1': results[horizon]['metrics']['f1'],
            'MCC': results[horizon]['metrics']['mcc'],
            'Precision': results[horizon]['metrics']['precision'],
            'Recall': results[horizon]['metrics']['recall'],
            'Optimal_k': results[horizon]['optimal_k']
        }
        for horizon in horizons
    }).T
    
    print("\n📈 Tabela Comparativa:")
    print(comparison_df.round(4))
    
    # Identificar melhor horizonte
    best_horizon_pr_auc = comparison_df['PR-AUC'].idxmax()
    best_horizon_f1 = comparison_df['F1'].idxmax()
    
    print(f"\n🏆 Melhores horizontes:")
    print(f"  Melhor PR-AUC: {best_horizon_pr_auc} ({comparison_df.loc[best_horizon_pr_auc, 'PR-AUC']:.4f})")
    print(f"  Melhor F1:     {best_horizon_f1} ({comparison_df.loc[best_horizon_f1, 'F1']:.4f})")
    
    # 10. Análise de correlação entre predições
    print(f"\n🔗 Correlação entre predições dos horizontes:")
    pred_matrix = pd.DataFrame({
        horizon: results[horizon]['predictions']['calibrated']
        for horizon in horizons
    })
    
    corr_matrix = pred_matrix.corr()
    print(corr_matrix.round(3))
    
    # Salvar comparação
    comparison_df.to_csv(f"{config.reports_path}/horizon_comparison_{experiment_name if MLFLOW_AVAILABLE else 'local'}.csv")
    
    # Limpeza final de memória
    del features_with_funding, pred_matrix, corr_matrix, comparison_df
    gc.collect()
    print(f"\n🧹 Pipeline finalizado - memória limpa")
    
    return results

print("✅ Função run_multi_horizon_pipeline definida")# %% [markdown]
# ## 8. Pipeline LSTM para Séries Temporais
# 
# Implementação de LSTM com:
# - Otimização Bayesiana via Optuna
# - PR-AUC como métrica alvo
# - Calibração isotônica
# - Threshold otimizado no validation
# - Export para produção via TorchScript

# %%
# Funções de preparação de dados sequenciais
def make_sequences(X_df: pd.DataFrame, y_series: pd.Series, seq_len: int):
    """
    Cria sequências para LSTM a partir de dados tabulares
    
    Args:
        X_df: Features
        y_series: Labels
        seq_len: Comprimento da sequência
        
    Returns:
        X_seq: Array de sequências [n_samples, seq_len, n_features]
        y_seq: Array de labels
        idx_seq: Índices pandas correspondentes
    """
    X = X_df.values.astype(np.float32)
    y = y_series.values.astype(np.int64)
    idx = X_df.index

    X_seq, y_seq, idx_seq = [], [], []
    for t in range(seq_len, len(X)):
        X_seq.append(X[t-seq_len:t])
        y_seq.append(y[t])
        idx_seq.append(idx[t])
    
    return np.array(X_seq), np.array(y_seq), pd.Index(idx_seq)

def train_val_test_split_time(X_df: pd.DataFrame, y: pd.Series, n_splits: int = 5):
    """
    Split temporal para treino/validação/teste
    
    Args:
        X_df: Features
        y: Labels
        n_splits: Número de splits para TimeSeriesSplit
        
    Returns:
        tr_idx: Índices de treino
        va_idx: Índices de validação
        te_idx: Índices de teste
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    splits = list(tscv.split(X_df))
    
    # Penúltimo split para validação
    (tr_idx, va_idx) = splits[-2]
    # Último split para teste
    (tr2_idx, te_idx) = splits[-1]
    
    # Treino = do início até fim do penúltimo split
    tr_idx_full = np.arange(0, va_idx[-1] + 1)
    
    return tr_idx_full, va_idx, te_idx

def build_lstm_tensors(X_df: pd.DataFrame, y: pd.Series, seq_len: int, 
                      tr_idx, va_idx, te_idx):
    """
    Prepara tensores para LSTM com escalonamento
    
    Args:
        X_df: Features
        y: Labels
        seq_len: Comprimento da sequência
        tr_idx, va_idx, te_idx: Índices dos splits
        
    Returns:
        Tupla com tensores de treino/val/test, índices e scaler
    """
    # Escalonar features com base no treino
    scaler = StandardScaler()
    X_train_df = X_df.iloc[tr_idx]
    scaler.fit(X_train_df.values)
    
    # Aplicar escalonamento
    Xs = pd.DataFrame(
        scaler.transform(X_df.values), 
        index=X_df.index, 
        columns=X_df.columns
    )
    
    # Gerar sequências
    X_seq, y_seq, idx_seq = make_sequences(Xs, y, seq_len)
    
    # Mapear índices originais para índices da sequência
    idx_map = pd.Series(range(len(idx_seq)), index=idx_seq)
    
    # Obter índices das sequências para cada split
    tr = idx_map[idx_seq.intersection(X_df.index[tr_idx])].dropna().astype(int).values
    va = idx_map[idx_seq.intersection(X_df.index[va_idx])].dropna().astype(int).values
    te = idx_map[idx_seq.intersection(X_df.index[te_idx])].dropna().astype(int).values
    
    # Separar tensores
    Xtr, ytr = X_seq[tr], y_seq[tr]
    Xva, yva = X_seq[va], y_seq[va]
    Xte, yte = X_seq[te], y_seq[te]
    
    return (Xtr, ytr, Xva, yva, Xte, yte, idx_seq[te], scaler)

# %%
# Modelo LSTM para classificação binária
if TORCH_AVAILABLE:
    class LSTMClassifier(nn.Module):
        """
        LSTM para classificação binária de séries temporais
        """
        def __init__(self, in_dim: int, hidden: int, layers: int, dropout: float):
            super().__init__()
            self.lstm = nn.LSTM(
                in_dim, hidden, 
                num_layers=layers, 
                batch_first=True, 
                dropout=dropout if layers > 1 else 0
            )
            self.head = nn.Linear(hidden, 1)
            
        def forward(self, x):
            # x: [batch, seq, feat]
            out, _ = self.lstm(x)
            # Usar apenas último timestep
            logit = self.head(out[:, -1, :])
            return logit.squeeze(1)  # Retorna logits (sem sigmoid)

# %%
# Funções de treinamento e avaliação
if TORCH_AVAILABLE:
    def train_one_epoch(model, optimizer, loss_fn, loader, device):
        """
        Treina modelo por uma época
        
        Args:
            model: Modelo LSTM
            optimizer: Otimizador
            loss_fn: Função de perda
            loader: DataLoader
            device: Device (cuda/cpu)
            
        Returns:
            Loss médio da época
        """
        model.train()
        total_loss = 0.0
        
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device).float()
            
            optimizer.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            
            # Gradient clipping para estabilidade
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            total_loss += loss.item() * xb.size(0)
            
        return total_loss / len(loader.dataset)
    
    @torch.no_grad()
    def eval_ap(model, loader, device):
        """
        Avalia modelo com Average Precision (PR-AUC)
        
        Args:
            model: Modelo LSTM
            loader: DataLoader
            device: Device
            
        Returns:
            ap: Average Precision score
            probs: Probabilidades preditas
            labels: Labels verdadeiros
        """
        model.eval()
        probs_list, labels_list = [], []
        
        for xb, yb in loader:
            xb = xb.to(device)
            logits = model(xb)
            probs = torch.sigmoid(logits).cpu().numpy()
            probs_list.append(probs)
            labels_list.append(yb.numpy())
        
        labels = np.concatenate(labels_list)
        probs = np.concatenate(probs_list)
        
        ap = average_precision_score(labels, probs)
        return ap, probs, labels

# %%
# Objetivo do Optuna para LSTM
if TORCH_AVAILABLE and OPTUNA_AVAILABLE:
    def objective_lstm(trial, X_train, y_train, X_val, y_val, seq_len, device):
        """
        Função objetivo para otimização com Optuna
        
        Args:
            trial: Trial do Optuna
            X_train, y_train: Dados de treino
            X_val, y_val: Dados de validação
            seq_len: Comprimento da sequência
            device: Device (cuda/cpu)
            
        Returns:
            Best Average Precision obtido
        """
        # Hiperparâmetros para otimizar
        hidden = trial.suggest_categorical("hidden", [64, 128, 256])
        layers = trial.suggest_int("layers", 1, 3)
        dropout = trial.suggest_float("dropout", 0.0, 0.4)
        lr = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
        wd = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
        batch_size = trial.suggest_categorical("batch_size", [64, 128, 256])
        
        # Criar modelo
        model = LSTMClassifier(
            X_train.shape[-1], hidden, layers, dropout
        ).to(device)
        
        # BCEWithLogitsLoss com peso para desbalanceamento
        pos_ratio = y_train.mean()
        pos_ratio = float(pos_ratio if pos_ratio > 0 else 1e-6)
        pos_weight = torch.tensor((1 - pos_ratio) / pos_ratio, device=device)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        
        # Otimizador
        optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=lr, 
            weight_decay=wd
        )
        
        # DataLoaders
        train_dataset = TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.float32)
        )
        val_dataset = TensorDataset(
            torch.tensor(X_val, dtype=torch.float32),
            torch.tensor(y_val, dtype=torch.float32)
        )
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            drop_last=True
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=512, 
            shuffle=False
        )
        
        # Treinar com early stopping via pruning
        best_ap = 0.0
        patience_counter = 0
        patience = 10
        
        for epoch in range(60):
            # Treinar
            train_loss = train_one_epoch(
                model, optimizer, loss_fn, train_loader, device
            )
            
            # Avaliar
            ap, _, _ = eval_ap(model, val_loader, device)
            
            # Reportar ao Optuna
            trial.report(ap, epoch)
            
            # Pruning
            if trial.should_prune():
                raise optuna.TrialPruned()
            
            # Track best
            if ap > best_ap:
                best_ap = ap
                patience_counter = 0
            else:
                patience_counter += 1
                
            # Early stopping
            if patience_counter >= patience:
                break
                
        return best_ap

# %%
# Pipeline de treinamento com Optuna
if TORCH_AVAILABLE and OPTUNA_AVAILABLE:
    def fit_lstm_with_optuna(Xtr, ytr, Xva, yva, n_trials=50, 
                            device="cuda" if torch.cuda.is_available() else "cpu"):
        """
        Treina LSTM com otimização Bayesiana
        
        Args:
            Xtr, ytr: Dados de treino
            Xva, yva: Dados de validação
            n_trials: Número de trials do Optuna
            device: Device para treino
            
        Returns:
            model: Modelo treinado
            best_params: Melhores hiperparâmetros
        """
        print(f"\n🔍 Otimização Bayesiana com Optuna ({n_trials} trials)")
        
        # Criar estudo
        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=42),
            pruner=optuna.pruners.MedianPruner()
        )
        
        # Otimizar
        study.optimize(
            lambda t: objective_lstm(t, Xtr, ytr, Xva, yva, Xtr.shape[1], device),
            n_trials=n_trials,
            show_progress_bar=True
        )
        
        best_params = study.best_params
        print(f"✅ Melhor AP em validação: {study.best_value:.4f}")
        print(f"📊 Melhores parâmetros: {best_params}")
        
        # Re-treinar com melhores parâmetros no conjunto completo (treino + val)
        print(f"\n🏋️ Treinando modelo final...")
        
        model = LSTMClassifier(
            Xtr.shape[-1], 
            best_params['hidden'],
            best_params['layers'],
            best_params['dropout']
        ).to(device)
        
        # Combinar treino e validação
        X_combined = np.concatenate([Xtr, Xva])
        y_combined = np.concatenate([ytr, yva])
        
        # Loss com peso para manter balanceamento de classes
        pos_ratio = y_combined.mean()
        pos_ratio = float(pos_ratio if pos_ratio > 0 else 1e-6)
        pos_weight = torch.tensor((1 - pos_ratio) / pos_ratio, device=device)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=best_params['lr'],
            weight_decay=best_params['weight_decay']
        )
        
        combined_dataset = TensorDataset(
            torch.tensor(X_combined, dtype=torch.float32),
            torch.tensor(y_combined, dtype=torch.float32)
        )
        
        combined_loader = DataLoader(
            combined_dataset,
            batch_size=best_params['batch_size'],
            shuffle=True,
            drop_last=True
        )
        
        # Treinar por mais épocas (1.5x)
        for epoch in range(90):
            train_loss = train_one_epoch(
                model, optimizer, loss_fn, combined_loader, device
            )
            if epoch % 10 == 0:
                print(f"  Época {epoch}: Loss = {train_loss:.4f}")
        
        return model, best_params

# %%
# Calibração e seleção de threshold
if TORCH_AVAILABLE:
    @torch.no_grad()
    def calibrate_and_choose_threshold(model, Xva, yva, device):
        """
        Calibra probabilidades e escolhe threshold ótimo
        
        Args:
            model: Modelo LSTM treinado
            Xva, yva: Dados de validação
            device: Device
            
        Returns:
            calibrator: Calibrador isotônico
            threshold: Threshold ótimo
        """
        from sklearn.isotonic import IsotonicRegression
        
        model.eval()
        
        # Obter probabilidades
        Xva_tensor = torch.tensor(Xva, dtype=torch.float32).to(device)
        logits = model(Xva_tensor)
        probs = torch.sigmoid(logits).cpu().numpy()
        
        # Calibração isotônica
        calibrator = IsotonicRegression(out_of_bounds='clip')
        probs_cal = calibrator.fit_transform(probs, yva)
        
        # Escolher threshold que maximiza F1
        precision, recall, thresholds = precision_recall_curve(yva, probs_cal)
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-12)
        
        # Ignorar último elemento (threshold = 1.0)
        best_idx = np.nanargmax(f1_scores[:-1])
        best_threshold = float(thresholds[best_idx]) if len(thresholds) > 0 else 0.5
        best_f1 = f1_scores[best_idx]
        
        print(f"\n📐 Calibração completa")
        print(f"  Threshold ótimo: {best_threshold:.3f}")
        print(f"  F1 em validação: {best_f1:.3f}")
        
        # Calcular Brier score
        from sklearn.metrics import brier_score_loss
        brier_before = brier_score_loss(yva, probs)
        brier_after = brier_score_loss(yva, probs_cal)
        print(f"  Brier Score: {brier_before:.4f} → {brier_after:.4f}")
        
        return calibrator, best_threshold
    
    @torch.no_grad()
    def predict_proba(model, X, device):
        """
        Prediz probabilidades para conjunto de dados
        
        Args:
            model: Modelo LSTM
            X: Features
            device: Device
            
        Returns:
            Probabilidades preditas
        """
        model.eval()
        X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
        logits = model(X_tensor)
        return torch.sigmoid(logits).cpu().numpy()

# %%
# Avaliação no teste e geração de sinais
if TORCH_AVAILABLE:
    def evaluate_on_test(model, Xte, yte, calibrator, threshold, test_index, device):
        """
        Avalia modelo no conjunto de teste
        
        Args:
            model: Modelo LSTM treinado
            Xte, yte: Dados de teste
            calibrator: Calibrador isotônico
            threshold: Threshold escolhido
            test_index: Índices do teste
            device: Device
            
        Returns:
            Dicionário com métricas e predições
        """
        # Predições
        probs = predict_proba(model, Xte, device)
        probs_cal = calibrator.transform(probs)
        preds = (probs_cal >= threshold).astype(int)
        
        # Métricas
        ap = average_precision_score(yte, probs_cal)
        f1 = f1_score(yte, preds)
        acc = accuracy_score(yte, preds)
        
        # MCC e Brier
        mcc = matthews_corrcoef(yte, preds)
        brier = brier_score_loss(yte, probs_cal)
        
        # Confusion matrix - forçar shape 2x2 para evitar erro quando só uma classe é predita
        cm = confusion_matrix(yte, preds, labels=[0, 1])
        
        # Desempacotar matriz de forma segura
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
        else:
            # Fallback se algo der errado
            tn = fp = fn = tp = 0
        
        # Sinais para backtest (-1 para short, +1 para long)
        signals = pd.Series((preds * 2 - 1), index=test_index)
        
        print(f"\n📊 Resultados no Teste:")
        print(f"  AP (PR-AUC): {ap:.4f}")
        print(f"  F1 Score:    {f1:.4f}")
        print(f"  Accuracy:    {acc:.4f}")
        print(f"  MCC:         {mcc:.4f}")
        print(f"  Brier Score: {brier:.4f}")
        print(f"\n  Confusion Matrix:")
        print(f"    TN: {tn:4d}  FP: {fp:4d}")
        print(f"    FN: {fn:4d}  TP: {tp:4d}")
        
        return {
            "ap": ap,
            "f1": f1,
            "accuracy": acc,
            "mcc": mcc,
            "brier": brier,
            "confusion_matrix": cm,
            "proba": probs_cal,
            "pred": preds,
            "signals": signals
        }

# %%
# Export para produção
if TORCH_AVAILABLE:
    def export_torchscript(model, in_dim, seq_len, path="lstm_model.pt", device="cpu"):
        """
        Exporta modelo para TorchScript
        
        Args:
            model: Modelo LSTM
            in_dim: Dimensão de entrada
            seq_len: Comprimento da sequência
            path: Caminho para salvar
            device: Device
            
        Returns:
            Caminho do arquivo salvo
        """
        model_cpu = model.to(device).eval()
        
        # Input dummy para tracing
        dummy_input = torch.randn(1, seq_len, in_dim).to(device)
        
        # Trace e salvar
        traced_model = torch.jit.trace(model_cpu, dummy_input)
        traced_model.save(path)
        
        print(f"✅ Modelo exportado para: {path}")
        return path

# %%
# Pipeline completo LSTM
def run_lstm_pipeline(X_df: pd.DataFrame, 
                     y_series: pd.Series,
                     seq_len: int = 64,
                     n_trials: int = 20,  # Reduzido para otimização de memória
                     device: str = None,
                     horizon: str = "lstm") -> Dict:
    """
    Pipeline completo para treinar LSTM com Optuna
    
    Args:
        X_df: DataFrame com features
        y_series: Series com labels binárias
        seq_len: Comprimento das sequências (default: 64 = 16 horas em 15min)
        n_trials: Número de trials do Optuna
        device: Device para treino (None = auto-detectar)
        horizon: Nome do horizonte para logging
        
    Returns:
        Dicionário com resultados compatível com run_multi_horizon_backtest
    """
    if not TORCH_AVAILABLE:
        print("⚠️ PyTorch não disponível - pulando LSTM")
        return {}
    
    if not OPTUNA_AVAILABLE:
        print("⚠️ Optuna não disponível - pulando otimização LSTM")
        return {}
    
    # Auto-detectar device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"\n{'='*80}")
    print(f"🚀 PIPELINE LSTM - Horizonte: {horizon}")
    print(f"{'='*80}")
    print(f"📊 Dataset: {len(X_df)} amostras, {X_df.shape[1]} features")
    print(f"⚙️  Device: {device}")
    print(f"📏 Sequência: {seq_len} timesteps")
    
    # 1. Split temporal
    print(f"\n1️⃣ Preparando splits temporais...")
    tr_idx, va_idx, te_idx = train_val_test_split_time(X_df, y_series)
    print(f"  Treino:    {len(tr_idx)} amostras")
    print(f"  Validação: {len(va_idx)} amostras")
    print(f"  Teste:     {len(te_idx)} amostras")
    
    # 2. Preparar tensores
    print(f"\n2️⃣ Gerando sequências e escalonando features...")
    (Xtr, ytr, Xva, yva, Xte, yte, test_index, scaler) = build_lstm_tensors(
        X_df, y_series, seq_len, tr_idx, va_idx, te_idx
    )
    print(f"  Sequências treino: {Xtr.shape}")
    print(f"  Sequências valid:  {Xva.shape}")
    print(f"  Sequências teste:  {Xte.shape}")
    
    # 3. Otimização com Optuna
    print(f"\n3️⃣ Otimização Bayesiana...")
    model, best_params = fit_lstm_with_optuna(
        Xtr, ytr, Xva, yva, n_trials=n_trials, device=device
    )
    
    # 4. Calibração e threshold
    print(f"\n4️⃣ Calibração de probabilidades...")
    calibrator, threshold = calibrate_and_choose_threshold(
        model, Xva, yva, device
    )
    
    # 5. Avaliação no teste
    print(f"\n5️⃣ Avaliação no conjunto de teste...")
    eval_results = evaluate_on_test(
        model, Xte, yte, calibrator, threshold, test_index, device
    )
    
    # 6. Export para produção
    print(f"\n6️⃣ Exportando modelo...")
    model_path = f"artifacts/models/lstm_{horizon}.pt"
    os.makedirs("artifacts/models", exist_ok=True)
    
    export_path = export_torchscript(
        model, Xtr.shape[-1], seq_len, 
        path=model_path, device="cpu"
    )
    
    # 7. Preparar resultados no formato esperado
    results = {
        horizon: {
            "best_params": best_params,
            "threshold": threshold,
            "test_metrics": {
                "accuracy": eval_results["accuracy"],
                "precision": eval_results["confusion_matrix"][1,1] / 
                            (eval_results["confusion_matrix"][1,1] + 
                             eval_results["confusion_matrix"][0,1] + 1e-10),
                "recall": eval_results["confusion_matrix"][1,1] / 
                         (eval_results["confusion_matrix"][1,1] + 
                          eval_results["confusion_matrix"][1,0] + 1e-10),
                "f1": eval_results["f1"],
                "pr_auc": eval_results["ap"],
                "mcc": eval_results["mcc"],
                "brier": eval_results["brier"]
            },
            "test_indices": test_index.tolist(),
            "predictions": {
                "proba": eval_results["proba"],
                "binary": eval_results["pred"]
            },
            "signals": eval_results["signals"],
            "confusion_matrix": eval_results["confusion_matrix"],
            "artifact": export_path,
            "scaler": scaler,
            "model_type": "lstm"
        }
    }
    
    # MLflow logging se disponível
    if MLFLOW_AVAILABLE:
        import mlflow
        
        mlflow.log_params({
            f"lstm_seq_len_{horizon}": seq_len,
            f"lstm_device_{horizon}": device,
            **{f"lstm_{k}_{horizon}": v for k, v in best_params.items()}
        })
        
        mlflow.log_metrics({
            f"lstm_pr_auc_test_{horizon}": eval_results["ap"],
            f"lstm_f1_test_{horizon}": eval_results["f1"],
            f"lstm_mcc_test_{horizon}": eval_results["mcc"],
            f"lstm_brier_test_{horizon}": eval_results["brier"]
        })
        
        mlflow.log_artifact(export_path)
    
    print(f"\n✅ Pipeline LSTM completo para horizonte {horizon}")
    print(f"{'='*80}")
    
    return results

print("✅ Pipeline LSTM definido e pronto para uso")

# %% [markdown]
# ## 9. Sistema de Backtest Multi-Horizonte

# %%
# Definir BacktestConfig e BacktestEngine caso não estejam disponíveis via import local
# NOTA: Esta é uma implementação de fallback quando os imports do projeto não estão disponíveis
# A implementação principal está em src/backtest/engine.py
# Ambas as implementações mantêm a mesma interface e lógica de custos
if not LOCAL_IMPORTS_AVAILABLE:
    print("⚠️ Usando implementação local do BacktestEngine (imports do projeto não disponíveis)")
    
    @dataclass
    class BacktestConfig:
        """Configuração para backtest - versão local de fallback"""
        initial_capital: float = 100000
        fee_bps: float = 5
        slippage_bps: float = 10
        funding_apr_est: float = 0.00
        borrow_apr_est: float = 0.00
        execution_rule: str = 'next_bar_open'
        max_leverage: float = 1.0
        position_mode: str = 'long_short'
        
    class BacktestEngine:
        """Engine de backtest com cálculo correto de P&L"""
        def __init__(self, config: BacktestConfig):
            self.config = config
            self.capital = config.initial_capital
            self.position = 0  # Direction: -1, 0, 1
            self.position_units = 0  # Actual units held
            self.entry_price = 0
            
        def run_backtest(self, df: pd.DataFrame, signals: pd.Series):
            """Executa backtest com P&L real e position sizing"""
            # Garantir alinhamento de índices
            perf = pd.DataFrame(index=signals.index)
            perf['signals'] = signals
            perf['open'] = df.loc[signals.index, 'open']
            perf['close'] = df.loc[signals.index, 'close']
            perf['returns'] = perf['close'].pct_change()
            
            # Inicializar colunas
            perf['position'] = 0
            perf['position_units'] = 0.0
            perf['position_value'] = 0.0
            perf['unrealized_pnl'] = 0.0
            perf['realized_pnl'] = 0.0
            perf['equity'] = self.config.initial_capital
            perf['costs'] = 0.0
            perf['trades'] = 0
            
            # State tracking
            equity = self.config.initial_capital
            position = 0
            position_units = 0
            entry_price = 0
            
            for i in range(len(perf)):
                current_price = perf['close'].iloc[i]
                
                # Calculate unrealized P&L
                if position_units != 0 and entry_price > 0:
                    if position > 0:  # Long
                        unrealized_pnl = position_units * (current_price - entry_price)
                    else:  # Short
                        unrealized_pnl = -position_units * (current_price - entry_price)
                    perf.iloc[i, perf.columns.get_loc('unrealized_pnl')] = unrealized_pnl
                
                # Execute previous bar's signal (t+1 execution)
                if i > 0:
                    prev_signal = signals.iloc[i-1]
                    
                    if prev_signal != position:
                        execution_price = perf['open'].iloc[i]
                        
                        # Apply slippage
                        if prev_signal > position:  # Buying
                            execution_price *= (1 + self.config.slippage_bps / 10000)
                        elif prev_signal < position:  # Selling
                            execution_price *= (1 - self.config.slippage_bps / 10000)
                        
                        # Close existing position
                        if position != 0 and position_units != 0:
                            close_value = abs(position_units * execution_price)
                            close_cost = close_value * (self.config.fee_bps / 10000)
                            
                            # Realize P&L
                            if position > 0:
                                realized_pnl = position_units * (execution_price - entry_price)
                            else:
                                realized_pnl = -position_units * (execution_price - entry_price)
                            
                            equity += realized_pnl - close_cost
                            perf.iloc[i, perf.columns.get_loc('realized_pnl')] = realized_pnl
                            perf.iloc[i, perf.columns.get_loc('costs')] += close_cost
                        
                        # Open new position
                        if prev_signal != 0:
                            # Calculate position size (using available equity)
                            position_value = equity * self.config.max_leverage * abs(prev_signal)
                            position_units = position_value / execution_price
                            
                            # Opening costs
                            open_cost = position_value * (self.config.fee_bps / 10000)
                            equity -= open_cost
                            
                            position = prev_signal
                            entry_price = execution_price
                            perf.iloc[i, perf.columns.get_loc('costs')] += open_cost
                            perf.iloc[i, perf.columns.get_loc('trades')] = 1
                        else:
                            position = 0
                            position_units = 0
                            entry_price = 0
                
                # Update state
                perf.iloc[i, perf.columns.get_loc('position')] = position
                perf.iloc[i, perf.columns.get_loc('position_units')] = position_units
                perf.iloc[i, perf.columns.get_loc('position_value')] = abs(position_units * current_price)
                perf.iloc[i, perf.columns.get_loc('equity')] = equity
            
            # Calculate net returns from equity curve
            perf['net_returns'] = perf['equity'].pct_change()
            
            # Extract trades
            trades = pd.DataFrame()
            trade_entries = perf[perf['trades'] == 1]
            if len(trade_entries) > 0:
                trade_list = []
                for i in range(len(trade_entries)):
                    if perf['realized_pnl'].iloc[perf.index.get_loc(trade_entries.index[i])] != 0:
                        trade_list.append({
                            'exit_time': trade_entries.index[i],
                            'pnl': perf['realized_pnl'].iloc[perf.index.get_loc(trade_entries.index[i])]
                        })
                trades = pd.DataFrame(trade_list)
            
            return perf, trades

def run_multi_horizon_backtest(df: pd.DataFrame,
                              results: Dict,
                              initial_capital: float = 100000,
                              fee_bps: float = 5,
                              slippage_bps: float = 10) -> Dict:
    """
    Executa backtest para múltiplos horizontes e compara performance
    
    Args:
        df: DataFrame com OHLC
        results: Resultados do pipeline multi-horizonte
        initial_capital: Capital inicial
        fee_bps: Taxa em basis points
        slippage_bps: Slippage em basis points
        
    Returns:
        Dict com resultados de backtest por horizonte
    """
    print("="*80)
    print("📊 BACKTEST MULTI-HORIZONTE")
    print("="*80)
    
    # Usar import local ou classe definida acima
    if LOCAL_IMPORTS_AVAILABLE:
        from src.backtest.engine import BacktestEngine, BacktestConfig
    
    backtest_results = {}
    
    for horizon, horizon_results in results.items():
        print(f"\n⏱️ Backtesting horizonte: {horizon}")
        print("-"*40)
        
        # Configurar backtest
        config = BacktestConfig(
            initial_capital=initial_capital,
            fee_bps=fee_bps,
            slippage_bps=slippage_bps,
            funding_apr_est=0.00,  # Simplificado para demo
            execution_rule='next_bar_open'
        )
        
        # Gerar sinais a partir das predições com índices corretos
        predictions = horizon_results['predictions']['binary']
        labels = horizon_results['labels']
        
        # Usar o índice do conjunto de teste após as máscaras
        # Isso garante alinhamento correto com os dados
        test_indices = horizon_results.get('test_indices', None)
        if test_indices is None:
            # Fallback se não tivermos os índices salvos
            test_start_idx = len(df) - len(predictions)
            test_indices = df.index[test_start_idx:test_start_idx + len(predictions)]
        
        signals = pd.Series(predictions * 2 - 1, index=test_indices)  # Converter 0/1 para -1/1
        
        # Executar backtest
        bt_engine = BacktestEngine(config)
        perf, trades = bt_engine.run_backtest(df.loc[signals.index], signals)
        
        # Métricas de trading - usar net_returns que inclui custos
        returns = perf['net_returns'].dropna()
        cumulative_return = (1 + returns).cumprod().iloc[-1] - 1 if len(returns) > 0 else 0
        
        # Sharpe Ratio
        if returns.std() > 0:
            sharpe = returns.mean() / returns.std() * np.sqrt(365 * 24 * 4)  # Anualizado para 15min
        else:
            sharpe = 0
            
        # Maximum Drawdown
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Calmar Ratio
        calmar = cumulative_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # Win rate
        winning_trades = trades[trades['pnl'] > 0] if len(trades) > 0 else pd.DataFrame()
        win_rate = len(winning_trades) / len(trades) if len(trades) > 0 else 0
        
        # Profit factor
        gross_profit = trades[trades['pnl'] > 0]['pnl'].sum() if len(trades) > 0 else 0
        gross_loss = abs(trades[trades['pnl'] < 0]['pnl'].sum()) if len(trades) > 0 else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
        
        # Turnover
        position_changes = signals.diff().abs()
        turnover = position_changes.sum() / len(signals)
        
        # Expected Value per trade
        ev_per_trade = trades['pnl'].mean() if len(trades) > 0 else 0
        
        print(f"\n📈 Métricas de Trading para {horizon}:")
        print(f"  Retorno Total:    {cumulative_return:+.2%}")
        print(f"  Sharpe Ratio:     {sharpe:.3f}")
        print(f"  Max Drawdown:     {max_drawdown:.2%}")
        print(f"  Calmar Ratio:     {calmar:.3f}")
        print(f"  Win Rate:         {win_rate:.2%}")
        print(f"  Profit Factor:    {profit_factor:.2f}")
        print(f"  Turnover:         {turnover:.3f}")
        print(f"  EV per Trade:     ${ev_per_trade:.2f}")
        print(f"  Num Trades:       {len(trades)}")
        
        # Comparar com Buy & Hold
        buy_hold_return = (df.loc[signals.index, 'close'].iloc[-1] / 
                          df.loc[signals.index, 'close'].iloc[0] - 1)
        outperformance = cumulative_return - buy_hold_return
        
        print(f"\n  Buy & Hold:       {buy_hold_return:+.2%}")
        print(f"  Outperformance:   {outperformance:+.2%}")
        
        # Salvar resultados
        backtest_results[horizon] = {
            'performance': perf,
            'trades': trades,
            'metrics': {
                'cumulative_return': cumulative_return,
                'sharpe_ratio': sharpe,
                'max_drawdown': max_drawdown,
                'calmar_ratio': calmar,
                'win_rate': win_rate,
                'profit_factor': profit_factor,
                'turnover': turnover,
                'ev_per_trade': ev_per_trade,
                'num_trades': len(trades),
                'buy_hold_return': buy_hold_return,
                'outperformance': outperformance
            },
            'signals': signals,
            'returns': returns
        }
    
    # Análise comparativa
    print(f"\n{'='*80}")
    print("🏆 COMPARAÇÃO ENTRE HORIZONTES")
    print(f"{'='*80}")
    
    comparison_metrics = pd.DataFrame({
        horizon: backtest_results[horizon]['metrics']
        for horizon in backtest_results.keys()
    }).T
    
    print("\n📊 Tabela Comparativa de Backtest:")
    print(comparison_metrics.round(3))
    
    # Identificar melhor horizonte por diferentes métricas
    best_return = comparison_metrics['cumulative_return'].idxmax()
    best_sharpe = comparison_metrics['sharpe_ratio'].idxmax()
    best_calmar = comparison_metrics['calmar_ratio'].idxmax()
    
    print(f"\n🥇 Melhores Horizontes:")
    print(f"  Melhor Retorno: {best_return} ({comparison_metrics.loc[best_return, 'cumulative_return']:+.2%})")
    print(f"  Melhor Sharpe:  {best_sharpe} ({comparison_metrics.loc[best_sharpe, 'sharpe_ratio']:.3f})")
    print(f"  Melhor Calmar:  {best_calmar} ({comparison_metrics.loc[best_calmar, 'calmar_ratio']:.3f})")
    
    # Análise de correlação de retornos
    print(f"\n🔗 Correlação entre retornos dos horizontes:")
    returns_df = pd.DataFrame({
        horizon: backtest_results[horizon]['returns']
        for horizon in backtest_results.keys()
    })
    
    # Alinhar índices
    returns_df = returns_df.dropna()
    if len(returns_df) > 0:
        corr_matrix = returns_df.corr()
        print(corr_matrix.round(3))
    
    # Salvar resultados
    os.makedirs(config.artifacts_path + "/backtest", exist_ok=True)
    comparison_metrics.to_csv(f"{config.artifacts_path}/backtest/horizon_backtest_comparison.csv")
    
    # Plotar equity curves (opcional - salvando dados para visualização posterior)
    equity_curves = {}
    for horizon in backtest_results.keys():
        returns = backtest_results[horizon]['returns']
        equity = (1 + returns).cumprod()
        equity_curves[horizon] = equity
    
    equity_df = pd.DataFrame(equity_curves)
    equity_df.to_csv(f"{config.artifacts_path}/backtest/equity_curves.csv")
    
    print(f"\n✅ Resultados salvos em {config.artifacts_path}/backtest/")
    
    return backtest_results

print("✅ Função run_multi_horizon_backtest definida")

# %% [markdown]
# ## 9. Estratégia Ensemble

# %%
def create_ensemble_signals(results: Dict, 
                           weights: Dict = None,
                           voting: str = 'soft') -> pd.Series:
    """
    Cria sinais ensemble combinando múltiplos horizontes
    
    Args:
        results: Resultados do pipeline multi-horizonte
        weights: Pesos para cada horizonte (None = igual peso)
        voting: 'soft' (média ponderada) ou 'hard' (votação majoritária)
        
    Returns:
        Series com sinais combinados
    """
    if weights is None:
        weights = {h: 1.0/len(results) for h in results.keys()}
    
    # Coletar probabilidades calibradas
    probabilities = {}
    for horizon, horizon_results in results.items():
        probs = horizon_results['predictions']['calibrated']
        probabilities[horizon] = probs
    
    # Criar DataFrame alinhado
    prob_df = pd.DataFrame(probabilities)
    
    if voting == 'soft':
        # Média ponderada das probabilidades
        weighted_probs = sum(prob_df[h] * weights[h] for h in prob_df.columns)
        # Aplicar threshold médio dos horizontes
        avg_threshold = np.mean([results[h]['threshold'] for h in results.keys()])
        signals = (weighted_probs >= avg_threshold).astype(int) * 2 - 1
    else:  # voting == 'hard'
        # Votação majoritária
        binary_preds = pd.DataFrame({
            h: (prob_df[h] >= results[h]['threshold']).astype(int)
            for h in prob_df.columns
        })
        signals = (binary_preds.mean(axis=1) >= 0.5).astype(int) * 2 - 1
    
    return signals

print("✅ Função create_ensemble_signals definida")

# %% [markdown]
# ## 10. Funções de Demonstração

# %%
def create_sample_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cria features básicas para demonstração
    """
    features = pd.DataFrame(index=df.index)
    
    # Returns
    for period in [1, 5, 10, 20, 50]:
        features[f'return_{period}'] = df['close'].pct_change(period)
    
    # Moving averages
    for period in [10, 20, 50, 100]:
        features[f'ma_{period}'] = df['close'].rolling(period).mean() / df['close'] - 1
    
    # Volume
    features['volume_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
    features['volume_ma_20'] = df['volume'].rolling(20).mean()
    
    # Volatility
    features['volatility_20'] = df['close'].pct_change().rolling(20).std()
    features['high_low_ratio'] = df['high'] / df['low'] - 1
    
    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    features['rsi'] = 100 - (100 / (1 + rs))
    
    # Bollinger Bands
    ma_20 = df['close'].rolling(20).mean()
    std_20 = df['close'].rolling(20).std()
    features['bb_upper'] = (ma_20 + 2 * std_20) / df['close'] - 1
    features['bb_lower'] = (ma_20 - 2 * std_20) / df['close'] - 1
    features['bb_width'] = features['bb_upper'] - features['bb_lower']
    
    # Price position
    features['price_position'] = (df['close'] - df['low']) / (df['high'] - df['low'])
    
    return features

def generate_sample_data(n_samples: int = 10000, freq: str = '15min') -> pd.DataFrame:
    """
    Gera dados OHLCV sintéticos para teste
    """
    np.random.seed(42)
    
    dates = pd.date_range('2023-01-01', periods=n_samples, freq=freq)
    
    # Simular preço com tendência e volatilidade
    returns = np.random.randn(n_samples) * 0.01  # 1% vol
    price = 100 * np.exp(returns.cumsum())
    
    df = pd.DataFrame(index=dates)
    df['close'] = price
    
    # Gerar OHLV a partir do close
    df['open'] = df['close'] * (1 + np.random.randn(n_samples) * 0.001)
    df['high'] = df[['open', 'close']].max(axis=1) * (1 + np.abs(np.random.randn(n_samples)) * 0.002)
    df['low'] = df[['open', 'close']].min(axis=1) * (1 - np.abs(np.random.randn(n_samples)) * 0.002)
    df['volume'] = np.random.exponential(1000, n_samples) * (1 + np.abs(returns) * 10)
    
    # Garantir consistência OHLC
    df['high'] = df[['open', 'high', 'close']].max(axis=1)
    df['low'] = df[['open', 'low', 'close']].min(axis=1)
    
    return df

def demo_multi_horizon_pipeline():
    """
    Demonstração completa do pipeline multi-horizonte
    """
    print("="*80)
    print("🚀 DEMONSTRAÇÃO DO PIPELINE MULTI-HORIZONTE")
    print("="*80)
    
    # 1. Gerar dados sintéticos
    print("\n📊 Gerando dados sintéticos...")
    df = generate_sample_data(n_samples=10000)
    print(f"  Dados gerados: {len(df)} barras de 15min")
    print(f"  Período: {df.index[0]} a {df.index[-1]}")
    
    # 2. Criar features
    print("\n🔧 Criando features...")
    features = create_sample_features(df)
    
    # Adicionar features de calendário
    crypto_features = Crypto24x7Features()
    features = pd.concat([
        features,
        crypto_features.create_calendar_features(df),
        crypto_features.create_session_features(df)
    ], axis=1)
    
    print(f"  Features criadas: {len(features.columns)}")
    print(f"  Features: {', '.join(features.columns[:10])}...")
    
    # 3. Executar pipeline multi-horizonte
    print("\n🎯 Executando pipeline multi-horizonte...")
    results = run_multi_horizon_pipeline(
        df=df,
        features=features,
        horizons=['15m', '30m', '60m', '120m'],
        n_trials=10  # Reduzido para demo
    )
    
    # 4. Executar backtest
    print("\n📊 Executando backtest multi-horizonte...")
    backtest_results = run_multi_horizon_backtest(df, results)
    
    # 5. Criar sinais ensemble
    print("\n🎯 Criando estratégia ensemble...")
    ensemble_signals = create_ensemble_signals(results, voting='soft')
    print(f"  Sinais ensemble criados: {len(ensemble_signals)}")
    
    print("\n✅ Demonstração concluída!")
    
    return results, backtest_results

print("✅ Funções de demonstração definidas")

# %% [markdown]
# ## 11. Como Usar o Pipeline

# %%
print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                        GUIA DE USO DO PIPELINE                              ║
╚══════════════════════════════════════════════════════════════════════════════╝

1. CONFIGURAÇÃO INICIAL:
   ```python
   # Configurar ambiente determinístico
   setup_deterministic_environment(seed=42)
   
   # Configurar projeto
   config = ProjectConfig()
   config.create_directories()
   ```

2. CARREGAR SEUS DADOS:
   ```python
   # Opção 1: Dados locais
   df = pd.read_csv('seu_arquivo.csv', index_col='timestamp', parse_dates=True)
   
   # Opção 2: API (se disponível)
   from src.data.binance_loader import CryptoDataLoader
   loader = CryptoDataLoader()
   df = loader.fetch_ohlcv('BTCUSDT', '15m', limit=10000)
   ```

3. CRIAR FEATURES:
   ```python
   # Features básicas
   features = create_sample_features(df)
   
   # Adicionar features crypto 24/7
   crypto_features = Crypto24x7Features()
   # Definir período de funding baseado no contrato (480 min para maioria, 60 min para alguns)
   funding_period = 480  # Ajustar conforme símbolo/exchange
   features = pd.concat([
       features,
       crypto_features.create_calendar_features(df),
       crypto_features.create_session_features(df),
       crypto_features.create_funding_features(df, funding_period_minutes=funding_period)
   ], axis=1)
   ```

4. TREINAR MODELOS MULTI-HORIZONTE:
   ```python
   results = run_multi_horizon_pipeline(
       df=df,
       features=features,
       horizons=['15m', '30m', '60m', '120m'],
       test_size=0.2,
       val_size=0.2,
       n_trials=50  # Aumentar para produção
   )
   ```

5. EXECUTAR BACKTEST:
   ```python
   backtest_results = run_multi_horizon_backtest(
       df=df,
       results=results,
       initial_capital=100000,
       fee_bps=5,
       slippage_bps=10
   )
   ```

6. CRIAR ESTRATÉGIA ENSEMBLE:
   ```python
   # Combinar sinais de múltiplos horizontes
   ensemble_signals = create_ensemble_signals(
       results,
       weights={'15m': 0.2, '30m': 0.3, '60m': 0.3, '120m': 0.2},
       voting='soft'
   )
   ```

7. DEMO RÁPIDA:
   ```python
   # Executar demonstração completa com dados sintéticos
   results, backtest_results = demo_multi_horizon_pipeline()
   ```

NOTAS IMPORTANTES:
- Sempre use dados de 15 minutos como base
- Horizontes são múltiplos de 15min (15m, 30m, 60m, 120m)
- PR-AUC é a métrica principal (não ROC-AUC)
- Calibração de probabilidades é obrigatória
- Backtest usa execução t+1 (sinal em t, execução em t+1)
- Custos incluem fees e slippage

Para mais informações, consulte a documentação em docs/
""")

# %% [markdown]
# ## 12. Executar Demonstração

# %%
# Para executar a demonstração, descomente a linha abaixo:
# results, backtest_results = demo_multi_horizon_pipeline()

# %% [markdown]
# ## 13. Suíte Completa de Testes - Validação de Requisitos PRD

# %%
class ModelTestSuite:
    """
    Suíte completa de testes para validar todos os requisitos dos PRDs
    Inclui testes de integridade, performance e requisitos econômicos
    """
    
    def __init__(self, config: ProjectConfig = None):
        self.config = config or ProjectConfig()
        self.test_results = {}
        
    def test_temporal_leakage(self, df: pd.DataFrame, features: pd.DataFrame, 
                              labels: pd.Series) -> Tuple[bool, Dict]:
        """
        Teste de vazamento temporal - CRÍTICO
        Requisito PRD: Sem vazamento temporal comprovado
        """
        print("\n🔍 Testando vazamento temporal...")
        
        passed = True
        metrics = {}
        
        # Verificar se features usam informação futura
        for col in features.columns:
            if 'future' in col.lower() or 'next' in col.lower():
                passed = False
                metrics[f'feature_{col}'] = 'SUSPEITA: nome sugere informação futura'
        
        # Verificar correlação com retornos futuros não shiftados
        future_returns = df['close'].pct_change().shift(-1)  # Retorno futuro
        
        for col in features.columns:
            if features[col].notna().sum() > 100:  # Só testar se tiver dados suficientes
                corr = features[col].corr(future_returns)
                if abs(corr) > 0.95:  # Correlação muito alta é suspeita
                    passed = False
                    metrics[f'correlation_{col}'] = f'ALTA: {corr:.3f}'
        
        # Verificar alinhamento temporal de labels
        label_returns = labels.shift(-1)  # Labels devem estar no futuro
        label_feature_corr = labels.corr(features.mean(axis=1))
        
        if abs(label_feature_corr) > 0.8:
            passed = False
            metrics['label_alignment'] = f'PROBLEMA: correlação {label_feature_corr:.3f}'
        
        metrics['status'] = 'PASS' if passed else 'FAIL'
        print(f"  Resultado: {'✅ PASS' if passed else '❌ FAIL'}")
        
        return passed, metrics
    
    def test_data_quality(self, df: pd.DataFrame, features: pd.DataFrame) -> Tuple[bool, Dict]:
        """
        Teste de qualidade de dados
        Requisito PRD: Dados limpos e consistentes
        """
        print("\n🔍 Testando qualidade de dados...")
        
        passed = True
        metrics = {}
        
        # Verificar NaN
        nan_ratio = features.isna().sum().sum() / (len(features) * len(features.columns))
        metrics['nan_ratio'] = nan_ratio
        if nan_ratio > 0.1:  # Mais de 10% NaN é problemático
            passed = False
        
        # Verificar consistência OHLC
        ohlc_errors = 0
        ohlc_errors += (df['high'] < df['low']).sum()
        ohlc_errors += (df['high'] < df['open']).sum()
        ohlc_errors += (df['high'] < df['close']).sum()
        ohlc_errors += (df['low'] > df['open']).sum()
        ohlc_errors += (df['low'] > df['close']).sum()
        
        metrics['ohlc_errors'] = ohlc_errors
        if ohlc_errors > 0:
            passed = False
        
        # Verificar outliers extremos (> 10 desvios padrão)
        outliers = 0
        for col in features.select_dtypes(include=[np.number]).columns:
            z_scores = np.abs(stats.zscore(features[col].dropna()))
            outliers += (z_scores > 10).sum()
        
        metrics['extreme_outliers'] = outliers
        if outliers > len(features) * 0.01:  # Mais de 1% outliers extremos
            passed = False
        
        metrics['status'] = 'PASS' if passed else 'FAIL'
        print(f"  Resultado: {'✅ PASS' if passed else '❌ FAIL'}")
        
        return passed, metrics
    
    def test_model_performance(self, results: Dict) -> Tuple[bool, Dict]:
        """
        Teste de performance do modelo
        Requisito PRD: PR-AUC acima do baseline, F1 adequado
        """
        print("\n🔍 Testando performance dos modelos...")
        
        passed = True
        metrics = {}
        
        baseline_pr_auc = 0.5  # Baseline aleatório
        min_acceptable_f1 = 0.3  # Mínimo aceitável para crypto
        
        for horizon, result in results.items():
            horizon_metrics = result['metrics']
            
            # PR-AUC deve ser melhor que baseline
            pr_auc = horizon_metrics['pr_auc']
            metrics[f'{horizon}_pr_auc'] = pr_auc
            if pr_auc <= baseline_pr_auc:
                passed = False
                metrics[f'{horizon}_pr_auc_status'] = 'FAIL: abaixo do baseline'
            
            # F1 score mínimo
            f1 = horizon_metrics['f1']
            metrics[f'{horizon}_f1'] = f1
            if f1 < min_acceptable_f1:
                passed = False
                metrics[f'{horizon}_f1_status'] = 'FAIL: F1 muito baixo'
            
            # MCC (Matthews Correlation Coefficient)
            mcc = horizon_metrics['mcc']
            metrics[f'{horizon}_mcc'] = mcc
            if mcc < 0:  # MCC negativo indica pior que aleatório
                passed = False
                metrics[f'{horizon}_mcc_status'] = 'FAIL: MCC negativo'
        
        metrics['status'] = 'PASS' if passed else 'FAIL'
        print(f"  Resultado: {'✅ PASS' if passed else '❌ FAIL'}")
        
        return passed, metrics
    
    def test_calibration(self, results: Dict) -> Tuple[bool, Dict]:
        """
        Teste de calibração de probabilidades
        Requisito PRD: Calibração dentro de ±2 p.p.
        """
        print("\n🔍 Testando calibração de probabilidades...")
        
        passed = True
        metrics = {}
        
        for horizon, result in results.items():
            predictions = result['predictions']['calibrated']
            labels = result['labels']
            
            # Calcular ECE (Expected Calibration Error)
            n_bins = 10
            bin_boundaries = np.linspace(0, 1, n_bins + 1)
            bin_lowers = bin_boundaries[:-1]
            bin_uppers = bin_boundaries[1:]
            
            ece = 0
            for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
                in_bin = (predictions > bin_lower) & (predictions <= bin_upper)
                prop_in_bin = in_bin.mean()
                
                if prop_in_bin > 0:
                    accuracy_in_bin = labels[in_bin].mean()
                    avg_confidence_in_bin = predictions[in_bin].mean()
                    ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
            
            metrics[f'{horizon}_ece'] = ece
            
            # ECE deve ser < 0.02 (2%)
            if ece > 0.02:
                passed = False
                metrics[f'{horizon}_ece_status'] = f'FAIL: ECE {ece:.3f} > 0.02'
            
            # Brier Score (menor é melhor)
            brier = brier_score_loss(labels, predictions)
            metrics[f'{horizon}_brier'] = brier
            
            if brier > 0.25:  # Brier > 0.25 indica má calibração
                passed = False
                metrics[f'{horizon}_brier_status'] = f'FAIL: Brier {brier:.3f} > 0.25'
        
        metrics['status'] = 'PASS' if passed else 'FAIL'
        print(f"  Resultado: {'✅ PASS' if passed else '❌ FAIL'}")
        
        return passed, metrics
    
    def test_economic_metrics(self, backtest_results: Dict) -> Tuple[bool, Dict]:
        """
        Teste de métricas econômicas
        Requisito PRD: Sharpe > 1.0, DSR > 0.8
        """
        print("\n🔍 Testando métricas econômicas...")
        
        passed = True
        metrics = {}
        
        min_sharpe = 1.0  # Requisito PRD
        min_dsr = 0.8     # Requisito PRD
        max_acceptable_drawdown = 0.25  # Max 25% drawdown
        
        for horizon, result in backtest_results.items():
            horizon_metrics = result['metrics']
            
            # Sharpe Ratio
            sharpe = horizon_metrics['sharpe_ratio']
            metrics[f'{horizon}_sharpe'] = sharpe
            if sharpe < min_sharpe:
                passed = False
                metrics[f'{horizon}_sharpe_status'] = f'FAIL: Sharpe {sharpe:.2f} < {min_sharpe}'
            
            # Maximum Drawdown
            mdd = abs(horizon_metrics['max_drawdown'])
            metrics[f'{horizon}_max_drawdown'] = mdd
            if mdd > max_acceptable_drawdown:
                passed = False
                metrics[f'{horizon}_mdd_status'] = f'FAIL: MDD {mdd:.2%} > {max_acceptable_drawdown:.0%}'
            
            # Calmar Ratio
            calmar = horizon_metrics['calmar_ratio']
            metrics[f'{horizon}_calmar'] = calmar
            
            # Win Rate
            win_rate = horizon_metrics['win_rate']
            metrics[f'{horizon}_win_rate'] = win_rate
            if win_rate < 0.45:  # Win rate muito baixo
                metrics[f'{horizon}_win_rate_status'] = f'WARNING: Win rate {win_rate:.1%} baixo'
            
            # Calcular DSR (Deflated Sharpe Ratio) simplificado
            # DSR = Sharpe * sqrt(T) / sqrt(1 + skew^2/4 + (kurt-3)^2/24)
            returns = result.get('returns', pd.Series([0]))
            if len(returns) > 30:
                skew = returns.skew()
                kurt = returns.kurt()
                T = len(returns) / (365 * 24 * 4)  # Anos de dados
                dsr = sharpe * np.sqrt(T) / np.sqrt(1 + skew**2/4 + (kurt-3)**2/24)
                metrics[f'{horizon}_dsr'] = dsr
                
                if dsr < min_dsr:
                    passed = False
                    metrics[f'{horizon}_dsr_status'] = f'FAIL: DSR {dsr:.2f} < {min_dsr}'
        
        metrics['status'] = 'PASS' if passed else 'FAIL'
        print(f"  Resultado: {'✅ PASS' if passed else '❌ FAIL'}")
        
        return passed, metrics
    
    def test_feature_importance_stability(self, results: Dict) -> Tuple[bool, Dict]:
        """
        Teste de estabilidade das feature importances
        Requisito PRD: Feature importances consistentes
        """
        print("\n🔍 Testando estabilidade de feature importance...")
        
        passed = True
        metrics = {}
        
        # Coletar top features de cada horizonte
        top_features_by_horizon = {}
        for horizon, result in results.items():
            top_10 = result['feature_importance'].head(10)['feature'].tolist()
            top_features_by_horizon[horizon] = set(top_10)
        
        # Verificar overlap entre horizontes
        horizons = list(results.keys())
        for i in range(len(horizons)-1):
            h1, h2 = horizons[i], horizons[i+1]
            overlap = len(top_features_by_horizon[h1] & top_features_by_horizon[h2])
            overlap_ratio = overlap / 10
            
            metrics[f'overlap_{h1}_{h2}'] = overlap_ratio
            
            if overlap_ratio < 0.3:  # Menos de 30% de overlap é suspeito
                passed = False
                metrics[f'overlap_{h1}_{h2}_status'] = f'FAIL: apenas {overlap_ratio:.1%} overlap'
        
        metrics['status'] = 'PASS' if passed else 'FAIL'
        print(f"  Resultado: {'✅ PASS' if passed else '❌ FAIL'}")
        
        return passed, metrics
    
    def test_execution_realism(self, backtest_results: Dict) -> Tuple[bool, Dict]:
        """
        Teste de realismo da execução
        Requisito: Execução t+1, custos aplicados
        """
        print("\n🔍 Testando realismo da execução...")
        
        passed = True
        metrics = {}
        
        for horizon, result in backtest_results.items():
            # Verificar turnover
            turnover = result['metrics'].get('turnover', 0)
            metrics[f'{horizon}_turnover'] = turnover
            
            if turnover > 10:  # Turnover muito alto é irrealista
                passed = False
                metrics[f'{horizon}_turnover_status'] = f'FAIL: turnover {turnover:.1f} muito alto'
            
            # Verificar se custos foram aplicados
            if 'outperformance' in result['metrics']:
                outperf = result['metrics']['outperformance']
                if outperf > 0.5:  # Outperformance > 50% é suspeito
                    metrics[f'{horizon}_outperf_warning'] = f'WARNING: outperformance {outperf:.1%} muito alto'
        
        metrics['status'] = 'PASS' if passed else 'FAIL'
        print(f"  Resultado: {'✅ PASS' if passed else '❌ FAIL'}")
        
        return passed, metrics
    
    def run_all_tests(self, df: pd.DataFrame, features: pd.DataFrame, 
                      results: Dict, backtest_results: Dict) -> Dict:
        """
        Executa todos os testes e gera relatório completo
        """
        print("\n" + "="*80)
        print("🧪 EXECUTANDO SUÍTE COMPLETA DE TESTES")
        print("="*80)
        
        all_results = {}
        
        # Preparar labels para teste (usar do primeiro horizonte)
        first_horizon = list(results.keys())[0]
        labeler = results[first_horizon]['labeler']
        labels = labeler.create_labels(df)
        
        # 1. Teste de vazamento temporal
        passed, metrics = self.test_temporal_leakage(df, features, labels)
        all_results['temporal_leakage'] = {'passed': passed, 'metrics': metrics}
        
        # 2. Teste de qualidade de dados
        passed, metrics = self.test_data_quality(df, features)
        all_results['data_quality'] = {'passed': passed, 'metrics': metrics}
        
        # 3. Teste de performance do modelo
        passed, metrics = self.test_model_performance(results)
        all_results['model_performance'] = {'passed': passed, 'metrics': metrics}
        
        # 4. Teste de calibração
        passed, metrics = self.test_calibration(results)
        all_results['calibration'] = {'passed': passed, 'metrics': metrics}
        
        # 5. Teste de métricas econômicas
        passed, metrics = self.test_economic_metrics(backtest_results)
        all_results['economic_metrics'] = {'passed': passed, 'metrics': metrics}
        
        # 6. Teste de estabilidade de features
        passed, metrics = self.test_feature_importance_stability(results)
        all_results['feature_stability'] = {'passed': passed, 'metrics': metrics}
        
        # 7. Teste de realismo de execução
        passed, metrics = self.test_execution_realism(backtest_results)
        all_results['execution_realism'] = {'passed': passed, 'metrics': metrics}
        
        # Resumo final
        self.print_test_summary(all_results)
        
        return all_results
    
    def print_test_summary(self, results: Dict):
        """
        Imprime resumo dos testes
        """
        print("\n" + "="*80)
        print("📊 RESUMO DOS TESTES")
        print("="*80)
        
        total_tests = len(results)
        passed_tests = sum(1 for r in results.values() if r['passed'])
        
        print(f"\nTotal de testes: {total_tests}")
        print(f"Testes aprovados: {passed_tests}")
        print(f"Taxa de aprovação: {passed_tests/total_tests:.1%}")
        
        print("\n📋 Detalhes por teste:")
        print("-"*50)
        
        for test_name, result in results.items():
            status = "✅ PASS" if result['passed'] else "❌ FAIL"
            print(f"{test_name:25s}: {status}")
            
            # Mostrar métricas críticas se falhou
            if not result['passed']:
                for key, value in result['metrics'].items():
                    if 'FAIL' in str(value) or 'status' in key:
                        print(f"  └─ {key}: {value}")
        
        print("\n" + "="*80)
        
        # Verificação dos requisitos PRD
        print("\n🎯 REQUISITOS PRD:")
        print("-"*50)
        
        # Requisitos críticos
        requirements = {
            'Sem vazamento temporal': results['temporal_leakage']['passed'],
            'PR-AUC acima do baseline': 'model_performance' in results and results['model_performance']['passed'],
            'Sharpe > 1.0': False,  # Verificar nas métricas
            'DSR > 0.8': False,  # Verificar nas métricas
            'Calibração < 2%': results.get('calibration', {}).get('passed', False),
            'Features estáveis': results.get('feature_stability', {}).get('passed', False)
        }
        
        # Verificar Sharpe e DSR
        if 'economic_metrics' in results:
            metrics = results['economic_metrics']['metrics']
            # Verificar se algum horizonte passou no Sharpe
            sharpe_passed = any(
                metrics.get(f'{h}_sharpe', 0) >= 1.0 
                for h in ['15m', '30m', '60m', '120m']
            )
            requirements['Sharpe > 1.0'] = sharpe_passed
            
            # Verificar DSR
            dsr_passed = any(
                metrics.get(f'{h}_dsr', 0) >= 0.8
                for h in ['15m', '30m', '60m', '120m']
                if f'{h}_dsr' in metrics
            )
            requirements['DSR > 0.8'] = dsr_passed
        
        for req, passed in requirements.items():
            status = "✅" if passed else "❌"
            print(f"{status} {req}")
        
        # Conclusão
        all_requirements_met = all(requirements.values())
        
        print("\n" + "="*80)
        if all_requirements_met:
            print("🎉 TODOS OS REQUISITOS PRD FORAM ATENDIDOS!")
        else:
            print("⚠️ ALGUNS REQUISITOS PRD NÃO FORAM ATENDIDOS")
            print("   Revise os testes falhados e ajuste o modelo")
        print("="*80)


# Função para executar os testes
def run_model_tests(df: pd.DataFrame = None, features: pd.DataFrame = None,
                    results: Dict = None, backtest_results: Dict = None) -> Dict:
    """
    Executa a suíte completa de testes
    
    Se não fornecer dados, executa com dados de demo
    """
    if df is None or features is None or results is None:
        print("Gerando dados de demonstração para testes...")
        df = generate_sample_data(10000)
        features = create_sample_features(df)
        
        # Adicionar features crypto
        crypto_features = Crypto24x7Features()
        features = pd.concat([
            features,
            crypto_features.create_calendar_features(df),
            crypto_features.create_session_features(df)
        ], axis=1)
        
        # Executar pipeline
        print("Executando pipeline para gerar resultados...")
        results = run_multi_horizon_pipeline(
            df, features, 
            horizons=['15m', '30m'],  # Menos horizontes para teste rápido
            n_trials=5  # Poucos trials para teste
        )
        
        # Executar backtest
        print("Executando backtest...")
        backtest_results = run_multi_horizon_backtest(df, results)
    
    # Executar testes
    test_suite = ModelTestSuite()
    test_results = test_suite.run_all_tests(df, features, results, backtest_results)
    
    return test_results


# Para executar os testes:
# test_results = run_model_tests()

print("✅ Suíte de testes definida")
