"""Sistema de Labeling Adaptativo baseado em Volatilidade."""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import ta
import structlog
from functools import partial

log = structlog.get_logger()


class VolatilityEstimators:
    """Estimadores de volatilidade para diferentes condições de mercado."""
    
    @staticmethod
    def atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
        """Average True Range - robusto para gaps."""
        indicator = ta.volatility.AverageTrueRange(
            high=df['high'], 
            low=df['low'], 
            close=df['close'], 
            window=window
        )
        return indicator.average_true_range()
    
    @staticmethod
    def garman_klass(df: pd.DataFrame, window: int = 20) -> pd.Series:
        """Garman-Klass estimator usando OHLC."""
        log_hl = np.log(df['high'] / df['low']) ** 2
        log_co = np.log(df['close'] / df['open']) ** 2
        
        gk = np.sqrt(0.5 * log_hl - (2 * np.log(2) - 1) * log_co)
        return gk.rolling(window).mean()
    
    @staticmethod
    def yang_zhang(df: pd.DataFrame, window: int = 20) -> pd.Series:
        """Yang-Zhang estimator - mais preciso para mercados 24/7."""
        log_ho = np.log(df['high'] / df['open'])
        log_lo = np.log(df['low'] / df['open'])
        log_co = np.log(df['close'] / df['open'])
        
        log_oc = np.log(df['open'] / df['close'].shift(1))
        log_oc_sq = log_oc ** 2
        
        log_cc = np.log(df['close'] / df['close'].shift(1))
        log_cc_sq = log_cc ** 2
        
        rs = log_ho * (log_ho - log_co) + log_lo * (log_lo - log_co)
        
        k = 0.34 / (1.34 + (window + 1) / (window - 1))
        
        # Componentes
        open_vol = log_oc_sq.rolling(window).mean()
        close_vol = log_cc_sq.rolling(window).mean()
        rs_vol = rs.rolling(window).mean()
        
        yz = np.sqrt(open_vol + k * close_vol + (1 - k) * rs_vol)
        
        return yz
    
    @staticmethod
    def parkinson(df: pd.DataFrame, window: int = 20) -> pd.Series:
        """Parkinson estimator - usa apenas high/low."""
        hl_ratio = np.log(df['high'] / df['low']) ** 2
        factor = 1 / (4 * np.log(2))
        
        return np.sqrt(factor * hl_ratio.rolling(window).mean())
    
    @staticmethod
    def realized_volatility(df: pd.DataFrame, window: int = 20) -> pd.Series:
        """Volatilidade realizada clássica."""
        returns = np.log(df['close'] / df['close'].shift(1))
        return returns.rolling(window).std() * np.sqrt(252)  # Anualizada


class AdaptiveLabeler:
    """
    Sistema de rotulagem adaptativo baseado em volatilidade.
    Substitui o Triple Barrier por um sistema mais robusto e interpretável.
    Suporta múltiplos horizontes alinhados com timeframe de 15m.
    
    Formula: label = sign(r_future) if |r_future| > τ else 0
    onde τ = k × σ̂ × sqrt(horizon)
    """
    
    def __init__(
        self,
        horizon_bars: int = 1,
        k: float = 1.0,
        vol_estimator: str = 'yang_zhang',
        vol_window: int = 20,
        neutral_zone: bool = True,
        min_threshold: float = 0.001,
        max_threshold: float = 0.10
    ):
        """
        Inicializa o labeler adaptativo.
        
        Args:
            horizon_bars: Janela futura para calcular retorno (em barras)
            k: Multiplicador do threshold (hiperparâmetro a otimizar)
            vol_estimator: Estimador de volatilidade a usar
            vol_window: Janela para cálculo de volatilidade
            neutral_zone: Se True, cria zona morta entre thresholds
            min_threshold: Threshold mínimo permitido
            max_threshold: Threshold máximo permitido
        """
        self.horizon_bars = horizon_bars
        self.k = k
        self.vol_estimator = vol_estimator
        self.vol_window = vol_window
        self.neutral_zone = neutral_zone
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold
        self.volatility_estimators = VolatilityEstimators()
        
        # Mapeamento de horizontes em minutos para bars de 15m
        self.horizon_map = {
            '15m': 1,    # 15 minutos = 1 bar
            '30m': 2,    # 30 minutos = 2 bars
            '60m': 4,    # 60 minutos = 4 bars
            '120m': 8,   # 120 minutos = 8 bars
            '240m': 16,  # 240 minutos = 16 bars
            '480m': 32   # 480 minutos = 32 bars
        }
        
        log.info(
            "adaptive_labeler_initialized",
            horizon_bars=horizon_bars,
            k=k,
            vol_estimator=vol_estimator,
            neutral_zone=neutral_zone
        )
    
    def calculate_volatility(self, df: pd.DataFrame) -> pd.Series:
        """
        Calcula volatilidade usando estimador selecionado.
        
        Args:
            df: DataFrame com OHLC
            
        Returns:
            Series com volatilidade estimada
        """
        estimator_map = {
            'atr': self.volatility_estimators.atr,
            'garman_klass': self.volatility_estimators.garman_klass,
            'yang_zhang': self.volatility_estimators.yang_zhang,
            'parkinson': self.volatility_estimators.parkinson,
            'realized': self.volatility_estimators.realized_volatility
        }
        
        if self.vol_estimator not in estimator_map:
            raise ValueError(f"Estimador {self.vol_estimator} não suportado")
        
        return estimator_map[self.vol_estimator](df, self.vol_window)
    
    def calculate_adaptive_threshold(self, df: pd.DataFrame) -> pd.Series:
        """
        Calcula threshold adaptativo baseado em volatilidade.
        
        Formula: τ = k × σ̂ × sqrt(horizon)
        
        Args:
            df: DataFrame com dados OHLC
            
        Returns:
            Series com threshold adaptativo para cada barra
        """
        # Calcular volatilidade
        volatility = self.calculate_volatility(df)
        
        # Ajustar pelo horizonte (raiz quadrada do tempo)
        horizon_adjustment = np.sqrt(self.horizon_bars)
        
        # Calcular threshold
        threshold = self.k * volatility * horizon_adjustment
        
        # Aplicar limites
        threshold = threshold.clip(lower=self.min_threshold, upper=self.max_threshold)
        
        return threshold
    
    def create_labels(
        self, 
        df: pd.DataFrame,
        return_column: Optional[str] = None
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Cria labels baseados em threshold adaptativo.
        
        Args:
            df: DataFrame com dados OHLC
            return_column: Coluna opcional com retornos futuros pré-calculados
            
        Returns:
            Tupla com:
            - DataFrame com labels adicionados
            - Dicionário com estatísticas
        """
        log.info(
            "creating_adaptive_labels",
            rows=len(df),
            horizon_bars=self.horizon_bars,
            vol_estimator=self.vol_estimator
        )
        
        df = df.copy()
        
        # Calcular retorno futuro se não fornecido
        if return_column is None:
            df['future_return'] = (
                df['close'].shift(-self.horizon_bars) / df['close'] - 1
            )
        else:
            df['future_return'] = df[return_column]
        
        # Calcular threshold adaptativo
        df['threshold'] = self.calculate_adaptive_threshold(df)
        
        # Criar labels
        df['label'] = 0  # Inicializar como neutro
        
        if self.neutral_zone:
            # Com zona neutra: -1, 0, 1
            df.loc[df['future_return'] > df['threshold'], 'label'] = 1   # Long
            df.loc[df['future_return'] < -df['threshold'], 'label'] = -1 # Short
        else:
            # Sem zona neutra: apenas -1, 1
            df.loc[df['future_return'] > 0, 'label'] = 1   # Long
            df.loc[df['future_return'] <= 0, 'label'] = -1 # Short
        
        # Remover últimas barras sem label (sem futuro conhecido)
        df.loc[df.index[-self.horizon_bars:], 'label'] = np.nan
        
        # Estatísticas
        stats = self.get_label_statistics(df)
        
        log.info(
            "labels_created",
            total=stats['total_samples'],
            long_pct=stats['long_rate'],
            short_pct=stats['short_rate'],
            neutral_pct=stats['neutral_rate'],
            avg_threshold=df['threshold'].mean()
        )
        
        return df, stats
    
    def get_label_statistics(self, df: pd.DataFrame) -> Dict:
        """
        Calcula estatísticas dos labels.
        
        Args:
            df: DataFrame com coluna 'label'
            
        Returns:
            Dicionário com estatísticas
        """
        if 'label' not in df.columns:
            raise ValueError("DataFrame não tem coluna 'label'")
        
        # Remover NaN para estatísticas
        labels = df['label'].dropna()
        total = len(labels)
        
        if total == 0:
            return {
                'total_samples': 0,
                'long_count': 0,
                'short_count': 0,
                'neutral_count': 0,
                'long_rate': 0.0,
                'short_rate': 0.0,
                'neutral_rate': 0.0,
                'class_balance': 0.0
            }
        
        label_counts = labels.value_counts()
        
        stats = {
            'total_samples': total,
            'long_count': int(label_counts.get(1, 0)),
            'short_count': int(label_counts.get(-1, 0)),
            'neutral_count': int(label_counts.get(0, 0)),
            'long_rate': float(label_counts.get(1, 0) / total),
            'short_rate': float(label_counts.get(-1, 0) / total),
            'neutral_rate': float(label_counts.get(0, 0) / total)
        }
        
        # Calcular balanço entre classes (excluindo neutro se existir)
        non_neutral = label_counts.get(1, 0) + label_counts.get(-1, 0)
        if non_neutral > 0:
            stats['class_balance'] = float(
                min(label_counts.get(1, 0), label_counts.get(-1, 0)) / 
                max(label_counts.get(1, 0), label_counts.get(-1, 0))
            )
        else:
            stats['class_balance'] = 0.0
        
        # Adicionar estatísticas de threshold se disponível
        if 'threshold' in df.columns:
            threshold_clean = df['threshold'].dropna()
            stats.update({
                'threshold_mean': float(threshold_clean.mean()),
                'threshold_std': float(threshold_clean.std()),
                'threshold_min': float(threshold_clean.min()),
                'threshold_max': float(threshold_clean.max())
            })
        
        return stats
    
    def calculate_sample_weights(
        self, 
        df: pd.DataFrame,
        method: str = 'balanced'
    ) -> np.ndarray:
        """
        Calcula pesos de amostra para lidar com desbalanceamento.
        
        Args:
            df: DataFrame com labels
            method: Método de peso ('balanced', 'sqrt', 'none')
            
        Returns:
            Array com pesos normalizados
        """
        if 'label' not in df.columns:
            raise ValueError("DataFrame precisa ter coluna 'label'")
        
        labels = df['label'].values
        n = len(labels)
        
        if method == 'none':
            return np.ones(n)
        
        # Calcular pesos baseado na frequência das classes
        unique_labels, counts = np.unique(labels[~np.isnan(labels)], return_counts=True)
        
        if len(unique_labels) == 0:
            return np.ones(n)
        
        # Criar mapa de pesos
        total_samples = counts.sum()
        n_classes = len(unique_labels)
        
        if method == 'balanced':
            # Peso inversamente proporcional à frequência
            class_weights = total_samples / (n_classes * counts)
        elif method == 'sqrt':
            # Raiz quadrada do peso balanceado (menos agressivo)
            class_weights = np.sqrt(total_samples / (n_classes * counts))
        else:
            raise ValueError(f"Método {method} não suportado")
        
        # Mapear pesos para cada amostra
        weight_map = dict(zip(unique_labels, class_weights))
        weights = np.array([
            weight_map.get(label, 1.0) if not np.isnan(label) else 0.0 
            for label in labels
        ])
        
        # Normalizar para que a soma seja igual ao número de amostras não-NaN
        non_nan_count = np.sum(~np.isnan(labels))
        if weights.sum() > 0:
            weights = weights / weights.sum() * non_nan_count
        
        log.info(
            "sample_weights_calculated",
            method=method,
            min_weight=weights.min(),
            max_weight=weights.max(),
            mean_weight=weights.mean()
        )
        
        return weights
    
    def optimize_k(
        self,
        df: pd.DataFrame,
        k_values: List[float],
        metric: str = 'balanced_accuracy'
    ) -> Tuple[float, Dict]:
        """
        Otimiza o valor de k usando grid search.
        
        Args:
            df: DataFrame com dados
            k_values: Lista de valores de k para testar
            metric: Métrica para otimizar
            
        Returns:
            Tupla com melhor k e resultados
        """
        results = {}
        
        for k in k_values:
            self.k = k
            df_labeled, stats = self.create_labels(df)
            
            # Calcular métrica baseada na distribuição de labels
            if metric == 'balanced_accuracy':
                # Penalizar distribuições muito desbalanceadas
                score = stats['class_balance']
            elif metric == 'coverage':
                # Maximizar cobertura (minimizar neutros)
                score = 1.0 - stats['neutral_rate']
            else:
                score = 0.0
            
            results[k] = {
                'score': score,
                'stats': stats
            }
        
        # Encontrar melhor k
        best_k = max(results.keys(), key=lambda k: results[k]['score'])
        self.k = best_k
        
        log.info(
            "k_optimized",
            best_k=best_k,
            best_score=results[best_k]['score'],
            metric=metric
        )
        
        return best_k, results


# Manter compatibilidade com código existente
TripleBarrierLabeler = AdaptiveLabeler  # Alias temporário para migração suave