"""Triple Barrier Method para labeling robusto."""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import ta
import structlog
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial

log = structlog.get_logger()


class TripleBarrierLabeler:
    """Implementação do Triple Barrier Method de Marcos López de Prado.
    
    Cria labels para classificação baseado em três barreiras:
    1. Take profit (barreira superior)
    2. Stop loss (barreira inferior)  
    3. Tempo máximo de holding (barreira vertical)
    """

    def __init__(
        self,
        pt_multiplier: float = 2.0,
        sl_multiplier: float = 1.5,
        max_holding_period: int = 100,
        min_ret: float = 0.0001,
        use_atr: bool = True,
        parallel: bool = False,
    ):
        """Inicializa o labeler.
        
        Args:
            pt_multiplier: Multiplicador para take profit (ATR ou fixo)
            sl_multiplier: Multiplicador para stop loss (ATR ou fixo)
            max_holding_period: Período máximo de holding (barras)
            min_ret: Retorno mínimo para considerar como sinal
            use_atr: Se True usa ATR, senão usa percentual fixo
            parallel: Se True usa processamento paralelo
        """
        self.pt_multiplier = pt_multiplier
        self.sl_multiplier = sl_multiplier
        self.max_holding_period = max_holding_period
        self.min_ret = min_ret
        self.use_atr = use_atr
        self.parallel = parallel

    def apply_triple_barrier(
        self, df: pd.DataFrame, side: Optional[pd.Series] = None
    ) -> Tuple[pd.DataFrame, List[Dict]]:
        """Aplica triple barrier nos dados.
        
        Args:
            df: DataFrame com OHLCV e opcionalmente ATR
            side: Série com direção esperada (1=long, -1=short, 0=flat)
                  Se None, assume sempre long
                  
        Returns:
            Tupla com:
            - DataFrame com labels adicionados
            - Lista com informações detalhadas das barreiras
        """
        log.info(
            "applying_triple_barrier",
            rows=len(df),
            use_atr=self.use_atr,
            max_holding=self.max_holding_period,
        )
        
        # Calcular ATR se necessário e não existir
        if self.use_atr and "atr_14" not in df.columns:
            df["atr_14"] = ta.volatility.AverageTrueRange(
                df["high"], df["low"], df["close"], window=14
            ).average_true_range()
            log.info("atr_calculated")
        
        # Se side não fornecido, assume sempre long
        if side is None:
            side = pd.Series(1, index=df.index)
        
        # Aplicar barreiras
        if self.parallel:
            labels, barrier_info = self._apply_parallel(df, side)
        else:
            labels, barrier_info = self._apply_sequential(df, side)
        
        # Adicionar labels ao DataFrame
        df["label"] = labels
        
        # Adicionar meta-labels (binário)
        df["meta_label"] = (df["label"] != 0).astype(int)
        
        # Estatísticas
        label_counts = df["label"].value_counts()
        log.info(
            "labels_created",
            total=len(labels),
            long_wins=label_counts.get(1, 0),
            losses=label_counts.get(-1, 0),
            neutrals=label_counts.get(0, 0),
        )
        
        return df, barrier_info

    def _apply_sequential(self, df: pd.DataFrame, side: pd.Series) -> Tuple[List, List]:
        """Aplica barreiras sequencialmente."""
        labels = []
        barrier_info = []
        
        n = len(df)
        
        for i in range(n - 1):
            # Skip se side é neutro
            if side.iloc[i] == 0:
                labels.append(0)
                barrier_info.append({
                    "entry_idx": i,
                    "exit_idx": i,
                    "entry_price": df["close"].iloc[i],
                    "exit_price": df["close"].iloc[i],
                    "label": 0,
                    "exit_reason": "no_position",
                })
                continue
            
            # Calcular barreiras para esta posição
            result = self._calculate_barrier_touch(
                df, i, side.iloc[i], min(i + self.max_holding_period, n - 1)
            )
            
            labels.append(result["label"])
            barrier_info.append(result)
        
        # Última barra sempre neutra (não há futuro para avaliar)
        labels.append(0)
        barrier_info.append({
            "entry_idx": n - 1,
            "exit_idx": n - 1,
            "label": 0,
            "exit_reason": "end_of_data",
        })
        
        return labels, barrier_info

    def _apply_parallel(self, df: pd.DataFrame, side: pd.Series) -> Tuple[List, List]:
        """Aplica barreiras em paralelo usando multiprocessing."""
        n = len(df)
        
        # Função parcial com DataFrame fixo
        calc_func = partial(self._calculate_barrier_touch_wrapper, df=df, side=side)
        
        # Criar índices para processar
        indices = list(range(n - 1))
        
        # Processar em paralelo
        results = [None] * n
        
        with ProcessPoolExecutor() as executor:
            futures = {executor.submit(calc_func, i): i for i in indices}
            
            for future in as_completed(futures):
                idx = futures[future]
                results[idx] = future.result()
        
        # Última barra
        results[n - 1] = {
            "label": 0,
            "entry_idx": n - 1,
            "exit_idx": n - 1,
            "exit_reason": "end_of_data",
        }
        
        # Separar labels e barrier_info
        labels = [r["label"] for r in results]
        barrier_info = results
        
        return labels, barrier_info

    def _calculate_barrier_touch_wrapper(self, i: int, df: pd.DataFrame, side: pd.Series) -> Dict:
        """Wrapper para cálculo paralelo."""
        if side.iloc[i] == 0:
            return {
                "entry_idx": i,
                "exit_idx": i,
                "label": 0,
                "exit_reason": "no_position",
            }
        
        return self._calculate_barrier_touch(
            df, i, side.iloc[i], min(i + self.max_holding_period, len(df) - 1)
        )

    def _calculate_barrier_touch(
        self, df: pd.DataFrame, entry_idx: int, side: int, max_idx: int
    ) -> Dict:
        """Calcula qual barreira é tocada primeiro.
        
        Args:
            df: DataFrame com dados
            entry_idx: Índice de entrada
            side: Direção (1=long, -1=short)
            max_idx: Índice máximo (barreira de tempo)
            
        Returns:
            Dicionário com informações da barreira tocada
        """
        entry_price = df["close"].iloc[entry_idx]
        
        # Calcular níveis das barreiras
        if self.use_atr:
            atr = df["atr_14"].iloc[entry_idx]
            if pd.isna(atr) or atr <= 0:
                atr = df["close"].iloc[entry_idx] * 0.02  # Fallback 2%
            
            pt_distance = self.pt_multiplier * atr
            sl_distance = self.sl_multiplier * atr
        else:
            # Usar percentual fixo
            pt_distance = entry_price * self.pt_multiplier / 100
            sl_distance = entry_price * self.sl_multiplier / 100
        
        # Definir barreiras baseado no side
        if side == 1:  # Long
            pt_barrier = entry_price + pt_distance
            sl_barrier = entry_price - sl_distance
        else:  # Short
            pt_barrier = entry_price - pt_distance
            sl_barrier = entry_price + sl_distance
        
        # Procurar qual barreira é tocada primeiro
        for j in range(entry_idx + 1, max_idx + 1):
            high = df["high"].iloc[j]
            low = df["low"].iloc[j]
            
            # Para posição long
            if side == 1:
                # Check take profit
                if high >= pt_barrier:
                    return {
                        "entry_idx": entry_idx,
                        "exit_idx": j,
                        "entry_price": entry_price,
                        "exit_price": pt_barrier,
                        "pt_barrier": pt_barrier,
                        "sl_barrier": sl_barrier,
                        "label": 1,
                        "exit_reason": "take_profit",
                        "return": (pt_barrier - entry_price) / entry_price,
                    }
                
                # Check stop loss
                if low <= sl_barrier:
                    return {
                        "entry_idx": entry_idx,
                        "exit_idx": j,
                        "entry_price": entry_price,
                        "exit_price": sl_barrier,
                        "pt_barrier": pt_barrier,
                        "sl_barrier": sl_barrier,
                        "label": -1,
                        "exit_reason": "stop_loss",
                        "return": (sl_barrier - entry_price) / entry_price,
                    }
            
            # Para posição short
            else:
                # Check take profit (preço caindo)
                if low <= pt_barrier:
                    return {
                        "entry_idx": entry_idx,
                        "exit_idx": j,
                        "entry_price": entry_price,
                        "exit_price": pt_barrier,
                        "pt_barrier": pt_barrier,
                        "sl_barrier": sl_barrier,
                        "label": 1,
                        "exit_reason": "take_profit",
                        "return": (entry_price - pt_barrier) / entry_price,
                    }
                
                # Check stop loss (preço subindo)
                if high >= sl_barrier:
                    return {
                        "entry_idx": entry_idx,
                        "exit_idx": j,
                        "entry_price": entry_price,
                        "exit_price": sl_barrier,
                        "pt_barrier": pt_barrier,
                        "sl_barrier": sl_barrier,
                        "label": -1,
                        "exit_reason": "stop_loss",
                        "return": (entry_price - sl_barrier) / entry_price,
                    }
        
        # Barreira de tempo atingida
        exit_price = df["close"].iloc[max_idx]
        
        if side == 1:
            ret = (exit_price - entry_price) / entry_price
        else:
            ret = (entry_price - exit_price) / entry_price
        
        # Classificar baseado no retorno
        if ret > self.min_ret:
            label = 1
        elif ret < -self.min_ret:
            label = -1
        else:
            label = 0
        
        return {
            "entry_idx": entry_idx,
            "exit_idx": max_idx,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "pt_barrier": pt_barrier,
            "sl_barrier": sl_barrier,
            "label": label,
            "exit_reason": "max_holding",
            "return": ret,
        }

    def calculate_sample_weights(
        self, df: pd.DataFrame, barrier_info: List[Dict]
    ) -> np.ndarray:
        """Calcula pesos de amostra baseado em unicidade e retorno.
        
        Implementa o conceito de "uniqueness" do livro de López de Prado:
        - Eventos que se sobrepõem compartilham informação
        - Eventos únicos têm maior peso
        - Decaimento temporal para dar mais peso a eventos recentes
        
        Args:
            df: DataFrame com dados
            barrier_info: Lista com informações das barreiras
            
        Returns:
            Array com pesos normalizados
        """
        n = len(barrier_info)
        weights = np.ones(n)
        
        # 1. Peso por unicidade (eventos não sobrepostos)
        overlap_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i + 1, n):
                # Verificar sobreposição temporal
                if self._check_overlap(barrier_info[i], barrier_info[j]):
                    overlap_matrix[i, j] = 1
                    overlap_matrix[j, i] = 1
        
        # Calcular unicidade (inverso do número de sobreposições)
        overlaps = overlap_matrix.sum(axis=1)
        uniqueness = 1 / (1 + overlaps)
        weights *= uniqueness
        
        # 2. Peso por retorno absoluto (eventos com maior movimento são mais informativos)
        returns = np.array([abs(info.get("return", 0)) for info in barrier_info])
        returns_normalized = returns / (returns.max() + 1e-10)
        weights *= (0.5 + 0.5 * returns_normalized)  # Peso entre 0.5 e 1.0
        
        # 3. Decaimento temporal exponencial
        decay_factor = 0.99
        time_weights = decay_factor ** np.arange(n - 1, -1, -1)
        weights *= time_weights
        
        # 4. Peso por tipo de saída (opcional)
        # Dar mais peso para sinais claros (TP/SL) vs timeout
        exit_weights = np.array([
            1.2 if info["exit_reason"] in ["take_profit", "stop_loss"] else 0.8
            for info in barrier_info
        ])
        weights *= exit_weights
        
        # Normalizar para que a soma seja igual ao número de amostras
        weights = weights / weights.sum() * n
        
        log.info(
            "sample_weights_calculated",
            min_weight=weights.min(),
            max_weight=weights.max(),
            mean_weight=weights.mean(),
            std_weight=weights.std(),
        )
        
        return weights

    def _check_overlap(self, event1: Dict, event2: Dict) -> bool:
        """Verifica se dois eventos se sobrepõem no tempo."""
        start1, end1 = event1["entry_idx"], event1["exit_idx"]
        start2, end2 = event2["entry_idx"], event2["exit_idx"]
        
        # Eventos se sobrepõem se um começa antes do outro terminar
        return (start1 <= end2) and (start2 <= end1)

    def get_label_statistics(self, df: pd.DataFrame) -> Dict:
        """Calcula estatísticas dos labels.
        
        Args:
            df: DataFrame com coluna 'label'
            
        Returns:
            Dicionário com estatísticas
        """
        if "label" not in df.columns:
            raise ValueError("DataFrame não tem coluna 'label'")
        
        label_counts = df["label"].value_counts()
        total = len(df["label"].dropna())
        
        stats = {
            "total_samples": total,
            "long_wins": label_counts.get(1, 0),
            "losses": label_counts.get(-1, 0),
            "neutrals": label_counts.get(0, 0),
            "long_win_rate": label_counts.get(1, 0) / total if total > 0 else 0,
            "loss_rate": label_counts.get(-1, 0) / total if total > 0 else 0,
            "neutral_rate": label_counts.get(0, 0) / total if total > 0 else 0,
            "class_imbalance": max(label_counts.values()) / min(label_counts.values())
            if len(label_counts) > 1 else np.inf,
        }
        
        return stats

    def create_side_prediction(self, df: pd.DataFrame, method: str = "momentum") -> pd.Series:
        """Cria predição de side (direção) para o triple barrier.
        
        Args:
            df: DataFrame com dados
            method: Método para prever direção
                   - "momentum": Baseado em momentum
                   - "mean_reversion": Baseado em reversão à média
                   - "trend": Baseado em tendência de médias móveis
                   
        Returns:
            Série com predições de side (1, -1, 0)
        """
        if method == "momentum":
            # Momentum simples de 20 períodos
            momentum = df["close"].pct_change(20)
            side = pd.Series(0, index=df.index)
            side[momentum > 0.01] = 1   # Long se momentum positivo
            side[momentum < -0.01] = -1  # Short se momentum negativo
            
        elif method == "mean_reversion":
            # Z-score de 50 períodos
            mean = df["close"].rolling(50).mean()
            std = df["close"].rolling(50).std()
            zscore = (df["close"] - mean) / std
            
            side = pd.Series(0, index=df.index)
            side[zscore < -2] = 1   # Long se muito abaixo da média
            side[zscore > 2] = -1   # Short se muito acima da média
            
        elif method == "trend":
            # Cruzamento de médias móveis
            sma_fast = df["close"].rolling(20).mean()
            sma_slow = df["close"].rolling(50).mean()
            
            side = pd.Series(0, index=df.index)
            side[sma_fast > sma_slow] = 1   # Long se tendência de alta
            side[sma_fast < sma_slow] = -1  # Short se tendência de baixa
            
        else:
            # Método padrão: sempre long
            side = pd.Series(1, index=df.index)
        
        return side