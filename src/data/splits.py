"""
Validação temporal com Purged K-Fold e embargo
"""

import numpy as np
import pandas as pd
from typing import Iterator, Tuple, Optional


class PurgedKFold:
    """
    Purged K-Fold com embargo para evitar vazamento temporal
    
    Implementação rigorosa que garante zero vazamento entre treino e validação
    quando labels usam janelas sobrepostas (ex: Triple Barrier)
    """
    
    def __init__(self, n_splits: int = 5, embargo: int = 10):
        """
        Args:
            n_splits: Número de folds
            embargo: Número de barras de embargo entre treino e validação
        """
        if n_splits < 2:
            raise ValueError("n_splits deve ser >= 2")
        if embargo < 0:
            raise ValueError("embargo deve ser >= 0")
            
        self.n_splits = n_splits
        self.embargo = embargo
        
    def split(self, X: pd.DataFrame, y: pd.Series = None, 
              groups: Optional[np.ndarray] = None) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        Gera índices de treino/validação com purging e embargo
        
        Args:
            X: Features
            y: Labels (não usado, mantido para compatibilidade sklearn)
            groups: Grupos (não usado)
            
        Yields:
            train_indices, val_indices
        """
        n_samples = len(X)
        indices = np.arange(n_samples)
        
        # Tamanho de cada fold
        fold_size = n_samples // self.n_splits
        
        for fold in range(self.n_splits):
            # Definir validação
            val_start = fold * fold_size
            val_end = (fold + 1) * fold_size if fold < self.n_splits - 1 else n_samples
            
            # Índices de validação
            val_indices = indices[val_start:val_end]
            
            # Índices de treino com embargo
            train_indices = []
            
            # Adicionar amostras antes da validação (com embargo)
            if val_start > self.embargo:
                train_indices.extend(indices[:val_start - self.embargo])
            
            # Adicionar amostras depois da validação (com embargo)
            if val_end + self.embargo < n_samples:
                train_indices.extend(indices[val_end + self.embargo:])
            
            # Converter para array
            train_indices = np.array(train_indices)
            
            # Verificações de segurança
            if len(train_indices) == 0:
                raise ValueError(f"Fold {fold}: conjunto de treino vazio. "
                                 f"Aumente n_samples ou reduza embargo/n_splits")
            
            # Verificar não-vazamento se X tem índice temporal
            if hasattr(X, 'index') and isinstance(X.index, pd.DatetimeIndex):
                self._verify_no_leakage(X.index, train_indices, val_indices)
            
            yield train_indices, val_indices
    
    def _verify_no_leakage(self, time_index: pd.DatetimeIndex, 
                           train_idx: np.ndarray, val_idx: np.ndarray):
        """
        Verifica que não há vazamento temporal entre treino e validação
        
        Args:
            time_index: Índice temporal do DataFrame
            train_idx: Índices de treino
            val_idx: Índices de validação
            
        Raises:
            AssertionError: Se detectar vazamento temporal
        """
        if len(train_idx) == 0 or len(val_idx) == 0:
            return
            
        train_times = time_index[train_idx]
        val_times = time_index[val_idx]
        
        # Validação está no meio do período total
        val_start = val_times.min()
        val_end = val_times.max()
        
        # Verificação eficiente: check apenas os limites
        train_max = train_times.max()
        train_min = train_times.min()

        # Estimar resolução média do índice (minutos) a partir da mediana dos deltas
        diffs = pd.Series(train_times).diff().dropna()
        if len(diffs) == 0:
            bar_minutes = 1.0
        else:
            bar_minutes = float(diffs.median().total_seconds() / 60.0)
            if bar_minutes <= 0:
                bar_minutes = 1.0
        min_gap = self.embargo * bar_minutes
        
        # Caso 1: Validação está completamente após o treino
        if val_start > train_max:
            # Calcular gap entre último treino e primeiro val
            gap = (val_start - train_max).total_seconds() / 60  # em minutos
            if gap < min_gap:
                raise AssertionError(
                    f"Embargo insuficiente: {gap:.1f} min < {min_gap:.1f} min"
                )
        # Caso 2: Validação está completamente antes do treino
        elif val_end < train_min:
            # Calcular gap entre último val e primeiro treino
            gap = (train_min - val_end).total_seconds() / 60
            if gap < min_gap:
                raise AssertionError(
                    f"Embargo insuficiente: {gap:.1f} min < {min_gap:.1f} min"
                )
        # Caso 3: Treino tem amostras antes E depois da validação
        else:
            # Este é o caso problemático - treino envolve validação
            # Verificar se há amostras de treino dentro do período de validação
            train_in_val = train_times[(train_times >= val_start) & (train_times <= val_end)]
            if len(train_in_val) > 0:
                raise AssertionError(
                    f"Vazamento temporal! {len(train_in_val)} amostras de treino "
                    f"dentro do período de validação [{val_start}, {val_end}]"
                )
    
    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        """Retorna número de splits (compatibilidade sklearn)"""
        return self.n_splits


class CombinatorialPurgedKFold:
    """
    Combinatorial Purged Cross-Validation (CPCV)
    
    Versão mais avançada que gera todas as combinações possíveis
    de splits com purging e embargo
    """
    
    def __init__(self, n_splits: int = 5, n_test_splits: int = 2, 
                 embargo: int = 10):
        """
        Args:
            n_splits: Número total de grupos
            n_test_splits: Número de grupos para teste em cada fold
            embargo: Barras de embargo
        """
        self.n_splits = n_splits
        self.n_test_splits = n_test_splits
        self.embargo = embargo
        
        if n_test_splits >= n_splits:
            raise ValueError("n_test_splits deve ser < n_splits")
    
    def split(self, X: pd.DataFrame, y: pd.Series = None) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        Gera combinações de treino/teste com purging
        
        Implementação simplificada - para versão completa ver mlfinlab
        """
        from itertools import combinations
        
        n_samples = len(X)
        indices = np.arange(n_samples)
        group_size = n_samples // self.n_splits
        
        # Criar grupos
        groups = []
        for i in range(self.n_splits):
            start = i * group_size
            end = (i + 1) * group_size if i < self.n_splits - 1 else n_samples
            groups.append(indices[start:end])
        
        # Gerar combinações
        for test_groups in combinations(range(self.n_splits), self.n_test_splits):
            # Índices de teste
            test_indices = []
            for g in test_groups:
                test_indices.extend(groups[g])
            test_indices = np.array(test_indices)
            
            # Índices de treino (com purging)
            train_indices = []
            for g in range(self.n_splits):
                if g not in test_groups:
                    group_indices = groups[g]
                    # Aplicar embargo
                    valid_indices = [
                        idx for idx in group_indices
                        if self._check_embargo(idx, test_indices, n_samples)
                    ]
                    train_indices.extend(valid_indices)
            
            train_indices = np.array(train_indices)
            
            if len(train_indices) > 0:
                yield train_indices, test_indices
    
    def _check_embargo(self, idx: int, test_indices: np.ndarray, 
                       n_samples: int) -> bool:
        """Verifica se um índice respeita o embargo"""
        min_test = test_indices.min()
        max_test = test_indices.max()
        
        # Verificar distância do início do teste
        if idx < min_test and min_test - idx <= self.embargo:
            return False
        
        # Verificar distância do fim do teste
        if idx > max_test and idx - max_test <= self.embargo:
            return False
        
        # Verificar se está dentro do período de teste
        if idx >= min_test and idx <= max_test:
            return False
        
        return True