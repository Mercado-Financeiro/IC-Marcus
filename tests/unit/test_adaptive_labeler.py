"""Testes unitários para o AdaptiveLabeler."""

import pytest
import numpy as np
import pandas as pd
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.features.labels import AdaptiveLabeler, VolatilityEstimators


class TestVolatilityEstimators:
    """Testes para estimadores de volatilidade."""
    
    @pytest.fixture
    def sample_ohlcv_data(self):
        """Cria dados OHLCV de teste."""
        np.random.seed(42)
        n = 100
        
        # Gerar preços realísticos
        base_price = 100.0
        returns = np.random.normal(0, 0.02, n)  # 2% vol diária
        prices = [base_price]
        
        for ret in returns:
            prices.append(prices[-1] * (1 + ret))
        
        prices = prices[1:]  # Remove base_price
        
        # Criar OHLCV baseado nos preços
        df = pd.DataFrame({
            'open': prices,
            'close': np.roll(prices, -1),
            'high': [max(o, c) * (1 + np.random.uniform(0, 0.01)) for o, c in zip(prices, np.roll(prices, -1))],
            'low': [min(o, c) * (1 - np.random.uniform(0, 0.01)) for o, c in zip(prices, np.roll(prices, -1))],
            'volume': np.random.uniform(1000, 10000, n)
        })
        
        # Corrigir últimas barras
        df.iloc[-1, 1] = df.iloc[-1, 0]  # close = open na última barra
        df.iloc[-1, 2] = df.iloc[-1, 0] * 1.01  # high ajustado
        df.iloc[-1, 3] = df.iloc[-1, 0] * 0.99  # low ajustado
        
        df.index = pd.date_range('2024-01-01', periods=n, freq='15min')
        return df
    
    def test_atr_calculation(self, sample_ohlcv_data):
        """Testa cálculo de ATR."""
        atr = VolatilityEstimators.atr(sample_ohlcv_data, window=14)
        
        assert len(atr) == len(sample_ohlcv_data)
        assert not atr.isna().all(), "ATR não deve ser completamente NaN"
        assert (atr >= 0).all(), "ATR deve ser sempre positivo"
        assert atr.iloc[-1] > 0, "ATR final deve ser positivo"
    
    def test_yang_zhang_calculation(self, sample_ohlcv_data):
        """Testa cálculo de Yang-Zhang."""
        yz = VolatilityEstimators.yang_zhang(sample_ohlcv_data, window=20)
        
        assert len(yz) == len(sample_ohlcv_data)
        # Após período de warmup, deve ser positivo
        yz_valid = yz.dropna()
        assert len(yz_valid) > 0, "Deve ter alguns valores válidos"
        assert (yz_valid >= 0).all(), "Yang-Zhang deve ser sempre positivo (valores não-NaN)"
    
    def test_garman_klass_calculation(self, sample_ohlcv_data):
        """Testa cálculo de Garman-Klass."""
        gk = VolatilityEstimators.garman_klass(sample_ohlcv_data, window=20)
        
        assert len(gk) == len(sample_ohlcv_data)
        gk_valid = gk.dropna()
        assert len(gk_valid) > 0, "Deve ter alguns valores válidos"
        assert (gk_valid >= 0).all(), "Garman-Klass deve ser sempre positivo (valores não-NaN)"
    
    def test_parkinson_calculation(self, sample_ohlcv_data):
        """Testa cálculo de Parkinson."""
        park = VolatilityEstimators.parkinson(sample_ohlcv_data, window=20)
        
        assert len(park) == len(sample_ohlcv_data)
        park_valid = park.dropna()
        assert len(park_valid) > 0, "Deve ter alguns valores válidos"
        assert (park_valid >= 0).all(), "Parkinson deve ser sempre positivo (valores não-NaN)"
    
    def test_realized_volatility(self, sample_ohlcv_data):
        """Testa volatilidade realizada."""
        real_vol = VolatilityEstimators.realized_volatility(sample_ohlcv_data, window=20)
        
        assert len(real_vol) == len(sample_ohlcv_data)
        real_vol_valid = real_vol.dropna()
        assert len(real_vol_valid) > 0, "Deve ter alguns valores válidos"
        assert (real_vol_valid >= 0).all(), "Volatilidade realizada deve ser sempre positiva (valores não-NaN)"


class TestAdaptiveLabeler:
    """Testes para o AdaptiveLabeler."""
    
    @pytest.fixture
    def sample_ohlcv_data(self):
        """Cria dados OHLCV de teste."""
        np.random.seed(42)
        n = 200  # Mais dados para testes de horizonte
        
        # Gerar série de preços com tendência e volatilidade
        base_price = 50000.0  # Bitcoin-like price
        trend = 0.0001  # Slight upward trend
        volatility = 0.02
        
        prices = [base_price]
        for i in range(n):
            shock = np.random.normal(trend, volatility)
            new_price = prices[-1] * (1 + shock)
            prices.append(new_price)
        
        prices = prices[1:]  # Remove first element
        
        # Criar OHLC realístico
        df = pd.DataFrame(index=pd.date_range('2024-01-01', periods=n, freq='15min'))
        df['open'] = prices
        df['close'] = np.roll(prices, -1)
        df['close'].iloc[-1] = df['open'].iloc[-1]  # Fix last row
        
        # High/Low com spread realístico
        spread = 0.001  # 0.1% spread típico
        for i in range(len(df)):
            o, c = df['open'].iloc[i], df['close'].iloc[i]
            high_shock = np.random.uniform(0, spread)
            low_shock = np.random.uniform(0, spread)
            
            df.loc[df.index[i], 'high'] = max(o, c) * (1 + high_shock)
            df.loc[df.index[i], 'low'] = min(o, c) * (1 - low_shock)
        
        df['volume'] = np.random.uniform(1000, 50000, n)
        
        return df
    
    def test_labeler_initialization(self):
        """Testa inicialização do labeler."""
        labeler = AdaptiveLabeler(
            horizon_bars=4,
            k=1.0,
            vol_estimator='atr',
            neutral_zone=True
        )
        
        assert labeler.horizon_bars == 4
        assert labeler.k == 1.0
        assert labeler.vol_estimator == 'atr'
        assert labeler.neutral_zone is True
        
        # Verificar mapeamento de horizontes
        assert labeler.horizon_map['60m'] == 4
        assert labeler.horizon_map['480m'] == 32
    
    def test_volatility_calculation(self, sample_ohlcv_data):
        """Testa cálculo de volatilidade."""
        labeler = AdaptiveLabeler(vol_estimator='atr')
        
        volatility = labeler.calculate_volatility(sample_ohlcv_data)
        
        assert len(volatility) == len(sample_ohlcv_data)
        assert (volatility > 0).any(), "Deve ter volatilidade positiva"
        assert not volatility.isna().all(), "Não deve ser completamente NaN"
    
    def test_adaptive_threshold_calculation(self, sample_ohlcv_data):
        """Testa cálculo de threshold adaptativo."""
        labeler = AdaptiveLabeler(horizon_bars=4, k=1.5)
        
        threshold = labeler.calculate_adaptive_threshold(sample_ohlcv_data)
        
        assert len(threshold) == len(sample_ohlcv_data)
        threshold_valid = threshold.dropna()
        assert len(threshold_valid) > 0, "Deve ter thresholds válidos"
        assert (threshold_valid >= labeler.min_threshold).all(), "Threshold deve respeitar mínimo"
        assert (threshold_valid <= labeler.max_threshold).all(), "Threshold deve respeitar máximo"
        
        # Threshold deve escalar com volatilidade
        volatility = labeler.calculate_volatility(sample_ohlcv_data)
        vol_valid = volatility.dropna()
        thresh_valid = threshold.dropna()
        
        # Alinhar índices para correlação
        common_idx = vol_valid.index.intersection(thresh_valid.index)
        if len(common_idx) > 10:  # Só calcular correlação se tiver dados suficientes
            correlation = np.corrcoef(thresh_valid.loc[common_idx], vol_valid.loc[common_idx])[0, 1]
            assert correlation > 0.3, f"Threshold deve correlacionar com volatilidade (r={correlation:.3f})"
    
    def test_label_creation_with_neutral_zone(self, sample_ohlcv_data):
        """Testa criação de labels com zona neutra."""
        labeler = AdaptiveLabeler(
            horizon_bars=4,
            k=1.0,
            neutral_zone=True
        )
        
        df_labeled, stats = labeler.create_labels(sample_ohlcv_data)
        
        # Verificar colunas adicionadas
        assert 'future_return' in df_labeled.columns
        assert 'threshold' in df_labeled.columns
        assert 'label' in df_labeled.columns
        
        # Verificar range de labels
        labels_clean = df_labeled['label'].dropna()
        unique_labels = labels_clean.unique()
        assert set(unique_labels).issubset({-1, 0, 1}), "Labels devem ser -1, 0, 1"
        
        # Deve ter pelo menos 2 classes
        assert len(unique_labels) >= 2, "Deve ter pelo menos 2 classes"
        
        # Verificar estatísticas
        assert stats['total_samples'] > 0
        assert 0 <= stats['long_rate'] <= 1
        assert 0 <= stats['short_rate'] <= 1
        assert 0 <= stats['neutral_rate'] <= 1
        assert abs(stats['long_rate'] + stats['short_rate'] + stats['neutral_rate'] - 1.0) < 1e-6
    
    def test_label_creation_without_neutral_zone(self, sample_ohlcv_data):
        """Testa criação de labels sem zona neutra (binário)."""
        labeler = AdaptiveLabeler(
            horizon_bars=4,
            k=1.0,
            neutral_zone=False
        )
        
        df_labeled, stats = labeler.create_labels(sample_ohlcv_data)
        
        # Verificar que só tem labels -1 e 1
        labels_clean = df_labeled['label'].dropna()
        unique_labels = set(labels_clean.unique())
        assert unique_labels == {-1, 1}, "Sem zona neutra deve ter apenas -1 e 1"
        
        # Taxa de neutros deve ser 0
        assert stats['neutral_rate'] == 0.0, "Neutral rate deve ser 0 sem zona neutra"
    
    def test_no_future_leakage(self, sample_ohlcv_data):
        """Verifica que não há vazamento de futuro."""
        labeler = AdaptiveLabeler(horizon_bars=8)
        
        df_labeled, stats = labeler.create_labels(sample_ohlcv_data)
        
        # Últimas 8 barras devem ser NaN
        last_labels = df_labeled['label'].iloc[-8:]
        assert last_labels.isna().all(), "Últimas barras devem ser NaN (sem futuro)"
        
        # Verificar que retornos futuros são calculados corretamente
        future_returns = df_labeled['future_return'].dropna()
        assert len(future_returns) >= len(sample_ohlcv_data) - 8
    
    def test_different_vol_estimators(self, sample_ohlcv_data):
        """Testa diferentes estimadores de volatilidade."""
        estimators = ['atr', 'yang_zhang', 'garman_klass', 'parkinson', 'realized']
        
        results = {}
        
        for estimator in estimators:
            labeler = AdaptiveLabeler(vol_estimator=estimator)
            df_labeled, stats = labeler.create_labels(sample_ohlcv_data)
            results[estimator] = stats
        
        # Todos devem produzir resultados válidos
        for estimator, stats in results.items():
            assert stats['total_samples'] > 0, f"Estimador {estimator} deve produzir amostras"
            assert len(set([stats['long_rate'], stats['short_rate'], stats['neutral_rate']])) > 1, \
                f"Estimador {estimator} deve produzir distribuição variada"
    
    def test_sample_weights_calculation(self, sample_ohlcv_data):
        """Testa cálculo de pesos de amostra."""
        labeler = AdaptiveLabeler()
        df_labeled, _ = labeler.create_labels(sample_ohlcv_data)
        
        # Testar diferentes métodos de peso
        for method in ['balanced', 'sqrt', 'none']:
            weights = labeler.calculate_sample_weights(df_labeled, method=method)
            
            assert len(weights) == len(df_labeled)
            assert (weights >= 0).all(), f"Pesos do método {method} devem ser não-negativos"
            
            if method == 'none':
                assert np.allclose(weights, 1.0), "Método 'none' deve retornar pesos unitários"
            else:
                # Pesos devem variar para métodos de balanceamento
                assert weights.std() > 0, f"Método {method} deve produzir pesos variados"
    
    def test_k_optimization(self, sample_ohlcv_data):
        """Testa otimização do parâmetro k."""
        labeler = AdaptiveLabeler()
        
        k_values = [0.5, 1.0, 1.5, 2.0]
        best_k, results = labeler.optimize_k(
            sample_ohlcv_data, 
            k_values, 
            metric='balanced_accuracy'
        )
        
        assert best_k in k_values, "Melhor k deve estar na lista testada"
        assert len(results) == len(k_values), "Deve ter resultado para cada k"
        
        # Verificar que o k foi setado
        assert labeler.k == best_k, "Labeler deve usar o melhor k encontrado"
        
        # Resultados devem ter métricas válidas
        for k, result in results.items():
            assert 'score' in result
            assert 'stats' in result
            assert 0 <= result['score'] <= 1, "Score deve estar entre 0 e 1"
    
    def test_edge_case_constant_prices(self):
        """Testa caso extremo com preços constantes."""
        # Criar dados com preços constantes
        n = 50
        constant_price = 100.0
        
        df = pd.DataFrame({
            'open': [constant_price] * n,
            'high': [constant_price] * n,
            'low': [constant_price] * n,
            'close': [constant_price] * n,
            'volume': [1000] * n
        }, index=pd.date_range('2024-01-01', periods=n, freq='15min'))
        
        labeler = AdaptiveLabeler(horizon_bars=4)
        df_labeled, stats = labeler.create_labels(df)
        
        # Com preços constantes, deve ter volatilidade próxima a zero
        volatility = labeler.calculate_volatility(df)
        vol_valid = volatility.dropna()
        if len(vol_valid) > 0:
            assert vol_valid.max() <= labeler.min_threshold * 2, \
                f"Volatilidade máxima deve ser baixa com preços constantes (max={vol_valid.max():.6f})"
        
        # Labels devem ser principalmente neutras
        labels_clean = df_labeled['label'].dropna()
        if len(labels_clean) > 0:
            neutral_rate = (labels_clean == 0).mean()
            assert neutral_rate >= 0.8, "Maioria deve ser neutra com preços constantes"
    
    def test_different_horizons(self, sample_ohlcv_data):
        """Testa diferentes horizontes de predição."""
        horizons = [1, 4, 8, 16]  # 15min, 1h, 2h, 4h
        
        results = {}
        
        for horizon in horizons:
            labeler = AdaptiveLabeler(horizon_bars=horizon)
            df_labeled, stats = labeler.create_labels(sample_ohlcv_data)
            results[horizon] = stats
        
        # Horizontes maiores devem ter threshold maiores (devido a sqrt scaling)
        labeler_short = AdaptiveLabeler(horizon_bars=1, k=1.0)
        labeler_long = AdaptiveLabeler(horizon_bars=16, k=1.0)
        
        thresh_short = labeler_short.calculate_adaptive_threshold(sample_ohlcv_data)
        thresh_long = labeler_long.calculate_adaptive_threshold(sample_ohlcv_data)
        
        # Threshold longo deve ser maior que curto (devido ao sqrt(horizon))
        ratio = thresh_long.mean() / thresh_short.mean()
        expected_ratio = np.sqrt(16 / 1)  # sqrt(16) / sqrt(1)
        
        assert 3 < ratio < 5, f"Ratio de threshold deve ser ~{expected_ratio}, got {ratio}"
    
    def test_statistics_consistency(self, sample_ohlcv_data):
        """Testa consistência das estatísticas."""
        labeler = AdaptiveLabeler()
        df_labeled, stats = labeler.create_labels(sample_ohlcv_data)
        
        # Verificar que contagens batem com rates
        expected_total = stats['long_count'] + stats['short_count'] + stats['neutral_count']
        assert expected_total == stats['total_samples'], "Contagens devem somar ao total"
        
        # Verificar rates
        expected_long_rate = stats['long_count'] / stats['total_samples']
        expected_short_rate = stats['short_count'] / stats['total_samples']
        expected_neutral_rate = stats['neutral_count'] / stats['total_samples']
        
        assert abs(stats['long_rate'] - expected_long_rate) < 1e-10
        assert abs(stats['short_rate'] - expected_short_rate) < 1e-10
        assert abs(stats['neutral_rate'] - expected_neutral_rate) < 1e-10
        
        # Verificar threshold stats se presentes
        if 'threshold_mean' in stats:
            thresholds = df_labeled['threshold'].dropna()
            assert abs(stats['threshold_mean'] - thresholds.mean()) < 1e-10
            assert abs(stats['threshold_std'] - thresholds.std()) < 1e-10


class TestLegacyCompatibility:
    """Testes de compatibilidade com código legado."""
    
    def test_triple_barrier_alias(self):
        """Testa que o alias TripleBarrierLabeler ainda funciona."""
        from src.features.labels import TripleBarrierLabeler
        
        # Deve ser o mesmo que AdaptiveLabeler
        assert TripleBarrierLabeler is AdaptiveLabeler
        
        # Deve conseguir instanciar
        labeler = TripleBarrierLabeler()
        assert isinstance(labeler, AdaptiveLabeler)
    
    def test_backward_compatible_parameters(self):
        """Testa que parâmetros antigos ainda funcionam."""
        # Parâmetros do TripleBarrierLabeler mapeados para AdaptiveLabeler
        labeler = AdaptiveLabeler(
            horizon_bars=4,  # era max_holding_period
            k=1.0,          # era pt_multiplier
            vol_estimator='atr'  # era use_atr=True
        )
        
        assert labeler.horizon_bars == 4
        assert labeler.k == 1.0
        assert labeler.vol_estimator == 'atr'