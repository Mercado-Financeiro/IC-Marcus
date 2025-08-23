"""Testes para feature engineering e data quality."""

import pytest
import numpy as np
import pandas as pd
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.features.engineering import FeatureEngineer
from src.features.labels import AdaptiveLabeler


class TestFeatureEngineering:
    """Testes para feature engineering."""
    
    @pytest.fixture
    def sample_data(self):
        """Cria dados de teste."""
        np.random.seed(42)
        n = 1000
        df = pd.DataFrame({
            'open': np.random.randn(n) * 10 + 100,
            'high': np.random.randn(n) * 10 + 105,
            'low': np.random.randn(n) * 10 + 95,
            'close': np.random.randn(n) * 10 + 100,
            'volume': np.abs(np.random.randn(n) * 1000 + 10000)
        })
        df.index = pd.date_range('2024-01-01', periods=n, freq='15min')
        return df
    
    @pytest.fixture
    def edge_case_data(self):
        """Dados com casos extremos."""
        df = pd.DataFrame({
            'open': [100, 0, 100, -100, np.inf],
            'high': [105, 105, 0, 105, 105],
            'low': [95, 95, 95, 0, -np.inf],
            'close': [100, 100, 0, 100, 100],
            'volume': [1000, 0, -1000, np.inf, np.nan]
        })
        df.index = pd.date_range('2024-01-01', periods=5, freq='15min')
        return df
    
    def test_no_nan_or_inf_in_features(self, sample_data):
        """Verifica que features não têm NaN ou Inf após limpeza."""
        fe = FeatureEngineer(
            lookback_periods=[5, 10, 20],
            technical_indicators=['rsi', 'macd']
        )
        
        df_features = fe.create_all_features(sample_data)
        
        # Remove NaN/Inf como no pipeline
        df_features = df_features.replace([np.inf, -np.inf], np.nan)
        df_features = df_features.dropna()
        
        # Verificar que não há NaN ou Inf restantes
        assert not df_features.isna().any().any(), "Features contêm NaN"
        assert not np.isinf(df_features.values).any(), "Features contêm Inf"
        assert len(df_features) > 0, "Nenhuma amostra restante após limpeza"
    
    def test_division_by_zero_protection(self, edge_case_data):
        """Testa proteção contra divisão por zero."""
        fe = FeatureEngineer(lookback_periods=[2])
        
        # Não deve lançar exceção
        df_features = fe.create_all_features(edge_case_data)
        
        # Verificar que price_to_sma não tem Inf
        price_to_sma_cols = [col for col in df_features.columns if 'price_to_sma' in col]
        for col in price_to_sma_cols:
            assert not np.isinf(df_features[col].values).any(), f"{col} contém Inf"
    
    def test_log_negative_protection(self, edge_case_data):
        """Testa proteção contra log de valores negativos."""
        fe = FeatureEngineer()
        
        # Não deve lançar exceção
        df_features = fe.create_all_features(edge_case_data)
        
        # Verificar que returns não tem NaN de log negativo
        if 'returns' in df_features.columns:
            # Pode ter NaN do shift, mas não deve ter -inf
            non_nan = df_features['returns'].dropna()
            assert not np.isinf(non_nan.values).any(), "Returns contém Inf de log negativo"
    
    def test_feature_consistency(self, sample_data):
        """Testa que features são consistentes entre runs."""
        fe = FeatureEngineer(lookback_periods=[10], technical_indicators=['rsi'])
        
        # Criar features duas vezes
        df1 = fe.create_all_features(sample_data.copy())
        df2 = fe.create_all_features(sample_data.copy())
        
        # Devem ser idênticas
        pd.testing.assert_frame_equal(df1, df2, check_dtype=False)
    
    def test_temporal_ordering_preserved(self, sample_data):
        """Verifica que ordem temporal é preservada."""
        fe = FeatureEngineer()
        df_features = fe.create_all_features(sample_data)
        
        # Index deve continuar ordenado
        assert df_features.index.is_monotonic_increasing, "Ordem temporal violada"
    
    def test_scaler_fit_transform_separate(self, sample_data):
        """Testa que fit e transform funcionam separadamente."""
        fe = FeatureEngineer(scaler_type='robust')
        
        # Split temporal
        split_idx = len(sample_data) // 2
        train_data = sample_data.iloc[:split_idx]
        test_data = sample_data.iloc[split_idx:]
        
        # Fit no treino
        train_features = fe.fit_transform(train_data)
        
        # Transform no teste (sem fit)
        test_features = fe.transform(test_data)
        
        # Verificar que funcionou
        assert len(train_features) > 0
        assert len(test_features) > 0
        assert fe.scaler is not None, "Scaler não foi fitado"


class TestTripleBarrier:
    """Testes para Triple Barrier labeling."""
    
    @pytest.fixture
    def price_data(self):
        """Cria série de preços."""
        np.random.seed(42)
        prices = 100 * np.exp(np.cumsum(np.random.randn(1000) * 0.01))
        return pd.Series(
            prices,
            index=pd.date_range('2024-01-01', periods=1000, freq='15min')
        )
    
    def test_label_distribution(self, price_data):
        """Verifica distribuição de labels."""
        labeler = AdaptiveLabeler(
            horizon_bars=4,
            k=1.0,
            vol_estimator='atr'
        )
        
        df = pd.DataFrame({
            'open': price_data * 0.99,
            'high': price_data * 1.01,
            'low': price_data * 0.98,
            'close': price_data,
            'volume': [1000] * len(price_data)
        })
        df_labeled, stats = labeler.create_labels(df)
        
        # Deve ter pelo menos 2 classes
        unique_labels = df_labeled['label'].dropna().unique()
        assert len(unique_labels) >= 2, f"Apenas {len(unique_labels)} classe(s) encontrada(s)"
        
        # Estatísticas devem ter informações válidas
        assert stats['total_samples'] > 0, "Nenhuma amostra processada"
        assert stats['long_rate'] + stats['short_rate'] + stats['neutral_rate'] == 1.0, "Rates não somam 1"
    
    def test_binary_conversion(self, price_data):
        """Testa conversão para binário."""
        labeler = AdaptiveLabeler(neutral_zone=False)  # Sem zona neutra = binário
        
        df = pd.DataFrame({
            'open': price_data * 0.99,
            'high': price_data * 1.01,
            'low': price_data * 0.98,
            'close': price_data,
            'volume': [1000] * len(price_data)
        })
        df_labeled, stats = labeler.create_labels(df)
        
        # Verificar conversão
        labels_clean = df_labeled['label'].dropna()
        assert labels_clean.isin([-1, 1]).all(), "Labels não são binárias (-1, 1)"
        assert len(labels_clean) > 0, "Nenhum label válido gerado"
        
        # Com neutral_zone=False, não deve ter neutros
        assert (labels_clean == 0).sum() == 0, "Ainda há labels neutras com neutral_zone=False"
    
    def test_no_future_leakage(self, price_data):
        """Verifica que não há vazamento do futuro."""
        labeler = AdaptiveLabeler(horizon_bars=10)
        
        df = pd.DataFrame({
            'open': price_data * 0.99,
            'high': price_data * 1.01,
            'low': price_data * 0.98,
            'close': price_data,
            'volume': [1000] * len(price_data)
        })
        df_labeled, stats = labeler.create_labels(df)
        
        # Labels devem ser NaN para as últimas horizon_bars barras
        last_labels = df_labeled['label'].iloc[-10:]
        assert last_labels.isna().sum() > 0, "Labels calculadas muito perto do fim (possível leak)"


class TestDataQuality:
    """Testes de qualidade de dados."""
    
    def test_remove_nan_inf_pipeline(self):
        """Testa pipeline de limpeza de NaN/Inf."""
        df = pd.DataFrame({
            'feat1': [1, 2, np.nan, 4, 5],
            'feat2': [1, np.inf, 3, 4, 5],
            'feat3': [1, 2, 3, -np.inf, 5],
            'label': [0, 1, 0, 1, 0]
        })
        
        # Pipeline de limpeza
        initial_len = len(df)
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.dropna()
        
        # Verificações
        assert len(df) == 2, f"Esperava 2 linhas limpas, obteve {len(df)}"
        assert not df.isna().any().any()
        assert not np.isinf(df.values).any()
    
    def test_sufficient_data_after_cleaning(self):
        """Verifica que resta dados suficientes após limpeza."""
        # Dados com muitos NaN
        df = pd.DataFrame({
            'feat1': [np.nan] * 95 + [1] * 5,
            'feat2': [1] * 100,
            'label': [0, 1] * 50
        })
        
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.dropna()
        
        # Deve ter pelo menos algumas amostras
        assert len(df) >= 5, f"Apenas {len(df)} amostras após limpeza"
    
    def test_feature_variance_check(self):
        """Testa detecção de features constantes."""
        df = pd.DataFrame({
            'constant_feat': [1] * 100,
            'normal_feat': np.random.randn(100),
            'label': np.random.randint(0, 2, 100)
        })
        
        # Detectar features constantes
        constant_features = df.columns[df.std() == 0].tolist()
        
        assert 'constant_feat' in constant_features
        assert 'normal_feat' not in constant_features
        assert 'label' not in constant_features


if __name__ == "__main__":
    pytest.main([__file__, "-v"])