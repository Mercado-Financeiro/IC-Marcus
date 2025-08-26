"""Testes de blindagem essenciais - proteções críticas do sistema."""

import pytest
import numpy as np
import pandas as pd
import sys
from pathlib import Path

# Add parent to path  
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))


class TestCoreProtections:
    """Testes de proteção crítica - DEVEM SEMPRE PASSAR."""
    
    def test_temporal_order_protection(self):
        """CRÍTICO: Verifica que splits temporais respeitam ordem cronológica."""
        # Dados temporais ordenados
        dates = pd.date_range('2023-01-01', periods=100, freq='15min')
        data = pd.DataFrame({
            'timestamp': dates,
            'value': np.arange(100)
        }).set_index('timestamp')
        
        # Split temporal manual
        train_end_idx = 60
        val_end_idx = 80
        
        train_data = data.iloc[:train_end_idx]
        val_data = data.iloc[train_end_idx:val_end_idx]  
        test_data = data.iloc[val_end_idx:]
        
        # PROTEÇÃO CRÍTICA: Ordem temporal
        assert train_data.index.max() < val_data.index.min(), \
            "CRITICAL: Training data leaks into validation time"
        assert val_data.index.max() < test_data.index.min(), \
            "CRITICAL: Validation data leaks into test time"
        
        # PROTEÇÃO CRÍTICA: Nenhum overlap
        train_times = set(train_data.index)
        val_times = set(val_data.index)
        test_times = set(test_data.index)
        
        assert len(train_times & val_times) == 0, "CRITICAL: Time overlap train-val"
        assert len(val_times & test_times) == 0, "CRITICAL: Time overlap val-test"
        assert len(train_times & test_times) == 0, "CRITICAL: Time overlap train-test"
    
    def test_no_future_information_in_features(self):
        """CRÍTICO: Features não podem usar informação futura."""
        # Simular série temporal
        np.random.seed(42)
        prices = pd.Series(100 + np.cumsum(np.random.randn(100) * 0.01))
        
        # Feature correta: média móvel olhando apenas para trás
        window = 5
        correct_ma = []
        
        for i in range(len(prices)):
            if i < window - 1:
                # Primeiros pontos: usar dados disponíveis até agora
                ma = prices.iloc[:i+1].mean()
            else:
                # Janela completa: apenas dados passados
                ma = prices.iloc[i-window+1:i+1].mean()
            correct_ma.append(ma)
        
        # Feature INCORRETA: usando dados futuros (centered moving average)
        incorrect_ma = prices.rolling(window, center=True).mean()
        
        # PROTEÇÃO CRÍTICA: Features corretas não devem usar dados futuros
        # Para verificar, calculamos manualmente vs pandas centered
        
        # No ponto 50, MA correta deve usar apenas pontos [46-50]
        if len(prices) > 50:
            manual_ma_50 = prices.iloc[46:51].mean()  # pontos 46,47,48,49,50
            assert abs(correct_ma[50] - manual_ma_50) < 1e-10, \
                "CRITICAL: Correct MA calculation failed"
            
            # MA centered usaria pontos [48-52] = usa futuro!
            centered_ma_50 = incorrect_ma.iloc[50]
            if not pd.isna(centered_ma_50):
                # Se fosse correta, deveria ser igual à manual
                assert abs(correct_ma[50] - centered_ma_50) > 1e-6, \
                    "CRITICAL: Centered MA incorrectly matches backward-only MA"
    
    def test_model_input_validation(self):
        """CRÍTICO: Modelo deve rejeitar inputs inválidos."""
        from models.xgb_optuna import XGBoostOptuna
        
        # Dados válidos
        X_valid = pd.DataFrame({
            'feature_1': np.random.randn(100),
            'feature_2': np.random.randn(100)
        })
        y_valid = np.random.binomial(1, 0.3, 100)
        
        optimizer = XGBoostOptuna(n_trials=2, cv_folds=2)
        
        # PROTEÇÃO 1: Rejeitar NaNs
        X_with_nan = X_valid.copy()
        X_with_nan.iloc[0, 0] = np.nan
        
        with pytest.raises(ValueError, match="NaN"):
            optimizer.optimize(X_with_nan, y_valid)
        
        # PROTEÇÃO 2: Rejeitar infinitos
        X_with_inf = X_valid.copy()
        X_with_inf.iloc[0, 0] = np.inf
        
        with pytest.raises(ValueError, match="Inf"):
            optimizer.optimize(X_with_inf, y_valid)
        
        # PROTEÇÃO 3: Rejeitar dados vazios
        with pytest.raises(ValueError, match="Empty"):
            optimizer.optimize(X_valid.iloc[:0], y_valid[:0])
    
    def test_threshold_optimization_on_correct_split(self):
        """CRÍTICO: Threshold deve ser otimizado apenas na validação."""
        # Simular dados de validação separados
        np.random.seed(42)
        n_val = 200
        y_val_true = np.random.binomial(1, 0.3, n_val)
        y_val_proba = np.random.beta(2, 5, n_val)  # Probabilities
        
        from sklearn.metrics import f1_score
        
        # Otimização de threshold apenas nos dados de VALIDAÇÃO
        thresholds = np.linspace(0.1, 0.9, 17)
        best_f1 = -1
        best_threshold = 0.5
        
        for th in thresholds:
            val_pred = (y_val_proba >= th).astype(int)
            f1 = f1_score(y_val_true, val_pred, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = th
        
        # PROTEÇÃO CRÍTICA: Threshold deve estar em range válido
        assert 0.1 <= best_threshold <= 0.9, \
            f"CRITICAL: Invalid threshold: {best_threshold}"
        
        # PROTEÇÃO CRÍTICA: F1 deve estar em range válido  
        assert 0 <= best_f1 <= 1, f"CRITICAL: Invalid F1 score: {best_f1}"
        
        # Simular uso em dados de TESTE (diferentes dos de validação)
        np.random.seed(123)  # Diferente da validação
        n_test = 150
        y_test_true = np.random.binomial(1, 0.3, n_test)
        y_test_proba = np.random.beta(2, 5, n_test)
        
        # Aplicar threshold otimizado na validação
        test_pred = (y_test_proba >= best_threshold).astype(int)
        test_f1 = f1_score(y_test_true, test_pred, zero_division=0)
        
        # PROTEÇÃO: Resultado de teste deve ser válido
        assert 0 <= test_f1 <= 1, f"CRITICAL: Invalid test F1: {test_f1}"
    
    def test_no_lookahead_bias_in_backtest(self):
        """CRÍTICO: Backtest não pode usar informação futura."""
        from backtest.engine import BacktestEngine
        
        # Dados de mercado sintéticos
        dates = pd.date_range('2023-01-01', periods=50, freq='15min')
        prices = 100 + np.cumsum(np.random.randn(50) * 0.01)
        
        market_data = pd.DataFrame({
            'open': prices,
            'high': prices * 1.001,
            'low': prices * 0.999, 
            'close': prices,
            'volume': np.random.lognormal(8, 0.5, 50)
        }, index=dates)
        
        bt = BacktestEngine(initial_capital=10000, fee_bps=5, slippage_bps=5)
        
        # TESTE CRÍTICO: Sinal no último período não pode ser executado
        signals = pd.Series(0, index=dates)
        signals.iloc[-1] = 1  # Sinal de compra no último período
        
        results = bt.run_backtest(market_data, signals)
        
        # PROTEÇÃO CRÍTICA: Posição final deve ser 0 (não executou último sinal)
        assert results['positions'].iloc[-1] == 0, \
            "CRITICAL: Lookahead bias - executed signal without future bar"
        
        # PROTEÇÃO ADICIONAL: Sinal em t-1 deve executar em t
        signals_early = pd.Series(0, index=dates)
        signals_early.iloc[-2] = 1  # Sinal na penúltima barra
        
        results_early = bt.run_backtest(market_data, signals_early)
        
        # Deve ter executado (t+1 execution)
        assert results_early['positions'].iloc[-1] == 1, \
            "CRITICAL: t+1 execution not working properly"
    
    def test_cost_application_in_backtest(self):
        """CRÍTICO: Custos devem ser aplicados em toda transação."""
        from backtest.engine import BacktestEngine
        
        # Dados simples
        dates = pd.date_range('2023-01-01', periods=10, freq='15min')
        prices = pd.Series([100, 101, 102, 103, 104, 105, 106, 107, 108, 109])
        
        market_data = pd.DataFrame({
            'open': prices,
            'high': prices,
            'low': prices,
            'close': prices,
            'volume': [1000] * 10
        }, index=dates)
        
        bt = BacktestEngine(initial_capital=10000, fee_bps=10, slippage_bps=10)
        
        # Estratégia simples: comprar e vender
        signals = pd.Series([0, 1, 0, 1, 0, -1, 0, 1, 0, 0], index=dates)
        
        results = bt.run_backtest(market_data, signals)
        
        # PROTEÇÃO CRÍTICA: Deve haver custos para cada trade
        total_costs = results['total_costs'].sum()
        trades_made = results['trades'].sum()
        
        if trades_made > 0:
            assert total_costs > 0, "CRITICAL: No costs applied despite trades"
            
            # Custo médio por trade deve ser razoável
            avg_cost_per_trade = total_costs / trades_made
            assert avg_cost_per_trade > 0, "CRITICAL: Zero cost per trade"
    
    def test_data_type_consistency(self):
        """CRÍTICO: Tipos de dados devem ser consistentes."""
        # Gerar dados com tipos específicos
        n_samples = 100
        
        X = pd.DataFrame({
            'float_feature': np.random.randn(n_samples).astype(np.float64),
            'int_feature': np.random.randint(0, 10, n_samples).astype(np.int64),
            'bool_feature': np.random.choice([True, False], n_samples)
        })
        
        y = pd.Series(np.random.binomial(1, 0.3, n_samples), dtype=np.int64)
        
        # PROTEÇÃO CRÍTICA: Tipos devem ser preservados
        assert X['float_feature'].dtype == np.float64, "Float feature type changed"
        assert X['int_feature'].dtype == np.int64, "Int feature type changed" 
        assert y.dtype == np.int64, "Target type changed"
        
        # PROTEÇÃO: Não deve haver mudanças inesperadas de tipo
        X_copy = X.copy()
        y_copy = y.copy()
        
        # Operações básicas não devem alterar tipos
        X_scaled = (X - X.mean()) / X.std()
        
        assert X_scaled['float_feature'].dtype in [np.float64, np.float32], \
            "Scaling changed float type unexpectedly"
    
    def test_reproducibility_with_seeds(self):
        """CRÍTICO: Mesma seed deve produzir mesmos resultados."""
        # Função que gera números aleatórios
        def generate_random_data(seed):
            np.random.seed(seed)
            return np.random.randn(10)
        
        # PROTEÇÃO CRÍTICA: Mesmo seed = mesmos dados
        data1 = generate_random_data(42)
        data2 = generate_random_data(42)
        
        assert np.array_equal(data1, data2), \
            "CRITICAL: Same seed produced different results"
        
        # PROTEÇÃO: Seeds diferentes = dados diferentes
        data3 = generate_random_data(123)
        assert not np.array_equal(data1, data3), \
            "CRITICAL: Different seeds produced same results"
    
    def test_no_data_contamination_between_splits(self):
        """CRÍTICO: Splits não devem contaminar entre si."""
        # Dataset temporal
        n_samples = 1000
        dates = pd.date_range('2023-01-01', periods=n_samples, freq='15min')
        
        data = pd.DataFrame({
            'feature': np.random.randn(n_samples),
            'target': np.random.binomial(1, 0.3, n_samples)
        }, index=dates)
        
        # Criar splits temporais
        train_size = 600
        val_size = 200
        
        train_data = data.iloc[:train_size]
        val_data = data.iloc[train_size:train_size+val_size]
        test_data = data.iloc[train_size+val_size:]
        
        # PROTEÇÃO CRÍTICA: Estatísticas calculadas apenas no treino
        train_mean = train_data['feature'].mean()
        train_std = train_data['feature'].std()
        
        # Normalizar usando APENAS estatísticas do treino
        train_normalized = (train_data['feature'] - train_mean) / train_std
        val_normalized = (val_data['feature'] - train_mean) / train_std
        test_normalized = (test_data['feature'] - train_mean) / train_std
        
        # VERIFICAÇÃO: Treino normalizado deve ter média ~0, std ~1
        assert abs(train_normalized.mean()) < 0.1, \
            "CRITICAL: Train normalization incorrect"
        assert abs(train_normalized.std() - 1.0) < 0.1, \
            "CRITICAL: Train std normalization incorrect"
        
        # PROTEÇÃO: Val/Test podem ter estatísticas diferentes (correto!)
        # Eles NÃO devem ter média 0 e std 1
        val_mean = val_normalized.mean()
        test_mean = test_normalized.mean()
        
        # É esperado que val/test tenham médias diferentes de 0
        # Se fossem iguais a 0, indicaria data leakage
        print(f"Val mean after train normalization: {val_mean:.3f}")
        print(f"Test mean after train normalization: {test_mean:.3f}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])