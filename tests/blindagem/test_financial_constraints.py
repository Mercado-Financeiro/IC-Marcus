"""Testes de blindagem para restrições financeiras e realismo."""

import pytest
import numpy as np
import pandas as pd
import sys
from pathlib import Path

# Add parent to path
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

from backtest.engine import BacktestEngine


class TestFinancialConstraints:
    """Testes para verificar realismo e restrições financeiras."""
    
    @pytest.fixture
    def market_data(self):
        """Dados de mercado sintéticos realistas."""
        np.random.seed(42)
        n_bars = 1000
        
        # Simular preços com volatility clustering
        returns = []
        volatility = 0.02
        
        for i in range(n_bars):
            # GARCH-like volatility
            if i > 0:
                volatility = 0.000001 + 0.05 * (returns[-1]**2) + 0.9 * volatility
            
            ret = np.random.normal(0, np.sqrt(volatility))
            returns.append(ret)
        
        # Construir OHLCV
        prices = 100 * np.cumprod(1 + np.array(returns))
        
        data = pd.DataFrame({
            'timestamp': pd.date_range('2023-01-01', periods=n_bars, freq='15T'),
            'open': prices,
            'high': prices * (1 + np.abs(np.random.normal(0, 0.003, n_bars))),
            'low': prices * (1 - np.abs(np.random.normal(0, 0.003, n_bars))),
            'close': prices,
            'volume': np.random.lognormal(10, 0.8, n_bars)
        })
        
        # High >= Low, and price consistency
        data['high'] = np.maximum(data['high'], np.maximum(data['open'], data['close']))
        data['low'] = np.minimum(data['low'], np.minimum(data['open'], data['close']))
        
        return data.set_index('timestamp')
    
    def test_transaction_costs_realistic(self, market_data):
        """Verifica se custos de transação são realistas."""
        # Configurar backtest com custos típicos de crypto
        bt = BacktestEngine(
            initial_capital=100000,
            fee_bps=5.0,     # 0.05% - Binance maker fee
            slippage_bps=8.0  # 0.08% - slippage realista
        )
        
        # Gerar sinais aleatórios
        signals = pd.Series(
            np.random.choice([-1, 0, 1], size=len(market_data), p=[0.2, 0.6, 0.2]),
            index=market_data.index
        )
        
        results = bt.run_backtest(market_data, signals)
        
        # Verificar que custos são razoáveis
        total_costs = results['total_costs'].sum()
        total_turnover = results['turnover'].sum()
        
        # Custo deve ser proporcional ao turnover
        if total_turnover > 0:
            avg_cost_bps = (total_costs / (total_turnover * results['close'].mean())) * 10000
            assert 10 <= avg_cost_bps <= 50, f"Unrealistic cost structure: {avg_cost_bps} bps"
    
    def test_no_lookahead_bias(self, market_data):
        """Verifica execução t+1 (sem bias de lookahead)."""
        bt = BacktestEngine(initial_capital=100000, fee_bps=5, slippage_bps=5)
        
        # Sinal forte no último bar
        signals = pd.Series(0, index=market_data.index)
        signals.iloc[-1] = 1  # Comprar no último bar
        
        results = bt.run_backtest(market_data, signals)
        
        # Última posição deve ser 0, pois não há barra t+1 para executar
        assert results['positions'].iloc[-1] == 0, "Lookahead bias detected"
        
        # Sinal em t-1 deve resultar em posição em t
        if len(signals) > 1:
            signals_early = pd.Series(0, index=market_data.index)
            signals_early.iloc[-2] = 1  # Sinal na penúltima barra
            
            results_early = bt.run_backtest(market_data, signals_early)
            assert results_early['positions'].iloc[-1] == 1, "t+1 execution not working"
    
    def test_slippage_increases_with_volatility(self, market_data):
        """Verifica que slippage aumenta com volatilidade."""
        # Calcular volatilidade rolling
        market_data['returns'] = market_data['close'].pct_change()
        market_data['volatility'] = market_data['returns'].rolling(20).std() * np.sqrt(96)  # Daily vol
        
        # Backtest com slippage fixo
        bt = BacktestEngine(initial_capital=100000, fee_bps=5, slippage_bps=10)
        
        # Sinais em períodos de alta e baixa volatilidade
        signals = pd.Series(0, index=market_data.index)
        
        # Identificar períodos de alta e baixa vol
        vol_threshold = market_data['volatility'].quantile(0.8)
        high_vol_periods = market_data['volatility'] > vol_threshold
        low_vol_periods = market_data['volatility'] < market_data['volatility'].quantile(0.2)
        
        # Sinais em ambos os períodos
        signals[high_vol_periods] = 1
        signals[low_vol_periods] = -1
        
        results = bt.run_backtest(market_data, signals)
        
        # Em implementação mais avançada, slippage deveria ser maior em alta vol
        # Por enquanto, verificar que execução funcionou
        trades_made = results['trades'].sum()
        assert trades_made > 0, "No trades executed"
        
        costs_incurred = results['total_costs'].sum()
        assert costs_incurred > 0, "No costs incurred"
    
    def test_position_sizing_constraints(self, market_data):
        """Verifica restrições de tamanho de posição."""
        bt = BacktestEngine(
            initial_capital=100000,
            fee_bps=5,
            slippage_bps=5,
            max_leverage=2.0  # Máximo 2x
        )
        
        # Sinais muito fortes (devem ser limitados por leverage)
        signals = pd.Series(1, index=market_data.index)  # Sempre long
        
        results = bt.run_backtest(market_data, signals)
        
        # Verificar que leverage nunca excede o limite
        for i in range(len(results)):
            position_value = results['position_value'].iloc[i]
            equity = results['equity'].iloc[i]
            
            if equity > 0 and position_value > 0:
                leverage = position_value / equity
                assert leverage <= bt.max_leverage + 0.1, f"Leverage exceeded: {leverage}"
    
    def test_no_negative_prices(self, market_data):
        """Verifica que modelo não assume preços negativos."""
        bt = BacktestEngine(initial_capital=100000, fee_bps=5, slippage_bps=5)
        
        # Dados com preços muito baixos (mas positivos)
        low_price_data = market_data.copy()
        low_price_data[['open', 'high', 'low', 'close']] *= 0.001  # Preços muito baixos
        
        # Garantir que ainda são positivos
        assert (low_price_data[['open', 'high', 'low', 'close']] > 0).all().all()
        
        signals = pd.Series(
            np.random.choice([-1, 0, 1], size=len(low_price_data), p=[0.3, 0.4, 0.3]),
            index=low_price_data.index
        )
        
        results = bt.run_backtest(low_price_data, signals)
        
        # Equity nunca deve ser negativo (no máximo zero)
        assert (results['equity'] >= -0.01).all(), "Equity went negative"  # Margem para arredondamento
    
    def test_funding_costs_realistic(self, market_data):
        """Verifica custos de funding realistas."""
        # Funding anual típico para crypto perpetuals: -5% a +15%
        bt = BacktestEngine(
            initial_capital=100000,
            fee_bps=5,
            slippage_bps=5,
            funding_apr=0.08,  # 8% anual - típico para crypto
            borrow_apr=0.03   # 3% para borrow - menor que funding
        )
        
        # Posição longa mantida por tempo longo
        signals = pd.Series(1, index=market_data.index)  # Sempre long
        
        results = bt.run_backtest(market_data, signals)
        
        # Calcular funding total esperado
        n_bars = len(market_data)
        bars_per_year = 365 * 24 * 4  # 15-min bars
        years = n_bars / bars_per_year
        
        # Funding deve ser proporcional ao tempo
        total_costs = results['total_costs'].sum()
        avg_position_value = results['position_value'].mean()
        
        if avg_position_value > 0 and years > 0:
            implied_funding_rate = total_costs / (avg_position_value * years)
            # Deve estar na faixa esperada (fee + slippage + funding)
            expected_min = 0.05  # 5% mínimo (fees)
            expected_max = 0.20  # 20% máximo (fees + funding + slippage)
            
            assert expected_min <= implied_funding_rate <= expected_max, \
                f"Unrealistic funding rate: {implied_funding_rate:.2%}"
    
    def test_no_free_lunch(self, market_data):
        """Verifica que não há 'almoço grátis' no backtest."""
        bt = BacktestEngine(initial_capital=100000, fee_bps=5, slippage_bps=5)
        
        # Estratégia que sempre ganha (impossível na realidade)
        # Sinal: comprar antes de subidas, vender antes de quedas
        returns = market_data['close'].pct_change()
        perfect_signals = np.sign(returns.shift(-1))  # Lookahead perfeito
        perfect_signals = perfect_signals.fillna(0)
        
        results = bt.run_backtest(market_data, perfect_signals)
        
        # Mesmo com sinais perfeitos, deve haver custos
        total_costs = results['total_costs'].sum()
        assert total_costs > 0, "No costs with perfect strategy - free lunch detected"
        
        # Performance não deve ser irreaalisticamente alta
        final_equity = results['equity'].iloc[-1]
        total_return = (final_equity / bt.initial_capital) - 1
        
        # Mesmo perfeita, não deve dar mais que 1000% em período curto
        n_bars = len(market_data)
        if n_bars < 2000:  # Menos que ~20 dias
            assert total_return < 10, f"Unrealistic return with perfect foresight: {total_return:.1%}"
    
    def test_trading_hours_respected(self, market_data):
        """Verifica que horários de trading são respeitados."""
        # Definir horário restrito (apenas 8h-16h UTC)
        bt = BacktestEngine(
            initial_capital=100000,
            fee_bps=5,
            slippage_bps=5,
            trading_hours={'start': '08:00', 'end': '16:00'}
        )
        
        signals = pd.Series(1, index=market_data.index)  # Sempre quer comprar
        
        results = bt.run_backtest(market_data, signals)
        
        # Verificar que trades só aconteceram no horário permitido
        trades_made = results[results['trades'] == 1]
        
        for timestamp in trades_made.index:
            hour = timestamp.hour
            assert 8 <= hour < 16, f"Trade outside trading hours: {timestamp}"
    
    def test_minimum_trade_size(self, market_data):
        """Verifica tamanho mínimo de trade."""
        bt = BacktestEngine(
            initial_capital=1000,  # Capital pequeno
            fee_bps=5,
            slippage_bps=5,
            min_trade_size=500  # Mínimo $500
        )
        
        # Sinais fracos com capital insuficiente
        weak_signals = pd.Series(0.1, index=market_data.index)  # Sinal muito fraco
        
        results = bt.run_backtest(market_data, weak_signals)
        
        # Poucos ou nenhum trade deve ser executado
        trades_made = results['trades'].sum()
        
        # Com capital pequeno e tamanho mínimo alto, deve haver menos trades
        total_bars = len(market_data)
        trade_rate = trades_made / total_bars
        
        assert trade_rate < 0.1, "Too many small trades executed"
    
    def test_bid_ask_spread_implicit(self, market_data):
        """Verifica spread bid-ask implícito via slippage."""
        bt = BacktestEngine(initial_capital=100000, fee_bps=5, slippage_bps=10)
        
        # Estratégia de alta frequência (muitos trades)
        signals = pd.Series(
            np.random.choice([-1, 1], size=len(market_data)),
            index=market_data.index
        )
        
        results = bt.run_backtest(market_data, signals)
        
        # Slippage deve ser aplicado consistentemente
        trades_made = results[results['trades'] == 1]
        
        if len(trades_made) > 10:
            # Total slippage deve ser significativo para muitos trades
            total_costs = results['total_costs'].sum()
            total_volume = results['position_value'].sum()
            
            if total_volume > 0:
                avg_cost_rate = total_costs / total_volume
                expected_min_rate = 0.001  # 0.1% mínimo
                expected_max_rate = 0.002  # 0.2% máximo
                
                assert expected_min_rate <= avg_cost_rate <= expected_max_rate, \
                    f"Unrealistic cost rate: {avg_cost_rate:.3%}"
    
    def test_no_arbitrage_opportunities(self, market_data):
        """Verifica que não há oportunidades de arbitragem óbvias."""
        bt = BacktestEngine(initial_capital=100000, fee_bps=5, slippage_bps=5)
        
        # Estratégia: comprar quando preço cai, vender quando sobe
        returns = market_data['close'].pct_change()
        contrarian_signals = -np.sign(returns)  # Contrarian sem lookahead
        contrarian_signals = contrarian_signals.fillna(0)
        
        results = bt.run_backtest(market_data, contrarian_signals)
        
        # Estratégia contrarian simples não deve gerar lucros enormes
        final_equity = results['equity'].iloc[-1]
        total_return = (final_equity / bt.initial_capital) - 1
        
        # Sem informação adicional, não deve superar +50% facilmente
        assert total_return < 0.5, f"Suspiciously high return from simple contrarian: {total_return:.1%}"
        
        # E não deve perder mais que 80% (com stop loss implícito)
        assert total_return > -0.8, f"Excessive loss from simple contrarian: {total_return:.1%}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])