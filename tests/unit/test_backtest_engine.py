"""Tests for backtest engine with t+1 execution."""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))


class TestBacktestEngine:
    """Test suite for backtest engine."""
    
    def test_initialization(self):
        """Test BacktestEngine initialization."""
        from src.backtest.engine import BacktestEngine
        
        engine = BacktestEngine(
            initial_capital=100000,
            fee_bps=5,
            slippage_bps=5,
            max_leverage=1.0
        )
        
        assert engine.initial_capital == 100000
        assert engine.fee_bps == 5
        assert engine.slippage_bps == 5
        assert engine.max_leverage == 1.0
    
    def test_t_plus_1_execution(self, sample_ohlcv_data):
        """Critical test: verify t+1 execution."""
        from src.backtest.engine import BacktestEngine
        
        engine = BacktestEngine()
        
        # Create signals (signal at t, execution at t+1)
        signals = pd.Series(0, index=sample_ohlcv_data.index)
        signals.iloc[10] = 1  # Buy signal at bar 10
        signals.iloc[20] = 0  # Exit signal at bar 20
        signals.iloc[30] = -1  # Short signal at bar 30
        
        results = engine.run_backtest(sample_ohlcv_data, signals)
        
        # Check that positions change at t+1
        positions = results['positions']
        
        # Signal at 10, position should change at 11
        assert positions.iloc[10] == 0, "Position changed at signal bar (should be t+1)"
        assert positions.iloc[11] == 1, "Position didn't change at t+1"
        
        # Signal at 20, position should change at 21
        assert positions.iloc[20] == 1, "Position changed early"
        assert positions.iloc[21] == 0, "Position didn't change at t+1"
        
        # Signal at 30, position should change at 31
        assert positions.iloc[30] == 0, "Position changed early"
        assert positions.iloc[31] == -1, "Position didn't change at t+1"
    
    def test_calculate_costs(self):
        """Test cost calculation."""
        from src.backtest.engine import BacktestEngine
        
        engine = BacktestEngine(fee_bps=10, slippage_bps=5)
        
        # Test single trade cost
        trade_value = 10000
        cost = engine.calculate_costs(trade_value, holding_period=1)
        
        # Fee + slippage = 15 bps = 0.15%
        expected_cost = trade_value * 0.0015
        assert abs(cost - expected_cost) < 0.01
        
        # Test with funding
        engine.funding_apr = 0.05  # 5% annual
        cost_with_funding = engine.calculate_costs(trade_value, holding_period=24)  # 1 day
        
        # Should include funding cost
        assert cost_with_funding > cost
    
    def test_position_sizing(self):
        """Test position sizing methods."""
        from src.backtest.engine import BacktestEngine
        
        engine = BacktestEngine(initial_capital=100000)
        
        # Fixed sizing
        size = engine.calculate_position_size(
            method='fixed',
            capital=100000,
            signal_strength=1.0
        )
        assert size > 0
        assert size <= 100000  # Can't exceed capital
        
        # Kelly sizing
        size_kelly = engine.calculate_position_size(
            method='kelly',
            capital=100000,
            signal_strength=0.6,  # 60% win probability
            kelly_fraction=0.25
        )
        assert size_kelly > 0
        assert size_kelly < size  # Kelly should be more conservative
        
        # Volatility targeting
        size_vol = engine.calculate_position_size(
            method='volatility_target',
            capital=100000,
            signal_strength=1.0,
            volatility=0.02,  # 2% daily vol
            vol_target=0.01  # 1% target
        )
        assert size_vol > 0
        assert size_vol < 100000
    
    def test_calculate_metrics(self, sample_ohlcv_data):
        """Test performance metrics calculation."""
        from src.backtest.engine import BacktestEngine
        
        engine = BacktestEngine()
        
        # Create simple buy-and-hold signals
        signals = pd.Series(1, index=sample_ohlcv_data.index)
        
        results = engine.run_backtest(sample_ohlcv_data, signals)
        metrics = engine.calculate_metrics(results)
        
        # Check required metrics
        required_metrics = [
            'total_return', 'annualized_return', 'volatility',
            'sharpe_ratio', 'sortino_ratio', 'calmar_ratio',
            'max_drawdown', 'max_drawdown_duration',
            'win_rate', 'profit_factor', 'dsr',
            'total_trades', 'turnover', 'total_costs'
        ]
        
        for metric in required_metrics:
            assert metric in metrics, f"Missing metric: {metric}"
        
        # Validate metric ranges
        assert -1 <= metrics['total_return'] <= 10  # Reasonable return
        assert -1 <= metrics['max_drawdown'] <= 0  # Drawdown is negative
        assert metrics['total_trades'] >= 0
        assert metrics['total_costs'] >= 0
    
    def test_deflated_sharpe_ratio(self):
        """Test DSR calculation."""
        from src.backtest.engine import BacktestEngine
        
        engine = BacktestEngine()
        
        # Test DSR calculation
        sharpe = 1.5
        n_trials = 100
        T = 252  # One year of data
        
        dsr = engine.calculate_dsr(sharpe, n_trials, T)
        
        # DSR should be lower than regular Sharpe
        assert dsr < sharpe
        assert dsr > 0
    
    def test_execution_at_open_price(self, sample_ohlcv_data):
        """Test that execution happens at open price of t+1."""
        from src.backtest.engine import BacktestEngine
        
        engine = BacktestEngine()
        
        # Create a single buy signal
        signals = pd.Series(0, index=sample_ohlcv_data.index)
        signals.iloc[10] = 1
        
        results = engine.run_backtest(sample_ohlcv_data, signals)
        
        # Entry should be at open price of bar 11
        entry_price = results['entry_prices'].iloc[11]
        expected_price = sample_ohlcv_data['open'].iloc[11]
        
        # Allow for slippage
        assert abs(entry_price - expected_price) / expected_price < 0.01
    
    def test_no_look_ahead_bias(self, sample_ohlcv_data):
        """Test that there's no look-ahead bias."""
        from src.backtest.engine import BacktestEngine
        
        engine = BacktestEngine()
        
        # Create signals based on future returns (intentional look-ahead)
        future_returns = sample_ohlcv_data['close'].pct_change().shift(-1)
        signals = (future_returns > 0).astype(int)
        
        # Even with perfect foresight signals, execution at t+1 should limit profits
        results = engine.run_backtest(sample_ohlcv_data, signals)
        
        # The backtest should not use future information
        # Positions should lag signals by 1 bar
        for i in range(1, len(results) - 1):
            if signals.iloc[i] != signals.iloc[i-1]:  # Signal change
                # Position shouldn't change immediately
                assert results['positions'].iloc[i] == results['positions'].iloc[i-1]
    
    def test_short_selling(self, sample_ohlcv_data):
        """Test short selling functionality."""
        from src.backtest.engine import BacktestEngine
        
        engine = BacktestEngine(allow_short=True)
        
        # Create short signals
        signals = pd.Series(0, index=sample_ohlcv_data.index)
        signals.iloc[10:20] = -1  # Short position
        
        results = engine.run_backtest(sample_ohlcv_data, signals)
        
        # Check short positions
        assert any(results['positions'] == -1), "No short positions created"
        
        # Short should profit from price decreases
        # (This is a simplified check)
        short_period = results.iloc[11:21]  # Execution at t+1
        if sample_ohlcv_data['close'].iloc[11] > sample_ohlcv_data['close'].iloc[20]:
            # Price decreased, short should profit
            assert short_period['pnl'].sum() > 0
    
    def test_max_leverage_constraint(self, sample_ohlcv_data):
        """Test that leverage constraints are respected."""
        from src.backtest.engine import BacktestEngine
        
        engine = BacktestEngine(
            initial_capital=100000,
            max_leverage=0.5  # Conservative 50% exposure
        )
        
        # Create full position signals
        signals = pd.Series(1, index=sample_ohlcv_data.index)
        
        results = engine.run_backtest(sample_ohlcv_data, signals)
        
        # Check that position size respects leverage
        max_position_value = results['position_value'].abs().max()
        assert max_position_value <= 100000 * 0.5 * 1.1  # Allow 10% buffer for P&L
    
    def test_trading_hours_filter(self, sample_ohlcv_data):
        """Test trading hours filtering."""
        from src.backtest.engine import BacktestEngine
        
        engine = BacktestEngine(
            trading_hours={'start': '09:00', 'end': '17:00'}
        )
        
        # Create signals at all hours
        signals = pd.Series(1, index=sample_ohlcv_data.index)
        
        results = engine.run_backtest(sample_ohlcv_data, signals)
        
        # Trades should only happen during trading hours
        trades = results[results['trades'] != 0]
        for timestamp in trades.index:
            hour = timestamp.hour
            assert 9 <= hour < 17, f"Trade outside hours at {timestamp}"


class TestBacktestEngineEdgeCases:
    """Test edge cases for backtest engine."""
    
    def test_empty_signals(self, sample_ohlcv_data):
        """Test with no signals (all zeros)."""
        from src.backtest.engine import BacktestEngine
        
        engine = BacktestEngine()
        signals = pd.Series(0, index=sample_ohlcv_data.index)
        
        results = engine.run_backtest(sample_ohlcv_data, signals)
        
        # Should have no trades
        assert results['positions'].sum() == 0
        assert results['pnl'].sum() == 0
        assert results['equity'].iloc[-1] == engine.initial_capital
    
    def test_single_bar_data(self):
        """Test with single bar of data."""
        from src.backtest.engine import BacktestEngine
        
        dates = pd.date_range('2023-01-01', periods=1, freq='1h', tz='UTC')
        df = pd.DataFrame({
            'open': [100],
            'high': [101],
            'low': [99],
            'close': [100],
            'volume': [1000]
        }, index=dates)
        
        signals = pd.Series([1], index=dates)
        
        engine = BacktestEngine()
        results = engine.run_backtest(df, signals)
        
        # Should handle gracefully
        assert len(results) == 1
        assert results['positions'].iloc[0] == 0  # Can't execute on single bar
    
    def test_all_long_signals(self, sample_ohlcv_data):
        """Test with all long signals."""
        from src.backtest.engine import BacktestEngine
        
        engine = BacktestEngine()
        signals = pd.Series(1, index=sample_ohlcv_data.index)
        
        results = engine.run_backtest(sample_ohlcv_data, signals)
        
        # Should be mostly long (after first bar for t+1)
        assert (results['positions'][1:] == 1).all()
    
    def test_rapid_signal_changes(self, sample_ohlcv_data):
        """Test with rapidly changing signals."""
        from src.backtest.engine import BacktestEngine
        
        engine = BacktestEngine()
        
        # Alternate between long and short every bar
        signals = pd.Series(
            [1, -1] * (len(sample_ohlcv_data) // 2),
            index=sample_ohlcv_data.index[:len(sample_ohlcv_data) // 2 * 2]
        )
        
        results = engine.run_backtest(sample_ohlcv_data[:len(signals)], signals)
        
        # High turnover should result in high costs
        assert results['total_costs'].sum() > 0
        assert results['turnover'].mean() > 0.5