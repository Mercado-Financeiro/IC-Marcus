"""Tests for Triple Barrier labeling method."""

import pytest
import numpy as np
import pandas as pd
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.features.labels import TripleBarrierLabeler


class TestTripleBarrierLabeler:
    """Test suite for Triple Barrier labeling."""
    
    def test_basic_labeling(self, sample_ohlcv_data):
        """Test basic triple barrier labeling."""
        labeler = TripleBarrierLabeler(
            pt_multiplier=2.0,
            sl_multiplier=1.5,
            max_holding_period=10
        )
        
        df, barrier_info = labeler.apply_triple_barrier(sample_ohlcv_data)
        
        # Check label column exists
        assert 'label' in df.columns, "Label column not created"
        assert 'meta_label' in df.columns, "Meta label column not created"
        
        # Check labels are valid
        unique_labels = df['label'].dropna().unique()
        assert all(label in [-1, 0, 1] for label in unique_labels), \
            f"Invalid labels found: {unique_labels}"
        
        # Check barrier info
        assert len(barrier_info) == len(df), "Barrier info length mismatch"
        
        # Check each barrier info has required fields
        required_fields = ['entry_idx', 'exit_idx', 'label', 'exit_reason']
        for info in barrier_info[:10]:  # Check first 10
            for field in required_fields:
                assert field in info, f"Missing field: {field}"
    
    def test_take_profit_hit(self):
        """Test that take profit barrier is correctly detected."""
        # Create synthetic data with strong upward movement
        dates = pd.date_range('2023-01-01', periods=100, freq='15min', tz='UTC')
        
        prices = [100.0]
        for i in range(1, 100):
            if i == 5:  # Strong upward movement
                prices.append(110.0)
            else:
                prices.append(prices[-1] + np.random.randn() * 0.1)
        
        df = pd.DataFrame({
            'open': prices,
            'high': [p + 0.5 for p in prices],
            'low': [p - 0.5 for p in prices],
            'close': prices,
            'volume': [1000] * 100
        }, index=dates)
        
        labeler = TripleBarrierLabeler(
            pt_multiplier=0.05,  # 5% take profit
            sl_multiplier=0.03,  # 3% stop loss
            max_holding_period=10,
            use_atr=False
        )
        
        df, barrier_info = labeler.apply_triple_barrier(df)
        
        # First position should hit take profit
        assert barrier_info[0]['label'] == 1, "Take profit not detected"
        assert barrier_info[0]['exit_reason'] == 'take_profit', \
            f"Wrong exit reason: {barrier_info[0]['exit_reason']}"
    
    def test_stop_loss_hit(self):
        """Test that stop loss barrier is correctly detected."""
        # Create synthetic data with strong downward movement
        dates = pd.date_range('2023-01-01', periods=100, freq='15min', tz='UTC')
        
        prices = [100.0]
        for i in range(1, 100):
            if i == 5:  # Strong downward movement
                prices.append(90.0)
            else:
                prices.append(prices[-1] + np.random.randn() * 0.1)
        
        df = pd.DataFrame({
            'open': prices,
            'high': [p + 0.5 for p in prices],
            'low': [p - 0.5 for p in prices],
            'close': prices,
            'volume': [1000] * 100
        }, index=dates)
        
        labeler = TripleBarrierLabeler(
            pt_multiplier=0.05,  # 5% take profit
            sl_multiplier=0.03,  # 3% stop loss
            max_holding_period=10,
            use_atr=False
        )
        
        df, barrier_info = labeler.apply_triple_barrier(df)
        
        # First position should hit stop loss
        assert barrier_info[0]['label'] == -1, "Stop loss not detected"
        assert barrier_info[0]['exit_reason'] == 'stop_loss', \
            f"Wrong exit reason: {barrier_info[0]['exit_reason']}"
    
    def test_max_holding_period(self):
        """Test that max holding period is respected."""
        # Create flat data (no significant movement)
        dates = pd.date_range('2023-01-01', periods=100, freq='15min', tz='UTC')
        
        df = pd.DataFrame({
            'open': [100.0] * 100,
            'high': [100.5] * 100,
            'low': [99.5] * 100,
            'close': [100.0] * 100,
            'volume': [1000] * 100
        }, index=dates)
        
        max_holding = 5
        labeler = TripleBarrierLabeler(
            pt_multiplier=0.10,  # 10% take profit (won't be hit)
            sl_multiplier=0.10,  # 10% stop loss (won't be hit)
            max_holding_period=max_holding,
            use_atr=False
        )
        
        df, barrier_info = labeler.apply_triple_barrier(df)
        
        # Check that exits happen at max holding period
        for i in range(min(10, len(barrier_info) - max_holding)):
            if barrier_info[i]['exit_reason'] == 'max_holding':
                assert barrier_info[i]['exit_idx'] - barrier_info[i]['entry_idx'] == max_holding, \
                    f"Max holding period not respected for entry {i}"
    
    def test_sample_weights(self, sample_ohlcv_data):
        """Test sample weight calculation."""
        labeler = TripleBarrierLabeler()
        
        df, barrier_info = labeler.apply_triple_barrier(sample_ohlcv_data)
        weights = labeler.calculate_sample_weights(df, barrier_info)
        
        # Check weights properties
        assert len(weights) == len(barrier_info), "Weights length mismatch"
        assert all(w > 0 for w in weights), "Negative or zero weights found"
        assert np.isclose(weights.mean(), len(weights) / len(weights)), \
            "Weights not properly normalized"
        
        # Check that overlapping events have lower weights
        # This is a simplified check
        assert weights.std() > 0, "All weights are identical"
    
    def test_with_side_prediction(self, sample_ohlcv_data):
        """Test labeling with side predictions."""
        labeler = TripleBarrierLabeler()
        
        # Create side predictions
        side = pd.Series(
            np.random.choice([-1, 0, 1], size=len(sample_ohlcv_data)),
            index=sample_ohlcv_data.index
        )
        
        df, barrier_info = labeler.apply_triple_barrier(sample_ohlcv_data, side)
        
        # Check that neutral positions (side=0) get neutral labels
        for i, s in enumerate(side):
            if s == 0 and i < len(barrier_info):
                assert barrier_info[i]['label'] == 0, \
                    f"Non-zero label for neutral position at {i}"
    
    def test_atr_calculation(self, sample_ohlcv_data):
        """Test that ATR is calculated when not present."""
        labeler = TripleBarrierLabeler(use_atr=True)
        
        # Remove ATR if it exists
        if 'atr_14' in sample_ohlcv_data.columns:
            sample_ohlcv_data = sample_ohlcv_data.drop('atr_14', axis=1)
        
        df, barrier_info = labeler.apply_triple_barrier(sample_ohlcv_data)
        
        # Check ATR was calculated
        assert 'atr_14' in df.columns, "ATR not calculated"
        assert df['atr_14'].notna().sum() > 0, "ATR is all NaN"
    
    def test_label_statistics(self, sample_ohlcv_data):
        """Test label statistics calculation."""
        labeler = TripleBarrierLabeler()
        
        df, _ = labeler.apply_triple_barrier(sample_ohlcv_data)
        stats = labeler.get_label_statistics(df)
        
        # Check required statistics
        required_stats = [
            'total_samples', 'long_wins', 'losses', 'neutrals',
            'long_win_rate', 'loss_rate', 'neutral_rate'
        ]
        
        for stat in required_stats:
            assert stat in stats, f"Missing statistic: {stat}"
        
        # Check rates sum to ~1
        total_rate = stats['long_win_rate'] + stats['loss_rate'] + stats['neutral_rate']
        assert np.isclose(total_rate, 1.0, rtol=0.01), \
            f"Rates don't sum to 1: {total_rate}"
    
    @pytest.mark.parametrize("method", ["momentum", "mean_reversion", "trend"])
    def test_side_prediction_methods(self, sample_ohlcv_data, method):
        """Test different side prediction methods."""
        labeler = TripleBarrierLabeler()
        
        side = labeler.create_side_prediction(sample_ohlcv_data, method=method)
        
        # Check side values are valid
        assert all(s in [-1, 0, 1] for s in side.dropna()), \
            f"Invalid side values for method {method}"
        
        # Check index matches
        assert len(side) == len(sample_ohlcv_data), "Side length mismatch"
    
    def test_reproducibility(self, sample_ohlcv_data):
        """Test that labeling is reproducible."""
        labeler = TripleBarrierLabeler(
            pt_multiplier=2.0,
            sl_multiplier=1.5,
            max_holding_period=10
        )
        
        # Run twice
        df1, barrier_info1 = labeler.apply_triple_barrier(sample_ohlcv_data.copy())
        df2, barrier_info2 = labeler.apply_triple_barrier(sample_ohlcv_data.copy())
        
        # Check labels are identical
        pd.testing.assert_series_equal(df1['label'], df2['label'])
        
        # Check barrier info is identical
        for info1, info2 in zip(barrier_info1[:10], barrier_info2[:10]):
            assert info1['label'] == info2['label']
            assert info1['exit_reason'] == info2['exit_reason']


class TestTripleBarrierEdgeCases:
    """Test edge cases for Triple Barrier labeling."""
    
    def test_empty_dataframe(self):
        """Test with empty DataFrame."""
        labeler = TripleBarrierLabeler()
        
        df = pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume'])
        
        # Should handle gracefully
        result_df, barrier_info = labeler.apply_triple_barrier(df)
        
        assert len(result_df) == 0
        assert len(barrier_info) == 0
    
    def test_single_row(self):
        """Test with single row of data."""
        dates = pd.date_range('2023-01-01', periods=1, freq='15min', tz='UTC')
        
        df = pd.DataFrame({
            'open': [100.0],
            'high': [101.0],
            'low': [99.0],
            'close': [100.5],
            'volume': [1000]
        }, index=dates)
        
        labeler = TripleBarrierLabeler()
        
        result_df, barrier_info = labeler.apply_triple_barrier(df)
        
        # Last row should always be neutral
        assert result_df['label'].iloc[-1] == 0
    
    def test_extreme_barriers(self):
        """Test with extreme barrier values."""
        dates = pd.date_range('2023-01-01', periods=100, freq='15min', tz='UTC')
        
        df = pd.DataFrame({
            'open': [100.0] * 100,
            'high': [101.0] * 100,
            'low': [99.0] * 100,
            'close': [100.0] * 100,
            'volume': [1000] * 100
        }, index=dates)
        
        # Very tight barriers
        labeler_tight = TripleBarrierLabeler(
            pt_multiplier=0.001,  # 0.1%
            sl_multiplier=0.001,  # 0.1%
            max_holding_period=100,
            use_atr=False
        )
        
        df_tight, _ = labeler_tight.apply_triple_barrier(df.copy())
        
        # Very wide barriers
        labeler_wide = TripleBarrierLabeler(
            pt_multiplier=1.0,  # 100%
            sl_multiplier=1.0,  # 100%
            max_holding_period=100,
            use_atr=False
        )
        
        df_wide, _ = labeler_wide.apply_triple_barrier(df.copy())
        
        # With tight barriers, more labels should be hit
        # With wide barriers, most should be neutral (time exit)
        tight_hits = (df_tight['label'] != 0).sum()
        wide_hits = (df_wide['label'] != 0).sum()
        
        # This is a sanity check - exact values depend on data
        assert tight_hits >= wide_hits