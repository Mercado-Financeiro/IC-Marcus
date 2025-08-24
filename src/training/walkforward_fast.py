#!/usr/bin/env python
"""
Fast Walk-Forward Analysis with Dual Thresholds
Quick version with simplified threshold optimization
"""

import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import average_precision_score, matthews_corrcoef, brier_score_loss
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier

from src.data.binance_loader import CryptoDataLoader
from src.features.engineering import FeatureEngineer
from src.eval.walkforward_db import WalkForwardDB
from src.utils.logging import log as logger


def create_labels(close: pd.Series, horizon: int = 2, threshold: float = 0.0005) -> pd.Series:
    """Create binary labels for classification"""
    fut_ret = close.pct_change(horizon).shift(-horizon)
    y = (fut_ret > threshold).astype(int)
    return y


def fast_dual_thresholds(p_val, y_val, fee_bps=8, slippage_bps=4):
    """Fast dual threshold selection based on percentiles"""
    # Use percentiles for fast selection
    # Long threshold: aim for top 20-30% signals
    # Short threshold: aim for bottom 20-30% signals
    
    # Start with reasonable defaults
    tau_long = np.percentile(p_val, 70)  # Top 30%
    tau_short = np.percentile(p_val, 30)  # Bottom 30%
    
    # Ensure minimum dead zone
    if tau_long - tau_short < 0.2:
        tau_long = min(0.7, tau_short + 0.2)
    
    # Clamp to reasonable ranges
    tau_long = np.clip(tau_long, 0.55, 0.8)
    tau_short = np.clip(tau_short, 0.2, 0.45)
    
    return float(tau_long), float(tau_short)


def run_fast_walkforward():
    """Fast walk-forward execution"""
    
    # Configuration
    config = {
        'symbol': 'BTCUSDT',
        'timeframe': '15m',
        'start_date': '2024-05-01',  # Extended period
        'end_date': '2024-07-15',
        'lookback': 8,  # Reduced lookback
        'horizon': 2,
        'label_threshold': 0.0005,
        'val_frac': 0.15,
        'test_size': 200,  # Smaller test size
        'fee_bps': 8,
        'slippage_bps': 4,
        'n_splits': 3
    }
    
    print("=" * 60)
    print("FAST WALK-FORWARD ANALYSIS")
    print("=" * 60)
    
    # Load data
    if Path('/tmp/btc_data.parquet').exists():
        print("Loading cached data...")
        df = pd.read_parquet('/tmp/btc_data.parquet')
        # Filter to requested date range
        df = df[(df.index >= config['start_date']) & (df.index <= config['end_date'] + 'T23:59:59')]
    else:
        print("Downloading data...")
        loader = CryptoDataLoader(use_cache=False)
        df = loader.fetch_ohlcv(
            config['symbol'], 
            config['timeframe'],
            config['start_date'], 
            config['end_date']
        )
    
    print(f"Data: {len(df)} rows, {df.index[0]} to {df.index[-1]}")
    
    # Create labels
    y_all = create_labels(df["close"], config['horizon'], config['label_threshold'])
    df = df[:len(y_all)]
    
    # Walk-forward splits
    n = len(df)
    test_size = config['test_size']
    n_splits = config['n_splits']
    lookback = config['lookback']
    embargo = max(lookback - 1, config['horizon'])
    
    results = []
    
    for fold_num in range(n_splits):
        print(f"\nFold {fold_num + 1}/{n_splits}:")
        
        # Calculate split indices
        test_end = n - (n_splits - fold_num - 1) * (test_size // 2)
        test_start = test_end - test_size
        
        if test_start <= 0 or test_end > n:
            continue
            
        val_end = test_start - embargo
        if val_end <= lookback:
            continue
            
        trval_len = val_end - lookback
        tr_end = lookback + int(trval_len * (1 - config['val_frac']))
        
        try:
            # Prepare data splits
            df_tr = df.iloc[lookback:tr_end]
            df_va = df.iloc[tr_end + embargo:val_end]
            df_te = df.iloc[test_start:test_end]
            
            y_tr = y_all.iloc[lookback:tr_end]
            y_va = y_all.iloc[tr_end + embargo:val_end]
            y_te = y_all.iloc[test_start:test_end]
            
            # Quick feature engineering (skip logging)
            fe = FeatureEngineer(scaler_type=None)
            logger.disabled = True  # Disable logging for speed
            
            Xtr_df = fe.create_all_features(df_tr)
            Xva_df = fe.create_all_features(df_va)
            Xte_df = fe.create_all_features(df_te)
            
            logger.disabled = False
            
            # Align labels
            y_tr_aligned = y_tr.loc[Xtr_df.index]
            y_va_aligned = y_va.loc[Xva_df.index]
            y_te_aligned = y_te.loc[Xte_df.index]
            
            # Check data size
            if len(Xtr_df) < 100 or len(Xva_df) < 20 or len(Xte_df) < 20:
                print(f"  Skipping: insufficient data")
                continue
            
            # Align features
            common_cols = sorted(set(Xtr_df.columns) & set(Xva_df.columns) & set(Xte_df.columns))
            Xtr_df = Xtr_df[common_cols]
            Xva_df = Xva_df[common_cols]
            Xte_df = Xte_df[common_cols]
            
            # Scale
            scaler = MinMaxScaler()
            Xtr = scaler.fit_transform(Xtr_df.values)
            Xva = scaler.transform(Xva_df.values)
            Xte = scaler.transform(Xte_df.values)
            
            # Train simple model
            model = XGBClassifier(
                n_estimators=100,  # Reduced for speed
                max_depth=3,
                learning_rate=0.1,
                random_state=42,
                n_jobs=-1,
                eval_metric="logloss"
            )
            model.fit(Xtr, y_tr_aligned.values)
            
            # Calibrate
            calib = CalibratedClassifierCV(model, method="isotonic", cv="prefit")
            calib.fit(Xva, y_va_aligned.values)
            
            # Get thresholds fast
            pva = calib.predict_proba(Xva)[:, 1]
            tau_long, tau_short = fast_dual_thresholds(pva, y_va_aligned.values)
            
            # Test predictions
            pte = calib.predict_proba(Xte)[:, 1]
            
            # Calculate positions with dead zone
            pos = np.zeros_like(pte)
            pos[pte >= tau_long] = 1
            pos[pte <= tau_short] = -1
            
            # Metrics
            pos_long = (pos == 1).astype(int)
            yhat = pos_long
            
            auprc = float(average_precision_score(y_te_aligned.values, pte))
            mcc = float(matthews_corrcoef(y_te_aligned.values, yhat))
            brier = float(brier_score_loss(y_te_aligned.values, pte))
            
            # Trading metrics
            rets = df_te["close"].pct_change().fillna(0.0)
            rets = rets.loc[Xte_df.index].values
            
            pos_exec = np.r_[0, pos[:-1]]
            switches = np.abs(np.diff(np.r_[0, pos_exec]))
            cost = (config['fee_bps'] + config['slippage_bps']) / 1e4
            pnl = pos_exec * rets - switches * cost
            
            sharpe = float(np.mean(pnl) / (np.std(pnl) + 1e-12) * np.sqrt(252 * 96))
            n_trades = int(switches.sum())
            action_rate = float((pos != 0).mean())
            neutral_rate = float((pos == 0).mean())
            
            results.append({
                'fold': fold_num + 1,
                'auprc': auprc,
                'mcc': mcc,
                'brier': brier,
                'sharpe': sharpe,
                'n_trades': n_trades,
                'action_rate': action_rate,
                'neutral_rate': neutral_rate,
                'tau_long': tau_long,
                'tau_short': tau_short
            })
            
            print(f"  AUPRC={auprc:.3f}, MCC={mcc:.3f}, Sharpe={sharpe:.2f}")
            print(f"  Thresholds: Long={tau_long:.3f}, Short={tau_short:.3f}")
            print(f"  Action={action_rate:.1%}, Neutral={neutral_rate:.1%}, Trades={n_trades}")
            
        except Exception as e:
            print(f"  Error: {e}")
    
    # Summary
    if results:
        df_results = pd.DataFrame(results)
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        print(df_results[['fold', 'auprc', 'mcc', 'sharpe', 'action_rate', 'neutral_rate']].round(3))
        print("\nAverages:")
        print(f"  AUPRC: {df_results['auprc'].mean():.3f} ± {df_results['auprc'].std():.3f}")
        print(f"  MCC: {df_results['mcc'].mean():.3f} ± {df_results['mcc'].std():.3f}")
        print(f"  Sharpe: {df_results['sharpe'].mean():.2f} ± {df_results['sharpe'].std():.2f}")
        print(f"  Action Rate: {df_results['action_rate'].mean():.1%}")
        print(f"  Neutral Rate: {df_results['neutral_rate'].mean():.1%}")


if __name__ == "__main__":
    run_fast_walkforward()