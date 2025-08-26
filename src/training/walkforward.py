#!/usr/bin/env python
"""
Walk-Forward Analysis with Database Storage
Executes walk-forward analysis and stores results incrementally in SQLite
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


def _sanitize_fit(X: pd.DataFrame, clip_q: Tuple[float, float] = (0.001, 0.999)):
    """Compute train-based sanitation stats: clip quantiles and medians for imputation."""
    qlow, qhigh = X.quantile(clip_q[0]), X.quantile(clip_q[1])
    med = X.median()
    return {"qlow": qlow, "qhigh": qhigh, "med": med}


def _sanitize_apply(X: pd.DataFrame, stats) -> pd.DataFrame:
    Xc = X.copy()
    Xc = Xc.replace([np.inf, -np.inf], np.nan)
    # Clip using train quantiles
    Xc = Xc.clip(lower=stats["qlow"], upper=stats["qhigh"], axis=1)
    # Impute NaNs with train median
    Xc = Xc.fillna(stats["med"])
    # Any remaining NaNs -> 0
    Xc = Xc.fillna(0.0)
    return Xc


def choose_dual_thresholds_by_ev(p_val, y_val, fee_bps, slippage_bps, min_action_rate=0.03, max_action_rate=0.5):
    """Choose independent long and short thresholds with dead zone"""
    # Independent grids for long and short thresholds (reduced for speed)
    long_grid = np.linspace(0.5, 0.8, 16)  # Long entry thresholds
    short_grid = np.linspace(0.2, 0.5, 16)  # Short entry thresholds
    
    best_ev = -1e9
    best_t_long = 0.65
    best_t_short = 0.35
    
    c = (fee_bps + slippage_bps) / 1e4
    label = (y_val * 2 - 1).astype(int)
    
    for t_long in long_grid:
        for t_short in short_grid:
            # Ensure dead zone exists (t_short < t_long)
            if t_short >= t_long - 0.1:  # Minimum 10% dead zone
                continue
                
            # Calculate positions with dead zone
            pos = np.zeros_like(p_val)
            pos[p_val >= t_long] = 1   # Long when p >= t_long
            pos[p_val <= t_short] = -1  # Short when p <= t_short
            # Neutral when t_short < p < t_long (dead zone)
            
            action_rate = (pos != 0).mean()
            
            # Skip if action rate is outside bounds
            if action_rate < min_action_rate or action_rate > max_action_rate:
                continue
                
            # Calculate switches and costs
            switches = np.abs(np.diff(np.r_[0, pos]))
            n_trades = switches.sum()
            
            # Skip if no trades
            if n_trades == 0:
                continue
            
            # Calculate expected value
            gross_pnl = pos * label
            cost_per_trade = switches * c
            net_pnl = gross_pnl - cost_per_trade
            ev = float(np.mean(net_pnl))
            
            if ev > best_ev:
                best_ev = ev
                best_t_long = t_long
                best_t_short = t_short
    
    return float(best_t_long), float(best_t_short)


def run_walkforward_with_db():
    """Main walk-forward execution with database storage"""
    
    # Configuration - Extended period for robust analysis
    config = {
        'symbol': 'BTCUSDT',
        'timeframe': '15m',
        'start_date': '2023-01-01',  # Extended: 1.5 years
        'end_date': '2024-07-31',     # Extended end date
        'lookback': 96,               # Back to original (1 day)
        'horizon': 5,                 # Back to original
        'label_threshold': 0.0005,
        'val_frac': 0.15,
        'test_size': 1500,            # Increased for more data
        'fee_bps': 8,
        'slippage_bps': 4,
        'n_splits': 5                 # More splits with more data
    }
    
    print("=" * 60)
    print("WALK-FORWARD ANALYSIS WITH DATABASE STORAGE")
    print("=" * 60)
    print(f"Symbol: {config['symbol']}")
    print(f"Period: {config['start_date']} to {config['end_date']}")
    print(f"Parameters: lookback={config['lookback']}, horizon={config['horizon']}")
    print("=" * 60)
    
    # Load data - prioritize full cached dataset
    try:
        # First, try to load from full dataset
        full_data_path = Path('data/processed/btcusdt_15m_full.parquet')
        if full_data_path.exists():
            print(f"Loading data from {full_data_path}...")
            df = pd.read_parquet(full_data_path)
            # Filter to requested date range
            df = df[(df.index >= config['start_date']) & (df.index <= config['end_date'] + 'T23:59:59')]
            print(f"Loaded {len(df)} rows from cache")
        # Fallback to temp cache
        elif Path('/tmp/btc_data.parquet').exists():
            print("Loading data from /tmp/btc_data.parquet...")
            df = pd.read_parquet('/tmp/btc_data.parquet')
            # Filter to requested date range
            df = df[(df.index >= config['start_date']) & (df.index <= config['end_date'] + 'T23:59:59')]
        else:
            print("Downloading fresh data...")
            print("Note: Run 'python download_full_data.py' first for faster loading")
            loader = CryptoDataLoader(use_cache=False)
            df = loader.fetch_ohlcv(
                config['symbol'], 
                config['timeframe'],
                config['start_date'], 
                config['end_date']
            )
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    print(f"Data loaded: {len(df)} rows")
    print(f"Period: {df.index[0]} to {df.index[-1]}")
    
    # Create labels
    y_all = create_labels(df["close"], config['horizon'], config['label_threshold'])
    df = df[:len(y_all)]
    
    # Initialize database
    with WalkForwardDB() as db:
        run_id = db.create_run(config)
        print(f"\nDatabase run ID: {run_id}")
        print("-" * 40)
        
        # Create time series splits
        n = len(df)
        test_size = config['test_size']
        n_splits = config['n_splits']
        lookback = config['lookback']
        embargo = max(lookback - 1, config['horizon'])
        
        valid_folds = 0
        
        for fold_num in range(n_splits):
            fold_start_time = time.time()
            
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
            
            print(f"\nFold {fold_num + 1}/{n_splits}:")
            print(f"  Train: {lookback}:{tr_end} ({tr_end - lookback} samples)")
            print(f"  Val: {tr_end + embargo}:{val_end} ({val_end - tr_end - embargo} samples)")
            print(f"  Test: {test_start}:{test_end} ({test_size} samples)")
            
            try:
                # Prepare data splits
                df_tr = df.iloc[lookback:tr_end]
                df_va = df.iloc[tr_end + embargo:val_end]
                df_te = df.iloc[test_start:test_end]
                
                y_tr = y_all.iloc[lookback:tr_end]
                y_va = y_all.iloc[tr_end + embargo:val_end]
                y_te = y_all.iloc[test_start:test_end]
                
                # Feature engineering
                print("  Creating features...")
                fe = FeatureEngineer(scaler_type=None)
                
                Xtr_df = fe.create_all_features(df_tr)
                Xva_df = fe.create_all_features(df_va)
                Xte_df = fe.create_all_features(df_te)
                
                # Align labels with features
                y_tr_aligned = y_tr.loc[Xtr_df.index]
                y_va_aligned = y_va.loc[Xva_df.index]
                y_te_aligned = y_te.loc[Xte_df.index]
                
                # Check if we have enough data
                if len(Xtr_df) < 100 or len(Xva_df) < 20 or len(Xte_df) < 20:
                    print(f"  Skipping fold: insufficient data after features")
                    continue
                
                # Align feature columns
                common_cols = sorted(set(Xtr_df.columns) & set(Xva_df.columns) & set(Xte_df.columns))
                Xtr_df = Xtr_df[common_cols]
                Xva_df = Xva_df[common_cols]
                Xte_df = Xte_df[common_cols]

                # Sanitize features using train-based stats (no leakage)
                stats_san = _sanitize_fit(Xtr_df)
                Xtr_df = _sanitize_apply(Xtr_df, stats_san)
                Xva_df = _sanitize_apply(Xva_df, stats_san)
                Xte_df = _sanitize_apply(Xte_df, stats_san)

                # Scale features
                scaler = MinMaxScaler()
                Xtr = scaler.fit_transform(Xtr_df.values)
                Xva = scaler.transform(Xva_df.values)
                Xte = scaler.transform(Xte_df.values)
                
                # Train model
                print("  Training XGBoost...")
                model = XGBClassifier(
                    n_estimators=300,
                    max_depth=4,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42,
                    n_jobs=-1,
                    eval_metric="aucpr"
                )
                model.fit(Xtr, y_tr_aligned.values)
                
                # Calibrate on validation
                print("  Calibrating...")
                calib = CalibratedClassifierCV(model, method="isotonic", cv="prefit")
                calib.fit(Xva, y_va_aligned.values)
                
                # Choose dual thresholds
                pva = calib.predict_proba(Xva)[:, 1]
                tau_long, tau_short = choose_dual_thresholds_by_ev(
                    pva, y_va_aligned.values, 
                    config['fee_bps'], config['slippage_bps'],
                    min_action_rate=0.02, max_action_rate=0.4
                )
                
                # Test predictions
                pte = calib.predict_proba(Xte)[:, 1]
                
                # Calculate positions with dead zone
                pos = np.zeros_like(pte)
                pos[pte >= tau_long] = 1   # Long when p >= tau_long
                pos[pte <= tau_short] = -1  # Short when p <= tau_short
                # Neutral when tau_short < p < tau_long (dead zone)
                
                pos_long = (pos == 1).astype(int)
                pos_short = (pos == -1).astype(int)
                
                # Calculate metrics
                yhat = pos_long  # For classification metrics
                
                # ML Metrics
                auprc = float(average_precision_score(y_te_aligned.values, pte))
                p_base = float(y_te_aligned.mean())
                auprc_norm = float((auprc - p_base) / (1 - p_base + 1e-12))
                mcc = float(matthews_corrcoef(y_te_aligned.values, yhat))
                brier = float(brier_score_loss(y_te_aligned.values, pte))
                
                # Trading metrics
                rets = df_te["close"].pct_change().fillna(0.0)
                rets = rets.loc[Xte_df.index].values
                
                pos_exec = np.r_[0, pos[:-1]]  # Execute at next bar
                switches = np.abs(np.diff(np.r_[0, pos_exec]))
                cost = (config['fee_bps'] + config['slippage_bps']) / 1e4
                pnl = pos_exec * rets - switches * cost
                
                sharpe = float(np.mean(pnl) / (np.std(pnl) + 1e-12) * np.sqrt(252 * 96))
                turnover = float(switches.mean())
                n_trades = int(switches.sum())
                
                # Position stats
                action_rate = float((pos != 0).mean())
                long_rate = float(pos_long.mean())
                short_rate = float(pos_short.mean())
                neutral_rate = float((pos == 0).mean())
                
                # Probability distribution
                prob_stats = {
                    'prob_min': float(pte.min()),
                    'prob_p25': float(np.percentile(pte, 25)),
                    'prob_median': float(np.median(pte)),
                    'prob_p75': float(np.percentile(pte, 75)),
                    'prob_max': float(pte.max())
                }
                
                # Save to database
                fold_result = {
                    'train_start': lookback,
                    'train_end': tr_end,
                    'val_start': tr_end + embargo,
                    'val_end': val_end,
                    'test_start': test_start,
                    'test_end': test_end,
                    'prevalence': p_base,
                    'auprc': auprc,
                    'auprc_norm': auprc_norm,
                    'mcc': mcc,
                    'brier_score': brier,
                    'threshold': None,  # Legacy
                    'threshold_long': tau_long,
                    'threshold_short': tau_short,
                    'n_trades': n_trades,
                    'turnover': turnover,
                    'pnl_mean': float(pnl.mean()),
                    'sharpe_ratio': sharpe,
                    'long_rate': long_rate,
                    'short_rate': short_rate,
                    'action_rate': action_rate,
                    'execution_time_seconds': time.time() - fold_start_time,
                    **prob_stats
                }
                
                db.save_fold_result(run_id, fold_num + 1, fold_result)
                valid_folds += 1
                
                print(f"  Results: AUPRC={auprc:.3f}, MCC={mcc:.3f}, Sharpe={sharpe:.2f}")
                print(f"  Thresholds: Long={tau_long:.3f}, Short={tau_short:.3f}")
                print(f"  Trading: {n_trades} trades, action={action_rate:.1%}, neutral={neutral_rate:.1%}")
                print(f"  ✓ Saved to database")
                
            except Exception as e:
                print(f"  ✗ Error in fold {fold_num + 1}: {e}")
                db.save_fold_result(run_id, fold_num + 1, {'error_message': str(e)})
        
        # Update run statistics
        db.update_run_stats(run_id, valid_folds)
        
        # Print summary
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        print(f"Valid folds: {valid_folds}/{n_splits}")
        
        if valid_folds > 0:
            stats = db.get_aggregate_stats(run_id)
            print(f"\nAggregate Statistics:")
            print(f"  AUPRC: {stats.get('avg_auprc', 0):.3f} ± {stats.get('std_auprc', 0):.3f}")
            print(f"  MCC: {stats.get('avg_mcc', 0):.3f} ± {stats.get('std_mcc', 0):.3f}")
            print(f"  Sharpe: {stats.get('avg_sharpe', 0):.2f} ± {stats.get('std_sharpe', 0):.2f}")
            print(f"  Total trades: {stats.get('total_trades', 0)}")
            print(f"  Avg action rate: {stats.get('avg_action_rate', 0):.1%}")
            
            # Export to CSV
            csv_path = f"artifacts/reports/walkforward_run_{run_id}.csv"
            db.export_to_csv(run_id, csv_path)
            
        print(f"\n✓ Results stored in database (run_id={run_id})")
        print(f"  Database: artifacts/walkforward_results.db")


if __name__ == "__main__":
    run_walkforward_with_db()
