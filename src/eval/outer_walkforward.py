from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import average_precision_score, matthews_corrcoef, brier_score_loss
from sklearn.preprocessing import MinMaxScaler

from src.data.binance_loader import CryptoDataLoader
from src.features.engineering import FeatureEngineer
from src.utils.logging import log as logger


@dataclass
class TemporalConfig:
    lookback: int = 96  # ~1 day on 15m
    horizon: int = 5    # bars ahead used for label
    embargo: int = 96   # will be max(lookback-1, horizon) if not provided
    val_frac: float = 0.2


def create_labels(close: pd.Series, horizon: int = 5, threshold: float = 0.002) -> pd.Series:
    fut_ret = close.pct_change(horizon).shift(-horizon)
    y = (fut_ret > threshold).astype(int)
    return y


def choose_threshold_by_ev(p_val: np.ndarray, y_val: np.ndarray, fee_bps: float, slippage_bps: float, min_action_rate: float = 0.03) -> float:
    # Expanded grid for better coverage
    grid = np.linspace(0.2, 0.8, 61)
    best_ev, best_t = -1e9, 0.5
    c = (fee_bps + slippage_bps) / 1e4
    label = (y_val * 2 - 1).astype(int)
    
    for t in grid:
        long = (p_val >= t).astype(int)
        short = (p_val <= 1 - t).astype(int)
        pos = long - short
        
        # Calculate action rate
        action_rate = (pos != 0).mean()
        
        # Skip if action rate is too low
        if action_rate < min_action_rate:
            continue
            
        # simple turnover cost when switching
        switches = np.abs(np.diff(np.r_[0, pos]))
        n_trades = switches.sum()
        
        # Base EV calculation
        ev_base = float(np.mean(pos * label) - switches.mean() * c)
        
        # Penalize zero trades
        penalty = 0.001 if n_trades == 0 else 0
        ev = ev_base - penalty
        
        if ev > best_ev:
            best_ev, best_t = ev, t
            
    # Log diagnostic info
    logger.info("Threshold optimization", 
                best_threshold=best_t, 
                best_ev=best_ev,
                grid_range=(grid.min(), grid.max()))
    
    return float(best_t)


def ece_score(y_true: np.ndarray, p_pred: np.ndarray, n_bins: int = 15) -> float:
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(p_pred)
    for i in range(n_bins):
        mask = (p_pred >= bins[i]) & (p_pred < bins[i + 1])
        if mask.any():
            conf = float(p_pred[mask].mean())
            acc = float(y_true[mask].mean())
            ece += (mask.sum() / max(1, n)) * abs(acc - conf)
    return float(ece)


def embargo_cuts(n: int, test_size: int, cfg: TemporalConfig, n_splits: int = 4) -> List[Tuple[int, int, int, int]]:
    """Build anchored outer splits with embargo between val and test.

    Returns list of tuples (train_start, train_end, val_end, test_end)
    where train is [train_start:train_end), val is [train_end+E:val_end),
    test is [val_end+E:test_end).
    """
    L, H = cfg.lookback, cfg.horizon
    E = max(cfg.embargo, L - 1, H)
    cuts: List[Tuple[int, int, int, int]] = []
    for j in range(n_splits):
        test_end = n - (n_splits - j - 1) * test_size
        test_start = test_end - test_size
        if test_start <= 0 or test_end > n:
            continue
        val_end = test_start - E
        if val_end <= L:
            continue
        trval_len = val_end - L
        tr_end = L + int(trval_len * (1 - cfg.val_frac))
        cuts.append((0, tr_end, val_end, test_end))
    return cuts


def build_model_from_artifact() -> object:
    """Try to reuse best params from last saved wrapper; else default XGB."""
    import joblib  # type: ignore
    from glob import glob
    from xgboost import XGBClassifier  # type: ignore

    paths = sorted(glob("artifacts/models/xgboost_optuna_*.pkl"))
    if not paths:
        return XGBClassifier(
            n_estimators=600,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.0,
            reg_lambda=2.0,
            tree_method="hist",
            random_state=42,
            n_jobs=-1,
            eval_metric="aucpr",
        )
    wrap = joblib.load(paths[-1])
    try:
        best = getattr(wrap, "best_model", None)
        if best is not None:
            params = best.get_params()
            params.update({"random_state": 42, "n_jobs": -1, "eval_metric": "aucpr"})
            return XGBClassifier(**params)
    except Exception:
        pass
    # Fallback
    return XGBClassifier(
        n_estimators=600,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.0,
        reg_lambda=2.0,
        tree_method="hist",
        random_state=42,
        n_jobs=-1,
        eval_metric="aucpr",
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument("--timeframe", default="15m")
    ap.add_argument("--start", default="2024-06-01")
    ap.add_argument("--end", default="2024-07-02")
    ap.add_argument("--lookback", type=int, default=16)  # Reduced from 96
    ap.add_argument("--horizon", type=int, default=3)   # Reduced from 5
    ap.add_argument("--label_th", type=float, default=0.001)  # Reduced from 0.002
    ap.add_argument("--val_frac", type=float, default=0.15)  # Reduced from 0.2
    ap.add_argument("--test_size", type=int, default=300)  # Reduced from 400
    ap.add_argument("--fee_bps", type=float, default=8.0)
    ap.add_argument("--slippage_bps", type=float, default=4.0)
    args = ap.parse_args()

    loader = CryptoDataLoader(use_cache=True)
    df = loader.fetch_ohlcv(args.symbol, args.timeframe, args.start, args.end)
    # Build labels on full df then slice
    y_all = create_labels(df["close"], horizon=args.horizon, threshold=args.label_th)
    # Align
    df = df.loc[y_all.index]
    y_all = y_all.loc[df.index]
    # Remove NaNs
    mask = ~y_all.isna()
    df = df.loc[mask]
    y_all = y_all.loc[mask]

    n = len(df)
    cfg = TemporalConfig(lookback=args.lookback, horizon=args.horizon, embargo=max(args.lookback - 1, args.horizon), val_frac=args.val_frac)
    cuts = embargo_cuts(n, args.test_size, cfg, n_splits=4)
    if not cuts:
        print("no_cuts_generated; reduce test_size or adjust dates")
        return

    rows = []
    reports_dir = Path("artifacts/reports")
    reports_dir.mkdir(parents=True, exist_ok=True)

    for (train_start, tr_end, val_end, te_end) in cuts:
        L = cfg.lookback
        E = max(cfg.embargo, cfg.lookback - 1, cfg.horizon)
        tr_idx = slice(L, tr_end)
        va_idx = slice(tr_end, val_end)
        te_idx = slice(val_end + E, te_end)
        if tr_idx.stop <= tr_idx.start or va_idx.stop <= va_idx.start or te_idx.stop <= te_idx.start:
            continue
        # Ensure validation window is large enough to survive rolling drops
        if (va_idx.stop - va_idx.start) < max(300, L + 50):
            continue

        df_tr = df.iloc[tr_idx]
        df_va = df.iloc[va_idx]
        df_te = df.iloc[te_idx]
        y_tr = y_all.iloc[tr_idx].astype(int)
        y_va = y_all.iloc[va_idx].astype(int)
        y_te = y_all.iloc[te_idx].astype(int)

        # Feature engineering per split (causal windows)
        fe = FeatureEngineer(scaler_type=None)
        Xtr_df = fe.create_all_features(df_tr)
        Xva_df = fe.create_all_features(df_va)
        Xte_df = fe.create_all_features(df_te)
        
        # Align y with X (features creation drops initial rows)
        y_tr_aligned = y_tr.loc[Xtr_df.index]
        y_va_aligned = y_va.loc[Xva_df.index]
        y_te_aligned = y_te.loc[Xte_df.index]
        
        # Align feature sets
        common_cols = sorted(set(Xtr_df.columns) & set(Xva_df.columns) & set(Xte_df.columns))
        Xtr_df = Xtr_df[common_cols]
        Xva_df = Xva_df[common_cols]
        Xte_df = Xte_df[common_cols]

        # Check if we have data after preprocessing
        if len(Xtr_df) == 0 or len(Xva_df) == 0 or len(Xte_df) == 0:
            logger.warning("Empty dataset after preprocessing", 
                       train_size=len(Xtr_df), val_size=len(Xva_df), test_size=len(Xte_df))
            
            # Skip this fold if validation or test is empty
            if len(Xva_df) == 0 or len(Xte_df) == 0:
                logger.warning("Skipping fold due to empty validation/test set")
                continue
                
        # Scale based on train only
        scaler = MinMaxScaler()
        Xtr = scaler.fit_transform(Xtr_df.values)
        Xva = scaler.transform(Xva_df.values)
        Xte = scaler.transform(Xte_df.values)

        # Model
        model = build_model_from_artifact()
        model.fit(Xtr, y_tr_aligned.values)

        # Calibration on validation
        calib = CalibratedClassifierCV(model, method="isotonic", cv="prefit")
        calib.fit(Xva, y_va_aligned.values)
        pva = calib.predict_proba(Xva)[:, 1]
        tau = choose_threshold_by_ev(pva, y_va_aligned.values, args.fee_bps, args.slippage_bps, min_action_rate=0.03)

        # Test metrics
        pte = calib.predict_proba(Xte)[:, 1]
        
        # Log probability distribution for diagnostics
        logger.info("Probability distribution",
                   min=float(pte.min()),
                   p25=float(np.percentile(pte, 25)),
                   median=float(np.median(pte)),
                   p75=float(np.percentile(pte, 75)),
                   max=float(pte.max()),
                   fold=i)
        
        # Implement long/short positions based on threshold
        pos_long = (pte >= tau).astype(int)
        pos_short = (pte <= (1 - tau)).astype(int)
        pos = pos_long - pos_short  # +1 for long, -1 for short, 0 for neutral
        
        # Log action rates
        action_rate = (pos != 0).mean()
        logger.info("Trading signals",
                   action_rate=float(action_rate),
                   long_rate=float(pos_long.mean()),
                   short_rate=float(pos_short.mean()),
                   threshold=tau,
                   fold=i)
        
        # For classification metrics, use long signals only
        yhat = pos_long
        
        pr = float(average_precision_score(y_te_aligned.values, pte))
        p_base = float(y_te_aligned.mean())
        pr_norm = float((pr - p_base) / (1 - p_base + 1e-12))
        mcc = float(matthews_corrcoef(y_te_aligned.values, yhat))
        brier = float(brier_score_loss(y_te_aligned.values, pte))
        ece = float(ece_score(y_te_aligned.values, pte))

        # Simple t+1 backtest based on direction proxy
        # Build returns from closes in test window (aligned with features)
        rets_full = df_te["close"].pct_change().fillna(0.0)
        rets = rets_full.loc[Xte_df.index].values
        # execute at next bar open: shift position
        pos_exec = np.r_[0, pos[:-1]]
        switches = np.abs(np.diff(np.r_[0, pos_exec]))
        cost = (args.fee_bps + args.slippage_bps) / 1e4
        pnl = pos_exec * rets - switches * cost
        sharpe = float(np.mean(pnl) / (np.std(pnl) + 1e-12) * np.sqrt(252))
        turnover = float(switches.mean())

        rows.append(
            {
                "train": f"{int(L)}:{int(tr_end)}",
                "val": f"{int(tr_end+E)}:{int(val_end)}",
                "test": f"{int(val_end+E)}:{int(te_end)}",
                "p": p_base,
                "AUPRC": pr,
                "AUPRC_norm": pr_norm,
                "MCC": mcc,
                "Brier": brier,
                "ECE": ece,
                "tau": tau,
                "Trades": int(switches.sum()),
                "Turnover": turnover,
                "PnL_mean": float(np.mean(pnl)),
                "Sharpe": sharpe,
            }
        )

    report = pd.DataFrame(rows)
    out_csv = reports_dir / "outer_walkforward_xgb.csv"
    report.to_csv(out_csv, index=False)
    # Print concise summary
    if len(report):
        summ = report[["AUPRC", "AUPRC_norm", "MCC", "Brier", "ECE", "Sharpe", "Turnover"]].mean().to_dict()
        print({k: float(v) for k, v in summ.items()})
        print(f"saved: {out_csv}")
    else:
        print("no_rows_in_report")


if __name__ == "__main__":
    main()
