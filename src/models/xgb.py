from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

from ..utils.logging import log
from ..utils.config import load_yaml
from .utils_common import try_import
from .feature_select_ga import genetic_feature_selection
from .postproc import calibrate_and_threshold, choose_threshold_by_ev_tplus1
from ..data.splits import purged_kfold_index


def cli_train():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/xgb.yaml")
    ap.add_argument("--fast", action="store_true")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    seed = int(cfg.get("seed", 42))
    costs = (cfg.get("costs") or {"fee_bps": 5, "slippage_bps": 5})

    # Check deps
    xgb = try_import("xgboost")
    sk = try_import("sklearn")

    if xgb is None or sk is None:
        log.warning(
            "degraded_mode: missing deps; skipping training",
            needs=["xgboost", "scikit-learn"],
        )
        return

    # Minimal synthetic dataset for offline demo if no data loader
    import numpy as np  # type: ignore

    n = 1000 if not args.fast else 200
    p = 20
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    # synthetic label with weak signal
    linear = X[:, 0] * 0.8 + X[:, 1] * 0.5 + rng.normal(scale=0.5, size=n)
    y = (linear > 0).astype(int)
    feature_names = [f"f{i}" for i in range(p)]
    # synthetic next-bar returns aligned to t+1 execution; correlate with label
    # positive label tends to positive next return
    ret_noise = rng.normal(loc=0.0, scale=0.001, size=n)
    returns_next = (y * 2 - 1) * 0.002 + ret_noise

    # GA feature selection (fitness = mean val accuracy proxy -> here EV-lite)
    def fitness(mask: List[int]) -> float:
        sel = [i for i, b in enumerate(mask) if b == 1]
        if not sel:
            return -1.0
        Xs = X[:, sel]
        scores = []
        for tr, va in purged_kfold_index(range(len(y)), n_splits=3, embargo=2):
            model = xgb.XGBClassifier(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=4,
                subsample=0.8,
                colsample_bytree=0.8,
                tree_method="hist",
                random_state=seed,
                n_jobs=1,
                eval_metric="logloss",
            )
            model.fit(Xs[tr], y[tr])
            p_val = model.predict_proba(Xs[va])[:, 1]
            # EV proxy with threshold optimized per-fold; use label-signed returns as fallback
            y_signed = (y[va] * 2 - 1)
            th = choose_threshold_by_ev_tplus1(
                p_val,
                y_signed,
                costs.get("fee_bps", 5),
                costs.get("slippage_bps", 5),
            )
            side = (p_val >= th).astype(int)
            gross = float(np.mean(side * y_signed))
            turn = float(np.mean(np.abs(np.diff(side, prepend=0))))
            ev = gross - turn * (costs["fee_bps"] + costs["slippage_bps"]) / 1e4
            scores.append(ev)
        return float(np.mean(scores))

    n_features = p
    mask, score = genetic_feature_selection(
        n_features=n_features,
        fitness_fn=fitness,
        population_size=20 if args.fast else 40,
        generations=3 if args.fast else 10,
        seed=seed,
    )
    sel_idx = [i for i, b in enumerate(mask) if b == 1]
    if not sel_idx:
        sel_idx = list(range(min(5, n_features)))
    X_sel = X[:, sel_idx]

    # Final model
    # class imbalance handling
    pos = max(1, int((y == 1).sum()))
    neg = max(1, int((y == 0).sum()))
    scale_pos_weight = max(0.1, min(10.0, neg / pos))

    model = xgb.XGBClassifier(
        n_estimators=500 if not args.fast else 200,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method="hist",
        random_state=seed,
        n_jobs=1,
        eval_metric="logloss",
        scale_pos_weight=scale_pos_weight,
    )

    # Split train/val
    n_tr = int(0.8 * n)
    X_tr, X_val = X_sel[:n_tr], X_sel[n_tr:]
    y_tr, y_val = y[:n_tr], y[n_tr:]
    # early stopping on temporal validation
    model.fit(
        X_tr,
        y_tr,
        eval_set=[(X_val, y_val)],
        verbose=False,
        early_stopping_rounds=(10 if args.fast else 50),
    )

    # Calibrate + thresholds, with EV tuning using real t+1 returns (synthetic here)
    returns_val_next = returns_next[n_tr:]
    calibrator, metrics = calibrate_and_threshold(
        model,
        X_tr,
        y_tr,
        X_val,
        y_val,
        costs,
        calibration_method="both",
        returns_val_next=returns_val_next,
    )

    # Persist minimal artifacts
    out_dir = Path("artifacts/models/xgb_stub")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "selected_features.txt").write_text("\n".join([feature_names[i] for i in sel_idx]))
    (out_dir / "metrics.txt").write_text("\n".join([f"{k}: {v}" for k, v in metrics.items()]))
    log.info(
        "train_complete",
        selected_features=len(sel_idx),
        th_ev=metrics.get("threshold_ev", 0.5),
    )


if __name__ == "__main__":
    cli_train()
