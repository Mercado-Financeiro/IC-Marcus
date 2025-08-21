from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import average_precision_score, f1_score

from data.dataset import fit_transform_no_leak, load_ohlcv_csv, resample_ohlcv
from src.xgboost.xgboost import (
    XGBConfig,
    make_tabular_features,
    purged_walk_forward,
    triple_barrier_labels,
)


def select_csv_from_dir(data_dir: str, symbol: Optional[str], pattern: Optional[str]) -> str:
    base = Path(data_dir)
    candidates = sorted([p for p in base.rglob("*.csv")])
    if not candidates:
        raise FileNotFoundError(f"Nenhum CSV encontrado em {base}")
    if pattern:
        filtered = [p for p in candidates if pattern in p.name]
        if filtered:
            candidates = filtered
    elif symbol:
        filtered = [p for p in candidates if symbol in p.name]
        if filtered:
            candidates = filtered
    return str(candidates[0])


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Optuna BO para XGBoost (classificação PR-AUC/F1)")
    p.add_argument("--data-dir", type=str, default="src/data/raw")
    p.add_argument("--csv", type=str, default=None)
    p.add_argument("--csv-pattern", type=str, default=None)
    p.add_argument("--symbol", type=str, default="BTCUSDT")
    p.add_argument("--rule", type=str, default="1min")
    p.add_argument("--horizon", type=int, default=60)

    # Walk-forward purgado
    p.add_argument("--n_splits", type=int, default=6)
    p.add_argument("--train_min_points", type=int, default=2000)
    p.add_argument("--val_points", type=int, default=1000)
    p.add_argument("--embargo_points", type=int, default=60)
    p.add_argument("--step", type=int, default=1000)

    # Optuna
    p.add_argument("--n_trials", type=int, default=60)
    p.add_argument("--pruner", type=str, default="median", choices=["median", "asha"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--study", type=str, default="xgb_bo")

    # Target metric
    p.add_argument("--metric", type=str, default="pr_auc", choices=["pr_auc", "f1"])  # maximize
    p.add_argument("--calibrate", action="store_true")
    p.add_argument("--calib_method", type=str, default="sigmoid", choices=["sigmoid", "isotonic"])
    return p.parse_args()


def build_data(csv_path: str, rule: str) -> pd.DataFrame:
    df = load_ohlcv_csv(csv_path)
    df = resample_ohlcv(df, rule)
    return df


def make_pruner(kind: str):
    if kind == "asha":
        return optuna.pruners.SuccessiveHalvingPruner()
    return optuna.pruners.MedianPruner()


def objective_factory(df: pd.DataFrame, args: argparse.Namespace):
    def objective(trial: optuna.Trial) -> float:
        # Hyperparams
        lags = trial.suggest_categorical("lags", [(1, 3, 5, 10, 20), (1, 2, 4, 8, 16)])
        windows = trial.suggest_categorical("windows", [(5, 15, 60, 240), (10, 30, 120, 480)])
        lr = trial.suggest_float("learning_rate", 0.03, 0.15)
        max_depth = trial.suggest_int("max_depth", 3, 8)
        min_child_weight = trial.suggest_float("min_child_weight", 1.0, 10.0)
        subsample = trial.suggest_float("subsample", 0.6, 0.95)
        colsample_bytree = trial.suggest_float("colsample_bytree", 0.5, 0.95)
        gamma = trial.suggest_float("gamma", 0.0, 5.0)
        reg_lambda = trial.suggest_float("reg_lambda", 0.5, 5.0)
        reg_alpha = trial.suggest_float("reg_alpha", 0.0, 2.0)
        n_estimators = trial.suggest_int("n_estimators", 300, 1500)

        # Features
        df_feat = make_tabular_features(df, lags=list(lags), windows=list(windows))
        # Labels TB (+1 vs resto)
        y_all = triple_barrier_labels(df_feat, horizon=args.horizon)
        y_all = (y_all == 1).astype(int)
        feature_cols = [c for c in df_feat.columns if c not in {"open", "high", "low", "close", "volume"}]
        X_all = df_feat[feature_cols]
        mask = y_all.notna()
        X_all = X_all.loc[mask]
        y_all = y_all.loc[mask]

        # Splits purgados
        splits = purged_walk_forward(
            index=X_all.index,
            n_splits=args.n_splits,
            train_min_points=args.train_min_points,
            val_points=args.val_points,
            embargo_points=args.embargo_points,
            step=args.step,
        )
        if not splits:
            raise optuna.TrialPruned("Sem splits válidos")

        scores: List[float] = []
        for (idx_tr, idx_va) in splits:
            Xtr, Xva = X_all.iloc[idx_tr], X_all.iloc[idx_va]
            ytr, yva = y_all.iloc[idx_tr], y_all.iloc[idx_va]

            Xtr_s, Xva_s, _Xte_s, _ = fit_transform_no_leak(Xtr, Xva, Xva, scaler_kind="standard")

            params = dict(
                tree_method="hist",
                learning_rate=lr,
                max_depth=max_depth,
                min_child_weight=min_child_weight,
                subsample=subsample,
                colsample_bytree=colsample_bytree,
                gamma=gamma,
                reg_lambda=reg_lambda,
                reg_alpha=reg_alpha,
                n_estimators=n_estimators,
                objective="binary:logistic",
                eval_metric="logloss",
            )
            # scale_pos_weight baseline
            pos = max(1, int((ytr == 1).sum()))
            neg = max(1, int((ytr == 0).sum()))
            params["scale_pos_weight"] = neg / pos

            model = xgb.XGBClassifier(**params)
            model.fit(Xtr_s, ytr, eval_set=[(Xva_s, yva)], verbose=False, early_stopping_rounds=100)

            proba_va = model.predict_proba(Xva_s)[:, 1]
            # Calibração opcional dentro do split
            if args.calibrate:
                calibrator = CalibratedClassifierCV(model, method=args.calib_method, cv="prefit")
                calibrator.fit(Xva_s, yva)
                proba_va = calibrator.predict_proba(Xva_s)[:, 1]

            if args.metric == "pr_auc":
                score = float(average_precision_score(yva, proba_va))
            else:
                # threshold por F1 no próprio val
                taus = np.unique(np.percentile(proba_va, np.linspace(1, 99, 25)))
                best = 0.0
                for t in taus:
                    preds = (proba_va >= t).astype(int)
                    best = max(best, f1_score(yva, preds, zero_division=0))
                score = float(best)

            scores.append(score)

            # Relato e pruning
            trial.report(float(np.mean(scores)), step=len(scores))
            if trial.should_prune():
                raise optuna.TrialPruned()

        # Maximizar métrica: Optuna maximiza via direction, mas aqui devolvemos a média
        return float(np.mean(scores))

    return objective


def main() -> None:
    args = parse_args()

    if args.csv is None:
        csv_path = select_csv_from_dir(args.data_dir, args.symbol, args.csv_pattern)
        print({"csv_selected": csv_path})
    else:
        csv_path = args.csv

    df = build_data(csv_path, args.rule)

    optuna.logging.set_verbosity(optuna.logging.INFO)
    sampler = optuna.samplers.TPESampler(seed=args.seed)
    study = optuna.create_study(direction="maximize", study_name=args.study, sampler=sampler, pruner=make_pruner(args.pruner))
    study.optimize(objective_factory(df, args), n_trials=args.n_trials, gc_after_trial=True)

    print({"best_value": study.best_value, "best_params": study.best_params})


if __name__ == "__main__":
    main()
