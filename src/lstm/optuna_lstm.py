from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import optuna
import pandas as pd
import tensorflow as tf
from optuna.integration import TFKerasPruningCallback

from data.dataset import (
    add_base_features,
    fit_transform_no_leak,
    load_ohlcv_csv,
    make_horizon_target,
    resample_ohlcv,
    rolling_origin_splits,
)
from data.metrics import rmse as rmse_metric
from lstm.lstm import LSTMConfig, build_lstm_model, compile_regression_model


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
    p = argparse.ArgumentParser(description="Optuna BO para LSTM (regressão de retornos)")
    p.add_argument("--data-dir", type=str, default="src/data/raw")
    p.add_argument("--csv", type=str, default=None)
    p.add_argument("--csv-pattern", type=str, default=None)
    p.add_argument("--symbol", type=str, default="BTCUSDT")
    p.add_argument("--rule", type=str, default="1min")
    p.add_argument("--horizon", type=int, default=5)

    p.add_argument("--train", type=str, nargs=2, default=["2018-01-01", "2025-07-31"])
    p.add_argument("--val", type=str, nargs=2, default=["2025-08-01", "2025-08-15"])
    p.add_argument("--test", type=str, nargs=2, default=["2025-08-16", "2025-08-31"])

    # Rolling-origin config
    p.add_argument("--ro_splits", type=int, default=5)
    p.add_argument("--ro_step", type=int, default=1000)
    p.add_argument("--ro_val", type=int, default=500)
    p.add_argument("--ro_test", type=int, default=500)

    # Optuna config
    p.add_argument("--n_trials", type=int, default=50)
    p.add_argument("--pruner", type=str, default="median", choices=["median", "asha"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--study", type=str, default="lstm_bo")
    return p.parse_args()


def build_data(csv_path: str, rule: str, horizon: int, train: Tuple[str, str], val: Tuple[str, str], test: Tuple[str, str]):
    df = load_ohlcv_csv(csv_path)
    df = resample_ohlcv(df, rule)
    df = add_base_features(df)
    y = make_horizon_target(df, horizon)
    features = ["open", "high", "low", "close", "volume", "ret_1", "vol_realizada_30"]
    X = df[features].copy()
    mask = y.notna()
    X = X.loc[mask]
    y = y.loc[mask]
    return X, y


def make_pruner(kind: str):
    if kind == "asha":
        return optuna.pruners.SuccessiveHalvingPruner()
    return optuna.pruners.MedianPruner()


def objective_factory(
    X_all: pd.DataFrame,
    y_all: pd.Series,
    args: argparse.Namespace,
):
    def objective(trial: optuna.Trial) -> float:
        # Espaço de busca
        lookback = trial.suggest_int("lookback", 32, 256, step=16)
        hidden_size = trial.suggest_int("hidden_size", 64, 256, step=32)
        num_layers = trial.suggest_int("num_layers", 1, 3)
        dropout = trial.suggest_float("dropout", 0.0, 0.5)
        lr = trial.suggest_float("lr", 1e-4, 3e-3, log=True)
        batch_size = trial.suggest_int("batch_size", 64, 512, step=64)
        bidirectional = trial.suggest_categorical("bidirectional", [False, True])

        # Splits rolling-origin
        splits = rolling_origin_splits(
            index=X_all.index,
            n_splits=args.ro_splits,
            train_min_points=max(lookback + args.ro_step, lookback + args.ro_val),
            val_points=args.ro_val,
            test_points=args.ro_test,
            step=args.ro_step,
        )
        if not splits:
            raise optuna.TrialPruned("Splits insuficientes para os parâmetros atuais")

        rmses: List[float] = []
        split_id = 0
        for (idx_tr, idx_va, _idx_te) in splits:
            split_id += 1
            Xtr, Xva = X_all.iloc[idx_tr], X_all.iloc[idx_va]
            ytr, yva = y_all.iloc[idx_tr], y_all.iloc[idx_va]

            # Escalonamento sem vazamento
            Xtr_s, Xva_s, _Xte_s, _scaler = fit_transform_no_leak(Xtr, Xva, Xva, scaler_kind="standard")

            # Janelamento
            from data.dataset import windowify

            Xtr_w, ytr_w = windowify(Xtr_s, ytr, lookback=lookback)
            Xva_w, yva_w = windowify(Xva_s, yva, lookback=lookback)

            if len(Xtr_w) == 0 or len(Xva_w) == 0:
                raise optuna.TrialPruned("Janelas insuficientes")

            cfg = LSTMConfig(
                input_size=Xtr_s.shape[1],
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout,
                bidirectional=bidirectional,
                output_size=1,
            )
            model = build_lstm_model(cfg, lookback=lookback)
            model = compile_regression_model(model, lr=lr, loss="mse")

            callbacks = [
                tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
                tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6),
                TFKerasPruningCallback(trial, monitor="val_loss"),
            ]

            model.fit(
                Xtr_w,
                ytr_w,
                validation_data=(Xva_w, yva_w),
                epochs=50,
                batch_size=batch_size,
                verbose=0,
                callbacks=callbacks,
                shuffle=False,
            )
            y_pred = model.predict(Xva_w, verbose=0).reshape(-1)
            rmse_val = rmse_metric(yva_w, y_pred)
            rmses.append(rmse_val)

            # Reporta progresso e permite pruning por split
            trial.report(float(np.mean(rmses)), step=split_id)
            if trial.should_prune():
                raise optuna.TrialPruned()

        return float(np.mean(rmses))

    return objective


def main() -> None:
    args = parse_args()

    # Seleção de dataset
    if args.csv is None:
        csv_path = select_csv_from_dir(args.data_dir, args.symbol, args.csv_pattern)
        print({"csv_selected": csv_path})
    else:
        csv_path = args.csv

    X, y = build_data(csv_path, args.rule, args.horizon, tuple(args.train), tuple(args.val), tuple(args.test))

    optuna.logging.set_verbosity(optuna.logging.INFO)
    sampler = optuna.samplers.TPESampler(seed=args.seed)
    study = optuna.create_study(
        direction="minimize",
        study_name=args.study,
        sampler=sampler,
        pruner=make_pruner(args.pruner),
    )
    study.optimize(objective_factory(X, y, args), n_trials=args.n_trials, gc_after_trial=True)

    print({"best_value": study.best_value, "best_params": study.best_params})


if __name__ == "__main__":
    main()
