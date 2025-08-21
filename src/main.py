from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import tensorflow as tf

from data.dataset import (
    add_base_features,
    fit_transform_no_leak,
    load_ohlcv_csv,
    make_horizon_target,
    resample_ohlcv,
    temporal_split,
    rolling_origin_splits,
)
from lstm.lstm import LSTMConfig, build_lstm_model, compile_regression_model
from data.metrics import rmse as rmse_metric, mae as mae_metric, smape as smape_metric, mase as mase_metric


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Treino LSTM para previsão de retornos curtos")
    parser.add_argument("--csv", type=str, default=None, help="Caminho para CSV OHLCV (opcional se usar --data-dir)")
    parser.add_argument("--data-dir", type=str, default="src/data/raw", help="Diretório com CSVs OHLCV")
    parser.add_argument("--csv-pattern", type=str, default=None, help="Substring para escolher arquivo no data-dir (ex.: BTCUSDT)")
    parser.add_argument("--rule", type=str, default="1min", help="Frequência (ex.: 1min,5min,15min)")
    parser.add_argument("--horizon", type=int, default=1, help="Horizonte H em passos")
    parser.add_argument("--lookback", type=int, default=64, help="Tamanho da janela de entrada")
    parser.add_argument("--scaler", type=str, default="standard", choices=["standard", "robust", "minmax"], help="Tipo de scaler")
    parser.add_argument("--train", type=str, nargs=2, default=["2018-01-01", "2025-07-31"], help="Intervalo treino [ini fim]")
    parser.add_argument("--val", type=str, nargs=2, default=["2025-08-01", "2025-08-15"], help="Intervalo validação [ini fim]")
    parser.add_argument("--test", type=str, nargs=2, default=["2025-08-16", "2025-08-31"], help="Intervalo teste [ini fim]")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--bidirectional", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str, default="lstm_training.log")
    parser.add_argument("--rolling", action="store_true", help="Executa avaliação rolling-origin")
    parser.add_argument("--ro_splits", type=int, default=5)
    parser.add_argument("--ro_step", type=int, default=500)
    parser.add_argument("--ro_val", type=int, default=500)
    parser.add_argument("--ro_test", type=int, default=500)
    parser.add_argument("--binance", action="store_true", help="Baixa OHLCV da Binance ao invés de CSV")
    parser.add_argument("--symbol", type=str, default="BTCUSDT", help="Símbolo Binance (ex.: BTCUSDT)")
    parser.add_argument("--interval", type=str, default="1m", help="Intervalo Binance (ex.: 1m,5m,15m)")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    tf.keras.utils.set_random_seed(seed)
    np.random.seed(seed)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    # Carregamento e resample
    if args.binance:
        from data.binance_ingest import fetch_ohlcv_binance
        # Converte datas para ms UTC, se fornecidas
        start_ms = int(pd.Timestamp(args.train[0], tz="UTC").timestamp() * 1000)
        end_ms = int(pd.Timestamp(args.test[1], tz="UTC").timestamp() * 1000)
        df = fetch_ohlcv_binance(symbol=args.symbol, interval=args.interval, start_ms=start_ms, end_ms=end_ms)
        # Se intervalo difere de rule, reamostra
        if args.interval != args.rule:
            df = resample_ohlcv(df, args.rule)
    else:
        # Seleciona arquivo a partir de --csv ou --data-dir
        csv_path = args.csv
        if csv_path is None:
            base = Path(args.data_dir)
            if not base.exists():
                raise FileNotFoundError(f"Diretório não encontrado: {base}")
            candidates = sorted([p for p in base.rglob("*.csv")])
            if not candidates:
                raise FileNotFoundError(f"Nenhum CSV encontrado em {base}")
            if args.csv_pattern:
                filtered = [p for p in candidates if args.csv_pattern in p.name]
                if filtered:
                    candidates = filtered
            elif args.symbol:
                filtered = [p for p in candidates if args.symbol in p.name]
                if filtered:
                    candidates = filtered
            csv_path = str(candidates[0])
            print({"csv_selected": csv_path})
        df = load_ohlcv_csv(csv_path)
        df = resample_ohlcv(df, args.rule)
    df = add_base_features(df)

    # Target de regressão H-passos
    y = make_horizon_target(df, args.horizon)

    # Seleção de features base
    features = ["open", "high", "low", "close", "volume", "ret_1", "vol_realizada_30"]
    X = df[features].copy()
    mask = y.notna()
    X = X.loc[mask]
    y = y.loc[mask]

    # Splits temporais
    df_train, df_val, df_test = temporal_split(X, tuple(args.train), tuple(args.val), tuple(args.test))
    y_train, y_val, y_test = temporal_split(y.to_frame("y"), tuple(args.train), tuple(args.val), tuple(args.test))
    y_train = y_train["y"]
    y_val = y_val["y"]
    y_test = y_test["y"]

    # Escalonamento sem vazamento
    Xtr, Xva, Xte, scaler = fit_transform_no_leak(df_train, df_val, df_test, scaler_kind=args.scaler)

    # Janelamento
    from data.dataset import windowify

    Xtr_w, ytr_w = windowify(Xtr, y_train, lookback=args.lookback)
    Xva_w, yva_w = windowify(Xva, y_val, lookback=args.lookback)
    Xte_w, yte_w = windowify(Xte, y_test, lookback=args.lookback)

    input_size = Xtr.shape[1]
    cfg = LSTMConfig(
        input_size=input_size,
        hidden_size=args.hidden,
        num_layers=args.layers,
        dropout=args.dropout,
        bidirectional=args.bidirectional,
        output_size=1,
    )

    model = build_lstm_model(cfg, lookback=args.lookback)
    model = compile_regression_model(model, lr=args.lr, loss="mse")

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6),
        tf.keras.callbacks.CSVLogger(args.out, append=True),
    ]

    model.fit(
        Xtr_w,
        ytr_w,
        validation_data=(Xva_w, yva_w),
        epochs=args.epochs,
        batch_size=args.batch,
        verbose=2,
        callbacks=callbacks,
        shuffle=False,
    )

    # Avaliação
    y_pred = model.predict(Xte_w, verbose=0).reshape(-1)
    rmse = rmse_metric(yte_w, y_pred)
    mae = mae_metric(yte_w, y_pred)
    smape = smape_metric(yte_w, y_pred)
    # Para MASE, usa série de treino do alvo (y_train) como referência
    mase = mase_metric(yte_w, y_pred, y_train.values)

    print({"rmse": rmse, "mae": mae, "smape": smape, "mase": mase, "n_test": int(len(yte_w))})

    if args.rolling:
        print("Executando rolling-origin...")
        # Concatena treino+val+teste para gerar splits por índice
        X_all = pd.concat([df_train, df_val, df_test])
        y_all = pd.concat([y_train, y_val, y_test])
        splits = rolling_origin_splits(
            index=X_all.index,
            n_splits=args.ro_splits,
            train_min_points=max(args.lookback + args.ro_step, args.lookback + args.ro_val),
            val_points=args.ro_val,
            test_points=args.ro_test,
            step=args.ro_step,
        )

        scores = []
        for (idx_tr, idx_va, idx_te) in splits:
            Xtr_ro, Xva_ro, Xte_ro = X_all.iloc[idx_tr], X_all.iloc[idx_va], X_all.iloc[idx_te]
            ytr_ro, yva_ro, yte_ro = y_all.iloc[idx_tr], y_all.iloc[idx_va], y_all.iloc[idx_te]

            Xtr_ro, Xva_ro, Xte_ro, _ = fit_transform_no_leak(Xtr_ro, Xva_ro, Xte_ro, scaler_kind=args.scaler)
            from data.dataset import windowify
            Xtr_w, ytr_w = windowify(Xtr_ro, ytr_ro, lookback=args.lookback)
            Xva_w, yva_w = windowify(Xva_ro, yva_ro, lookback=args.lookback)
            Xte_w, yte_w = windowify(Xte_ro, yte_ro, lookback=args.lookback)

            model_ro = build_lstm_model(cfg, lookback=args.lookback)
            model_ro = compile_regression_model(model_ro, lr=args.lr, loss="mse")
            callbacks_ro = [
                tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
                tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6),
            ]

            model_ro.fit(
                Xtr_w,
                ytr_w,
                validation_data=(Xva_w, yva_w),
                epochs=args.epochs,
                batch_size=args.batch,
                verbose=0,
                callbacks=callbacks_ro,
                shuffle=False,
            )
            y_pred_ro = model_ro.predict(Xte_w, verbose=0).reshape(-1)
            scores.append(
                {
                    "rmse": rmse_metric(yte_w, y_pred_ro),
                    "mae": mae_metric(yte_w, y_pred_ro),
                    "smape": smape_metric(yte_w, y_pred_ro),
                    "mase": mase_metric(yte_w, y_pred_ro, ytr_ro.values),
                    "n_test": int(len(yte_w)),
                }
            )

        # Resumo
        if scores:
            agg = {k: float(np.mean([s[k] for s in scores])) for k in ["rmse", "mae", "smape", "mase"]}
            print({"rolling_avg": agg, "splits": len(scores)})


if __name__ == "__main__":
    main()
