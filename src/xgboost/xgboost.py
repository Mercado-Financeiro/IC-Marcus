from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    fbeta_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

from data.dataset import (
    add_base_features,
    fit_transform_no_leak,
    load_ohlcv_csv,
    make_horizon_target,
    resample_ohlcv,
)


# =============================
# Features
# =============================


def make_tabular_features(
    df: pd.DataFrame,
    lags: List[int],
    windows: List[int],
) -> pd.DataFrame:
    out = df.copy()

    log_close = np.log(out["close"]).astype(float)
    out["ret_1"] = log_close.diff(1)

    # Lags de retorno
    for k in lags:
        out[f"ret_{k}"] = log_close.diff(k)

    # Volatilidades e médias móveis
    for w in windows:
        out[f"rv_{w}"] = out["ret_1"].rolling(w, min_periods=w).std()
        out[f"sma_close_{w}"] = out["close"].rolling(w, min_periods=w).mean()
        out[f"z_close_{w}"] = (out["close"] - out[f"sma_close_{w}"]) / (out[f"sma_close_{w}"] .rolling(w, min_periods=w).std())

    # Range-based proxies (Garman-Klass aprox.)
    hl = np.log(out["high"]) - np.log(out["low"])
    co = np.log(out["close"]) - np.log(out["open"])
    out["gk_proxy"] = 0.5 * (hl ** 2) - (2.0 * np.log(2) - 1) * (co ** 2)

    out = out.replace([np.inf, -np.inf], np.nan).dropna()
    return out


# =============================
# Triple Barrier (aproximação OHLC)
# =============================


def triple_barrier_labels(
    df: pd.DataFrame,
    horizon: int,
    vol_window: int = 30,
    pt_mult: float = 2.0,
    sl_mult: float = 1.5,
) -> pd.Series:
    """Retorna rótulos {+1, 0, -1} usando barreiras superior/inferior por múltiplos da vol e barreira temporal H.

    Aproxima com OHLC: verifica, entre t+1..t+H, se o máximo de log(high/close_t) cruza +pt ou o mínimo de log(low/close_t) cruza -sl. Caso nenhum, 0.
    """
    out_index = df.index
    log_close = np.log(df["close"]).values
    log_high = np.log(df["high"]).values
    log_low = np.log(df["low"]).values

    # Volatilidade realizada de 1-passo como proxy
    vol = pd.Series(np.log(df["close"]).diff(1).rolling(vol_window, min_periods=vol_window).std(), index=out_index).values

    labels = np.zeros(len(df), dtype=int)

    for i in range(len(df) - horizon):
        sigma = vol[i]
        if not np.isfinite(sigma) or sigma <= 0:
            continue
        upper = pt_mult * sigma
        lower = -sl_mult * sigma
        ref = log_close[i]
        # janela futura
        fut_high = log_high[i + 1 : i + horizon + 1]
        fut_low = log_low[i + 1 : i + horizon + 1]
        # Retornos máximos/mínimos relativos
        max_ret = np.max(fut_high - ref)
        min_ret = np.min(fut_low - ref)
        if max_ret >= upper and min_ret <= lower:
            # em ambiguidade, prioriza o primeiro a ocorrer: aproximação usando extremos não dá ordem; escolhe neutro
            labels[i] = 0
        elif max_ret >= upper:
            labels[i] = 1
        elif min_ret <= lower:
            labels[i] = -1
        else:
            labels[i] = 0

    return pd.Series(labels, index=out_index)


# =============================
# Purged walk-forward com embargo
# =============================


def purged_walk_forward(
    index: pd.DatetimeIndex,
    n_splits: int,
    train_min_points: int,
    val_points: int,
    embargo_points: int,
    step: int,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    n = len(index)
    splits: List[Tuple[np.ndarray, np.ndarray]] = []
    start_train_end = train_min_points

    for k in range(n_splits):
        train_end = start_train_end + k * step
        val_start = train_end + embargo_points
        val_end = val_start + val_points
        if val_end > n:
            break
        idx_train = np.arange(0, train_end)
        idx_val = np.arange(val_start, val_end)
        splits.append((idx_train, idx_val))
    return splits


# =============================
# Calibração e métricas
# =============================


def expected_calibration_error(y_true: np.ndarray, proba: np.ndarray, n_bins: int = 15) -> float:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    inds = np.digitize(proba, bins) - 1
    ece = 0.0
    for b in range(n_bins):
        mask = inds == b
        if np.any(mask):
            conf = np.mean(proba[mask])
            acc = np.mean((proba[mask] >= 0.5) == (y_true[mask] == 1))
            ece += np.abs(acc - conf) * np.mean(mask)
    return float(ece)


def select_threshold_by_f1(y_true: np.ndarray, proba: np.ndarray) -> Tuple[float, Dict[str, float]]:
    thresholds = np.unique(np.percentile(proba, np.linspace(1, 99, 99)))
    best_tau, best_f1 = 0.5, -1.0
    for t in thresholds:
        preds = (proba >= t).astype(int)
        f1 = f1_score(y_true, preds, zero_division=0)
        if f1 > best_f1:
            best_f1, best_tau = f1, float(t)
    prec = precision_score(y_true, (proba >= best_tau).astype(int), zero_division=0)
    rec = recall_score(y_true, (proba >= best_tau).astype(int), zero_division=0)
    return best_tau, {"f1": float(best_f1), "precision": float(prec), "recall": float(rec)}


# =============================
# Quantile objective
# =============================


def quantile_objective(q: float):
    def _obj(y_pred: np.ndarray, dtrain: xgb.DMatrix):
        y = dtrain.get_label()
        e = y_pred - y
        grad = np.where(e < 0.0, q - 1.0, q)
        hess = np.full_like(grad, 1e-6)
        return grad, hess

    return _obj


def pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, q: float) -> float:
    e = y_true - y_pred
    return float(np.mean(np.maximum(q * e, (q - 1) * e)))


# =============================
# Treino principal
# =============================


@dataclass
class XGBConfig:
    task: str  # 'class' | 'reg' | 'quantile'
    horizon: int
    lags: Tuple[int, ...] = (1, 3, 5, 10, 20)
    windows: Tuple[int, ...] = (5, 15, 60, 240)
    scaler: str = "standard"

    # XGBoost
    tree_method: str = "hist"
    device: Optional[str] = None  # e.g., 'cuda:0'
    n_estimators: int = 1000
    learning_rate: float = 0.07
    max_depth: int = 5
    min_child_weight: float = 1.0
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    gamma: float = 0.0
    reg_lambda: float = 1.0
    reg_alpha: float = 0.0
    early_stopping_rounds: int = 100
    scale_pos_weight: Optional[float] = None

    # Quantile
    quantiles: Tuple[float, ...] = (0.1, 0.5, 0.9)


def train_xgb_pipeline(
    df: pd.DataFrame,
    cfg: XGBConfig,
    train: Tuple[str, str],
    val: Tuple[str, str],
    test: Tuple[str, str],
    calibrate: bool = True,
    calib_method: str = "sigmoid",
) -> Dict[str, object]:
    # Features tabulares
    df_feat = make_tabular_features(df, lags=list(cfg.lags), windows=list(cfg.windows))

    # Targets
    if cfg.task == "class":
        y_all = triple_barrier_labels(df_feat, horizon=cfg.horizon)
        # Binário: +1 vs resto
        y_all = (y_all == 1).astype(int)
    elif cfg.task == "reg":
        y_all = make_horizon_target(df_feat, cfg.horizon)
    elif cfg.task == "quantile":
        y_all = make_horizon_target(df_feat, cfg.horizon)
    else:
        raise ValueError("Tarefa inválida: use 'class', 'reg' ou 'quantile'")

    # Seleção de features (exclui alvo e preços brutos para evitar vazamento óbvio)
    feature_cols = [c for c in df_feat.columns if c not in {"open", "high", "low", "close", "volume"}]
    X_all = df_feat[feature_cols]

    # Alinha e remove NaNs
    mask = y_all.notna()
    X_all = X_all.loc[mask]
    y_all = y_all.loc[mask]

    # Splits por data
    t0, t1 = pd.to_datetime(train[0], utc=True), pd.to_datetime(train[1], utc=True)
    v0, v1 = pd.to_datetime(val[0], utc=True), pd.to_datetime(val[1], utc=True)
    s0, s1 = pd.to_datetime(test[0], utc=True), pd.to_datetime(test[1], utc=True)
    X_tr = X_all.loc[(X_all.index >= t0) & (X_all.index <= t1)]
    X_va = X_all.loc[(X_all.index >= v0) & (X_all.index <= v1)]
    X_te = X_all.loc[(X_all.index >= s0) & (X_all.index <= s1)]
    y_tr = y_all.loc[X_tr.index]
    y_va = y_all.loc[X_va.index]
    y_te = y_all.loc[X_te.index]

    # Escalonamento sem vazamento
    X_tr_s, X_va_s, X_te_s, scaler = fit_transform_no_leak(X_tr, X_va, X_te, scaler_kind=cfg.scaler)

    results: Dict[str, object] = {}

    if cfg.task == "class":
        params = dict(
            tree_method=cfg.tree_method,
            learning_rate=cfg.learning_rate,
            max_depth=cfg.max_depth,
            min_child_weight=cfg.min_child_weight,
            subsample=cfg.subsample,
            colsample_bytree=cfg.colsample_bytree,
            gamma=cfg.gamma,
            reg_lambda=cfg.reg_lambda,
            reg_alpha=cfg.reg_alpha,
            n_estimators=cfg.n_estimators,
            objective="binary:logistic",
            eval_metric="logloss",
        )
        if cfg.device:
            params["device"] = cfg.device
            params["tree_method"] = "gpu_hist"

        # scale_pos_weight
        if cfg.scale_pos_weight is None:
            pos = max(1, int((y_tr == 1).sum()))
            neg = max(1, int((y_tr == 0).sum()))
            spw = neg / pos
        else:
            spw = cfg.scale_pos_weight
        params["scale_pos_weight"] = spw

        model = xgb.XGBClassifier(**params)
        model.fit(
            X_tr_s,
            y_tr,
            eval_set=[(X_va_s, y_va)],
            verbose=False,
            early_stopping_rounds=cfg.early_stopping_rounds,
        )

        proba_va = model.predict_proba(X_va_s)[:, 1]
        proba_te = model.predict_proba(X_te_s)[:, 1]

        # Calibração (opcional) com Platt/Isotonic
        if calibrate:
            calibrator = CalibratedClassifierCV(model, method=calib_method, cv="prefit")
            calibrator.fit(X_va_s, y_va)
            proba_va = calibrator.predict_proba(X_va_s)[:, 1]
            proba_te = calibrator.predict_proba(X_te_s)[:, 1]
            results["calibrator"] = calibrator

        # Threshold via F1
        tau, t_metrics = select_threshold_by_f1(y_va.values, proba_va)
        preds_te = (proba_te >= tau).astype(int)

        metrics = {
            "val_ap": float(average_precision_score(y_va, proba_va)),
            "val_roc": float(roc_auc_score(y_va, proba_va)) if y_va.nunique() == 2 else float("nan"),
            "val_brier": float(brier_score_loss(y_va, proba_va)),
            "val_ece": float(expected_calibration_error(y_va.values, proba_va)),
            "val_f1": float(t_metrics["f1"]),
            "val_precision": float(t_metrics["precision"]),
            "val_recall": float(t_metrics["recall"]),
            "tau": float(tau),
            "test_ap": float(average_precision_score(y_te, proba_te)),
            "test_roc": float(roc_auc_score(y_te, proba_te)) if y_te.nunique() == 2 else float("nan"),
            "test_brier": float(brier_score_loss(y_te, proba_te)),
            "test_ece": float(expected_calibration_error(y_te.values, proba_te)),
            "test_f1": float(f1_score(y_te, preds_te, zero_division=0)),
            "test_precision": float(precision_score(y_te, preds_te, zero_division=0)),
            "test_recall": float(recall_score(y_te, preds_te, zero_division=0)),
        }

        # SHAP (amostra)
        explainer = shap.TreeExplainer(model)
        sample_rows = min(2000, len(X_te_s))
        shap_vals = explainer.shap_values(X_te_s.iloc[:sample_rows])
        results.update({
            "model": model,
            "metrics": metrics,
            "proba_te": proba_te,
            "preds_te": preds_te,
            "feature_importance_gain": getattr(model, "feature_importances_", None),
            "shap_values_sample": shap_vals,
            "features": feature_cols,
            "scaler": scaler,
        })
        return results

    if cfg.task == "reg":
        params = dict(
            tree_method=cfg.tree_method,
            learning_rate=cfg.learning_rate,
            max_depth=cfg.max_depth,
            min_child_weight=cfg.min_child_weight,
            subsample=cfg.subsample,
            colsample_bytree=cfg.colsample_bytree,
            gamma=cfg.gamma,
            reg_lambda=cfg.reg_lambda,
            reg_alpha=cfg.reg_alpha,
            n_estimators=cfg.n_estimators,
            objective="reg:squarederror",
        )
        if cfg.device:
            params["device"] = cfg.device
            params["tree_method"] = "gpu_hist"

        model = xgb.XGBRegressor(**params)
        model.fit(
            X_tr_s,
            y_tr,
            eval_set=[(X_va_s, y_va)],
            verbose=False,
            early_stopping_rounds=cfg.early_stopping_rounds,
        )
        y_pred = model.predict(X_te_s)
        from data.metrics import rmse as rmse_metric, mae as mae_metric, mase as mase_metric

        metrics = {
            "rmse": rmse_metric(y_te.values, y_pred),
            "mae": mae_metric(y_te.values, y_pred),
            "mase": mase_metric(y_te.values, y_pred, y_tr.values),
        }

        results.update({
            "model": model,
            "metrics": metrics,
            "y_pred": y_pred,
            "features": feature_cols,
            "scaler": scaler,
        })
        return results

    if cfg.task == "quantile":
        # DMatrix + custom objective para cada quantil
        Xtrm = xgb.DMatrix(X_tr_s, label=y_tr.values)
        Xvam = xgb.DMatrix(X_va_s, label=y_va.values)
        Xtem = xgb.DMatrix(X_te_s, label=y_te.values)
        base_params = {
            "tree_method": cfg.tree_method,
            "eta": cfg.learning_rate,
            "max_depth": cfg.max_depth,
            "min_child_weight": cfg.min_child_weight,
            "subsample": cfg.subsample,
            "colsample_bytree": cfg.colsample_bytree,
            "gamma": cfg.gamma,
            "lambda": cfg.reg_lambda,
            "alpha": cfg.reg_alpha,
            "objective": "reg:squarederror",
        }
        if cfg.device:
            base_params["device"] = cfg.device
            base_params["tree_method"] = "gpu_hist"

        preds: Dict[float, np.ndarray] = {}
        pinballs: Dict[float, float] = {}
        boosters: Dict[float, xgb.Booster] = {}
        for q in cfg.quantiles:
            booster = xgb.train(
                params=base_params,
                dtrain=Xtrm,
                num_boost_round=cfg.n_estimators,
                evals=[(Xtrm, "trn"), (Xvam, "val")],
                obj=quantile_objective(q),
                verbose_eval=False,
                early_stopping_rounds=cfg.early_stopping_rounds,
            )
            yq = booster.predict(Xtem)
            preds[q] = yq
            pinballs[q] = pinball_loss(y_te.values, yq, q)
            boosters[q] = booster

        # Coverage aproximada
        q10, q50, q90 = preds[cfg.quantiles[0]], preds[cfg.quantiles[1]], preds[cfg.quantiles[2]]
        coverage = float(np.mean((y_te.values >= q10) & (y_te.values <= q90)))

        results.update({
            "boosters": boosters,
            "pred_q": preds,
            "pinball": pinballs,
            "coverage_q10_q90": coverage,
            "features": feature_cols,
            "scaler": scaler,
        })
        return results

    raise AssertionError("Caminho não alcançável")


# =============================
# CLI
# =============================


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Treino XGBoost para cripto: class/reg/quantile")
    p.add_argument("--task", type=str, choices=["class", "reg", "quantile"], required=True)
    p.add_argument("--horizon", type=int, default=60)
    p.add_argument("--rule", type=str, default="1min")
    p.add_argument("--data-dir", type=str, default="src/data/raw")
    p.add_argument("--csv", type=str, default=None)
    p.add_argument("--csv-pattern", type=str, default=None)
    p.add_argument("--symbol", type=str, default="BTCUSDT")

    p.add_argument("--train", type=str, nargs=2, default=["2018-01-01", "2025-07-31"])
    p.add_argument("--val", type=str, nargs=2, default=["2025-08-01", "2025-08-15"])
    p.add_argument("--test", type=str, nargs=2, default=["2025-08-16", "2025-08-31"])

    p.add_argument("--lags", type=int, nargs="*", default=[1, 3, 5, 10, 20])
    p.add_argument("--windows", type=int, nargs="*", default=[5, 15, 60, 240])
    p.add_argument("--scaler", type=str, default="standard", choices=["standard", "robust", "minmax"])

    p.add_argument("--n_estimators", type=int, default=1000)
    p.add_argument("--learning_rate", type=float, default=0.07)
    p.add_argument("--max_depth", type=int, default=5)
    p.add_argument("--min_child_weight", type=float, default=1.0)
    p.add_argument("--subsample", type=float, default=0.8)
    p.add_argument("--colsample_bytree", type=float, default=0.8)
    p.add_argument("--gamma", type=float, default=0.0)
    p.add_argument("--reg_lambda", type=float, default=1.0)
    p.add_argument("--reg_alpha", type=float, default=0.0)
    p.add_argument("--early_stopping_rounds", type=int, default=100)
    p.add_argument("--device", type=str, default=None)

    p.add_argument("--calibrate", action="store_true")
    p.add_argument("--calib_method", type=str, default="sigmoid", choices=["sigmoid", "isotonic"])

    p.add_argument("--quantiles", type=float, nargs="*", default=[0.1, 0.5, 0.9])

    return p.parse_args()


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


def main() -> None:
    args = parse_args()

    # Carrega dados
    if args.csv is None:
        csv_path = select_csv_from_dir(args.data_dir, args.symbol, args.csv_pattern)
        print({"csv_selected": csv_path})
    else:
        csv_path = args.csv
    df = load_ohlcv_csv(csv_path)
    df = resample_ohlcv(df, args.rule)

    cfg = XGBConfig(
        task=args.task,
        horizon=args.horizon,
        lags=tuple(args.lags),
        windows=tuple(args.windows),
        scaler=args.scaler,
        tree_method="hist",
        device=args.device,
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        max_depth=args.max_depth,
        min_child_weight=args.min_child_weight,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        gamma=args.gamma,
        reg_lambda=args.reg_lambda,
        reg_alpha=args.reg_alpha,
        early_stopping_rounds=args.early_stopping_rounds,
        quantiles=tuple(args.quantiles),
    )

    results = train_xgb_pipeline(
        df=df,
        cfg=cfg,
        train=tuple(args.train),
        val=tuple(args.val),
        test=tuple(args.test),
        calibrate=args.calibrate if args.task == "class" else False,
        calib_method=args.calib_method,
    )

    # Imprime resumo das principais métricas
    if args.task == "class":
        print(results["metrics"])
    elif args.task == "reg":
        print(results["metrics"])
    elif args.task == "quantile":
        out = {"pinball": results["pinball"], "coverage_q10_q90": results["coverage_q10_q90"]}
        print(out)


if __name__ == "__main__":
    main()
