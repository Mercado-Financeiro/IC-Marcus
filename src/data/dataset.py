from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler


# =============================
# Carregamento e padronização
# =============================


def _standardize_ohlcv_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Padroniza nomes de colunas para ['ts','open','high','low','close','volume'].

    Aceita variações comuns de timestamp e OHLCV. Lança ValueError se
    colunas essenciais estiverem ausentes.
    """
    renamed: Dict[str, str] = {}
    cols = {c.lower(): c for c in df.columns}

    # Timestamp candidates
    ts_candidates = [
        "timestamp",
        "ts",
        "time",
        "date",
        "datetime",
        "open_time",
        "opentime",
        "open time",
    ]
    ts_col = next((cols[c] for c in ts_candidates if c in cols), None)
    if ts_col:
        renamed[ts_col] = "ts"

    # Price/volume
    mapping_candidates = {
        "open": ["open", "o"],
        "high": ["high", "h"],
        "low": ["low", "l"],
        "close": ["close", "c"],
        "volume": ["volume", "vol", "v"],
    }
    for target, candidates in mapping_candidates.items():
        found = next((cols[c] for c in candidates if c in cols), None)
        if found:
            renamed[found] = target

    df = df.rename(columns=renamed)

    required = {"ts", "open", "high", "low", "close", "volume"}
    missing = required - set(df.columns.str.lower())
    if missing:
        raise ValueError(f"Colunas OHLCV ausentes: {sorted(missing)}")

    return df[["ts", "open", "high", "low", "close", "volume"]]


def load_ohlcv_csv(file_path: str, tz: str = "UTC") -> pd.DataFrame:
    """Carrega OHLCV de CSV, padroniza colunas e indexa por datetime (UTC).

    Espera colunas equivalentes a timestamp e OHLCV.
    """
    df = pd.read_csv(file_path)
    df = _standardize_ohlcv_columns(df)
    ts = pd.to_datetime(df["ts"], utc=True, errors="coerce")
    if tz and tz.upper() != "UTC":
        ts = ts.dt.tz_convert("UTC")
    df["ts"] = ts
    df = df.dropna(subset=["ts"]).sort_values("ts").reset_index(drop=True)
    df = df.set_index("ts")
    return df


def resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Reamostra OHLCV para a `rule` (ex.: '1min','5min','15min')."""
    o = df["open"].resample(rule).first()
    h = df["high"].resample(rule).max()
    l = df["low"].resample(rule).min()
    c = df["close"].resample(rule).last()
    v = df["volume"].resample(rule).sum()
    out = pd.concat({"open": o, "high": h, "low": l, "close": c, "volume": v}, axis=1)
    out = out.dropna(how="any")
    return out


# =============================
# Features e targets
# =============================


def add_base_features(df: pd.DataFrame, window_vol: int = 30) -> pd.DataFrame:
    """Adiciona features mínimas sem vazamento (até t):
    - ret_1: retorno log de 1 passo
    - vol_realizada_{window_vol}: desvio padrão dos retornos de 1 passo
    """
    out = df.copy()
    log_close = np.log(out["close"])  # sem vazamento: usa até t
    out["ret_1"] = log_close.diff(1)
    out[f"vol_realizada_{window_vol}"] = out["ret_1"].rolling(window_vol, min_periods=window_vol).std()
    out = out.dropna()
    return out


def make_horizon_target(df: pd.DataFrame, horizon: int) -> pd.Series:
    """Target de regressão: retorno log H passos à frente.
    y_t = log(close_{t+H}) - log(close_t)
    """
    log_close = np.log(df["close"])
    return log_close.shift(-horizon) - log_close


# =============================
# Splits temporais
# =============================


@dataclass
class TimeSplits:
    train: Tuple[pd.Timestamp, pd.Timestamp]
    val: Tuple[pd.Timestamp, pd.Timestamp]
    test: Tuple[pd.Timestamp, pd.Timestamp]


def temporal_split(
    df: pd.DataFrame,
    train: Tuple[str, str],
    val: Tuple[str, str],
    test: Tuple[str, str],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Realiza splits por janelas de datas inclusivas [ini, fim]."""
    t0, t1 = pd.to_datetime(train[0], utc=True), pd.to_datetime(train[1], utc=True)
    v0, v1 = pd.to_datetime(val[0], utc=True), pd.to_datetime(val[1], utc=True)
    s0, s1 = pd.to_datetime(test[0], utc=True), pd.to_datetime(test[1], utc=True)

    df_train = df.loc[(df.index >= t0) & (df.index <= t1)].copy()
    df_val = df.loc[(df.index >= v0) & (df.index <= v1)].copy()
    df_test = df.loc[(df.index >= s0) & (df.index <= s1)].copy()
    return df_train, df_val, df_test


def rolling_origin_splits(
    index: pd.DatetimeIndex,
    n_splits: int,
    train_min_points: int,
    val_points: int,
    test_points: int,
    step: int,
) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Gera índices para rolling-origin crescente.

    Retorna lista de tuplas (idx_train, idx_val, idx_test) como arrays de posições.
    """
    n = len(index)
    splits: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    start_train_end = train_min_points

    for k in range(n_splits):
        train_end = start_train_end + k * step
        val_end = train_end + val_points
        test_end = val_end + test_points

        if test_end > n:
            break

        idx_train = np.arange(0, train_end)
        idx_val = np.arange(train_end, val_end)
        idx_test = np.arange(val_end, test_end)
        splits.append((idx_train, idx_val, idx_test))

    return splits


# =============================
# Escalonamento sem vazamento
# =============================


def get_scaler(kind: str):
    kind = (kind or "standard").lower()
    if kind == "standard":
        return StandardScaler()
    if kind == "robust":
        return RobustScaler()
    if kind == "minmax":
        return MinMaxScaler()
    raise ValueError(f"Scaler inválido: {kind}")


def fit_transform_no_leak(
    df_train_X: pd.DataFrame,
    df_val_X: pd.DataFrame,
    df_test_X: pd.DataFrame,
    scaler_kind: str = "standard",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, object]:
    """Ajusta scaler apenas no treino e transforma val/test."""
    scaler = get_scaler(scaler_kind)
    scaler.fit(df_train_X.values)
    Xtr = pd.DataFrame(scaler.transform(df_train_X.values), index=df_train_X.index, columns=df_train_X.columns)
    Xva = pd.DataFrame(scaler.transform(df_val_X.values), index=df_val_X.index, columns=df_val_X.columns)
    Xte = pd.DataFrame(scaler.transform(df_test_X.values), index=df_test_X.index, columns=df_test_X.columns)
    return Xtr, Xva, Xte, scaler


# =============================
# Janelamento (windowing)
# =============================


def windowify(
    X: pd.DataFrame,
    y: pd.Series,
    lookback: int,
    stride: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """Gera janelas deslizantes X[t-lookback+1:t] -> y[t].

    - X: features indexado por datetime
    - y: alvo alinhado ao mesmo índice de X (inclui shift futuro)
    - Retorna arrays (N, lookback, n_features) e (N,)
    """
    Xv = X.values
    yv = y.values
    n = len(X)
    n_feat = X.shape[1]
    windows: List[np.ndarray] = []
    targets: List[float] = []

    for end in range(lookback, n, stride):
        x_win = Xv[end - lookback : end]
        y_val = yv[end - 1]  # alvo no último instante da janela
        if np.isfinite(y_val) and not np.isnan(x_win).any():
            windows.append(x_win)
            targets.append(y_val)

    if not windows:
        return np.empty((0, lookback, n_feat)), np.empty((0,))

    X_out = np.stack(windows, axis=0)
    y_out = np.asarray(targets)
    return X_out, y_out


# =============================
# Utilidades de métricas auxiliares
# =============================


def naive_scale(y_train: np.ndarray, m: int = 1, eps: float = 1e-8) -> float:
    """Denominador para MASE com sazonalidade m (m=1 => naive simples)."""
    if len(y_train) <= m:
        return float(np.mean(np.abs(np.diff(y_train)))) + eps
    return float(np.mean(np.abs(y_train[m:] - y_train[:-m]))) + eps
