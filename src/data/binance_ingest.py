from __future__ import annotations

import time
from typing import List

import pandas as pd
import requests

from .dataset import _standardize_ohlcv_columns


BINANCE_BASE = "https://api.binance.com"


def _kline_to_row(k: List) -> dict:
    # Campos spot klines: https://binance-docs.github.io/apidocs/spot/en/#kline-candlestick-data
    return {
        "ts": int(k[0]),  # open time ms
        "open": float(k[1]),
        "high": float(k[2]),
        "low": float(k[3]),
        "close": float(k[4]),
        "volume": float(k[5]),
    }


def fetch_ohlcv_binance(
    symbol: str,
    interval: str = "1m",
    start_ms: int | None = None,
    end_ms: int | None = None,
    limit: int = 1000,
    sleep_s: float = 0.2,
) -> pd.DataFrame:
    """Baixa OHLCV (klines) da Binance Spot API em páginas até cobrir [start_ms, end_ms].

    - interval: '1m','5m','15m','1h','1d' etc.
    - timestamps em milissegundos UTC
    """
    url = f"{BINANCE_BASE}/api/v3/klines"
    params = {"symbol": symbol.upper(), "interval": interval, "limit": limit}
    cur_start = start_ms
    out: List[dict] = []

    while True:
        if cur_start is not None:
            params["startTime"] = int(cur_start)
        if end_ms is not None:
            params["endTime"] = int(end_ms)
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        if not data:
            break
        rows = [_kline_to_row(k) for k in data]
        out.extend(rows)
        # Próxima página começa no close time + 1 ms do último item
        last_open_time = data[-1][0]
        if cur_start is not None and (end_ms is not None and last_open_time >= end_ms) or len(data) < limit:
            break
        cur_start = last_open_time + 1
        time.sleep(sleep_s)

    if not out:
        return pd.DataFrame(columns=["ts", "open", "high", "low", "close", "volume"])  # vazio

    df = pd.DataFrame(out)
    # Padroniza e indexa em UTC
    df = _standardize_ohlcv_columns(df)
    df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    df = df.set_index("ts").sort_index()
    return df


