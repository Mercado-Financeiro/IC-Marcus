from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List

from ..utils.config import load_yaml
from ..utils.logging import log


@dataclass
class Costs:
    fee_bps: float = 5.0
    slippage_bps: float = 10.0


def run_backtest(prices: List[float], signals: List[int], costs: Costs) -> dict:
    """Toy t+1 backtest: decision at t, execute at open of t+1.

    - Long-only toggling based on signal {0,1}
    - PnL = position[t] * return[t+1] - turnover_cost
    """
    n = min(len(prices), len(signals))
    if n < 3:
        return {"sharpe": 0.0, "mdd": 0.0, "ev": 0.0, "turnover": 0.0}
    # log returns proxy
    rets = [0.0] + [float(prices[i] / prices[i - 1] - 1.0) for i in range(1, n)]
    pos = [0] * n
    for t in range(n - 1):
        pos[t + 1] = signals[t]
    pnl = []
    turnover = 0
    for t in range(1, n):
        pnl.append(pos[t - 1] * rets[t])
        if pos[t] != pos[t - 1]:
            turnover += 1
            pnl[-1] -= (costs.fee_bps + costs.slippage_bps) / 1e4
    from math import sqrt

    mean = sum(pnl) / max(1, len(pnl))
    var = sum((x - mean) ** 2 for x in pnl) / max(1, len(pnl))
    sharpe = mean / (var ** 0.5 + 1e-12) * sqrt(252)
    eq = 0.0
    peak = 0.0
    mdd = 0.0
    for r in pnl:
        eq += r
        peak = max(peak, eq)
        mdd = min(mdd, eq - peak)
    return {
        "sharpe": float(sharpe),
        "mdd": float(mdd),
        "ev": float(mean),
        "turnover": float(turnover / max(1, len(pnl))),
    }


@dataclass
class BacktestConfig:
    fee_bps: float = 5.0
    slippage_bps: float = 10.0
    execution_rule: str = "next_bar_open"


class BacktestEngine:
    """Compatibility shim exposing a simple API used by the dashboard."""

    def __init__(self, config: BacktestConfig | None = None):
        self.config = config or BacktestConfig()

    def run(self, prices: List[float], signals: List[int]) -> dict:
        c = Costs(fee_bps=self.config.fee_bps, slippage_bps=self.config.slippage_bps)
        return run_backtest(prices, signals, c)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/backtest.yaml")
    ap.add_argument("--fast", action="store_true")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    c = Costs(**((cfg.get("costs")) or {}))
    # Toy price and signals for smoke
    prices = [100 + i * 0.1 for i in range(300 if not args.fast else 60)]
    # simple MA cross proxy signal
    signals = [1 if (i % 10) > 5 else 0 for i in range(len(prices))]
    metrics = run_backtest(prices, signals, c)
    out = Path("artifacts/reports")
    out.mkdir(parents=True, exist_ok=True)
    (out / "backtest_metrics.txt").write_text("\n".join([f"{k}: {v}" for k, v in metrics.items()]))
    log.info("backtest_complete", **metrics, exec_rule="next_bar_open")


if __name__ == "__main__":
    main()
