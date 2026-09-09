"""Example: EMA-cross strategy run through the engine on synthetic data."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backtester import Backtester, Signal, metrics


def make_synthetic_ohlcv(n: int = 500, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    steps = rng.normal(loc=0.0002, scale=0.01, size=n)
    close = 100 * np.exp(np.cumsum(steps))
    high = close * (1 + np.abs(rng.normal(0, 0.003, n)))
    low = close * (1 - np.abs(rng.normal(0, 0.003, n)))
    open_ = close * (1 + rng.normal(0, 0.001, n))
    index = pd.date_range("2025-01-01", periods=n, freq="1h")
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close}, index=index)


def ema_cross_strategy(fast: int = 12, slow: int = 26, atr_mult: float = 2.0):
    def strategy_fn(i: int, history: pd.DataFrame):
        if i < slow + 1:
            return None
        close = history["close"]
        fast_ema = close.ewm(span=fast, adjust=False).mean()
        slow_ema = close.ewm(span=slow, adjust=False).mean()

        cross_up = fast_ema.iloc[-2] <= slow_ema.iloc[-2] and fast_ema.iloc[-1] > slow_ema.iloc[-1]
        cross_down = fast_ema.iloc[-2] >= slow_ema.iloc[-2] and fast_ema.iloc[-1] < slow_ema.iloc[-1]

        high_low_range = (history["high"] - history["low"]).rolling(14).mean().iloc[-1]
        if pd.isna(high_low_range) or high_low_range <= 0:
            return None

        if cross_up:
            return Signal(direction=1, stop_price=history["close"].iloc[-1] - atr_mult * high_low_range)
        if cross_down:
            return Signal(direction=-1, stop_price=history["close"].iloc[-1] + atr_mult * high_low_range)
        return None

    return strategy_fn


if __name__ == "__main__":
    data = make_synthetic_ohlcv()
    bt = Backtester(data, ema_cross_strategy(), initial_equity=10_000, risk_per_trade=0.01)
    result = bt.run()
    for k, v in metrics.summary(result).items():
        print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")
