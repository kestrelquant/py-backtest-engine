import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backtester import Backtester, Signal, metrics


def make_trending_data(n=50):
    index = pd.date_range("2025-01-01", periods=n, freq="1D")
    close = np.linspace(100, 150, n)
    df = pd.DataFrame(
        {
            "open": close - 0.1,
            "high": close + 0.5,
            "low": close - 0.5,
            "close": close,
        },
        index=index,
    )
    return df


def test_engine_opens_and_holds_a_long_on_trend():
    data = make_trending_data()

    def strategy_fn(i, history):
        if i == 5:
            return Signal(direction=1, stop_price=history["close"].iloc[-1] - 5)
        return None

    bt = Backtester(data, strategy_fn, initial_equity=10_000, risk_per_trade=0.01)
    result = bt.run()

    assert len(result.trades) == 1
    trade = result.trades[0]
    assert trade.direction == 1
    assert result.equity_curve.iloc[-1] > result.equity_curve.iloc[0]


def test_stop_loss_exits_the_trade():
    n = 20
    index = pd.date_range("2025-01-01", periods=n, freq="1D")
    close = np.array([100.0] * 5 + [90.0] * (n - 5))
    data = pd.DataFrame(
        {"open": close, "high": close + 1, "low": close - 1, "close": close}, index=index
    )

    def strategy_fn(i, history):
        if i == 2:
            return Signal(direction=1, stop_price=95.0)
        return None

    bt = Backtester(data, strategy_fn, initial_equity=10_000, risk_per_trade=0.01)
    result = bt.run()

    assert len(result.closed_trades) == 1
    assert result.closed_trades[0].exit_price == 95.0


def test_no_lookahead_entry_fills_next_bar_open():
    data = make_trending_data()
    seen_indices = []

    def strategy_fn(i, history):
        seen_indices.append(i)
        assert len(history) == i + 1  # never sees beyond the current bar
        if i == 3:
            return Signal(direction=1, stop_price=history["close"].iloc[-1] - 5)
        return None

    bt = Backtester(data, strategy_fn, initial_equity=10_000, risk_per_trade=0.01)
    result = bt.run()

    trade = result.trades[0]
    assert trade.entry_time == data.index[4]
    assert trade.entry_price == data.iloc[4]["open"]


def test_metrics_summary_keys():
    data = make_trending_data()
    bt = Backtester(data, lambda i, h: None, initial_equity=10_000)
    result = bt.run()
    s = metrics.summary(result)
    assert set(s.keys()) == {"trades", "total_return", "max_drawdown", "sharpe_ratio", "win_rate"}
