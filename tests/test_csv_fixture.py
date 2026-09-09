"""Runs the engine against a CSV file on disk instead of an in-memory
DataFrame -- the earlier tests all build data with numpy/pandas directly,
which doesn't exercise the read_csv/index-parsing path a real user hits.
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backtester import Backtester, Signal

FIXTURE = Path(__file__).resolve().parents[1] / "examples" / "fixtures" / "sample_ohlcv.csv"


def test_fixture_file_exists():
    assert FIXTURE.exists(), "run examples/fixtures/generate_fixture.py to create it"


def test_backtester_runs_against_csv_loaded_data():
    data = pd.read_csv(FIXTURE, parse_dates=["time"], index_col="time")
    assert {"open", "high", "low", "close"}.issubset(data.columns)

    def buy_and_hold(i, history):
        if i == 10:
            return Signal(direction=1, stop_price=history["close"].iloc[-1] * 0.5)
        return None

    bt = Backtester(data, buy_and_hold, initial_equity=10_000, risk_per_trade=0.01)
    result = bt.run()

    assert len(result.trades) == 1
    assert result.equity_curve.index.equals(data.index)
