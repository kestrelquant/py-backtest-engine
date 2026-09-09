import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backtester import Backtester, Signal, grid_search, walk_forward
from backtester import metrics as metrics_module


def make_trending_data(n=200):
    index = pd.date_range("2025-01-01", periods=n, freq="1D")
    close = np.linspace(100, 100 + n * 0.3, n) + np.sin(np.linspace(0, 20, n))
    df = pd.DataFrame(
        {"open": close - 0.1, "high": close + 0.6, "low": close - 0.6, "close": close}, index=index
    )
    return df


def threshold_strategy_factory(params: dict):
    """A toy strategy: go long once every `period` bars, sized off a fixed
    stop distance. Deterministic and parameter-sensitive enough to make
    grid_search's ranking behavior checkable."""
    period = params["period"]
    stop = params.get("stop", 3.0)

    def strategy_fn(i, history):
        if i > 0 and i % period == 0:
            return Signal(direction=1, stop_price=history["close"].iloc[-1] - stop)
        return None

    return strategy_fn


def test_grid_search_returns_all_combinations_sorted_desc():
    data = make_trending_data()
    results = grid_search(
        data,
        threshold_strategy_factory,
        param_grid={"period": [10, 20, 40], "stop": [2.0, 4.0]},
        score_fn=metrics_module.total_return,
    )

    assert len(results) == 6  # 3 * 2 combinations
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)


def test_grid_search_params_round_trip():
    data = make_trending_data()
    results = grid_search(
        data,
        threshold_strategy_factory,
        param_grid={"period": [15], "stop": [3.0]},
    )
    assert results[0].params == {"period": 15, "stop": 3.0}


def test_walk_forward_produces_expected_number_of_windows():
    data = make_trending_data(n=200)
    windows = walk_forward(
        data,
        threshold_strategy_factory,
        param_grid={"period": [10, 20], "stop": [3.0]},
        train_size=100,
        test_size=50,
    )
    # start=0: train[0:100] test[100:150]; start=50: train[50:150] test[150:200]
    assert len(windows) == 2
    assert windows[0].test_start == data.index[100]
    assert windows[0].test_end == data.index[149]


def test_walk_forward_test_window_never_overlaps_its_own_training_window():
    data = make_trending_data(n=200)
    windows = walk_forward(
        data,
        threshold_strategy_factory,
        param_grid={"period": [10, 20], "stop": [3.0]},
        train_size=100,
        test_size=50,
    )
    for w in windows:
        assert w.test_start > w.train_end
