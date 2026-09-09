"""Parameter grid search and walk-forward analysis on top of Backtester.

Walk-forward exists because grid search alone answers "what parameters
would have worked best on this whole history" -- which is exactly the
kind of question that produces parameters overfit to one sample. Walk-
forward instead re-optimizes on a rolling training window and reports
performance only on the following out-of-sample window it was never
fit against, then slides forward.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Callable

import pandas as pd

from .engine import Backtester, BacktestResult
from . import metrics as metrics_module


@dataclass
class GridResult:
    params: dict
    result: BacktestResult
    score: float


def grid_search(
    data: pd.DataFrame,
    strategy_factory: Callable[[dict], Callable],
    param_grid: dict[str, list],
    score_fn: Callable[[BacktestResult], float] = metrics_module.sharpe_ratio,
    initial_equity: float = 10_000.0,
    risk_per_trade: float = 0.01,
) -> list[GridResult]:
    keys = list(param_grid.keys())
    combos = [dict(zip(keys, values)) for values in itertools.product(*param_grid.values())]

    results = []
    for params in combos:
        strategy_fn = strategy_factory(params)
        bt = Backtester(data, strategy_fn, initial_equity=initial_equity, risk_per_trade=risk_per_trade)
        result = bt.run()
        score = score_fn(result)
        results.append(GridResult(params=params, result=result, score=score))

    return sorted(results, key=lambda r: r.score, reverse=True)


@dataclass
class WalkForwardWindow:
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    best_params: dict
    train_score: float
    test_result: BacktestResult
    test_score: float


def walk_forward(
    data: pd.DataFrame,
    strategy_factory: Callable[[dict], Callable],
    param_grid: dict[str, list],
    train_size: int,
    test_size: int,
    score_fn: Callable[[BacktestResult], float] = metrics_module.sharpe_ratio,
    initial_equity: float = 10_000.0,
    risk_per_trade: float = 0.01,
) -> list[WalkForwardWindow]:
    if train_size <= 0 or test_size <= 0:
        raise ValueError("train_size and test_size must be positive")

    windows: list[WalkForwardWindow] = []
    start = 0

    while start + train_size + test_size <= len(data):
        train_slice = data.iloc[start : start + train_size]
        test_slice = data.iloc[start + train_size : start + train_size + test_size]

        ranked = grid_search(
            train_slice,
            strategy_factory,
            param_grid,
            score_fn=score_fn,
            initial_equity=initial_equity,
            risk_per_trade=risk_per_trade,
        )
        best = ranked[0]

        # Evaluate the winning params on the *following* window only --
        # this is the out-of-sample number that actually means something.
        test_strategy_fn = strategy_factory(best.params)
        test_bt = Backtester(test_slice, test_strategy_fn, initial_equity=initial_equity, risk_per_trade=risk_per_trade)
        test_result = test_bt.run()
        test_score = score_fn(test_result)

        windows.append(
            WalkForwardWindow(
                train_start=train_slice.index[0],
                train_end=train_slice.index[-1],
                test_start=test_slice.index[0],
                test_end=test_slice.index[-1],
                best_params=best.params,
                train_score=best.score,
                test_result=test_result,
                test_score=test_score,
            )
        )

        start += test_size

    return windows
