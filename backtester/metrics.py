from __future__ import annotations

import numpy as np
import pandas as pd

from .engine import BacktestResult


def total_return(result: BacktestResult) -> float:
    curve = result.equity_curve.dropna()
    if curve.empty:
        return 0.0
    return curve.iloc[-1] / curve.iloc[0] - 1.0


def max_drawdown(result: BacktestResult) -> float:
    curve = result.equity_curve.dropna()
    if curve.empty:
        return 0.0
    running_max = curve.cummax()
    drawdown = curve / running_max - 1.0
    return drawdown.min()


def sharpe_ratio(result: BacktestResult, periods_per_year: int = 252) -> float:
    curve = result.equity_curve.dropna()
    returns = curve.pct_change().dropna()
    if returns.std() == 0 or returns.empty:
        return 0.0
    return float(np.sqrt(periods_per_year) * returns.mean() / returns.std())


def win_rate(result: BacktestResult) -> float:
    closed = result.closed_trades
    if not closed:
        return 0.0
    wins = sum(1 for t in closed if t.pnl(t.exit_price) > 0)
    return wins / len(closed)


def summary(result: BacktestResult) -> dict:
    return {
        "trades": len(result.closed_trades),
        "total_return": total_return(result),
        "max_drawdown": max_drawdown(result),
        "sharpe_ratio": sharpe_ratio(result),
        "win_rate": win_rate(result),
    }
