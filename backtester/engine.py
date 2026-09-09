"""A small event-driven backtest engine for single-instrument OHLCV strategies.

Deliberately not vectorized: fills happen bar-by-bar against the *next*
bar's open (no lookahead), position sizing is risk-based, and every fill
is logged. That makes it slower than a pure vectorized backtest but means
the numbers it produces are the numbers a strategy could actually have
gotten -- vectorized backtests are easy to accidentally let peek at the
same bar's close it signaled on.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class Trade:
    entry_time: pd.Timestamp
    entry_price: float
    direction: int  # 1 = long, -1 = short
    size: float
    stop_price: float
    exit_time: pd.Timestamp | None = None
    exit_price: float | None = None

    @property
    def is_open(self) -> bool:
        return self.exit_time is None

    def pnl(self, price: float) -> float:
        return (price - self.entry_price) * self.direction * self.size


@dataclass
class BacktestResult:
    equity_curve: pd.Series
    trades: list[Trade] = field(default_factory=list)

    @property
    def closed_trades(self) -> list[Trade]:
        return [t for t in self.trades if not t.is_open]


class Backtester:
    """
    `strategy_fn(bar_index, history_df) -> Signal | None` is called once
    per closed bar. `history_df` only contains bars up to and including
    the current one, so a strategy cannot see the future by construction.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        strategy_fn,
        initial_equity: float = 10_000.0,
        risk_per_trade: float = 0.01,
    ):
        required = {"open", "high", "low", "close"}
        missing = required - set(data.columns)
        if missing:
            raise ValueError(f"data is missing required columns: {missing}")

        self.data = data
        self.strategy_fn = strategy_fn
        self.initial_equity = initial_equity
        self.risk_per_trade = risk_per_trade

    def run(self) -> BacktestResult:
        equity = self.initial_equity
        equity_curve = pd.Series(index=self.data.index, dtype=float)
        trades: list[Trade] = []
        open_trade: Trade | None = None

        for i in range(len(self.data)):
            row = self.data.iloc[i]

            if open_trade is not None:
                hit_stop = (
                    row["low"] <= open_trade.stop_price
                    if open_trade.direction == 1
                    else row["high"] >= open_trade.stop_price
                )
                if hit_stop:
                    open_trade.exit_time = self.data.index[i]
                    open_trade.exit_price = open_trade.stop_price
                    equity += open_trade.pnl(open_trade.stop_price)
                    open_trade = None

            equity_curve.iloc[i] = equity + (
                open_trade.pnl(row["close"]) if open_trade is not None else 0.0
            )

            if i + 1 < len(self.data):
                signal = self.strategy_fn(i, self.data.iloc[: i + 1])
                next_open = self.data.iloc[i + 1]["open"]

                # A signal opposite to the open position closes it at the
                # next bar's open before (optionally) opening the new one --
                # without this, only stop-losses ever closed a trade, which
                # silently excluded every winner from closed_trades/win_rate.
                if signal is not None and open_trade is not None and signal.direction != open_trade.direction:
                    open_trade.exit_time = self.data.index[i + 1]
                    open_trade.exit_price = next_open
                    equity += open_trade.pnl(next_open)
                    open_trade = None

                if signal is not None and open_trade is None:
                    stop_distance = abs(next_open - signal.stop_price)
                    if stop_distance > 0:
                        risk_money = equity * self.risk_per_trade
                        size = risk_money / stop_distance
                        open_trade = Trade(
                            entry_time=self.data.index[i + 1],
                            entry_price=next_open,
                            direction=signal.direction,
                            size=size,
                            stop_price=signal.stop_price,
                        )
                        trades.append(open_trade)

        if open_trade is not None:
            last_close = self.data.iloc[-1]["close"]
            open_trade.exit_time = self.data.index[-1]
            open_trade.exit_price = last_close
            equity += open_trade.pnl(last_close)

        return BacktestResult(equity_curve=equity_curve.ffill(), trades=trades)


@dataclass
class Signal:
    direction: int  # 1 = long, -1 = short
    stop_price: float
