# py-backtest-engine

A small event-driven backtest engine for single-instrument OHLCV
strategies: bar-by-bar simulation, fills on the next bar's open, risk-based
position sizing, and a stop-loss check before a strategy even gets a
chance to signal on that bar.

## Why event-driven instead of vectorized

A vectorized backtest computes signals across the whole series at once,
which makes it easy to accidentally let a signal use information from the
same bar's close it's supposed to act on. This engine calls
`strategy_fn(bar_index, history_df)` once per bar with `history_df`
truncated to that bar — the strategy physically cannot see future rows.
Slower, but the numbers it produces are numbers a strategy could actually
have gotten live.

## Usage

```python
from backtester import Backtester, Signal, metrics

def my_strategy(i, history):
    if some_condition(history):
        return Signal(direction=1, stop_price=history["close"].iloc[-1] - 1.0)
    return None

bt = Backtester(ohlcv_df, my_strategy, initial_equity=10_000, risk_per_trade=0.01)
result = bt.run()
print(metrics.summary(result))
```

See [examples/ema_cross_strategy.py](examples/ema_cross_strategy.py) for a
full runnable example on synthetic data.

## Parameter search and walk-forward analysis

`grid_search` ranks parameter combinations by a score function (Sharpe by
default) — useful, and also exactly how you overfit a strategy to one
sample if that's all you do with it.

`walk_forward` is the check on that: it re-runs `grid_search` on a
rolling training window, then evaluates only the winning parameters on
the *following* window, which the search never saw. It slides forward by
`test_size` and repeats. The in-sample and out-of-sample scores routinely
disagree — that disagreement is the point, not a bug.

```python
from backtester import grid_search, walk_forward

ranked = grid_search(data, strategy_factory, param_grid={"fast": [8, 12], "slow": [26, 40]})
best = ranked[0]  # highest score on the whole series -- likely optimistic

windows = walk_forward(data, strategy_factory, param_grid={"fast": [8, 12], "slow": [26, 40]},
                        train_size=500, test_size=250)
for w in windows:
    print(w.best_params, w.test_score)  # score on data the search never touched
```

`strategy_factory(params: dict) -> strategy_fn` adapts a strategy's
keyword arguments to what both functions expect — see
`ema_cross_strategy_factory` in the example.

## Metrics

`backtester.metrics.summary()` returns trade count, total return, max
drawdown, Sharpe ratio, and win rate.

## Tests

```bash
pip install -r requirements.txt pytest
python -m pytest tests/
```

## License

MIT — see [LICENSE](LICENSE).
