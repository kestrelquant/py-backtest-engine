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
