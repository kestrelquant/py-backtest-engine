# Contributing

## Setup

```bash
pip install -r requirements-examples.txt pytest
python -m pytest tests/
```

## The one rule that matters here: no lookahead

`Backtester.run()` calls `strategy_fn(i, history)` with `history` sliced
to `self.data.iloc[:i+1]` — never more. Every PR that touches `engine.py`
should keep `test_no_lookahead_entry_fills_next_bar_open` passing, and if
you add a new fill path (a limit order type, a same-bar exit), add a test
in the same style: assert on `len(history) == i + 1` inside the strategy
function, don't just trust the slicing.

## Adding a metric

Add a function to `backtester/metrics.py` taking a `BacktestResult` and
returning a float, then add it to `summary()`. Test it against a
hand-constructed `BacktestResult` with a known answer — not just "runs
without crashing."

## Adding an optimizer feature

`grid_search` and `walk_forward` in `backtester/optimizer.py` both take a
`strategy_factory: dict -> strategy_fn`. Keep that signature — it's what
lets both functions work with any strategy without knowing its
parameters ahead of time.

## Regenerating the example fixture

```bash
python examples/fixtures/generate_fixture.py
```

`test_csv_fixture.py` only checks structure (required columns, one
deterministic trade), not exact backtest numbers, so regenerating the
fixture won't break the test suite — but it does change the printed
output of `examples/run_from_config.py`, so mention it in the PR if
you regenerate it.
