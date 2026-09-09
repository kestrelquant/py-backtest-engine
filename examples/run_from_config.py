"""Run a backtest from a YAML config instead of hardcoded parameters.

    python examples/run_from_config.py config.yaml

This is the shape a real usage of this engine takes: point it at a CSV of
OHLCV data and a config describing which strategy and parameters to run,
rather than editing a script's constants by hand for every run.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backtester import Backtester, metrics, walk_forward
from ema_cross_strategy import ema_cross_strategy, ema_cross_strategy_factory


def load_ohlcv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["time"], index_col="time")
    required = {"open", "high", "low", "close"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")
    return df


def main(config_path: str) -> None:
    config_file = Path(config_path)
    config = yaml.safe_load(config_file.read_text())

    data_path = config_file.parent / config["data"]["path"]
    data = load_ohlcv(data_path)
    print(f"loaded {len(data)} bars from {data_path}")

    strategy_params = {k: v for k, v in config["strategy"].items() if k != "name"}
    strategy_fn = ema_cross_strategy(**strategy_params)

    bt = Backtester(
        data,
        strategy_fn,
        initial_equity=config["backtest"]["initial_equity"],
        risk_per_trade=config["backtest"]["risk_per_trade"],
    )
    result = bt.run()

    print("\n-- single run --")
    for k, v in metrics.summary(result).items():
        print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")

    opt = config.get("optimize", {})
    if opt.get("enabled"):
        print("\n-- walk-forward (out-of-sample) --")
        windows = walk_forward(
            data,
            ema_cross_strategy_factory,
            param_grid=opt["param_grid"],
            train_size=opt["train_size"],
            test_size=opt["test_size"],
        )
        for w in windows:
            print(f"train [{w.train_start.date()}..{w.train_end.date()}] best={w.best_params} "
                  f"-> test oos_sharpe={w.test_score:.4f}")


if __name__ == "__main__":
    config_arg = sys.argv[1] if len(sys.argv) > 1 else str(Path(__file__).parent / "config.yaml")
    main(config_arg)
