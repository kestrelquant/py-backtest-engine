"""Regenerates sample_ohlcv.csv. Not part of the test suite -- run by
hand if the fixture needs to change:

    python examples/fixtures/generate_fixture.py

The fixture is synthetic (geometric random walk), not real market data --
see examples/README.md for why.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # examples/
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root

from ema_cross_strategy import make_synthetic_ohlcv

if __name__ == "__main__":
    data = make_synthetic_ohlcv(n=500, seed=42)
    out = Path(__file__).parent / "sample_ohlcv.csv"
    data.to_csv(out, index_label="time")
    print(f"wrote {len(data)} rows to {out}")
