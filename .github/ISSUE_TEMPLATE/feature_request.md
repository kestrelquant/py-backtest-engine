---
name: Feature request
about: A metric, fill model, or optimization mode that's missing
title: ""
labels: enhancement
---

**What's missing**


**Where it belongs**
- [ ] `backtester/engine.py` — fill/position logic
- [ ] `backtester/metrics.py` — a new summary statistic
- [ ] `backtester/optimizer.py` — grid search / walk-forward

**No-lookahead check**
If this touches fill timing or the data a strategy can see, say
explicitly what a strategy is and isn't allowed to see at bar `i` under
the proposed change — this engine's whole value is not leaking the
future, so this needs to be explicit, not assumed.
