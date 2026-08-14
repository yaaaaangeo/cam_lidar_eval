# Contributing

## Setup

```bash
git clone https://github.com/YOUR_ORG/cam-lidar-eval.git
cd cam-lidar-eval
pip install -e .
```

## Running tests

```bash
./run_tests.sh                       # everything
python3 tests/test_edge_alignment.py # a single file
```

Every test file works standalone (no pytest required), but `pytest tests/`
also works if you prefer it.

## Code layout

See the README's "Architecture" section (§4) for the directory map. A few
conventions worth knowing before contributing:

- **Modules are flat top-level packages** (`input`, `geometry`,
  `evaluation`, `quality`, `visualization`, `report`, `app`), not nested
  under a single namespace. Keep new modules consistent with this — import
  as `from evaluation.edge_alignment import ...`, not relative imports.
- **Metrics return dataclasses, not plain dicts.** Serialization to
  JSON/HTML happens exclusively in `report/builder.py` — that's the one
  place that decides what's report-worthy and sanitizes NaN/Inf. Don't
  hand-serialize a metric result elsewhere.
- **Every threshold is sensor-relative**, derived from `quality.noise_floor`
  multipliers — avoid introducing new hardcoded absolute-pixel thresholds.
- **New metrics should validate against a synthetic scene with a known
  ground truth**, the same way M2/M3/M4 do (see `tests/test_edge_alignment.py`'s
  `_make_synthetic_scene`): assert the metric moves in the *correct
  direction* as you perturb calibration, not just that it runs without
  crashing.
- **Advanced (Phase-5) metrics must never affect `quality_score`.** They're
  opt-in diagnostics (`--advanced`), not part of the MVP scored set.

## Adding a new metric

1. Implement it in `evaluation/your_metric.py`, returning a dataclass with
   at least a `classification` field (`"GOOD" | "WARNING" | "BAD" | "FAIL"`).
2. Add a synthetic-scene test in `tests/test_your_metric.py`.
3. If it's MVP-scored, wire it into `quality/quality_score.py`. If it's
   advanced/supplementary, add a `*_summary()` function in
   `report/builder.py` and wire it into `report/html.py`'s advanced section
   instead (see `plane_consistency_summary` for the pattern).
4. Wire it into `app/cli.py` if it should be runnable from the CLI.
