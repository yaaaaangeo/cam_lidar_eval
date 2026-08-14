# Cam–LiDAR Calibration Evaluation Tool

GT-free quality evaluation for an **existing** camera–LiDAR extrinsic
calibration (`T_CL`). This tool does not compute a new calibration — it
tells you how much to trust one you already have, without needing a ground
truth transform.

> Full design rationale lives in `evaluation_metric_spec.md`. This README
> is the practical "how do I run this / how is it built" companion.

---

## 1. Why GT-free?

In production you almost never have a ground-truth `T_CL` to compare
against. What you *can* measure without one:

- Does the calibration project LiDAR structure onto matching image edges *right now*? (**M2**)
- Does it hold up across different time windows, or was it only ever good in one scene? (**M3**)
- Does its accuracy stay stable frame-to-frame, or does it spike unpredictably? (**M4**)
- Is the projection even structurally sane before trusting M2–M4 at all? (**M0**)

Everything here answers "is this T consistent and stable," never "is this T
correct to N mm/degrees" — that question needs a ground truth, which is out
of scope by design.

---

## 2. Install

```bash
git clone https://github.com/YOUR_ORG/cam-lidar-eval.git
cd cam-lidar-eval
pip install -e .
```

This installs `numpy`, `opencv-python`, `scipy`, `matplotlib`, and `PyYAML`,
and adds a `cam-lidar-eval` console command. No GPU, no ROS, no external
services required.

Prefer plain pip without an editable install? `pip install -r requirements.txt`
works too (then run via `python -m app.cli ...` from the repo root instead
of the `cam-lidar-eval` command).

---

## 3. Quickstart

```bash
# no data required — runs a built-in synthetic scene
cam-lidar-eval --demo --output-dir out/

# a scenario where calibration degrades partway through the sequence
cam-lidar-eval --demo --scenario drift --output-dir out/

# open out/report.html in a browser
```

(Not installed via pip? Run `python -m app.cli ...` from the repo root instead.)

Against real data:

```bash
cam-lidar-eval --config my_config.yaml --output-dir out/
```

See `app/cli.py`'s module docstring for the full YAML config schema
(camera intrinsics/distortion, LiDAR sensor spec, extrinsic, evaluation
parameters). A minimal example:

```yaml
camera:
  image_dir: /data/images
  width: 1920
  height: 1080
  model: pinhole
  intrinsics: {fx: 1400.0, fy: 1400.0, cx: 960.0, cy: 540.0}
  distortion: {model: plumb_bob, coeffs: {k1: -0.1, k2: 0.02, p1: 0.0, p2: 0.0}}

lidar:
  pcd_dir: /data/pointclouds
  sensor_spec:
    horizontal_resolution_deg: 0.2
    channels: 32
    vertical_fov_deg: 40.0
    range_accuracy_m: 0.02

extrinsic:
  parent: lidar
  child: camera
  translation: [0.12, -0.04, 0.15]
  rotation: [0.5, -1.2, 90.1]
  rotation_format: rpy_deg

evaluation:
  n_blocks: 4
  min_frames_per_block: 30
  min_frames_m4: 30
  depth_jump_threshold_m: 0.3
  edge_radius_px: 3.0
```

Useful flags: `--fail-on-bad` (nonzero exit if overall quality is BAD/FAIL —
for CI), `--advanced` (also runs Phase-5 diagnostics), `--no-visuals`
(skip image generation, much faster), `--weights geometry=0.5,...`
(override category weighting), `--frame-index N` (pick which frame M2's
headline number comes from).

---

## 4. Architecture

```
              Existing System
                    │
        ┌───────────┴───────────┐
      Camera                  LiDAR
        │                       │
        └──────────┬────────────┘
                    ▼
             Existing T_CL
                    │
                    ▼
      ┌──────────────────────────┐
      │ M0 Sanity Gate (gate,    │   evaluation/sanity_gate.py
      │ not a score)             │
      └──────────────┬───────────┘
                      ▼
   ┌─────────────────────────────────────┐
   │            Evaluation Engine          │
   │  M2 Geometry │ M3 Generalization │   │  evaluation/edge_alignment.py
   │              │ M4 Stability      │   │  evaluation/holdout_consistency.py
   └───────┬──────┴──────────┬────────┘   │  evaluation/multiframe_consistency.py
           ▼                 ▼
   quality/noise_floor.py → floor(Z), sensor-relative thresholds
   quality/normalization.py → 0-100 score curve, anchored to floor(Z) multipliers
   quality/quality_score.py → Geometry/Generalization/Stability → Overall Quality
                      │
                      ▼
        visualization/  (overlay, trajectory, histogram — embedded as PNG)
                      │
                      ▼
             report/ (JSON + self-contained HTML)
                      │
                      ▼
                 app/cli.py  (ties everything together)

   (optional, --advanced)
   evaluation/plane_consistency.py   — dominant-plane boundary alignment
   evaluation/perturbation.py        — is T_CL near a local error minimum?
   evaluation/temporal_drift.py      — does error trend over the sequence?
```

### Directory map

```
cam_lidar_eval/
├── input/                    Parse raw camera/LiDAR/extrinsic data into standard models
│   ├── camera.py               CameraModel, image_dir loader
│   ├── lidar.py                LidarModel, PCD/PLY loaders (hand-written parsers)
│   ├── extrinsic.py            T_CL/T_LC normalization + sanity check (rotation validity, units)
│   └── dataset.py              Camera↔LiDAR timestamp sync, time_blocks() for M3
│
├── geometry/                  Pure math, no I/O
│   ├── transform.py             SE(3): rpy/quaternion→matrix, compose, invert, transform_points
│   └── projection.py            Pinhole/fisheye projection, LiDAR→image projection pipeline
│
├── evaluation/                 The metrics themselves
│   ├── sanity_gate.py           M0 — pass/fail gate, not scored
│   ├── edge_alignment.py        M2 — LiDAR edge points vs image edges (distance transform)
│   ├── holdout_consistency.py   M3 — same T across contiguous time blocks
│   ├── multiframe_consistency.py M4 — per-frame error stability, outlier detection
│   ├── plane_consistency.py     [advanced] dominant-plane boundary alignment
│   ├── perturbation.py          [advanced] local-minimum check via small T_CL nudges
│   └── temporal_drift.py        [advanced] linear-trend test on M4's per-frame sequence
│
├── quality/                    Turning px measurements into judgments
│   ├── noise_floor.py           floor(Z): sensor-relative "best possible" px uncertainty
│   ├── normalization.py         floor(Z)-anchored 0-100 score curve
│   └── quality_score.py         Geometry/Generalization/Stability → Overall Quality
│
├── visualization/               PNG generation for the HTML report
│   ├── overlay.py                Projected LiDAR points on the image, GOOD/WARNING/BAD colored
│   ├── trajectory.py             M4 per-frame error line chart (matplotlib)
│   └── histogram.py              M2 per-point error distribution (matplotlib)
│
├── report/                     JSON + HTML report generation
│   ├── builder.py                Assembles one plain-dict report structure from all results
│   ├── json.py                   Strict NaN-safe JSON serialization
│   └── html.py                   Self-contained dark-theme HTML report (base64-embedded images)
│
├── app/
│   └── cli.py                   Entry point: config/demo → pipeline → report → console summary
│
├── tests/                      249 tests across 20 files (see §7)
│
├── pyproject.toml               Package metadata, dependencies, `cam-lidar-eval` console script
├── requirements.txt             Plain-pip alternative to `pip install -e .`
├── run_tests.sh                 Runs the full test suite, CI-friendly exit code
├── .github/workflows/ci.yml     GitHub Actions: install + test + CLI smoke test on Python 3.10-3.12
└── LICENSE                      MIT
```

---

## 5. Metric reference

| | What it measures | Scored? | Category |
|---|---|---|---|
| **M0** Sanity Gate | Is the T_CL + data combination even structurally sane (FOV coverage, depth sanity, occlusion plausibility)? | No — pass/fail gate | Data quality |
| **M2** Edge Alignment | Does projected LiDAR structure land on real image edges, right now? | Yes | Geometry |
| **M3** Hold-out Consistency | Does the same T_CL perform consistently across different contiguous time windows? | Yes | Generalization |
| **M4** Multi-frame Consistency | Does per-frame error stay stable, or do specific frames spike? | Yes | Stability |
| **Plane Consistency** *(advanced)* | Does the dominant flat surface's outline line up with its image silhouette? | No | Geometry (supplementary) |
| **Perturbation Sensitivity** *(advanced)* | Is T_CL near a local minimum of the error surface, or would a small nudge do better? | No | Sensitivity |
| **Temporal Drift** *(advanced)* | Does error trend up/down over the sequence (vs just being noisy)? | No | Stability (supplementary) |

### The sensor-relative noise floor: `floor(Z)`

Every threshold in this tool (GOOD/WARNING/BAD boundaries, the 0-100 score
curve) is expressed as a multiple of `floor(Z)` — the theoretical best-case
px uncertainty for a *specific* camera+LiDAR combination at a given depth,
not a fixed absolute pixel count:

```
floor(Z) = sqrt(floor_angular² + floor_range(Z)² + floor_edge²)

floor_angular       = fx · θ_res                    (LiDAR angular resolution)
floor_range(Z)      = fx · baseline · σ_r / Z²       (LiDAR range noise, falls with distance²)
floor_edge          ≈ 0.5 px                          (edge-detector sub-pixel limit)
```

This means a 16-channel LiDAR paired with a 4K camera and a 128-channel
LiDAR paired with a VGA camera get *different, appropriately scaled*
thresholds instead of one-size-fits-all pixel cutoffs. See
`quality/noise_floor.py` for the full derivation and fallback rules when
sensor specs aren't fully known.

### 0-100 scoring

Scores use a curve anchored exactly at the classification boundaries, so a
score and a classification can never contradict each other:

```
score(r) = 100 / (1 + (r / warning_mult)^p),   r = value_px / floor_px

r = 0            → 100
r = good_mult     → 80   (GOOD/WARNING boundary)
r = warning_mult  → 50   (WARNING/BAD boundary)
r → ∞             → 0
```

Verified by property-based tests (`tests/test_normalization.py`) across
500 random ratios per multiplier scheme.

### Quality Score aggregation

Geometry (M2) / Generalization (M3) / Stability (M4) are weighted equally
(1/3 each) by default — there's no data-driven basis yet to weight one
higher, and the weights are a parameter (`--weights` on the CLI), not a
hardcoded constant. If a category's metric FAILed outright, it's excluded
and the remaining weights are renormalized (never silently scored as 0 —
"couldn't measure" and "measured as terrible" are different things).

---

## 6. Report output

Every run produces:

- **`report.json`** — machine-readable, strictly valid JSON (NaN/Inf
  sanitized to `null`, `allow_nan=False` enforced as a backstop). Meant for
  CI pipelines, dashboards, or further tooling.
- **`report.html`** — a single self-contained file (images embedded as
  base64, so nothing else needs to ship alongside it). Dark
  "instrument-panel" theme: an overall-quality gauge (pure CSS
  `conic-gradient`, no JS), GOOD/WARNING/BAD/FAIL badges used consistently
  everywhere, per-metric detail tables, and (unless `--no-visuals`) the
  actual LiDAR-on-image overlay, an M4 error trajectory chart, and an M2
  error histogram.

---

## 7. Testing

```bash
./run_tests.sh                            # run every test file, aggregate pass/fail
python3 tests/test_noise_floor.py         # or run any single file directly
```

`run_tests.sh` exits non-zero if anything failed, and is exactly what
`.github/workflows/ci.yml` runs on every push/PR (Python 3.10-3.12).

No pytest dependency required — every test file has a built-in runner
(`if __name__ == "__main__":` at the bottom) that prints PASS/FAIL per test
and exits non-zero on failure, so `python3 tests/test_X.py` works standalone
or under CI.

**249 tests across 20 files**, all passing:

| File | Tests | Covers |
|---|---|---|
| `test_transform.py` | 18 | SE(3) math |
| `test_projection.py` | 13 | Pinhole/fisheye projection |
| `test_camera.py` | 8 | Camera loader |
| `test_lidar.py` | 13 | PCD/PLY parsers, LiDAR loader |
| `test_extrinsic.py` | 11 | Rotation formats, T_CL/T_LC direction handling |
| `test_dataset.py` | 9 | Timestamp sync, time_blocks() |
| `test_noise_floor.py` | 21 | floor(Z) derivation and fallback rules |
| `test_normalization.py` | 21 | Score curve, incl. 500-sample property test |
| `test_edge_alignment.py` | 15 | M2, incl. synthetic depth-step scene |
| `test_holdout_consistency.py` | 8 | M3, incl. drift-detection scenario |
| `test_multiframe_consistency.py` | 9 | M4, incl. outlier detection |
| `test_sanity_gate.py` | 10 | M0, incl. occlusion-violation detection |
| `test_plane_consistency.py` | 10 | Plane fitting + boundary alignment |
| `test_perturbation.py` | 7 | Local-minimum detection |
| `test_temporal_drift.py` | 9 | Trend regression, significance gating |
| `test_quality_score.py` | 12 | Category aggregation, weight handling |
| `test_report.py` | 20 | JSON/HTML generation, NaN safety, visuals embedding |
| `test_visualization.py` | 13 | Overlay/trajectory/histogram rendering |
| `test_m0_report_integration.py` | 2 | M0 → report end-to-end |
| `test_cli.py` | 20 | Demo mode, config loading, full pipeline, exit codes |

Every MVP metric (M2/M3/M4) is validated against a **synthetic scene with
a known, controllable ground truth** (a depth step positioned to exactly
match a drawn image edge under a known `T_CL`), so tests assert the actual
numbers move the *correct direction* as calibration is perturbed — not just
that the code runs without crashing.

Several real bugs were caught this way during development and are called
out in code comments where they were fixed: a `floor(Z)` worst-case-axis
selection bug, a perspective row-shift artifact in an early synthetic
scene, and a grid-quantization sensitivity artifact in the perturbation
test scene.

---

## 8. Known limitations / not implemented

- **rosbag / ROS topic sources** are stubbed (`NotImplementedError`) —
  no ROS deserialization dependency in this environment. `image_dir` /
  `pcd_dir` sources are fully implemented.
- **Re-calibration repeatability** (spec §13's "Level 2": re-run
  calibration on subsets, compare resulting T's) is explicitly out of
  scope — this tool evaluates an *existing* calibration, it doesn't
  compute new ones.
- **Photometric consistency** was deferred per the original design notes
  (illumination/exposure/reflectance confound calibration quality too
  easily for a first pass).
- **GT mode** (accuracy against a real ground-truth transform, for
  research/benchmark use) is a distinct mode described in the spec but not
  built here — this tool is GT-free mode only.
- Advanced metrics (`--advanced`) are diagnostic and **do not affect
  quality_score** — by design, since they're less battle-tested than the
  MVP set and answer different questions (local optimality, trend, a
  single-surface check) rather than contributing to the headline score.
