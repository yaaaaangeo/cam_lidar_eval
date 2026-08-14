# Changelog

All notable changes to this project are documented here.
This project follows [Semantic Versioning](https://semver.org/).

## [0.1.0] — Initial release

MVP + Phase 5 advanced diagnostics, per `evaluation_metric_spec.md`.

### Added

- **Input loaders** (`input/`): camera (image_dir), LiDAR (PCD ASCII/binary,
  PLY ASCII), extrinsic (rpy/quaternion/matrix, automatic T_CL/T_LC
  direction handling), timestamp-synced dataset construction.
- **Geometry** (`geometry/`): SE(3) transform utilities, pinhole/fisheye
  projection.
- **M0 Sanity Gate**: FOV coverage, depth-distribution, and occlusion
  plausibility checks — not scored, gates whether M2-M4 are meaningful.
- **MVP scored metrics**:
  - **M2 Edge Alignment** — projected LiDAR depth-discontinuity points vs
    image edges (Canny + distance transform).
  - **M3 Hold-out Consistency** — same T_CL evaluated across contiguous
    time blocks.
  - **M4 Multi-frame Consistency** — per-frame error stability and outlier
    detection.
- **Sensor-relative noise floor** (`quality/noise_floor.py`): `floor(Z)`
  derived from LiDAR angular resolution, range noise, and edge-detector
  limits — all GOOD/WARNING/BAD thresholds scale with sensor capability
  instead of using fixed pixel cutoffs.
- **0-100 scoring** (`quality/normalization.py`): score curve anchored
  exactly at classification boundaries.
- **Quality Score aggregation** (`quality/quality_score.py`): Geometry /
  Generalization / Stability → Overall Quality, equal weights by default,
  graceful exclusion of FAILed categories.
- **Visualization** (`visualization/`): LiDAR-on-image overlay, M4 error
  trajectory chart, M2 error histogram.
- **Reports** (`report/`): strict NaN-safe JSON, self-contained dark-theme
  HTML with embedded visuals.
- **CLI** (`app/cli.py`): `--demo` (synthetic scenes) and `--config`
  (real data via YAML) entry points, `--fail-on-bad` for CI use.
- **Phase 5 advanced diagnostics** (opt-in via `--advanced`, never affect
  quality_score):
  - **Plane Consistency** — dominant-plane boundary alignment via RANSAC.
  - **Perturbation Sensitivity** — is T_CL near a local error minimum?
  - **Temporal Drift** — statistically-gated linear trend test on M4's
    per-frame sequence.
- 249 tests across 20 files, each independently runnable
  (`python3 tests/test_X.py`, no pytest required).
- GitHub Actions CI (`.github/workflows/ci.yml`), Python 3.10-3.12.

### Known limitations

- rosbag / ROS topic sources are stubbed (no ROS deserialization
  dependency in this environment).
- Re-calibration repeatability, photometric consistency, and GT
  (ground-truth) mode are explicitly out of scope for this release — see
  README §8.
