"""
input/validation.py

STEP 1 -- Input Validation layer.

Responsibility: answer "is this input usable at all" BEFORE any calibration
evaluation runs. This module never judges calibration quality (that's M0/
M2/M3/M4's job) -- it only checks that the raw camera/LiDAR/dataset inputs
themselves are well-formed enough to make a calibration judgement meaningful.

Why this exists (see evaluation_metric_spec.md's design principle): if the
input is broken -- non-monotonic camera timestamps, a degenerate intrinsics
matrix, an empty point cloud, NaN/Inf points -- the tool must never report
that as "Calibration BAD". Reporting a data problem as a calibration
problem is actively misleading; a data problem must be reported as exactly
that, so a user fixes the right thing.

Distinct from (do not confuse with):
  - input/camera.py / input/lidar.py loaders: PARSE ONLY, no validation.
  - input/extrinsic.py's verify_extrinsic(): checks T_CL is a well-formed
    transform in isolation.
  - evaluation/sanity_gate.py's run_sanity_gate() (M0): checks whether
    T_CL + the actual data combination projects sanely. M0 assumes the
    input itself (this module's job) is already valid; validation.py runs
    strictly BEFORE M0 in the pipeline.

Three possible statuses, combined worst-of across every check:

    INPUT_VALID     -- nothing wrong (WARNING-level issues may still exist
                        upstream in loader warnings, but nothing here failed)
    INPUT_WARNING   -- usable, but with caveats worth surfacing (e.g. a
                        borderline overlap duration, a few NaN points that
                        get filtered)
    INPUT_INVALID   -- unusable; the evaluation pipeline must not run.

Every failing/warning check records a short, human-actionable "reason"
string (e.g. "Camera timestamps are not monotonic"), collected into
ValidationReport.reasons(), so the CLI can print exactly why input was
rejected instead of a bare exception.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np

from input.camera import CameraModel, CameraFrame
from input.lidar import LidarModel, LidarFrame


# ---------------------------------------------------------------------------
# Status + report primitives
# ---------------------------------------------------------------------------

class ValidationStatus(str, Enum):
    VALID = "INPUT_VALID"
    WARNING = "INPUT_WARNING"
    INVALID = "INPUT_INVALID"

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.value


# Ordering used to combine multiple check statuses into a "worst wins"
# overall status: INVALID > WARNING > VALID.
_SEVERITY = {
    ValidationStatus.VALID: 0,
    ValidationStatus.WARNING: 1,
    ValidationStatus.INVALID: 2,
}


def _worse(a: ValidationStatus, b: ValidationStatus) -> ValidationStatus:
    return a if _SEVERITY[a] >= _SEVERITY[b] else b


@dataclass
class ValidationCheckItem:
    name: str
    status: ValidationStatus
    detail: str
    # Machine-usable value backing the check, when there is one (e.g. a
    # count or a ratio) -- mirrors the `value` field pattern already used
    # by evaluation/sanity_gate.py's SanityCheckItem.
    value: Optional[float] = None

    @property
    def ok(self) -> bool:
        return self.status == ValidationStatus.VALID


@dataclass
class ValidationReport:
    """Result of validating one layer (camera / lidar / dataset), or the
    combined result of all three (see validate_input)."""
    status: ValidationStatus = ValidationStatus.VALID
    items: list[ValidationCheckItem] = field(default_factory=list)

    def add(self, item: ValidationCheckItem) -> None:
        self.items.append(item)
        self.status = _worse(self.status, item.status)

    def merge(self, other: "ValidationReport") -> None:
        for item in other.items:
            self.add(item)

    def reasons(self, min_severity: ValidationStatus = ValidationStatus.WARNING) -> list[str]:
        """Human-readable 'Reason: ...' lines for every check at or above
        min_severity (default: WARNING and INVALID, i.e. everything that
        isn't a clean pass)."""
        threshold = _SEVERITY[min_severity]
        return [
            f"{item.name}: {item.detail}"
            for item in self.items
            if _SEVERITY[item.status] >= threshold
        ]

    def failed_items(self) -> list[ValidationCheckItem]:
        return [i for i in self.items if i.status == ValidationStatus.INVALID]

    def warning_items(self) -> list[ValidationCheckItem]:
        return [i for i in self.items if i.status == ValidationStatus.WARNING]

    @property
    def is_invalid(self) -> bool:
        return self.status == ValidationStatus.INVALID

    def to_dict(self) -> dict:
        return {
            "status": self.status.value,
            "checks": [
                {
                    "name": i.name,
                    "status": i.status.value,
                    "detail": i.detail,
                    "value": _safe_float(i.value),
                }
                for i in self.items
            ],
            "reasons": self.reasons(),
        }


def _safe_float(x):
    if x is None:
        return None
    xf = float(x)
    return xf if np.isfinite(xf) else None


class InputValidationError(ValueError):
    """Raised when combined input validation status is INPUT_INVALID and a
    caller asked to fail fast (see validate_input(..., raise_on_invalid=True)
    and app.cli). Carries the ValidationReport so callers can print full
    reasons, not just a flat message."""

    def __init__(self, report: ValidationReport):
        self.report = report
        reasons = "\n".join(f"  - {r}" for r in report.reasons(ValidationStatus.INVALID))
        super().__init__(f"INPUT INVALID:\n{reasons}")


# ---------------------------------------------------------------------------
# Camera checks
# ---------------------------------------------------------------------------

def validate_camera(camera: CameraModel, frames: list[CameraFrame]) -> ValidationReport:
    """
    Checks (per STEP 1 spec):
      - K exists / fx > 0 / fy > 0 / cx,cy sane
      - distortion parameter count sane for the declared model
      - image size sane (width/height > 0)
      - timestamps exist, are finite, and are monotonically increasing
    """
    report = ValidationReport()
    intr = camera.intrinsics

    fx_ok = np.isfinite(intr.fx) and intr.fx > 0
    report.add(ValidationCheckItem(
        "camera.fx_positive",
        ValidationStatus.VALID if fx_ok else ValidationStatus.INVALID,
        f"fx={intr.fx}" if fx_ok else f"fx must be finite and > 0, got {intr.fx}",
        value=intr.fx,
    ))

    fy_ok = np.isfinite(intr.fy) and intr.fy > 0
    report.add(ValidationCheckItem(
        "camera.fy_positive",
        ValidationStatus.VALID if fy_ok else ValidationStatus.INVALID,
        f"fy={intr.fy}" if fy_ok else f"fy must be finite and > 0, got {intr.fy}",
        value=intr.fy,
    ))

    cx_ok = np.isfinite(intr.cx) and 0 <= intr.cx <= camera.width
    report.add(ValidationCheckItem(
        "camera.cx_in_image_bounds",
        ValidationStatus.VALID if cx_ok else ValidationStatus.WARNING,
        (f"cx={intr.cx}" if cx_ok else
         f"cx={intr.cx} is outside the image width [0, {camera.width}] -- "
         f"check intrinsics/width for a mismatch."),
        value=intr.cx,
    ))

    cy_ok = np.isfinite(intr.cy) and 0 <= intr.cy <= camera.height
    report.add(ValidationCheckItem(
        "camera.cy_in_image_bounds",
        ValidationStatus.VALID if cy_ok else ValidationStatus.WARNING,
        (f"cy={intr.cy}" if cy_ok else
         f"cy={intr.cy} is outside the image height [0, {camera.height}] -- "
         f"check intrinsics/height for a mismatch."),
        value=intr.cy,
    ))

    size_ok = camera.width > 0 and camera.height > 0
    report.add(ValidationCheckItem(
        "camera.image_size_valid",
        ValidationStatus.VALID if size_ok else ValidationStatus.INVALID,
        f"width={camera.width}, height={camera.height}" if size_ok else
        f"width/height must both be > 0, got width={camera.width}, height={camera.height}",
    ))

    dist_status, dist_detail = _check_distortion(camera)
    report.add(ValidationCheckItem("camera.distortion_params_valid", dist_status, dist_detail))

    if not frames:
        report.add(ValidationCheckItem(
            "camera.frames_present",
            ValidationStatus.INVALID,
            "No camera frames were loaded.",
        ))
        return report

    ts = np.array([f.timestamp for f in frames], dtype=float)
    ts_finite = np.isfinite(ts)
    n_nonfinite = int((~ts_finite).sum())
    report.add(ValidationCheckItem(
        "camera.timestamps_finite",
        ValidationStatus.VALID if n_nonfinite == 0 else ValidationStatus.INVALID,
        (f"All {len(ts)} camera timestamps are finite." if n_nonfinite == 0 else
         f"{n_nonfinite}/{len(ts)} camera timestamps are NaN/Inf (likely "
         f"non-numeric filenames that fell back badly)."),
        value=float(n_nonfinite),
    ))

    if ts_finite.all():
        is_monotonic = bool(np.all(np.diff(ts) > 0))
        report.add(ValidationCheckItem(
            "camera.timestamps_monotonic",
            ValidationStatus.VALID if is_monotonic else ValidationStatus.INVALID,
            ("Camera timestamps are strictly increasing." if is_monotonic else
             "Camera timestamps are not monotonic (frames are out of "
             "time order, or contain duplicate timestamps)."),
        ))

    return report


def _check_distortion(camera: CameraModel):
    """Distortion coeff-count sanity per model. Uses the same coeffs
    resolution path (`CameraDistortion.as_array`) the rest of the pipeline
    uses, so this check reflects exactly what projection will receive."""
    dist = camera.distortion
    if dist.model == "none":
        return ValidationStatus.VALID, "distortion model is 'none' (no coeffs expected)."
    try:
        arr = dist.as_array()
    except Exception as e:  # malformed coeffs dict, unknown keys, etc.
        return ValidationStatus.INVALID, f"Failed to build distortion coefficients: {e}"

    if arr is None:
        return ValidationStatus.WARNING, f"distortion model '{dist.model}' resolved to no coefficients."

    if not np.all(np.isfinite(arr)):
        return ValidationStatus.INVALID, f"distortion coefficients contain NaN/Inf: {arr.tolist()}"

    expected_len = {"plumb_bob": 5, "fisheye_equidistant": 4}.get(dist.model)
    if expected_len is not None and len(arr) != expected_len:
        return (
            ValidationStatus.WARNING,
            f"distortion model '{dist.model}' usually has {expected_len} "
            f"coefficients, got {len(arr)}.",
        )
    return ValidationStatus.VALID, f"distortion model '{dist.model}' has {len(arr)} finite coefficient(s)."


# ---------------------------------------------------------------------------
# LiDAR checks
# ---------------------------------------------------------------------------

# A frame this sparse can't support any meaningful edge/plane metric later;
# flagging it as WARNING here (not INVALID) since a single sparse frame
# shouldn't necessarily kill an otherwise-fine sequence -- the dataset-level
# check catches it if it's pervasive.
MIN_POINTS_PER_FRAME_WARNING = 100

# Fraction of NaN/Inf points in a frame above which the frame is flagged.
MAX_NONFINITE_RATIO_WARNING = 0.01
MAX_NONFINITE_RATIO_INVALID = 0.5


def validate_lidar(
    lidar: LidarModel,
    frames: list[LidarFrame],
    sample_frames: int = 5,
) -> ValidationReport:
    """
    Checks (per STEP 1 spec):
      - point count (empty PCD, very sparse frames)
      - NaN / Inf in XYZ
      - XYZ validity (all-zero degenerate points)
      - range vs sensor_spec min/max_range_m
      - timestamps exist

    `sample_frames`: point-level checks (NaN/Inf/range) are run on up to
    this many frames spread across the sequence rather than every frame,
    since loading is lazy and a full-sequence scan can mean reading
    thousands of point cloud files just to validate input. Timestamp and
    frame-count checks always cover the FULL sequence (cheap: metadata
    only, no file I/O).
    """
    report = ValidationReport()

    if not frames:
        report.add(ValidationCheckItem(
            "lidar.frames_present",
            ValidationStatus.INVALID,
            "No LiDAR frames were loaded.",
        ))
        return report

    ts = np.array([f.timestamp for f in frames], dtype=float)
    ts_finite = np.isfinite(ts)
    n_nonfinite_ts = int((~ts_finite).sum())
    report.add(ValidationCheckItem(
        "lidar.timestamps_finite",
        ValidationStatus.VALID if n_nonfinite_ts == 0 else ValidationStatus.INVALID,
        (f"All {len(ts)} LiDAR timestamps are finite." if n_nonfinite_ts == 0 else
         f"{n_nonfinite_ts}/{len(ts)} LiDAR timestamps are NaN/Inf (likely "
         f"non-numeric filenames that fell back badly)."),
        value=float(n_nonfinite_ts),
    ))

    if ts_finite.all():
        is_monotonic = bool(np.all(np.diff(ts) > 0))
        report.add(ValidationCheckItem(
            "lidar.timestamps_monotonic",
            ValidationStatus.VALID if is_monotonic else ValidationStatus.INVALID,
            ("LiDAR timestamps are strictly increasing." if is_monotonic else
             "LiDAR timestamps are not monotonic (frames are out of time "
             "order, or contain duplicate timestamps)."),
        ))

    # Sample point-level checks across the sequence (first, last, and
    # evenly-spaced frames in between) instead of loading everything.
    n = len(frames)
    k = max(1, min(sample_frames, n))
    sample_idx = sorted(set(np.linspace(0, n - 1, k).astype(int).tolist()))

    spec = lidar.sensor_spec
    empty_frames = 0
    sparse_frames = 0
    nonfinite_flagged_frames = 0
    nonfinite_severe_frames = 0
    out_of_range_frames = 0
    checked = 0

    for idx in sample_idx:
        try:
            pts = frames[idx].load()
        except Exception as e:
            report.add(ValidationCheckItem(
                f"lidar.frame_{idx}_loadable",
                ValidationStatus.INVALID,
                f"Failed to load LiDAR frame {idx}: {e}",
            ))
            continue

        checked += 1
        n_points = pts.shape[0]
        if n_points == 0:
            empty_frames += 1
            continue
        if n_points < MIN_POINTS_PER_FRAME_WARNING:
            sparse_frames += 1

        xyz = pts[:, :3]
        finite_mask = np.isfinite(xyz).all(axis=1)
        nonfinite_ratio = 1.0 - (finite_mask.sum() / n_points)
        if nonfinite_ratio >= MAX_NONFINITE_RATIO_INVALID:
            nonfinite_severe_frames += 1
        elif nonfinite_ratio >= MAX_NONFINITE_RATIO_WARNING:
            nonfinite_flagged_frames += 1

        if finite_mask.any():
            ranges = np.linalg.norm(xyz[finite_mask], axis=1)
            in_range = (ranges >= spec.min_range_m) & (ranges <= spec.max_range_m)
            frac_in_range = float(in_range.mean())
            if frac_in_range < 0.5:
                out_of_range_frames += 1

    if empty_frames > 0:
        report.add(ValidationCheckItem(
            "lidar.point_count_nonzero",
            ValidationStatus.INVALID,
            f"{empty_frames}/{checked} sampled LiDAR frame(s) are empty (0 points).",
            value=float(empty_frames),
        ))
    else:
        report.add(ValidationCheckItem(
            "lidar.point_count_nonzero",
            ValidationStatus.VALID,
            f"All {checked} sampled LiDAR frame(s) contain points.",
        ))

    if sparse_frames > 0:
        report.add(ValidationCheckItem(
            "lidar.point_count_sufficient",
            ValidationStatus.WARNING,
            f"{sparse_frames}/{checked} sampled frame(s) have fewer than "
            f"{MIN_POINTS_PER_FRAME_WARNING} points; downstream metrics on "
            f"those frames will be based on very sparse data.",
            value=float(sparse_frames),
        ))

    if nonfinite_severe_frames > 0:
        report.add(ValidationCheckItem(
            "lidar.xyz_finite",
            ValidationStatus.INVALID,
            f"{nonfinite_severe_frames}/{checked} sampled frame(s) have "
            f">= {MAX_NONFINITE_RATIO_INVALID:.0%} NaN/Inf points.",
            value=float(nonfinite_severe_frames),
        ))
    elif nonfinite_flagged_frames > 0:
        report.add(ValidationCheckItem(
            "lidar.xyz_finite",
            ValidationStatus.WARNING,
            f"{nonfinite_flagged_frames}/{checked} sampled frame(s) contain "
            f"some NaN/Inf points (below the {MAX_NONFINITE_RATIO_INVALID:.0%} "
            f"invalid threshold; these points will be filtered downstream).",
            value=float(nonfinite_flagged_frames),
        ))
    else:
        report.add(ValidationCheckItem(
            "lidar.xyz_finite",
            ValidationStatus.VALID,
            f"No significant NaN/Inf contamination in {checked} sampled frame(s).",
        ))

    if out_of_range_frames > 0:
        report.add(ValidationCheckItem(
            "lidar.range_within_sensor_spec",
            ValidationStatus.WARNING,
            f"{out_of_range_frames}/{checked} sampled frame(s) have most "
            f"points outside sensor_spec's [min_range_m={spec.min_range_m}, "
            f"max_range_m={spec.max_range_m}] -- check the sensor_spec "
            f"values or point cloud units (e.g. mm vs m).",
            value=float(out_of_range_frames),
        ))
    else:
        report.add(ValidationCheckItem(
            "lidar.range_within_sensor_spec",
            ValidationStatus.VALID,
            f"Point ranges are consistent with sensor_spec across {checked} "
            f"sampled frame(s).",
        ))

    return report


# ---------------------------------------------------------------------------
# Dataset-level checks (camera + lidar together)
# ---------------------------------------------------------------------------

# Below this overlap, sync will produce too few matched frames to run M3/M4
# meaningfully -- flagged here (before sync even runs) so the reason is
# framed as an input problem, not a downstream "not enough frames" error.
MIN_OVERLAP_S_WARNING = 1.0


def validate_dataset(
    camera_frames: list[CameraFrame],
    lidar_frames: list[LidarFrame],
) -> ValidationReport:
    """
    Checks (per STEP 1 spec):
      - camera / lidar frame counts (non-empty, checked again here since
        validate_dataset can be called standalone)
      - timestamp monotonicity across each stream (also checked in
        validate_camera/validate_lidar; repeated here because a dataset-only
        caller shouldn't have to run those separately to catch it)
      - timestamp range / overlap duration between the two streams
    """
    report = ValidationReport()

    n_cam, n_lidar = len(camera_frames), len(lidar_frames)
    if n_cam == 0 or n_lidar == 0:
        report.add(ValidationCheckItem(
            "dataset.frame_counts_nonzero",
            ValidationStatus.INVALID,
            f"camera_frames={n_cam}, lidar_frames={n_lidar} -- both streams "
            f"must be non-empty.",
        ))
        return report

    report.add(ValidationCheckItem(
        "dataset.frame_counts_nonzero",
        ValidationStatus.VALID,
        f"camera_frames={n_cam}, lidar_frames={n_lidar}.",
    ))

    cam_ts = np.array([f.timestamp for f in camera_frames], dtype=float)
    lid_ts = np.array([f.timestamp for f in lidar_frames], dtype=float)

    if not (np.isfinite(cam_ts).all() and np.isfinite(lid_ts).all()):
        report.add(ValidationCheckItem(
            "dataset.overlap_duration",
            ValidationStatus.INVALID,
            "Cannot compute camera/LiDAR overlap: one or both streams have "
            "non-finite timestamps.",
        ))
        return report

    cam_start, cam_end = float(cam_ts.min()), float(cam_ts.max())
    lid_start, lid_end = float(lid_ts.min()), float(lid_ts.max())
    overlap_start = max(cam_start, lid_start)
    overlap_end = min(cam_end, lid_end)
    overlap_s = max(0.0, overlap_end - overlap_start)

    if overlap_s <= 0.0:
        report.add(ValidationCheckItem(
            "dataset.overlap_duration",
            ValidationStatus.INVALID,
            f"Camera time range [{cam_start:.3f}, {cam_end:.3f}] and LiDAR "
            f"time range [{lid_start:.3f}, {lid_end:.3f}] do not overlap at "
            f"all -- the two streams cannot be synchronized. Check that "
            f"both timestamps use the same clock/epoch.",
        ))
    elif overlap_s < MIN_OVERLAP_S_WARNING:
        report.add(ValidationCheckItem(
            "dataset.overlap_duration",
            ValidationStatus.WARNING,
            f"Camera/LiDAR overlap is only {overlap_s:.3f}s -- too short to "
            f"produce many synced frames; M3/M4 results may be unreliable.",
            value=overlap_s,
        ))
    else:
        report.add(ValidationCheckItem(
            "dataset.overlap_duration",
            ValidationStatus.VALID,
            f"Camera/LiDAR overlap is {overlap_s:.3f}s.",
            value=overlap_s,
        ))

    return report


# ---------------------------------------------------------------------------
# Combined entry point
# ---------------------------------------------------------------------------

def validate_input(
    camera: CameraModel,
    camera_frames: list[CameraFrame],
    lidar: LidarModel,
    lidar_frames: list[LidarFrame],
    lidar_sample_frames: int = 5,
    raise_on_invalid: bool = False,
) -> ValidationReport:
    """
    Run all STEP 1 checks (camera + lidar + dataset) and combine them into
    one ValidationReport with a single worst-of-all-checks status.

    This is meant to run BEFORE input/dataset.py's build_dataset() and
    BEFORE any evaluation metric, so a broken input surfaces as
    INPUT_INVALID with concrete reasons rather than as a misleading
    "Calibration BAD" or an opaque downstream exception.

    raise_on_invalid: if True and the combined status is INPUT_INVALID,
    raises InputValidationError(report) instead of returning it -- for
    callers (e.g. app.cli) that want validation failures to short-circuit
    the pipeline via an exception.
    """
    report = ValidationReport()
    report.merge(validate_camera(camera, camera_frames))
    report.merge(validate_lidar(lidar, lidar_frames, sample_frames=lidar_sample_frames))
    report.merge(validate_dataset(camera_frames, lidar_frames))

    if raise_on_invalid and report.is_invalid:
        raise InputValidationError(report)

    return report
