"""
evaluation/perturbation.py

Advanced/Phase-5 metric: Perturbation Sensitivity (see
evaluation_metric_spec.md section 14 / STEP11). Not part of the MVP
scored set.

Idea: nudge the existing T_CL by small amounts along each of its 6 DOF
(translation x/y/z, rotation roll/pitch/yaw) in both directions, re-run M2
at each nudged T, and check whether the ORIGINAL T already has the lowest
error among all the nudges. If some nearby T consistently does better,
that's evidence the current calibration is not sitting at even a local
optimum -- useful signal even without ground truth, since it doesn't
require knowing the "correct" T, only whether nearby alternatives are
better or worse.

STEP11 -- Calibration Sensitivity Analysis (see evaluation_metric_spec.md's
STEP11, "이제 기존 calibration을 일부러 흔들어봅니다"): beyond the binary
"is T_CL at a local minimum" question above, this module also reports, PER
AXIS, how MUCH M2's error changes as that axis is perturbed by increasing
amounts -- exactly the spec's own worked example:

    Parameter   Sensitivity
    Yaw         ██████████ HIGH
    Tx          █████████  HIGH
    Pitch       ████       MEDIUM
    Ty          ██         LOW
    Roll        █          LOW
    Tz          █          LOW

This is the data STEP12's Root Cause Diagnosis Engine will use later (e.g.
"right-side error is high AND yaw sensitivity is HIGH -> yaw misalignment
is a plausible cause") -- this module only produces the sensitivity
numbers themselves, not the diagnosis.

Delta grids match the spec exactly:
  rotation:    ±0.05°, ±0.1°, ±0.2°, ±0.5°, ±1.0° (roll/pitch/yaw, each independently)
  translation: ±1mm, ±5mm, ±10mm, ±20mm (tx/ty/tz, each independently)
  timestamp:   ±5ms, ±10ms, ±20ms, ±50ms, ±100ms (see below)

Per-axis sensitivity classification is sensor-relative (consistent with
this project's floor(Z) design principle throughout): HIGH if even the
SMALLEST configured perturbation already moves mean_px by more than
floor_px (more error than ordinary sensor noise would explain); MEDIUM if
only the LARGEST configured perturbation does; LOW if even the largest
configured perturbation stays within the noise floor.

Timestamp sensitivity is opt-in and reuses STEP5's motion.deskew math:
"if the platform were moving with velocity_mps/velocity_rps and sync were
off by Δt, where would every LiDAR point actually have been". Passing
point_times_s=zeros(N) and reference_time_s=Δt into
motion.deskew.deskew_points_constant_velocity turns its normal PER-POINT
within-scan correction into a single UNIFORM rigid shift of the whole
point cloud by Δt -- exactly what "the whole frame was captured Δt off"
means, reusing already-tested code rather than a parallel implementation.
Requires an assumed/known platform velocity (this tool has no independent
way to measure one -- same limitation STEP5's own --deskew-* flags have);
without one, timestamp sensitivity is simply not computed rather than
guessed.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Optional
import os

import numpy as np

from evaluation.edge_alignment import evaluate_edge_alignment
from geometry.transform import rpy_to_rotation_matrix
from input.camera import CameraModel
from quality.noise_floor import LidarSensorSpecForFloor


DEFAULT_TRANSLATION_DELTAS_M = (0.001, 0.005, 0.010, 0.020)     # 1/5/10/20 mm, per STEP11 spec
DEFAULT_ROTATION_DELTAS_DEG = (0.05, 0.1, 0.2, 0.5, 1.0)        # per STEP11 spec
DEFAULT_TIMESTAMP_DELTAS_S = (0.005, 0.010, 0.020, 0.050, 0.100)  # 5/10/20/50/100 ms, per STEP11 spec
DEFAULT_LOCAL_MINIMUM_TOLERANCE_PX = 0.05

_TRANSLATION_AXES = ("tx", "ty", "tz")
_ROTATION_AXES = ("roll_deg", "pitch_deg", "yaw_deg")
_TIMESTAMP_AXIS = "timestamp"

_SENSITIVITY_ORDER = {"HIGH": 2, "MEDIUM": 1, "LOW": 0}


@dataclass
class PerturbationSample:
    axis: str
    direction: str   # "+" | "-"
    delta: float      # meters (translation axes), degrees (rotation axes), or seconds (timestamp axis)
    mean_px: float
    valid: bool        # False if this perturbed T's M2 evaluation FAILed
    warnings: list[str] = field(default_factory=list)


@dataclass
class AxisSensitivity:
    """STEP11: per-axis sensitivity summary -- the data behind the spec's
    HIGH/MEDIUM/LOW bar chart."""
    axis: str
    classification: str          # "HIGH" | "MEDIUM" | "LOW"
    small_delta_effect_px: float  # mean |mean_px - baseline| at the smallest configured magnitude
    large_delta_effect_px: float  # mean |mean_px - baseline| at the largest configured magnitude


@dataclass
class PerturbationResult:
    classification: str    # "AT_LOCAL_MINIMUM" | "NOT_AT_LOCAL_MINIMUM" | "FAIL"
    baseline_mean_px: float
    samples: list[PerturbationSample]
    best_sample: Optional[PerturbationSample]
    is_local_minimum: bool
    improvement_margin_px: float   # baseline - best.mean_px; positive means some nudge did better
    warnings: list[str] = field(default_factory=list)
    # STEP11 -- per-axis sensitivity ranking (see module docstring). Empty
    # if the baseline itself FAILed (nothing to measure sensitivity against).
    axis_sensitivities: list[AxisSensitivity] = field(default_factory=list)
    timestamp_sensitivity_computed: bool = False


def _perturb_translation(T_CL: np.ndarray, axis_idx: int, delta: float) -> np.ndarray:
    T = T_CL.copy()
    T[axis_idx, 3] += delta
    return T


def _perturb_rotation(T_CL: np.ndarray, axis: str, delta_deg: float) -> np.ndarray:
    roll = delta_deg if axis == "roll_deg" else 0.0
    pitch = delta_deg if axis == "pitch_deg" else 0.0
    yaw = delta_deg if axis == "yaw_deg" else 0.0
    dR = rpy_to_rotation_matrix(roll, pitch, yaw, degrees=True)
    T = T_CL.copy()
    T[:3, :3] = dR @ T_CL[:3, :3]
    return T


def _perturb_timestamp_points(points_lidar: np.ndarray, delta_s: float,
                               linear_velocity_mps: np.ndarray, angular_velocity_rps: np.ndarray) -> np.ndarray:
    """
    Rigidly shift the WHOLE point cloud by delta_s of platform motion --
    see this module's docstring for why passing point_times_s=zeros(N)
    and reference_time_s=delta_s into motion.deskew's per-point-time-
    aware function produces exactly a uniform shift instead of its normal
    within-scan correction.
    """
    from motion.deskew import deskew_points_constant_velocity
    n = points_lidar.shape[0]
    result = deskew_points_constant_velocity(
        points_lidar, scan_period_s=max(abs(delta_s), 1.0),
        linear_velocity_mps=linear_velocity_mps, angular_velocity_rps=angular_velocity_rps,
        point_times_s=np.zeros(n), reference_time_s=delta_s,
    )
    return result.points_deskewed


def _evaluate_one_sample(
    image: np.ndarray,
    points_lidar: np.ndarray,
    T_perturbed: np.ndarray,
    camera: CameraModel,
    lidar_spec: LidarSensorSpecForFloor,
    edge_alignment_kwargs: dict,
    axis_name: str,
    direction: str,
    delta: float,
) -> PerturbationSample:
    """Run M2 at one perturbed T (or perturbed points_lidar, for the
    timestamp axis) and package the result as a PerturbationSample.
    Factored out as a standalone function (rather than inlined in the
    loop) so it can be submitted to a thread pool -- each call is
    independent of every other, making this embarrassingly parallel."""
    result = evaluate_edge_alignment(
        image=image, points_lidar=points_lidar, T_CL=T_perturbed,
        camera=camera, lidar_spec=lidar_spec, **edge_alignment_kwargs,
    )
    valid = result.classification != "FAIL"
    return PerturbationSample(
        axis=axis_name, direction=direction, delta=delta,
        mean_px=result.mean_px if valid else float("nan"),
        valid=valid, warnings=list(result.warnings) if not valid else [],
    )


def _compute_axis_sensitivities(
    samples: list[PerturbationSample],
    baseline_mean_px: float,
    floor_px: float,
) -> list[AxisSensitivity]:
    """
    STEP11: for each axis present in `samples`, compare mean_px at the
    SMALLEST vs LARGEST configured magnitude against baseline_mean_px,
    and classify HIGH/MEDIUM/LOW relative to floor_px (see module
    docstring for the exact rule). Axes with no valid samples at all are
    skipped (not reported as LOW -- "no data" and "confirmed low
    sensitivity" are different things).
    """
    by_axis: dict[str, list[PerturbationSample]] = {}
    for s in samples:
        if s.valid:
            by_axis.setdefault(s.axis, []).append(s)

    sensitivities = []
    for axis, axis_samples in by_axis.items():
        deltas = sorted({s.delta for s in axis_samples})
        if not deltas:
            continue
        small_delta, large_delta = deltas[0], deltas[-1]

        small_samples = [s for s in axis_samples if s.delta == small_delta]
        large_samples = [s for s in axis_samples if s.delta == large_delta]
        small_effect = float(np.mean([abs(s.mean_px - baseline_mean_px) for s in small_samples]))
        large_effect = float(np.mean([abs(s.mean_px - baseline_mean_px) for s in large_samples]))

        if floor_px > 0 and small_effect > floor_px:
            classification = "HIGH"
        elif floor_px > 0 and large_effect > floor_px:
            classification = "MEDIUM"
        else:
            classification = "LOW"

        sensitivities.append(AxisSensitivity(
            axis=axis, classification=classification,
            small_delta_effect_px=small_effect, large_delta_effect_px=large_effect,
        ))

    sensitivities.sort(key=lambda a: (_SENSITIVITY_ORDER[a.classification], a.large_delta_effect_px), reverse=True)
    return sensitivities


def evaluate_perturbation_sensitivity(
    image: np.ndarray,
    points_lidar: np.ndarray,
    T_CL: np.ndarray,
    camera: CameraModel,
    lidar_spec: LidarSensorSpecForFloor,
    translation_deltas_m: tuple = DEFAULT_TRANSLATION_DELTAS_M,
    rotation_deltas_deg: tuple = DEFAULT_ROTATION_DELTAS_DEG,
    timestamp_deltas_s: tuple = DEFAULT_TIMESTAMP_DELTAS_S,
    local_minimum_tolerance_px: float = DEFAULT_LOCAL_MINIMUM_TOLERANCE_PX,
    linear_velocity_mps: Optional[np.ndarray] = None,
    angular_velocity_rps: Optional[np.ndarray] = None,
    edge_alignment_kwargs: Optional[dict] = None,
    max_workers: Optional[int] = None,
) -> PerturbationResult:
    """
    Evaluate M2 at T_CL and at small perturbations of T_CL along each of
    its 6 DOF (both directions, each configured magnitude, per STEP11's
    exact delta grids), and check whether T_CL already has the lowest
    per-point mean error among all of them (within local_minimum_tolerance_px,
    to absorb measurement noise). Also computes STEP11's per-axis
    sensitivity ranking (see module docstring).

    linear_velocity_mps / angular_velocity_rps: if EITHER is given, also
    perturbs along a "timestamp" axis (see _perturb_timestamp_points) --
    requires an assumed/known platform velocity, since this tool has no
    independent way to measure one. Leaving both None (default) skips
    timestamp sensitivity entirely (PerturbationResult.
    timestamp_sensitivity_computed stays False) rather than guessing.

    FAILs if the baseline M2 evaluation itself FAILs (nothing meaningful
    to compare perturbations against).

    All perturbation samples (translation + rotation + optional
    timestamp) are independent M2 evaluations against the same
    image/camera/lidar_spec -- an embarrassingly parallel workload,
    run concurrently via a ThreadPoolExecutor (see the original
    docstring notes on why threads help here despite Python's GIL:
    each M2 call spends most of its time inside numpy/OpenCV/SciPy C
    extensions that release the GIL). Results are collected via
    executor.map, which preserves input order regardless of which
    thread finishes first -- so `samples` is deterministic between runs.
    """
    edge_alignment_kwargs = edge_alignment_kwargs or {}
    warnings: list[str] = []

    baseline = evaluate_edge_alignment(
        image=image, points_lidar=points_lidar, T_CL=T_CL,
        camera=camera, lidar_spec=lidar_spec, **edge_alignment_kwargs,
    )
    if baseline.classification == "FAIL":
        warnings.append("Baseline M2 evaluation FAILed; cannot assess perturbation sensitivity.")
        return PerturbationResult(
            classification="FAIL", baseline_mean_px=float("nan"), samples=[],
            best_sample=None, is_local_minimum=False, improvement_margin_px=float("nan"),
            warnings=warnings,
        )

    # Build the full task list up front: (T_perturbed, points_for_this_sample,
    # axis, direction, delta). Translation/rotation samples perturb T_CL
    # with the ORIGINAL points_lidar; the timestamp axis (if enabled)
    # perturbs points_lidar itself with the ORIGINAL T_CL.
    tasks: list[tuple[np.ndarray, np.ndarray, str, str, float]] = []
    for axis_name, axis_idx in zip(_TRANSLATION_AXES, range(3)):
        for delta in translation_deltas_m:
            for sign, direction in ((1.0, "+"), (-1.0, "-")):
                tasks.append((_perturb_translation(T_CL, axis_idx, sign * delta), points_lidar,
                              axis_name, direction, delta))
    for axis_name in _ROTATION_AXES:
        for delta in rotation_deltas_deg:
            for sign, direction in ((1.0, "+"), (-1.0, "-")):
                tasks.append((_perturb_rotation(T_CL, axis_name, sign * delta), points_lidar,
                              axis_name, direction, delta))

    timestamp_sensitivity_computed = linear_velocity_mps is not None or angular_velocity_rps is not None
    if timestamp_sensitivity_computed:
        lin_v = linear_velocity_mps if linear_velocity_mps is not None else np.zeros(3)
        ang_v = angular_velocity_rps if angular_velocity_rps is not None else np.zeros(3)
        for delta in timestamp_deltas_s:
            for sign, direction in ((1.0, "+"), (-1.0, "-")):
                shifted_points = _perturb_timestamp_points(points_lidar, sign * delta, lin_v, ang_v)
                tasks.append((T_CL, shifted_points, _TIMESTAMP_AXIS, direction, delta))
    else:
        warnings.append(
            "Timestamp sensitivity not computed: no linear_velocity_mps/angular_velocity_rps "
            "given (this tool has no independent way to measure platform velocity)."
        )

    workers = min(len(tasks), max_workers or os.cpu_count() or 4)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        samples: list[PerturbationSample] = list(executor.map(
            lambda t: _evaluate_one_sample(
                image, t[1], t[0], camera, lidar_spec, edge_alignment_kwargs, t[2], t[3], t[4],
            ),
            tasks,
        ))

    valid_samples = [s for s in samples if s.valid]
    n_invalid = len(samples) - len(valid_samples)
    if n_invalid > 0:
        warnings.append(f"{n_invalid}/{len(samples)} perturbation samples FAILed M2 evaluation and were excluded.")

    if not valid_samples:
        warnings.append("All perturbation samples FAILed; cannot assess local-minimum status.")
        return PerturbationResult(
            classification="FAIL", baseline_mean_px=baseline.mean_px, samples=samples,
            best_sample=None, is_local_minimum=False, improvement_margin_px=float("nan"),
            warnings=warnings, timestamp_sensitivity_computed=timestamp_sensitivity_computed,
        )

    best_sample = min(valid_samples, key=lambda s: s.mean_px)
    improvement_margin = baseline.mean_px - best_sample.mean_px
    is_local_minimum = improvement_margin <= local_minimum_tolerance_px

    if not is_local_minimum:
        warnings.append(
            f"Nudging T_CL along {best_sample.axis} ({best_sample.direction}{best_sample.delta}) "
            f"reduced mean error by {improvement_margin:.3f}px -- current T may not be locally optimal "
            f"along this axis."
        )

    axis_sensitivities = _compute_axis_sensitivities(samples, baseline.mean_px, baseline.floor_px)

    return PerturbationResult(
        classification="AT_LOCAL_MINIMUM" if is_local_minimum else "NOT_AT_LOCAL_MINIMUM",
        baseline_mean_px=baseline.mean_px,
        samples=samples,
        best_sample=best_sample,
        is_local_minimum=is_local_minimum,
        improvement_margin_px=improvement_margin,
        warnings=warnings,
        axis_sensitivities=axis_sensitivities,
        timestamp_sensitivity_computed=timestamp_sensitivity_computed,
    )
