"""
evaluation/perturbation.py

Advanced/Phase-5 metric: Perturbation Sensitivity (see
evaluation_metric_spec.md section 14). Not part of the MVP scored set.

Idea: nudge the existing T_CL by small amounts along each of its 6 DOF
(translation x/y/z, rotation roll/pitch/yaw) in both directions, re-run M2
at each nudged T, and check whether the ORIGINAL T already has the lowest
error among all the nudges. If some nearby T consistently does better,
that's evidence the current calibration is not sitting at even a local
optimum -- useful signal even without ground truth, since it doesn't
require knowing the "correct" T, only whether nearby alternatives are
better or worse.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from evaluation.edge_alignment import evaluate_edge_alignment
from geometry.transform import rpy_to_rotation_matrix
from input.camera import CameraModel
from quality.noise_floor import LidarSensorSpecForFloor


DEFAULT_TRANSLATION_DELTAS_M = (0.01, 0.02)
DEFAULT_ROTATION_DELTAS_DEG = (0.1, 0.2)
DEFAULT_LOCAL_MINIMUM_TOLERANCE_PX = 0.05

_TRANSLATION_AXES = ("tx", "ty", "tz")
_ROTATION_AXES = ("roll_deg", "pitch_deg", "yaw_deg")


@dataclass
class PerturbationSample:
    axis: str
    direction: str   # "+" | "-"
    delta: float      # meters (translation axes) or degrees (rotation axes)
    mean_px: float
    valid: bool        # False if this perturbed T's M2 evaluation FAILed
    warnings: list[str] = field(default_factory=list)


@dataclass
class PerturbationResult:
    classification: str    # "AT_LOCAL_MINIMUM" | "NOT_AT_LOCAL_MINIMUM" | "FAIL"
    baseline_mean_px: float
    samples: list[PerturbationSample]
    best_sample: Optional[PerturbationSample]
    is_local_minimum: bool
    improvement_margin_px: float   # baseline - best.mean_px; positive means some nudge did better
    warnings: list[str] = field(default_factory=list)


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


def evaluate_perturbation_sensitivity(
    image: np.ndarray,
    points_lidar: np.ndarray,
    T_CL: np.ndarray,
    camera: CameraModel,
    lidar_spec: LidarSensorSpecForFloor,
    translation_deltas_m: tuple = DEFAULT_TRANSLATION_DELTAS_M,
    rotation_deltas_deg: tuple = DEFAULT_ROTATION_DELTAS_DEG,
    local_minimum_tolerance_px: float = DEFAULT_LOCAL_MINIMUM_TOLERANCE_PX,
    edge_alignment_kwargs: Optional[dict] = None,
) -> PerturbationResult:
    """
    Evaluate M2 at T_CL and at small perturbations of T_CL along each of
    its 6 DOF (both directions, each configured magnitude), and check
    whether T_CL already has the lowest per-point mean error among all of
    them (within local_minimum_tolerance_px, to absorb measurement noise).

    FAILs if the baseline M2 evaluation itself FAILs (nothing meaningful
    to compare perturbations against).
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

    samples: list[PerturbationSample] = []

    for axis_name, axis_idx in zip(_TRANSLATION_AXES, range(3)):
        for delta in translation_deltas_m:
            for sign, direction in ((1.0, "+"), (-1.0, "-")):
                T_perturbed = _perturb_translation(T_CL, axis_idx, sign * delta)
                result = evaluate_edge_alignment(
                    image=image, points_lidar=points_lidar, T_CL=T_perturbed,
                    camera=camera, lidar_spec=lidar_spec, **edge_alignment_kwargs,
                )
                valid = result.classification != "FAIL"
                samples.append(PerturbationSample(
                    axis=axis_name, direction=direction, delta=delta,
                    mean_px=result.mean_px if valid else float("nan"),
                    valid=valid, warnings=list(result.warnings) if not valid else [],
                ))

    for axis_name in _ROTATION_AXES:
        for delta in rotation_deltas_deg:
            for sign, direction in ((1.0, "+"), (-1.0, "-")):
                T_perturbed = _perturb_rotation(T_CL, axis_name, sign * delta)
                result = evaluate_edge_alignment(
                    image=image, points_lidar=points_lidar, T_CL=T_perturbed,
                    camera=camera, lidar_spec=lidar_spec, **edge_alignment_kwargs,
                )
                valid = result.classification != "FAIL"
                samples.append(PerturbationSample(
                    axis=axis_name, direction=direction, delta=delta,
                    mean_px=result.mean_px if valid else float("nan"),
                    valid=valid, warnings=list(result.warnings) if not valid else [],
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
            warnings=warnings,
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

    return PerturbationResult(
        classification="AT_LOCAL_MINIMUM" if is_local_minimum else "NOT_AT_LOCAL_MINIMUM",
        baseline_mean_px=baseline.mean_px,
        samples=samples,
        best_sample=best_sample,
        is_local_minimum=is_local_minimum,
        improvement_margin_px=improvement_margin,
        warnings=warnings,
    )
