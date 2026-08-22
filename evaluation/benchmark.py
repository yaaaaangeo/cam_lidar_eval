"""
evaluation/benchmark.py

STEP 15 -- Benchmark / Regression Test (see evaluation_metric_spec.md's
STEP15, the roadmap's final step). Per the spec's own framing, this isn't
really "one more feature" so much as a call to FORMALIZE a check that
should already have been running all along: given a synthetic scene with
a KNOWN correct calibration, does this tool's own error metric (M2)
actually get WORSE as the evaluated T_CL is nudged farther from that known
truth? And separately -- given a synthetic scene with a KNOWN injected
problem (a real misalignment, real dynamic contamination, ...), does
evaluation.root_cause's diagnosis engine actually name that problem as
its top candidate?

Two kinds of benchmark, both built entirely from machinery earlier STEPs
already produced (no new evaluation logic here -- this module only
ORCHESTRATES and CHECKS):

  1. Monotonicity benchmark (spec's own worked example):
         GT (0 deg) -> +0.1 deg yaw -> +0.2 deg yaw -> +0.5 deg yaw
     and the system should report INCREASING M2 error at each step, for
     EVERY one of roll/pitch/yaw/tx/ty/tz (spec: "각 축 테스트... 전부
     합니다"), plus a timestamp axis using STEP11's motion.deskew-based
     timestamp perturbation (reused directly from evaluation.perturbation,
     not reimplemented here).

  2. Diagnosis accuracy benchmark (spec's own framing: "Known problem ->
     System -> Correct diagnosis?"): construct a scene with an ACTUAL
     geometric problem injected (e.g. evaluate against a T_CL that's
     really rotated by a known yaw offset from the scene's true geometry,
     or a scene with a real moving-object contamination band), run the
     REAL pipeline (M2 -> spatial analysis -> dynamic filtering ->
     perturbation sensitivity -> root cause diagnosis) end to end with NO
     mocked intermediate results, and check that
     evaluation.root_cause.diagnose_root_cause's TOP-ranked candidate
     names the actual injected cause. This is deliberately a stronger
     check than evaluation/root_cause.py's own unit tests (which use
     lightweight fakes for speed and rule-isolation) -- it exercises the
     real, wired-together pipeline the way a person running the CLI
     actually would.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from evaluation.edge_alignment import evaluate_edge_alignment
from evaluation.perturbation import _perturb_translation, _perturb_rotation, _perturb_timestamp_points


DEFAULT_ROTATION_BENCHMARK_DELTAS_DEG = (0.0, 0.1, 0.2, 0.5)   # spec's own literal example sequence
DEFAULT_TRANSLATION_BENCHMARK_DELTAS_M = (0.0, 0.005, 0.010, 0.020)
DEFAULT_TIMESTAMP_BENCHMARK_DELTAS_S = (0.0, 0.010, 0.020, 0.050, 0.100)  # spec's own 0/10/20/50/100ms

_ROTATION_AXES = ("roll_deg", "pitch_deg", "yaw_deg")
_TRANSLATION_AXES = ("tx", "ty", "tz")

DEFAULT_MONOTONIC_TOLERANCE_PX = 0.05  # absorbs sub-pixel measurement noise; see check_monotonic_nondecreasing


@dataclass
class MonotonicitySample:
    delta: float       # perturbation magnitude: degrees (rotation axes), meters (translation), or seconds (timestamp)
    mean_px: float
    valid: bool = True  # False if M2 FAILed at this perturbation level


@dataclass
class AxisMonotonicityResult:
    axis: str
    samples: list[MonotonicitySample]  # in increasing delta order, samples[0].delta == 0 (ground truth)
    is_monotonic: bool
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "axis": self.axis,
            "is_monotonic": self.is_monotonic,
            "samples": [{"delta": s.delta, "mean_px": s.mean_px, "valid": s.valid} for s in self.samples],
            "warnings": list(self.warnings),
        }


def check_monotonic_nondecreasing(values: list[float], tolerance: float = DEFAULT_MONOTONIC_TOLERANCE_PX) -> bool:
    """
    True if every consecutive pair is non-decreasing within `tolerance`
    (values[i+1] >= values[i] - tolerance for all i) -- the spec's own
    "0.1 < 0.2 < 0.5 순으로 나빠지는가" check, with a small absolute
    tolerance to absorb ordinary sub-pixel measurement noise rather than
    demanding mathematically perfect monotonicity from a real (noisy)
    pixel-error measurement.
    """
    for i in range(1, len(values)):
        if values[i] < values[i - 1] - tolerance:
            return False
    return True


def _run_axis_samples(
    image: np.ndarray,
    points_lidar: np.ndarray,
    T_CL_ground_truth: np.ndarray,
    camera,
    lidar_spec,
    axis: str,
    deltas,
    edge_alignment_kwargs: dict,
) -> AxisMonotonicityResult:
    samples = []
    warnings = []
    for delta in deltas:
        if axis in _ROTATION_AXES:
            T_eval = _perturb_rotation(T_CL_ground_truth, axis, delta)
        elif axis in _TRANSLATION_AXES:
            axis_idx = _TRANSLATION_AXES.index(axis)
            T_eval = _perturb_translation(T_CL_ground_truth, axis_idx, delta)
        else:
            raise ValueError(f"Unknown axis {axis!r}; expected one of {_ROTATION_AXES + _TRANSLATION_AXES}")

        result = evaluate_edge_alignment(
            image=image, points_lidar=points_lidar, T_CL=T_eval,
            camera=camera, lidar_spec=lidar_spec, **edge_alignment_kwargs,
        )
        valid = result.classification != "FAIL"
        samples.append(MonotonicitySample(delta=delta, mean_px=result.mean_px if valid else float("nan"), valid=valid))
        if not valid:
            warnings.append(f"{axis} delta={delta}: M2 FAILed ({'; '.join(result.warnings) or 'no detail'}).")

    valid_values = [s.mean_px for s in samples if s.valid]
    is_monotonic = check_monotonic_nondecreasing(valid_values) if len(valid_values) >= 2 else False
    if len(valid_values) < 2:
        warnings.append(f"{axis}: fewer than 2 valid samples; cannot assess monotonicity.")

    return AxisMonotonicityResult(axis=axis, samples=samples, is_monotonic=is_monotonic, warnings=warnings)


def run_rotation_translation_monotonicity_benchmark(
    image: np.ndarray,
    points_lidar: np.ndarray,
    T_CL_ground_truth: np.ndarray,
    camera,
    lidar_spec,
    axes: tuple = _ROTATION_AXES + _TRANSLATION_AXES,
    rotation_deltas_deg: tuple = DEFAULT_ROTATION_BENCHMARK_DELTAS_DEG,
    translation_deltas_m: tuple = DEFAULT_TRANSLATION_BENCHMARK_DELTAS_M,
    edge_alignment_kwargs: Optional[dict] = None,
) -> dict:
    """
    STEP15's own worked example, generalized to every one of the spec's
    six axes ("각 축 테스트... 전부 합니다"): for each axis, evaluate M2
    at T_CL_ground_truth perturbed by each configured delta (starting at
    0 = the ground truth itself), and check that mean_px only ever gets
    WORSE (or stays about the same) as the perturbation grows -- never
    meaningfully better. A tool whose own error metric doesn't respect
    this basic sanity property can't be trusted to diagnose anything.

    Returns {axis_name: AxisMonotonicityResult}.
    """
    edge_alignment_kwargs = edge_alignment_kwargs or {}
    results = {}
    for axis in axes:
        deltas = rotation_deltas_deg if axis in _ROTATION_AXES else translation_deltas_m
        results[axis] = _run_axis_samples(
            image, points_lidar, T_CL_ground_truth, camera, lidar_spec, axis, deltas, edge_alignment_kwargs,
        )
    return results


def run_timestamp_monotonicity_benchmark(
    image: np.ndarray,
    points_lidar: np.ndarray,
    T_CL: np.ndarray,
    camera,
    lidar_spec,
    linear_velocity_mps: np.ndarray,
    angular_velocity_rps: Optional[np.ndarray] = None,
    timestamp_deltas_s: tuple = DEFAULT_TIMESTAMP_BENCHMARK_DELTAS_S,
    edge_alignment_kwargs: Optional[dict] = None,
) -> AxisMonotonicityResult:
    """
    STEP15's timestamp benchmark (spec: "0ms / 10ms / 20ms / 50ms /
    100ms"): reuses STEP11's motion.deskew-based timestamp perturbation
    (evaluation.perturbation._perturb_timestamp_points) to rigidly shift
    the WHOLE point cloud by each Δt under an assumed constant platform
    velocity, and checks that M2's error only gets worse as the
    simulated sync error grows -- larger timing mismatches under real
    motion should never coincidentally look BETTER than a smaller one.

    Requires a nonzero linear_velocity_mps and/or angular_velocity_rps --
    at zero velocity every Δt is a no-op (see motion.deskew's own
    documented behavior) and this benchmark would be checking nothing.
    """
    edge_alignment_kwargs = edge_alignment_kwargs or {}
    angular_velocity_rps = angular_velocity_rps if angular_velocity_rps is not None else np.zeros(3)
    if np.allclose(linear_velocity_mps, 0) and np.allclose(angular_velocity_rps, 0):
        raise ValueError(
            "run_timestamp_monotonicity_benchmark needs a nonzero linear_velocity_mps or "
            "angular_velocity_rps -- at zero velocity every timestamp delta is a no-op "
            "(see motion.deskew), so there would be nothing to benchmark."
        )

    samples = []
    warnings = []
    for delta in timestamp_deltas_s:
        shifted_points = _perturb_timestamp_points(points_lidar, delta, linear_velocity_mps, angular_velocity_rps)
        result = evaluate_edge_alignment(
            image=image, points_lidar=shifted_points, T_CL=T_CL,
            camera=camera, lidar_spec=lidar_spec, **edge_alignment_kwargs,
        )
        valid = result.classification != "FAIL"
        samples.append(MonotonicitySample(delta=delta, mean_px=result.mean_px if valid else float("nan"), valid=valid))
        if not valid:
            warnings.append(f"timestamp delta={delta}s: M2 FAILed.")

    valid_values = [s.mean_px for s in samples if s.valid]
    is_monotonic = check_monotonic_nondecreasing(valid_values) if len(valid_values) >= 2 else False
    if len(valid_values) < 2:
        warnings.append("timestamp: fewer than 2 valid samples; cannot assess monotonicity.")

    return AxisMonotonicityResult(axis="timestamp", samples=samples, is_monotonic=is_monotonic, warnings=warnings)


@dataclass
class DiagnosisBenchmarkCase:
    name: str
    expected_top_cause: str
    actual_top_cause: Optional[str]
    passed: bool
    all_causes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "name": self.name, "expected_top_cause": self.expected_top_cause,
            "actual_top_cause": self.actual_top_cause, "passed": self.passed,
            "all_causes": list(self.all_causes),
        }


def check_diagnosis_case(name: str, expected_top_cause: str, diagnosis_result) -> DiagnosisBenchmarkCase:
    """
    STEP15's "Known problem -> System -> Correct diagnosis?" check, applied
    to an ALREADY-COMPUTED evaluation.root_cause.RootCauseDiagnosisResult
    (built by running the real pipeline against a scene with a known
    injected problem -- see this module's docstring). Passes if
    expected_top_cause is the FIRST (highest-ranked) candidate; a correct
    cause buried below a stronger false positive still counts as a miss,
    since the whole point of ranking is that the top entry is what a
    person would act on first.
    """
    causes = [c.cause for c in diagnosis_result.candidates]
    actual_top = causes[0] if causes else None
    return DiagnosisBenchmarkCase(
        name=name, expected_top_cause=expected_top_cause, actual_top_cause=actual_top,
        passed=(actual_top == expected_top_cause), all_causes=causes,
    )
