"""
evaluation/multiframe_consistency.py

M4. Multi-frame Consistency (see evaluation_metric_spec.md v0.4).

Measures whether a FIXED, existing T_CL produces a stable per-frame error
across the whole frame sequence -- i.e. whether the calibration is reliable
frame-to-frame, or whether specific frames blow up (momentary misalignment,
outlier scenes, sync glitches, etc).

Unlike M3 (which pools frames into contiguous TIME BLOCKS to check
generalization across time windows), M4 evaluates EACH FRAME INDEPENDENTLY
and looks at the distribution of per-frame error directly -- this is what
catches a single bad frame that a block-level average would smooth over.

Pipeline:
  1. For every synced frame, run M2 (edge_alignment) independently:
       E_i = M2(T_fixed, Frame_i)
  2. Collect the valid per-frame mean_px values.
  3. Aggregate Mean/STD/P95/Max across frames, plus STEP10's robust
     alternatives to plain mean/std (see below).
  4. Flag outlier frames using outlier_method (default "hampel" -- see
     STEP10's docstring section below; "multiplier" -- the original 5x
     median rule -- remains available for comparison/back-compat).
  5. Classify STD against the sensor-relative floor(Z) using the STD
     multiplier scheme (1x / 3x, same scheme M3 uses).

STEP10 -- robust statistics (see evaluation_metric_spec.md's STEP10,
"단순 5 x median 대신 MAD / IQR / Hampel / robust z-score 등을
사용합니다"): a plain mean/std/multiplier-of-median is fragile -- a
single extreme frame inflates BOTH the mean AND the std it's supposedly
being measured against, and "5x the median" has no notion of how SPREAD
OUT the non-outlier frames already are (5x median might be a huge margin
in a noisy dataset, or a tiny one in a very clean one). This module now
always computes, regardless of which method is used for the pass/fail
is_outlier flag:
  - MAD (median absolute deviation), scaled by the standard 1.4826
    constant so it's directly comparable to a normal distribution's std
    -- this module's "mad_px".
  - IQR (interquartile range, Q3 - Q1) -- "iqr_px", with q1_px/q3_px kept
    too for transparency.
  - a robust z-score per frame: (frame_mean - median) / mad_px -- like a
    normal z-score, but built from median/MAD instead of mean/std, so a
    handful of outliers can't drag the very yardstick used to judge them.
  - Hampel identifier: outlier if |robust z-score| > hampel_k (default
    3.0, a standard choice for Hampel's X84 rule) -- this is
    evaluate_multiframe_consistency's new DEFAULT outlier_method,
    replacing the plain multiplier-of-median rule per the spec's explicit
    "instead of" framing, while "multiplier" and "iqr" remain selectable
    (outlier_method parameter) for comparison or back-compat.

Also per STEP10's explicit requirement, three separate ratios are always
reported instead of being folded into one pass/fail number:
  - valid_ratio: fraction of ALL frames that produced a usable M2 result
    (didn't FAIL outright -- too few edge points, etc).
  - failure_ratio: 1 - valid_ratio (frames M2 couldn't evaluate at all).
  - outlier_ratio: of the VALID frames, what fraction were flagged as
    outliers by outlier_method. A high failure_ratio and a high
    outlier_ratio are different problems (the calibration might be fine
    on frames it CAN evaluate, but usable on very few of them; or it
    might evaluate everywhere but be wildly inconsistent) -- conflating
    them into one number, as the pre-STEP10 version implicitly did,
    hides which problem is actually present.

Failure conditions (per spec):
  - fewer than min_frames total frames (default 30, per spec's open item
    "통계적 유의성 위해 몇 프레임 이상 필요한지, 예: 최소 30")
  - fewer than 2 valid (non-FAIL) frames (can't compute STD)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from evaluation.edge_alignment import evaluate_edge_alignment, EdgeAlignmentResult
from input.dataset import EvaluationDataset
from quality.noise_floor import (
    LidarSensorSpecForFloor,
    classify,
    STD_GOOD_MULTIPLIER,
    STD_WARNING_MULTIPLIER,
)


DEFAULT_MIN_FRAMES = 30
DEFAULT_OUTLIER_MULTIPLIER = 5.0
DEFAULT_HAMPEL_K = 3.0
DEFAULT_IQR_K = 1.5
DEFAULT_OUTLIER_METHOD = "hampel"
_MAD_NORMAL_CONSTANT = 1.4826  # scales raw MAD to be std-comparable under a normal distribution


def compute_robust_stats(values: np.ndarray) -> dict:
    """
    STEP10: median, scaled MAD, and IQR (with Q1/Q3) for an array of
    values -- the shared primitive behind Hampel/IQR outlier detection
    and the always-reported mad_px/iqr_px diagnostic fields. Returns NaN
    fields for empty input rather than raising, so callers can check
    np.isfinite on the way out instead of wrapping every call in a
    length check.
    """
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return {"median": float("nan"), "mad": float("nan"), "q1": float("nan"),
                "q3": float("nan"), "iqr": float("nan")}
    median = float(np.median(values))
    mad_raw = float(np.median(np.abs(values - median)))
    mad = _MAD_NORMAL_CONSTANT * mad_raw
    q1, q3 = np.percentile(values, [25, 75])
    return {"median": median, "mad": mad, "q1": float(q1), "q3": float(q3), "iqr": float(q3 - q1)}


@dataclass
class FrameResult:
    frame_index: int
    timestamp: float
    mean_px: float
    num_edge_points: int
    representative_depth_m: float
    floor_px: float
    classification: str    # GOOD | WARNING | BAD | FAIL (per-frame M2 classification)
    is_outlier: bool = False
    robust_z_score: float = float("nan")  # STEP10: (mean_px - median) / mad_px across valid frames
    warnings: list[str] = field(default_factory=list)


@dataclass
class MultiFrameConsistencyResult:
    frame_results: list[FrameResult]
    num_frames_total: int
    num_valid_frames: int
    num_failed_frames: int
    num_outlier_frames: int
    outlier_frame_indices: list[int]
    mean_across_frames_px: float
    std_across_frames_px: float
    median_across_frames_px: float
    p95_across_frames_px: float
    max_across_frames_px: float
    floor_px: float
    classification: str    # GOOD | WARNING | BAD | FAIL
    warnings: list[str] = field(default_factory=list)

    # STEP10 -- robust statistics + separated ratios (see module docstring).
    outlier_method: str = DEFAULT_OUTLIER_METHOD
    mad_px: float = float("nan")
    iqr_px: float = float("nan")
    q1_px: float = float("nan")
    q3_px: float = float("nan")
    valid_ratio: float = float("nan")
    failure_ratio: float = float("nan")
    outlier_ratio: float = float("nan")


def _flag_outliers(
    frame_means: np.ndarray,
    outlier_method: str,
    outlier_multiplier: float,
    hampel_k: float,
    iqr_k: float,
    robust: dict,
    warnings: list[str],
) -> tuple[np.ndarray, float]:
    """
    Returns (is_outlier boolean array, threshold-or-nan used for
    "multiplier"-method reporting/back-compat). Guards each method's own
    degenerate case (a spread statistic of ~0 -- e.g. every frame agrees
    almost exactly) the same way the original multiplier rule already
    did for median~0: fall back to a small absolute pixel epsilon rather
    than flagging near-arbitrary noise as "infinitely many multiples over
    zero".
    """
    median = robust["median"]
    n = frame_means.shape[0]

    if outlier_method == "multiplier":
        if median > 1e-6:
            threshold = outlier_multiplier * median
        else:
            threshold = max(outlier_multiplier * 0.1, 0.5)
            warnings.append(
                "Median per-frame error is ~0px; outlier detection (multiplier method) "
                "fell back to an absolute threshold since the rule is degenerate at median=0."
            )
        return frame_means > threshold, threshold

    if outlier_method == "hampel":
        mad = robust["mad"]
        if mad > 1e-6:
            z = (frame_means - median) / mad
        else:
            # every frame within a hair of the median -- MAD-based z-score
            # is degenerate (near-divide-by-zero); fall back to an
            # absolute epsilon distance from the median instead.
            z = (frame_means - median) / max(0.05, median * 0.05)
            warnings.append(
                "MAD of per-frame error is ~0px; outlier detection (hampel method) "
                "fell back to an absolute threshold since robust z-scores are "
                "degenerate when MAD=0."
            )
        return np.abs(z) > hampel_k, float("nan")

    if outlier_method == "iqr":
        iqr = robust["iqr"]
        if iqr > 1e-6:
            lo, hi = robust["q1"] - iqr_k * iqr, robust["q3"] + iqr_k * iqr
        else:
            lo, hi = median - 0.5, median + 0.5
            warnings.append(
                "IQR of per-frame error is ~0px; outlier detection (iqr method) "
                "fell back to an absolute threshold since Tukey fences are "
                "degenerate when IQR=0."
            )
        return (frame_means < lo) | (frame_means > hi), float("nan")

    raise ValueError(f"Unknown outlier_method {outlier_method!r}; expected 'multiplier', 'hampel', or 'iqr'.")


def evaluate_multiframe_consistency(
    dataset: EvaluationDataset,
    lidar_spec: LidarSensorSpecForFloor,
    min_frames: int = DEFAULT_MIN_FRAMES,
    outlier_multiplier: float = DEFAULT_OUTLIER_MULTIPLIER,
    outlier_method: str = DEFAULT_OUTLIER_METHOD,
    hampel_k: float = DEFAULT_HAMPEL_K,
    iqr_k: float = DEFAULT_IQR_K,
    edge_alignment_kwargs: Optional[dict] = None,
) -> MultiFrameConsistencyResult:
    """
    Compute M4 Multi-frame Consistency: run M2 independently on every synced
    frame in the dataset (using the fixed dataset.extrinsic.T_CL), then
    report the Mean/STD/P95/Max of per-frame error and flag outlier frames.

    outlier_method: "hampel" (default, STEP10) | "multiplier" (the
    original 5x-median rule, kept for back-compat/comparison) | "iqr"
    (Tukey's fences). Regardless of which is chosen, mad_px/iqr_px/
    q1_px/q3_px and every frame's robust_z_score are ALWAYS computed and
    reported (see this module's docstring) -- outlier_method only picks
    which one drives the is_outlier flag.
    """
    edge_alignment_kwargs = edge_alignment_kwargs or {}
    warnings: list[str] = []

    total_frames = len(dataset.frames)
    if total_frames < min_frames:
        warnings.append(
            f"Dataset has {total_frames} frame(s), below min_frames={min_frames}. "
            f"Multi-frame Consistency requires enough frames for the STD/P95 "
            f"statistics to be meaningful."
        )
        return _fail_result(total_frames, warnings, outlier_method=outlier_method)

    frame_results: list[FrameResult] = []
    for sf in dataset.frames:
        image = sf.camera_frame.load()
        points = sf.lidar_frame.load()
        result: EdgeAlignmentResult = evaluate_edge_alignment(
            image=image, points_lidar=points, T_CL=dataset.extrinsic.T_CL,
            camera=dataset.camera, lidar_spec=lidar_spec, **edge_alignment_kwargs,
        )
        frame_results.append(FrameResult(
            frame_index=sf.index,
            timestamp=sf.timestamp,
            mean_px=result.mean_px,
            num_edge_points=result.num_edge_points,
            representative_depth_m=result.representative_depth_m,
            floor_px=result.floor_px,
            classification=result.classification,
            warnings=list(result.warnings),
        ))

    valid = [f for f in frame_results if f.classification != "FAIL"]
    failed = [f for f in frame_results if f.classification == "FAIL"]
    valid_ratio = len(valid) / total_frames if total_frames > 0 else float("nan")
    failure_ratio = len(failed) / total_frames if total_frames > 0 else float("nan")

    if len(failed) > 0:
        warnings.append(f"{len(failed)}/{total_frames} frame(s) failed M2 evaluation and were excluded.")

    if len(valid) < 2:
        warnings.append(
            f"Only {len(valid)} valid frame(s) after excluding failures; "
            f"need >= 2 to compute STD. Cannot assess consistency."
        )
        return _fail_result(total_frames, warnings, frame_results=frame_results,
                             num_failed=len(failed), outlier_method=outlier_method,
                             valid_ratio=valid_ratio, failure_ratio=failure_ratio)

    frame_means = np.array([f.mean_px for f in valid])
    robust = compute_robust_stats(frame_means)
    median_px = robust["median"]

    is_outlier, _legacy_threshold = _flag_outliers(
        frame_means, outlier_method, outlier_multiplier, hampel_k, iqr_k, robust, warnings,
    )
    mad = robust["mad"]
    for f, z_flag, mean_val in zip(valid, is_outlier, frame_means):
        f.is_outlier = bool(z_flag)
        f.robust_z_score = float((mean_val - median_px) / mad) if mad > 1e-9 else float("nan")

    outliers = [f for f in valid if f.is_outlier]
    outlier_ratio = len(outliers) / len(valid) if len(valid) > 0 else float("nan")

    mean_across = float(np.mean(frame_means))
    std_across = float(np.std(frame_means, ddof=1))
    p95_across = float(np.percentile(frame_means, 95))
    max_across = float(np.max(frame_means))

    floor_px = float(np.median([f.floor_px for f in valid]))

    overall_classification = classify(std_across, floor_px, STD_GOOD_MULTIPLIER, STD_WARNING_MULTIPLIER)

    if outliers:
        warnings.append(
            f"{len(outliers)} outlier frame(s) detected ({outlier_method} method): "
            f"frame indices {[f.frame_index for f in outliers]}."
        )

    return MultiFrameConsistencyResult(
        frame_results=frame_results,
        num_frames_total=total_frames,
        num_valid_frames=len(valid),
        num_failed_frames=len(failed),
        num_outlier_frames=len(outliers),
        outlier_frame_indices=[f.frame_index for f in outliers],
        mean_across_frames_px=mean_across,
        std_across_frames_px=std_across,
        median_across_frames_px=median_px,
        p95_across_frames_px=p95_across,
        max_across_frames_px=max_across,
        floor_px=floor_px,
        classification=overall_classification,
        warnings=warnings,
        outlier_method=outlier_method,
        mad_px=mad,
        iqr_px=robust["iqr"],
        q1_px=robust["q1"],
        q3_px=robust["q3"],
        valid_ratio=valid_ratio,
        failure_ratio=failure_ratio,
        outlier_ratio=outlier_ratio,
    )


def _fail_result(
    total_frames: int,
    warnings: list[str],
    frame_results: Optional[list[FrameResult]] = None,
    num_failed: int = 0,
    outlier_method: str = DEFAULT_OUTLIER_METHOD,
    valid_ratio: float = float("nan"),
    failure_ratio: float = float("nan"),
) -> MultiFrameConsistencyResult:
    return MultiFrameConsistencyResult(
        frame_results=frame_results or [],
        num_frames_total=total_frames,
        num_valid_frames=0,
        num_failed_frames=num_failed,
        num_outlier_frames=0,
        outlier_frame_indices=[],
        mean_across_frames_px=float("nan"),
        std_across_frames_px=float("nan"),
        median_across_frames_px=float("nan"),
        p95_across_frames_px=float("nan"),
        max_across_frames_px=float("nan"),
        floor_px=float("nan"),
        classification="FAIL",
        warnings=warnings,
        outlier_method=outlier_method,
        valid_ratio=valid_ratio,
        failure_ratio=failure_ratio,
        outlier_ratio=float("nan"),
    )
