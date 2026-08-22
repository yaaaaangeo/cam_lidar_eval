"""
evaluation/dynamic_filter.py

STEP 8 -- Dynamic Object Filtering (see evaluation_metric_spec.md's STEP 8).

Problem: a moving object visible in the camera frame (e.g. a car) may have
already moved by the time the LiDAR captures the same region (even within
one scan, and more so across the ~10-100ms typical camera-LiDAR sync
offset -- see STEP2's Δt estimation). Projecting that LiDAR return under
the CURRENT extrinsic then lands on the WRONG part of the image -- not
because the calibration is wrong, but because the object itself moved.
Without separating this out, M2 conflates "the calibration is off" with
"a car drove through the scene", making the score look worse than the
calibration actually is, in a way that's specific to whichever frames
happened to contain moving traffic.

This module has no object detector (no camera-side semantic segmentation,
no bounding boxes) -- this codebase is deliberately dependency-light
(numpy/opencv/scipy only, see quality/noise_floor.py's own docstring for
the same design principle), and bundling a real detector (YOLO or
similar) would be a large, heavy dependency for what should stay an
evaluation tool. Two dependency-free options remain, both implemented
here:

  1. classify_points_by_motion_consistency (the spec's "2차 구현",
     promoted to primary here since it's the only genuinely free-standing
     option): reuses STEP4's range-image structure
     (geometry.range_image.build_range_image) across a small WINDOW of
     nearby frames. A (ring, azimuth) cell whose range value is STABLE
     across the window is almost certainly a static surface; one whose
     range JUMPS AROUND is either a moving object passing through that
     cell, or -- and this is the method's central, unavoidable caveat --
     the PLATFORM ITSELF moved between frames, which looks identical to
     an object moving from a single sensor's point of view without an
     independent ego-motion estimate. This method is therefore only
     directly meaningful for an (approximately) STATIONARY platform, or
     one where ego-motion has already been compensated for by some other
     means; see this module's classify_points_by_motion_consistency
     docstring for how a caller can flag this. A genuinely general
     solution needs real odometry/SLAM, which is out of scope here (the
     spec's own "나중에는" ["later on"] framing for the motion-consistency
     approach acknowledges the same gap).

  2. apply_external_dynamic_mask (the spec's "1차 구현" in spirit, without
     bundling a detector): if the caller already has per-point dynamic/
     static labels from THEIR OWN object detector or tracker, this is a
     one-line pass-through that plugs those results into the same
     downstream M2 comparison (compare_with_without_dynamic_filtering)
     the motion-consistency method feeds -- so a real detector can be
     substituted in without touching any other code, once one is
     available.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from geometry.range_image import (
    build_range_image,
    compute_azimuth_rad,
    derive_ring_index,
    DEFAULT_NUM_RINGS,
    DEFAULT_NUM_AZIMUTH_BINS,
)


DEFAULT_RANGE_STD_THRESHOLD_M = 0.3
DEFAULT_MIN_FRAMES_PRESENT = 3

STATIC = 0
DYNAMIC = 1
UNKNOWN = 2

_LABEL_NAMES = {STATIC: "static", DYNAMIC: "dynamic", UNKNOWN: "unknown"}


@dataclass
class MotionConsistencyResult:
    """
    Cell-level (ring, azimuth) classification from
    classify_points_by_motion_consistency, plus the machinery to map it
    back onto any individual frame's own points.
    """
    cell_label: np.ndarray       # (num_rings, num_azimuth_bins) int, one of STATIC/DYNAMIC/UNKNOWN
    cell_range_std_m: np.ndarray  # (num_rings, num_azimuth_bins) float, NaN where UNKNOWN
    cell_num_frames_present: np.ndarray  # (num_rings, num_azimuth_bins) int
    num_rings: int
    num_azimuth_bins: int
    num_frames_used: int
    range_std_threshold_m: float
    min_frames_present: int

    def label_counts(self) -> dict:
        return {
            _LABEL_NAMES[label]: int(np.sum(self.cell_label == label))
            for label in (STATIC, DYNAMIC, UNKNOWN)
        }


def classify_points_by_motion_consistency(
    frames_points: list[np.ndarray],
    num_rings: int = DEFAULT_NUM_RINGS,
    num_azimuth_bins: int = DEFAULT_NUM_AZIMUTH_BINS,
    vertical_fov_deg: Optional[float] = None,
    range_std_threshold_m: float = DEFAULT_RANGE_STD_THRESHOLD_M,
    min_frames_present: int = DEFAULT_MIN_FRAMES_PRESENT,
) -> MotionConsistencyResult:
    """
    Classify every (ring, azimuth) range-image cell as STATIC, DYNAMIC, or
    UNKNOWN, from how stable its range value is across `frames_points` (a
    small window of nearby LiDAR frames -- the caller picks how many and
    which; a handful of frames immediately around the one being evaluated
    is typical).

    IMPORTANT CAVEAT (see this module's docstring): this method assumes
    the sensor platform itself is approximately stationary across the
    window, or that ego-motion has already been compensated for. A moving
    platform makes the ENTIRE static scene look "dynamic" under this
    method (every cell's range genuinely does change frame to frame, for
    a reason that has nothing to do with actual moving objects), which
    would swamp real dynamic-object detection. Do not use this on a
    moving-platform frame window without separately compensating for the
    platform's own motion first.

    A cell needs data in at least min_frames_present of the given frames
    to be classified STATIC or DYNAMIC at all; otherwise (too little
    data -- occluded most of the window, at the very edge of the FOV,
    etc.) it's UNKNOWN, distinct from "we checked and it's static".
    range_std_threshold_m is compared against the STANDARD DEVIATION
    (not e.g. peak-to-peak range) of that cell's range values across the
    frames where it had data, so a cell that's simply noisy in a small,
    sensor-precision way stays STATIC -- only a cell whose range
    genuinely swings around (a real object passing through, or occluding
    then un-occluding a farther static surface) crosses the bar.
    """
    n_frames = len(frames_points)
    if n_frames == 0:
        cell_label = np.full((num_rings, num_azimuth_bins), UNKNOWN, dtype=np.int64)
        return MotionConsistencyResult(
            cell_label=cell_label,
            cell_range_std_m=np.full((num_rings, num_azimuth_bins), np.nan),
            cell_num_frames_present=np.zeros((num_rings, num_azimuth_bins), dtype=np.int64),
            num_rings=num_rings, num_azimuth_bins=num_azimuth_bins, num_frames_used=0,
            range_std_threshold_m=range_std_threshold_m, min_frames_present=min_frames_present,
        )

    range_stack = np.full((n_frames, num_rings, num_azimuth_bins), np.nan, dtype=np.float64)
    for i, points in enumerate(frames_points):
        if points.shape[0] == 0:
            continue
        ri = build_range_image(points, num_rings=num_rings, num_azimuth_bins=num_azimuth_bins,
                                vertical_fov_deg=vertical_fov_deg)
        range_stack[i] = ri.range_m

    present_mask = np.isfinite(range_stack)
    num_present = present_mask.sum(axis=0)  # (num_rings, num_azimuth_bins)

    with np.errstate(invalid="ignore"):
        import warnings as _warnings
        with _warnings.catch_warnings():
            # cells with 0 or 1 present frames produce an all-NaN or
            # single-value nanstd computation, which numpy warns about
            # ("Degrees of freedom <= 0 for slice") -- already handled
            # below via `enough_data` (min_frames_present), so the
            # warning is noise, not a signal of an actual problem.
            _warnings.simplefilter("ignore", category=RuntimeWarning)
            cell_std = np.nanstd(range_stack, axis=0)  # NaN where all-NaN

    enough_data = num_present >= min_frames_present
    is_dynamic = enough_data & (cell_std > range_std_threshold_m)
    is_static = enough_data & ~is_dynamic

    cell_label = np.full((num_rings, num_azimuth_bins), UNKNOWN, dtype=np.int64)
    cell_label[is_static] = STATIC
    cell_label[is_dynamic] = DYNAMIC

    cell_range_std_m = np.where(enough_data, cell_std, np.nan)

    return MotionConsistencyResult(
        cell_label=cell_label,
        cell_range_std_m=cell_range_std_m,
        cell_num_frames_present=num_present.astype(np.int64),
        num_rings=num_rings, num_azimuth_bins=num_azimuth_bins, num_frames_used=n_frames,
        range_std_threshold_m=range_std_threshold_m, min_frames_present=min_frames_present,
    )


def dynamic_point_mask(
    points: np.ndarray,
    result: MotionConsistencyResult,
    vertical_fov_deg: Optional[float] = None,
    treat_unknown_as_dynamic: bool = False,
) -> np.ndarray:
    """
    Map a MotionConsistencyResult's cell-level classification onto an
    individual frame's own points (typically the reference/headline frame
    being evaluated, NOT necessarily one of the frames used to build
    `result`, as long as it's the same sensor/geometry). Returns a
    boolean mask, True where the point falls in a DYNAMIC cell (i.e.
    "exclude this point").

    treat_unknown_as_dynamic: if False (default), UNKNOWN cells are NOT
    excluded -- "we don't have enough data to say" is treated as
    innocent-until-proven-dynamic, so filtering doesn't silently discard
    points just because they sit at the edge of the classification
    window's coverage. Set True for a more conservative filter that also
    excludes anything not positively confirmed static.
    """
    n = points.shape[0]
    if n == 0:
        return np.zeros(0, dtype=bool)

    ring = derive_ring_index(points, num_rings=result.num_rings, vertical_fov_deg=vertical_fov_deg)
    azimuth = compute_azimuth_rad(points)
    azimuth_bin = np.clip(
        np.floor(azimuth / (2.0 * np.pi) * result.num_azimuth_bins).astype(np.int64),
        0, result.num_azimuth_bins - 1,
    )

    valid = ring >= 0
    labels = np.full(n, UNKNOWN, dtype=np.int64)
    labels[valid] = result.cell_label[ring[valid], azimuth_bin[valid]]

    if treat_unknown_as_dynamic:
        return labels != STATIC
    return labels == DYNAMIC


def apply_external_dynamic_mask(points: np.ndarray, dynamic_mask: np.ndarray) -> np.ndarray:
    """
    STEP 8's "1차 구현" in spirit, without bundling a detector: if the
    caller already has a per-point dynamic/static label from their OWN
    object detector or tracker (e.g. projected 2D bounding boxes ->
    per-point membership, or a 3D tracker's own moving-object segmentation),
    this is a one-line pass-through -- kept as a named function (rather
    than callers just indexing with `~dynamic_mask` themselves) so it's
    the same shape of call as classify_points_by_motion_consistency's
    dynamic_point_mask, and so compare_with_without_dynamic_filtering
    below can accept EITHER source uniformly.
    """
    dynamic_mask = np.asarray(dynamic_mask, dtype=bool)
    if dynamic_mask.shape[0] != points.shape[0]:
        raise ValueError(
            f"dynamic_mask length ({dynamic_mask.shape[0]}) must match points length ({points.shape[0]})"
        )
    return dynamic_mask


@dataclass
class DynamicFilteringComparison:
    """
    STEP 8's headline output, matching evaluation_metric_spec.md's own
    example format:

        M2 overall           : 3.1 px
        M2 static only       : 1.2 px
        Dynamic contamination: 38%

    "overall" is M2 computed on every LiDAR edge point (today's default
    behavior, unchanged); "static_only" is M2 computed with dynamic_mask
    applied (moving-object points removed before edge extraction, see
    evaluation.edge_alignment.evaluate_edge_alignment's dynamic_mask
    parameter). dynamic_contamination_ratio is how much of the ORIGINAL
    edge-point set was excluded as dynamic -- the fraction of "why is M2
    worse than it should be" that's attributable to moving objects rather
    than the calibration itself.
    """
    overall_mean_px: float
    overall_classification: str
    overall_num_edge_points: int
    static_only_mean_px: float
    static_only_classification: str
    static_only_num_edge_points: int
    dynamic_contamination_ratio: float
    num_dynamic_points_removed: int


def compare_with_without_dynamic_filtering(
    image: np.ndarray,
    points_lidar: np.ndarray,
    T_CL: np.ndarray,
    camera,
    lidar_spec,
    dynamic_mask: np.ndarray,
    **edge_alignment_kwargs,
) -> DynamicFilteringComparison:
    """
    Run M2 twice -- once on every edge point ("overall"), once with
    dynamic_mask applied ("static only") -- and report the difference.
    This is what actually answers STEP 8's "이 순간부터 '왜 안 맞는지'를
    설명할 수 있습니다" ["from this point on, we can explain WHY it
    doesn't line up"]: a big gap between overall and static-only means
    much of the apparent misalignment was moving objects, not the
    calibration; overall ≈ static-only means the calibration's own error
    is the real story, dynamic contamination isn't meaningfully in play.

    edge_alignment_kwargs are forwarded to BOTH evaluate_edge_alignment
    calls unchanged (e.g. depth_jump_threshold_m, use_correspondence_matching),
    so the two runs are directly comparable -- only dynamic_mask differs
    between them.
    """
    from evaluation.edge_alignment import evaluate_edge_alignment  # local import avoids a cycle at module load

    dynamic_mask = np.asarray(dynamic_mask, dtype=bool)
    if dynamic_mask.shape[0] != points_lidar.shape[0]:
        raise ValueError(
            f"dynamic_mask length ({dynamic_mask.shape[0]}) must match "
            f"points_lidar length ({points_lidar.shape[0]})"
        )

    overall = evaluate_edge_alignment(
        image=image, points_lidar=points_lidar, T_CL=T_CL, camera=camera, lidar_spec=lidar_spec,
        **edge_alignment_kwargs,
    )
    static_only = evaluate_edge_alignment(
        image=image, points_lidar=points_lidar, T_CL=T_CL, camera=camera, lidar_spec=lidar_spec,
        dynamic_mask=dynamic_mask, **edge_alignment_kwargs,
    )

    if overall.num_edge_points > 0:
        contamination = 1.0 - (static_only.num_edge_points / overall.num_edge_points)
    else:
        contamination = float("nan")

    return DynamicFilteringComparison(
        overall_mean_px=overall.mean_px,
        overall_classification=overall.classification,
        overall_num_edge_points=overall.num_edge_points,
        static_only_mean_px=static_only.mean_px,
        static_only_classification=static_only.classification,
        static_only_num_edge_points=static_only.num_edge_points,
        dynamic_contamination_ratio=contamination,
        num_dynamic_points_removed=int(dynamic_mask.sum()),
    )
