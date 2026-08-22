"""
evaluation/holdout_consistency.py

M3. Hold-out Consistency (see evaluation_metric_spec.md v0.4).

Measures whether a FIXED, existing T_CL performs consistently across
different contiguous time blocks of the dataset -- i.e. whether the
calibration is generalizing across the whole sequence rather than being
"accidentally okay" only in the scene/time window it happens to fit.

Pipeline:
  1. Split the synced dataset into N contiguous time blocks
     (EvaluationDataset.time_blocks -- no random shuffling, per spec).
  2. For each block, run M2 (edge_alignment) independently per frame,
     pool all edge-point errors across the block's frames, and compute one
     aggregate M2 result for that block.
  3. Collect each block's mean_px into a distribution across blocks; compute
     Mean/STD/range across blocks.
  4. Classify STD against the sensor-relative floor(Z), using the STD
     multiplier scheme (1x / 3x, per spec) -- NOT the M2 2x/5x scheme,
     since this is measuring spread, not per-point offset.

STEP10 -- scene metadata + instability diagnosis (see
evaluation_metric_spec.md's STEP10, "M3 BAD가 아니라 M3 instability /
Cause: Long-range scenes 처럼 설명할 수 있게 합니다"): a bare "M3 is
WARNING/BAD" tells you THAT the calibration doesn't generalize across time
blocks, but not WHY -- was one block just further away, sparser, or
covering less of the frame than the others? Each BlockResult now also
records:
  - representative_depth_m (already existed)
  - edge_density: average M2 edge points per valid frame in the block
  - num_points_avg: average LiDAR points projected per valid frame
  - fov_coverage: fraction of the image's area spanned by this block's
    pooled edge points' bounding box (a cheap proxy for "did this block's
    scene actually exercise most of the frame, or just a small patch")
  - dynamic_ratio: optional (None unless the caller supplies per-frame
    dynamic masks -- see evaluate_holdout_consistency's dynamic_masks
    parameter); fraction of pooled points flagged as dynamic (STEP8)

diagnose_instability(...) then compares the WORST block's scene metadata
against the median of the other blocks' and reports which metric(s)
deviate the most, in plain language ("Long-range scenes", "Sparse edge
structure", etc) -- a specific, actionable candidate cause, not just a
worse number.

Failure conditions (per spec):
  - fewer than 3 valid blocks (statistically meaningless)
  - a block with fewer frames than min_frames_per_block is excluded
    (with a warning) rather than silently included
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from evaluation.edge_alignment import evaluate_edge_alignment, EdgeAlignmentResult
from input.camera import CameraModel
from input.dataset import EvaluationDataset, SyncedFrame
from quality.noise_floor import (
    LidarSensorSpecForFloor,
    resolve_floor_inputs,
    compute_floor,
    classify,
    STD_GOOD_MULTIPLIER,
    STD_WARNING_MULTIPLIER,
    M2_GOOD_MULTIPLIER,
    M2_WARNING_MULTIPLIER,
)


MIN_VALID_BLOCKS = 3
DIAGNOSIS_RELATIVE_DIFF_THRESHOLD = 0.5  # 50% -- how far a metric must deviate from the other blocks' median to be named as a candidate cause

# (metric_name, "above median" label, "below median" label)
_DIAGNOSIS_METRICS = [
    ("representative_depth_m", "Long-range scenes", "Short-range scenes"),
    ("edge_density", "Dense edge structure", "Sparse edge structure"),
    ("num_points_avg", "Dense point cloud", "Sparse point cloud"),
    ("fov_coverage", "Wide FOV coverage", "Narrow FOV coverage"),
]


@dataclass
class BlockResult:
    block_index: int
    frame_indices: list[int]
    num_frames_total: int
    num_frames_valid: int   # frames where per-frame M2 did not FAIL
    num_frames_failed: int
    mean_px: float
    median_px: float
    p95_px: float
    num_edge_points: int
    representative_depth_m: float
    floor_px: float
    classification: str    # GOOD | WARNING | BAD | EXCLUDED | FAIL
    warnings: list[str] = field(default_factory=list)

    # STEP10 -- scene metadata (see module docstring)
    edge_density: float = float("nan")
    num_points_avg: float = float("nan")
    fov_coverage: float = float("nan")
    dynamic_ratio: Optional[float] = None


@dataclass
class HoldoutConsistencyResult:
    block_results: list[BlockResult]
    num_valid_blocks: int
    block_means_px: list[float]
    mean_across_blocks_px: float
    std_across_blocks_px: float
    range_px: float
    floor_px: float
    classification: str    # GOOD | WARNING | BAD | FAIL
    warnings: list[str] = field(default_factory=list)
    instability_diagnosis: Optional[dict] = None  # STEP10 -- see diagnose_instability


def diagnose_instability(
    block_results: list["BlockResult"],
    relative_diff_threshold: float = DIAGNOSIS_RELATIVE_DIFF_THRESHOLD,
) -> Optional[dict]:
    """
    STEP10: compare the block with the WORST mean_px against the median
    of the OTHER valid blocks' own scene metadata, and report which
    metric(s) deviate by more than relative_diff_threshold (default 50%)
    as candidate explanations -- e.g. "this block's error is worse, AND
    its representative depth is 2x the other blocks' -- long-range scenes
    are a plausible cause", not just "this block is worse".

    Returns None if there are fewer than MIN_VALID_BLOCKS valid blocks
    (not enough to compare against) -- same threshold
    evaluate_holdout_consistency itself uses to decide M3 is meaningful
    at all. Always computed when there's enough data, regardless of the
    overall classification -- a caller (e.g. the report layer) can choose
    to only surface it when M3 isn't GOOD.
    """
    valid_blocks = [b for b in block_results if b.classification in ("GOOD", "WARNING", "BAD")]
    if len(valid_blocks) < MIN_VALID_BLOCKS:
        return None

    worst_block = max(valid_blocks, key=lambda b: b.mean_px)
    others = [b for b in valid_blocks if b.block_index != worst_block.block_index]
    if not others:
        return None

    candidates = []
    for metric_name, above_label, below_label in _DIAGNOSIS_METRICS:
        worst_value = getattr(worst_block, metric_name)
        other_values = [getattr(b, metric_name) for b in others]
        other_values = [v for v in other_values if np.isfinite(v)]
        if not np.isfinite(worst_value) or not other_values:
            continue
        baseline = float(np.median(other_values))
        if abs(baseline) < 1e-9:
            continue  # can't compute a meaningful relative difference against ~0
        relative_diff = (worst_value - baseline) / abs(baseline)
        if abs(relative_diff) < relative_diff_threshold:
            continue
        candidates.append({
            "metric": metric_name,
            "worst_block_value": worst_value,
            "other_blocks_median": baseline,
            "relative_diff": relative_diff,
            "explanation": above_label if relative_diff > 0 else below_label,
        })

    candidates.sort(key=lambda c: abs(c["relative_diff"]), reverse=True)
    return {
        "worst_block_index": worst_block.block_index,
        "worst_block_mean_px": worst_block.mean_px,
        "candidates": candidates,
    }


def _fov_coverage(edge_pixels: Optional[np.ndarray], image_width: int, image_height: int) -> float:
    """Fraction of the image's area spanned by edge_pixels' bounding box
    -- a cheap, dependency-free proxy for 'did this block's scene actually
    exercise most of the frame, or just a small patch of it'. NaN if
    there are no edge pixels to measure."""
    if edge_pixels is None or edge_pixels.shape[0] == 0 or image_width <= 0 or image_height <= 0:
        return float("nan")
    u_span = float(edge_pixels[:, 0].max() - edge_pixels[:, 0].min())
    v_span = float(edge_pixels[:, 1].max() - edge_pixels[:, 1].min())
    return (u_span * v_span) / (image_width * image_height)


def _evaluate_block(
    block_index: int,
    frames: list[SyncedFrame],
    T_CL: np.ndarray,
    camera: CameraModel,
    lidar_spec: LidarSensorSpecForFloor,
    min_frames_per_block: int,
    edge_alignment_kwargs: dict,
    dynamic_masks: Optional[dict] = None,
) -> BlockResult:
    frame_indices = [f.index for f in frames]

    if len(frames) < min_frames_per_block:
        return BlockResult(
            block_index=block_index,
            frame_indices=frame_indices,
            num_frames_total=len(frames),
            num_frames_valid=0,
            num_frames_failed=0,
            mean_px=float("nan"), median_px=float("nan"), p95_px=float("nan"),
            num_edge_points=0, representative_depth_m=float("nan"), floor_px=float("nan"),
            classification="EXCLUDED",
            warnings=[
                f"Block {block_index} has {len(frames)} frames, below "
                f"min_frames_per_block={min_frames_per_block}; excluded from aggregation."
            ],
        )

    per_frame_results: list[EdgeAlignmentResult] = []
    for sf in frames:
        image = sf.camera_frame.load()
        points = sf.lidar_frame.load()
        frame_kwargs = dict(edge_alignment_kwargs)
        if dynamic_masks is not None and sf.index in dynamic_masks:
            frame_kwargs["dynamic_mask"] = dynamic_masks[sf.index]
        result = evaluate_edge_alignment(
            image=image, points_lidar=points, T_CL=T_CL,
            camera=camera, lidar_spec=lidar_spec, **frame_kwargs,
        )
        per_frame_results.append(result)

    valid_results = [r for r in per_frame_results if r.classification != "FAIL"]
    num_failed = len(per_frame_results) - len(valid_results)

    if not valid_results:
        return BlockResult(
            block_index=block_index,
            frame_indices=frame_indices,
            num_frames_total=len(frames),
            num_frames_valid=0,
            num_frames_failed=num_failed,
            mean_px=float("nan"), median_px=float("nan"), p95_px=float("nan"),
            num_edge_points=0, representative_depth_m=float("nan"), floor_px=float("nan"),
            classification="FAIL",
            warnings=[f"Block {block_index}: all {len(frames)} frames failed M2 evaluation."],
        )

    pooled_errors = np.concatenate([r.edge_point_errors_px for r in valid_results])
    pooled_depth = float(np.median([r.representative_depth_m for r in valid_results]))
    num_edge_points = int(sum(r.num_edge_points for r in valid_results))
    edge_density = num_edge_points / len(valid_results)
    num_points_avg = float(np.mean([r.num_projected_points for r in valid_results]))

    pooled_pixels = np.concatenate([r.edge_point_pixels for r in valid_results if r.edge_point_pixels is not None])
    fov_coverage = _fov_coverage(pooled_pixels, camera.width, camera.height)

    dynamic_ratio = None
    if dynamic_masks is not None:
        block_frame_indices_with_masks = [sf.index for sf in frames if sf.index in dynamic_masks]
        if block_frame_indices_with_masks:
            total_pts = sum(dynamic_masks[idx].shape[0] for idx in block_frame_indices_with_masks)
            dynamic_pts = sum(int(dynamic_masks[idx].sum()) for idx in block_frame_indices_with_masks)
            dynamic_ratio = dynamic_pts / total_pts if total_pts > 0 else None

    floor_inputs = resolve_floor_inputs(
        fx_px=camera.intrinsics.fx, T_CL=T_CL, lidar_spec=lidar_spec,
        edge_localization_floor_px=camera.edge_localization_floor_px,
    )
    floor_px = compute_floor(floor_inputs, pooled_depth)

    mean_px = float(np.mean(pooled_errors))
    block_classification = classify(mean_px, floor_px, M2_GOOD_MULTIPLIER, M2_WARNING_MULTIPLIER)

    warnings = list(floor_inputs.fallback_warnings)
    if num_failed > 0:
        warnings.append(f"Block {block_index}: {num_failed}/{len(frames)} frames failed M2 and were excluded from pooling.")

    return BlockResult(
        block_index=block_index,
        frame_indices=frame_indices,
        num_frames_total=len(frames),
        num_frames_valid=len(valid_results),
        num_frames_failed=num_failed,
        mean_px=mean_px,
        median_px=float(np.median(pooled_errors)),
        p95_px=float(np.percentile(pooled_errors, 95)),
        num_edge_points=num_edge_points,
        representative_depth_m=pooled_depth,
        floor_px=floor_px,
        classification=block_classification,
        warnings=warnings,
        edge_density=edge_density,
        num_points_avg=num_points_avg,
        fov_coverage=fov_coverage,
        dynamic_ratio=dynamic_ratio,
    )


def evaluate_holdout_consistency(
    dataset: EvaluationDataset,
    lidar_spec: LidarSensorSpecForFloor,
    n_blocks: int = 4,
    min_frames_per_block: int = 30,
    edge_alignment_kwargs: Optional[dict] = None,
    dynamic_masks: Optional[dict] = None,
) -> HoldoutConsistencyResult:
    """
    Compute M3 Hold-out Consistency: split dataset.frames into n_blocks
    contiguous time blocks, run M2 independently (pooled) per block, and
    report the Mean/STD/range of block-level error across blocks.

    dataset.extrinsic.T_CL is treated as fixed and applied identically to
    every block -- this is the whole point of M3 (does the SAME calibration
    generalize, not "what's the best T for each block").

    dynamic_masks: STEP10/STEP8 -- optional {frame_index: dynamic_mask}
    dict (dynamic_mask a boolean array aligned with that frame's LiDAR
    points, per evaluation.dynamic_filter). If given, frames present in
    this dict have their dynamic points excluded from M2 (via
    evaluate_edge_alignment's dynamic_mask parameter) AND contribute to
    each block's dynamic_ratio scene metadata; frames not present are
    evaluated unfiltered and don't contribute to dynamic_ratio. None
    (default) skips dynamic filtering entirely and leaves dynamic_ratio
    as None on every block.
    """
    edge_alignment_kwargs = edge_alignment_kwargs or {}
    warnings: list[str] = []

    raw_blocks = dataset.time_blocks(n_blocks)

    block_results = [
        _evaluate_block(
            block_index=i, frames=block, T_CL=dataset.extrinsic.T_CL,
            camera=dataset.camera, lidar_spec=lidar_spec,
            min_frames_per_block=min_frames_per_block,
            edge_alignment_kwargs=edge_alignment_kwargs,
            dynamic_masks=dynamic_masks,
        )
        for i, block in enumerate(raw_blocks)
    ]

    for b in block_results:
        warnings.extend(f"[block {b.block_index}] {w}" for w in b.warnings)

    valid_blocks = [b for b in block_results if b.classification in ("GOOD", "WARNING", "BAD")]

    if len(valid_blocks) < MIN_VALID_BLOCKS:
        warnings.append(
            f"Only {len(valid_blocks)} valid block(s) (need >= {MIN_VALID_BLOCKS}) -- "
            f"Hold-out Consistency is not statistically meaningful with this few blocks. "
            f"Consider a longer dataset, fewer n_blocks, or a smaller min_frames_per_block."
        )
        return HoldoutConsistencyResult(
            block_results=block_results,
            num_valid_blocks=len(valid_blocks),
            block_means_px=[b.mean_px for b in valid_blocks],
            mean_across_blocks_px=float("nan"),
            std_across_blocks_px=float("nan"),
            range_px=float("nan"),
            floor_px=float("nan"),
            classification="FAIL",
            warnings=warnings,
        )

    block_means = [b.mean_px for b in valid_blocks]
    mean_across = float(np.mean(block_means))
    std_across = float(np.std(block_means, ddof=1))  # sample std across blocks
    range_across = float(np.max(block_means) - np.min(block_means))

    # Representative floor for STD classification: median of the valid
    # blocks' own floor(Z) values (each already reflects that block's
    # representative depth).
    floor_px = float(np.median([b.floor_px for b in valid_blocks]))

    overall_classification = classify(std_across, floor_px, STD_GOOD_MULTIPLIER, STD_WARNING_MULTIPLIER)

    instability_diagnosis = diagnose_instability(block_results)
    if instability_diagnosis is not None and instability_diagnosis["candidates"] and overall_classification != "GOOD":
        top = instability_diagnosis["candidates"][0]
        warnings.append(
            f"Block {instability_diagnosis['worst_block_index']} has the highest error "
            f"({instability_diagnosis['worst_block_mean_px']:.2f}px) and its {top['metric']} "
            f"differs from other blocks by {top['relative_diff']:+.0%} -- possible cause: {top['explanation']}."
        )

    return HoldoutConsistencyResult(
        block_results=block_results,
        num_valid_blocks=len(valid_blocks),
        block_means_px=block_means,
        mean_across_blocks_px=mean_across,
        std_across_blocks_px=std_across,
        range_px=range_across,
        floor_px=floor_px,
        classification=overall_classification,
        warnings=warnings,
        instability_diagnosis=instability_diagnosis,
    )
