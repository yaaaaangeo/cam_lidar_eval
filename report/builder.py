"""
report/builder.py

Assembles the plain-dict "report" structure consumed by report/json.py and
report/html.py, from the dataclass results produced by evaluation/*.py and
quality/quality_score.py.

Deliberately builds a plain dict (not dataclasses-all-the-way-down) because
this IS the JSON/HTML serialization boundary -- one place that decides
exactly what's included, in what shape, and sanitizes non-JSON-safe values
(NaN/Inf -> null, numpy scalars -> python scalars). Per-point visualization
arrays (EdgeAlignmentResult.edge_point_pixels / edge_point_errors_px) are
intentionally excluded from the report; those belong to
visualization/overlay.py (not yet implemented) since a report reader wants
summary statistics, not raw per-point arrays.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional

import math
import numpy as np

from evaluation.edge_alignment import EdgeAlignmentResult
from evaluation.holdout_consistency import HoldoutConsistencyResult
from evaluation.multiframe_consistency import MultiFrameConsistencyResult
from input.dataset import EvaluationDataset
from quality.quality_score import QualityScoreResult


# Kept in sync with pyproject.toml's [project].version by hand (no
# single-source-of-truth mechanism yet -- see CONTRIBUTING.md's release
# checklist). Bump this whenever pyproject.toml's version changes.
TOOL_VERSION = "0.2.0"


def _num(x: Any) -> Optional[float]:
    """Sanitize a numeric value for JSON: NaN/Inf -> None, numpy scalar ->
    python float, everything else passed through math.isfinite's gate."""
    if x is None:
        return None
    if isinstance(x, (np.floating, np.integer)):
        x = x.item()
    if isinstance(x, bool):
        return x
    if isinstance(x, (int, float)):
        if isinstance(x, float) and not math.isfinite(x):
            return None
        return x
    return x


def _matrix_to_list(T: np.ndarray) -> list[list[float]]:
    return [[float(v) for v in row] for row in np.asarray(T, dtype=float)]


# ---------------------------------------------------------------------------
# Per-metric summary dicts
# ---------------------------------------------------------------------------

def spatial_analysis_summary(analysis) -> Optional[dict]:
    """
    STEP 9 -- Depth/Spatial Error Analysis report section. Takes an
    already-computed evaluation.spatial_analysis.SpatialAnalysisResult
    (or None) -- build_report computes it once and shares it with
    STEP12's root_cause_summary below, rather than each recomputing it
    independently.
    """
    if analysis is None:
        return None
    return analysis.to_dict()


def m2_summary(m2: EdgeAlignmentResult) -> dict:
    return {
        "metric": "M2",
        "name": "Edge Alignment (Projection Consistency)",
        "classification": m2.classification,
        "mean_px": _num(m2.mean_px),
        "median_px": _num(m2.median_px),
        "p95_px": _num(m2.p95_px),
        "max_px": _num(m2.max_px),
        "num_edge_points": m2.num_edge_points,
        "num_projected_points": m2.num_projected_points,
        "representative_depth_m": _num(m2.representative_depth_m),
        "floor_px": _num(m2.floor_px),
        "warnings": list(m2.warnings),
        # STEP6 -- correspondence-matching diagnostics. None (all three)
        # when use_correspondence_matching=False.
        "num_matched": m2.num_matched,
        "num_unmatched": m2.num_unmatched,
        "match_rate": _num(m2.match_rate),
        # STEP7 -- Noise/Uncertainty Model summary stats. Per-point arrays
        # (edge_point_floor_px, edge_point_normalized_errors,
        # edge_point_depths_m) are intentionally NOT included here, same
        # reasoning as edge_point_pixels/edge_point_errors_px above: a
        # report reader wants summary statistics, not raw per-point
        # arrays -- those belong to visualization.uncertainty_plot.
        "mean_normalized_error": _num(m2.mean_normalized_error),
        "median_normalized_error": _num(m2.median_normalized_error),
        "p95_normalized_error": _num(m2.p95_normalized_error),
    }


def m3_summary(m3: HoldoutConsistencyResult) -> dict:
    return {
        "metric": "M3",
        "name": "Hold-out Consistency (Generalization)",
        "classification": m3.classification,
        "num_valid_blocks": m3.num_valid_blocks,
        "mean_across_blocks_px": _num(m3.mean_across_blocks_px),
        "std_across_blocks_px": _num(m3.std_across_blocks_px),
        "range_px": _num(m3.range_px),
        "floor_px": _num(m3.floor_px),
        "blocks": [
            {
                "block_index": b.block_index,
                "num_frames_total": b.num_frames_total,
                "num_frames_valid": b.num_frames_valid,
                "mean_px": _num(b.mean_px),
                "median_px": _num(b.median_px),
                "p95_px": _num(b.p95_px),
                "classification": b.classification,
                # STEP10 -- scene metadata (see evaluation/holdout_consistency.py)
                "representative_depth_m": _num(b.representative_depth_m),
                "edge_density": _num(b.edge_density),
                "num_points_avg": _num(b.num_points_avg),
                "fov_coverage": _num(b.fov_coverage),
                "dynamic_ratio": _num(b.dynamic_ratio) if b.dynamic_ratio is not None else None,
            }
            for b in m3.block_results
        ],
        # STEP10 -- instability diagnosis (see evaluation.holdout_consistency.diagnose_instability)
        "instability_diagnosis": m3.instability_diagnosis,
        "warnings": list(m3.warnings),
    }


def m4_summary(m4: MultiFrameConsistencyResult) -> dict:
    return {
        "metric": "M4",
        "name": "Multi-frame Consistency (Stability)",
        "classification": m4.classification,
        "num_frames_total": m4.num_frames_total,
        "num_valid_frames": m4.num_valid_frames,
        "num_failed_frames": m4.num_failed_frames,
        "num_outlier_frames": m4.num_outlier_frames,
        "outlier_frame_indices": list(m4.outlier_frame_indices),
        "mean_across_frames_px": _num(m4.mean_across_frames_px),
        "std_across_frames_px": _num(m4.std_across_frames_px),
        "median_across_frames_px": _num(m4.median_across_frames_px),
        "p95_across_frames_px": _num(m4.p95_across_frames_px),
        "max_across_frames_px": _num(m4.max_across_frames_px),
        "floor_px": _num(m4.floor_px),
        # STEP10 -- robust statistics + separated ratios (see
        # evaluation/multiframe_consistency.py's module docstring)
        "outlier_method": m4.outlier_method,
        "mad_px": _num(m4.mad_px),
        "iqr_px": _num(m4.iqr_px),
        "q1_px": _num(m4.q1_px),
        "q3_px": _num(m4.q3_px),
        "valid_ratio": _num(m4.valid_ratio),
        "failure_ratio": _num(m4.failure_ratio),
        "outlier_ratio": _num(m4.outlier_ratio),
        # per-frame trajectory, trimmed to the essentials needed for a
        # trend chart (frame_index, mean_px, is_outlier) -- omits per-frame
        # warnings/edge-point counts to keep this from ballooning on long
        # sequences.
        "frame_trajectory": [
            {
                "frame_index": f.frame_index,
                "timestamp": f.timestamp,
                "mean_px": _num(f.mean_px),
                "classification": f.classification,
                "is_outlier": f.is_outlier,
                "robust_z_score": _num(f.robust_z_score),
            }
            for f in m4.frame_results
        ],
        "warnings": list(m4.warnings),
    }


def quality_summary(q: QualityScoreResult) -> dict:
    return {
        "overall_score": _num(q.overall_score),
        "overall_classification": q.overall_classification,
        "num_valid_categories": q.num_valid_categories,
        "weights_used": q.weights_used,
        "categories": [
            {
                "name": c.name,
                "metric": c.metric_name,
                "valid": c.valid,
                "score": _num(c.score),
                "classification": c.classification,
                "raw_value_px": _num(c.raw_value_px),
                "floor_px": _num(c.floor_px),
                "summary": {k: _num(v) if isinstance(v, (int, float, np.floating, np.integer)) else v
                            for k, v in c.summary.items()},
            }
            for c in q.categories
        ],
        "warnings": list(q.warnings),
    }


def m0_summary(m0: Optional[dict]) -> Optional[dict]:
    """
    Placeholder passthrough for the M0 Sanity Gate result. m0 is expected
    to already be a plain dict (see evaluation/sanity_gate.py) -- kept as
    an Optional pass-through here so build_report doesn't need to change
    once M0 exists; None simply omits the section.
    """
    return m0


def synchronization_summary(dataset: EvaluationDataset) -> Optional[dict]:
    """
    STEP 2 -- Timestamp Synchronization report section (input/dataset.py's
    SyncStats). None if sync never ran (shouldn't normally happen once
    build_dataset has been called, but mirrors m0_summary's Optional
    passthrough pattern for consistency/robustness).
    """
    if dataset.sync_stats is None:
        return None
    return dataset.sync_stats.to_dict()


def motion_deskew_summary(deskew_compare: Optional[dict]) -> Optional[dict]:
    """
    STEP 5 -- Motion Deskew report section. deskew_compare is expected to
    already be a plain dict (motion.deskew.compare_before_after's output)
    -- opt-in, so None (the default) simply omits the section, matching
    input_validation/synchronization's Optional passthrough pattern.
    Deskewing needs an external platform-velocity input this tool has no
    way to measure on its own (see motion/deskew.py's module docstring),
    so unlike STEP1/STEP2 it's never computed unless the caller (app.cli's
    --deskew-* flags) explicitly supplies one.
    """
    return deskew_compare


def dynamic_filter_summary(comparison) -> Optional[dict]:
    """
    STEP 8 -- Dynamic Object Filtering report section. `comparison` is an
    evaluation.dynamic_filter.DynamicFilteringComparison (or None). Opt-in
    like STEP5's deskew -- needs a multi-frame window (for the
    motion-consistency classifier) or an externally-supplied mask,
    neither of which is assumed by default, so None simply omits the
    section.
    """
    if comparison is None:
        return None
    return {
        "overall_mean_px": _num(comparison.overall_mean_px),
        "overall_classification": comparison.overall_classification,
        "overall_num_edge_points": comparison.overall_num_edge_points,
        "static_only_mean_px": _num(comparison.static_only_mean_px),
        "static_only_classification": comparison.static_only_classification,
        "static_only_num_edge_points": comparison.static_only_num_edge_points,
        "dynamic_contamination_ratio": _num(comparison.dynamic_contamination_ratio),
        "num_dynamic_points_removed": comparison.num_dynamic_points_removed,
    }


def plane_consistency_summary(result) -> dict:
    return {
        "metric": "Plane Consistency",
        "classification": result.classification,
        "plane_found": result.plane_found,
        "inlier_ratio": _num(result.inlier_ratio),
        "num_inliers": result.num_inliers,
        "num_boundary_points": result.num_boundary_points,
        "mean_px": _num(result.mean_px),
        "median_px": _num(result.median_px),
        "p95_px": _num(result.p95_px),
        "floor_px": _num(result.floor_px),
        "warnings": list(result.warnings),
    }


def perturbation_summary(result) -> dict:
    return {
        "metric": "Perturbation Sensitivity",
        "classification": result.classification,
        "baseline_mean_px": _num(result.baseline_mean_px),
        "is_local_minimum": result.is_local_minimum,
        "improvement_margin_px": _num(result.improvement_margin_px),
        "best_sample": (
            {
                "axis": result.best_sample.axis,
                "direction": result.best_sample.direction,
                "delta": result.best_sample.delta,
                "mean_px": _num(result.best_sample.mean_px),
            } if result.best_sample else None
        ),
        "num_samples": len(result.samples),
        "warnings": list(result.warnings),
        # STEP11 -- per-axis sensitivity ranking (see evaluation/perturbation.py)
        "axis_sensitivities": [
            {
                "axis": a.axis,
                "classification": a.classification,
                "small_delta_effect_px": _num(a.small_delta_effect_px),
                "large_delta_effect_px": _num(a.large_delta_effect_px),
            }
            for a in result.axis_sensitivities
        ],
        "timestamp_sensitivity_computed": result.timestamp_sensitivity_computed,
    }


def temporal_drift_summary(result) -> dict:
    return {
        "metric": "Temporal Drift",
        "classification": result.classification,
        "slope_px_per_frame": _num(result.slope_px_per_frame),
        "r_value": _num(result.r_value),
        "p_value": _num(result.p_value),
        "is_statistically_significant": result.is_statistically_significant,
        "total_drift_px": _num(result.total_drift_px),
        "floor_px": _num(result.floor_px),
        "num_frames_used": result.num_frames_used,
        "warnings": list(result.warnings),
    }


def advanced_summary(
    plane_result=None,
    perturbation_result=None,
    temporal_drift_result=None,
) -> Optional[dict]:
    """
    Bundle whichever advanced (Phase-5) metrics were actually run into one
    'advanced' report section. Returns None if none were provided, so
    build_report can omit the section entirely for MVP-only runs.
    """
    if plane_result is None and perturbation_result is None and temporal_drift_result is None:
        return None
    return {
        "plane_consistency": plane_consistency_summary(plane_result) if plane_result is not None else None,
        "perturbation": perturbation_summary(perturbation_result) if perturbation_result is not None else None,
        "temporal_drift": temporal_drift_summary(temporal_drift_result) if temporal_drift_result is not None else None,
    }


# ---------------------------------------------------------------------------
# Top-level report assembly
# ---------------------------------------------------------------------------

def build_report(
    dataset: EvaluationDataset,
    m2_result: EdgeAlignmentResult,
    m3_result: HoldoutConsistencyResult,
    m4_result: MultiFrameConsistencyResult,
    quality_result: QualityScoreResult,
    m0_result: Optional[dict] = None,
    n_blocks: Optional[int] = None,
    min_frames_m4: Optional[int] = None,
    extra_warnings: Optional[list[str]] = None,
    plane_result=None,
    perturbation_result=None,
    temporal_drift_result=None,
    deskew_compare: Optional[dict] = None,
    dynamic_filter_comparison=None,
) -> dict:
    """
    Assemble the full report dict. This is the single source of truth for
    "what goes in a report" -- report/json.py and report/html.py both
    consume this same structure so they can never drift out of sync with
    each other.

    plane_result / perturbation_result / temporal_drift_result are optional
    Phase-5 "advanced" metrics (evaluation/plane_consistency.py,
    evaluation/perturbation.py, evaluation/temporal_drift.py). None of them
    contribute to quality_score -- they're supplementary diagnostics, not
    part of the MVP scored set -- and are simply omitted from the report
    if not provided.

    deskew_compare: STEP5 -- motion.deskew.compare_before_after(...)'s
    output dict, if the caller ran deskewing (opt-in; see app.cli's
    --deskew-* flags). None (default) omits the "motion_deskew" section.

    dynamic_filter_comparison: STEP8 -- an
    evaluation.dynamic_filter.DynamicFilteringComparison, if the caller
    ran the dynamic-object-filtering comparison (opt-in; see app.cli's
    --dynamic-filter flag). None (default) omits the "dynamic_filter"
    section.
    """
    generated_at = datetime.now(timezone.utc).isoformat()

    metadata = {
        "tool_version": TOOL_VERSION,
        "generated_at": generated_at,
        "camera": {
            "model": dataset.camera.model,
            "width": dataset.camera.width,
            "height": dataset.camera.height,
            "fx": dataset.camera.intrinsics.fx,
            "fy": dataset.camera.intrinsics.fy,
            "cx": dataset.camera.intrinsics.cx,
            "cy": dataset.camera.intrinsics.cy,
            "distortion_model": dataset.camera.distortion.model,
        },
        "lidar": {
            "source_kind": dataset.lidar.source.kind,
        },
        "extrinsic": {
            "parent": dataset.extrinsic.parent,
            "child": dataset.extrinsic.child,
            "T_CL": _matrix_to_list(dataset.extrinsic.T_CL),
            "baseline_m": float(np.linalg.norm(dataset.extrinsic.T_CL[:3, 3])),
        },
        "dataset": {
            "num_synced_frames": len(dataset.frames),
            "sync_max_time_diff_ms": dataset.sync_config.max_time_diff_ms,
            "n_blocks_for_m3": n_blocks,
            "min_frames_for_m4": min_frames_m4,
        },
    }

    dataset_warnings = list(dataset.warnings) if dataset.warnings else []

    from evaluation.spatial_analysis import analyze_depth_and_spatial_from_result
    spatial_analysis_result = analyze_depth_and_spatial_from_result(
        m2_result, dataset.camera.width, dataset.camera.height,
    )

    from evaluation.root_cause import diagnose_root_cause
    root_cause_result = diagnose_root_cause(
        sync_stats=dataset.sync_stats, m2=m2_result, m3=m3_result, m4=m4_result,
        spatial_analysis=spatial_analysis_result,
        dynamic_filter_comparison=dynamic_filter_comparison,
        perturbation_result=perturbation_result,
    )

    from quality.confidence_coverage import compute_quality_confidence_coverage
    confidence_coverage_result = compute_quality_confidence_coverage(
        quality_result, sync_stats=dataset.sync_stats, m2=m2_result, m3=m3_result, m4=m4_result,
        spatial_analysis=spatial_analysis_result, input_validation=dataset.input_validation,
        n_blocks=n_blocks,
    )

    report = {
        "metadata": metadata,
        "input_validation": dataset.input_validation,
        "synchronization": synchronization_summary(dataset),
        "motion_deskew": motion_deskew_summary(deskew_compare),
        "dynamic_filter": dynamic_filter_summary(dynamic_filter_comparison),
        "m0_sanity_gate": m0_summary(m0_result),
        "m2_edge_alignment": m2_summary(m2_result),
        "m2_spatial_analysis": spatial_analysis_summary(spatial_analysis_result),
        "m3_holdout_consistency": m3_summary(m3_result),
        "m4_multiframe_consistency": m4_summary(m4_result),
        "quality_score": quality_summary(quality_result),
        "advanced": advanced_summary(plane_result, perturbation_result, temporal_drift_result),
        "root_cause_diagnosis": root_cause_result.to_dict(),
        "quality_confidence_coverage": confidence_coverage_result.to_dict(),
        "warnings": dataset_warnings + list(extra_warnings or []),
    }
    return report
