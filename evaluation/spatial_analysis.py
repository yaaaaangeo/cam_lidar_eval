"""
evaluation/spatial_analysis.py

STEP 9 -- Depth / Spatial Error Analysis (see evaluation_metric_spec.md's
STEP 9, "이 단계에서부터 단순 평균을 버립니다" -- "from this step on, we
stop relying on a single average").

M2's mean_px is ONE number for the whole frame. That's fine as a pass/fail
gate, but it can't tell "far-range error is fine, it's just noisy up
close" apart from "far-range error is actually bad" -- both can produce
the exact same mean. This module breaks M2's per-point errors down along
two independent axes:

  1. Depth bins: 0-10m, 10-20m, 20-30m, 30-50m, 50m+ (spec's own bin
     edges). A monotonic increase in mean/median error across these bins
     is the spec's own worked example: "원거리에서만 calibration error가
     커진다" ("the error only grows at long range") -- a specific,
     actionable diagnosis a single mean_px can never produce on its own.
  2. Camera region: two SEPARATE partitions of the image, not one 3x3
     grid -- LEFT/CENTER/RIGHT (horizontal thirds) and TOP/CENTER/BOTTOM
     (vertical thirds), each independently telling you whether error
     concentrates on one side (often a rotation-axis clue: worse on
     left+right edges than center suggests yaw; worse top vs bottom
     suggests pitch or a vertical translation offset) without needing a
     denser grid to get that signal.

For each bin/region, five numbers are reported (per the spec): mean,
median, p95, std, and two counts -- valid_count (points that found a
genuine STEP6 correspondence) and failure_count (points that didn't,
i.e. STEP6's unmatched -- penalized at the max search radius, not
excluded, so they still pull mean/std toward "bad" in that bin, which is
correct: a bin where the matcher can't even find correspondences IS a
worse bin, not a bin with less data). Every reported bin also carries the
RAW error array so a caller building further diagnostics doesn't need to
recompute the grouping.

Requires evaluate_edge_alignment to have been run with
use_correspondence_matching=True (the default since STEP6) -- valid vs
failure counts have no meaning under the old pure-nearest-distance mode,
since every point there just gets a distance value.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


DEPTH_BIN_EDGES = [0.0, 10.0, 20.0, 30.0, 50.0, float("inf")]
DEPTH_BIN_LABELS = ["0-10m", "10-20m", "20-30m", "30-50m", "50m+"]

HORIZONTAL_REGIONS = ["LEFT", "CENTER", "RIGHT"]
VERTICAL_REGIONS = ["TOP", "CENTER", "BOTTOM"]


@dataclass
class BinStats:
    """Per-bin/region statistics -- exactly the five numbers STEP9 asks
    for, plus the label and raw data for anything built on top of this."""
    label: str
    mean_px: float
    median_px: float
    p95_px: float
    std_px: float
    valid_count: int
    failure_count: int
    errors_px: np.ndarray = field(repr=False)

    def to_dict(self) -> dict:
        def _safe(x):
            xf = float(x)
            return xf if np.isfinite(xf) else None
        return {
            "label": self.label,
            "mean_px": _safe(self.mean_px),
            "median_px": _safe(self.median_px),
            "p95_px": _safe(self.p95_px),
            "std_px": _safe(self.std_px),
            "valid_count": self.valid_count,
            "failure_count": self.failure_count,
            "total_count": self.valid_count + self.failure_count,
        }


def _bin_stats(label: str, errors: np.ndarray, matched: Optional[np.ndarray]) -> BinStats:
    n = errors.shape[0]
    if n == 0:
        return BinStats(label=label, mean_px=float("nan"), median_px=float("nan"),
                         p95_px=float("nan"), std_px=float("nan"),
                         valid_count=0, failure_count=0, errors_px=errors)
    if matched is not None:
        valid_count = int(matched.sum())
        failure_count = int((~matched).sum())
    else:
        valid_count, failure_count = n, 0
    return BinStats(
        label=label,
        mean_px=float(np.mean(errors)),
        median_px=float(np.median(errors)),
        p95_px=float(np.percentile(errors, 95)),
        std_px=float(np.std(errors)),
        valid_count=valid_count,
        failure_count=failure_count,
        errors_px=errors,
    )


def bin_by_depth(depths_m: np.ndarray) -> list:
    """Assign each depth to a DEPTH_BIN_LABELS index via DEPTH_BIN_EDGES
    (right-open intervals: [0,10), [10,20), ..., [50, inf)). Returns a
    list of bin indices, same length as depths_m."""
    # np.digitize with right=False gives 1-indexed bins matching our
    # edges (excluding the implicit -inf..0 and last..inf sentinels) --
    # subtract 1 and clip so results land in [0, len(labels)-1].
    idx = np.digitize(depths_m, DEPTH_BIN_EDGES[1:-1], right=False)
    return np.clip(idx, 0, len(DEPTH_BIN_LABELS) - 1)


def bin_by_horizontal_region(pixels_u: np.ndarray, image_width: int) -> list:
    """LEFT/CENTER/RIGHT by pixel-u thirds of image_width. Returns a list
    of HORIZONTAL_REGIONS indices, same length as pixels_u."""
    third = image_width / 3.0
    idx = np.clip((pixels_u // third).astype(np.int64), 0, 2)
    return idx


def bin_by_vertical_region(pixels_v: np.ndarray, image_height: int) -> list:
    """TOP/CENTER/BOTTOM by pixel-v thirds of image_height. Returns a
    list of VERTICAL_REGIONS indices, same length as pixels_v."""
    third = image_height / 3.0
    idx = np.clip((pixels_v // third).astype(np.int64), 0, 2)
    return idx


@dataclass
class SpatialAnalysisResult:
    depth_bins: dict            # label -> BinStats, in DEPTH_BIN_LABELS order
    horizontal_regions: dict    # label -> BinStats, in HORIZONTAL_REGIONS order
    vertical_regions: dict      # label -> BinStats, in VERTICAL_REGIONS order
    depth_trend: Optional[str]  # "increases_with_depth" | "stable" | "decreases_with_depth" | None (too little data)

    def to_dict(self) -> dict:
        return {
            "depth_bins": {k: v.to_dict() for k, v in self.depth_bins.items()},
            "horizontal_regions": {k: v.to_dict() for k, v in self.horizontal_regions.items()},
            "vertical_regions": {k: v.to_dict() for k, v in self.vertical_regions.items()},
            "depth_trend": self.depth_trend,
        }


def _detect_depth_trend(depth_bins: dict, min_bins_with_data: int = 3) -> Optional[str]:
    """
    A minimal, defensible version of the spec's own worked diagnosis
    ("원거리에서만 calibration error가 커진다"): checks whether mean_px
    across POPULATED depth bins (in depth order) is monotonically
    non-decreasing (increases_with_depth), monotonically non-increasing
    (decreases_with_depth), or neither (stable -- no clear directional
    trend). Returns None if fewer than min_bins_with_data bins have any
    data at all -- not enough range to say anything about a trend.
    """
    populated = [(label, stats.mean_px) for label, stats in depth_bins.items()
                 if stats.valid_count + stats.failure_count > 0 and np.isfinite(stats.mean_px)]
    if len(populated) < min_bins_with_data:
        return None
    means = [m for _, m in populated]
    diffs = np.diff(means)
    if np.all(diffs >= -1e-9):
        return "increases_with_depth"
    if np.all(diffs <= 1e-9):
        return "decreases_with_depth"
    return "stable"


def analyze_depth_and_spatial(
    edge_point_errors_px: np.ndarray,
    edge_point_depths_m: np.ndarray,
    edge_point_pixels: np.ndarray,
    image_width: int,
    image_height: int,
    edge_point_matched: Optional[np.ndarray] = None,
) -> SpatialAnalysisResult:
    """
    Break down M2's per-point errors by depth bin and by camera region
    (see this module's docstring for the exact bins/regions and what
    "failure_count" means). Takes the raw per-point arrays directly
    (rather than an EdgeAlignmentResult) so it has no import-time
    dependency on evaluation.edge_alignment -- callers typically pass
    result.edge_point_errors_px / .edge_point_depths_m / .edge_point_pixels
    / .edge_point_matched straight through; see
    analyze_depth_and_spatial_from_result for that convenience wrapper.
    """
    errors = np.asarray(edge_point_errors_px, dtype=np.float64)
    depths = np.asarray(edge_point_depths_m, dtype=np.float64)
    pixels = np.asarray(edge_point_pixels, dtype=np.float64)
    matched = np.asarray(edge_point_matched, dtype=bool) if edge_point_matched is not None else None

    depth_idx = bin_by_depth(depths)
    h_idx = bin_by_horizontal_region(pixels[:, 0], image_width)
    v_idx = bin_by_vertical_region(pixels[:, 1], image_height)

    depth_bins = {}
    for i, label in enumerate(DEPTH_BIN_LABELS):
        sel = depth_idx == i
        depth_bins[label] = _bin_stats(label, errors[sel], matched[sel] if matched is not None else None)

    horizontal_regions = {}
    for i, label in enumerate(HORIZONTAL_REGIONS):
        sel = h_idx == i
        horizontal_regions[label] = _bin_stats(label, errors[sel], matched[sel] if matched is not None else None)

    vertical_regions = {}
    for i, label in enumerate(VERTICAL_REGIONS):
        sel = v_idx == i
        vertical_regions[label] = _bin_stats(label, errors[sel], matched[sel] if matched is not None else None)

    return SpatialAnalysisResult(
        depth_bins=depth_bins,
        horizontal_regions=horizontal_regions,
        vertical_regions=vertical_regions,
        depth_trend=_detect_depth_trend(depth_bins),
    )


def analyze_depth_and_spatial_from_result(edge_alignment_result, image_width: int, image_height: int) -> Optional[SpatialAnalysisResult]:
    """
    Convenience wrapper taking an EdgeAlignmentResult directly (as
    returned by evaluation.edge_alignment.evaluate_edge_alignment).
    Returns None if the result FAILed or lacks the needed per-point
    arrays (e.g. it predates STEP7's edge_point_depths_m).
    """
    if (edge_alignment_result.classification == "FAIL"
            or edge_alignment_result.edge_point_errors_px is None
            or edge_alignment_result.edge_point_depths_m is None
            or edge_alignment_result.edge_point_pixels is None):
        return None
    return analyze_depth_and_spatial(
        edge_alignment_result.edge_point_errors_px,
        edge_alignment_result.edge_point_depths_m,
        edge_alignment_result.edge_point_pixels,
        image_width, image_height,
        edge_point_matched=edge_alignment_result.edge_point_matched,
    )
