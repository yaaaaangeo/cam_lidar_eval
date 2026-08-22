"""
visualization/uncertainty_plot.py

STEP 7 -- Noise/Uncertainty Model visualization: a scatter of each M2 edge
point's actual pixel error against its own depth, with the sensor-relative
floor(Z) curve (quality.noise_floor.compute_floor_array) drawn on top as a
reference. This is the direct visual answer to STEP 7's core question --
"is this error bigger than sensor noise would explain AT THIS POINT'S OWN
RANGE" -- made literal: points sitting ON or BELOW the floor(Z) curve are
consistent with ordinary sensor noise (GOOD, however large their raw pixel
error looks in isolation); points sitting well ABOVE the curve are the ones
actually worth worrying about, regardless of their absolute distance from
the sensor.

Distinct from visualization/histogram.py's error histogram (which shows
the DISTRIBUTION of raw errors against one frame-representative floor(Z)
value) -- this view is depth-resolved, showing WHERE ALONG THE DEPTH AXIS
the error sits relative to what's expected there, not just how big it is
overall.
"""

from __future__ import annotations

from io import BytesIO
from typing import Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless: no display server needed, safe for CLI/report generation
import matplotlib.pyplot as plt

from quality.noise_floor import FloorInputs, compute_floor_array, M2_GOOD_MULTIPLIER, M2_WARNING_MULTIPLIER


_BG = "#0D1117"
_SURFACE = "#161B22"
_TEXT = "#8B98AC"
_GRID = "#2A3244"
_GOOD = "#3FB950"
_WARNING = "#D29922"
_BAD = "#F85149"
_FLOOR_CURVE_COLOR = "#7DD3FC"


def _point_colors(normalized_errors: np.ndarray, good_mult: float, warning_mult: float) -> list:
    colors = []
    for ratio in normalized_errors:
        if ratio < good_mult:
            colors.append(_GOOD)
        elif ratio < warning_mult:
            colors.append(_WARNING)
        else:
            colors.append(_BAD)
    return colors


def render_uncertainty_plot_png(
    depths_m: np.ndarray,
    errors_px: np.ndarray,
    floor_inputs: FloorInputs,
    good_mult: float = M2_GOOD_MULTIPLIER,
    warning_mult: float = M2_WARNING_MULTIPLIER,
    dpi: int = 130,
    point_size: float = 8.0,
) -> Optional[bytes]:
    """
    Scatter depth (x) vs actual error (y), each point colored GOOD/
    WARNING/BAD by its own normalized_error = actual_error / floor(its
    own depth) -- same multiplier scheme M2's aggregate classification
    uses, just applied per-point instead of to the frame mean. The
    floor(Z) curve itself (and its good/warning multiplier bands) is
    drawn across the full observed depth range as a reference: a point
    ABOVE the curve has more error than its own depth's sensor noise
    would explain; a point at or below it is consistent with ordinary
    noise, however large its raw pixel error might look on its own.

    Returns None if depths_m/errors_px are empty.
    """
    depths_m = np.asarray(depths_m, dtype=np.float64)
    errors_px = np.asarray(errors_px, dtype=np.float64)
    finite = np.isfinite(depths_m) & np.isfinite(errors_px) & (depths_m > 0)
    depths_m = depths_m[finite]
    errors_px = errors_px[finite]
    if depths_m.size == 0:
        return None

    normalized = errors_px / compute_floor_array(floor_inputs, depths_m)
    colors = _point_colors(normalized, good_mult, warning_mult)

    z_min, z_max = float(depths_m.min()), float(depths_m.max())
    if z_max <= z_min:
        # every point at (near enough) the same depth -- still show a
        # single-column scatter rather than degenerating the curve's x-range
        z_min, z_max = z_min * 0.9, z_max * 1.1 if z_max > 0 else 1.0

    curve_z = np.linspace(z_min, z_max, 200)
    curve_floor = compute_floor_array(floor_inputs, curve_z)

    fig = plt.figure(figsize=(8.2, 4.2), dpi=dpi)
    try:
        ax = fig.add_subplot(111)
        fig.patch.set_facecolor(_BG)
        ax.set_facecolor(_SURFACE)

        ax.fill_between(curve_z, 0, curve_floor * good_mult, color=_GOOD, alpha=0.10, zorder=1)
        ax.fill_between(curve_z, curve_floor * good_mult, curve_floor * warning_mult,
                         color=_WARNING, alpha=0.08, zorder=1)

        ax.plot(curve_z, curve_floor, color=_FLOOR_CURVE_COLOR, linewidth=1.5, zorder=2,
                label="floor(Z) -- expected sensor noise")
        ax.plot(curve_z, curve_floor * good_mult, color=_GOOD, linewidth=1.0, linestyle="--",
                zorder=2, alpha=0.8, label=f"{good_mult:g}x floor (GOOD boundary)")
        ax.plot(curve_z, curve_floor * warning_mult, color=_BAD, linewidth=1.0, linestyle="--",
                zorder=2, alpha=0.8, label=f"{warning_mult:g}x floor (BAD boundary)")

        ax.scatter(depths_m, errors_px, c=colors, s=point_size, alpha=0.75, zorder=3,
                   edgecolors="none")

        ax.set_xlabel("Depth Z (m)", color=_TEXT, fontsize=9)
        ax.set_ylabel("Actual error (px)", color=_TEXT, fontsize=9)
        ax.set_title("Per-point error vs. sensor-relative noise floor", color=_TEXT, fontsize=10)
        ax.tick_params(colors=_TEXT, labelsize=8)
        for spine in ax.spines.values():
            spine.set_color(_GRID)
        ax.grid(True, color=_GRID, linewidth=0.5, alpha=0.6)
        ax.legend(loc="upper right", frameon=False, labelcolor=_TEXT, fontsize=7)

        fig.tight_layout()
        buf = BytesIO()
        fig.savefig(buf, format="png", facecolor=fig.get_facecolor())
        return buf.getvalue()
    except Exception:
        # Consistent with the other visualization modules: a broken/
        # partial matplotlib install shouldn't crash the caller over one
        # optional diagnostic image.
        return None
    finally:
        plt.close(fig)


def render_uncertainty_plot_from_result(edge_alignment_result, floor_inputs: FloorInputs, **kwargs) -> Optional[bytes]:
    """
    Convenience wrapper taking an EdgeAlignmentResult directly (as
    returned by evaluation.edge_alignment.evaluate_edge_alignment).
    Returns None if the result FAILed or lacks per-point uncertainty
    data (e.g. it predates STEP7, or use_correspondence_matching's
    per-point fields were never populated for some other reason).
    """
    if (edge_alignment_result.classification == "FAIL"
            or edge_alignment_result.edge_point_depths_m is None
            or edge_alignment_result.edge_point_errors_px is None):
        return None
    return render_uncertainty_plot_png(
        edge_alignment_result.edge_point_depths_m,
        edge_alignment_result.edge_point_errors_px,
        floor_inputs,
        **kwargs,
    )
