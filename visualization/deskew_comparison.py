"""
visualization/deskew_comparison.py

STEP 5 -- Motion Deskew visualization: the literal "정지 상황과 이동
상황에서 deskew 전/후 차이를 비교" (compare deskew before/after, in
stationary vs moving cases) completion criterion from
evaluation_metric_spec.md's STEP 5, made visible.

Two panels:
  left  -- bird's-eye view (X vs Y, looking straight down) with the
           ORIGINAL ("skewed", as-measured) points in one color and the
           DESKEWED points in another, so any motion-induced shift is
           directly visible as the two point sets pulling apart. At zero
           platform velocity the two sets are IDENTICAL (see
           motion.deskew's exact-no-op guarantee at v=w=0), so this panel
           is the most direct "stationary vs moving" comparison there is:
           call this function once with zero velocity and once with the
           platform's real/estimated velocity, and compare the two
           images.
  right -- histogram of per-point correction magnitude (motion.deskew.
           DeskewResult.correction_m), so the SIZE of the effect is
           quantified, not just its direction.
"""

from __future__ import annotations

from io import BytesIO
from typing import Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless: no display server needed, safe for CLI/report generation
import matplotlib.pyplot as plt

from motion.deskew import DeskewResult, deskew_points_constant_velocity


_BG = "#0D1117"
_SURFACE = "#161B22"
_TEXT = "#8B98AC"
_GRID = "#2A3244"
_BEFORE_COLOR = "#58A6FF"   # blue -- matches camera_frustum's LiDAR-origin color
_AFTER_COLOR = "#F0883E"    # orange -- matches camera_frustum's camera/frustum color

_DEFAULT_MAX_POINTS = 20_000


def render_deskew_comparison_png(
    points_before: np.ndarray,
    result: DeskewResult,
    dpi: int = 130,
    point_size: float = 1.2,
    max_points: int = _DEFAULT_MAX_POINTS,
    seed: int = 0,
) -> Optional[bytes]:
    """
    Render the before/after BEV overlay + correction-magnitude histogram
    described in this module's docstring. Returns None if there are no
    points to plot.

    max_points: if points_before has more than this many points,
    deterministically (fixed seed) subsample BOTH before and after sets
    with the SAME indices (so a given dot's before/after pair stays
    visually connectable) -- keeps the render responsive on dense clouds.
    """
    n = points_before.shape[0]
    if n == 0:
        return None

    before = np.asarray(points_before, dtype=np.float64)[:, :3]
    after = result.points_deskewed[:, :3]
    correction = result.correction_m

    if n > max_points:
        rng = np.random.default_rng(seed)
        keep = rng.choice(n, size=max_points, replace=False)
        before, after, correction = before[keep], after[keep], correction[keep]

    fig = plt.figure(figsize=(11.0, 5), dpi=dpi)
    try:
        fig.patch.set_facecolor(_BG)

        # --- Panel 1: BEV before/after overlay ------------------------------
        ax_bev = fig.add_subplot(1, 2, 1)
        ax_bev.set_facecolor(_SURFACE)
        ax_bev.scatter(before[:, 0], before[:, 1], c=_BEFORE_COLOR, s=point_size, marker=".",
                        linewidths=0, alpha=0.6, label="Before (skewed)")
        ax_bev.scatter(after[:, 0], after[:, 1], c=_AFTER_COLOR, s=point_size, marker=".",
                        linewidths=0, alpha=0.6, label="After (deskewed)")
        ax_bev.set_xlabel("X (m)", color=_TEXT, fontsize=8)
        ax_bev.set_ylabel("Y (m)", color=_TEXT, fontsize=8)
        ax_bev.set_title("Bird's-eye view: before vs after", color=_TEXT, fontsize=10)
        ax_bev.tick_params(colors=_TEXT, labelsize=7)
        ax_bev.grid(color=_GRID, linewidth=0.5)
        for spine in ax_bev.spines.values():
            spine.set_color(_GRID)
        ax_bev.set_aspect("equal", adjustable="datalim")
        legend = ax_bev.legend(loc="upper right", fontsize=7, facecolor=_SURFACE, edgecolor=_GRID, labelcolor=_TEXT)
        for handle in legend.legend_handles:
            handle.set_alpha(1.0)

        # --- Panel 2: correction-magnitude histogram ------------------------
        ax_hist = fig.add_subplot(1, 2, 2)
        ax_hist.set_facecolor(_SURFACE)
        if correction.size > 0 and np.any(correction > 0):
            ax_hist.hist(correction, bins=40, color=_AFTER_COLOR, edgecolor=_BG, linewidth=0.3)
        ax_hist.set_xlabel("Per-point correction (m)", color=_TEXT, fontsize=8)
        ax_hist.set_ylabel("Point count", color=_TEXT, fontsize=8)
        ax_hist.set_title(
            f"Correction magnitude  (mean {result.mean_correction_m:.3f}m, max {result.max_correction_m:.3f}m)",
            color=_TEXT, fontsize=10,
        )
        ax_hist.tick_params(colors=_TEXT, labelsize=7)
        ax_hist.grid(color=_GRID, linewidth=0.5)
        for spine in ax_hist.spines.values():
            spine.set_color(_GRID)

        fig.tight_layout()
        buf = BytesIO()
        fig.savefig(buf, format="png", facecolor=fig.get_facecolor())
        return buf.getvalue()
    except Exception:
        # Consistent with visualization.camera_frustum / colorized_pointcloud
        # / range_image: a broken/partial matplotlib install shouldn't crash
        # the caller over one optional diagnostic image.
        return None
    finally:
        plt.close(fig)


def render_deskew_comparison_from_points(
    points_lidar: np.ndarray,
    scan_period_s: float,
    linear_velocity_mps: np.ndarray,
    angular_velocity_rps: np.ndarray,
    reference_time_s: float = 0.0,
    point_times_s: Optional[np.ndarray] = None,
    azimuth_at_scan_start_rad: float = 0.0,
    clockwise: bool = False,
    **render_kwargs,
) -> Optional[bytes]:
    """
    Convenience wrapper: runs deskew_points_constant_velocity and renders
    the result in one call, mirroring the other visualization modules'
    *_from_frame/*_from_points helpers.
    """
    points_lidar = np.asarray(points_lidar, dtype=np.float64)
    if points_lidar.shape[0] == 0:
        return None
    result = deskew_points_constant_velocity(
        points_lidar, scan_period_s=scan_period_s,
        linear_velocity_mps=linear_velocity_mps, angular_velocity_rps=angular_velocity_rps,
        point_times_s=point_times_s, reference_time_s=reference_time_s,
        azimuth_at_scan_start_rad=azimuth_at_scan_start_rad, clockwise=clockwise,
    )
    return render_deskew_comparison_png(points_lidar, result, **render_kwargs)
