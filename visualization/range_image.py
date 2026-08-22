"""
visualization/range_image.py

STEP 4 -- LiDAR Ring / Topology visualization: renders the range image
(geometry.range_image.RangeImage) as a 2D heatmap -- ring rows x azimuth
columns, colored by range -- with LiDAR-native depth-discontinuity cells
(geometry.range_image.compute_edge_cell_mask) highlighted on top.

This is the direct visual counterpart to STEP 3's projection_overlay: that
one asks "does the projected point cloud's SHAPE look right in image
space", this one asks "does the sensor's OWN scan structure make sense,
and where does IT think the depth discontinuities are" -- independent of
any camera projection, extrinsic, or intrinsics at all. A person can look
at this and see e.g. "ring 12 has a big gap" (occlusion / a missed
return), "the marked edges trace a clean object silhouette" (sane data),
or "the whole image is noisy speckle" (a bad depth_jump_threshold_m, or a
genuinely noisy sensor) -- all without any calibration in the loop.
"""

from __future__ import annotations

from io import BytesIO
from typing import Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless: no display server needed, safe for CLI/report generation
import matplotlib.pyplot as plt

from geometry.range_image import (
    RangeImage,
    build_range_image,
    compute_edge_cell_mask,
    DEFAULT_NUM_RINGS,
    DEFAULT_NUM_AZIMUTH_BINS,
    DEFAULT_DEPTH_JUMP_THRESHOLD_M,
)


_BG = "#0D1117"
_SURFACE = "#161B22"
_TEXT = "#8B98AC"
_GRID = "#2A3244"
_EDGE_MARKER_COLOR = "#F0883E"


def render_range_image_png(
    range_image: RangeImage,
    edge_cell_mask: Optional[np.ndarray] = None,
    dpi: int = 130,
    cmap: str = "turbo",
    edge_marker_size: float = 3.0,
) -> Optional[bytes]:
    """
    Render a range image as a 2D heatmap (ring rows x azimuth columns),
    colored by range (near=cool/warm per the `cmap`, empty cells shown in
    a distinct dark "no return" color rather than a misleading zero).

    edge_cell_mask: optional boolean array, same shape as
    range_image.range_m -- cells to highlight as LiDAR-native depth
    discontinuities (see geometry.range_image.compute_edge_cell_mask).
    If not given, no edge markers are drawn -- just the raw range image.

    Returns None if the range image has no populated cells at all
    (nothing to show).
    """
    range_m = range_image.range_m
    populated = np.isfinite(range_m)
    if not populated.any():
        return None

    fig_h = max(2.5, min(8.0, range_image.num_rings * 0.18 + 1.0))
    fig = plt.figure(figsize=(10.5, fig_h), dpi=dpi)
    try:
        fig.patch.set_facecolor(_BG)
        ax = fig.add_subplot(111)
        ax.set_facecolor(_SURFACE)

        cmap_obj = matplotlib.colormaps[cmap].copy()
        cmap_obj.set_bad(color=_SURFACE)  # NaN (no return) cells -> surface color, not a misleading value

        im = ax.imshow(
            np.ma.masked_invalid(range_m),
            aspect="auto", cmap=cmap_obj, interpolation="nearest", origin="lower",
        )
        cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.015)
        cbar.set_label("Range (m)", color=_TEXT, fontsize=8)
        cbar.ax.tick_params(colors=_TEXT, labelsize=7)
        cbar.outline.set_edgecolor(_GRID)

        if edge_cell_mask is not None and edge_cell_mask.any():
            edge_rows, edge_cols = np.nonzero(edge_cell_mask)
            ax.scatter(
                edge_cols, edge_rows, s=edge_marker_size, c=_EDGE_MARKER_COLOR,
                marker=".", linewidths=0, label="LiDAR-native edge",
            )
            ax.legend(loc="upper right", fontsize=7, facecolor=_SURFACE, edgecolor=_GRID, labelcolor=_TEXT)

        ax.set_xlabel("Azimuth bin", color=_TEXT, fontsize=8)
        ax.set_ylabel("Ring", color=_TEXT, fontsize=8)
        ax.set_title(
            f"LiDAR range image  ({range_image.num_rings} rings \u00d7 {range_image.num_azimuth_bins} azimuth bins)",
            color=_TEXT, fontsize=10,
        )
        ax.tick_params(colors=_TEXT, labelsize=7)
        for spine in ax.spines.values():
            spine.set_color(_GRID)

        fig.tight_layout()
        buf = BytesIO()
        fig.savefig(buf, format="png", facecolor=fig.get_facecolor())
        return buf.getvalue()
    except Exception:
        # Consistent with visualization.camera_frustum / colorized_pointcloud:
        # a broken/partial matplotlib install shouldn't crash the caller
        # over one optional diagnostic image.
        return None
    finally:
        plt.close(fig)


def render_range_image_from_points(
    points_lidar: np.ndarray,
    ring: Optional[np.ndarray] = None,
    num_rings: int = DEFAULT_NUM_RINGS,
    num_azimuth_bins: int = DEFAULT_NUM_AZIMUTH_BINS,
    vertical_fov_deg: Optional[float] = None,
    depth_jump_threshold_m: float = DEFAULT_DEPTH_JUMP_THRESHOLD_M,
    wrap_azimuth: bool = True,
    show_edges: bool = True,
    **render_kwargs,
) -> Optional[bytes]:
    """
    Convenience wrapper mirroring the other visualization modules'
    *_from_frame helpers: builds the range image AND (if show_edges) the
    LiDAR-native edge mask directly from a raw LiDAR frame's points, then
    renders in one call.
    """
    points_lidar = np.asarray(points_lidar, dtype=np.float64)
    if points_lidar.shape[0] == 0:
        return None

    ri = build_range_image(
        points_lidar, ring=ring, num_rings=num_rings, num_azimuth_bins=num_azimuth_bins,
        vertical_fov_deg=vertical_fov_deg,
    )
    edge_cell_mask = None
    if show_edges:
        edge_cell_mask = compute_edge_cell_mask(
            ri, depth_jump_threshold_m=depth_jump_threshold_m, wrap_azimuth=wrap_azimuth,
        )
    return render_range_image_png(ri, edge_cell_mask=edge_cell_mask, **render_kwargs)
