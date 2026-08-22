"""
geometry/range_image.py

STEP 4 -- LiDAR Ring / Topology (see evaluation_metric_spec.md's STEP 4).

Builds a "range image": LiDAR points organized into a 2D grid of ring
(vertical/channel index) x azimuth (horizontal scan angle), each cell
holding that laser return's range (distance from the sensor). This is the
LiDAR's own native scan structure -- and it's what lets us tell "this
point sits at a genuine depth discontinuity, AS THE SENSOR ACTUALLY
SCANNED IT" (a real object silhouette / occlusion boundary) apart from
"this point happens to land near other points that are far away, after
being flattened into a 2D IMAGE-SPACE PROJECTION" (which is what
evaluation.edge_alignment.extract_lidar_edge_points does today -- a
useful, cheap, projection-only heuristic, but one that can be fooled by
projection-induced coincidences, especially at oblique viewing angles or
sparse far-range returns, since it only ever looks at where points landed
on the image, never at how they were actually laid out by the scanner).

Ring index: if the point cloud carries an explicit per-point ring/channel
index (common for e.g. Velodyne/Ouster PCD or PointCloud2 data with a
"ring" field), pass it in via the `ring` parameter for the most accurate
result. Otherwise, a ring index is DERIVED from each point's vertical
(elevation) angle, binned into `num_rings` equal-width bins across the
observed elevation range -- this works on ANY (N,3) point cloud with no
special file-format support needed, which is the common case for this
tool's existing pcd_dir/ply_dir/rosbag loaders today (none of which
currently parse a ring field). Both paths produce the same downstream
RangeImage structure, so the rest of the pipeline (edge extraction,
visualization) doesn't need to know or care which one was used.

Azimuth convention: azimuth = atan2(y, x) in the LiDAR's own frame,
wrapped to [0, 2*pi). This matches the common mounting convention (+X
forward, +Y left, +Z up -- REP-103) already assumed elsewhere in this
codebase (see geometry/transform.py's docstring); azimuth 0 is "straight
ahead".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


DEFAULT_NUM_RINGS = 32
DEFAULT_NUM_AZIMUTH_BINS = 900  # 0.4 deg/bin at 360 deg -- fine enough to
                                  # resolve most real occlusion boundaries
                                  # without an excessively sparse grid on
                                  # modest point counts.
DEFAULT_DEPTH_JUMP_THRESHOLD_M = 0.3


def compute_azimuth_rad(points: np.ndarray) -> np.ndarray:
    """azimuth = atan2(y, x), wrapped to [0, 2*pi)."""
    az = np.arctan2(points[:, 1], points[:, 0])
    return np.mod(az, 2.0 * np.pi)


def compute_elevation_rad(points: np.ndarray) -> np.ndarray:
    """elevation = atan2(z, sqrt(x^2 + y^2)) -- angle above the XY plane."""
    horizontal = np.hypot(points[:, 0], points[:, 1])
    return np.arctan2(points[:, 2], horizontal)


def derive_ring_index(
    points: np.ndarray,
    num_rings: int = DEFAULT_NUM_RINGS,
    vertical_fov_deg: Optional[float] = None,
) -> np.ndarray:
    """
    Derive a pseudo-ring index (0..num_rings-1) for each point by binning
    its elevation angle into num_rings equal-width bins.

    If vertical_fov_deg is given, bins span exactly [-fov/2, +fov/2]
    (matching the sensor's actual spec, so ring boundaries are stable
    across frames/scenes even if a particular frame doesn't populate the
    sensor's full vertical range). Otherwise, bins span the OBSERVED
    elevation range in `points` -- still useful (points at similar
    elevation still land in the same ring), but ring boundaries can shift
    frame-to-frame with whatever elevations happened to be present.

    Points with non-finite elevation (shouldn't normally occur for
    finite, non-degenerate XYZ) are assigned ring -1 (invalid).
    """
    n = points.shape[0]
    if n == 0:
        return np.zeros(0, dtype=np.int64)

    elevation = compute_elevation_rad(points)
    finite = np.isfinite(elevation)

    if vertical_fov_deg is not None:
        half_fov = np.radians(vertical_fov_deg) / 2.0
        lo, hi = -half_fov, half_fov
    else:
        if not finite.any():
            return np.full(n, -1, dtype=np.int64)
        lo, hi = float(np.min(elevation[finite])), float(np.max(elevation[finite]))
        if hi <= lo:
            # degenerate (all points at the same elevation, e.g. a single
            # flat synthetic plane) -- everything falls in ring 0.
            ring = np.where(finite, 0, -1).astype(np.int64)
            return ring

    span = hi - lo
    normalized = (elevation - lo) / span if span > 0 else np.zeros(n)
    ring = np.floor(normalized * num_rings).astype(np.int64)
    ring = np.clip(ring, 0, num_rings - 1)
    ring = np.where(finite, ring, -1)
    return ring


@dataclass
class RangeImage:
    """
    2D LiDAR range image: rows = ring index, columns = azimuth bin.

    range_m: (num_rings, num_azimuth_bins), NaN where no point landed in
    that cell.
    point_index: (num_rings, num_azimuth_bins) int, -1 where empty --
    maps each populated cell back to a row index in the ORIGINAL points
    array passed to build_range_image, so callers can recover full point
    data (not just range) for any cell.
    ring: (N,) the ring index actually used for each input point (either
    the caller-supplied `ring` array, or the derived pseudo-ring).
    azimuth_bin: (N,) the azimuth bin index actually used for each point.
    """
    range_m: np.ndarray
    point_index: np.ndarray
    ring: np.ndarray
    azimuth_bin: np.ndarray
    num_rings: int
    num_azimuth_bins: int


def build_range_image(
    points: np.ndarray,
    ring: Optional[np.ndarray] = None,
    num_rings: int = DEFAULT_NUM_RINGS,
    num_azimuth_bins: int = DEFAULT_NUM_AZIMUTH_BINS,
    vertical_fov_deg: Optional[float] = None,
) -> RangeImage:
    """
    Organize points into a (num_rings, num_azimuth_bins) range image.

    ring: optional (N,) explicit per-point ring/channel index. If given,
    used as-is (num_rings is then just the array shape -- values outside
    [0, num_rings) are clamped, matching derive_ring_index's contract).
    If None, ring is derived from elevation angle via derive_ring_index.

    When multiple points fall in the same (ring, azimuth_bin) cell (can
    happen with a coarse grid, or an unstructured/non-native-order point
    cloud), the NEAREST (smallest range) point wins -- consistent with
    this module's overall "closer surface is what defines a visible
    boundary" stance (see extract_lidar_native_edge_points).
    """
    points = np.asarray(points, dtype=np.float64)
    n = points.shape[0]

    if ring is None:
        ring = derive_ring_index(points, num_rings=num_rings, vertical_fov_deg=vertical_fov_deg)
    else:
        ring = np.asarray(ring)
        num_rings = int(max(num_rings, (ring.max() + 1) if n > 0 and ring.size > 0 else num_rings))
        ring = np.clip(ring, -1, num_rings - 1)

    range_m = np.full((num_rings, num_azimuth_bins), np.nan, dtype=np.float64)
    point_index = np.full((num_rings, num_azimuth_bins), -1, dtype=np.int64)

    if n == 0:
        return RangeImage(
            range_m=range_m, point_index=point_index,
            ring=np.zeros(0, dtype=np.int64), azimuth_bin=np.zeros(0, dtype=np.int64),
            num_rings=num_rings, num_azimuth_bins=num_azimuth_bins,
        )

    azimuth = compute_azimuth_rad(points)
    azimuth_bin = np.floor(azimuth / (2.0 * np.pi) * num_azimuth_bins).astype(np.int64)
    azimuth_bin = np.clip(azimuth_bin, 0, num_azimuth_bins - 1)

    point_range = np.linalg.norm(points[:, :3], axis=1)

    valid = (ring >= 0) & np.isfinite(point_range)
    valid_idx = np.nonzero(valid)[0]
    # Process points sorted by DESCENDING range first, so the final write
    # to each cell (numpy fancy-index assignment keeps the LAST write) is
    # the nearest point -- vectorized equivalent of "nearest point wins"
    # without a Python loop.
    order = valid_idx[np.argsort(-point_range[valid_idx])]
    r_idx = ring[order]
    a_idx = azimuth_bin[order]
    range_m[r_idx, a_idx] = point_range[order]
    point_index[r_idx, a_idx] = order

    return RangeImage(
        range_m=range_m, point_index=point_index,
        ring=ring, azimuth_bin=azimuth_bin,
        num_rings=num_rings, num_azimuth_bins=num_azimuth_bins,
    )


def compute_edge_cell_mask(
    range_image: RangeImage,
    depth_jump_threshold_m: float = DEFAULT_DEPTH_JUMP_THRESHOLD_M,
    wrap_azimuth: bool = True,
) -> np.ndarray:
    """
    Cell-level version of extract_lidar_native_edge_points: returns a
    boolean (num_rings, num_azimuth_bins) mask over RANGE-IMAGE CELLS
    (not original points) marking near-side depth-discontinuity cells.
    Factored out from extract_lidar_native_edge_points so visualization
    code (which wants to highlight CELLS on the range image itself) and
    point-selection code (which wants a mask over the ORIGINAL points
    array) can both build on the same underlying comparison without
    duplicating it -- see extract_lidar_native_edge_points' docstring for
    the near-side-only / wrap_azimuth semantics, which apply identically
    here.
    """
    if wrap_azimuth:
        left = np.roll(range_image.range_m, shift=1, axis=1)
        right = np.roll(range_image.range_m, shift=-1, axis=1)
    else:
        left = np.full_like(range_image.range_m, np.nan)
        right = np.full_like(range_image.range_m, np.nan)
        left[:, 1:] = range_image.range_m[:, :-1]
        right[:, :-1] = range_image.range_m[:, 1:]

    center = range_image.range_m
    left_jump = (left - center) > depth_jump_threshold_m
    right_jump = (right - center) > depth_jump_threshold_m
    return np.isfinite(center) & (left_jump | right_jump)


def extract_lidar_native_edge_points(
    points: np.ndarray,
    ring: Optional[np.ndarray] = None,
    num_rings: int = DEFAULT_NUM_RINGS,
    num_azimuth_bins: int = DEFAULT_NUM_AZIMUTH_BINS,
    vertical_fov_deg: Optional[float] = None,
    depth_jump_threshold_m: float = DEFAULT_DEPTH_JUMP_THRESHOLD_M,
    wrap_azimuth: bool = True,
) -> np.ndarray:
    """
    Identify which points sit on a LiDAR-NATIVE depth discontinuity: for
    each populated range-image cell, compare its range to its immediate
    LEFT and RIGHT neighbor WITHIN THE SAME RING (i.e. adjacent laser
    pulses in the actual scan order, not adjacent after camera
    projection). If either neighbor's range differs by more than
    depth_jump_threshold_m, this cell sits at an occlusion boundary.

    Only the NEAR side (smaller range) of a boundary is kept as an edge
    point, mirroring evaluation.edge_alignment.extract_lidar_edge_points'
    same "closer surface defines the visible silhouette" logic -- the
    far-side surface is occluded there, not edge-defining.

    Returns a boolean mask over the ORIGINAL `points` array (length N,
    same order as input) -- NOT over range-image cells -- so this can be
    used as a drop-in point-selection mask alongside the existing
    projection-space extract_lidar_edge_points.

    wrap_azimuth: if True (default), azimuth bin 0's left neighbor is the
    last bin and vice versa (the scan is a full 360-degree circle, so
    "adjacent" genuinely wraps around). Set False to treat the first/last
    azimuth bins as having no left/right neighbor respectively -- useful
    if the point cloud is known to be a partial (non-360) scan, where
    wrapping would incorrectly compare unrelated far-apart azimuths.
    """
    n = points.shape[0]
    if n == 0:
        return np.zeros(0, dtype=bool)

    ri = build_range_image(
        points, ring=ring, num_rings=num_rings, num_azimuth_bins=num_azimuth_bins,
        vertical_fov_deg=vertical_fov_deg,
    )
    edge_cell = compute_edge_cell_mask(ri, depth_jump_threshold_m=depth_jump_threshold_m, wrap_azimuth=wrap_azimuth)

    edge_mask = np.zeros(n, dtype=bool)
    edge_rows, edge_cols = np.nonzero(edge_cell)
    if edge_rows.size > 0:
        edge_point_indices = ri.point_index[edge_rows, edge_cols]
        edge_point_indices = edge_point_indices[edge_point_indices >= 0]
        edge_mask[edge_point_indices] = True
    return edge_mask
