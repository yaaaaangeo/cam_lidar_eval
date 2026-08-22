"""
visualization/dynamic_filter_overlay.py

STEP 8 -- Dynamic Object Filtering visualization: every valid projected
LiDAR point drawn on the camera image, colored by its
evaluation.dynamic_filter classification (STATIC/DYNAMIC/UNKNOWN). Makes
the "how much of the scene is contaminated by moving objects, and where"
question in evaluation.dynamic_filter.DynamicFilteringComparison literal
and inspectable, the same way visualization.projection_overlay makes
STEP 3's raw projection sanity check literal.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import cv2

from geometry.projection import project_lidar_to_image
from geometry.range_image import compute_azimuth_rad, derive_ring_index
from evaluation.dynamic_filter import MotionConsistencyResult, STATIC, DYNAMIC, UNKNOWN


_COLOR_BGR = {
    "static": (79, 185, 63),     # matches report --good (#3FB950), BGR
    "dynamic": (81, 81, 248),    # matches report --bad (#F85149), BGR
    "unknown": (170, 170, 170),  # neutral gray
}

_POINT_RADIUS = 2
_POINT_THICKNESS = -1  # filled


@dataclass
class DynamicOverlayResult:
    image: np.ndarray
    num_input_points: int
    num_valid_points: int
    num_static: int
    num_dynamic: int
    num_unknown: int


def render_dynamic_filter_overlay(
    image: np.ndarray,
    points_lidar: np.ndarray,
    T_CL: np.ndarray,
    K: np.ndarray,
    dist_coeffs: Optional[np.ndarray],
    image_width: int,
    image_height: int,
    motion_result: MotionConsistencyResult,
    camera_model: str = "pinhole",
    min_depth_m: float = 0.05,
    vertical_fov_deg: Optional[float] = None,
    point_radius: int = _POINT_RADIUS,
) -> DynamicOverlayResult:
    """
    Project points_lidar into `image` via T_CL and draw every valid point
    colored by its STATIC (green) / DYNAMIC (red) / UNKNOWN (gray)
    classification from `motion_result`
    (evaluation.dynamic_filter.classify_points_by_motion_consistency's
    output). Colors match visualization.overlay's GOOD/BAD convention so
    "red points here" reads the same way "red points there" does.
    """
    canvas = image.copy()
    if canvas.ndim == 2:
        canvas = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)

    points_lidar = np.asarray(points_lidar, dtype=np.float64)
    proj = project_lidar_to_image(
        points_lidar, T_CL, K, dist_coeffs, image_width, image_height,
        camera_model=camera_model, min_depth_m=min_depth_m,
    )

    num_static = num_dynamic = num_unknown = 0

    if proj.num_valid_points > 0:
        valid_points = points_lidar[proj.source_indices]
        ring = derive_ring_index(valid_points, num_rings=motion_result.num_rings, vertical_fov_deg=vertical_fov_deg)
        azimuth = compute_azimuth_rad(valid_points)
        azimuth_bin = np.clip(
            np.floor(azimuth / (2.0 * np.pi) * motion_result.num_azimuth_bins).astype(np.int64),
            0, motion_result.num_azimuth_bins - 1,
        )
        labels = np.full(valid_points.shape[0], UNKNOWN, dtype=np.int64)
        in_range = ring >= 0
        labels[in_range] = motion_result.cell_label[ring[in_range], azimuth_bin[in_range]]

        num_static = int(np.sum(labels == STATIC))
        num_dynamic = int(np.sum(labels == DYNAMIC))
        num_unknown = int(np.sum(labels == UNKNOWN))

        label_names = {STATIC: "static", DYNAMIC: "dynamic", UNKNOWN: "unknown"}
        for (u, v), label in zip(proj.pixels, labels):
            color = _COLOR_BGR[label_names[label]]
            cv2.circle(canvas, (int(round(u)), int(round(v))), point_radius, color, _POINT_THICKNESS,
                       lineType=cv2.LINE_AA)

    return DynamicOverlayResult(
        image=canvas,
        num_input_points=proj.num_input_points,
        num_valid_points=proj.num_valid_points,
        num_static=num_static,
        num_dynamic=num_dynamic,
        num_unknown=num_unknown,
    )


def render_dynamic_filter_overlay_from_frame(
    image: np.ndarray,
    points_lidar: np.ndarray,
    T_CL: np.ndarray,
    camera,
    motion_result: MotionConsistencyResult,
    **kwargs,
) -> Optional[DynamicOverlayResult]:
    """
    Convenience wrapper mirroring visualization.projection_overlay's
    *_from_frame helper: takes the raw frame data plus a CameraModel.
    Returns None if the image's pixel dimensions don't match the camera
    config, same guard used elsewhere in this codebase.
    """
    if image.ndim < 2 or image.shape[0] != camera.height or image.shape[1] != camera.width:
        return None
    return render_dynamic_filter_overlay(
        image, points_lidar, T_CL,
        K=camera.K(), dist_coeffs=camera.dist_coeffs(),
        image_width=camera.width, image_height=camera.height,
        camera_model=camera.projection_model_name(),
        motion_result=motion_result,
        **kwargs,
    )


def encode_png(image_bgr: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", image_bgr)
    if not ok:
        raise RuntimeError("Failed to encode dynamic filter overlay image to PNG.")
    return buf.tobytes()
