"""
visualization/projection_overlay.py

STEP 3 -- Projection verification: the "does this even look right" visual
sanity check called out explicitly in evaluation_metric_spec.md's STEP 3
("처음부터 반드시 overlay를 만드세요... RGB image + projected LiDAR가 눈으로
맞는지 확인합니다").

Distinct from visualization/overlay.py's render_overlay(): that function
draws only the M2 edge-alignment result's MATCHED edge points, colored by
their GOOD/WARNING/BAD error classification -- it requires M2 to have
already run and only shows a curated subset of points. This module draws
EVERY valid projected LiDAR point directly on the raw camera image, colored
by depth, and requires nothing but the extrinsic + intrinsics + a frame.
That makes it usable as a first, cheap sanity check before M2 (or any
scoring) runs at all: does the overall SHAPE of the projected point cloud
follow the image's real geometry (building edges, road edges, object
silhouettes)? A grossly wrong T_CL, a swapped axis, or a bad intrinsics
matrix is usually obvious at a glance here, long before it shows up as a
specific pixel-error number.

Reuses geometry.projection.project_lidar_to_image -- the same function
M0/M2/the colorized point cloud view all use -- so "which points are
valid" (in front of the camera, inside image bounds) never drifts out of
sync with the rest of the tool.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import cv2

from geometry.projection import project_lidar_to_image


# Depth colormap range, in meters. Points at/below _DEPTH_NEAR are drawn in
# the "near" color; at/above _DEPTH_FAR in the "far" color; values in
# between are linearly interpolated through OpenCV's TURBO colormap (a
# perceptually-reasonable, high-contrast rainbow -- easy to eyeball "close
# vs far" at a glance, which is exactly what this view is for).
_DEFAULT_DEPTH_NEAR_M = 1.0
_DEFAULT_DEPTH_FAR_M = 50.0

_POINT_RADIUS = 2
_POINT_THICKNESS = -1  # filled


@dataclass
class ProjectionOverlayResult:
    image: np.ndarray            # BGR image with points drawn on top
    num_input_points: int
    num_valid_points: int        # points actually drawn (in front + in bounds)
    depth_near_m: float
    depth_far_m: float


def render_projection_overlay(
    image: np.ndarray,
    points_lidar: np.ndarray,
    T_CL: np.ndarray,
    K: np.ndarray,
    dist_coeffs: Optional[np.ndarray],
    image_width: int,
    image_height: int,
    camera_model: str = "pinhole",
    min_depth_m: float = 0.05,
    depth_near_m: float = _DEFAULT_DEPTH_NEAR_M,
    depth_far_m: float = _DEFAULT_DEPTH_FAR_M,
    point_radius: int = _POINT_RADIUS,
) -> ProjectionOverlayResult:
    """
    Project ALL of points_lidar into `image` via T_CL and draw every valid
    one directly on a copy of the image, colored by depth (near=warm,
    far=cool via OpenCV's TURBO colormap). Unlike visualization.overlay's
    render_overlay, this has no dependency on M2 having run -- it only
    needs the extrinsic and intrinsics, making it usable as a first,
    cheap "does projection look sane" check.

    depth_near_m / depth_far_m: the depth range the colormap is stretched
    across. Points closer than depth_near_m or farther than depth_far_m
    are clamped to the colormap's endpoints, not excluded -- this view is
    about SHAPE, not a hard depth cutoff.
    """
    if depth_far_m <= depth_near_m:
        raise ValueError(
            f"depth_far_m ({depth_far_m}) must be > depth_near_m ({depth_near_m})"
        )

    canvas = image.copy()
    if canvas.ndim == 2:
        canvas = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)

    points_lidar = np.asarray(points_lidar, dtype=np.float64)
    proj = project_lidar_to_image(
        points_lidar, T_CL, K, dist_coeffs, image_width, image_height,
        camera_model=camera_model, min_depth_m=min_depth_m,
    )

    if proj.num_valid_points > 0:
        depth_norm = np.clip(
            (proj.depths - depth_near_m) / (depth_far_m - depth_near_m), 0.0, 1.0
        )
        # cv2.applyColorMap expects a (N,1) uint8 array; TURBO maps 0->deep
        # blue/purple (near) through green/yellow to 255->deep red (far).
        depth_u8 = (depth_norm * 255.0).astype(np.uint8).reshape(-1, 1)
        colors = cv2.applyColorMap(depth_u8, cv2.COLORMAP_TURBO).reshape(-1, 3)

        for (u, v), color in zip(proj.pixels, colors):
            cv2.circle(
                canvas, (int(round(u)), int(round(v))), point_radius,
                (int(color[0]), int(color[1]), int(color[2])), _POINT_THICKNESS,
                lineType=cv2.LINE_AA,
            )

    return ProjectionOverlayResult(
        image=canvas,
        num_input_points=proj.num_input_points,
        num_valid_points=proj.num_valid_points,
        depth_near_m=depth_near_m,
        depth_far_m=depth_far_m,
    )


def render_projection_overlay_from_frame(
    image: np.ndarray,
    points_lidar: np.ndarray,
    T_CL: np.ndarray,
    camera,
    **kwargs,
) -> Optional[ProjectionOverlayResult]:
    """
    Convenience wrapper mirroring visualization.overlay's
    render_overlay_from_result / visualization.colorized_pointcloud's
    render_colorized_pointcloud_from_frame: takes the raw frame data plus
    a CameraModel (as already used throughout the pipeline). Returns None
    if the image's pixel dimensions don't match the camera config (same
    guard used elsewhere) rather than raising, so callers building a
    best-effort visuals dict can skip this one gracefully.
    """
    if image.ndim < 2 or image.shape[0] != camera.height or image.shape[1] != camera.width:
        return None
    return render_projection_overlay(
        image, points_lidar, T_CL,
        K=camera.K(), dist_coeffs=camera.dist_coeffs(),
        image_width=camera.width, image_height=camera.height,
        camera_model=camera.projection_model_name(),
        **kwargs,
    )


def encode_png(image_bgr: np.ndarray) -> bytes:
    """Encode a BGR image array to PNG bytes (for embedding or writing to disk)."""
    ok, buf = cv2.imencode(".png", image_bgr)
    if not ok:
        raise RuntimeError("Failed to encode projection overlay image to PNG.")
    return buf.tobytes()


def save_projection_overlay_png(image_bgr: np.ndarray, path: str) -> None:
    ok = cv2.imwrite(path, image_bgr)
    if not ok:
        raise RuntimeError(f"Failed to write projection overlay image to {path!r}.")
