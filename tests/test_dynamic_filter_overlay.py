import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from geometry.projection import intrinsics_matrix
from evaluation.dynamic_filter import classify_points_by_motion_consistency, STATIC, DYNAMIC
from visualization.dynamic_filter_overlay import (
    render_dynamic_filter_overlay,
    render_dynamic_filter_overlay_from_frame,
    encode_png,
)
from input.camera import CameraIntrinsics, CameraDistortion, CameraModel, CameraSource


def _make_camera(width=200, height=200, fx=100.0, fy=100.0, cx=100.0, cy=100.0):
    intr = CameraIntrinsics(fx=fx, fy=fy, cx=cx, cy=cy)
    dist = CameraDistortion(model="none")
    source = CameraSource(kind="image_dir", path="/fake")
    return CameraModel(width=width, height=height, model="pinhole",
                        intrinsics=intr, distortion=dist, source=source)


def _ring_scan(num_rings, num_az, ranges_per_azimuth):
    elevations = np.radians(np.linspace(-5, 5, num_rings))
    azimuths = np.linspace(0, 2 * np.pi, num_az, endpoint=False)
    points = []
    for el in elevations:
        for i, az in enumerate(azimuths):
            r = ranges_per_azimuth[i]
            x = r * np.cos(el) * np.cos(az)
            y = r * np.cos(el) * np.sin(az)
            z = r * np.sin(el)
            points.append([x, y, z])
    return np.array(points)


def _lidar_to_camera_T_CL():
    """A proper rotation (det=+1, standard REP-103 lidar/base_link ->
    camera-optical-frame convention) mapping LiDAR +X (forward) to camera
    +Z (forward), so a LiDAR-convention scan (as built by _ring_scan)
    actually projects in front of the camera under this T_CL -- using
    np.eye(4) would silently put every point's "forward" component in
    camera Y/X instead of Z, landing them all behind the camera (filtered
    out by project_lidar_to_image's min_depth_m) and making every test in
    this file trivially pass on an empty overlay."""
    from geometry.transform import to_homogeneous
    R = np.array([
        [0.0, -1.0, 0.0],
        [0.0, 0.0, -1.0],
        [1.0, 0.0, 0.0],
    ])
    return to_homogeneous(R, np.zeros(3))


def _make_classification(num_rings=8, num_az=36, moving_bins=slice(10, 15)):
    base = np.full(num_az, 10.0)
    rng = np.random.default_rng(0)
    frames = []
    for _ in range(5):
        ranges = base.copy() + rng.normal(0, 0.01, num_az)
        ranges[moving_bins] = rng.uniform(3.0, 9.0, moving_bins.stop - moving_bins.start)
        frames.append(_ring_scan(num_rings, num_az, ranges))
    return classify_points_by_motion_consistency(
        frames, num_rings=num_rings, num_azimuth_bins=num_az,
        range_std_threshold_m=0.3, min_frames_present=3,
    )


def test_render_dynamic_filter_overlay_draws_points_and_counts():
    num_rings, num_az = 8, 36
    result = _make_classification(num_rings, num_az)
    points = _ring_scan(num_rings, num_az, np.full(num_az, 10.0))
    K = intrinsics_matrix(100, 100, 100, 100)
    T_CL = _lidar_to_camera_T_CL()

    overlay = render_dynamic_filter_overlay(
        np.zeros((200, 200, 3), dtype=np.uint8), points, T_CL, K, None,
        image_width=200, image_height=200, motion_result=result,
    )
    assert overlay.num_input_points == points.shape[0]
    assert overlay.num_valid_points > 0
    assert overlay.num_static + overlay.num_dynamic + overlay.num_unknown == overlay.num_valid_points
    assert overlay.image.sum() > 0  # something got drawn


def test_render_dynamic_filter_overlay_does_not_mutate_input():
    num_rings, num_az = 8, 36
    result = _make_classification(num_rings, num_az)
    points = _ring_scan(num_rings, num_az, np.full(num_az, 10.0))
    K = intrinsics_matrix(100, 100, 100, 100)
    image = np.zeros((200, 200, 3), dtype=np.uint8)
    render_dynamic_filter_overlay(image, points, _lidar_to_camera_T_CL(), K, None, 200, 200, motion_result=result)
    assert image.sum() == 0


def test_render_dynamic_filter_overlay_empty_when_no_valid_points():
    num_rings, num_az = 4, 8
    result = _make_classification(num_rings, num_az, moving_bins=slice(1, 2))
    points = np.array([[0.0, 0.0, -5.0]])  # behind camera -> no valid projection
    K = intrinsics_matrix(100, 100, 100, 100)
    overlay = render_dynamic_filter_overlay(
        np.zeros((200, 200, 3), dtype=np.uint8), points, np.eye(4), K, None, 200, 200, motion_result=result,
    )
    assert overlay.num_valid_points == 0
    assert overlay.num_static == overlay.num_dynamic == overlay.num_unknown == 0


def test_render_dynamic_filter_overlay_from_frame_matches_camera_dims():
    camera = _make_camera()
    num_rings, num_az = 8, 36
    result = _make_classification(num_rings, num_az)
    points = _ring_scan(num_rings, num_az, np.full(num_az, 10.0))
    image = np.zeros((200, 200, 3), dtype=np.uint8)
    overlay = render_dynamic_filter_overlay_from_frame(
        image, points, _lidar_to_camera_T_CL(), camera, motion_result=result,
    )
    assert overlay is not None
    assert overlay.num_valid_points > 0


def test_render_dynamic_filter_overlay_from_frame_returns_none_on_shape_mismatch():
    camera = _make_camera(width=200, height=200)
    wrong_image = np.zeros((10, 10, 3), dtype=np.uint8)
    result = _make_classification()
    overlay = render_dynamic_filter_overlay_from_frame(
        wrong_image, np.zeros((1, 3)), np.eye(4), camera, motion_result=result,
    )
    assert overlay is None


def test_encode_png_roundtrip_shape():
    image = np.zeros((50, 50, 3), dtype=np.uint8)
    png_bytes = encode_png(image)
    assert isinstance(png_bytes, bytes)
    assert png_bytes[:8] == b"\x89PNG\r\n\x1a\n"


if __name__ == "__main__":
    test_fns = [obj for name, obj in list(globals().items()) if name.startswith("test_")]
    passed, failed = 0, 0
    for fn in test_fns:
        try:
            fn()
            print(f"PASS  {fn.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"FAIL  {fn.__name__}: {e}")
            failed += 1
        except Exception as e:
            print(f"ERROR {fn.__name__}: {type(e).__name__}: {e}")
            failed += 1
    print(f"\n{passed} passed, {failed} failed")
    sys.exit(1 if failed else 0)
