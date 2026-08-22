import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from geometry.projection import intrinsics_matrix
from visualization.projection_overlay import (
    render_projection_overlay,
    render_projection_overlay_from_frame,
    encode_png,
)
from input.camera import CameraIntrinsics, CameraDistortion, CameraModel, CameraSource


def _make_camera(width=64, height=48, fx=50.0, fy=50.0, cx=32.0, cy=24.0):
    intr = CameraIntrinsics(fx=fx, fy=fy, cx=cx, cy=cy)
    dist = CameraDistortion(model="none")
    source = CameraSource(kind="image_dir", path="/fake")
    return CameraModel(width=width, height=height, model="pinhole",
                        intrinsics=intr, distortion=dist, source=source)


def test_render_projection_overlay_draws_valid_points_only():
    image = np.zeros((48, 64, 3), dtype=np.uint8)
    K = intrinsics_matrix(50, 50, 32, 24)
    T_CL = np.eye(4)
    points = np.array([
        [0.0, 0.0, 5.0],    # in front, in bounds
        [0.0, 0.0, -5.0],   # behind camera -> excluded
        [1000.0, 0.0, 1.0],  # projects far off-image -> excluded
    ])
    result = render_projection_overlay(image, points, T_CL, K, None, image_width=64, image_height=48)
    assert result.num_input_points == 3
    assert result.num_valid_points == 1
    assert result.image.shape == (48, 64, 3)
    # the canvas should not be all-zero anymore (a point got drawn)
    assert result.image.sum() > 0


def test_render_projection_overlay_does_not_mutate_input_image():
    image = np.zeros((48, 64, 3), dtype=np.uint8)
    K = intrinsics_matrix(50, 50, 32, 24)
    T_CL = np.eye(4)
    points = np.array([[0.0, 0.0, 5.0]])
    render_projection_overlay(image, points, T_CL, K, None, image_width=64, image_height=48)
    assert image.sum() == 0, "input image must not be mutated in place"


def test_render_projection_overlay_empty_when_no_valid_points():
    image = np.full((48, 64, 3), 100, dtype=np.uint8)
    K = intrinsics_matrix(50, 50, 32, 24)
    T_CL = np.eye(4)
    points = np.array([[0.0, 0.0, -1.0], [0.0, 0.0, -2.0]])  # all behind camera
    result = render_projection_overlay(image, points, T_CL, K, None, image_width=64, image_height=48)
    assert result.num_valid_points == 0
    # canvas should be unchanged from the base image (no points drawn)
    assert np.array_equal(result.image, image)


def test_render_projection_overlay_near_and_far_points_get_different_colors():
    """Sanity check that the depth colormap actually varies with depth --
    a near point and a far point should not be drawn in the same color."""
    image = np.zeros((200, 200, 3), dtype=np.uint8)
    K = intrinsics_matrix(50, 50, 100, 100)
    T_CL = np.eye(4)
    # two points landing at clearly different pixels so we can sample each
    points = np.array([
        [-1.0, 0.0, 2.0],   # near
        [1.0, 0.0, 40.0],   # far
    ])
    result = render_projection_overlay(
        image, points, T_CL, K, None, image_width=200, image_height=200,
        depth_near_m=1.0, depth_far_m=50.0, point_radius=5,
    )
    assert result.num_valid_points == 2
    # sample colors at the two projected pixel locations (approximately)
    near_color = result.image[100, 100 - 25]  # roughly where the near point lands
    far_color = result.image[100, 100 + 1]    # roughly where the far point lands
    assert not np.array_equal(near_color, far_color)


def test_render_projection_overlay_rejects_bad_depth_range():
    image = np.zeros((48, 64, 3), dtype=np.uint8)
    K = intrinsics_matrix(50, 50, 32, 24)
    T_CL = np.eye(4)
    points = np.array([[0.0, 0.0, 5.0]])
    try:
        render_projection_overlay(image, points, T_CL, K, None, image_width=64, image_height=48,
                                   depth_near_m=50.0, depth_far_m=1.0)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_render_projection_overlay_from_frame_matches_camera_dims():
    camera = _make_camera()
    image = np.zeros((48, 64, 3), dtype=np.uint8)
    T_CL = np.eye(4)
    points = np.array([[0.0, 0.0, 5.0]])
    result = render_projection_overlay_from_frame(image, points, T_CL, camera)
    assert result is not None
    assert result.num_valid_points == 1


def test_render_projection_overlay_from_frame_returns_none_on_shape_mismatch():
    camera = _make_camera(width=64, height=48)
    wrong_image = np.zeros((10, 10, 3), dtype=np.uint8)  # doesn't match camera dims
    T_CL = np.eye(4)
    points = np.array([[0.0, 0.0, 5.0]])
    result = render_projection_overlay_from_frame(wrong_image, points, T_CL, camera)
    assert result is None


def test_encode_png_roundtrip_shape():
    image = np.zeros((48, 64, 3), dtype=np.uint8)
    png_bytes = encode_png(image)
    assert isinstance(png_bytes, bytes)
    assert png_bytes[:8] == b"\x89PNG\r\n\x1a\n"  # PNG magic bytes


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
