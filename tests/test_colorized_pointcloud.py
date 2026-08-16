import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from visualization.colorized_pointcloud import (
    colorize_lidar_points,
    render_colorized_pointcloud_png,
    render_colorized_pointcloud_from_frame,
    ColorizedPointCloudResult,
)
from tests.test_holdout_consistency import _make_camera, _make_image, _make_base_points_cam_frame


def _camera_and_image_and_points():
    camera = _make_camera()
    image = _make_image()
    points = _make_base_points_cam_frame()
    return camera, image, points


def test_colorize_lidar_points_returns_one_color_per_point():
    camera, image, points = _camera_and_image_and_points()
    result = colorize_lidar_points(
        image, points, T_CL=np.eye(4), K=camera.K(), dist_coeffs=camera.dist_coeffs(),
        image_width=camera.width, image_height=camera.height, camera_model="pinhole",
    )
    assert result.num_colorized_points > 0
    assert result.points_cam.shape == (result.num_colorized_points, 3)
    assert result.colors_rgb.shape == (result.num_colorized_points, 3)
    assert result.colors_rgb.dtype == np.uint8


def test_colorize_lidar_points_samples_correct_side_of_split_image():
    # _make_image() is black on the left half (u < CX) and white on the
    # right half (u >= CX); _make_base_points_cam_frame() places points at
    # Z_NEAR for u < CX and Z_FAR for u >= CX -- so a correct T_CL=identity
    # projection should recover that split in the sampled colors.
    camera, image, points = _camera_and_image_and_points()
    result = colorize_lidar_points(
        image, points, T_CL=np.eye(4), K=camera.K(), dist_coeffs=camera.dist_coeffs(),
        image_width=camera.width, image_height=camera.height, camera_model="pinhole",
    )
    assert result.num_colorized_points > 0
    left_mask = result.pixels[:, 0] < camera.intrinsics.cx
    right_mask = ~left_mask
    assert left_mask.any() and right_mask.any()
    # Left-side samples should be dark (black bg), right-side bright (white bg).
    assert result.colors_rgb[left_mask].mean() < 50
    assert result.colors_rgb[right_mask].mean() > 200


def test_colorize_lidar_points_empty_when_all_points_behind_camera():
    camera, image, points = _camera_and_image_and_points()
    behind = points.copy()
    behind[:, 2] *= -1  # flip depth so every point is behind the camera
    result = colorize_lidar_points(
        image, behind, T_CL=np.eye(4), K=camera.K(), dist_coeffs=camera.dist_coeffs(),
        image_width=camera.width, image_height=camera.height, camera_model="pinhole",
    )
    assert result.num_colorized_points == 0
    assert result.points_cam.shape == (0, 3)
    assert result.colors_rgb.shape == (0, 3)


def test_colorize_lidar_points_subsamples_deterministically():
    camera, image, points = _camera_and_image_and_points()
    kwargs = dict(
        T_CL=np.eye(4), K=camera.K(), dist_coeffs=camera.dist_coeffs(),
        image_width=camera.width, image_height=camera.height, camera_model="pinhole",
        max_points=10, seed=42,
    )
    r1 = colorize_lidar_points(image, points, **kwargs)
    r2 = colorize_lidar_points(image, points, **kwargs)
    assert r1.num_colorized_points == 10
    assert np.array_equal(r1.points_cam, r2.points_cam)
    assert np.array_equal(r1.colors_rgb, r2.colors_rgb)
    # subsampling shouldn't change how many points *passed projection*
    assert r1.num_valid_points == r2.num_valid_points > 10


def test_render_colorized_pointcloud_png_returns_bytes():
    camera, image, points = _camera_and_image_and_points()
    result = colorize_lidar_points(
        image, points, T_CL=np.eye(4), K=camera.K(), dist_coeffs=camera.dist_coeffs(),
        image_width=camera.width, image_height=camera.height, camera_model="pinhole",
    )
    png = render_colorized_pointcloud_png(result)
    assert png is not None
    assert isinstance(png, bytes)
    assert png[:8] == b"\x89PNG\r\n\x1a\n"  # PNG magic bytes


def test_render_colorized_pointcloud_png_returns_none_when_empty():
    empty = ColorizedPointCloudResult(
        points_cam=np.zeros((0, 3)), colors_rgb=np.zeros((0, 3), dtype=np.uint8),
        pixels=np.zeros((0, 2)), num_input_points=5, num_valid_points=0, num_colorized_points=0,
    )
    assert render_colorized_pointcloud_png(empty) is None


def test_render_colorized_pointcloud_from_frame_convenience_wrapper():
    camera, image, points = _camera_and_image_and_points()
    png = render_colorized_pointcloud_from_frame(image, points, T_CL=np.eye(4), camera=camera)
    assert png is not None
    assert png[:8] == b"\x89PNG\r\n\x1a\n"


def test_colorize_lidar_points_retains_lidar_frame_points():
    camera, image, points = _camera_and_image_and_points()
    result = colorize_lidar_points(
        image, points, T_CL=np.eye(4), K=camera.K(), dist_coeffs=camera.dist_coeffs(),
        image_width=camera.width, image_height=camera.height, camera_model="pinhole",
    )
    assert result.points_lidar is not None
    assert result.points_lidar.shape == (result.num_colorized_points, 3)
    # T_CL is identity here, so lidar-frame and camera-frame points coincide.
    assert np.allclose(result.points_lidar, result.points_cam)
