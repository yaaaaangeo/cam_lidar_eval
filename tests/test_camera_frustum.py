import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from visualization.camera_frustum import (
    compute_frustum_geometry,
    render_camera_frustum_png,
    render_camera_frustum_from_dataset,
    FrustumGeometry,
)
from input.camera import CameraIntrinsics
from tests.test_holdout_consistency import _make_dataset, _make_camera


WIDTH, HEIGHT = 640, 480
FX = FY = 500.0
CX, CY = 320.0, 240.0


def _K():
    return CameraIntrinsics(FX, FY, CX, CY).as_matrix()


def test_compute_frustum_geometry_identity_extrinsic():
    # Identity T_CL: camera and lidar share the same origin/orientation.
    geom = compute_frustum_geometry(np.eye(4), _K(), WIDTH, HEIGHT, depth_m=10.0)
    assert np.allclose(geom.camera_origin, [0.0, 0.0, 0.0])
    assert np.allclose(geom.camera_axes, np.eye(3))
    assert geom.baseline_m == 0.0
    assert geom.far_corners.shape == (4, 3)
    # All far corners should sit at depth 10 along the camera's (== lidar's) Z axis.
    assert np.allclose(geom.far_corners[:, 2], 10.0)


def test_compute_frustum_geometry_translated_extrinsic_moves_camera_origin():
    # T_CL: p_cam = T_CL @ p_lidar. A pure translation t means points
    # shift by -t when going lidar->camera, so the camera's origin in the
    # lidar frame is at -T_CL's translation (since T_LC = inverse of a
    # pure translation just negates it for R=I).
    T_CL = np.eye(4)
    T_CL[:3, 3] = [1.0, 2.0, 3.0]
    geom = compute_frustum_geometry(T_CL, _K(), WIDTH, HEIGHT, depth_m=5.0)
    assert np.allclose(geom.camera_origin, [-1.0, -2.0, -3.0])
    assert np.isclose(geom.baseline_m, np.linalg.norm([1.0, 2.0, 3.0]))


def test_compute_frustum_geometry_fov_matches_intrinsics():
    geom = compute_frustum_geometry(np.eye(4), _K(), WIDTH, HEIGHT, depth_m=10.0)
    expected_hfov = np.degrees(2.0 * np.arctan((WIDTH / 2.0) / FX))
    expected_vfov = np.degrees(2.0 * np.arctan((HEIGHT / 2.0) / FY))
    assert np.isclose(geom.hfov_deg, expected_hfov)
    assert np.isclose(geom.vfov_deg, expected_vfov)


def test_compute_frustum_geometry_far_corners_scale_with_depth():
    geom_near = compute_frustum_geometry(np.eye(4), _K(), WIDTH, HEIGHT, depth_m=5.0)
    geom_far = compute_frustum_geometry(np.eye(4), _K(), WIDTH, HEIGHT, depth_m=10.0)
    near_half_width = geom_near.far_corners[:, 0].max()
    far_half_width = geom_far.far_corners[:, 0].max()
    assert np.isclose(far_half_width, near_half_width * 2.0)


def test_render_camera_frustum_png_returns_valid_png_bytes():
    png = render_camera_frustum_png(np.eye(4), _K(), WIDTH, HEIGHT, depth_m=8.0)
    assert isinstance(png, bytes)
    assert png[:8] == b"\x89PNG\r\n\x1a\n"


def test_render_camera_frustum_png_with_context_points():
    points = np.random.default_rng(0).uniform(-2, 2, size=(500, 3))
    points[:, 2] = np.abs(points[:, 2]) + 3.0  # keep points roughly in front
    png = render_camera_frustum_png(np.eye(4), _K(), WIDTH, HEIGHT, points_lidar=points)
    assert png[:8] == b"\x89PNG\r\n\x1a\n"


def test_render_camera_frustum_png_subsamples_large_clouds():
    points = np.random.default_rng(1).uniform(-2, 2, size=(20000, 3))
    points[:, 2] = np.abs(points[:, 2]) + 3.0
    # Should not raise / hang, and should still produce a valid PNG even
    # with far more points than max_points.
    png = render_camera_frustum_png(np.eye(4), _K(), WIDTH, HEIGHT, points_lidar=points, max_points=1000)
    assert png[:8] == b"\x89PNG\r\n\x1a\n"


def test_auto_depth_used_when_depth_m_not_given():
    # With no explicit depth_m, points_lidar's depth distribution should
    # drive the frustum size (implicitly exercised via no exception and a
    # valid PNG; explicit numeric check done through compute_frustum_geometry
    # directly for the auto-depth helper isn't exposed publicly, so this
    # is a smoke test of the auto-depth code path in render).
    points = np.zeros((100, 3))
    points[:, 2] = 20.0  # all points far away
    png = render_camera_frustum_png(np.eye(4), _K(), WIDTH, HEIGHT, points_lidar=points)
    assert png[:8] == b"\x89PNG\r\n\x1a\n"


def test_render_camera_frustum_from_dataset_convenience_wrapper():
    dataset = _make_dataset([0.0] * 10)
    png = render_camera_frustum_from_dataset(dataset)
    assert isinstance(png, bytes)
    assert png[:8] == b"\x89PNG\r\n\x1a\n"


def test_render_camera_frustum_from_dataset_raises_on_empty_dataset():
    dataset = _make_dataset([])
    try:
        render_camera_frustum_from_dataset(dataset)
        assert False, "expected ValueError for empty dataset"
    except ValueError:
        pass
