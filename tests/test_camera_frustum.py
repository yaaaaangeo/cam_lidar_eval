import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from visualization.camera_frustum import (
    compute_frustum_geometry,
    render_camera_frustum_png,
    render_camera_frustum_from_dataset,
    auto_frustum_depth,
)
from input.camera import CameraIntrinsics
from tests.test_holdout_consistency import _make_dataset


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


def test_auto_frustum_depth_uses_75th_percentile_of_point_depths():
    points = np.zeros((100, 3))
    points[:, 2] = np.linspace(1.0, 41.0, 100)  # depths 1..41 in the lidar/camera frame (identity T_CL)
    depth = auto_frustum_depth(points, np.eye(4))
    assert np.isclose(depth, np.percentile(points[:, 2], 75))


def test_auto_frustum_depth_falls_back_when_no_points():
    assert auto_frustum_depth(None, np.eye(4), fallback=12.0) == 12.0
    assert auto_frustum_depth(np.zeros((0, 3)), np.eye(4), fallback=12.0) == 12.0


def test_auto_depth_used_when_depth_m_not_given():
    # With no explicit depth_m, points_lidar's depth distribution should
    # drive the frustum size -- exercised end-to-end here (via no
    # exception and a valid PNG); auto_frustum_depth itself is unit-tested
    # directly above.
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


# ---------------------------------------------------------------------------
# Graceful degradation when mpl_toolkits.mplot3d isn't usable in this
# environment (e.g. a system + pip matplotlib version mismatch -- see the
# module docstring / _MPL3D_AVAILABLE comment in visualization.camera_frustum)
# ---------------------------------------------------------------------------

def test_render_camera_frustum_png_returns_none_when_mpl3d_unavailable():
    import visualization.camera_frustum as cf
    original = cf._MPL3D_AVAILABLE
    cf._MPL3D_AVAILABLE = False
    try:
        png = cf.render_camera_frustum_png(np.eye(4), _K(), WIDTH, HEIGHT, depth_m=8.0)
        assert png is None
    finally:
        cf._MPL3D_AVAILABLE = original


def test_render_camera_frustum_from_dataset_returns_none_when_mpl3d_unavailable():
    import visualization.camera_frustum as cf
    original = cf._MPL3D_AVAILABLE
    cf._MPL3D_AVAILABLE = False
    try:
        dataset = _make_dataset([0.0] * 10)
        png = cf.render_camera_frustum_from_dataset(dataset)
        assert png is None
        # still raises for an empty dataset even in the degraded-3D state --
        # that's a caller bug, not an environment issue, and shouldn't be
        # silently swallowed into a None
        empty_dataset = _make_dataset([])
        try:
            cf.render_camera_frustum_from_dataset(empty_dataset)
            assert False, "expected ValueError for empty dataset"
        except ValueError:
            pass
    finally:
        cf._MPL3D_AVAILABLE = original


def test_render_camera_frustum_png_returns_none_on_unexpected_plotting_error():
    """Even with mpl_toolkits importable, some OTHER failure inside the
    plotting call (e.g. `projection='3d'` registration itself failing in a
    genuinely broken environment) must also degrade to None, not crash the
    caller -- simulated here by forcing add_subplot to raise."""
    import visualization.camera_frustum as cf
    import matplotlib.figure

    original_add_subplot = matplotlib.figure.Figure.add_subplot

    def _broken_add_subplot(self, *args, **kwargs):
        raise RuntimeError("simulated broken 3d projection registration")

    matplotlib.figure.Figure.add_subplot = _broken_add_subplot
    try:
        png = cf.render_camera_frustum_png(np.eye(4), _K(), WIDTH, HEIGHT, depth_m=8.0)
        assert png is None
    finally:
        matplotlib.figure.Figure.add_subplot = original_add_subplot


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
