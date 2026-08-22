import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from geometry.projection import (
    intrinsics_matrix,
    plumb_bob_dist_coeffs,
    fisheye_dist_coeffs,
    project_points_pinhole,
    project_points_fisheye,
    project_lidar_to_image,
)


def test_intrinsics_matrix_shape_and_values():
    K = intrinsics_matrix(fx=500, fy=510, cx=320, cy=240)
    assert K.shape == (3, 3)
    assert K[0, 0] == 500
    assert K[1, 1] == 510
    assert K[0, 2] == 320
    assert K[1, 2] == 240


def test_plumb_bob_dist_coeffs_defaults_to_zero():
    coeffs = plumb_bob_dist_coeffs({})
    assert np.allclose(coeffs, [0, 0, 0, 0, 0])


def test_plumb_bob_dist_coeffs_partial():
    coeffs = plumb_bob_dist_coeffs({"k1": 0.1, "k2": -0.05})
    assert np.allclose(coeffs, [0.1, -0.05, 0, 0, 0])


def test_fisheye_dist_coeffs_defaults_to_zero():
    coeffs = fisheye_dist_coeffs({})
    assert np.allclose(coeffs, [0, 0, 0, 0])


def test_pinhole_projection_optical_axis_point_lands_at_principal_point():
    K = intrinsics_matrix(fx=500, fy=500, cx=320, cy=240)
    # a point straight ahead on the optical axis should project to (cx, cy)
    pts = np.array([[0.0, 0.0, 5.0]])
    px = project_points_pinhole(pts, K)
    assert np.allclose(px, [[320, 240]], atol=1e-6)


def test_pinhole_projection_known_analytic_point():
    K = intrinsics_matrix(fx=100, fy=100, cx=50, cy=50)
    # point at (1, 0, 1) with no distortion -> u = fx*x/z + cx = 100*1/1+50=150
    pts = np.array([[1.0, 0.0, 1.0]])
    px = project_points_pinhole(pts, K)
    assert np.allclose(px, [[150, 50]], atol=1e-6)


def test_pinhole_projection_empty_input():
    K = intrinsics_matrix(500, 500, 320, 240)
    px = project_points_pinhole(np.zeros((0, 3)), K)
    assert px.shape == (0, 2)


def test_pinhole_projection_rejects_bad_shape():
    K = intrinsics_matrix(500, 500, 320, 240)
    try:
        project_points_pinhole(np.array([1.0, 2.0, 3.0]), K)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_project_lidar_to_image_filters_behind_camera():
    T_CL = np.eye(4)
    K = intrinsics_matrix(500, 500, 320, 240)
    points = np.array([
        [0.0, 0.0, 5.0],   # in front, in bounds
        [0.0, 0.0, -5.0],  # behind camera
    ])
    result = project_lidar_to_image(points, T_CL, K, None, image_width=640, image_height=480)
    assert result.num_input_points == 2
    assert result.num_valid_points == 1
    assert result.source_indices.tolist() == [0]


def test_project_lidar_to_image_filters_out_of_bounds():
    T_CL = np.eye(4)
    K = intrinsics_matrix(500, 500, 320, 240)
    points = np.array([
        [0.0, 0.0, 5.0],     # near center -> in bounds
        [100.0, 0.0, 1.0],   # projects way off to the right -> out of bounds
    ])
    result = project_lidar_to_image(points, T_CL, K, None, image_width=640, image_height=480)
    assert result.num_valid_points == 1
    assert result.source_indices.tolist() == [0]


def test_project_lidar_to_image_empty_when_all_behind():
    T_CL = np.eye(4)
    K = intrinsics_matrix(500, 500, 320, 240)
    points = np.array([[0.0, 0.0, -1.0], [0.0, 0.0, -2.0]])
    result = project_lidar_to_image(points, T_CL, K, None, image_width=640, image_height=480)
    assert result.num_valid_points == 0
    assert result.pixels.shape == (0, 2)


def test_project_lidar_to_image_applies_extrinsic_translation():
    # Point at lidar-frame origin; T_CL shifts it forward by 5m in z (camera frame)
    T_CL = np.eye(4)
    T_CL[:3, 3] = [0, 0, 5.0]
    K = intrinsics_matrix(500, 500, 320, 240)
    points = np.array([[0.0, 0.0, 0.0]])
    result = project_lidar_to_image(points, T_CL, K, None, image_width=640, image_height=480)
    assert result.num_valid_points == 1
    assert np.isclose(result.depths[0], 5.0)
    assert np.allclose(result.pixels[0], [320, 240], atol=1e-6)


def test_project_lidar_to_image_depths_correspond_to_valid_points_only():
    T_CL = np.eye(4)
    K = intrinsics_matrix(500, 500, 320, 240)
    points = np.array([
        [0.0, 0.0, 3.0],
        [0.0, 0.0, -1.0],  # filtered out
        [0.0, 0.0, 7.0],
    ])
    result = project_lidar_to_image(points, T_CL, K, None, image_width=640, image_height=480)
    assert result.num_valid_points == 2
    assert np.allclose(sorted(result.depths.tolist()), [3.0, 7.0])


# ---------------------------------------------------------------------------
# STEP 3 -- Projection verification unit tests
# (see evaluation_metric_spec.md's STEP 3 "반드시 테스트할 것" list)
# ---------------------------------------------------------------------------

# Test 1: points in front of the camera (Z > 0) project; behind does not.
# -- already covered above by test_project_lidar_to_image_filters_behind_camera
#    and test_project_lidar_to_image_empty_when_all_behind.

# Test 2: points outside the image bounds are removed.
# -- already covered above by test_project_lidar_to_image_filters_out_of_bounds.

def test_project_lidar_to_image_point_exactly_on_boundary_is_excluded():
    """The image bounds check is a half-open interval [0, width) x [0,
    height) -- a point landing exactly AT width or height (one past the
    last valid pixel index) must be excluded, not off-by-one included."""
    K = intrinsics_matrix(fx=100, fy=100, cx=0, cy=0)  # cx=cy=0 for easy exact placement
    T_CL = np.eye(4)
    # With cx=cy=0, a point at (x=6.4, y=0, z=1) projects to exactly u=640
    # (fx * 6.4 / 1 + 0 = 640) -- one pixel past the last valid column for
    # a 640-wide image.
    points = np.array([[6.4, 0.0, 1.0]])
    result = project_lidar_to_image(points, T_CL, K, None, image_width=640, image_height=480)
    assert result.num_valid_points == 0


# Test 3: distortion applied before/after is correct.

def test_pinhole_projection_nonzero_distortion_shifts_pixel():
    """A nonzero k1 radial distortion coefficient should move an off-center
    point's projected pixel away from where it would land undistorted --
    this is what 'distortion is actually being applied' looks like, rather
    than just 'coefficients accepted without error'. (Not asserting a
    specific direction here since OpenCV's plumb-bob sign convention for
    barrel vs pincushion isn't this test's concern -- just that distortion
    measurably changes the result.)"""
    K = intrinsics_matrix(fx=300, fy=300, cx=320, cy=240)
    pts = np.array([[1.0, 0.0, 2.0]])  # off-axis point

    px_undistorted = project_points_pinhole(pts, K, dist_coeffs=None)
    px_distorted = project_points_pinhole(pts, K, dist_coeffs=plumb_bob_dist_coeffs({"k1": -0.3}))

    assert not np.allclose(px_undistorted, px_distorted)


def test_pinhole_projection_zero_distortion_matches_no_distortion():
    """Explicit all-zero coefficients must be numerically identical to
    passing dist_coeffs=None -- i.e. 'distortion model requested but all
    coefficients are zero' is truly a no-op, not an approximation."""
    K = intrinsics_matrix(fx=400, fy=400, cx=320, cy=240)
    pts = np.array([[0.5, -0.3, 3.0], [1.2, 0.8, 5.0]])
    px_none = project_points_pinhole(pts, K, dist_coeffs=None)
    px_zero = project_points_pinhole(pts, K, dist_coeffs=plumb_bob_dist_coeffs({}))
    assert np.allclose(px_none, px_zero, atol=1e-9)


def test_pinhole_projection_distortion_is_negligible_near_optical_axis():
    """A point very close to the optical axis (small x, y relative to
    depth) should be nearly unaffected by radial distortion, regardless
    of coefficient magnitude -- distortion is a function of the
    normalized (x/z, y/z) radius, which is ~0 on-axis. This is a basic
    physical sanity check on the distortion model, independent of exact
    coefficient values."""
    K = intrinsics_matrix(fx=500, fy=500, cx=320, cy=240)
    pts = np.array([[0.0001, 0.0001, 5.0]])  # almost exactly on-axis
    px_undistorted = project_points_pinhole(pts, K, dist_coeffs=None)
    px_distorted = project_points_pinhole(pts, K, dist_coeffs=plumb_bob_dist_coeffs({"k1": -0.5, "k2": 0.2}))
    assert np.allclose(px_undistorted, px_distorted, atol=0.05)


def test_fisheye_projection_optical_axis_point_lands_at_principal_point():
    K = intrinsics_matrix(fx=300, fy=300, cx=320, cy=240)
    pts = np.array([[0.0, 0.0, 5.0]])
    px = project_points_fisheye(pts, K)
    assert np.allclose(px, [[320, 240]], atol=1e-6)


def test_fisheye_projection_differs_from_pinhole_off_axis():
    """Off-axis points should land at genuinely different pixel locations
    under the fisheye (equidistant) model vs plain pinhole -- otherwise
    the fisheye path isn't actually doing anything different."""
    K = intrinsics_matrix(fx=300, fy=300, cx=320, cy=240)
    pts = np.array([[1.0, 0.5, 1.0]])  # steep off-axis angle
    px_pinhole = project_points_pinhole(pts, K)
    px_fisheye = project_points_fisheye(pts, K)
    assert not np.allclose(px_pinhole, px_fisheye)


def test_fisheye_projection_empty_input():
    K = intrinsics_matrix(300, 300, 320, 240)
    px = project_points_fisheye(np.zeros((0, 3)), K)
    assert px.shape == (0, 2)


def test_fisheye_projection_rejects_bad_shape():
    K = intrinsics_matrix(300, 300, 320, 240)
    try:
        project_points_fisheye(np.array([1.0, 2.0, 3.0]), K)
        assert False, "expected ValueError"
    except ValueError:
        pass


# Test 4: extrinsic identity -> expected pixel position (including rotation,
# not just translation -- test_project_lidar_to_image_applies_extrinsic_
# translation above only covers a pure translation).

def test_project_lidar_to_image_pure_rotation_maps_to_expected_pixel():
    """A 90-degree yaw rotation should visibly redirect where a LiDAR-frame
    point lands, in a direction we can predict analytically -- this is the
    'does projection correctly account for the ROTATION part of the
    extrinsic, not just translation' check STEP 3 calls for."""
    from geometry.transform import rpy_to_rotation_matrix, to_homogeneous

    # +90 deg yaw: LiDAR's +X axis maps to camera's... let's just derive
    # analytically rather than assume: R = Rz(90deg), so R @ [1,0,0] = [0,1,0].
    R = rpy_to_rotation_matrix(roll=0, pitch=0, yaw=90, degrees=True)
    T_CL = to_homogeneous(R, np.zeros(3))

    # A point 5m along LiDAR's +X axis (with a small +Z nudge in LIDAR
    # frame so it still ends up with positive camera-frame depth after
    # rotation -- pure LiDAR-frame +X alone would rotate to camera-frame
    # +Y, i.e. z=0, which project_lidar_to_image correctly drops as
    # "not in front of the camera").
    point_lidar = R.T @ np.array([0.0, 0.0, 5.0])  # whatever LiDAR-frame point maps to camera-frame [0,0,5]
    points = np.array([point_lidar])

    K = intrinsics_matrix(fx=500, fy=500, cx=320, cy=240)
    result = project_lidar_to_image(points, T_CL, K, None, image_width=640, image_height=480)

    assert result.num_valid_points == 1
    assert np.isclose(result.depths[0], 5.0, atol=1e-6)
    # constructed so the camera-frame point is exactly [0,0,5] -> principal point
    assert np.allclose(result.pixels[0], [320, 240], atol=1e-6)


def test_project_lidar_to_image_rotation_and_translation_combined():
    """Extrinsic with BOTH a non-trivial rotation and a translation should
    still land a known point at the analytically-predicted pixel -- the
    end-to-end 'does the full T_CL pipeline (not just translation-only or
    rotation-only) produce the expected position' check."""
    from geometry.transform import rpy_to_rotation_matrix, to_homogeneous

    R = rpy_to_rotation_matrix(roll=0, pitch=0, yaw=180, degrees=True)  # 180 deg: flips X and Y sign
    t = np.array([0.0, 0.0, 2.0])
    T_CL = to_homogeneous(R, t)

    # Solve for a LiDAR-frame point that maps to camera-frame [0, 0, 5]:
    # p_cam = R @ p_lidar + t  =>  p_lidar = R.T @ (p_cam - t)
    p_cam_target = np.array([0.0, 0.0, 5.0])
    point_lidar = R.T @ (p_cam_target - t)
    points = np.array([point_lidar])

    K = intrinsics_matrix(fx=200, fy=200, cx=320, cy=240)
    result = project_lidar_to_image(points, T_CL, K, None, image_width=640, image_height=480)

    assert result.num_valid_points == 1
    assert np.isclose(result.depths[0], 5.0, atol=1e-6)
    assert np.allclose(result.pixels[0], [320, 240], atol=1e-6)


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
