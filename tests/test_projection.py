import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from geometry.projection import (
    intrinsics_matrix,
    plumb_bob_dist_coeffs,
    fisheye_dist_coeffs,
    project_points_pinhole,
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
