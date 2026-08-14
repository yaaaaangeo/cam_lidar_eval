import sys
import os
import math

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from geometry.transform import (
    rpy_to_rotation_matrix,
    quaternion_to_rotation_matrix,
    is_valid_rotation_matrix,
    to_homogeneous,
    invert_transform,
    compose_transforms,
    transform_points,
)


def test_identity_rpy_is_identity():
    R = rpy_to_rotation_matrix(0, 0, 0)
    assert np.allclose(R, np.eye(3))


def test_yaw_90deg_rotates_x_to_y():
    R = rpy_to_rotation_matrix(0, 0, 90, degrees=True)
    p = np.array([1.0, 0.0, 0.0])
    p_rot = R @ p
    assert np.allclose(p_rot, [0, 1, 0], atol=1e-9)


def test_roll_90deg_rotates_y_to_z():
    R = rpy_to_rotation_matrix(90, 0, 0, degrees=True)
    p = np.array([0.0, 1.0, 0.0])
    p_rot = R @ p
    assert np.allclose(p_rot, [0, 0, 1], atol=1e-9)


def test_rpy_produces_valid_rotation_matrix():
    R = rpy_to_rotation_matrix(12.3, -45.6, 78.9, degrees=True)
    ok, diag = is_valid_rotation_matrix(R)
    assert ok, diag


def test_quaternion_identity():
    R = quaternion_to_rotation_matrix(0, 0, 0, 1)
    assert np.allclose(R, np.eye(3), atol=1e-9)


def test_quaternion_matches_rpy_for_yaw_90():
    # quaternion for 90deg yaw about Z: (0, 0, sin(45deg), cos(45deg))
    s = math.sin(math.radians(45))
    c = math.cos(math.radians(45))
    R_quat = quaternion_to_rotation_matrix(0, 0, s, c)
    R_rpy = rpy_to_rotation_matrix(0, 0, 90, degrees=True)
    assert np.allclose(R_quat, R_rpy, atol=1e-9)


def test_quaternion_unnormalized_still_works():
    # scale a valid quaternion by 2x -- should normalize internally
    s = math.sin(math.radians(45)) * 2
    c = math.cos(math.radians(45)) * 2
    R = quaternion_to_rotation_matrix(0, 0, s, c)
    ok, diag = is_valid_rotation_matrix(R)
    assert ok, diag


def test_quaternion_zero_norm_raises():
    try:
        quaternion_to_rotation_matrix(0, 0, 0, 0)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_is_valid_rotation_matrix_rejects_reflection():
    R = np.eye(3)
    R[0, 0] = -1  # reflection, det = -1
    ok, diag = is_valid_rotation_matrix(R)
    assert not ok
    assert not diag["det_ok"]


def test_is_valid_rotation_matrix_rejects_non_orthogonal():
    R = np.eye(3) * 1.1  # det != 1 and not orthogonal
    ok, diag = is_valid_rotation_matrix(R)
    assert not ok


def test_to_homogeneous_shape_and_content():
    R = rpy_to_rotation_matrix(0, 0, 90, degrees=True)
    t = np.array([1.0, 2.0, 3.0])
    T = to_homogeneous(R, t)
    assert T.shape == (4, 4)
    assert np.allclose(T[:3, :3], R)
    assert np.allclose(T[:3, 3], t)
    assert np.allclose(T[3, :], [0, 0, 0, 1])


def test_invert_transform_round_trip():
    R = rpy_to_rotation_matrix(10, 20, 30, degrees=True)
    t = np.array([0.5, -1.2, 3.0])
    T = to_homogeneous(R, t)
    T_inv = invert_transform(T)
    identity_check = T @ T_inv
    assert np.allclose(identity_check, np.eye(4), atol=1e-9)


def test_compose_transforms_matches_matrix_multiplication():
    T_a = to_homogeneous(rpy_to_rotation_matrix(0, 0, 90, degrees=True), [1, 0, 0])
    T_b = to_homogeneous(rpy_to_rotation_matrix(0, 0, 0, degrees=True), [0, 1, 0])
    T_composed = compose_transforms(T_a, T_b)
    assert np.allclose(T_composed, T_a @ T_b)


def test_transform_points_identity():
    T = np.eye(4)
    pts = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    out = transform_points(T, pts)
    assert np.allclose(out, pts)


def test_transform_points_translation_only():
    T = to_homogeneous(np.eye(3), [10, 0, 0])
    pts = np.array([[0.0, 0.0, 0.0]])
    out = transform_points(T, pts)
    assert np.allclose(out, [[10, 0, 0]])


def test_transform_points_rotation_and_translation():
    R = rpy_to_rotation_matrix(0, 0, 90, degrees=True)
    T = to_homogeneous(R, [1, 0, 0])
    pts = np.array([[1.0, 0.0, 0.0]])
    out = transform_points(T, pts)
    # rotate (1,0,0) by 90deg yaw -> (0,1,0), then translate by (1,0,0) -> (1,1,0)
    assert np.allclose(out, [[1, 1, 0]], atol=1e-9)


def test_transform_points_accepts_3x4():
    T = to_homogeneous(np.eye(3), [1, 2, 3])[:3, :]
    pts = np.array([[0.0, 0.0, 0.0]])
    out = transform_points(T, pts)
    assert np.allclose(out, [[1, 2, 3]])


def test_transform_points_rejects_bad_shape():
    T = np.eye(4)
    try:
        transform_points(T, np.array([1.0, 2.0, 3.0]))  # 1D, not (N,3)
        assert False, "expected ValueError"
    except ValueError:
        pass


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
