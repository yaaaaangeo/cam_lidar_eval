import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from input.extrinsic import (
    ExtrinsicRaw,
    load_extrinsic,
    verify_extrinsic,
)
from geometry.transform import rpy_to_rotation_matrix, invert_transform, to_homogeneous


def test_load_extrinsic_rpy_deg_lidar_to_camera():
    raw = ExtrinsicRaw(
        parent="lidar", child="camera",
        translation=(0.1, -0.05, 0.2),
        rotation=(0.0, 0.0, 90.0),
        rotation_format="rpy_deg",
        unit="m",
    )
    model = load_extrinsic(raw)
    assert model.T_CL.shape == (4, 4)
    expected_R = rpy_to_rotation_matrix(0, 0, 90, degrees=True)
    assert np.allclose(model.T_CL[:3, :3], expected_R, atol=1e-9)
    assert np.allclose(model.T_CL[:3, 3], [0.1, -0.05, 0.2])


def test_load_extrinsic_direction_flip_camera_to_lidar_is_inverted():
    """
    If the user provides T_LC (parent=camera, child=lidar) instead of T_CL,
    load_extrinsic must invert it so the returned T_CL always means the same
    thing: p_cam = T_CL @ p_lidar. This is the core defense against the
    'T_CL vs T_LC swapped' mistake.
    """
    R = rpy_to_rotation_matrix(10, 20, 30, degrees=True)
    t = np.array([0.3, 0.1, -0.2])
    T_LC = to_homogeneous(R, t)  # camera -> lidar direction

    raw = ExtrinsicRaw(
        parent="camera", child="lidar",
        translation=tuple(t.tolist()),
        rotation=(10.0, 20.0, 30.0),
        rotation_format="rpy_deg",
        unit="m",
    )
    model = load_extrinsic(raw)
    expected_T_CL = invert_transform(T_LC)
    assert np.allclose(model.T_CL, expected_T_CL, atol=1e-9)


def test_load_extrinsic_quaternion():
    import math
    s = math.sin(math.radians(45))
    c = math.cos(math.radians(45))
    raw = ExtrinsicRaw(
        parent="lidar", child="camera",
        translation=(0, 0, 0),
        rotation=(0, 0, s, c),
        rotation_format="quaternion",
    )
    model = load_extrinsic(raw)
    expected_R = rpy_to_rotation_matrix(0, 0, 90, degrees=True)
    assert np.allclose(model.T_CL[:3, :3], expected_R, atol=1e-6)


def test_load_extrinsic_matrix4x4():
    R = rpy_to_rotation_matrix(0, 0, 45, degrees=True)
    t = np.array([1, 2, 3])
    T = to_homogeneous(R, t)
    raw = ExtrinsicRaw(
        parent="lidar", child="camera",
        translation=(0, 0, 0),  # ignored when rotation_format is matrix4x4
        rotation=T.tolist(),
        rotation_format="matrix4x4",
    )
    model = load_extrinsic(raw)
    assert np.allclose(model.T_CL, T)


def test_load_extrinsic_unit_conversion_cm():
    raw = ExtrinsicRaw(
        parent="lidar", child="camera",
        translation=(10.0, 20.0, 30.0),  # cm
        rotation=(0, 0, 0),
        rotation_format="rpy_deg",
        unit="cm",
    )
    model = load_extrinsic(raw)
    assert np.allclose(model.T_CL[:3, 3], [0.1, 0.2, 0.3])  # converted to meters


def test_load_extrinsic_unit_conversion_mm():
    raw = ExtrinsicRaw(
        parent="lidar", child="camera",
        translation=(100.0, 200.0, 300.0),  # mm
        rotation=(0, 0, 0),
        rotation_format="rpy_deg",
        unit="mm",
    )
    model = load_extrinsic(raw)
    assert np.allclose(model.T_CL[:3, 3], [0.1, 0.2, 0.3])


def test_load_extrinsic_invalid_parent_child_raises():
    raw = ExtrinsicRaw(
        parent="lidar", child="lidar",  # invalid combo
        translation=(0, 0, 0),
        rotation=(0, 0, 0),
        rotation_format="rpy_deg",
    )
    try:
        load_extrinsic(raw)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_verify_extrinsic_passes_for_valid_transform():
    raw = ExtrinsicRaw(
        parent="lidar", child="camera",
        translation=(0.1, 0.0, 0.2),
        rotation=(1.0, 2.0, 3.0),
        rotation_format="rpy_deg",
    )
    model = load_extrinsic(raw)
    report = verify_extrinsic(model)
    assert report.all_passed, report.failed_items()


def test_verify_extrinsic_fails_for_non_finite_translation():
    raw = ExtrinsicRaw(
        parent="lidar", child="camera",
        translation=(float("nan"), 0.0, 0.0),
        rotation=(0, 0, 0),
        rotation_format="rpy_deg",
    )
    model = load_extrinsic(raw)
    report = verify_extrinsic(model)
    assert not report.all_passed
    names = [i.name for i in report.failed_items()]
    assert "translation_finite" in names


def test_verify_extrinsic_flags_implausible_translation_magnitude():
    """Catches the classic mm-treated-as-m unit mistake: e.g. a rig with a
    1500m 'baseline' is obviously wrong for a camera-lidar pair."""
    raw = ExtrinsicRaw(
        parent="lidar", child="camera",
        translation=(1500.0, 0.0, 0.0),
        rotation=(0, 0, 0),
        rotation_format="rpy_deg",
        unit="m",
    )
    model = load_extrinsic(raw)
    report = verify_extrinsic(model, max_plausible_translation_m=100.0)
    names = [i.name for i in report.failed_items()]
    assert "translation_magnitude_plausible" in names


def test_verify_extrinsic_flags_invalid_rotation_matrix():
    raw = ExtrinsicRaw(
        parent="lidar", child="camera",
        translation=(0, 0, 0),
        rotation=[[2, 0, 0], [0, 1, 0], [0, 0, 1]],  # not a valid rotation
        rotation_format="matrix3x3",
    )
    model = load_extrinsic(raw)
    report = verify_extrinsic(model)
    assert not report.all_passed
    names = [i.name for i in report.failed_items()]
    assert "rotation_valid" in names


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
