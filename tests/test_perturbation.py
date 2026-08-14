import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from evaluation.perturbation import (
    evaluate_perturbation_sensitivity, _perturb_translation, _perturb_rotation,
)
from tests.test_holdout_consistency import _make_camera, _make_image, _make_lidar_spec, _make_base_points_cam_frame


def test_perturb_translation_shifts_correct_axis():
    T = np.eye(4)
    T_perturbed = _perturb_translation(T, axis_idx=0, delta=0.05)
    assert np.isclose(T_perturbed[0, 3], 0.05)
    assert np.isclose(T_perturbed[1, 3], 0.0)
    assert np.allclose(T_perturbed[:3, :3], np.eye(3))
    assert np.allclose(T[:3, 3], [0, 0, 0])  # original untouched


def test_perturb_rotation_applies_small_rotation():
    T = np.eye(4)
    T_perturbed = _perturb_rotation(T, "yaw_deg", 5.0)
    assert not np.allclose(T_perturbed[:3, :3], np.eye(3))
    assert np.allclose(T_perturbed[:3, 3], [0, 0, 0])
    assert np.allclose(T[:3, :3], np.eye(3))  # original untouched


def _make_robust_depth_step_scene(n=60000, seed=42):
    """
    A randomly-sampled (not grid-based) depth-step scene, used specifically
    for perturbation tests. The grid-based scene shared with M2/M3/M4 tests
    (_make_base_points_cam_frame) places points on an exact regular lattice,
    which makes the near/far depth assignment threshold (u < cx) align with
    point positions in a way that's hypersensitive to sub-pixel nudges --
    a tiny perturbation can snap many points across the threshold at once,
    producing discontinuous, asymmetric behavior unrepresentative of real
    sensor data. Random sampling avoids this quantization artifact.
    """
    import cv2
    width, height = 640, 480
    fx = fy = 500.0
    cx, cy = 320.0, 240.0
    z_near, z_far = 5.0, 10.0

    image = np.zeros((height, width), dtype=np.uint8)
    image[:, int(cx):] = 255
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    rng = np.random.RandomState(seed)
    uu = rng.uniform(0, width - 1, n)
    vv = rng.uniform(0, height - 1, n)
    zz = np.where(uu < cx, z_near, z_far)
    xx = (uu - cx) * zz / fx
    yy = (vv - cy) * zz / fy
    points = np.stack([xx, yy, zz], axis=1)
    return image, points


def test_perturbation_at_local_minimum_when_t_is_already_correct():
    """
    When T_CL exactly matches how the synthetic scene was constructed,
    nudging it in any direction should only make things worse (or at
    least not meaningfully better) -- i.e. it should register as being at
    a local minimum.
    """
    camera = _make_camera()
    image, points = _make_robust_depth_step_scene()
    result = evaluate_perturbation_sensitivity(
        image=image, points_lidar=points, T_CL=np.eye(4), camera=camera,
        lidar_spec=_make_lidar_spec(),
        translation_deltas_m=(0.01,), rotation_deltas_deg=(0.1,),
        edge_alignment_kwargs={"depth_jump_threshold_m": 1.0},
    )
    assert result.classification != "FAIL", result.warnings
    assert result.is_local_minimum, (
        f"baseline={result.baseline_mean_px}, best={result.best_sample.mean_px if result.best_sample else None}"
    )
    assert len(result.samples) == 12  # 3 translation + 3 rotation axes, x1 delta x2 directions


def test_perturbation_not_at_local_minimum_when_t_is_off():
    """
    Start from a slightly WRONG T_CL (offset from how the scene was built).
    At least one perturbation nudging back toward the true alignment
    should reduce error -- i.e. NOT a local minimum.
    """
    camera = _make_camera()
    image, points = _make_robust_depth_step_scene()
    T_off = np.eye(4)
    T_off[0, 3] = 0.05  # offset by 5cm

    result = evaluate_perturbation_sensitivity(
        image=image, points_lidar=points, T_CL=T_off, camera=camera,
        lidar_spec=_make_lidar_spec(),
        translation_deltas_m=(0.02, 0.05), rotation_deltas_deg=(0.1,),
        edge_alignment_kwargs={"depth_jump_threshold_m": 1.0},
    )
    assert result.classification != "FAIL", result.warnings
    assert not result.is_local_minimum
    assert result.improvement_margin_px > 0
    # the best sample should be a translation-x nudge in the direction that
    # cancels the 5cm offset (i.e. negative, back toward zero)
    assert result.best_sample.axis == "tx"
    assert result.best_sample.direction == "-"


def test_perturbation_sample_count_matches_axes_and_deltas():
    camera = _make_camera()
    image, points = _make_robust_depth_step_scene()
    result = evaluate_perturbation_sensitivity(
        image=image, points_lidar=points, T_CL=np.eye(4), camera=camera,
        lidar_spec=_make_lidar_spec(),
        translation_deltas_m=(0.01, 0.02), rotation_deltas_deg=(0.1, 0.2),
        edge_alignment_kwargs={"depth_jump_threshold_m": 1.0},
    )
    # 6 axes total (3 translation + 3 rotation) x 2 deltas x 2 directions = 24
    assert len(result.samples) == 24


def test_perturbation_fails_gracefully_on_failed_baseline():
    camera = _make_camera()
    blank = np.zeros((camera.height, camera.width, 3), dtype=np.uint8)
    points = _make_base_points_cam_frame()
    result = evaluate_perturbation_sensitivity(
        image=blank, points_lidar=points, T_CL=np.eye(4), camera=camera,
        lidar_spec=_make_lidar_spec(), edge_alignment_kwargs={"depth_jump_threshold_m": 1.0},
    )
    assert result.classification == "FAIL"
    assert result.samples == []
    assert result.best_sample is None


def test_perturbation_baseline_mean_px_matches_direct_m2_call():
    from evaluation.edge_alignment import evaluate_edge_alignment
    camera = _make_camera()
    image = _make_image()
    points = _make_base_points_cam_frame()
    direct = evaluate_edge_alignment(image=image, points_lidar=points, T_CL=np.eye(4),
                                      camera=camera, lidar_spec=_make_lidar_spec(),
                                      depth_jump_threshold_m=1.0)
    result = evaluate_perturbation_sensitivity(
        image=image, points_lidar=points, T_CL=np.eye(4), camera=camera,
        lidar_spec=_make_lidar_spec(), translation_deltas_m=(0.01,), rotation_deltas_deg=(0.1,),
        edge_alignment_kwargs={"depth_jump_threshold_m": 1.0},
    )
    assert abs(result.baseline_mean_px - direct.mean_px) < 1e-9


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
