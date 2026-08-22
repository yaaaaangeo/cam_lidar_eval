import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from evaluation.perturbation import (
    evaluate_perturbation_sensitivity, _perturb_translation, _perturb_rotation,
    _perturb_timestamp_points, _compute_axis_sensitivities,
    PerturbationSample, AxisSensitivity,
    DEFAULT_TRANSLATION_DELTAS_M, DEFAULT_ROTATION_DELTAS_DEG, DEFAULT_TIMESTAMP_DELTAS_S,
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


# ---------------------------------------------------------------------------
# Threaded execution (parallelized perturbation samples)
# ---------------------------------------------------------------------------

def test_perturbation_results_deterministic_regardless_of_worker_count():
    # executor.map preserves input order regardless of which thread
    # finishes first, so the result must be identical (sample order,
    # best_sample, classification) whether run with 1 worker or many.
    camera = _make_camera()
    image = _make_image()
    points = _make_base_points_cam_frame()
    lidar_spec = _make_lidar_spec()
    kwargs = dict(image=image, points_lidar=points, T_CL=np.eye(4), camera=camera,
                  lidar_spec=lidar_spec, edge_alignment_kwargs={"depth_jump_threshold_m": 1.0})

    r1 = evaluate_perturbation_sensitivity(**kwargs, max_workers=1)
    r8 = evaluate_perturbation_sensitivity(**kwargs, max_workers=8)

    assert r1.classification == r8.classification
    assert len(r1.samples) == len(r8.samples)
    for s1, s8 in zip(r1.samples, r8.samples):
        assert s1.axis == s8.axis and s1.direction == s8.direction and s1.delta == s8.delta
        assert np.isclose(s1.mean_px, s8.mean_px, equal_nan=True)
    assert np.isclose(r1.baseline_mean_px, r8.baseline_mean_px)
    assert np.isclose(r1.improvement_margin_px, r8.improvement_margin_px)


def test_perturbation_samples_run_concurrently():
    # Prove the thread pool actually overlaps work rather than running
    # samples one at a time: patch evaluate_edge_alignment (as imported
    # into evaluation.perturbation's namespace) with a slow stand-in and
    # compare max_workers=1 (fully sequential) against max_workers=8.
    # Compares relative timing rather than an absolute threshold, since
    # absolute wall-clock time is sensitive to CI/sandbox overhead
    # (thread creation, single-core virtualization, etc.) in a way that
    # makes a fixed-time assertion flaky; the *ratio* between the two
    # runs isolates the effect of concurrency from that noise.
    import time
    from unittest.mock import patch
    import evaluation.perturbation as pert_module
    from evaluation.edge_alignment import evaluate_edge_alignment as real_eval

    camera = _make_camera()
    image = _make_image()
    points = _make_base_points_cam_frame()
    lidar_spec = _make_lidar_spec()

    def slow_eval(*args, **kwargs):
        time.sleep(0.03)
        return real_eval(*args, **kwargs)

    def timed_run(max_workers):
        with patch.object(pert_module, "evaluate_edge_alignment", side_effect=slow_eval):
            t0 = time.perf_counter()
            pert_module.evaluate_perturbation_sensitivity(
                image, points, np.eye(4), camera, lidar_spec,
                edge_alignment_kwargs={"depth_jump_threshold_m": 1.0}, max_workers=max_workers,
            )
            return time.perf_counter() - t0

    sequential_time = timed_run(max_workers=1)
    parallel_time = timed_run(max_workers=8)

    # Generous margin (parallel must be no more than 80% of sequential) --
    # this only needs to detect "concurrency is happening at all", not
    # measure a specific speedup factor.
    assert parallel_time < sequential_time * 0.8, (
        f"expected concurrent speedup: sequential={sequential_time:.2f}s, parallel={parallel_time:.2f}s"
    )


# ---------------------------------------------------------------------------
# STEP11 -- Calibration Sensitivity Analysis: delta grids, per-axis
# sensitivity ranking, and timestamp axis
# ---------------------------------------------------------------------------

def test_default_delta_grids_match_step11_spec():
    assert DEFAULT_ROTATION_DELTAS_DEG == (0.05, 0.1, 0.2, 0.5, 1.0)
    assert DEFAULT_TRANSLATION_DELTAS_M == (0.001, 0.005, 0.010, 0.020)
    assert DEFAULT_TIMESTAMP_DELTAS_S == (0.005, 0.010, 0.020, 0.050, 0.100)


def _sample(axis, direction, delta, mean_px):
    return PerturbationSample(axis=axis, direction=direction, delta=delta, mean_px=mean_px, valid=True)


def test_compute_axis_sensitivities_classifies_high_when_small_delta_exceeds_floor():
    baseline = 1.0
    floor_px = 1.0
    samples = [
        _sample("yaw_deg", "+", 0.05, baseline + 5.0), _sample("yaw_deg", "-", 0.05, baseline + 5.0),
        _sample("yaw_deg", "+", 1.0, baseline + 8.0), _sample("yaw_deg", "-", 1.0, baseline + 8.0),
    ]
    result = _compute_axis_sensitivities(samples, baseline, floor_px)
    assert len(result) == 1
    assert result[0].classification == "HIGH"


def test_compute_axis_sensitivities_classifies_medium_when_only_large_delta_exceeds_floor():
    baseline = 1.0
    floor_px = 1.0
    samples = [
        _sample("pitch_deg", "+", 0.05, baseline + 0.1), _sample("pitch_deg", "-", 0.05, baseline + 0.1),
        _sample("pitch_deg", "+", 1.0, baseline + 3.0), _sample("pitch_deg", "-", 1.0, baseline + 3.0),
    ]
    result = _compute_axis_sensitivities(samples, baseline, floor_px)
    assert result[0].classification == "MEDIUM"


def test_compute_axis_sensitivities_classifies_low_when_nothing_exceeds_floor():
    baseline = 1.0
    floor_px = 1.0
    samples = [
        _sample("tz", "+", 0.001, baseline + 0.01), _sample("tz", "-", 0.001, baseline + 0.01),
        _sample("tz", "+", 0.020, baseline + 0.1), _sample("tz", "-", 0.020, baseline + 0.1),
    ]
    result = _compute_axis_sensitivities(samples, baseline, floor_px)
    assert result[0].classification == "LOW"


def test_compute_axis_sensitivities_sorted_high_to_low():
    baseline = 1.0
    floor_px = 1.0
    samples = [
        _sample("tz", "+", 0.001, baseline + 0.01), _sample("tz", "-", 0.001, baseline + 0.01),
        _sample("yaw_deg", "+", 0.05, baseline + 5.0), _sample("yaw_deg", "-", 0.05, baseline + 5.0),
        _sample("pitch_deg", "+", 0.05, baseline + 0.1), _sample("pitch_deg", "-", 0.05, baseline + 0.1),
        _sample("pitch_deg", "+", 1.0, baseline + 3.0), _sample("pitch_deg", "-", 1.0, baseline + 3.0),
    ]
    result = _compute_axis_sensitivities(samples, baseline, floor_px)
    classes = [r.classification for r in result]
    assert classes == sorted(classes, key=lambda c: {"HIGH": 2, "MEDIUM": 1, "LOW": 0}[c], reverse=True)
    assert result[0].axis == "yaw_deg"


def test_compute_axis_sensitivities_skips_axis_with_no_valid_samples():
    invalid = PerturbationSample(axis="roll_deg", direction="+", delta=0.05, mean_px=float("nan"), valid=False)
    result = _compute_axis_sensitivities([invalid], 1.0, 1.0)
    assert result == []


def test_perturb_timestamp_points_matches_direct_deskew_call():
    from motion.deskew import deskew_points_constant_velocity
    points = np.random.default_rng(0).uniform(-5, 5, size=(20, 3))
    v = np.array([2.0, 0.0, 0.0])
    w = np.array([0.0, 0.0, 0.3])
    delta_s = 0.02

    shifted = _perturb_timestamp_points(points, delta_s, v, w)
    expected = deskew_points_constant_velocity(
        points, scan_period_s=1.0, linear_velocity_mps=v, angular_velocity_rps=w,
        point_times_s=np.zeros(20), reference_time_s=delta_s,
    ).points_deskewed
    assert np.allclose(shifted, expected)


def test_perturb_timestamp_points_zero_velocity_is_noop():
    points = np.random.default_rng(1).uniform(-5, 5, size=(10, 3))
    shifted = _perturb_timestamp_points(points, 0.05, np.zeros(3), np.zeros(3))
    assert np.allclose(shifted, points)


def test_evaluate_perturbation_sensitivity_skips_timestamp_axis_by_default():
    camera = _make_camera()
    image, points = _make_robust_depth_step_scene()
    result = evaluate_perturbation_sensitivity(
        image=image, points_lidar=points, T_CL=np.eye(4), camera=camera,
        lidar_spec=_make_lidar_spec(),
        translation_deltas_m=(0.01,), rotation_deltas_deg=(0.1,),
        edge_alignment_kwargs={"depth_jump_threshold_m": 1.0},
    )
    assert result.timestamp_sensitivity_computed is False
    assert not any(s.axis == "timestamp" for s in result.samples)
    assert any("Timestamp sensitivity not computed" in w for w in result.warnings)


def test_evaluate_perturbation_sensitivity_computes_timestamp_axis_when_velocity_given():
    camera = _make_camera()
    image, points = _make_robust_depth_step_scene()
    result = evaluate_perturbation_sensitivity(
        image=image, points_lidar=points, T_CL=np.eye(4), camera=camera,
        lidar_spec=_make_lidar_spec(),
        translation_deltas_m=(0.01,), rotation_deltas_deg=(0.1,),
        timestamp_deltas_s=(0.01, 0.02),
        linear_velocity_mps=np.array([3.0, 0.0, 0.0]),
        edge_alignment_kwargs={"depth_jump_threshold_m": 1.0},
    )
    assert result.timestamp_sensitivity_computed is True
    timestamp_samples = [s for s in result.samples if s.axis == "timestamp"]
    assert len(timestamp_samples) == 2 * 2  # 2 deltas x 2 directions


def test_evaluate_perturbation_sensitivity_result_has_axis_sensitivities():
    # Uses the dense regular-grid scene (not _make_robust_depth_step_scene's
    # sparse random scatter) -- roll/pitch perturbations on a purely
    # vertical depth boundary can drop a sparse random scene's edge-point
    # count below min_edge_points even at STEP11's smallest configured
    # delta (0.05deg), which is a real property of that scene's local
    # point density near the boundary, not a bug in axis sensitivity
    # itself; the dense grid here stays well above the threshold at every
    # configured delta so all 6 axes produce valid samples.
    camera = _make_camera()
    image = _make_image()
    points = _make_base_points_cam_frame()
    result = evaluate_perturbation_sensitivity(
        image=image, points_lidar=points, T_CL=np.eye(4), camera=camera,
        lidar_spec=_make_lidar_spec(),
        translation_deltas_m=(0.001, 0.02), rotation_deltas_deg=(0.05, 1.0),
        edge_alignment_kwargs={"depth_jump_threshold_m": 1.0},
    )
    assert len(result.axis_sensitivities) == 6  # 3 translation + 3 rotation
    assert all(a.classification in ("HIGH", "MEDIUM", "LOW") for a in result.axis_sensitivities)


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
