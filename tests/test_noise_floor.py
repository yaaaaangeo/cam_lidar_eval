"""
Unit tests for quality/noise_floor.py

Run with: python -m pytest tests/test_noise_floor.py -v
(or just: python tests/test_noise_floor.py, which runs a manual harness
 below without requiring pytest to be installed)
"""

import math
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from quality.noise_floor import (
    LidarSensorSpecForFloor,
    resolve_floor_inputs,
    floor_angular,
    floor_range,
    floor_edge,
    compute_floor,
    compute_floor_array,
    compute_floor_breakdown,
    classify,
    multiplier_thresholds,
    DEFAULT_ANGULAR_RESOLUTION_DEG,
    DEFAULT_RANGE_ACCURACY_M,
    DEFAULT_EDGE_LOCALIZATION_FLOOR_PX,
)


def identity_T(tx=0.1, ty=0.0, tz=0.05):
    T = np.eye(4)
    T[:3, 3] = [tx, ty, tz]
    return T


# ---------------------------------------------------------------------------
# Fallback resolution tests
# ---------------------------------------------------------------------------

def test_resolve_uses_explicit_horizontal_vertical_worst_case():
    spec = LidarSensorSpecForFloor(horizontal_resolution_deg=0.1, vertical_resolution_deg=0.4)
    inputs = resolve_floor_inputs(fx_px=1000.0, T_CL=identity_T(), lidar_spec=spec)
    assert not inputs.used_angular_fallback
    assert math.isclose(math.degrees(inputs.theta_res_rad), 0.4, rel_tol=1e-6)


def test_resolve_approximates_from_channels_and_fov():
    spec = LidarSensorSpecForFloor(channels=32, vertical_fov_deg=40.0)
    inputs = resolve_floor_inputs(fx_px=1000.0, T_CL=identity_T(), lidar_spec=spec)
    assert inputs.used_angular_fallback  # approximation counts as fallback
    assert math.isclose(math.degrees(inputs.theta_res_rad), 40.0 / 32, rel_tol=1e-6)
    assert any("approximated" in w for w in inputs.fallback_warnings)


def test_resolve_vertical_approximation_used_even_when_horizontal_given():
    """
    Regression test: previously, providing horizontal_resolution_deg alone
    would silently skip the channels+FOV vertical approximation entirely,
    even when the vertical axis (unresolved) was actually coarser. The
    worst-case axis must always be considered.
    """
    spec = LidarSensorSpecForFloor(
        horizontal_resolution_deg=0.1,   # fine horizontal resolution
        channels=32,
        vertical_fov_deg=40.0,           # -> approx vertical = 1.25 deg, much coarser
    )
    inputs = resolve_floor_inputs(fx_px=1000.0, T_CL=identity_T(), lidar_spec=spec)
    assert inputs.used_angular_fallback  # vertical had to be approximated
    # worst case must be the coarser (larger) vertical approximation, not the
    # finer horizontal value
    assert math.isclose(math.degrees(inputs.theta_res_rad), 40.0 / 32, rel_tol=1e-6)


def test_resolve_horizontal_only_no_vertical_info_at_all():
    """If only horizontal is given and vertical can't be approximated either,
    we use horizontal alone (no fallback needed for the resolvable axis)."""
    spec = LidarSensorSpecForFloor(horizontal_resolution_deg=0.15)
    inputs = resolve_floor_inputs(fx_px=1000.0, T_CL=identity_T(), lidar_spec=spec)
    assert not inputs.used_angular_fallback
    assert math.isclose(math.degrees(inputs.theta_res_rad), 0.15, rel_tol=1e-6)


def test_resolve_falls_back_to_default_when_nothing_provided():
    spec = LidarSensorSpecForFloor()
    inputs = resolve_floor_inputs(fx_px=1000.0, T_CL=identity_T(), lidar_spec=spec)
    assert inputs.used_angular_fallback
    assert math.isclose(math.degrees(inputs.theta_res_rad), DEFAULT_ANGULAR_RESOLUTION_DEG)
    assert inputs.used_range_accuracy_fallback
    assert math.isclose(inputs.sigma_r_m, DEFAULT_RANGE_ACCURACY_M)
    assert inputs.used_edge_floor_fallback
    assert math.isclose(inputs.floor_edge_px, DEFAULT_EDGE_LOCALIZATION_FLOOR_PX)
    assert len(inputs.fallback_warnings) == 3


def test_resolve_explicit_range_accuracy_no_fallback():
    spec = LidarSensorSpecForFloor(
        horizontal_resolution_deg=0.2, range_accuracy_m=0.03
    )
    inputs = resolve_floor_inputs(fx_px=1000.0, T_CL=identity_T(), lidar_spec=spec)
    assert not inputs.used_range_accuracy_fallback
    assert math.isclose(inputs.sigma_r_m, 0.03)


def test_baseline_extracted_from_translation_norm():
    T = identity_T(tx=0.3, ty=0.4, tz=0.0)  # 3-4-5 triangle -> norm 0.5
    spec = LidarSensorSpecForFloor(horizontal_resolution_deg=0.2, range_accuracy_m=0.02)
    inputs = resolve_floor_inputs(fx_px=1000.0, T_CL=T, lidar_spec=spec)
    assert math.isclose(inputs.baseline_m, 0.5, rel_tol=1e-9)


def test_invalid_fx_raises():
    spec = LidarSensorSpecForFloor()
    try:
        resolve_floor_inputs(fx_px=0.0, T_CL=identity_T(), lidar_spec=spec)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_invalid_T_shape_raises():
    spec = LidarSensorSpecForFloor()
    try:
        resolve_floor_inputs(fx_px=1000.0, T_CL=np.eye(3), lidar_spec=spec)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_accepts_3x4_T():
    spec = LidarSensorSpecForFloor(horizontal_resolution_deg=0.2)
    T = identity_T()[:3, :]  # 3x4
    inputs = resolve_floor_inputs(fx_px=1000.0, T_CL=T, lidar_spec=spec)
    assert inputs.baseline_m > 0


# ---------------------------------------------------------------------------
# Term-level behavior tests
# ---------------------------------------------------------------------------

def make_inputs(fx=1000.0, theta_res_deg=0.2, baseline=0.1, sigma_r=0.02, floor_edge_px=0.5):
    from quality.noise_floor import FloorInputs
    return FloorInputs(
        fx_px=fx,
        theta_res_rad=math.radians(theta_res_deg),
        baseline_m=baseline,
        sigma_r_m=sigma_r,
        floor_edge_px=floor_edge_px,
    )


def test_floor_angular_is_distance_independent():
    inputs = make_inputs()
    v1 = floor_angular(inputs)
    v2 = floor_angular(inputs)  # no z argument at all -> literally constant
    assert v1 == v2
    expected = 1000.0 * math.radians(0.2)
    assert math.isclose(v1, expected)


def test_floor_range_decreases_with_distance_squared():
    inputs = make_inputs()
    near = floor_range(inputs, z_m=5.0)
    far = floor_range(inputs, z_m=10.0)
    # doubling distance -> quarter the range-noise contribution
    assert math.isclose(near / far, 4.0, rel_tol=1e-9)


def test_floor_range_zero_or_negative_z_raises():
    inputs = make_inputs()
    for bad_z in (0.0, -1.0):
        try:
            floor_range(inputs, z_m=bad_z)
            assert False, "expected ValueError"
        except ValueError:
            pass


def test_floor_edge_is_constant():
    inputs = make_inputs(floor_edge_px=0.7)
    assert floor_edge(inputs) == 0.7


def test_compute_floor_is_quadrature_sum():
    inputs = make_inputs()
    z = 8.0
    t1 = floor_angular(inputs)
    t2 = floor_range(inputs, z)
    t3 = floor_edge(inputs)
    expected = math.sqrt(t1**2 + t2**2 + t3**2)
    assert math.isclose(compute_floor(inputs, z), expected)


# ---------------------------------------------------------------------------
# STEP7 -- compute_floor_array (vectorized, per-point noise/uncertainty model)
# ---------------------------------------------------------------------------

def test_compute_floor_array_matches_scalar_pointwise():
    inputs = make_inputs()
    z_values = np.array([2.0, 5.0, 8.0, 20.0, 50.0])
    array_result = compute_floor_array(inputs, z_values)
    scalar_result = np.array([compute_floor(inputs, float(z)) for z in z_values])
    assert np.allclose(array_result, scalar_result)


def test_compute_floor_array_decreases_with_depth():
    # floor_range's contribution shrinks as 1/Z^2, so floor(Z) should be
    # monotonically non-increasing with depth (same "distance_independent
    # angular/edge terms + shrinking range term" logic as the scalar
    # floor_monotonic_decrease_with_distance test below).
    inputs = make_inputs()
    z_values = np.array([1.0, 2.0, 5.0, 10.0, 30.0, 100.0])
    floors = compute_floor_array(inputs, z_values)
    assert np.all(np.diff(floors) <= 1e-9)


def test_compute_floor_array_empty_input():
    inputs = make_inputs()
    result = compute_floor_array(inputs, np.zeros(0))
    assert result.shape == (0,)


def test_compute_floor_array_rejects_nonpositive_depth():
    inputs = make_inputs()
    try:
        compute_floor_array(inputs, np.array([5.0, 0.0, 10.0]))
        assert False, "expected ValueError"
    except ValueError:
        pass
    try:
        compute_floor_array(inputs, np.array([5.0, -1.0, 10.0]))
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_compute_floor_array_same_point_a_vs_point_b_example():
    """The exact motivating example from evaluation_metric_spec.md's STEP7:
    two points with the SAME actual pixel error but different depths (and
    therefore different expected noise) should normalize to very
    different normalized_error values -- Point A (large floor, closer to
    the sensor's angular/edge-dominated regime) reads as unremarkable,
    Point B (tiny floor, far away where range noise has shrunk to
    nothing) reads as a real problem."""
    inputs = make_inputs()
    z_near = np.array([2.0])   # larger floor(Z) here (range term still significant)
    z_far = np.array([80.0])   # smaller floor(Z) here (range term negligible at 1/Z^2)
    floor_near = compute_floor_array(inputs, z_near)[0]
    floor_far = compute_floor_array(inputs, z_far)[0]
    assert floor_near > floor_far

    actual_error_px = 1.8
    normalized_near = actual_error_px / floor_near
    normalized_far = actual_error_px / floor_far
    assert normalized_far > normalized_near, (
        "the same raw pixel error at a farther depth (smaller floor) should "
        "normalize to a LARGER (more concerning) ratio"
    )


def test_compute_floor_breakdown_identifies_dominant_term_far_range():
    # At very large distance, range term should vanish, angular/edge dominate.
    inputs = make_inputs(theta_res_deg=1.0)  # make angular clearly larger than edge
    breakdown = compute_floor_breakdown(inputs, z_m=1000.0)
    assert breakdown.term_range_px < 1e-4  # negligible relative to angular/edge terms
    assert breakdown.dominant_term == "angular"


def test_compute_floor_breakdown_identifies_dominant_term_close_range_large_baseline():
    inputs = make_inputs(baseline=2.0, sigma_r=0.05, theta_res_deg=0.05)
    breakdown = compute_floor_breakdown(inputs, z_m=1.0)
    assert breakdown.dominant_term == "range"


def test_floor_monotonic_decrease_with_distance():
    inputs = make_inputs()
    zs = [1.0, 2.0, 5.0, 10.0, 50.0]
    values = [compute_floor(inputs, z) for z in zs]
    assert all(values[i] > values[i + 1] for i in range(len(values) - 1))
    # and it should asymptote toward sqrt(angular^2 + edge^2), never below it
    asymptote = math.sqrt(floor_angular(inputs) ** 2 + floor_edge(inputs) ** 2)
    assert values[-1] > asymptote


# ---------------------------------------------------------------------------
# Threshold / classification tests
# ---------------------------------------------------------------------------

def test_multiplier_thresholds():
    bounds = multiplier_thresholds(floor_px=1.0, good_mult=2.0, warning_mult=5.0)
    assert bounds["good_below_px"] == 2.0
    assert bounds["warning_below_px"] == 5.0


def test_classify_good_warning_bad():
    floor_px = 1.0
    assert classify(1.5, floor_px, good_mult=2.0, warning_mult=5.0) == "GOOD"
    assert classify(3.0, floor_px, good_mult=2.0, warning_mult=5.0) == "WARNING"
    assert classify(6.0, floor_px, good_mult=2.0, warning_mult=5.0) == "BAD"


def test_classify_boundary_is_exclusive_good():
    floor_px = 1.0
    # exactly at 2x floor -> not GOOD (boundary goes to WARNING)
    assert classify(2.0, floor_px, good_mult=2.0, warning_mult=5.0) == "WARNING"
    # exactly at 5x floor -> not WARNING (boundary goes to BAD)
    assert classify(5.0, floor_px, good_mult=2.0, warning_mult=5.0) == "BAD"


# ---------------------------------------------------------------------------
# Manual runner (in case pytest isn't available in the environment)
# ---------------------------------------------------------------------------

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
