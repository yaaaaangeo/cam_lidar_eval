import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from motion.deskew import (
    estimate_point_time_offsets,
    deskew_points_constant_velocity,
    compare_before_after,
    DeskewResult,
)


# ---------------------------------------------------------------------------
# estimate_point_time_offsets
# ---------------------------------------------------------------------------

def test_estimate_point_time_offsets_explicit_times_passthrough():
    points = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    explicit = np.array([0.02, 0.07])
    offsets = estimate_point_time_offsets(points, scan_period_s=0.1, point_times_s=explicit)
    assert np.allclose(offsets, explicit)


def test_estimate_point_time_offsets_explicit_times_clipped_to_scan_period():
    points = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    explicit = np.array([-0.01, 0.15])  # slightly out of [0, 0.1] range
    offsets = estimate_point_time_offsets(points, scan_period_s=0.1, point_times_s=explicit)
    assert np.allclose(offsets, [0.0, 0.1])


def test_estimate_point_time_offsets_azimuth_based_start_and_quarter_and_half():
    # az=0 -> t=0; az=pi/2 -> t=period/4; az=pi -> t=period/2
    points = np.array([
        [1.0, 0.0, 0.0],   # az = 0
        [0.0, 1.0, 0.0],   # az = pi/2
        [-1.0, 0.0, 0.0],  # az = pi
    ])
    offsets = estimate_point_time_offsets(points, scan_period_s=0.1)
    assert np.isclose(offsets[0], 0.0)
    assert np.isclose(offsets[1], 0.025)
    assert np.isclose(offsets[2], 0.05)


def test_estimate_point_time_offsets_respects_scan_start_azimuth():
    points = np.array([[0.0, 1.0, 0.0]])  # az = pi/2
    offsets = estimate_point_time_offsets(points, scan_period_s=0.1, azimuth_at_scan_start_rad=np.pi / 2)
    assert np.isclose(offsets[0], 0.0)  # this point IS the scan start now


def test_estimate_point_time_offsets_clockwise_flag_reverses_direction():
    points = np.array([[0.0, 1.0, 0.0]])  # az = pi/2 -> 1/4 of the way around CCW
    ccw = estimate_point_time_offsets(points, scan_period_s=0.1, clockwise=False)
    cw = estimate_point_time_offsets(points, scan_period_s=0.1, clockwise=True)
    assert np.isclose(ccw[0], 0.025)
    assert np.isclose(cw[0], 0.075)  # same point is 3/4 of the way around going CW


def test_estimate_point_time_offsets_empty_input():
    offsets = estimate_point_time_offsets(np.zeros((0, 3)), scan_period_s=0.1)
    assert offsets.shape == (0,)


# ---------------------------------------------------------------------------
# deskew_points_constant_velocity -- stationary (no-op) cases
# ---------------------------------------------------------------------------

def test_deskew_zero_velocity_is_exact_noop():
    points = np.random.default_rng(0).uniform(-10, 10, size=(50, 3))
    result = deskew_points_constant_velocity(
        points, scan_period_s=0.1,
        linear_velocity_mps=np.zeros(3), angular_velocity_rps=np.zeros(3),
        reference_time_s=0.03,
    )
    assert np.array_equal(result.points_deskewed, points)
    assert result.max_correction_m == 0.0
    assert result.mean_correction_m == 0.0


def test_deskew_zero_velocity_with_intensity_column_preserved():
    points = np.array([[1.0, 2.0, 3.0, 100.0], [4.0, 5.0, 6.0, 50.0]])  # xyz + intensity
    result = deskew_points_constant_velocity(
        points, scan_period_s=0.1, linear_velocity_mps=np.zeros(3), angular_velocity_rps=np.zeros(3),
    )
    assert np.array_equal(result.points_deskewed, points)
    assert np.array_equal(result.points_deskewed[:, 3], [100.0, 50.0])  # intensity untouched


def test_deskew_point_exactly_at_reference_time_is_unmoved_even_with_motion():
    # a point whose OWN time offset equals reference_time_s has dt=0,
    # regardless of nonzero velocity -- no correction needed for it.
    points = np.array([[5.0, 0.0, 0.0]])  # az = 0 -> t = 0
    result = deskew_points_constant_velocity(
        points, scan_period_s=0.1,
        linear_velocity_mps=np.array([3.0, 0.0, 0.0]), angular_velocity_rps=np.array([0, 0, 1.0]),
        reference_time_s=0.0,
    )
    assert np.allclose(result.points_deskewed, points, atol=1e-9)
    assert np.isclose(result.max_correction_m, 0.0, atol=1e-9)


# ---------------------------------------------------------------------------
# deskew_points_constant_velocity -- ground-truth numerical simulation
# ---------------------------------------------------------------------------

def _rodrigues(axis, theta):
    axis = axis / np.linalg.norm(axis)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)


def _simulate_ground_truth_local_at_reference(p_local_source, t_source, t_ref, v, w, dt_step=1e-5):
    """Independent, from-scratch numerical integration of a constant
    body-frame-velocity rigid motion, used ONLY in this test file to
    check deskew_points_constant_velocity's closed-form result against
    ground truth -- deliberately not sharing any code with
    motion/deskew.py's implementation, so a bug there wouldn't be
    self-confirming."""
    pos = np.zeros(3)
    R = np.eye(3)
    t = 0.0
    t_max = max(t_source, t_ref)
    steps = int(round(t_max / dt_step))
    snapshot = {}
    if 0.0 in (t_source, t_ref):
        snapshot[0.0] = (R.copy(), pos.copy())
    for _ in range(steps):
        pos = pos + R @ (v * dt_step)
        if np.linalg.norm(w) > 0:
            R = R @ _rodrigues(w, np.linalg.norm(w) * dt_step)
        t += dt_step
        if abs(t - t_source) < dt_step / 2:
            snapshot["source"] = (R.copy(), pos.copy())
        if abs(t - t_ref) < dt_step / 2:
            snapshot["ref"] = (R.copy(), pos.copy())

    R_s, pos_s = snapshot.get("source", snapshot.get(0.0))
    R_r, pos_r = snapshot.get("ref", snapshot.get(0.0))
    P_world = R_s @ p_local_source + pos_s
    return R_r.T @ (P_world - pos_r)


def test_deskew_matches_ground_truth_simulation_translation_and_rotation():
    v = np.array([2.0, 0.5, 0.0])
    w = np.array([0.0, 0.0, 1.0])
    t_source, t_ref = 0.03, 0.08
    p_local_source = np.array([3.0, -1.0, 0.5])

    expected = _simulate_ground_truth_local_at_reference(p_local_source, t_source, t_ref, v, w)

    result = deskew_points_constant_velocity(
        p_local_source.reshape(1, 3), scan_period_s=0.1,
        linear_velocity_mps=v, angular_velocity_rps=w,
        point_times_s=np.array([t_source]), reference_time_s=t_ref,
    )
    # atol here accounts for the reference simulator's own first-order
    # Euler integration error (dt_step=1e-5 over ~8000 steps for the
    # coupled rotation+translation case) -- NOT slack in deskew's own
    # math, which is an exact closed-form Rodrigues formula. For scale: a
    # genuine sign/direction bug in the deskew formula (caught during
    # development) produced an error around 0.46 on this same case --
    # ~180x larger than the numerical noise floor this atol allows.
    assert np.allclose(result.points_deskewed[0], expected, atol=5e-3)


def test_deskew_matches_ground_truth_simulation_pure_translation():
    v = np.array([1.5, -0.7, 0.2])
    w = np.zeros(3)
    t_source, t_ref = 0.01, 0.09
    p_local_source = np.array([10.0, 2.0, -1.0])

    expected = _simulate_ground_truth_local_at_reference(p_local_source, t_source, t_ref, v, w)
    result = deskew_points_constant_velocity(
        p_local_source.reshape(1, 3), scan_period_s=0.1,
        linear_velocity_mps=v, angular_velocity_rps=w,
        point_times_s=np.array([t_source]), reference_time_s=t_ref,
    )
    assert np.allclose(result.points_deskewed[0], expected, atol=1e-6)
    # pure translation, no rotation: correction should equal exactly -v*dt
    dt = t_ref - t_source
    assert np.allclose(result.points_deskewed[0], p_local_source - v * dt, atol=1e-6)


def test_deskew_matches_ground_truth_simulation_pure_rotation():
    v = np.zeros(3)
    w = np.array([0.3, -0.2, 0.9])
    t_source, t_ref = 0.02, 0.07
    p_local_source = np.array([4.0, 0.0, 1.0])

    expected = _simulate_ground_truth_local_at_reference(p_local_source, t_source, t_ref, v, w)
    result = deskew_points_constant_velocity(
        p_local_source.reshape(1, 3), scan_period_s=0.1,
        linear_velocity_mps=v, angular_velocity_rps=w,
        point_times_s=np.array([t_source]), reference_time_s=t_ref,
    )
    assert np.allclose(result.points_deskewed[0], expected, atol=1e-3)
    # pure rotation preserves distance from origin
    assert np.isclose(np.linalg.norm(result.points_deskewed[0]), np.linalg.norm(p_local_source), atol=1e-6)


def test_deskew_correction_grows_with_distance_from_reference_time():
    """Points farther (in time) from reference_time_s should generally
    accumulate larger corrections under constant nonzero velocity -- a
    basic sanity trend check independent of the exact ground-truth math
    above."""
    scan_period = 0.1
    times = np.linspace(0.0, scan_period, 20)
    # place points along a circle so distance from origin doesn't
    # dominate the correction magnitude comparison
    points = np.stack([np.full(20, 5.0), np.zeros(20), np.zeros(20)], axis=1)
    result = deskew_points_constant_velocity(
        points, scan_period_s=scan_period,
        linear_velocity_mps=np.array([2.0, 0.0, 0.0]), angular_velocity_rps=np.zeros(3),
        point_times_s=times, reference_time_s=0.0,
    )
    # correction should be monotonically non-decreasing as |t - reference| grows
    assert np.all(np.diff(result.correction_m) >= -1e-9)


# ---------------------------------------------------------------------------
# DeskewResult / compare_before_after
# ---------------------------------------------------------------------------

def test_deskew_result_empty_points():
    result = deskew_points_constant_velocity(
        np.zeros((0, 3)), scan_period_s=0.1,
        linear_velocity_mps=np.array([1.0, 0, 0]), angular_velocity_rps=np.zeros(3),
    )
    assert result.points_deskewed.shape == (0, 3)
    assert result.mean_correction_m == 0.0
    assert result.max_correction_m == 0.0


def test_compare_before_after_summary_shape():
    points = np.random.default_rng(1).uniform(-5, 5, size=(30, 3))
    result = deskew_points_constant_velocity(
        points, scan_period_s=0.1,
        linear_velocity_mps=np.array([3.0, 0, 0]), angular_velocity_rps=np.array([0, 0, 0.5]),
        reference_time_s=0.05,
    )
    summary = compare_before_after(points, result)
    for key in ("num_points", "scan_period_s", "reference_time_s",
                "mean_correction_m", "max_correction_m", "p95_correction_m"):
        assert key in summary
    assert summary["num_points"] == 30
    assert summary["max_correction_m"] >= summary["mean_correction_m"] >= 0.0


def test_compare_before_after_stationary_vs_moving():
    """The STEP 5 completion criterion in its most literal form: compare
    the SAME points deskewed with zero velocity (stationary) vs nonzero
    velocity (moving)."""
    points = np.random.default_rng(2).uniform(-5, 5, size=(40, 3))
    stationary = deskew_points_constant_velocity(
        points, scan_period_s=0.1, linear_velocity_mps=np.zeros(3), angular_velocity_rps=np.zeros(3),
    )
    moving = deskew_points_constant_velocity(
        points, scan_period_s=0.1,
        linear_velocity_mps=np.array([5.0, 0.0, 0.0]), angular_velocity_rps=np.zeros(3),
    )
    stationary_summary = compare_before_after(points, stationary)
    moving_summary = compare_before_after(points, moving)
    assert stationary_summary["max_correction_m"] == 0.0
    assert moving_summary["max_correction_m"] > 0.0


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
