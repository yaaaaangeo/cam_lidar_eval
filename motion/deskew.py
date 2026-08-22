"""
motion/deskew.py

STEP 5 -- LiDAR Motion Deskew (see evaluation_metric_spec.md's STEP 5).

Problem: a single LiDAR "frame" isn't captured at one instant -- a
spinning LiDAR sweeps its full 360-degree scan over the whole frame
period (commonly ~0.1s at 10Hz), so each point is actually measured at a
slightly different time, in a slightly different sensor pose, if the
vehicle/robot is moving. The camera, by contrast, captures its frame at
one instant. Projecting an un-corrected ("skewed") LiDAR frame under a
single T_CL evaluated against one camera instant means most of the
frame's points are being projected as if they'd been measured at a time
they weren't -- for a moving platform, this shows up as smeared/doubled
edges that look like a calibration problem but are actually a motion
artifact.

This module does two things:
  1. estimate_point_time_offsets: recover each point's time offset WITHIN
     the frame. If the point cloud already carries explicit per-point
     timestamps (common for real sensor drivers), use those. Otherwise,
     approximate via azimuth angle: a spinning LiDAR sweeps azimuth
     roughly linearly over the scan period, so time_offset ~= azimuth /
     (2*pi) * scan_period -- the same "derive it from geometry when the
     explicit field isn't available" approach STEP 4's derive_ring_index
     already uses for ring, and a standard technique in real deskewing
     implementations (LOAM-family, Autoware) when a raw per-point time
     field isn't present.
  2. deskew_points_constant_velocity: given each point's time offset and
     a constant SE(3) velocity (linear + angular) for the platform over
     the scan period, move every point from "as measured at its own
     capture time" into a single reference time's sensor frame -- LERP
     for translation, exact axis-angle (Rodrigues) rotation for
     orientation, which is the standard constant-velocity deskew
     approximation (valid for the short, ~0.1s durations a single scan
     spans).

Necessarily needs an external ego-motion source (IMU, wheel odometry, a
prior scan-matching estimate, ...) -- this tool has no such input on its
own (see evaluation_metric_spec.md's Input Loader Spec: only camera+LiDAR
+ extrinsic), so the caller supplies the platform's velocity explicitly
(app.cli exposes this via opt-in --deskew-* flags). At zero velocity,
deskewing is exactly a no-op (see DeskewResult.max_correction_m == 0.0
for a stationary platform) -- this is the STEP 5 "정지 상황과 이동
상황에서 deskew 전/후 차이를 비교" comparison in its simplest form: the
same function, called with zero velocity vs nonzero velocity, on the same
points.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from geometry.range_image import compute_azimuth_rad


def estimate_point_time_offsets(
    points: np.ndarray,
    scan_period_s: float,
    point_times_s: Optional[np.ndarray] = None,
    azimuth_at_scan_start_rad: float = 0.0,
    clockwise: bool = False,
) -> np.ndarray:
    """
    Per-point time offset within the frame, relative to the scan's own
    start (i.e. in [0, scan_period_s], NOT relative to any external
    reference time -- deskew_points_constant_velocity's `reference_time_s`
    handles that separately).

    If point_times_s is given (explicit per-point timestamps -- e.g. a
    "time" or "t" field from the sensor driver, already resolved to
    seconds-since-scan-start by the caller), it's returned as-is
    (defensively clipped to [0, scan_period_s], in case of minor sensor
    timing jitter at the very start/end of a scan).

    Otherwise, offsets are approximated from azimuth angle
    (geometry.range_image.compute_azimuth_rad), assuming the scanner
    sweeps azimuth at a constant rate starting at azimuth_at_scan_start_rad
    and covering the full 2*pi over scan_period_s:

        offset = ((azimuth - azimuth_at_scan_start) mod 2*pi)
                 / (2*pi) * scan_period_s

    clockwise: whether the scanner sweeps azimuth in the increasing (CCW,
    default False -> counter-clockwise, matching compute_azimuth_rad's
    atan2 convention) or decreasing (CW, True) direction. Most spinning
    LiDARs are one or the other consistently; get this wrong and the
    deskew correction will be applied roughly BACKWARDS in time across
    the scan, so it matters -- check against a known-moving sequence
    (STEP 5's "이동 상황" comparison) if unsure which way the actual
    sensor spins.
    """
    n = points.shape[0]
    if n == 0:
        return np.zeros(0, dtype=np.float64)

    if point_times_s is not None:
        return np.clip(np.asarray(point_times_s, dtype=np.float64), 0.0, scan_period_s)

    azimuth = compute_azimuth_rad(points)
    delta = np.mod(azimuth - azimuth_at_scan_start_rad, 2.0 * np.pi)
    if clockwise:
        delta = np.mod(2.0 * np.pi - delta, 2.0 * np.pi)
    return delta / (2.0 * np.pi) * scan_period_s


def _rodrigues_batch(axis: np.ndarray, angles: np.ndarray) -> np.ndarray:
    """
    Batch axis-angle -> rotation matrix, for a SINGLE FIXED unit axis and
    an array of N angles (one per point) -- exactly the shape needed here,
    since angular velocity direction is constant across a scan while the
    integrated angle (|omega| * dt) varies per point. Returns (N, 3, 3).

    Standard Rodrigues' rotation formula:
        R = I + sin(theta) K + (1 - cos(theta)) K^2
    where K is the skew-symmetric cross-product matrix of `axis`.
    """
    n = angles.shape[0]
    kx, ky, kz = axis
    K = np.array([
        [0.0, -kz, ky],
        [kz, 0.0, -kx],
        [-ky, kx, 0.0],
    ])
    I = np.eye(3)
    sin_t = np.sin(angles)[:, None, None]
    cos_t = np.cos(angles)[:, None, None]
    K2 = K @ K
    return I[None, :, :] + sin_t * K[None, :, :] + (1.0 - cos_t) * K2[None, :, :]


@dataclass
class DeskewResult:
    points_deskewed: np.ndarray       # (N, 3) or (N, C) -- same shape/extra columns as input, xyz corrected
    point_time_offsets_s: np.ndarray  # (N,) time offset used for each point, relative to scan start
    correction_m: np.ndarray          # (N,) per-point displacement magnitude introduced by deskewing
    mean_correction_m: float
    max_correction_m: float
    scan_period_s: float
    reference_time_s: float


def deskew_points_constant_velocity(
    points: np.ndarray,
    scan_period_s: float,
    linear_velocity_mps: np.ndarray,
    angular_velocity_rps: np.ndarray,
    point_times_s: Optional[np.ndarray] = None,
    reference_time_s: float = 0.0,
    azimuth_at_scan_start_rad: float = 0.0,
    clockwise: bool = False,
) -> DeskewResult:
    """
    Move every point from "as measured, at its own capture time within
    the scan" into a single reference time's sensor frame, assuming the
    platform moved with CONSTANT linear_velocity_mps + angular_velocity_rps
    over the scan.

    reference_time_s: which instant (in [0, scan_period_s], scan-start-
    relative -- same convention as estimate_point_time_offsets) all
    points are corrected TO. Typically the camera's own capture instant
    within the LiDAR scan window, if known; 0.0 (scan start) or
    scan_period_s/2 (scan midpoint) are reasonable defaults when the
    camera's precise position within the scan window isn't known.

    For each point with time offset t (from estimate_point_time_offsets),
    dt = reference_time_s - t is how far forward (positive) or backward
    (negative) in time this point needs to be carried. Under a constant
    BODY-FRAME velocity (v, w), the sensor's world pose integrates as
    pos(t+dt) = pos(t) + R(t) @ (v*dt) and R(t+dt) = R(t) @ Rodrigues(w*dt).
    Re-deriving a fixed world point's local coordinates at time (t+dt)
    from its local coordinates at time t (see motion/deskew.py's test
    suite for a from-scratch numerical simulation confirming this):

        p_ref = Rodrigues(w*dt)^T @ (p_local - v*dt)

    i.e. subtract the body-frame translation FIRST, then apply the
    INVERSE (transpose) of the accumulated rotation -- NOT "rotate then
    translate" as a naive reading of "apply the delta transform" might
    suggest. Getting this backwards produces plausible-LOOKING but
    numerically wrong corrections (same order of magnitude, wrong
    direction) that would be very easy to miss without a ground-truth
    check, which is exactly why this docstring spells out the derivation
    instead of just asserting the result.

    This is the standard constant-velocity ("LERP translation + exact
    axis-angle rotation") deskew approximation -- valid because a single
    scan period is short (~0.1s for a typical 10Hz spinning LiDAR), so
    velocity changing within that window is a second-order effect.

    At zero velocity (both vectors all-zero), every dt maps to the
    identity transform, so points_deskewed is EXACTLY points and
    max_correction_m is EXACTLY 0.0 -- this is STEP 5's "stationary"
    comparison case in its simplest, most direct form.

    points may be (N, 3) or (N, C) with C > 3 (e.g. an intensity column);
    only the first 3 columns are corrected, any extra columns pass
    through unchanged.
    """
    points = np.asarray(points, dtype=np.float64)
    n = points.shape[0]
    linear_velocity_mps = np.asarray(linear_velocity_mps, dtype=np.float64).reshape(3)
    angular_velocity_rps = np.asarray(angular_velocity_rps, dtype=np.float64).reshape(3)

    if n == 0:
        return DeskewResult(
            points_deskewed=points.copy(),
            point_time_offsets_s=np.zeros(0), correction_m=np.zeros(0),
            mean_correction_m=0.0, max_correction_m=0.0,
            scan_period_s=scan_period_s, reference_time_s=reference_time_s,
        )

    xyz = points[:, :3]
    point_times = estimate_point_time_offsets(
        xyz, scan_period_s, point_times_s=point_times_s,
        azimuth_at_scan_start_rad=azimuth_at_scan_start_rad, clockwise=clockwise,
    )
    dt = reference_time_s - point_times

    omega_norm = float(np.linalg.norm(angular_velocity_rps))
    if omega_norm < 1e-12:
        # No rotation at all -- avoid the axis/0 division below and just
        # apply pure translation (R_delta == identity for every point).
        # Also the exact path taken for the "stationary" case.
        xyz_deskewed = xyz - dt[:, None] * linear_velocity_mps[None, :]
    else:
        axis = angular_velocity_rps / omega_norm
        angles = omega_norm * dt
        R_pos = _rodrigues_batch(axis, angles)          # Rodrigues(+w*dt), (N, 3, 3)
        R_delta = np.transpose(R_pos, (0, 2, 1))         # ^T == Rodrigues(-w*dt) -- see docstring derivation
        translated = xyz - dt[:, None] * linear_velocity_mps[None, :]
        xyz_deskewed = np.einsum("nij,nj->ni", R_delta, translated)

    points_deskewed = points.copy()
    points_deskewed[:, :3] = xyz_deskewed

    correction = np.linalg.norm(xyz_deskewed - xyz, axis=1)

    return DeskewResult(
        points_deskewed=points_deskewed,
        point_time_offsets_s=point_times,
        correction_m=correction,
        mean_correction_m=float(np.mean(correction)),
        max_correction_m=float(np.max(correction)),
        scan_period_s=scan_period_s,
        reference_time_s=reference_time_s,
    )


def compare_before_after(
    points: np.ndarray,
    result: DeskewResult,
) -> dict:
    """
    Small summary dict for the "정지 상황과 이동 상황에서 deskew 전/후
    차이를 비교" comparison STEP 5 asks for -- meant to be printed/logged
    or dropped straight into a report section (see report/builder.py's
    deskew_summary if wired in). Two calls to
    deskew_points_constant_velocity -- one with zero velocity (stationary
    baseline), one with the platform's actual estimated/measured velocity
    (moving case) -- produce two of these, and the DIFFERENCE between
    their correction magnitudes is exactly the "how much did motion
    matter here" answer.
    """
    return {
        "num_points": int(points.shape[0]),
        "scan_period_s": result.scan_period_s,
        "reference_time_s": result.reference_time_s,
        "mean_correction_m": result.mean_correction_m,
        "max_correction_m": result.max_correction_m,
        "p95_correction_m": float(np.percentile(result.correction_m, 95)) if result.correction_m.size else 0.0,
    }
