import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from evaluation.plane_consistency import fit_plane_ransac, evaluate_plane_consistency
from tests.test_holdout_consistency import _make_camera, _make_lidar_spec


def _flat_plane_points(n=2000, seed=0, z0=5.0, noise=0.005):
    """Points on a plane z = z0 (perpendicular to the optical axis)."""
    rng = np.random.RandomState(seed)
    x = rng.uniform(-2, 2, n)
    y = rng.uniform(-1.5, 1.5, n)
    z = np.full(n, z0) + rng.normal(0, noise, n)
    return np.stack([x, y, z], axis=1)


# ---------------------------------------------------------------------------
# fit_plane_ransac
# ---------------------------------------------------------------------------

def test_fit_plane_ransac_finds_flat_plane():
    points = _flat_plane_points()
    plane = fit_plane_ransac(points, distance_threshold_m=0.05, iterations=200)
    assert plane is not None
    assert plane.inlier_ratio > 0.9
    # normal should be roughly aligned with Z axis (since plane is z=const)
    assert abs(abs(plane.normal[2]) - 1.0) < 0.05


def test_fit_plane_ransac_handles_noisy_outliers():
    plane_points = _flat_plane_points(n=1800)
    rng = np.random.RandomState(1)
    outliers = rng.uniform(-5, 5, size=(200, 3))
    points = np.vstack([plane_points, outliers])
    plane = fit_plane_ransac(points, distance_threshold_m=0.05, iterations=300)
    assert plane is not None
    assert plane.inlier_ratio > 0.8  # majority should still be identified as the plane


def test_fit_plane_ransac_too_few_points():
    assert fit_plane_ransac(np.zeros((2, 3))) is None


def test_fit_plane_ransac_empty_points():
    assert fit_plane_ransac(np.zeros((0, 3))) is None


def test_fit_plane_ransac_no_dominant_plane_in_random_cloud():
    rng = np.random.RandomState(2)
    points = rng.uniform(-5, 5, size=(500, 3))
    plane = fit_plane_ransac(points, distance_threshold_m=0.01, iterations=200)
    # a fully random 3D cloud shouldn't have a plane covering a large fraction
    assert plane is None or plane.inlier_ratio < 0.3


# ---------------------------------------------------------------------------
# evaluate_plane_consistency (uses the M2 synthetic depth-step scene, but
# restricted to the near-depth plane so the "plane" IS the whole near surface)
# ---------------------------------------------------------------------------

def _make_two_depth_scene():
    import cv2
    width, height = 640, 480
    fx = fy = 500.0
    cx, cy = 320.0, 240.0
    z_near, z_far = 5.0, 10.0

    image = np.zeros((height, width), dtype=np.uint8)
    image[:, int(cx):] = 255
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    u_vals = np.linspace(0, width - 1, 220)
    v_vals = np.linspace(0, height - 1, 140)
    uu, vv = np.meshgrid(u_vals, v_vals)
    uu, vv = uu.ravel(), vv.ravel()
    zz = np.where(uu < cx, z_near, z_far)
    xx = (uu - cx) * zz / fx
    yy = (vv - cy) * zz / fy
    points = np.stack([xx, yy, zz], axis=1)
    return image, points


def test_evaluate_plane_consistency_finds_dominant_plane_and_scores_it():
    camera = _make_camera()
    image, points = _make_two_depth_scene()
    result = evaluate_plane_consistency(
        image=image, points_lidar=points, T_CL=np.eye(4), camera=camera,
        lidar_spec=_make_lidar_spec(), plane_distance_threshold_m=0.05,
    )
    assert result.plane_found
    assert result.inlier_ratio > 0.3  # one of the two depth planes should dominate
    assert result.classification != "FAIL", result.warnings
    assert result.num_boundary_points > 0


def test_evaluate_plane_consistency_fails_with_no_flat_surface():
    camera = _make_camera()
    image, _ = _make_two_depth_scene()
    rng = np.random.RandomState(3)
    scattered = rng.uniform(-3, 3, size=(500, 3))
    scattered[:, 2] = np.abs(scattered[:, 2]) + 1.0  # keep in front of camera
    result = evaluate_plane_consistency(
        image=image, points_lidar=scattered, T_CL=np.eye(4), camera=camera,
        lidar_spec=_make_lidar_spec(), plane_distance_threshold_m=0.01,
        min_inlier_ratio=0.5,
    )
    assert not result.plane_found
    assert result.classification == "FAIL"


def test_evaluate_plane_consistency_mean_px_and_floor_px_when_valid():
    camera = _make_camera()
    image, points = _make_two_depth_scene()
    result = evaluate_plane_consistency(
        image=image, points_lidar=points, T_CL=np.eye(4), camera=camera,
        lidar_spec=_make_lidar_spec(),
    )
    assert result.plane_found
    assert result.classification != "FAIL"
    assert result.mean_px >= 0
    assert result.floor_px > 0


def _make_plane_patch_scene():
    """
    A scene built specifically so the WHOLE convex-hull boundary of the
    fitted plane corresponds to a real drawn edge: a white rectangle on a
    black background, and a flat LiDAR plane patch positioned to project
    exactly onto that rectangle under the identity T_CL. Unlike
    _make_two_depth_scene (where the plane's hull includes FOV-clipping
    edges that have no corresponding real image edge, dominating the error
    with irrelevant sides), this isolates the actual boundary-alignment
    signal on all four sides.
    """
    import cv2
    width, height = 640, 480
    fx = fy = 500.0
    cx, cy = 320.0, 240.0
    z_plane = 5.0
    u0, u1, v0, v1 = 200, 440, 150, 330  # rectangle in pixel space

    image = np.zeros((height, width), dtype=np.uint8)
    image[v0:v1, u0:u1] = 255
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    u_vals = np.linspace(u0, u1 - 1, 60)
    v_vals = np.linspace(v0, v1 - 1, 45)
    uu, vv = np.meshgrid(u_vals, v_vals)
    uu, vv = uu.ravel(), vv.ravel()
    xx = (uu - cx) * z_plane / fx
    yy = (vv - cy) * z_plane / fy
    zz = np.full(uu.shape, z_plane)
    points = np.stack([xx, yy, zz], axis=1)
    return image, points


def test_evaluate_plane_consistency_higher_error_with_perturbed_extrinsic():
    camera = _make_camera()
    image, points = _make_plane_patch_scene()
    T_perturbed = np.eye(4)
    T_perturbed[0, 3] = 0.15

    result_correct = evaluate_plane_consistency(
        image=image, points_lidar=points, T_CL=np.eye(4), camera=camera, lidar_spec=_make_lidar_spec(),
    )
    result_perturbed = evaluate_plane_consistency(
        image=image, points_lidar=points, T_CL=T_perturbed, camera=camera, lidar_spec=_make_lidar_spec(),
    )
    assert result_correct.classification != "FAIL", result_correct.warnings
    assert result_perturbed.classification != "FAIL", result_perturbed.warnings
    assert result_correct.mean_px < 3.0, f"mean_px={result_correct.mean_px}"
    assert result_perturbed.mean_px > result_correct.mean_px


def test_evaluate_plane_consistency_too_few_boundary_points_fails():
    camera = _make_camera()
    image, points = _make_two_depth_scene()
    # only the near half-plane, but require an unreasonably high boundary count
    result = evaluate_plane_consistency(
        image=image, points_lidar=points, T_CL=np.eye(4), camera=camera,
        lidar_spec=_make_lidar_spec(), min_boundary_points=1_000_000,
    )
    assert result.plane_found
    assert result.classification == "FAIL"
    assert any("boundary points" in w for w in result.warnings)


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
