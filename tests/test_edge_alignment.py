import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import cv2

from evaluation.edge_alignment import (
    extract_lidar_edge_points,
    extract_image_edges,
    compute_distance_transform,
    sample_bilinear,
    evaluate_edge_alignment,
)
from input.camera import CameraModel, CameraIntrinsics, CameraDistortion, CameraSource
from input.lidar import LidarSensorSpec


# ---------------------------------------------------------------------------
# Unit tests for individual pipeline stages
# ---------------------------------------------------------------------------

def test_extract_lidar_edge_points_finds_depth_step():
    # Two clusters of points at the same pixel neighborhood but very
    # different depths -> should be flagged as a discontinuity, with only
    # the near-side (smaller depth) points kept.
    near_pixels = np.array([[100.0, 100.0], [101.0, 100.0], [100.0, 101.0], [101.0, 101.0]])
    far_pixels = np.array([[100.5, 100.5], [101.5, 100.5], [100.5, 101.5], [101.5, 101.5]])
    pixels = np.vstack([near_pixels, far_pixels])
    depths = np.array([5.0, 5.0, 5.0, 5.0, 15.0, 15.0, 15.0, 15.0])

    mask = extract_lidar_edge_points(pixels, depths, radius_px=3.0, depth_jump_threshold_m=0.3, min_neighbors=3)
    assert mask[:4].all()   # near-side points flagged
    assert not mask[4:].any()  # far-side points not flagged


def test_extract_lidar_edge_points_no_discontinuity():
    # All points at the same depth, close together -> no edges
    pixels = np.random.RandomState(0).uniform(0, 50, size=(30, 2))
    depths = np.full(30, 5.0)
    mask = extract_lidar_edge_points(pixels, depths, radius_px=5.0, depth_jump_threshold_m=0.3)
    assert not mask.any()


def test_extract_lidar_edge_points_empty_input():
    mask = extract_lidar_edge_points(np.zeros((0, 2)), np.zeros(0))
    assert mask.shape == (0,)


def test_extract_lidar_edge_points_respects_min_neighbors():
    # isolated point far from everything else -> insufficient neighbors, not flagged
    pixels = np.array([[0.0, 0.0], [500.0, 500.0]])
    depths = np.array([5.0, 50.0])
    mask = extract_lidar_edge_points(pixels, depths, radius_px=3.0, min_neighbors=3)
    assert not mask.any()


def test_extract_image_edges_detects_step_edge():
    img = np.zeros((100, 100), dtype=np.uint8)
    img[:, 50:] = 255  # vertical step at x=50
    edges = extract_image_edges(img)
    edge_cols = np.nonzero(edges.sum(axis=0))[0]
    assert len(edge_cols) > 0
    assert abs(int(np.mean(edge_cols)) - 50) <= 2


def test_extract_image_edges_blank_image_has_no_edges():
    img = np.full((100, 100), 128, dtype=np.uint8)
    edges = extract_image_edges(img)
    assert edges.sum() == 0


def test_compute_distance_transform_zero_at_edge():
    edge_map = np.zeros((50, 50), dtype=np.uint8)
    edge_map[25, 25] = 255  # single edge pixel
    dt = compute_distance_transform(edge_map)
    assert dt[25, 25] < 1e-6
    assert dt[25, 30] > 4.0  # 5 px away, should be roughly 5


def test_sample_bilinear_exact_pixel_centers():
    dt = np.arange(25, dtype=np.float64).reshape(5, 5)
    pixels = np.array([[2.0, 3.0]])  # (u=2, v=3) -> dt[3,2]
    val = sample_bilinear(dt, pixels)
    assert np.isclose(val[0], dt[3, 2])


def test_sample_bilinear_interpolates_midpoint():
    dt = np.zeros((5, 5))
    dt[2, 2] = 10.0
    dt[2, 3] = 20.0
    pixels = np.array([[2.5, 2.0]])  # halfway between column 2 and 3 at row 2
    val = sample_bilinear(dt, pixels)
    assert np.isclose(val[0], 15.0)


def test_sample_bilinear_clamps_out_of_bounds():
    dt = np.ones((5, 5)) * 7.0
    pixels = np.array([[-10.0, -10.0], [100.0, 100.0]])
    vals = sample_bilinear(dt, pixels)
    assert np.allclose(vals, [7.0, 7.0])


# ---------------------------------------------------------------------------
# End-to-end synthetic scenario:
#   A depth "step" in 3D that, when correctly projected via T_CL, lines up
#   exactly with a drawn vertical edge in the image. This validates the
#   FULL pipeline (projection -> edge extraction -> DT sampling) together,
#   and confirms error increases when T_CL is perturbed away from truth.
# ---------------------------------------------------------------------------

def _make_synthetic_scene():
    width, height = 640, 480
    fx = fy = 500.0
    cx, cy = 320.0, 240.0

    camera = CameraModel(
        width=width, height=height, model="pinhole",
        intrinsics=CameraIntrinsics(fx, fy, cx, cy),
        distortion=CameraDistortion(model="none"),
        source=CameraSource(kind="image_dir", path="."),
    )

    # image: vertical step at the image center
    image = np.zeros((height, width), dtype=np.uint8)
    image[:, int(cx):] = 255  # right half white -> vertical edge at u=cx
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    # Build LiDAR points by BACK-PROJECTING from a dense pixel grid, so the
    # resulting depth step lands exactly on u=cx for every row (avoiding
    # perspective row-shift that a naive world-space grid would introduce:
    # two flat world planes at different depths do NOT project to aligned
    # rows for y != 0, since v = fy*y/z + cy depends on z).
    u_vals = np.linspace(0, width - 1, 220)
    v_vals = np.linspace(0, height - 1, 140)
    uu, vv = np.meshgrid(u_vals, v_vals)
    uu = uu.ravel()
    vv = vv.ravel()

    z_near, z_far = 5.0, 10.0
    zz = np.where(uu < cx, z_near, z_far)
    xx = (uu - cx) * zz / fx
    yy = (vv - cy) * zz / fy
    points_lidar = np.stack([xx, yy, zz], axis=1)

    lidar_spec = LidarSensorSpec(horizontal_resolution_deg=0.05, vertical_resolution_deg=0.05, range_accuracy_m=0.02)

    return camera, image, points_lidar, lidar_spec


def test_edge_alignment_low_error_with_correct_extrinsic():
    camera, image, points_lidar, lidar_spec = _make_synthetic_scene()
    T_CL = np.eye(4)  # identity: lidar frame == camera frame axes, matches how points were built

    result = evaluate_edge_alignment(
        image=image, points_lidar=points_lidar, T_CL=T_CL,
        camera=camera, lidar_spec=lidar_spec,
        depth_jump_threshold_m=1.0,
    )
    assert result.classification != "FAIL", result.warnings
    assert result.num_edge_points > 0
    # correct calibration -> discontinuity should land right on the drawn edge
    assert result.mean_px < 3.0, f"mean_px={result.mean_px}"


def test_edge_alignment_higher_error_with_perturbed_extrinsic():
    camera, image, points_lidar, lidar_spec = _make_synthetic_scene()
    T_correct = np.eye(4)
    T_perturbed = np.eye(4)
    T_perturbed[0, 3] = 1.0  # shift 1m in x -> discontinuity moves ~fx*1/depth px in image

    result_correct = evaluate_edge_alignment(
        image=image, points_lidar=points_lidar, T_CL=T_correct,
        camera=camera, lidar_spec=lidar_spec, depth_jump_threshold_m=1.0,
    )
    result_perturbed = evaluate_edge_alignment(
        image=image, points_lidar=points_lidar, T_CL=T_perturbed,
        camera=camera, lidar_spec=lidar_spec, depth_jump_threshold_m=1.0,
    )
    assert result_correct.classification != "FAIL"
    assert result_perturbed.classification != "FAIL"
    assert result_perturbed.mean_px > result_correct.mean_px, (
        f"correct={result_correct.mean_px} perturbed={result_perturbed.mean_px}"
    )


def test_edge_alignment_fails_gracefully_on_empty_pointcloud():
    camera, image, _, lidar_spec = _make_synthetic_scene()
    result = evaluate_edge_alignment(
        image=image, points_lidar=np.zeros((0, 3)), T_CL=np.eye(4),
        camera=camera, lidar_spec=lidar_spec,
    )
    assert result.classification == "FAIL"
    assert np.isnan(result.mean_px)


def test_edge_alignment_fails_gracefully_on_blank_image():
    camera, _, points_lidar, lidar_spec = _make_synthetic_scene()
    blank_image = np.zeros((camera.height, camera.width, 3), dtype=np.uint8)
    result = evaluate_edge_alignment(
        image=blank_image, points_lidar=points_lidar, T_CL=np.eye(4),
        camera=camera, lidar_spec=lidar_spec, depth_jump_threshold_m=1.0,
    )
    assert result.classification == "FAIL"
    assert "no edges" in result.warnings[0].lower() or "No edges" in result.warnings[0]


def test_edge_alignment_result_carries_floor_and_classification_consistently():
    camera, image, points_lidar, lidar_spec = _make_synthetic_scene()
    result = evaluate_edge_alignment(
        image=image, points_lidar=points_lidar, T_CL=np.eye(4),
        camera=camera, lidar_spec=lidar_spec, depth_jump_threshold_m=1.0,
    )
    assert result.floor_px > 0
    assert result.classification in ("GOOD", "WARNING", "BAD")


# ---------------------------------------------------------------------------
# STEP7 -- Noise/Uncertainty Model: per-point expected noise + normalized error
# ---------------------------------------------------------------------------

def test_edge_alignment_carries_per_point_uncertainty_fields():
    camera, image, points_lidar, lidar_spec = _make_synthetic_scene()
    result = evaluate_edge_alignment(
        image=image, points_lidar=points_lidar, T_CL=np.eye(4),
        camera=camera, lidar_spec=lidar_spec, depth_jump_threshold_m=1.0,
    )
    assert result.edge_point_floor_px is not None
    assert result.edge_point_normalized_errors is not None
    assert result.edge_point_depths_m is not None
    assert result.edge_point_matched is not None
    assert result.edge_point_matched.shape == (result.num_edge_points,)
    assert result.edge_point_matched.sum() == result.num_matched
    assert result.edge_point_floor_px.shape == (result.num_edge_points,)
    assert result.edge_point_normalized_errors.shape == (result.num_edge_points,)
    assert result.edge_point_depths_m.shape == (result.num_edge_points,)
    assert np.all(result.edge_point_floor_px > 0)
    assert np.all(result.edge_point_depths_m > 0)
    assert result.mean_normalized_error is not None
    assert result.median_normalized_error is not None
    assert result.p95_normalized_error is not None
    assert result.mean_normalized_error >= 0


def test_edge_alignment_normalized_error_matches_manual_division():
    camera, image, points_lidar, lidar_spec = _make_synthetic_scene()
    result = evaluate_edge_alignment(
        image=image, points_lidar=points_lidar, T_CL=np.eye(4),
        camera=camera, lidar_spec=lidar_spec, depth_jump_threshold_m=1.0,
    )
    manual = result.edge_point_errors_px / result.edge_point_floor_px
    assert np.allclose(result.edge_point_normalized_errors, manual)


def test_edge_alignment_per_point_floor_matches_direct_computation_at_shared_depth():
    """In this scene, M2's edge extraction keeps only the NEAR-SIDE of the
    depth discontinuity (by design -- see extract_lidar_edge_points'
    docstring), so every edge point shares the same depth (z_near=5.0m).
    edge_point_floor_px should then be UNIFORM across all points and
    match quality.noise_floor.compute_floor called directly at that same
    depth -- a cross-check against the STEP7-specific unit tests in
    test_noise_floor.py (which use varying synthetic depths directly,
    where this scene's fixed geometry can't)."""
    from quality.noise_floor import resolve_floor_inputs, compute_floor
    camera, image, points_lidar, lidar_spec = _make_synthetic_scene()
    result = evaluate_edge_alignment(
        image=image, points_lidar=points_lidar, T_CL=np.eye(4),
        camera=camera, lidar_spec=lidar_spec, depth_jump_threshold_m=1.0,
    )
    assert np.allclose(result.edge_point_depths_m, 5.0)
    assert np.allclose(result.edge_point_floor_px, result.edge_point_floor_px[0])

    floor_inputs = resolve_floor_inputs(fx_px=camera.intrinsics.fx, T_CL=np.eye(4), lidar_spec=lidar_spec,
                                         edge_localization_floor_px=camera.edge_localization_floor_px)
    expected_floor = compute_floor(floor_inputs, 5.0)  # z_near from _make_synthetic_scene
    assert np.isclose(result.edge_point_floor_px[0], expected_floor, rtol=1e-6)


def test_edge_alignment_uncertainty_fields_none_on_fail():
    camera, image, _, lidar_spec = _make_synthetic_scene()
    result = evaluate_edge_alignment(
        image=image, points_lidar=np.zeros((0, 3)), T_CL=np.eye(4),
        camera=camera, lidar_spec=lidar_spec,
    )
    assert result.classification == "FAIL"
    assert result.edge_point_floor_px is None
    assert result.edge_point_normalized_errors is None
    assert result.edge_point_depths_m is None
    assert result.mean_normalized_error is None


# ---------------------------------------------------------------------------
# STEP8 -- Dynamic Object Filtering: dynamic_mask parameter
# ---------------------------------------------------------------------------

def test_edge_alignment_dynamic_mask_excludes_flagged_points():
    camera, image, points_lidar, lidar_spec = _make_synthetic_scene()
    # flag every point in the "near" (left) half as dynamic
    dynamic_mask = points_lidar[:, 2] < 7.0  # z_near=5.0 points

    baseline = evaluate_edge_alignment(
        image=image, points_lidar=points_lidar, T_CL=np.eye(4),
        camera=camera, lidar_spec=lidar_spec, depth_jump_threshold_m=1.0,
    )
    filtered = evaluate_edge_alignment(
        image=image, points_lidar=points_lidar, T_CL=np.eye(4),
        camera=camera, lidar_spec=lidar_spec, depth_jump_threshold_m=1.0,
        dynamic_mask=dynamic_mask,
    )
    # removing all near-side points removes the only edge points this
    # scene has (the discontinuity's near side is what M2 keeps) -> FAIL
    assert baseline.classification != "FAIL"
    assert filtered.classification == "FAIL"
    assert "No LiDAR points projected" in filtered.warnings[0] or "edge points found" in filtered.warnings[0]


def test_edge_alignment_dynamic_mask_all_false_matches_no_mask():
    camera, image, points_lidar, lidar_spec = _make_synthetic_scene()
    no_mask_result = evaluate_edge_alignment(
        image=image, points_lidar=points_lidar, T_CL=np.eye(4),
        camera=camera, lidar_spec=lidar_spec, depth_jump_threshold_m=1.0,
    )
    all_false_mask = np.zeros(points_lidar.shape[0], dtype=bool)
    masked_result = evaluate_edge_alignment(
        image=image, points_lidar=points_lidar, T_CL=np.eye(4),
        camera=camera, lidar_spec=lidar_spec, depth_jump_threshold_m=1.0,
        dynamic_mask=all_false_mask,
    )
    assert np.isclose(no_mask_result.mean_px, masked_result.mean_px)
    assert no_mask_result.num_edge_points == masked_result.num_edge_points


def test_edge_alignment_dynamic_mask_rejects_length_mismatch():
    camera, image, points_lidar, lidar_spec = _make_synthetic_scene()
    bad_mask = np.zeros(5, dtype=bool)  # wrong length
    try:
        evaluate_edge_alignment(
            image=image, points_lidar=points_lidar, T_CL=np.eye(4),
            camera=camera, lidar_spec=lidar_spec, dynamic_mask=bad_mask,
        )
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
