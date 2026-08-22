import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from evaluation.spatial_analysis import (
    bin_by_depth,
    bin_by_horizontal_region,
    bin_by_vertical_region,
    analyze_depth_and_spatial,
    analyze_depth_and_spatial_from_result,
    DEPTH_BIN_LABELS,
    HORIZONTAL_REGIONS,
    VERTICAL_REGIONS,
)


# ---------------------------------------------------------------------------
# binning helpers
# ---------------------------------------------------------------------------

def test_bin_by_depth_edges():
    depths = np.array([0.0, 5.0, 9.99, 10.0, 15.0, 19.99, 20.0, 29.99, 30.0, 49.99, 50.0, 100.0])
    idx = bin_by_depth(depths)
    expected = [0, 0, 0, 1, 1, 1, 2, 2, 3, 3, 4, 4]
    assert list(idx) == expected


def test_bin_by_depth_labels_length_matches_edges():
    assert len(DEPTH_BIN_LABELS) == 5


def test_bin_by_horizontal_region_thirds():
    width = 300
    pixels_u = np.array([0.0, 50.0, 99.0, 100.0, 150.0, 199.0, 200.0, 250.0, 299.0])
    idx = bin_by_horizontal_region(pixels_u, width)
    expected = [0, 0, 0, 1, 1, 1, 2, 2, 2]  # LEFT, CENTER, RIGHT
    assert list(idx) == expected


def test_bin_by_vertical_region_thirds():
    height = 300
    pixels_v = np.array([0.0, 50.0, 99.0, 100.0, 150.0, 199.0, 200.0, 250.0, 299.0])
    idx = bin_by_vertical_region(pixels_v, height)
    expected = [0, 0, 0, 1, 1, 1, 2, 2, 2]  # TOP, CENTER, BOTTOM
    assert list(idx) == expected


def test_bin_by_horizontal_region_clips_out_of_bounds():
    idx = bin_by_horizontal_region(np.array([-10.0, 10000.0]), 300)
    assert list(idx) == [0, 2]


# ---------------------------------------------------------------------------
# analyze_depth_and_spatial -- the spec's own worked example
# ---------------------------------------------------------------------------

def test_analyze_reproduces_spec_worked_example_error_increases_with_depth():
    """Directly build the exact scenario from evaluation_metric_spec.md's
    STEP9 example: mean error 0.8px at 0-10m, 1.0px at 10-20m, 1.8px at
    20-30m, 3.9px at 30-50m -- and confirm depth_trend correctly reads
    this as 'increases_with_depth'."""
    rng = np.random.default_rng(0)
    depths, errors, pixels = [], [], []
    bin_targets = [(5.0, 0.8), (15.0, 1.0), (25.0, 1.8), (40.0, 3.9)]
    for depth, target_mean in bin_targets:
        n = 50
        depths.extend([depth] * n)
        errors.extend(target_mean + rng.normal(0, 0.05, n))
        pixels.extend([[320.0, 240.0]] * n)  # all CENTER/CENTER, irrelevant to this test

    result = analyze_depth_and_spatial(
        np.array(errors), np.array(depths), np.array(pixels),
        image_width=640, image_height=480,
    )
    assert np.isclose(result.depth_bins["0-10m"].mean_px, 0.8, atol=0.05)
    assert np.isclose(result.depth_bins["10-20m"].mean_px, 1.0, atol=0.05)
    assert np.isclose(result.depth_bins["20-30m"].mean_px, 1.8, atol=0.05)
    assert np.isclose(result.depth_bins["30-50m"].mean_px, 3.9, atol=0.05)
    assert result.depth_bins["50m+"].valid_count + result.depth_bins["50m+"].failure_count == 0
    assert result.depth_trend == "increases_with_depth"


def test_analyze_stable_depth_trend_when_no_clear_direction():
    rng = np.random.default_rng(1)
    depths, errors, pixels = [], [], []
    bin_targets = [(5.0, 1.5), (15.0, 0.9), (25.0, 1.7), (40.0, 1.0)]  # non-monotonic
    for depth, target_mean in bin_targets:
        n = 50
        depths.extend([depth] * n)
        errors.extend(target_mean + rng.normal(0, 0.02, n))
        pixels.extend([[320.0, 240.0]] * n)

    result = analyze_depth_and_spatial(
        np.array(errors), np.array(depths), np.array(pixels),
        image_width=640, image_height=480,
    )
    assert result.depth_trend == "stable"


def test_analyze_depth_trend_none_with_too_little_data():
    # only 2 populated bins -- below min_bins_with_data=3, trend is None
    errors = np.array([1.0, 1.0, 2.0, 2.0])
    depths = np.array([5.0, 5.0, 15.0, 15.0])
    pixels = np.array([[320.0, 240.0]] * 4)
    result = analyze_depth_and_spatial(errors, depths, pixels, image_width=640, image_height=480)
    assert result.depth_trend is None


def test_analyze_valid_and_failure_counts_from_matched_array():
    errors = np.array([1.0, 2.0, 3.0, 15.0, 15.0])
    depths = np.array([5.0, 5.0, 5.0, 5.0, 5.0])
    pixels = np.array([[320.0, 240.0]] * 5)
    matched = np.array([True, True, True, False, False])

    result = analyze_depth_and_spatial(errors, depths, pixels, 640, 480, edge_point_matched=matched)
    bin_stats = result.depth_bins["0-10m"]
    assert bin_stats.valid_count == 3
    assert bin_stats.failure_count == 2
    assert bin_stats.valid_count + bin_stats.failure_count == 5


def test_analyze_without_matched_array_everything_counts_as_valid():
    errors = np.array([1.0, 2.0, 3.0])
    depths = np.array([5.0, 5.0, 5.0])
    pixels = np.array([[320.0, 240.0]] * 3)
    result = analyze_depth_and_spatial(errors, depths, pixels, 640, 480, edge_point_matched=None)
    bin_stats = result.depth_bins["0-10m"]
    assert bin_stats.valid_count == 3
    assert bin_stats.failure_count == 0


def test_analyze_horizontal_region_isolates_left_vs_right_error():
    """A scene where error is much worse on the LEFT than elsewhere --
    e.g. simulating a yaw misalignment that only shows up on one side."""
    n = 60
    errors_left = np.full(n, 8.0)
    errors_center = np.full(n, 1.0)
    errors_right = np.full(n, 1.0)
    errors = np.concatenate([errors_left, errors_center, errors_right])
    depths = np.full(3 * n, 10.0)
    pixels = np.concatenate([
        np.column_stack([np.full(n, 50.0), np.full(n, 240.0)]),   # LEFT third of a 640-wide image
        np.column_stack([np.full(n, 320.0), np.full(n, 240.0)]),  # CENTER
        np.column_stack([np.full(n, 600.0), np.full(n, 240.0)]),  # RIGHT
    ])
    result = analyze_depth_and_spatial(errors, depths, pixels, image_width=640, image_height=480)
    assert result.horizontal_regions["LEFT"].mean_px > result.horizontal_regions["CENTER"].mean_px
    assert result.horizontal_regions["LEFT"].mean_px > result.horizontal_regions["RIGHT"].mean_px
    assert np.isclose(result.horizontal_regions["CENTER"].mean_px, 1.0)


def test_analyze_vertical_region_isolates_top_vs_bottom_error():
    n = 60
    errors = np.concatenate([np.full(n, 1.0), np.full(n, 1.0), np.full(n, 7.0)])
    depths = np.full(3 * n, 10.0)
    pixels = np.concatenate([
        np.column_stack([np.full(n, 320.0), np.full(n, 50.0)]),   # TOP third of a 480-tall image
        np.column_stack([np.full(n, 320.0), np.full(n, 240.0)]),  # CENTER
        np.column_stack([np.full(n, 320.0), np.full(n, 450.0)]),  # BOTTOM
    ])
    result = analyze_depth_and_spatial(errors, depths, pixels, image_width=640, image_height=480)
    assert result.vertical_regions["BOTTOM"].mean_px > result.vertical_regions["TOP"].mean_px
    assert result.vertical_regions["BOTTOM"].mean_px > result.vertical_regions["CENTER"].mean_px


def test_bin_stats_to_dict_shape():
    errors = np.array([1.0, 2.0, 3.0])
    depths = np.array([5.0, 5.0, 5.0])
    pixels = np.array([[320.0, 240.0]] * 3)
    result = analyze_depth_and_spatial(errors, depths, pixels, 640, 480)
    d = result.depth_bins["0-10m"].to_dict()
    for key in ("label", "mean_px", "median_px", "p95_px", "std_px", "valid_count", "failure_count", "total_count"):
        assert key in d
    assert d["total_count"] == 3


def test_spatial_analysis_result_to_dict_shape():
    errors = np.array([1.0, 2.0, 3.0])
    depths = np.array([5.0, 5.0, 5.0])
    pixels = np.array([[320.0, 240.0]] * 3)
    result = analyze_depth_and_spatial(errors, depths, pixels, 640, 480)
    d = result.to_dict()
    assert set(d.keys()) == {"depth_bins", "horizontal_regions", "vertical_regions", "depth_trend"}
    assert set(d["depth_bins"].keys()) == set(DEPTH_BIN_LABELS)
    assert set(d["horizontal_regions"].keys()) == set(HORIZONTAL_REGIONS)
    assert set(d["vertical_regions"].keys()) == set(VERTICAL_REGIONS)


def test_analyze_empty_bin_has_nan_stats_not_crash():
    errors = np.array([1.0])
    depths = np.array([5.0])
    pixels = np.array([[320.0, 240.0]])
    result = analyze_depth_and_spatial(errors, depths, pixels, 640, 480)
    empty_bin = result.depth_bins["50m+"]
    assert empty_bin.valid_count == 0
    assert empty_bin.failure_count == 0
    assert np.isnan(empty_bin.mean_px)
    d = empty_bin.to_dict()
    assert d["mean_px"] is None  # NaN sanitized to None in to_dict


# ---------------------------------------------------------------------------
# analyze_depth_and_spatial_from_result
# ---------------------------------------------------------------------------

def test_analyze_from_result_convenience_wrapper():
    import sys as _sys
    from tests.test_edge_alignment import _make_synthetic_scene
    from evaluation.edge_alignment import evaluate_edge_alignment

    camera, image, points_lidar, lidar_spec = _make_synthetic_scene()
    result = evaluate_edge_alignment(image=image, points_lidar=points_lidar, T_CL=np.eye(4),
                                       camera=camera, lidar_spec=lidar_spec, depth_jump_threshold_m=1.0)
    spatial = analyze_depth_and_spatial_from_result(result, camera.width, camera.height)
    assert spatial is not None
    assert sum(s.valid_count + s.failure_count for s in spatial.depth_bins.values()) == result.num_edge_points


def test_analyze_from_result_none_on_fail():
    from tests.test_edge_alignment import _make_synthetic_scene
    from evaluation.edge_alignment import evaluate_edge_alignment

    camera, image, _, lidar_spec = _make_synthetic_scene()
    result = evaluate_edge_alignment(image=image, points_lidar=np.zeros((0, 3)), T_CL=np.eye(4),
                                       camera=camera, lidar_spec=lidar_spec)
    assert result.classification == "FAIL"
    spatial = analyze_depth_and_spatial_from_result(result, camera.width, camera.height)
    assert spatial is None


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
