import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from evaluation.dynamic_filter import (
    classify_points_by_motion_consistency,
    dynamic_point_mask,
    apply_external_dynamic_mask,
    compare_with_without_dynamic_filtering,
    MotionConsistencyResult,
    DynamicFilteringComparison,
    STATIC,
    DYNAMIC,
    UNKNOWN,
)


def _ring_scan(num_rings, num_az, ranges_per_azimuth):
    """Build a full structured scan: `ranges_per_azimuth` is an array of
    length num_az giving each azimuth column's range (same for every
    ring, for simplicity -- these tests only care about azimuth-indexed
    behavior)."""
    elevations = np.radians(np.linspace(-10, 10, num_rings))
    azimuths = np.linspace(0, 2 * np.pi, num_az, endpoint=False)
    points = []
    for el in elevations:
        for i, az in enumerate(azimuths):
            r = ranges_per_azimuth[i]
            x = r * np.cos(el) * np.cos(az)
            y = r * np.cos(el) * np.sin(az)
            z = r * np.sin(el)
            points.append([x, y, z])
    return np.array(points)


# ---------------------------------------------------------------------------
# classify_points_by_motion_consistency
# ---------------------------------------------------------------------------

def test_classify_all_static_scene_stable_ranges():
    num_rings, num_az = 8, 36
    base_ranges = np.full(num_az, 10.0)
    # 5 frames of the SAME static scene (tiny sensor noise only)
    rng = np.random.default_rng(0)
    frames = [_ring_scan(num_rings, num_az, base_ranges + rng.normal(0, 0.01, num_az)) for _ in range(5)]

    result = classify_points_by_motion_consistency(
        frames, num_rings=num_rings, num_azimuth_bins=num_az,
        range_std_threshold_m=0.3, min_frames_present=3,
    )
    counts = result.label_counts()
    assert counts["dynamic"] == 0
    assert counts["static"] > 0


def test_classify_one_moving_azimuth_band_flagged_dynamic():
    num_rings, num_az = 8, 36
    base_ranges = np.full(num_az, 10.0)
    moving_bins = slice(10, 15)  # a band of azimuths where range changes a lot frame-to-frame

    rng = np.random.default_rng(1)
    frames = []
    for i in range(6):
        ranges = base_ranges.copy() + rng.normal(0, 0.01, num_az)
        ranges[moving_bins] = rng.uniform(3.0, 9.0, moving_bins.stop - moving_bins.start)  # object moving through
        frames.append(_ring_scan(num_rings, num_az, ranges))

    result = classify_points_by_motion_consistency(
        frames, num_rings=num_rings, num_azimuth_bins=num_az,
        range_std_threshold_m=0.3, min_frames_present=3,
    )
    # the moving band should be DYNAMIC in every ring
    assert np.all(result.cell_label[:, moving_bins] == DYNAMIC)
    # everywhere else, the important property is NO FALSE POSITIVES (never
    # DYNAMIC outside the moving band) -- a small fraction of bins can
    # legitimately land as UNKNOWN due to azimuth-bin quantization at this
    # scene's coarse 36-bin resolution (the same discretization edge case
    # noted in geometry/range_image.py's own tests), so this doesn't
    # require every non-moving cell to be positively STATIC, only that
    # none are wrongly flagged DYNAMIC.
    static_bins = [i for i in range(num_az) if not (10 <= i < 15)]
    sub = result.cell_label[:, static_bins]
    assert not np.any(sub == DYNAMIC)
    assert np.mean(sub == STATIC) > 0.8  # the overwhelming majority correctly classified


def test_classify_insufficient_frames_present_is_unknown():
    num_rings, num_az = 4, 8
    # only 2 frames given, but min_frames_present=3 -- nothing should be
    # classifiable as static/dynamic, only UNKNOWN
    frames = [
        _ring_scan(num_rings, num_az, np.full(num_az, 10.0)),
        _ring_scan(num_rings, num_az, np.full(num_az, 10.0)),
    ]
    result = classify_points_by_motion_consistency(
        frames, num_rings=num_rings, num_azimuth_bins=num_az, min_frames_present=3,
    )
    counts = result.label_counts()
    assert counts["static"] == 0
    assert counts["dynamic"] == 0
    assert counts["unknown"] == num_rings * num_az


def test_classify_empty_frame_list():
    result = classify_points_by_motion_consistency([], num_rings=4, num_azimuth_bins=8)
    assert result.num_frames_used == 0
    counts = result.label_counts()
    assert counts["unknown"] == 4 * 8


def test_classify_threshold_controls_sensitivity():
    num_rings, num_az = 4, 12
    base = np.full(num_az, 10.0)
    band = slice(3, 6)
    rng = np.random.default_rng(2)
    frames = []
    for _ in range(5):
        ranges = base.copy()
        ranges[band] += rng.normal(0, 0.15, band.stop - band.start)  # small (15cm) jitter, not a real mover
        frames.append(_ring_scan(num_rings, num_az, ranges))

    strict = classify_points_by_motion_consistency(frames, num_rings=num_rings, num_azimuth_bins=num_az,
                                                     range_std_threshold_m=0.05, min_frames_present=3)
    loose = classify_points_by_motion_consistency(frames, num_rings=num_rings, num_azimuth_bins=num_az,
                                                    range_std_threshold_m=1.0, min_frames_present=3)
    assert strict.label_counts()["dynamic"] > 0    # 5cm threshold: 15cm jitter trips it
    assert loose.label_counts()["dynamic"] == 0    # 1m threshold: 15cm jitter doesn't


def test_motion_consistency_result_label_counts_sum_to_all_cells():
    num_rings, num_az = 6, 20
    frames = [_ring_scan(num_rings, num_az, np.full(num_az, 10.0)) for _ in range(4)]
    result = classify_points_by_motion_consistency(frames, num_rings=num_rings, num_azimuth_bins=num_az)
    counts = result.label_counts()
    assert sum(counts.values()) == num_rings * num_az


# ---------------------------------------------------------------------------
# dynamic_point_mask
# ---------------------------------------------------------------------------

def test_dynamic_point_mask_flags_points_in_dynamic_cells():
    num_rings, num_az = 4, 12
    base = np.full(num_az, 10.0)
    moving_bins = slice(2, 4)
    rng = np.random.default_rng(3)
    frames = []
    for _ in range(5):
        ranges = base.copy()
        ranges[moving_bins] = rng.uniform(3.0, 9.0, moving_bins.stop - moving_bins.start)
        frames.append(_ring_scan(num_rings, num_az, ranges))

    result = classify_points_by_motion_consistency(frames, num_rings=num_rings, num_azimuth_bins=num_az,
                                                     range_std_threshold_m=0.3, min_frames_present=3)

    # evaluate a NEW frame (the "reference" frame) with the same structure
    reference_ranges = base.copy()
    reference_ranges[moving_bins] = 6.0  # the object is somewhere in the moving band right now
    reference_points = _ring_scan(num_rings, num_az, reference_ranges)

    mask = dynamic_point_mask(reference_points, result)
    assert mask.sum() > 0
    # points NOT in the moving azimuth band should never be flagged
    az = np.arctan2(reference_points[:, 1], reference_points[:, 0]) % (2 * np.pi)
    az_bin = np.floor(az / (2 * np.pi) * num_az).astype(int)
    in_moving_band = (az_bin >= 2) & (az_bin < 4)
    assert not mask[~in_moving_band].any()


def test_dynamic_point_mask_empty_points():
    result = classify_points_by_motion_consistency([], num_rings=4, num_azimuth_bins=8)
    mask = dynamic_point_mask(np.zeros((0, 3)), result)
    assert mask.shape == (0,)


def test_dynamic_point_mask_treat_unknown_as_dynamic_flag():
    # a result with only UNKNOWN cells (not enough frames)
    result = classify_points_by_motion_consistency(
        [_ring_scan(4, 8, np.full(8, 10.0))], num_rings=4, num_azimuth_bins=8, min_frames_present=3,
    )
    points = _ring_scan(4, 8, np.full(8, 10.0))
    mask_default = dynamic_point_mask(points, result)
    mask_conservative = dynamic_point_mask(points, result, treat_unknown_as_dynamic=True)
    assert not mask_default.any()       # UNKNOWN is not excluded by default
    assert mask_conservative.all()      # but IS excluded under the conservative flag


# ---------------------------------------------------------------------------
# apply_external_dynamic_mask
# ---------------------------------------------------------------------------

def test_apply_external_dynamic_mask_passthrough():
    points = np.random.default_rng(0).uniform(-5, 5, size=(10, 3))
    mask = np.array([True, False, True, False, False, True, False, False, False, True])
    result = apply_external_dynamic_mask(points, mask)
    assert np.array_equal(result, mask)


def test_apply_external_dynamic_mask_rejects_length_mismatch():
    points = np.zeros((5, 3))
    mask = np.zeros(3, dtype=bool)
    try:
        apply_external_dynamic_mask(points, mask)
        assert False, "expected ValueError"
    except ValueError:
        pass


# ---------------------------------------------------------------------------
# compare_with_without_dynamic_filtering -- STEP8's headline output
# ---------------------------------------------------------------------------

def _make_synthetic_scene_with_moving_wedge():
    """A depth-step scene (matching evaluation/edge_alignment tests'
    _make_synthetic_scene) plus an azimuthal wedge of points contaminated
    with a "moving object" offset, so the calibration-affecting real edge
    is separate from the artificially bad wedge."""
    import cv2
    from input.camera import CameraModel, CameraIntrinsics, CameraDistortion, CameraSource

    width, height = 640, 480
    fx = fy = 500.0
    cx, cy = 320.0, 240.0

    camera = CameraModel(
        width=width, height=height, model="pinhole",
        intrinsics=CameraIntrinsics(fx, fy, cx, cy),
        distortion=CameraDistortion(model="none"),
        source=CameraSource(kind="image_dir", path="."),
    )
    image = np.zeros((height, width), dtype=np.uint8)
    image[:, int(cx):] = 255
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    u_vals = np.linspace(0, width - 1, 220)
    v_vals = np.linspace(0, height - 1, 140)
    uu, vv = np.meshgrid(u_vals, v_vals)
    uu, vv = uu.ravel(), vv.ravel()
    z_near, z_far = 5.0, 10.0
    zz = np.where(uu < cx, z_near, z_far)
    xx = (uu - cx) * zz / fx
    yy = (vv - cy) * zz / fy
    points_lidar = np.stack([xx, yy, zz], axis=1)

    from input.lidar import LidarSensorSpec
    lidar_spec = LidarSensorSpec(horizontal_resolution_deg=0.05, vertical_resolution_deg=0.05, range_accuracy_m=0.02)

    # "moving object" contamination: a spatially LOCALIZED band of
    # near-side points (simulating one object occupying part of the
    # frame, not scattered noise across the whole boundary) get a large
    # depth-independent shift, simulating a LiDAR return from an object
    # that had already moved by capture time (so it now projects to the
    # wrong place relative to the static scene under this T_CL). Kept
    # localized and a MINORITY of the near side so plenty of clean
    # boundary remains for the static-only run to still clear
    # min_edge_points.
    near_side = np.nonzero(points_lidar[:, 2] < 7.0)[0]
    near_y = points_lidar[near_side, 1]
    band = near_side[(near_y > np.percentile(near_y, 40)) & (near_y < np.percentile(near_y, 55))]
    dynamic_mask = np.zeros(points_lidar.shape[0], dtype=bool)
    dynamic_mask[band] = True
    rng = np.random.default_rng(7)
    points_lidar[band, 0] += rng.uniform(0.5, 1.5, size=band.size)  # shifts them off the true edge

    return camera, image, points_lidar, lidar_spec, dynamic_mask


def test_compare_with_without_dynamic_filtering_static_only_improves():
    camera, image, points_lidar, lidar_spec, dynamic_mask = _make_synthetic_scene_with_moving_wedge()
    comparison = compare_with_without_dynamic_filtering(
        image=image, points_lidar=points_lidar, T_CL=np.eye(4), camera=camera, lidar_spec=lidar_spec,
        dynamic_mask=dynamic_mask, depth_jump_threshold_m=1.0,
    )
    assert comparison.static_only_mean_px < comparison.overall_mean_px
    assert comparison.dynamic_contamination_ratio > 0.0
    assert comparison.num_dynamic_points_removed == int(dynamic_mask.sum())


def test_compare_with_without_dynamic_filtering_zero_contamination_when_mask_all_false():
    camera, image, points_lidar, lidar_spec, _ = _make_synthetic_scene_with_moving_wedge()
    no_dynamic = np.zeros(points_lidar.shape[0], dtype=bool)
    comparison = compare_with_without_dynamic_filtering(
        image=image, points_lidar=points_lidar, T_CL=np.eye(4), camera=camera, lidar_spec=lidar_spec,
        dynamic_mask=no_dynamic, depth_jump_threshold_m=1.0,
    )
    assert comparison.dynamic_contamination_ratio == 0.0
    assert np.isclose(comparison.overall_mean_px, comparison.static_only_mean_px)


def test_compare_with_without_dynamic_filtering_rejects_length_mismatch():
    camera, image, points_lidar, lidar_spec, _ = _make_synthetic_scene_with_moving_wedge()
    bad_mask = np.zeros(3, dtype=bool)
    try:
        compare_with_without_dynamic_filtering(
            image=image, points_lidar=points_lidar, T_CL=np.eye(4), camera=camera, lidar_spec=lidar_spec,
            dynamic_mask=bad_mask,
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
