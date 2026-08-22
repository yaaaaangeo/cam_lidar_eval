import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import cv2

from evaluation.edge_correspondence import (
    compute_edge_orientation_map,
    estimate_lidar_edge_orientations,
    find_correspondences,
    apply_local_consistency_filter,
    match_lidar_edges_to_image,
    CorrespondenceResult,
)


# ---------------------------------------------------------------------------
# compute_edge_orientation_map
# ---------------------------------------------------------------------------

def test_compute_edge_orientation_map_vertical_edge_has_vertical_tangent():
    image = np.zeros((100, 100), dtype=np.uint8)
    image[:, 50:] = 255  # vertical step -> edge runs vertically (tangent ~90 deg)
    om = compute_edge_orientation_map(image)
    assert om.edge_mask.sum() > 0
    rows, cols = np.nonzero(om.edge_mask)
    orientations = om.orientation_deg[rows, cols]
    # tangent should cluster near 90 (mod 180) for a vertical edge
    diffs = np.minimum(np.abs(orientations - 90.0), 180.0 - np.abs(orientations - 90.0))
    assert np.median(diffs) < 15.0


def test_compute_edge_orientation_map_horizontal_edge_has_horizontal_tangent():
    image = np.zeros((100, 100), dtype=np.uint8)
    image[50:, :] = 255  # horizontal step -> edge runs horizontally (tangent ~0 deg)
    om = compute_edge_orientation_map(image)
    rows, cols = np.nonzero(om.edge_mask)
    orientations = om.orientation_deg[rows, cols]
    diffs = np.minimum(orientations, 180.0 - orientations)  # distance to 0/180
    assert np.median(diffs) < 15.0


def test_compute_edge_orientation_map_strength_nonzero_at_edges():
    image = np.zeros((100, 100), dtype=np.uint8)
    image[:, 50:] = 255
    om = compute_edge_orientation_map(image)
    rows, cols = np.nonzero(om.edge_mask)
    assert np.all(om.strength[rows, cols] > 0)


def test_compute_edge_orientation_map_blank_image_no_edges():
    image = np.full((50, 50), 128, dtype=np.uint8)
    om = compute_edge_orientation_map(image)
    assert om.edge_mask.sum() == 0


# ---------------------------------------------------------------------------
# estimate_lidar_edge_orientations
# ---------------------------------------------------------------------------

def test_estimate_lidar_edge_orientations_vertical_line():
    pixels = np.array([[50.0, v] for v in range(90, 111)])  # constant u, varying v -> vertical
    orientations = estimate_lidar_edge_orientations(pixels, k=6)
    diffs = np.minimum(np.abs(orientations - 90.0), 180.0 - np.abs(orientations - 90.0))
    assert np.all(diffs < 5.0)


def test_estimate_lidar_edge_orientations_horizontal_line():
    pixels = np.array([[u, 50.0] for u in range(90, 111)])  # constant v, varying u -> horizontal
    orientations = estimate_lidar_edge_orientations(pixels, k=6)
    diffs = np.minimum(orientations, 180.0 - orientations)
    assert np.all(diffs < 5.0)


def test_estimate_lidar_edge_orientations_diagonal_line():
    pixels = np.array([[float(i), float(i)] for i in range(20)])  # 45-degree diagonal
    orientations = estimate_lidar_edge_orientations(pixels, k=6)
    diffs = np.minimum(np.abs(orientations - 45.0), 180.0 - np.abs(orientations - 45.0))
    assert np.all(diffs < 5.0)


def test_estimate_lidar_edge_orientations_single_point_is_nan():
    orientations = estimate_lidar_edge_orientations(np.array([[10.0, 10.0]]))
    assert np.isnan(orientations[0])


def test_estimate_lidar_edge_orientations_empty():
    orientations = estimate_lidar_edge_orientations(np.zeros((0, 2)))
    assert orientations.shape == (0,)


# ---------------------------------------------------------------------------
# find_correspondences -- THE key STEP 6 behavior: reject a closer but
# wrong-orientation candidate in favor of a farther but correctly-oriented
# one, exactly the failure mode the old nearest-distance matcher had no
# way to avoid.
# ---------------------------------------------------------------------------

def _build_distractor_scene():
    """
    A target LiDAR point at (100, 100) with several vertically-aligned
    LiDAR neighbor points (giving it a ~90 deg / vertical local
    orientation). Two candidate image edges:
      - a HORIZONTAL distractor segment, only 5px away (row 105) -- wrong
        orientation (~0 deg), should be REJECTED even though it's closer.
      - a VERTICAL correct segment, 15px away (column 115) -- right
        orientation (~90 deg), should be the one actually matched.

    Segments extend well past their closest point to the target so the
    point sampled is deep in each line's clean middle, not near a rounded
    end-cap -- cv2.line's caps curve and produce locally mixed-orientation
    pixels right at the endpoint, which would contaminate exactly the
    measurement this test depends on.
    """
    image = np.zeros((200, 200), dtype=np.uint8)
    # horizontal distractor: row 105, spanning well past column 100 on
    # both sides (closest point to (100,100) is (100,105), distance
    # exactly 5, deep in the segment's middle)
    cv2.line(image, (40, 105), (160, 105), 255, thickness=2)
    # vertical correct edge: column 115, spanning well past row 100 on
    # both sides (closest point to (100,100) is (115,100), distance
    # exactly 15, deep in the segment's middle)
    cv2.line(image, (115, 40), (115, 160), 255, thickness=2)

    lidar_pixels = np.array([[100.0, v] for v in range(90, 111)])  # vertical line through (100,100)
    return image, lidar_pixels


def test_find_correspondences_rejects_closer_wrong_orientation_candidate():
    image, lidar_pixels = _build_distractor_scene()
    om = compute_edge_orientation_map(image)
    orientations = estimate_lidar_edge_orientations(lidar_pixels, k=6)

    target_idx = 10  # (100, 100) -- the 11th point (v=100) in the vertical line
    assert lidar_pixels[target_idx].tolist() == [100.0, 100.0]

    result = find_correspondences(
        lidar_pixels, orientations, om,
        radii_px=(5.0, 10.0, 15.0), max_orientation_diff_deg=30.0,
    )

    assert result.matched[target_idx], "expected the target point to find a valid (correctly-oriented) match"
    # matched distance should reflect the CORRECT (vertical, ~15px away)
    # edge, not the closer-but-wrong-orientation horizontal one at ~5px.
    assert result.distance_px[target_idx] > 10.0, (
        f"matched distance {result.distance_px[target_idx]} suggests the wrong "
        f"(closer, horizontal) distractor was picked instead of the correct vertical edge"
    )
    assert result.matched_pixels[target_idx, 0] > 110  # landed near column 115, not row 105


def test_find_correspondences_naive_nearest_would_have_picked_the_distractor():
    """Sanity check that the distractor scene is actually adversarial --
    i.e. the OLD approach (pure nearest-any-edge distance) really would
    have picked the wrong, closer candidate, which is exactly what
    find_correspondences must avoid above."""
    image, lidar_pixels = _build_distractor_scene()
    from evaluation.edge_alignment import extract_image_edges, compute_distance_transform, sample_bilinear
    edge_map = extract_image_edges(image)
    dt = compute_distance_transform(edge_map)
    naive_distance = sample_bilinear(dt, np.array([[100.0, 100.0]]))[0]
    assert naive_distance < 10.0, (
        "expected the naive nearest-edge distance to be dominated by the "
        "closer (wrong-orientation) distractor -- if this fails, the test "
        "scene itself isn't adversarial enough to prove anything"
    )


def test_find_correspondences_no_valid_candidate_gets_max_radius_penalty():
    image = np.zeros((200, 200), dtype=np.uint8)  # no edges anywhere
    lidar_pixels = np.array([[100.0, v] for v in range(90, 111)])
    om = compute_edge_orientation_map(image)
    orientations = estimate_lidar_edge_orientations(lidar_pixels, k=6)
    result = find_correspondences(lidar_pixels, orientations, om, radii_px=(5.0, 10.0, 15.0))
    assert not result.matched.any()
    assert np.all(result.distance_px == 15.0)
    assert all(r == "no_candidate" for r in result.rejection_reason)


def test_find_correspondences_empty_input():
    image = np.zeros((50, 50), dtype=np.uint8)
    om = compute_edge_orientation_map(image)
    result = find_correspondences(np.zeros((0, 2)), np.zeros(0), om)
    assert result.matched.shape == (0,)


def test_find_correspondences_nan_orientation_point_stays_unmatched():
    """A LiDAR point with unknown (NaN) local orientation (e.g. a lone,
    isolated edge point with no neighbors) must never be allowed to just
    match the nearest thing regardless of orientation -- see
    estimate_lidar_edge_orientations' docstring."""
    image = np.zeros((200, 200), dtype=np.uint8)
    cv2.line(image, (95, 80), (95, 120), 255, thickness=2)  # a real, close edge
    lidar_pixels = np.array([[100.0, 100.0]])  # single point -> its own orientation is NaN
    om = compute_edge_orientation_map(image)
    orientations = estimate_lidar_edge_orientations(lidar_pixels)
    assert np.isnan(orientations[0])
    result = find_correspondences(lidar_pixels, orientations, om, radii_px=(5.0, 10.0, 15.0))
    assert not result.matched[0]


def test_find_correspondences_prefers_stronger_edge_among_valid_candidates():
    """Two candidates at similar distance and BOTH orientation-valid --
    the stronger (higher-contrast) one should win."""
    image = np.zeros((200, 200), dtype=np.uint8)
    # weak vertical edge (low contrast) close by
    cv2.line(image, (105, 85), (105, 115), 60, thickness=2)
    # strong vertical edge (full contrast) at a similar distance, other side
    cv2.line(image, (95, 85), (95, 115), 255, thickness=2)

    lidar_pixels = np.array([[100.0, v] for v in range(90, 111)])
    om = compute_edge_orientation_map(image, canny_low=20, canny_high=60)
    orientations = estimate_lidar_edge_orientations(lidar_pixels, k=6)
    result = find_correspondences(lidar_pixels, orientations, om, radii_px=(5.0, 10.0, 15.0),
                                   max_orientation_diff_deg=30.0)
    target_idx = 10
    if result.matched[target_idx]:
        # should have landed on the STRONG edge (column 95), not the weak one (column 105)
        assert abs(result.matched_pixels[target_idx, 0] - 95) < abs(result.matched_pixels[target_idx, 0] - 105)


# ---------------------------------------------------------------------------
# apply_local_consistency_filter
# ---------------------------------------------------------------------------

def test_local_consistency_filter_demotes_outlier_displacement():
    lidar_pixels = np.array([[float(i) * 10, 100.0] for i in range(10)])
    # 9 consistent matches (all shifted +5 in x), 1 wild outlier shifted +5 in y instead
    matched_pixels = lidar_pixels.copy()
    matched_pixels[:, 0] += 5.0
    matched_pixels[5, 0] -= 5.0  # undo the x shift for the outlier
    matched_pixels[5, 1] += 5.0  # outlier moves in y instead of x

    n = lidar_pixels.shape[0]
    raw = CorrespondenceResult(
        matched=np.ones(n, dtype=bool),
        matched_pixels=matched_pixels,
        distance_px=np.full(n, 5.0),
        orientation_diff_deg=np.zeros(n),
        strength=np.full(n, 100.0),
        num_candidates_considered=np.ones(n, dtype=np.int64),
        rejection_reason=[None] * n,
    )
    filtered = apply_local_consistency_filter(lidar_pixels, raw, k=5, max_angle_diff_deg=45.0)
    assert not filtered.matched[5], "the direction-outlier point should be demoted"
    assert filtered.matched[:5].all() and filtered.matched[6:].all(), "consistent points should stay matched"
    assert filtered.rejection_reason[5] == "consistency"


def test_local_consistency_filter_leaves_uniform_matches_alone():
    lidar_pixels = np.array([[float(i) * 10, 100.0] for i in range(10)])
    matched_pixels = lidar_pixels.copy()
    matched_pixels[:, 0] += 5.0  # every point shifted identically -> perfectly consistent

    n = lidar_pixels.shape[0]
    raw = CorrespondenceResult(
        matched=np.ones(n, dtype=bool), matched_pixels=matched_pixels,
        distance_px=np.full(n, 5.0), orientation_diff_deg=np.zeros(n),
        strength=np.full(n, 100.0), num_candidates_considered=np.ones(n, dtype=np.int64),
        rejection_reason=[None] * n,
    )
    filtered = apply_local_consistency_filter(lidar_pixels, raw, k=5)
    assert filtered.matched.all()


def test_local_consistency_filter_noop_when_too_few_matches():
    lidar_pixels = np.array([[0.0, 0.0], [10.0, 0.0]])
    matched_pixels = np.array([[5.0, 0.0], [15.0, 0.0]])
    raw = CorrespondenceResult(
        matched=np.array([True, True]), matched_pixels=matched_pixels,
        distance_px=np.array([5.0, 5.0]), orientation_diff_deg=np.zeros(2),
        strength=np.full(2, 100.0), num_candidates_considered=np.ones(2, dtype=np.int64),
        rejection_reason=[None, None],
    )
    filtered = apply_local_consistency_filter(lidar_pixels, raw, k=5)
    assert filtered.matched.all()  # too few (< 3) matched points anywhere -- left alone


def test_local_consistency_filter_empty_matched_set():
    lidar_pixels = np.zeros((3, 2))
    raw = CorrespondenceResult(
        matched=np.zeros(3, dtype=bool), matched_pixels=np.full((3, 2), np.nan),
        distance_px=np.full(3, 15.0), orientation_diff_deg=np.full(3, np.nan),
        strength=np.full(3, np.nan), num_candidates_considered=np.zeros(3, dtype=np.int64),
        rejection_reason=["no_candidate"] * 3,
    )
    filtered = apply_local_consistency_filter(lidar_pixels, raw, k=5)
    assert not filtered.matched.any()


# ---------------------------------------------------------------------------
# match_lidar_edges_to_image -- full pipeline
# ---------------------------------------------------------------------------

def test_match_lidar_edges_to_image_full_pipeline_on_distractor_scene():
    image, lidar_pixels = _build_distractor_scene()
    result = match_lidar_edges_to_image(lidar_pixels, image, radii_px=(5.0, 10.0, 15.0),
                                         max_orientation_diff_deg=30.0)
    target_idx = 10
    assert result.matched[target_idx]
    assert result.distance_px[target_idx] > 10.0


def test_match_lidar_edges_to_image_clean_vertical_edge_matches_precisely():
    image = np.zeros((200, 200), dtype=np.uint8)
    image[:, 100:] = 255  # clean vertical step at column 100
    lidar_pixels = np.array([[100.0, v] for v in range(80, 121)])
    result = match_lidar_edges_to_image(lidar_pixels, image)
    assert result.matched.sum() > 0
    matched_distances = result.distance_px[result.matched]
    assert np.mean(matched_distances) < 2.0  # should land right on the true edge


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
