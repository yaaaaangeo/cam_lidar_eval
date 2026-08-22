import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from geometry.range_image import (
    compute_azimuth_rad,
    compute_elevation_rad,
    derive_ring_index,
    build_range_image,
    compute_edge_cell_mask,
    extract_lidar_native_edge_points,
    RangeImage,
)


# ---------------------------------------------------------------------------
# Basic angle helpers
# ---------------------------------------------------------------------------

def test_compute_azimuth_rad_wraps_to_0_2pi():
    points = np.array([
        [1.0, 0.0, 0.0],   # az = 0
        [0.0, 1.0, 0.0],   # az = pi/2
        [-1.0, 0.0, 0.0],  # az = pi
        [0.0, -1.0, 0.0],  # az = 3pi/2 (not negative)
    ])
    az = compute_azimuth_rad(points)
    assert np.all(az >= 0) and np.all(az < 2 * np.pi)
    assert np.isclose(az[0], 0.0)
    assert np.isclose(az[1], np.pi / 2)
    assert np.isclose(az[2], np.pi)
    assert np.isclose(az[3], 3 * np.pi / 2)


def test_compute_elevation_rad_horizontal_and_up():
    points = np.array([
        [1.0, 0.0, 0.0],   # elevation 0
        [1.0, 0.0, 1.0],   # elevation 45 deg
        [0.0, 0.0, 5.0],   # straight up -- horizontal dist 0 -> elevation pi/2
    ])
    el = compute_elevation_rad(points)
    assert np.isclose(el[0], 0.0)
    assert np.isclose(el[1], np.pi / 4)
    assert np.isclose(el[2], np.pi / 2)


# ---------------------------------------------------------------------------
# Ring derivation (pseudo-ring from elevation)
# ---------------------------------------------------------------------------

def test_derive_ring_index_separates_two_elevation_clusters():
    low = np.array([[5.0, 0.0, 0.0]] * 5)   # elevation ~0
    high = np.array([[5.0, 0.0, 5.0]] * 5)  # elevation ~45deg
    points = np.vstack([low, high])
    ring = derive_ring_index(points, num_rings=2)
    assert ring.shape == (10,)
    # lower elevation -> lower ring index, higher elevation -> higher ring index
    assert np.all(ring[:5] == ring[0])
    assert np.all(ring[5:] == ring[5])
    assert ring[0] < ring[5]


def test_derive_ring_index_monotonic_with_elevation():
    angles_deg = np.linspace(-20, 20, 40)
    horiz = 10.0
    points = np.stack([
        horiz * np.ones_like(angles_deg),
        np.zeros_like(angles_deg),
        horiz * np.tan(np.radians(angles_deg)),
    ], axis=1)
    ring = derive_ring_index(points, num_rings=8)
    order = np.argsort(angles_deg)
    ring_sorted = ring[order]
    # ring index should be non-decreasing as elevation increases
    assert np.all(np.diff(ring_sorted) >= 0)


def test_derive_ring_index_degenerate_flat_plane_all_ring_zero():
    # every point at z=0 -> zero elevation spread -> the "degenerate"
    # single-ring branch, not a divide-by-zero crash.
    points = np.array([[float(i), 0.0, 0.0] for i in range(1, 6)])
    ring = derive_ring_index(points, num_rings=16)
    assert np.all(ring == 0)


def test_derive_ring_index_empty_input():
    ring = derive_ring_index(np.zeros((0, 3)), num_rings=16)
    assert ring.shape == (0,)


def test_derive_ring_index_respects_explicit_vertical_fov():
    # With vertical_fov_deg fixed, ring boundaries come from the SPEC, not
    # the observed data -- two frames with different observed elevation
    # spreads but the same fov should bin identical elevations identically.
    points_a = np.array([[10.0, 0.0, 0.0], [10.0, 0.0, 1.0]])  # small spread
    points_b = np.array([[10.0, 0.0, 0.0], [10.0, 0.0, 1.0], [10.0, 0.0, 8.0]])  # wider spread
    ring_a = derive_ring_index(points_a, num_rings=32, vertical_fov_deg=40.0)
    ring_b = derive_ring_index(points_b, num_rings=32, vertical_fov_deg=40.0)
    # the shared points (same elevation) should land in the SAME ring in
    # both calls, since the fov (not observed spread) defines the bins.
    assert ring_a[0] == ring_b[0]
    assert ring_a[1] == ring_b[1]


# ---------------------------------------------------------------------------
# build_range_image
# ---------------------------------------------------------------------------

def test_build_range_image_shape_and_empty_cells_are_nan():
    points = np.array([[5.0, 0.0, 0.0]])
    ri = build_range_image(points, num_rings=4, num_azimuth_bins=8)
    assert ri.range_m.shape == (4, 8)
    assert ri.point_index.shape == (4, 8)
    assert np.isnan(ri.range_m).sum() == 4 * 8 - 1  # exactly one cell populated
    assert (ri.point_index == -1).sum() == 4 * 8 - 1


def test_build_range_image_empty_points():
    ri = build_range_image(np.zeros((0, 3)), num_rings=4, num_azimuth_bins=8)
    assert ri.range_m.shape == (4, 8)
    assert np.all(np.isnan(ri.range_m))
    assert ri.ring.shape == (0,)


def test_build_range_image_nearest_point_wins_on_collision():
    # Two points landing in the SAME cell (same ring, same azimuth bin,
    # different range) -- the nearer one should be the one recorded.
    near = np.array([3.0, 0.0, 0.0])
    far = np.array([9.0, 0.0, 0.0])
    points = np.stack([far, near])  # far listed FIRST, to prove order doesn't matter
    ri = build_range_image(points, num_rings=1, num_azimuth_bins=4)
    populated = ~np.isnan(ri.range_m)
    assert populated.sum() == 1
    assert np.isclose(ri.range_m[populated][0], 3.0)
    # point_index should point at the NEAR point's original index (1), not far's (0)
    idx = ri.point_index[populated][0]
    assert idx == 1


def test_build_range_image_explicit_ring_used_when_given():
    points = np.array([[5.0, 0.0, 0.0], [5.0, 0.0, 100.0]])  # 2nd point's z would derive a very different ring
    explicit_ring = np.array([0, 0])  # force both into ring 0
    ri = build_range_image(points, ring=explicit_ring, num_rings=2, num_azimuth_bins=8)
    assert np.array_equal(ri.ring, explicit_ring)
    # both points share the same azimuth (y=0, x=5 for both) so they'd
    # collide in ring 0 -- nearest (both range ~5 and ~100) -- just check
    # ring 1 stayed completely empty since neither point was assigned there.
    assert np.all(np.isnan(ri.range_m[1]))


# ---------------------------------------------------------------------------
# extract_lidar_native_edge_points -- the core STEP 4 deliverable
# ---------------------------------------------------------------------------

def _half_near_half_far_ring(num_bins=36, near=5.0, far=10.0, near_count=18):
    """A single ring of points around a full circle: near_count consecutive
    azimuth bins at `near` range, the rest at `far` range -- produces
    exactly two boundaries (one at the near->far transition, one at the
    far->near wraparound)."""
    angles = (np.arange(num_bins) + 0.5) * (2 * np.pi / num_bins)
    ranges = np.where(np.arange(num_bins) < near_count, near, far)
    x = ranges * np.cos(angles)
    y = ranges * np.sin(angles)
    z = np.zeros(num_bins)
    return np.stack([x, y, z], axis=1)


def test_extract_lidar_native_edge_points_flags_only_near_side():
    num_bins = 36
    points = _half_near_half_far_ring(num_bins=num_bins, near=5.0, far=10.0, near_count=18)
    mask = extract_lidar_native_edge_points(
        points, num_rings=1, num_azimuth_bins=num_bins, depth_jump_threshold_m=1.0,
    )
    assert mask.sum() == 2
    # near-side edge points sit at bin index 17 (last near-range bin,
    # adjacent to the far side) and bin index 0 (first near-range bin,
    # adjacent -- via wraparound -- to the last far-range bin)
    assert mask[17] == True
    assert mask[0] == True
    # the FAR side of each boundary (bin 18, bin 35) must NOT be flagged
    assert mask[18] == False
    assert mask[35] == False
    # interior points of each uniform run must not be flagged
    assert mask[5] == False
    assert mask[25] == False


def test_extract_lidar_native_edge_points_no_wrap_disables_boundary_bin():
    num_bins = 36
    points = _half_near_half_far_ring(num_bins=num_bins, near=5.0, far=10.0, near_count=18)
    mask = extract_lidar_native_edge_points(
        points, num_rings=1, num_azimuth_bins=num_bins, depth_jump_threshold_m=1.0,
        wrap_azimuth=False,
    )
    # with wraparound disabled, bin 0 has no left neighbor to compare
    # against (bin 35's far range is no longer "adjacent"), so only the
    # genuine near->far transition at bin 17 remains flagged.
    assert mask.sum() == 1
    assert mask[17] == True
    assert mask[0] == False


def test_extract_lidar_native_edge_points_uniform_range_no_false_positives():
    num_bins = 36
    points = _half_near_half_far_ring(num_bins=num_bins, near=5.0, far=5.0, near_count=num_bins)
    mask = extract_lidar_native_edge_points(
        points, num_rings=1, num_azimuth_bins=num_bins, depth_jump_threshold_m=0.3,
    )
    assert mask.sum() == 0


def test_extract_lidar_native_edge_points_threshold_controls_sensitivity():
    num_bins = 36
    points = _half_near_half_far_ring(num_bins=num_bins, near=9.0, far=9.2, near_count=18)  # small 20cm step
    mask_strict = extract_lidar_native_edge_points(
        points, num_rings=1, num_azimuth_bins=num_bins, depth_jump_threshold_m=0.3,
    )
    mask_loose = extract_lidar_native_edge_points(
        points, num_rings=1, num_azimuth_bins=num_bins, depth_jump_threshold_m=0.05,
    )
    assert mask_strict.sum() == 0    # 20cm step doesn't clear a 30cm threshold
    assert mask_loose.sum() == 2     # but does clear a 5cm threshold


def test_extract_lidar_native_edge_points_empty_input():
    mask = extract_lidar_native_edge_points(np.zeros((0, 3)))
    assert mask.shape == (0,)


def test_extract_lidar_native_edge_points_multi_ring_independent():
    """Edges in one ring shouldn't be affected by (or bleed into) another
    ring -- each ring's azimuth adjacency is evaluated independently."""
    num_bins = 36
    ring0 = _half_near_half_far_ring(num_bins=num_bins, near=5.0, far=10.0, near_count=18)
    ring1 = _half_near_half_far_ring(num_bins=num_bins, near=5.0, far=5.0, near_count=num_bins)  # uniform
    ring1[:, 2] += 3.0  # lift to a different elevation so it lands in a different ring
    points = np.vstack([ring0, ring1])
    explicit_ring = np.concatenate([np.zeros(num_bins, dtype=int), np.ones(num_bins, dtype=int)])
    mask = extract_lidar_native_edge_points(
        points, ring=explicit_ring, num_rings=2, num_azimuth_bins=num_bins, depth_jump_threshold_m=1.0,
    )
    assert mask[:num_bins].sum() == 2       # ring 0 has its usual 2 edges
    assert mask[num_bins:].sum() == 0       # ring 1 is uniform -- no edges


# ---------------------------------------------------------------------------
# compute_edge_cell_mask -- the cell-level primitive both point extraction
# and visualization build on
# ---------------------------------------------------------------------------

def test_compute_edge_cell_mask_matches_point_level_extraction():
    """The cell-level mask and the point-level mask must agree: a point is
    flagged iff its (ring, azimuth_bin) cell is flagged."""
    num_bins = 36
    points = _half_near_half_far_ring(num_bins=num_bins, near=5.0, far=10.0, near_count=18)
    ri = build_range_image(points, num_rings=1, num_azimuth_bins=num_bins)
    cell_mask = compute_edge_cell_mask(ri, depth_jump_threshold_m=1.0)
    assert cell_mask.shape == (1, num_bins)
    assert cell_mask.sum() == 2
    assert cell_mask[0, 17] and cell_mask[0, 0]

    point_mask = extract_lidar_native_edge_points(points, num_rings=1, num_azimuth_bins=num_bins,
                                                    depth_jump_threshold_m=1.0)
    # cross-check: every point whose cell is flagged should itself be flagged
    for i in range(num_bins):
        r, c = ri.ring[i], ri.azimuth_bin[i]
        assert point_mask[i] == bool(cell_mask[r, c])


def test_compute_edge_cell_mask_all_nan_range_image_has_no_edges():
    ri = build_range_image(np.zeros((0, 3)), num_rings=4, num_azimuth_bins=8)
    cell_mask = compute_edge_cell_mask(ri, depth_jump_threshold_m=0.3)
    assert cell_mask.shape == (4, 8)
    assert not cell_mask.any()


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
