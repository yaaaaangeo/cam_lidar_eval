import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from geometry.range_image import build_range_image, compute_edge_cell_mask
from visualization.range_image import render_range_image_png, render_range_image_from_points


def _half_near_half_far_ring(num_bins=36, near=5.0, far=10.0, near_count=18):
    angles = (np.arange(num_bins) + 0.5) * (2 * np.pi / num_bins)
    ranges = np.where(np.arange(num_bins) < near_count, near, far)
    x = ranges * np.cos(angles)
    y = ranges * np.sin(angles)
    z = np.zeros(num_bins)
    return np.stack([x, y, z], axis=1)


def test_render_range_image_png_returns_valid_png():
    points = _half_near_half_far_ring()
    ri = build_range_image(points, num_rings=1, num_azimuth_bins=36)
    png = render_range_image_png(ri)
    assert png is not None
    assert png[:8] == b"\x89PNG\r\n\x1a\n"


def test_render_range_image_png_with_edge_mask():
    points = _half_near_half_far_ring()
    ri = build_range_image(points, num_rings=1, num_azimuth_bins=36)
    edge_mask = compute_edge_cell_mask(ri, depth_jump_threshold_m=1.0)
    assert edge_mask.sum() == 2
    png = render_range_image_png(ri, edge_cell_mask=edge_mask)
    assert png is not None
    assert png[:8] == b"\x89PNG\r\n\x1a\n"


def test_render_range_image_png_none_when_all_empty():
    ri = build_range_image(np.zeros((0, 3)), num_rings=4, num_azimuth_bins=8)
    png = render_range_image_png(ri)
    assert png is None


def test_render_range_image_png_edge_mask_with_no_edges_still_renders():
    points = _half_near_half_far_ring(near=5.0, far=5.0, near_count=36)  # uniform, no edges
    ri = build_range_image(points, num_rings=1, num_azimuth_bins=36)
    edge_mask = compute_edge_cell_mask(ri, depth_jump_threshold_m=0.3)
    assert not edge_mask.any()
    png = render_range_image_png(ri, edge_cell_mask=edge_mask)
    assert png is not None  # still renders the range image, just no markers


def test_render_range_image_from_points_convenience_wrapper():
    points = _half_near_half_far_ring()
    png = render_range_image_from_points(points, num_rings=1, num_azimuth_bins=36, depth_jump_threshold_m=1.0)
    assert png is not None
    assert png[:8] == b"\x89PNG\r\n\x1a\n"


def test_render_range_image_from_points_show_edges_false_skips_computation():
    points = _half_near_half_far_ring()
    png = render_range_image_from_points(points, num_rings=1, num_azimuth_bins=36, show_edges=False)
    assert png is not None  # still renders, just without edge markers


def test_render_range_image_from_points_empty_returns_none():
    png = render_range_image_from_points(np.zeros((0, 3)))
    assert png is None


def test_render_range_image_png_returns_none_on_broken_3d_env():
    """Consistent with camera_frustum.py / colorized_pointcloud.py: a
    broken/partial matplotlib install must degrade to None, not crash."""
    import matplotlib.figure
    points = _half_near_half_far_ring()
    ri = build_range_image(points, num_rings=1, num_azimuth_bins=36)

    original_add_subplot = matplotlib.figure.Figure.add_subplot

    def _broken_add_subplot(self, *args, **kwargs):
        raise RuntimeError("simulated broken plotting environment")

    matplotlib.figure.Figure.add_subplot = _broken_add_subplot
    try:
        png = render_range_image_png(ri)
        assert png is None
    finally:
        matplotlib.figure.Figure.add_subplot = original_add_subplot


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
