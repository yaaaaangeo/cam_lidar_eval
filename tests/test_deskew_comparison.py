import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from motion.deskew import deskew_points_constant_velocity
from visualization.deskew_comparison import (
    render_deskew_comparison_png,
    render_deskew_comparison_from_points,
)


def _random_points(n=200, seed=0):
    return np.random.default_rng(seed).uniform(-10, 10, size=(n, 3))


def test_render_deskew_comparison_png_returns_valid_png():
    points = _random_points()
    result = deskew_points_constant_velocity(
        points, scan_period_s=0.1,
        linear_velocity_mps=np.array([3.0, 0, 0]), angular_velocity_rps=np.array([0, 0, 0.5]),
    )
    png = render_deskew_comparison_png(points, result)
    assert png is not None
    assert png[:8] == b"\x89PNG\r\n\x1a\n"


def test_render_deskew_comparison_png_stationary_case_still_renders():
    points = _random_points()
    result = deskew_points_constant_velocity(
        points, scan_period_s=0.1, linear_velocity_mps=np.zeros(3), angular_velocity_rps=np.zeros(3),
    )
    assert result.max_correction_m == 0.0
    png = render_deskew_comparison_png(points, result)
    assert png is not None  # still renders (empty histogram, overlapping before/after)


def test_render_deskew_comparison_png_none_for_empty_points():
    points = np.zeros((0, 3))
    result = deskew_points_constant_velocity(
        points, scan_period_s=0.1, linear_velocity_mps=np.zeros(3), angular_velocity_rps=np.zeros(3),
    )
    png = render_deskew_comparison_png(points, result)
    assert png is None


def test_render_deskew_comparison_png_subsamples_large_clouds():
    points = _random_points(n=50_000)
    result = deskew_points_constant_velocity(
        points, scan_period_s=0.1,
        linear_velocity_mps=np.array([2.0, 0, 0]), angular_velocity_rps=np.zeros(3),
    )
    png = render_deskew_comparison_png(points, result, max_points=1000)
    assert png is not None


def test_render_deskew_comparison_from_points_convenience_wrapper():
    points = _random_points()
    png = render_deskew_comparison_from_points(
        points, scan_period_s=0.1,
        linear_velocity_mps=np.array([4.0, 1.0, 0.0]), angular_velocity_rps=np.array([0, 0, 0.3]),
    )
    assert png is not None
    assert png[:8] == b"\x89PNG\r\n\x1a\n"


def test_render_deskew_comparison_from_points_empty_returns_none():
    png = render_deskew_comparison_from_points(
        np.zeros((0, 3)), scan_period_s=0.1,
        linear_velocity_mps=np.zeros(3), angular_velocity_rps=np.zeros(3),
    )
    assert png is None


def test_render_deskew_comparison_png_returns_none_on_broken_plotting_env():
    """Consistent with camera_frustum.py / colorized_pointcloud.py /
    range_image.py: a broken/partial matplotlib install must degrade to
    None, not crash."""
    import matplotlib.figure
    points = _random_points()
    result = deskew_points_constant_velocity(
        points, scan_period_s=0.1,
        linear_velocity_mps=np.array([2.0, 0, 0]), angular_velocity_rps=np.zeros(3),
    )

    original_add_subplot = matplotlib.figure.Figure.add_subplot

    def _broken_add_subplot(self, *args, **kwargs):
        raise RuntimeError("simulated broken plotting environment")

    matplotlib.figure.Figure.add_subplot = _broken_add_subplot
    try:
        png = render_deskew_comparison_png(points, result)
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
