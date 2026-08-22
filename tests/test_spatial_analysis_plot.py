import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from evaluation.spatial_analysis import analyze_depth_and_spatial
from visualization.spatial_analysis_plot import render_spatial_analysis_png, render_spatial_analysis_from_result


def _make_analysis_result():
    rng = np.random.default_rng(0)
    depths, errors, pixels = [], [], []
    for depth, target_mean in [(5.0, 0.8), (15.0, 1.0), (25.0, 1.8), (40.0, 3.9)]:
        n = 40
        depths.extend([depth] * n)
        errors.extend(target_mean + rng.normal(0, 0.05, n))
        pixels.extend(rng.uniform([0, 0], [640, 480], size=(n, 2)))
    return analyze_depth_and_spatial(np.array(errors), np.array(depths), np.array(pixels),
                                      image_width=640, image_height=480)


def test_render_spatial_analysis_png_returns_valid_png():
    result = _make_analysis_result()
    png = render_spatial_analysis_png(result)
    assert png is not None
    assert png[:8] == b"\x89PNG\r\n\x1a\n"


def test_render_spatial_analysis_png_none_when_totally_empty():
    empty = analyze_depth_and_spatial(np.zeros(0), np.zeros(0), np.zeros((0, 2)), 640, 480)
    png = render_spatial_analysis_png(empty)
    assert png is None


def test_render_spatial_analysis_png_partial_data_still_renders():
    # only a couple of points, most bins empty
    errors = np.array([1.0, 2.0])
    depths = np.array([5.0, 5.0])
    pixels = np.array([[100.0, 100.0], [500.0, 400.0]])
    result = analyze_depth_and_spatial(errors, depths, pixels, 640, 480)
    png = render_spatial_analysis_png(result)
    assert png is not None


def test_render_spatial_analysis_from_result_convenience_wrapper():
    from tests.test_edge_alignment import _make_synthetic_scene
    from evaluation.edge_alignment import evaluate_edge_alignment

    camera, image, points_lidar, lidar_spec = _make_synthetic_scene()
    result = evaluate_edge_alignment(image=image, points_lidar=points_lidar, T_CL=np.eye(4),
                                       camera=camera, lidar_spec=lidar_spec, depth_jump_threshold_m=1.0)
    png = render_spatial_analysis_from_result(result, camera.width, camera.height)
    assert png is not None
    assert png[:8] == b"\x89PNG\r\n\x1a\n"


def test_render_spatial_analysis_from_result_none_on_fail():
    from tests.test_edge_alignment import _make_synthetic_scene
    from evaluation.edge_alignment import evaluate_edge_alignment

    camera, image, _, lidar_spec = _make_synthetic_scene()
    result = evaluate_edge_alignment(image=image, points_lidar=np.zeros((0, 3)), T_CL=np.eye(4),
                                       camera=camera, lidar_spec=lidar_spec)
    png = render_spatial_analysis_from_result(result, camera.width, camera.height)
    assert png is None


def test_render_spatial_analysis_png_returns_none_on_broken_plotting_env():
    """Consistent with the other visualization modules: a broken/partial
    matplotlib install must degrade to None, not crash."""
    import matplotlib.figure
    result = _make_analysis_result()

    original_add_subplot = matplotlib.figure.Figure.add_subplot

    def _broken_add_subplot(self, *args, **kwargs):
        raise RuntimeError("simulated broken plotting environment")

    matplotlib.figure.Figure.add_subplot = _broken_add_subplot
    try:
        png = render_spatial_analysis_png(result)
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
