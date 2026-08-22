import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from quality.noise_floor import LidarSensorSpecForFloor, resolve_floor_inputs
from visualization.uncertainty_plot import render_uncertainty_plot_png, render_uncertainty_plot_from_result


def _make_floor_inputs():
    spec = LidarSensorSpecForFloor(horizontal_resolution_deg=0.2, range_accuracy_m=0.02)
    T_CL = np.eye(4)
    T_CL[:3, 3] = [0.1, 0.0, 0.05]
    return resolve_floor_inputs(fx_px=500.0, T_CL=T_CL, lidar_spec=spec)


def test_render_uncertainty_plot_png_returns_valid_png():
    rng = np.random.default_rng(0)
    depths = rng.uniform(2.0, 50.0, size=200)
    floor_inputs = _make_floor_inputs()
    from quality.noise_floor import compute_floor_array
    errors = compute_floor_array(floor_inputs, depths) * rng.uniform(0.2, 1.5, size=200)
    png = render_uncertainty_plot_png(depths, errors, floor_inputs)
    assert png is not None
    assert png[:8] == b"\x89PNG\r\n\x1a\n"


def test_render_uncertainty_plot_png_empty_returns_none():
    floor_inputs = _make_floor_inputs()
    png = render_uncertainty_plot_png(np.zeros(0), np.zeros(0), floor_inputs)
    assert png is None


def test_render_uncertainty_plot_png_filters_nonfinite_and_nonpositive_depths():
    depths = np.array([5.0, np.nan, -1.0, 0.0, 10.0])
    errors = np.array([1.0, 2.0, 3.0, 4.0, np.inf])
    floor_inputs = _make_floor_inputs()
    png = render_uncertainty_plot_png(depths, errors, floor_inputs)
    # only depth=5.0 (error=1.0) survives filtering -- still enough to render
    assert png is not None


def test_render_uncertainty_plot_png_single_depth_value_still_renders():
    depths = np.full(20, 8.0)
    errors = np.random.default_rng(1).uniform(0.1, 3.0, size=20)
    floor_inputs = _make_floor_inputs()
    png = render_uncertainty_plot_png(depths, errors, floor_inputs)
    assert png is not None


def test_render_uncertainty_plot_from_result_convenience_wrapper():
    import sys as _sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from tests.test_edge_alignment import _make_synthetic_scene
    from evaluation.edge_alignment import evaluate_edge_alignment
    from quality.noise_floor import resolve_floor_inputs

    camera, image, points_lidar, lidar_spec = _make_synthetic_scene()
    result = evaluate_edge_alignment(image=image, points_lidar=points_lidar, T_CL=np.eye(4),
                                       camera=camera, lidar_spec=lidar_spec, depth_jump_threshold_m=1.0)
    floor_inputs = resolve_floor_inputs(fx_px=camera.intrinsics.fx, T_CL=np.eye(4), lidar_spec=lidar_spec,
                                         edge_localization_floor_px=camera.edge_localization_floor_px)
    png = render_uncertainty_plot_from_result(result, floor_inputs)
    assert png is not None
    assert png[:8] == b"\x89PNG\r\n\x1a\n"


def test_render_uncertainty_plot_from_result_none_on_fail():
    import sys as _sys
    from tests.test_edge_alignment import _make_synthetic_scene
    from evaluation.edge_alignment import evaluate_edge_alignment
    from quality.noise_floor import resolve_floor_inputs

    camera, image, _, lidar_spec = _make_synthetic_scene()
    result = evaluate_edge_alignment(image=image, points_lidar=np.zeros((0, 3)), T_CL=np.eye(4),
                                       camera=camera, lidar_spec=lidar_spec)
    assert result.classification == "FAIL"
    floor_inputs = resolve_floor_inputs(fx_px=camera.intrinsics.fx, T_CL=np.eye(4), lidar_spec=lidar_spec,
                                         edge_localization_floor_px=camera.edge_localization_floor_px)
    png = render_uncertainty_plot_from_result(result, floor_inputs)
    assert png is None


def test_render_uncertainty_plot_png_returns_none_on_broken_plotting_env():
    """Consistent with the other visualization modules: a broken/partial
    matplotlib install must degrade to None, not crash."""
    import matplotlib.figure
    depths = np.array([5.0, 10.0, 15.0])
    errors = np.array([1.0, 2.0, 3.0])
    floor_inputs = _make_floor_inputs()

    original_add_subplot = matplotlib.figure.Figure.add_subplot

    def _broken_add_subplot(self, *args, **kwargs):
        raise RuntimeError("simulated broken plotting environment")

    matplotlib.figure.Figure.add_subplot = _broken_add_subplot
    try:
        png = render_uncertainty_plot_png(depths, errors, floor_inputs)
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
