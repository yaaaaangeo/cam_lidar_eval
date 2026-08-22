import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.perturbation import AxisSensitivity
from visualization.sensitivity_plot import render_sensitivity_png, render_sensitivity_from_result


def _make_sensitivities():
    return [
        AxisSensitivity(axis="yaw_deg", classification="HIGH", small_delta_effect_px=2.0, large_delta_effect_px=8.0),
        AxisSensitivity(axis="tx", classification="HIGH", small_delta_effect_px=1.5, large_delta_effect_px=6.0),
        AxisSensitivity(axis="pitch_deg", classification="MEDIUM", small_delta_effect_px=0.2, large_delta_effect_px=2.0),
        AxisSensitivity(axis="ty", classification="LOW", small_delta_effect_px=0.05, large_delta_effect_px=0.3),
        AxisSensitivity(axis="roll_deg", classification="LOW", small_delta_effect_px=0.02, large_delta_effect_px=0.1),
        AxisSensitivity(axis="tz", classification="LOW", small_delta_effect_px=0.01, large_delta_effect_px=0.05),
    ]


def test_render_sensitivity_png_returns_valid_png():
    png = render_sensitivity_png(_make_sensitivities())
    assert png is not None
    assert png[:8] == b"\x89PNG\r\n\x1a\n"


def test_render_sensitivity_png_empty_returns_none():
    png = render_sensitivity_png([])
    assert png is None


def test_render_sensitivity_png_single_axis_still_renders():
    png = render_sensitivity_png([_make_sensitivities()[0]])
    assert png is not None


def test_render_sensitivity_png_accepts_dict_items():
    dict_items = [
        {"axis": "yaw_deg", "classification": "HIGH", "small_delta_effect_px": 2.0, "large_delta_effect_px": 8.0},
        {"axis": "tz", "classification": "LOW", "small_delta_effect_px": 0.01, "large_delta_effect_px": 0.05},
    ]
    png = render_sensitivity_png(dict_items)
    assert png is not None


def test_render_sensitivity_from_result_convenience_wrapper():
    class _FakeResult:
        axis_sensitivities = _make_sensitivities()
    png = render_sensitivity_from_result(_FakeResult())
    assert png is not None


def test_render_sensitivity_png_returns_none_on_broken_plotting_env():
    """Consistent with the other visualization modules: a broken/partial
    matplotlib install must degrade to None, not crash."""
    import matplotlib.figure
    original_add_subplot = matplotlib.figure.Figure.add_subplot

    def _broken_add_subplot(self, *args, **kwargs):
        raise RuntimeError("simulated broken plotting environment")

    matplotlib.figure.Figure.add_subplot = _broken_add_subplot
    try:
        png = render_sensitivity_png(_make_sensitivities())
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
