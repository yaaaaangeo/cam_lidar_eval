import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from evaluation.temporal_drift import evaluate_temporal_drift
from evaluation.multiframe_consistency import FrameResult, MultiFrameConsistencyResult


def _make_fake_m4(mean_px_sequence, floor_px=1.0):
    """Build a MultiFrameConsistencyResult directly from a chosen sequence
    of per-frame mean_px values, bypassing the full M2 pipeline -- this
    metric only reads m4_result.frame_results and .floor_px, so we can
    test its regression logic in isolation with exact, controlled inputs."""
    frame_results = [
        FrameResult(frame_index=i, timestamp=float(i), mean_px=v, num_edge_points=100,
                    representative_depth_m=5.0, floor_px=floor_px, classification="GOOD")
        for i, v in enumerate(mean_px_sequence)
    ]
    return MultiFrameConsistencyResult(
        frame_results=frame_results, num_frames_total=len(frame_results),
        num_valid_frames=len(frame_results), num_failed_frames=0, num_outlier_frames=0,
        outlier_frame_indices=[], mean_across_frames_px=float(np.mean(mean_px_sequence)),
        std_across_frames_px=float(np.std(mean_px_sequence, ddof=1)) if len(mean_px_sequence) > 1 else 0.0,
        median_across_frames_px=float(np.median(mean_px_sequence)), p95_across_frames_px=float(np.percentile(mean_px_sequence, 95)),
        max_across_frames_px=float(np.max(mean_px_sequence)), floor_px=floor_px, classification="GOOD",
    )


def test_temporal_drift_no_trend_is_good():
    rng = np.random.RandomState(0)
    sequence = 1.0 + rng.normal(0, 0.05, 50)  # flat, noisy around 1.0px
    m4 = _make_fake_m4(sequence, floor_px=1.0)
    result = evaluate_temporal_drift(m4)
    # The classification is what matters here, not whether p happened to dip
    # below alpha by chance on this particular noise draw: even when a tiny
    # spurious slope is "significant", its magnitude is far below the floor
    # multiplier threshold, so it must still classify as GOOD.
    assert result.classification == "GOOD"
    assert result.total_drift_px < result.floor_px * 1.0  # well under the GOOD multiplier boundary


def test_temporal_drift_strong_upward_trend_is_bad():
    x = np.arange(50)
    sequence = 0.5 + 0.5 * x  # strong, clean linear increase, no noise
    m4 = _make_fake_m4(sequence, floor_px=1.0)
    result = evaluate_temporal_drift(m4)
    assert result.is_statistically_significant
    assert result.slope_px_per_frame > 0
    assert result.classification == "BAD"


def test_temporal_drift_strong_downward_trend_detected():
    x = np.arange(50)
    sequence = 30.0 - 0.5 * x
    m4 = _make_fake_m4(sequence, floor_px=1.0)
    result = evaluate_temporal_drift(m4)
    assert result.is_statistically_significant
    assert result.slope_px_per_frame < 0
    assert result.total_drift_px > 0  # magnitude, always positive


def test_temporal_drift_mild_significant_trend_is_warning():
    x = np.arange(80)
    # slope chosen so total predicted drift (~slope * 79) lands inside the
    # WARNING band (1x-3x floor_px=1.0) rather than GOOD or BAD, while
    # staying statistically significant against low noise
    rng = np.random.RandomState(1)
    sequence = 1.0 + 0.02 * x + rng.normal(0, 0.02, 80)
    m4 = _make_fake_m4(sequence, floor_px=1.0)
    result = evaluate_temporal_drift(m4)
    assert result.is_statistically_significant
    assert result.classification == "WARNING", (
        f"total_drift_px={result.total_drift_px}, floor_px={result.floor_px}"
    )


def test_temporal_drift_fails_below_min_frames():
    m4 = _make_fake_m4([1.0, 2.0, 3.0], floor_px=1.0)  # only 3, default min_frames=5
    result = evaluate_temporal_drift(m4)
    assert result.classification == "FAIL"
    assert any("Only 3 valid frame" in w for w in result.warnings)


def test_temporal_drift_respects_custom_min_frames():
    m4 = _make_fake_m4([1.0, 2.0, 3.0], floor_px=1.0)
    result = evaluate_temporal_drift(m4, min_frames=3)
    assert result.classification != "FAIL"


def test_temporal_drift_excludes_failed_frames():
    frame_results = [
        FrameResult(0, 0.0, 1.0, 100, 5.0, 1.0, "GOOD"),
        FrameResult(1, 1.0, float("nan"), 0, float("nan"), float("nan"), "FAIL"),
        FrameResult(2, 2.0, 1.1, 100, 5.0, 1.0, "GOOD"),
        FrameResult(3, 3.0, 1.05, 100, 5.0, 1.0, "GOOD"),
        FrameResult(4, 4.0, 1.02, 100, 5.0, 1.0, "GOOD"),
        FrameResult(5, 5.0, 0.98, 100, 5.0, 1.0, "GOOD"),
    ]
    m4 = MultiFrameConsistencyResult(
        frame_results=frame_results, num_frames_total=6, num_valid_frames=5,
        num_failed_frames=1, num_outlier_frames=0, outlier_frame_indices=[],
        mean_across_frames_px=1.03, std_across_frames_px=0.05, median_across_frames_px=1.02,
        p95_across_frames_px=1.1, max_across_frames_px=1.1, floor_px=1.0, classification="GOOD",
    )
    result = evaluate_temporal_drift(m4, min_frames=5)
    assert result.num_frames_used == 5
    assert result.classification != "FAIL"


def test_temporal_drift_all_same_frame_index_fails():
    frame_results = [FrameResult(0, 0.0, 1.0 + i * 0.01, 100, 5.0, 1.0, "GOOD") for i in range(10)]
    for f in frame_results:
        f.frame_index = 0  # degenerate: all same x value
    m4 = MultiFrameConsistencyResult(
        frame_results=frame_results, num_frames_total=10, num_valid_frames=10,
        num_failed_frames=0, num_outlier_frames=0, outlier_frame_indices=[],
        mean_across_frames_px=1.0, std_across_frames_px=0.01, median_across_frames_px=1.0,
        p95_across_frames_px=1.05, max_across_frames_px=1.09, floor_px=1.0, classification="GOOD",
    )
    result = evaluate_temporal_drift(m4, min_frames=5)
    assert result.classification == "FAIL"


def test_temporal_drift_scales_with_floor():
    """The same absolute slope should classify differently depending on
    the sensor-relative floor (bigger floor = more tolerance)."""
    x = np.arange(50)
    sequence = 1.0 + 0.05 * x  # same trend for both

    m4_tight_floor = _make_fake_m4(sequence, floor_px=0.1)
    m4_loose_floor = _make_fake_m4(sequence, floor_px=100.0)

    result_tight = evaluate_temporal_drift(m4_tight_floor)
    result_loose = evaluate_temporal_drift(m4_loose_floor)

    assert result_tight.classification == "BAD"
    assert result_loose.classification == "GOOD"


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
