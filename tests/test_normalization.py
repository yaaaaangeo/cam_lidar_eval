import sys
import os
import math

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from quality.normalization import (
    multiplier_score,
    score_m2,
    score_std,
    score_to_classification,
    normalize,
    GOOD_WARNING_BOUNDARY_SCORE,
    WARNING_BAD_BOUNDARY_SCORE,
)
from quality.noise_floor import (
    classify,
    M2_GOOD_MULTIPLIER,
    M2_WARNING_MULTIPLIER,
    STD_GOOD_MULTIPLIER,
    STD_WARNING_MULTIPLIER,
)


# ---------------------------------------------------------------------------
# Anchor point tests
# ---------------------------------------------------------------------------

def test_score_is_100_at_zero_error():
    s = multiplier_score(0.0, floor_px=1.0, good_mult=2.0, warning_mult=5.0)
    assert math.isclose(s, 100.0, abs_tol=1e-6)


def test_score_is_80_at_good_multiplier_boundary():
    floor = 1.0
    s = multiplier_score(M2_GOOD_MULTIPLIER * floor, floor, M2_GOOD_MULTIPLIER, M2_WARNING_MULTIPLIER)
    assert math.isclose(s, GOOD_WARNING_BOUNDARY_SCORE, abs_tol=1e-6)


def test_score_is_50_at_warning_multiplier_boundary():
    floor = 1.0
    s = multiplier_score(M2_WARNING_MULTIPLIER * floor, floor, M2_GOOD_MULTIPLIER, M2_WARNING_MULTIPLIER)
    assert math.isclose(s, WARNING_BAD_BOUNDARY_SCORE, abs_tol=1e-6)


def test_score_anchors_hold_for_std_multiplier_scheme_too():
    floor = 2.5
    s_good = multiplier_score(STD_GOOD_MULTIPLIER * floor, floor, STD_GOOD_MULTIPLIER, STD_WARNING_MULTIPLIER)
    s_warn = multiplier_score(STD_WARNING_MULTIPLIER * floor, floor, STD_GOOD_MULTIPLIER, STD_WARNING_MULTIPLIER)
    assert math.isclose(s_good, 80.0, abs_tol=1e-6)
    assert math.isclose(s_warn, 50.0, abs_tol=1e-6)


def test_score_anchors_invariant_to_floor_scale():
    # anchors should hold regardless of the absolute floor_px value, since
    # only the ratio r = value/floor matters
    for floor in [0.1, 1.0, 5.0, 30.5]:
        s = multiplier_score(M2_GOOD_MULTIPLIER * floor, floor, M2_GOOD_MULTIPLIER, M2_WARNING_MULTIPLIER)
        assert math.isclose(s, 80.0, abs_tol=1e-6), f"failed at floor={floor}"


# ---------------------------------------------------------------------------
# Monotonicity and bounds
# ---------------------------------------------------------------------------

def test_score_monotonically_decreasing_with_value():
    floor = 1.0
    values = np.linspace(0, 50, 200)
    scores = [multiplier_score(v, floor, M2_GOOD_MULTIPLIER, M2_WARNING_MULTIPLIER) for v in values]
    assert all(scores[i] >= scores[i + 1] for i in range(len(scores) - 1))


def test_score_bounded_0_to_100():
    floor = 1.0
    for v in [0.0, 1.0, 5.0, 50.0, 1000.0]:
        s = multiplier_score(v, floor, M2_GOOD_MULTIPLIER, M2_WARNING_MULTIPLIER)
        assert 0.0 <= s <= 100.0


def test_score_approaches_zero_for_extreme_values():
    s = multiplier_score(10000.0, floor_px=1.0, good_mult=2.0, warning_mult=5.0)
    assert s < 1.0


# ---------------------------------------------------------------------------
# NaN / invalid input handling
# ---------------------------------------------------------------------------

def test_score_nan_for_nan_value():
    s = multiplier_score(float("nan"), floor_px=1.0, good_mult=2.0, warning_mult=5.0)
    assert math.isnan(s)


def test_score_nan_for_invalid_floor():
    assert math.isnan(multiplier_score(1.0, floor_px=0.0, good_mult=2.0, warning_mult=5.0))
    assert math.isnan(multiplier_score(1.0, floor_px=-1.0, good_mult=2.0, warning_mult=5.0))
    assert math.isnan(multiplier_score(1.0, floor_px=float("nan"), good_mult=2.0, warning_mult=5.0))


def test_score_rejects_negative_value():
    try:
        multiplier_score(-1.0, floor_px=1.0, good_mult=2.0, warning_mult=5.0)
        assert False, "expected ValueError"
    except ValueError:
        pass


# ---------------------------------------------------------------------------
# score_to_classification round-trip / consistency with classify()
# ---------------------------------------------------------------------------

def test_score_to_classification_boundaries():
    assert score_to_classification(100.0) == "GOOD"
    assert score_to_classification(80.0) == "GOOD"
    assert score_to_classification(79.999) == "WARNING"
    assert score_to_classification(50.0) == "WARNING"
    assert score_to_classification(49.999) == "BAD"
    assert score_to_classification(0.0) == "BAD"
    assert score_to_classification(float("nan")) == "FAIL"


def test_score_classification_agrees_with_classify_across_random_ratios():
    """
    The core correctness property: for ANY value/floor ratio, the
    classification derived from the SCORE must agree with classify()
    (quality.noise_floor's direct multiplier-based classifier). This is
    what makes "score 85" and "classification GOOD" never contradict.
    """
    rng = np.random.RandomState(42)
    floor = 1.0
    for good_mult, warning_mult in [(M2_GOOD_MULTIPLIER, M2_WARNING_MULTIPLIER),
                                     (STD_GOOD_MULTIPLIER, STD_WARNING_MULTIPLIER)]:
        for r in rng.uniform(0, warning_mult * 4, size=500):
            value = r * floor
            direct_classification = classify(value, floor, good_mult, warning_mult)
            s = multiplier_score(value, floor, good_mult, warning_mult)
            score_classification = score_to_classification(s)
            assert direct_classification == score_classification, (
                f"mismatch at r={r}: direct={direct_classification}, "
                f"via score({s})={score_classification}"
            )


# ---------------------------------------------------------------------------
# Convenience wrappers
# ---------------------------------------------------------------------------

def test_score_m2_matches_multiplier_score_with_m2_constants():
    floor = 0.664
    value = 0.959
    assert math.isclose(
        score_m2(value, floor),
        multiplier_score(value, floor, M2_GOOD_MULTIPLIER, M2_WARNING_MULTIPLIER),
    )


def test_score_std_matches_multiplier_score_with_std_constants():
    floor = 0.664
    value = 1.35
    assert math.isclose(
        score_std(value, floor),
        multiplier_score(value, floor, STD_GOOD_MULTIPLIER, STD_WARNING_MULTIPLIER),
    )


def test_normalize_bundles_ratio_score_and_classification():
    result = normalize(1.0, 1.0, M2_GOOD_MULTIPLIER, M2_WARNING_MULTIPLIER)
    assert math.isclose(result.ratio, 1.0)
    assert result.classification == "GOOD"  # r=1 < good_mult=2
    assert result.score > 80


def test_normalize_handles_nan_gracefully():
    result = normalize(float("nan"), 1.0, M2_GOOD_MULTIPLIER, M2_WARNING_MULTIPLIER)
    assert math.isnan(result.score)
    assert result.classification == "FAIL"
    assert math.isnan(result.ratio)


# ---------------------------------------------------------------------------
# Sanity check against real measured values from earlier M2/M3/M4 demos
# ---------------------------------------------------------------------------

def test_score_against_measured_m2_correct_calibration():
    # from the M2 demo: mean=0.959px, floor=0.664px -> classification GOOD
    s = score_m2(0.959, 0.664)
    assert score_to_classification(s) == "GOOD"
    assert s > 80


def test_score_against_measured_m2_bad_calibration():
    # from the M2 demo: mean=74.240px, floor=0.775px -> classification BAD, severely so
    s = score_m2(74.240, 0.775)
    assert score_to_classification(s) == "BAD"
    assert s < 5  # should be very close to 0 given how extreme the ratio is


def test_score_against_measured_m3_uniform_case():
    # from the M3 demo: std=0.000px -> should score essentially perfect
    s = score_std(0.000, 0.664)
    assert math.isclose(s, 100.0, abs_tol=1e-3)


def test_score_against_measured_m3_drifting_case():
    # from the M3 demo: std=3.777px, floor=0.664px -> classification BAD
    s = score_std(3.777, 0.664)
    assert score_to_classification(s) == "BAD"


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
