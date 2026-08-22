import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.root_cause import diagnose_root_cause, RootCauseCandidate, RootCauseDiagnosisResult


# ---------------------------------------------------------------------------
# Lightweight test doubles -- root_cause.py only ever reads a handful of
# attributes off each input, so minimal stand-ins (rather than the full
# real dataclasses from every other module) keep these tests focused on
# THIS module's rule logic.
# ---------------------------------------------------------------------------

class _SyncStats:
    def __init__(self, classification, estimated_offset_ms=17.4, offset_std_ms=2.1, drop_ratio=0.012):
        self.classification = classification
        self.estimated_offset_ms = estimated_offset_ms
        self.offset_std_ms = offset_std_ms
        self.drop_ratio = drop_ratio


class _BinStats:
    def __init__(self, mean_px, valid_count=100, failure_count=0):
        self.mean_px = mean_px
        self.valid_count = valid_count
        self.failure_count = failure_count


class _SpatialAnalysis:
    def __init__(self, horizontal_regions=None, vertical_regions=None, depth_bins=None, depth_trend=None):
        self.horizontal_regions = horizontal_regions or {}
        self.vertical_regions = vertical_regions or {}
        self.depth_bins = depth_bins or {}
        self.depth_trend = depth_trend


class _DynamicFilterComparison:
    def __init__(self, dynamic_contamination_ratio, overall_classification, overall_mean_px,
                 static_only_classification, static_only_mean_px):
        self.dynamic_contamination_ratio = dynamic_contamination_ratio
        self.overall_classification = overall_classification
        self.overall_mean_px = overall_mean_px
        self.static_only_classification = static_only_classification
        self.static_only_mean_px = static_only_mean_px


class _AxisSensitivity:
    def __init__(self, axis, classification):
        self.axis = axis
        self.classification = classification


class _PerturbationResult:
    def __init__(self, axis_sensitivities):
        self.axis_sensitivities = axis_sensitivities


class _M3:
    def __init__(self, classification, instability_diagnosis=None):
        self.classification = classification
        self.instability_diagnosis = instability_diagnosis


# ---------------------------------------------------------------------------
# TEMPORAL_OFFSET -- spec's own example
# ---------------------------------------------------------------------------

def test_temporal_offset_fires_on_bad_sync():
    result = diagnose_root_cause(sync_stats=_SyncStats("BAD"))
    causes = [c.cause for c in result.candidates]
    assert "TEMPORAL_OFFSET" in causes
    candidate = next(c for c in result.candidates if c.cause == "TEMPORAL_OFFSET")
    assert candidate.confidence == "HIGH"


def test_temporal_offset_medium_on_warning_sync():
    result = diagnose_root_cause(sync_stats=_SyncStats("WARNING"))
    candidate = next(c for c in result.candidates if c.cause == "TEMPORAL_OFFSET")
    assert candidate.confidence == "MEDIUM"


def test_temporal_offset_silent_on_good_sync():
    result = diagnose_root_cause(sync_stats=_SyncStats("GOOD"))
    assert not any(c.cause == "TEMPORAL_OFFSET" for c in result.candidates)


def test_temporal_offset_silent_when_sync_stats_none():
    result = diagnose_root_cause(sync_stats=None)
    assert result.candidates == []


# ---------------------------------------------------------------------------
# YAW_MISALIGNMENT -- spec's own example ("right-side error >> left-side,
# yaw sensitivity HIGH")
# ---------------------------------------------------------------------------

def test_yaw_misalignment_fires_on_right_side_asymmetry_and_high_sensitivity():
    spatial = _SpatialAnalysis(horizontal_regions={
        "LEFT": _BinStats(1.0), "CENTER": _BinStats(1.2), "RIGHT": _BinStats(8.0),
    })
    perturbation = _PerturbationResult([_AxisSensitivity("yaw_deg", "HIGH")])
    result = diagnose_root_cause(spatial_analysis=spatial, perturbation_result=perturbation)
    candidate = next(c for c in result.candidates if c.cause == "YAW_MISALIGNMENT")
    assert candidate.confidence == "HIGH"
    assert candidate.label == "Yaw misalignment"


def test_yaw_misalignment_low_confidence_without_sensitivity_data():
    spatial = _SpatialAnalysis(horizontal_regions={
        "LEFT": _BinStats(1.0), "CENTER": _BinStats(1.2), "RIGHT": _BinStats(8.0),
    })
    result = diagnose_root_cause(spatial_analysis=spatial)
    candidate = next(c for c in result.candidates if c.cause == "YAW_MISALIGNMENT")
    assert candidate.confidence == "LOW"


def test_yaw_misalignment_silent_when_error_uniformly_elevated_not_asymmetric():
    # all three regions elevated roughly equally -- NOT a one-sided pattern
    spatial = _SpatialAnalysis(horizontal_regions={
        "LEFT": _BinStats(5.0), "CENTER": _BinStats(5.2), "RIGHT": _BinStats(5.1),
    })
    perturbation = _PerturbationResult([_AxisSensitivity("yaw_deg", "HIGH")])
    result = diagnose_root_cause(spatial_analysis=spatial, perturbation_result=perturbation)
    assert not any(c.cause == "YAW_MISALIGNMENT" for c in result.candidates)


def test_yaw_misalignment_silent_without_spatial_analysis():
    perturbation = _PerturbationResult([_AxisSensitivity("yaw_deg", "HIGH")])
    result = diagnose_root_cause(perturbation_result=perturbation)
    assert not any(c.cause == "YAW_MISALIGNMENT" for c in result.candidates)
    # falls through to the generic unexplained-sensitivity catch-all instead
    assert any(c.cause == "UNEXPLAINED_SENSITIVITY_YAW_DEG" for c in result.candidates)


# ---------------------------------------------------------------------------
# PITCH_MISALIGNMENT -- same pattern, vertical axis
# ---------------------------------------------------------------------------

def test_pitch_misalignment_fires_on_top_bottom_asymmetry():
    spatial = _SpatialAnalysis(vertical_regions={
        "TOP": _BinStats(7.5), "CENTER": _BinStats(1.0), "BOTTOM": _BinStats(1.1),
    })
    perturbation = _PerturbationResult([_AxisSensitivity("pitch_deg", "MEDIUM")])
    result = diagnose_root_cause(spatial_analysis=spatial, perturbation_result=perturbation)
    candidate = next(c for c in result.candidates if c.cause == "PITCH_MISALIGNMENT")
    assert candidate.confidence == "MEDIUM"


# ---------------------------------------------------------------------------
# TX_MISALIGNMENT -- parallax depth signature
# ---------------------------------------------------------------------------

def test_tx_misalignment_fires_on_decreasing_depth_trend_and_high_sensitivity():
    spatial = _SpatialAnalysis(
        depth_trend="decreases_with_depth",
        depth_bins={"0-10m": _BinStats(6.0), "30-50m": _BinStats(0.8)},
    )
    perturbation = _PerturbationResult([_AxisSensitivity("tx", "HIGH")])
    result = diagnose_root_cause(spatial_analysis=spatial, perturbation_result=perturbation)
    candidate = next(c for c in result.candidates if c.cause == "TX_MISALIGNMENT")
    assert candidate.confidence == "HIGH"
    assert any("parallax" in e.lower() for e in candidate.evidence)


def test_tx_misalignment_silent_when_depth_trend_increases():
    spatial = _SpatialAnalysis(depth_trend="increases_with_depth")
    perturbation = _PerturbationResult([_AxisSensitivity("tx", "HIGH")])
    result = diagnose_root_cause(spatial_analysis=spatial, perturbation_result=perturbation)
    assert not any(c.cause == "TX_MISALIGNMENT" for c in result.candidates)
    # falls through to the generic catch-all since tx wasn't "claimed"
    assert any(c.cause == "UNEXPLAINED_SENSITIVITY_TX" for c in result.candidates)


def test_tx_misalignment_silent_when_depth_trend_stable():
    spatial = _SpatialAnalysis(depth_trend="stable")
    result = diagnose_root_cause(spatial_analysis=spatial)
    assert not any(c.cause == "TX_MISALIGNMENT" for c in result.candidates)


# ---------------------------------------------------------------------------
# DYNAMIC_CONTAMINATION -- spec's own example
# ---------------------------------------------------------------------------

def test_dynamic_contamination_fires_matching_spec_example():
    # spec's literal numbers: M2 overall BAD-ish, static only GOOD, 38% contamination
    comparison = _DynamicFilterComparison(
        dynamic_contamination_ratio=0.38, overall_classification="BAD", overall_mean_px=3.1,
        static_only_classification="GOOD", static_only_mean_px=1.2,
    )
    result = diagnose_root_cause(dynamic_filter_comparison=comparison)
    candidate = next(c for c in result.candidates if c.cause == "DYNAMIC_CONTAMINATION")
    assert candidate.confidence == "HIGH"  # 38% > 30% threshold


def test_dynamic_contamination_medium_confidence_band():
    comparison = _DynamicFilterComparison(
        dynamic_contamination_ratio=0.20, overall_classification="WARNING", overall_mean_px=2.0,
        static_only_classification="GOOD", static_only_mean_px=1.0,
    )
    result = diagnose_root_cause(dynamic_filter_comparison=comparison)
    candidate = next(c for c in result.candidates if c.cause == "DYNAMIC_CONTAMINATION")
    assert candidate.confidence == "MEDIUM"


def test_dynamic_contamination_silent_when_static_also_bad():
    # if static-only is ALSO bad, contamination isn't the (whole) story
    comparison = _DynamicFilterComparison(
        dynamic_contamination_ratio=0.5, overall_classification="BAD", overall_mean_px=5.0,
        static_only_classification="BAD", static_only_mean_px=4.5,
    )
    result = diagnose_root_cause(dynamic_filter_comparison=comparison)
    assert not any(c.cause == "DYNAMIC_CONTAMINATION" for c in result.candidates)


def test_dynamic_contamination_silent_below_low_threshold():
    comparison = _DynamicFilterComparison(
        dynamic_contamination_ratio=0.02, overall_classification="GOOD", overall_mean_px=1.0,
        static_only_classification="GOOD", static_only_mean_px=0.98,
    )
    result = diagnose_root_cause(dynamic_filter_comparison=comparison)
    assert not any(c.cause == "DYNAMIC_CONTAMINATION" for c in result.candidates)


# ---------------------------------------------------------------------------
# SCENE_DEPENDENT_INSTABILITY -- reuses STEP10's diagnose_instability output
# ---------------------------------------------------------------------------

def test_scene_dependent_instability_fires_from_m3_diagnosis():
    m3 = _M3("BAD", instability_diagnosis={
        "worst_block_index": 3, "worst_block_mean_px": 5.2,
        "candidates": [{"metric": "representative_depth_m", "relative_diff": 2.0, "explanation": "Long-range scenes"}],
    })
    result = diagnose_root_cause(m3=m3)
    candidate = next(c for c in result.candidates if c.cause == "SCENE_DEPENDENT_INSTABILITY")
    assert candidate.label == "Long-range scenes"
    assert candidate.confidence == "HIGH"  # relative_diff >= 1.0


def test_scene_dependent_instability_medium_confidence_below_100pct_diff():
    m3 = _M3("WARNING", instability_diagnosis={
        "worst_block_index": 1, "worst_block_mean_px": 2.0,
        "candidates": [{"metric": "edge_density", "relative_diff": -0.6, "explanation": "Sparse edge structure"}],
    })
    result = diagnose_root_cause(m3=m3)
    candidate = next(c for c in result.candidates if c.cause == "SCENE_DEPENDENT_INSTABILITY")
    assert candidate.confidence == "MEDIUM"


def test_scene_dependent_instability_silent_when_m3_good():
    m3 = _M3("GOOD", instability_diagnosis={
        "worst_block_index": 0, "worst_block_mean_px": 1.0,
        "candidates": [{"metric": "representative_depth_m", "relative_diff": 2.0, "explanation": "Long-range scenes"}],
    })
    result = diagnose_root_cause(m3=m3)
    assert not any(c.cause == "SCENE_DEPENDENT_INSTABILITY" for c in result.candidates)


def test_scene_dependent_instability_silent_when_no_candidates():
    m3 = _M3("BAD", instability_diagnosis={"worst_block_index": 0, "worst_block_mean_px": 1.0, "candidates": []})
    result = diagnose_root_cause(m3=m3)
    assert not any(c.cause == "SCENE_DEPENDENT_INSTABILITY" for c in result.candidates)


# ---------------------------------------------------------------------------
# Unexplained sensitivity catch-all
# ---------------------------------------------------------------------------

def test_unexplained_sensitivity_reports_unclaimed_axes():
    perturbation = _PerturbationResult([
        _AxisSensitivity("roll_deg", "HIGH"),
        _AxisSensitivity("tz", "MEDIUM"),
        _AxisSensitivity("ty", "LOW"),  # LOW is not reported at all
    ])
    result = diagnose_root_cause(perturbation_result=perturbation)
    causes = {c.cause: c for c in result.candidates}
    assert "UNEXPLAINED_SENSITIVITY_ROLL_DEG" in causes
    assert "UNEXPLAINED_SENSITIVITY_TZ" in causes
    assert "UNEXPLAINED_SENSITIVITY_TY" not in causes
    assert causes["UNEXPLAINED_SENSITIVITY_ROLL_DEG"].confidence == "LOW"


def test_unexplained_sensitivity_does_not_duplicate_claimed_axes():
    """If yaw is already explained by YAW_MISALIGNMENT, it shouldn't ALSO
    show up as a separate unexplained-sensitivity entry."""
    spatial = _SpatialAnalysis(horizontal_regions={
        "LEFT": _BinStats(1.0), "CENTER": _BinStats(1.2), "RIGHT": _BinStats(8.0),
    })
    perturbation = _PerturbationResult([_AxisSensitivity("yaw_deg", "HIGH")])
    result = diagnose_root_cause(spatial_analysis=spatial, perturbation_result=perturbation)
    assert not any(c.cause == "UNEXPLAINED_SENSITIVITY_YAW_DEG" for c in result.candidates)
    assert any(c.cause == "YAW_MISALIGNMENT" for c in result.candidates)


# ---------------------------------------------------------------------------
# Ranking / overall result shape
# ---------------------------------------------------------------------------

def test_ranking_sorts_high_before_medium_before_low():
    result = diagnose_root_cause(
        sync_stats=_SyncStats("WARNING"),  # MEDIUM
        dynamic_filter_comparison=_DynamicFilterComparison(0.38, "BAD", 3.1, "GOOD", 1.2),  # HIGH
        perturbation_result=_PerturbationResult([_AxisSensitivity("roll_deg", "MEDIUM")]),  # LOW
    )
    confidences = [c.confidence for c in result.candidates]
    order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
    assert confidences == sorted(confidences, key=lambda c: order[c])


def test_empty_result_when_no_inputs_given():
    result = diagnose_root_cause()
    assert result.candidates == []


def test_summary_lines_matches_spec_output_format():
    result = diagnose_root_cause(
        dynamic_filter_comparison=_DynamicFilterComparison(0.38, "BAD", 3.1, "GOOD", 1.2),
    )
    lines = result.summary_lines()
    assert len(lines) == 1
    assert lines[0].startswith("1. Dynamic object contamination")
    assert "HIGH" in lines[0]


def test_to_dict_shape():
    result = diagnose_root_cause(sync_stats=_SyncStats("BAD"))
    d = result.to_dict()
    assert "candidates" in d
    assert d["candidates"][0]["cause"] == "TEMPORAL_OFFSET"
    assert "evidence" in d["candidates"][0]


def test_spec_worked_example_end_to_end_ranking():
    """Reproduces the spec's own final example output structure:

        1. Yaw misalignment       HIGH
        2. Tx misalignment        MEDIUM
        3. Dynamic contamination  LOW
        4. Timestamp offset       LOW

    (using this test's own evidence combination, not literally the same
    numbers -- the point is that multiple simultaneous causes rank
    correctly relative to each other by confidence)."""
    spatial = _SpatialAnalysis(
        horizontal_regions={"LEFT": _BinStats(1.0), "CENTER": _BinStats(1.1), "RIGHT": _BinStats(9.0)},
        depth_trend="decreases_with_depth",
        depth_bins={"0-10m": _BinStats(3.0), "30-50m": _BinStats(1.0)},
    )
    perturbation = _PerturbationResult([
        _AxisSensitivity("yaw_deg", "HIGH"),
        _AxisSensitivity("tx", "MEDIUM"),
    ])
    dynamic = _DynamicFilterComparison(0.10, "WARNING", 2.0, "GOOD", 1.5)  # LOW band
    sync = _SyncStats("WARNING")  # would be MEDIUM on its own, but let's check ranking holds generally

    result = diagnose_root_cause(
        sync_stats=sync, spatial_analysis=spatial,
        dynamic_filter_comparison=dynamic, perturbation_result=perturbation,
    )
    causes_in_order = [c.cause for c in result.candidates]
    assert causes_in_order.index("YAW_MISALIGNMENT") < causes_in_order.index("TX_MISALIGNMENT")
    assert causes_in_order.index("TX_MISALIGNMENT") < causes_in_order.index("DYNAMIC_CONTAMINATION")


# ---------------------------------------------------------------------------
# STEP14 -- confirmations (the 🟢 "OK" half of the spec's own diagnosis
# panel example, mixed alongside 🔴🟠 problems in the same list)
# ---------------------------------------------------------------------------

def test_confirmation_fires_for_good_sync():
    result = diagnose_root_cause(sync_stats=_SyncStats("GOOD"))
    labels = [c.label for c in result.confirmations]
    assert "Timestamp sync" in labels


def test_no_confirmation_for_bad_sync():
    result = diagnose_root_cause(sync_stats=_SyncStats("BAD"))
    labels = [c.label for c in result.confirmations]
    assert "Timestamp sync" not in labels
    # and it shows up as a PROBLEM instead
    assert any(c.cause == "TEMPORAL_OFFSET" for c in result.candidates)


def test_confirmation_fires_for_negligible_dynamic_contamination():
    comparison = _DynamicFilterComparison(
        dynamic_contamination_ratio=0.01, overall_classification="GOOD", overall_mean_px=1.0,
        static_only_classification="GOOD", static_only_mean_px=0.99,
    )
    result = diagnose_root_cause(dynamic_filter_comparison=comparison)
    labels = [c.label for c in result.confirmations]
    assert "Dynamic object contamination" in labels


def test_confirmation_fires_for_good_m3():
    m3 = _M3("GOOD", instability_diagnosis=None)
    result = diagnose_root_cause(m3=m3)
    labels = [c.label for c in result.confirmations]
    assert "Block-to-block consistency (M3)" in labels


def test_confirmation_fires_when_all_sensitivities_low():
    perturbation = _PerturbationResult([
        _AxisSensitivity("roll_deg", "LOW"),
        _AxisSensitivity("tz", "LOW"),
    ])
    result = diagnose_root_cause(perturbation_result=perturbation)
    labels = [c.label for c in result.confirmations]
    assert "Calibration parameter sensitivity" in labels
    assert result.candidates == []  # nothing flagged as a problem either


def test_no_sensitivity_confirmation_when_any_axis_elevated():
    perturbation = _PerturbationResult([
        _AxisSensitivity("roll_deg", "LOW"),
        _AxisSensitivity("tz", "MEDIUM"),  # one non-LOW axis breaks the "all clear" confirmation
    ])
    result = diagnose_root_cause(perturbation_result=perturbation)
    labels = [c.label for c in result.confirmations]
    assert "Calibration parameter sensitivity" not in labels


def test_spec_mixed_diagnosis_panel_example():
    """Directly reproduces the spec's own mixed-panel example:

        🔴 Yaw misalignment
        🟠 Tx misalignment
        🟢 Timestamp OK
        🟢 Sensor quality OK

    -- problems (candidates) and confirmations coexisting in one result,
    not just a bare list of problems."""
    spatial = _SpatialAnalysis(
        horizontal_regions={"LEFT": _BinStats(1.0), "CENTER": _BinStats(1.1), "RIGHT": _BinStats(9.0)},
        depth_trend="decreases_with_depth",
        depth_bins={"0-10m": _BinStats(3.0), "30-50m": _BinStats(1.0)},
    )
    perturbation = _PerturbationResult([
        _AxisSensitivity("yaw_deg", "HIGH"),
        _AxisSensitivity("tx", "MEDIUM"),
    ])
    sync = _SyncStats("GOOD")  # 🟢 Timestamp OK
    dynamic = _DynamicFilterComparison(0.01, "GOOD", 1.0, "GOOD", 0.99)  # 🟢 Sensor quality OK proxy

    result = diagnose_root_cause(
        sync_stats=sync, spatial_analysis=spatial,
        dynamic_filter_comparison=dynamic, perturbation_result=perturbation,
    )
    causes = [c.cause for c in result.candidates]
    confirmation_labels = [c.label for c in result.confirmations]
    assert "YAW_MISALIGNMENT" in causes
    assert "TX_MISALIGNMENT" in causes
    assert "Timestamp sync" in confirmation_labels
    assert "Dynamic object contamination" in confirmation_labels


def test_to_dict_includes_confirmations():
    result = diagnose_root_cause(sync_stats=_SyncStats("GOOD"))
    d = result.to_dict()
    assert "confirmations" in d
    assert d["confirmations"][0]["label"] == "Timestamp sync"
    assert "detail" in d["confirmations"][0]


def test_empty_confirmations_when_no_inputs():
    result = diagnose_root_cause()
    assert result.confirmations == []


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
