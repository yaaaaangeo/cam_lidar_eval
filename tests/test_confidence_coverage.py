import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from quality.confidence_coverage import (
    compute_confidence,
    compute_coverage,
    compute_quality_confidence_coverage,
    ComponentScore,
    ConfidenceCoverageAxis,
)


# ---------------------------------------------------------------------------
# Lightweight test doubles (same philosophy as tests/test_root_cause.py:
# this module only reads a handful of attributes off each input).
# ---------------------------------------------------------------------------

class _SyncStats:
    def __init__(self, classification, estimated_offset_ms=5.0, drop_ratio=0.01):
        self.classification = classification
        self.estimated_offset_ms = estimated_offset_ms
        self.drop_ratio = drop_ratio


class _M2:
    def __init__(self, match_rate):
        self.match_rate = match_rate


class _M3:
    def __init__(self, num_valid_blocks, block_results=None):
        self.num_valid_blocks = num_valid_blocks
        self.block_results = block_results or []


class _BlockResult:
    def __init__(self, classification, fov_coverage):
        self.classification = classification
        self.fov_coverage = fov_coverage


class _M4:
    def __init__(self, valid_ratio):
        self.valid_ratio = valid_ratio


class _BinStats:
    def __init__(self, valid_count=10, failure_count=0):
        self.valid_count = valid_count
        self.failure_count = failure_count


class _SpatialAnalysis:
    def __init__(self, depth_bins=None, horizontal_regions=None, vertical_regions=None):
        self.depth_bins = depth_bins or {}
        self.horizontal_regions = horizontal_regions or {}
        self.vertical_regions = vertical_regions or {}


class _QualityResult:
    def __init__(self, overall_score, overall_classification):
        self.overall_score = overall_score
        self.overall_classification = overall_classification


# ---------------------------------------------------------------------------
# compute_confidence
# ---------------------------------------------------------------------------

def test_confidence_high_with_all_good_signals():
    result = compute_confidence(
        sync_stats=_SyncStats("GOOD"),
        m2=_M2(match_rate=0.98),
        m3=_M3(num_valid_blocks=4), n_blocks=4,
        m4=_M4(valid_ratio=1.0),
        input_validation={"status": "INPUT_VALID"},
    )
    assert result.score > 90
    assert result.classification == "GOOD"
    assert len(result.components) == 5


def test_confidence_low_with_all_poor_signals():
    result = compute_confidence(
        sync_stats=_SyncStats("BAD"),
        m2=_M2(match_rate=0.3),
        m3=_M3(num_valid_blocks=1), n_blocks=4,
        m4=_M4(valid_ratio=0.4),
        input_validation={"status": "INPUT_WARNING"},
    )
    assert result.score < 50
    assert result.classification == "BAD"


def test_confidence_empty_when_no_inputs():
    result = compute_confidence()
    assert result.score != result.score  # NaN
    assert result.classification == "FAIL"
    assert result.components == []


def test_confidence_averages_only_available_components():
    # only sync given -- GOOD sync alone should give a perfect score,
    # not be dragged down by "missing" components
    result = compute_confidence(sync_stats=_SyncStats("GOOD"))
    assert len(result.components) == 1
    assert result.score == 100.0


def test_confidence_input_validation_warning_reduces_score():
    valid = compute_confidence(input_validation={"status": "INPUT_VALID"})
    warning = compute_confidence(input_validation={"status": "INPUT_WARNING"})
    assert warning.score < valid.score


def test_confidence_m2_none_match_rate_skipped():
    """Old (non-STEP6) matching mode -- match_rate is None, not 0 -- must
    be skipped, not treated as zero confidence."""
    result = compute_confidence(m2=_M2(match_rate=None), sync_stats=_SyncStats("GOOD"))
    assert len(result.components) == 1  # only sync, m2 skipped
    assert result.score == 100.0


# ---------------------------------------------------------------------------
# compute_coverage
# ---------------------------------------------------------------------------

def test_coverage_high_when_all_bins_and_regions_populated():
    spatial = _SpatialAnalysis(
        depth_bins={"0-10m": _BinStats(10), "10-20m": _BinStats(10), "20-30m": _BinStats(10),
                    "30-50m": _BinStats(10), "50m+": _BinStats(10)},
        horizontal_regions={"LEFT": _BinStats(10), "CENTER": _BinStats(10), "RIGHT": _BinStats(10)},
        vertical_regions={"TOP": _BinStats(10), "CENTER": _BinStats(10), "BOTTOM": _BinStats(10)},
    )
    m3 = _M3(num_valid_blocks=4, block_results=[_BlockResult("GOOD", 0.9)] * 4)
    result = compute_coverage(spatial_analysis=spatial, m3=m3)
    assert result.score > 85
    assert result.classification == "GOOD"


def test_coverage_low_when_only_one_depth_bin_populated():
    spatial = _SpatialAnalysis(
        depth_bins={"0-10m": _BinStats(10), "10-20m": _BinStats(0), "20-30m": _BinStats(0),
                    "30-50m": _BinStats(0), "50m+": _BinStats(0)},
    )
    result = compute_coverage(spatial_analysis=spatial)
    assert result.score == 20.0  # 1/5 bins


def test_coverage_empty_when_no_inputs():
    result = compute_coverage()
    assert result.score != result.score  # NaN
    assert result.classification == "FAIL"


def test_coverage_fov_averages_only_valid_blocks():
    m3 = _M3(num_valid_blocks=2, block_results=[
        _BlockResult("GOOD", 0.8), _BlockResult("GOOD", 0.6), _BlockResult("EXCLUDED", 0.0),
    ])
    result = compute_coverage(m3=m3)
    fov_component = next(c for c in result.components if c.name == "fov_coverage")
    assert abs(fov_component.value - 0.7) < 1e-9  # mean(0.8, 0.6), EXCLUDED block skipped


def test_coverage_depth_bin_failure_count_still_counts_as_populated():
    # a bin with only FAILED points still had SOMETHING projected there --
    # "no data at all" vs "data but it was bad" are different signals;
    # coverage cares about the former.
    spatial = _SpatialAnalysis(depth_bins={
        "0-10m": _BinStats(valid_count=0, failure_count=5), "10-20m": _BinStats(0, 0),
    })
    result = compute_coverage(spatial_analysis=spatial)
    assert result.score == 50.0  # 1/2 bins populated


# ---------------------------------------------------------------------------
# compute_quality_confidence_coverage -- spec's own two contrasting examples
# ---------------------------------------------------------------------------

def test_spec_example_high_confidence_high_coverage():
    """Reproduces the spec's first worked example:
        Quality: 82/100  Confidence: 94/100  Coverage: 97/100
    """
    quality = _QualityResult(82.0, "WARNING")
    sync = _SyncStats("GOOD")
    m2 = _M2(match_rate=0.97)
    m3 = _M3(num_valid_blocks=4, block_results=[_BlockResult("GOOD", 0.9)] * 4)
    m4 = _M4(valid_ratio=0.98)
    spatial = _SpatialAnalysis(
        depth_bins={k: _BinStats(20) for k in ("0-10m", "10-20m", "20-30m", "30-50m", "50m+")},
        horizontal_regions={k: _BinStats(20) for k in ("LEFT", "CENTER", "RIGHT")},
        vertical_regions={k: _BinStats(20) for k in ("TOP", "CENTER", "BOTTOM")},
    )
    result = compute_quality_confidence_coverage(
        quality, sync_stats=sync, m2=m2, m3=m3, m4=m4, spatial_analysis=spatial,
        input_validation={"status": "INPUT_VALID"}, n_blocks=4,
    )
    assert result.quality_score == 82.0
    assert result.confidence.score > 85
    assert result.coverage.score > 85


def test_spec_example_low_confidence_low_coverage_same_quality():
    """Reproduces the spec's second worked example -- SAME Quality score
    (82) but LOW Confidence/Coverage, meaning a completely different
    situation despite the identical Quality number:
        Quality: 82/100  Confidence: 42/100  Coverage: 51/100
    """
    quality = _QualityResult(82.0, "WARNING")  # deliberately identical Quality to the test above
    sync = _SyncStats("BAD")
    m2 = _M2(match_rate=0.35)
    m3 = _M3(num_valid_blocks=1, block_results=[_BlockResult("GOOD", 0.3)])
    m4 = _M4(valid_ratio=0.5)
    spatial = _SpatialAnalysis(
        depth_bins={"0-10m": _BinStats(5), "10-20m": _BinStats(0), "20-30m": _BinStats(0),
                    "30-50m": _BinStats(0), "50m+": _BinStats(0)},
        horizontal_regions={"LEFT": _BinStats(5), "CENTER": _BinStats(0), "RIGHT": _BinStats(0)},
        vertical_regions={"TOP": _BinStats(5), "CENTER": _BinStats(0), "BOTTOM": _BinStats(0)},
    )
    result = compute_quality_confidence_coverage(
        quality, sync_stats=sync, m2=m2, m3=m3, m4=m4, spatial_analysis=spatial,
        input_validation={"status": "INPUT_WARNING"}, n_blocks=4,
    )
    assert result.quality_score == 82.0  # SAME quality as the high-confidence case
    assert result.confidence.score < 50
    assert result.coverage.score < 60
    # the whole point of STEP13: same Quality, very different trustworthiness
    assert result.confidence.classification in ("WARNING", "BAD")
    assert result.coverage.classification in ("WARNING", "BAD")


def test_to_dict_shape():
    quality = _QualityResult(75.0, "WARNING")
    result = compute_quality_confidence_coverage(quality, sync_stats=_SyncStats("GOOD"))
    d = result.to_dict()
    assert set(d.keys()) == {"quality", "confidence", "coverage"}
    assert d["quality"]["score"] == 75.0
    assert "components" in d["confidence"]


def test_summary_line_format():
    quality = _QualityResult(82.0, "WARNING")
    result = compute_quality_confidence_coverage(quality, sync_stats=_SyncStats("GOOD"))
    line = result.summary_line()
    assert "Quality: 82/100" in line
    assert "Confidence:" in line
    assert "Coverage:" in line


def test_nan_quality_score_handled_gracefully():
    quality = _QualityResult(float("nan"), "FAIL")
    result = compute_quality_confidence_coverage(quality)
    d = result.to_dict()
    assert d["quality"]["score"] is None
    line = result.summary_line()
    assert "Quality: N/A" in line


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
