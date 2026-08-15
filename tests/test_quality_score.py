import sys
import os
import math

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from quality.quality_score import compute_quality_score, DEFAULT_WEIGHTS
from evaluation.edge_alignment import evaluate_edge_alignment
from evaluation.holdout_consistency import evaluate_holdout_consistency
from evaluation.multiframe_consistency import evaluate_multiframe_consistency

from tests.test_holdout_consistency import (
    _make_dataset, _make_lidar_spec, _make_camera, _make_base_points_cam_frame,
)


def _run_all_metrics(x_drifts_per_frame, n_blocks=4, min_frames_block=5, min_frames_m4=30, **edge_kwargs):
    dataset = _make_dataset(x_drifts_per_frame)
    # M2: evaluate on the first frame directly (representative single-frame result)
    sf0 = dataset.frames[0]
    m2 = evaluate_edge_alignment(
        image=sf0.camera_frame.load(), points_lidar=sf0.lidar_frame.load(),
        T_CL=dataset.extrinsic.T_CL, camera=dataset.camera, lidar_spec=_make_lidar_spec(),
        **edge_kwargs,
    )
    m3 = evaluate_holdout_consistency(
        dataset, lidar_spec=_make_lidar_spec(), n_blocks=n_blocks,
        min_frames_per_block=min_frames_block, edge_alignment_kwargs=edge_kwargs,
    )
    m4 = evaluate_multiframe_consistency(
        dataset, lidar_spec=_make_lidar_spec(), min_frames=min_frames_m4,
        edge_alignment_kwargs=edge_kwargs,
    )
    return m2, m3, m4


# ---------------------------------------------------------------------------
# End-to-end: uniformly good calibration -> high overall score, GOOD
# ---------------------------------------------------------------------------

def test_quality_score_all_good_end_to_end():
    m2, m3, m4 = _run_all_metrics([0.0] * 40, depth_jump_threshold_m=1.0)
    result = compute_quality_score(m2, m3, m4)

    assert result.num_valid_categories == 3
    geo = result.category("geometry")
    gen = result.category("generalization")
    stab = result.category("stability")

    assert geo.valid and gen.valid and stab.valid
    assert geo.classification == "GOOD"
    assert gen.classification == "GOOD"
    assert stab.classification == "GOOD"
    assert result.overall_score > 80
    assert result.overall_classification == "GOOD"


def test_quality_score_drifting_calibration_lowers_generalization_and_stability():
    m2_u, m3_u, m4_u = _run_all_metrics([0.0] * 40, depth_jump_threshold_m=1.0)
    m2_d, m3_d, m4_d = _run_all_metrics(
        [0.0] * 20 + [0.08] * 20, depth_jump_threshold_m=1.0, edge_radius_px=8.0,
    )

    result_uniform = compute_quality_score(m2_u, m3_u, m4_u)
    result_drifting = compute_quality_score(m2_d, m3_d, m4_d)

    gen_u = result_uniform.category("generalization").score
    gen_d = result_drifting.category("generalization").score
    assert gen_d < gen_u

    assert result_drifting.overall_score < result_uniform.overall_score


# ---------------------------------------------------------------------------
# Weight handling
# ---------------------------------------------------------------------------

def test_quality_score_default_weights_are_equal():
    assert DEFAULT_WEIGHTS == {"geometry": 1 / 3, "generalization": 1 / 3, "stability": 1 / 3}


def test_quality_score_overall_is_weighted_average_when_all_valid():
    m2, m3, m4 = _run_all_metrics([0.0] * 40, depth_jump_threshold_m=1.0)
    result = compute_quality_score(m2, m3, m4)
    geo = result.category("geometry").score
    gen = result.category("generalization").score
    stab = result.category("stability").score
    expected = (geo + gen + stab) / 3.0
    assert math.isclose(result.overall_score, expected, abs_tol=1e-6)


def test_quality_score_custom_weights_are_applied():
    m2, m3, m4 = _run_all_metrics([0.0] * 40, depth_jump_threshold_m=1.0)
    weights = {"geometry": 1.0, "generalization": 0.0, "stability": 0.0}
    result = compute_quality_score(m2, m3, m4, weights=weights)
    geo = result.category("geometry").score
    assert math.isclose(result.overall_score, geo, abs_tol=1e-6)


def test_quality_score_rejects_wrong_weight_keys():
    m2, m3, m4 = _run_all_metrics([0.0] * 40, depth_jump_threshold_m=1.0)
    try:
        compute_quality_score(m2, m3, m4, weights={"geometry": 1.0, "foo": 0.0, "stability": 0.0})
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_quality_score_rejects_negative_weights():
    m2, m3, m4 = _run_all_metrics([0.0] * 40, depth_jump_threshold_m=1.0)
    try:
        compute_quality_score(m2, m3, m4, weights={"geometry": -1.0, "generalization": 1.0, "stability": 1.0})
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_quality_score_rejects_all_zero_weights():
    m2, m3, m4 = _run_all_metrics([0.0] * 40, depth_jump_threshold_m=1.0)
    try:
        compute_quality_score(m2, m3, m4, weights={"geometry": 0.0, "generalization": 0.0, "stability": 0.0})
        assert False, "expected ValueError"
    except ValueError:
        pass


# ---------------------------------------------------------------------------
# Category exclusion on FAIL
# ---------------------------------------------------------------------------

def test_quality_score_excludes_failed_category_and_renormalizes():
    m2, m3, _ = _run_all_metrics([0.0] * 40, depth_jump_threshold_m=1.0)
    # force M4 to FAIL by giving it too few frames relative to its min_frames
    dataset = _make_dataset([0.0] * 10)
    m4_fail = evaluate_multiframe_consistency(dataset, lidar_spec=_make_lidar_spec(), min_frames=30)
    assert m4_fail.classification == "FAIL"

    result = compute_quality_score(m2, m3, m4_fail)
    assert result.num_valid_categories == 2
    stab = result.category("stability")
    assert not stab.valid
    assert stab.classification == "FAIL"
    assert math.isnan(stab.score)

    geo = result.category("geometry").score
    gen = result.category("generalization").score
    expected = (geo + gen) / 2.0  # equal weights renormalized over 2 valid categories
    assert math.isclose(result.overall_score, expected, abs_tol=1e-6)
    assert any("excluded from the Overall Quality" in w for w in result.warnings)


def test_quality_score_partial_result_capped_at_warning_even_if_score_is_high():
    # M2 and M3 are both uniformly good (would score GOOD on their own),
    # but M4 FAILs outright (too few frames). Regression test for the bug
    # where a partial Overall Quality (here 1/3 categories, since M2 is
    # used as the drift scenario's single successful category in the CLI
    # report) could still be labeled "GOOD" -- which is misleading for a
    # --fail-on-bad CI gate, since it implies the calibration is fully
    # evaluated and trustworthy when most of it couldn't be measured.
    m2, m3, _ = _run_all_metrics([0.0] * 40, depth_jump_threshold_m=1.0)
    dataset = _make_dataset([0.0] * 10)
    m4_fail = evaluate_multiframe_consistency(dataset, lidar_spec=_make_lidar_spec(), min_frames=30)
    assert m4_fail.classification == "FAIL"

    result = compute_quality_score(m2, m3, m4_fail)
    assert result.num_valid_categories == 2
    # The underlying renormalized score is still high (M2 + M3 both GOOD)...
    assert result.overall_score > 80
    # ...but the classification must NOT read as GOOD, since a category
    # FAILing outright is worse information than a low-but-valid score.
    assert result.overall_classification == "WARNING"
    assert any("capped at WARNING" in w for w in result.warnings)


def test_quality_score_full_result_not_capped_when_all_categories_valid():
    # Sanity check the cap only applies to partial results: an all-valid,
    # all-GOOD result must still report GOOD, not be capped by accident.
    m2, m3, m4 = _run_all_metrics([0.0] * 40, depth_jump_threshold_m=1.0)
    result = compute_quality_score(m2, m3, m4)
    assert result.num_valid_categories == 3
    assert result.overall_classification == "GOOD"
    assert not any("capped at WARNING" in w for w in result.warnings)


def test_quality_score_all_categories_failed_gives_nan_overall():
    blank_camera = _make_camera()
    blank_image = np.zeros((blank_camera.height, blank_camera.width, 3), dtype=np.uint8)
    m2_fail = evaluate_edge_alignment(
        image=blank_image, points_lidar=_make_base_points_cam_frame(), T_CL=np.eye(4),
        camera=blank_camera, lidar_spec=_make_lidar_spec(),
    )
    assert m2_fail.classification == "FAIL"

    dataset_small = _make_dataset([0.0] * 2)
    m3_fail = evaluate_holdout_consistency(dataset_small, lidar_spec=_make_lidar_spec(), n_blocks=4, min_frames_per_block=30)
    m4_fail = evaluate_multiframe_consistency(dataset_small, lidar_spec=_make_lidar_spec(), min_frames=30)
    assert m3_fail.classification == "FAIL"
    assert m4_fail.classification == "FAIL"

    result = compute_quality_score(m2_fail, m3_fail, m4_fail)
    assert result.num_valid_categories == 0
    assert math.isnan(result.overall_score)
    assert result.overall_classification == "FAIL"
    assert any("All categories FAILed" in w for w in result.warnings)


# ---------------------------------------------------------------------------
# Structural sanity
# ---------------------------------------------------------------------------

def test_quality_score_categories_present_and_named_correctly():
    m2, m3, m4 = _run_all_metrics([0.0] * 40, depth_jump_threshold_m=1.0)
    result = compute_quality_score(m2, m3, m4)
    names = {c.name for c in result.categories}
    assert names == {"geometry", "generalization", "stability"}
    metric_names = {c.name: c.metric_name for c in result.categories}
    assert metric_names == {"geometry": "M2", "generalization": "M3", "stability": "M4"}


def test_quality_score_overall_bounded_0_100_when_valid():
    m2, m3, m4 = _run_all_metrics([0.0] * 40, depth_jump_threshold_m=1.0)
    result = compute_quality_score(m2, m3, m4)
    assert 0.0 <= result.overall_score <= 100.0


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
