import sys
import os
import json as _json
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math
import numpy as np

from report.builder import build_report, m0_summary
from report.json import to_json_string, write_json_report
from report.html import render_html_report, write_html_report

from evaluation.edge_alignment import evaluate_edge_alignment
from evaluation.holdout_consistency import evaluate_holdout_consistency
from evaluation.multiframe_consistency import evaluate_multiframe_consistency
from quality.quality_score import compute_quality_score

from tests.test_holdout_consistency import _make_dataset, _make_lidar_spec


def _full_pipeline(x_drifts, **edge_kwargs):
    dataset = _make_dataset(x_drifts)
    sf0 = dataset.frames[0]
    m2 = evaluate_edge_alignment(
        image=sf0.camera_frame.load(), points_lidar=sf0.lidar_frame.load(),
        T_CL=dataset.extrinsic.T_CL, camera=dataset.camera, lidar_spec=_make_lidar_spec(),
        **edge_kwargs,
    )
    m3 = evaluate_holdout_consistency(dataset, lidar_spec=_make_lidar_spec(), n_blocks=4,
                                       min_frames_per_block=5, edge_alignment_kwargs=edge_kwargs)
    m4 = evaluate_multiframe_consistency(dataset, lidar_spec=_make_lidar_spec(), min_frames=30,
                                          edge_alignment_kwargs=edge_kwargs)
    quality = compute_quality_score(m2, m3, m4)
    return dataset, m2, m3, m4, quality


# ---------------------------------------------------------------------------
# build_report structure
# ---------------------------------------------------------------------------

def test_build_report_has_expected_top_level_keys():
    dataset, m2, m3, m4, quality = _full_pipeline([0.0] * 40, depth_jump_threshold_m=1.0)
    report = build_report(dataset, m2, m3, m4, quality, n_blocks=4, min_frames_m4=30)
    expected_keys = {"metadata", "input_validation", "synchronization", "motion_deskew", "dynamic_filter",
                      "m0_sanity_gate", "m2_edge_alignment", "m2_spatial_analysis", "m3_holdout_consistency",
                      "m4_multiframe_consistency", "quality_score", "advanced", "root_cause_diagnosis",
                      "quality_confidence_coverage", "warnings"}
    assert set(report.keys()) == expected_keys


def test_build_report_metadata_contains_camera_lidar_extrinsic():
    dataset, m2, m3, m4, quality = _full_pipeline([0.0] * 40, depth_jump_threshold_m=1.0)
    report = build_report(dataset, m2, m3, m4, quality)
    meta = report["metadata"]
    assert meta["camera"]["width"] == dataset.camera.width
    assert meta["camera"]["fx"] == dataset.camera.intrinsics.fx
    assert meta["lidar"]["source_kind"] == dataset.lidar.source.kind
    assert len(meta["extrinsic"]["T_CL"]) == 4
    assert len(meta["extrinsic"]["T_CL"][0]) == 4
    assert meta["dataset"]["num_synced_frames"] == len(dataset.frames)


def test_build_report_m0_defaults_to_none():
    dataset, m2, m3, m4, quality = _full_pipeline([0.0] * 40, depth_jump_threshold_m=1.0)
    report = build_report(dataset, m2, m3, m4, quality)
    assert report["m0_sanity_gate"] is None


def test_build_report_m0_passthrough_when_provided():
    dataset, m2, m3, m4, quality = _full_pipeline([0.0] * 40, depth_jump_threshold_m=1.0)
    fake_m0 = {"passed": True, "checks": []}
    report = build_report(dataset, m2, m3, m4, quality, m0_result=fake_m0)
    assert report["m0_sanity_gate"] == fake_m0
    assert m0_summary(fake_m0) == fake_m0
    assert m0_summary(None) is None


def test_build_report_quality_section_matches_source():
    dataset, m2, m3, m4, quality = _full_pipeline([0.0] * 40, depth_jump_threshold_m=1.0)
    report = build_report(dataset, m2, m3, m4, quality)
    q = report["quality_score"]
    assert math.isclose(q["overall_score"], quality.overall_score, abs_tol=1e-6)
    assert q["overall_classification"] == quality.overall_classification
    assert len(q["categories"]) == 3


def test_build_report_m4_trajectory_matches_frame_count():
    dataset, m2, m3, m4, quality = _full_pipeline([0.0] * 40, depth_jump_threshold_m=1.0)
    report = build_report(dataset, m2, m3, m4, quality)
    assert len(report["m4_multiframe_consistency"]["frame_trajectory"]) == len(dataset.frames)


# ---------------------------------------------------------------------------
# JSON serialization
# ---------------------------------------------------------------------------

def test_json_report_is_valid_json_and_round_trips():
    dataset, m2, m3, m4, quality = _full_pipeline([0.0] * 40, depth_jump_threshold_m=1.0)
    report = build_report(dataset, m2, m3, m4, quality)
    s = to_json_string(report)
    parsed = _json.loads(s)  # must not raise
    assert parsed["quality_score"]["overall_classification"] == quality.overall_classification


def test_json_report_sanitizes_nan_to_null_on_failed_metric():
    # force M4 to FAIL (too few frames) -> its NaN fields must serialize as null
    dataset = _make_dataset([0.0] * 5)
    m4_fail = evaluate_multiframe_consistency(dataset, lidar_spec=_make_lidar_spec(), min_frames=30)
    assert m4_fail.classification == "FAIL"
    assert math.isnan(m4_fail.std_across_frames_px)

    sf0 = dataset.frames[0]
    m2 = evaluate_edge_alignment(image=sf0.camera_frame.load(), points_lidar=sf0.lidar_frame.load(),
                                  T_CL=dataset.extrinsic.T_CL, camera=dataset.camera,
                                  lidar_spec=_make_lidar_spec(), depth_jump_threshold_m=1.0)
    m3 = evaluate_holdout_consistency(dataset, lidar_spec=_make_lidar_spec(), n_blocks=4,
                                       min_frames_per_block=30)
    quality = compute_quality_score(m2, m3, m4_fail)

    report = build_report(dataset, m2, m3, m4_fail, quality)
    s = to_json_string(report)
    parsed = _json.loads(s)
    assert parsed["m4_multiframe_consistency"]["std_across_frames_px"] is None
    assert parsed["quality_score"]["categories"][2]["score"] is None  # stability category


def test_to_json_string_raises_on_unsanitized_nan():
    bad_report = {"value": float("nan")}
    try:
        to_json_string(bad_report)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_write_json_report_writes_file_matching_string():
    dataset, m2, m3, m4, quality = _full_pipeline([0.0] * 40, depth_jump_threshold_m=1.0)
    report = build_report(dataset, m2, m3, m4, quality)
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "report.json")
        write_json_report(report, path)
        assert os.path.exists(path)
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        assert content == to_json_string(report)
        _json.loads(content)  # still valid


# ---------------------------------------------------------------------------
# HTML rendering
# ---------------------------------------------------------------------------

def test_html_report_contains_title_and_overall_score():
    dataset, m2, m3, m4, quality = _full_pipeline([0.0] * 40, depth_jump_threshold_m=1.0)
    report = build_report(dataset, m2, m3, m4, quality)
    html = render_html_report(report)
    assert "<!DOCTYPE html>" in html
    assert "Cam" in html and "Calibration Quality" in html
    assert f"{quality.overall_score:.1f}" in html
    assert quality.overall_classification in html


def test_html_report_contains_all_category_badges():
    dataset, m2, m3, m4, quality = _full_pipeline([0.0] * 40, depth_jump_threshold_m=1.0)
    report = build_report(dataset, m2, m3, m4, quality)
    html = render_html_report(report)
    for cat in quality.categories:
        assert cat.classification in html


def test_html_report_escapes_warning_text():
    dataset, m2, m3, m4, quality = _full_pipeline([0.0] * 40, depth_jump_threshold_m=1.0)
    report = build_report(dataset, m2, m3, m4, quality,
                           extra_warnings=["<script>alert('xss')</script>"])
    html = render_html_report(report)
    assert "<script>alert" not in html
    assert "&lt;script&gt;" in html


def test_html_report_handles_failed_metric_gracefully():
    dataset = _make_dataset([0.0] * 5)
    m4_fail = evaluate_multiframe_consistency(dataset, lidar_spec=_make_lidar_spec(), min_frames=30)
    sf0 = dataset.frames[0]
    m2 = evaluate_edge_alignment(image=sf0.camera_frame.load(), points_lidar=sf0.lidar_frame.load(),
                                  T_CL=dataset.extrinsic.T_CL, camera=dataset.camera,
                                  lidar_spec=_make_lidar_spec(), depth_jump_threshold_m=1.0)
    m3 = evaluate_holdout_consistency(dataset, lidar_spec=_make_lidar_spec(), n_blocks=4,
                                       min_frames_per_block=30)
    quality = compute_quality_score(m2, m3, m4_fail)
    report = build_report(dataset, m2, m3, m4_fail, quality)
    html = render_html_report(report)  # must not raise on NaN/None values
    assert "&mdash;" in html  # NaN fields render as em-dash placeholder
    assert "FAIL" in html


# ---------------------------------------------------------------------------
# STEP1 (Input Validation) / STEP2 (Synchronization) / STEP3 (Projection
# Sanity Check) HTML sections
# ---------------------------------------------------------------------------

def test_html_report_omits_input_validation_section_when_none():
    # dataset.input_validation is None for this fixture (never ran STEP1
    # validation, e.g. --demo mode) -- the section must not appear at all,
    # not render as an empty/broken block.
    dataset, m2, m3, m4, quality = _full_pipeline([0.0] * 40, depth_jump_threshold_m=1.0)
    assert dataset.input_validation is None
    report = build_report(dataset, m2, m3, m4, quality)
    html = render_html_report(report)
    assert "Input Validation" not in html


def test_html_report_shows_input_validation_when_present():
    from input.validation import ValidationReport, ValidationCheckItem, ValidationStatus
    dataset, m2, m3, m4, quality = _full_pipeline([0.0] * 40, depth_jump_threshold_m=1.0)
    vreport = ValidationReport()
    vreport.add(ValidationCheckItem("camera.fx_positive", ValidationStatus.WARNING, "fx is suspiciously small"))
    dataset.input_validation = vreport.to_dict()
    report = build_report(dataset, m2, m3, m4, quality)
    html = render_html_report(report)
    assert "Input Validation" in html
    assert "INPUT_WARNING" in html
    assert "fx is suspiciously small" in html


def test_html_report_omits_synchronization_section_when_none():
    dataset, m2, m3, m4, quality = _full_pipeline([0.0] * 40, depth_jump_threshold_m=1.0)
    assert dataset.sync_stats is None
    report = build_report(dataset, m2, m3, m4, quality)
    html = render_html_report(report)
    assert "Synchronization" not in html


def test_html_report_shows_synchronization_when_present():
    from input.dataset import SyncStats
    dataset, m2, m3, m4, quality = _full_pipeline([0.0] * 40, depth_jump_threshold_m=1.0)
    dataset.sync_stats = SyncStats(
        num_camera_frames=40, num_lidar_frames=40, num_matched=38,
        num_camera_dropped=2, num_lidar_dropped=2, mean_time_diff_ms=5.0, max_time_diff_ms=12.0,
        estimated_offset_ms=17.4, offset_std_ms=2.1, drop_ratio=0.05, classification="GOOD",
    )
    report = build_report(dataset, m2, m3, m4, quality)
    html = render_html_report(report)
    assert "Synchronization" in html
    assert "38 / 40" in html
    assert "17.4" in html


def test_html_report_omits_projection_overlay_section_when_absent():
    dataset, m2, m3, m4, quality = _full_pipeline([0.0] * 40, depth_jump_threshold_m=1.0)
    report = build_report(dataset, m2, m3, m4, quality)
    html = render_html_report(report)  # no visuals dict passed at all
    assert "Projection Sanity Check" not in html


def test_html_report_shows_projection_overlay_when_present():
    dataset, m2, m3, m4, quality = _full_pipeline([0.0] * 40, depth_jump_threshold_m=1.0)
    report = build_report(dataset, m2, m3, m4, quality)
    fake_png = b"\x89PNG\r\n\x1a\n" + b"\x00" * 20
    html = render_html_report(report, visuals={"projection_overlay_png": fake_png})
    assert "Projection Sanity Check" in html


def test_html_report_omits_range_image_section_when_absent():
    dataset, m2, m3, m4, quality = _full_pipeline([0.0] * 40, depth_jump_threshold_m=1.0)
    report = build_report(dataset, m2, m3, m4, quality)
    html = render_html_report(report)  # no visuals dict passed at all
    assert "LiDAR Range Image" not in html


def test_html_report_shows_range_image_when_present():
    dataset, m2, m3, m4, quality = _full_pipeline([0.0] * 40, depth_jump_threshold_m=1.0)
    report = build_report(dataset, m2, m3, m4, quality)
    fake_png = b"\x89PNG\r\n\x1a\n" + b"\x00" * 20
    html = render_html_report(report, visuals={"range_image_png": fake_png})
    assert "LiDAR Range Image" in html


def test_html_report_m2_section_omits_uncertainty_plot_when_absent():
    dataset, m2, m3, m4, quality = _full_pipeline([0.0] * 40, depth_jump_threshold_m=1.0)
    report = build_report(dataset, m2, m3, m4, quality)
    html = render_html_report(report)  # no visuals dict passed at all
    # the M2 section's surrounding subtitle text always appears; the
    # actual <img> tag (identified by its distinctive alt text) must not,
    # when no PNG was supplied
    assert "Scatter of per-point error against depth" not in html


def test_html_report_m2_section_shows_uncertainty_plot_when_present():
    dataset, m2, m3, m4, quality = _full_pipeline([0.0] * 40, depth_jump_threshold_m=1.0)
    report = build_report(dataset, m2, m3, m4, quality)
    fake_png = b"\x89PNG\r\n\x1a\n" + b"\x00" * 20
    html = render_html_report(report, visuals={"uncertainty_plot_png": fake_png})
    assert "Scatter of per-point error against depth" in html


def test_html_report_m2_section_shows_normalized_error_and_match_rate_stats():
    dataset, m2, m3, m4, quality = _full_pipeline([0.0] * 40, depth_jump_threshold_m=1.0)
    assert m2.mean_normalized_error is not None  # STEP7 fields populated by default (use_correspondence_matching=True)
    report = build_report(dataset, m2, m3, m4, quality)
    html = render_html_report(report)
    assert "Mean normalized error" in html
    assert "Match rate (STEP6)" in html


def test_html_report_m4_section_shows_step10_robust_stats():
    dataset, m2, m3, m4, quality = _full_pipeline([0.0] * 40, depth_jump_threshold_m=1.0)
    assert m4.outlier_method == "hampel"
    report = build_report(dataset, m2, m3, m4, quality)
    html = render_html_report(report)
    assert "MAD" in html
    assert "IQR" in html
    assert "Valid ratio" in html
    assert "Failure ratio" in html
    assert "Outlier ratio" in html
    assert "hampel" in html


def test_html_report_m3_section_shows_step10_scene_columns():
    dataset, m2, m3, m4, quality = _full_pipeline([0.0] * 40, depth_jump_threshold_m=1.0)
    report = build_report(dataset, m2, m3, m4, quality, n_blocks=4, min_frames_m4=30)
    html = render_html_report(report)
    assert "Edge density" in html
    assert "FOV coverage" in html


def test_html_report_shows_spatial_analysis_section_by_default():
    """STEP9 needs no extra opt-in input (unlike STEP5/STEP8) -- it
    should show up automatically whenever M2 succeeded."""
    dataset, m2, m3, m4, quality = _full_pipeline([0.0] * 40, depth_jump_threshold_m=1.0)
    report = build_report(dataset, m2, m3, m4, quality)
    assert report["m2_spatial_analysis"] is not None
    html = render_html_report(report)
    assert "Depth / Spatial Analysis" in html
    assert "0-10m" in html
    assert "LEFT" in html and "CENTER" in html and "RIGHT" in html
    assert "TOP" in html and "BOTTOM" in html


def test_html_report_spatial_analysis_omitted_when_m2_fails():
    dataset = _make_dataset([0.0] * 5)
    m4_fail = evaluate_multiframe_consistency(dataset, lidar_spec=_make_lidar_spec(), min_frames=30)
    sf0 = dataset.frames[0]
    m2_fail = evaluate_edge_alignment(image=sf0.camera_frame.load(), points_lidar=np.zeros((0, 3)),
                                       T_CL=dataset.extrinsic.T_CL, camera=dataset.camera,
                                       lidar_spec=_make_lidar_spec())
    m3_fail = evaluate_holdout_consistency(dataset, lidar_spec=_make_lidar_spec(), n_blocks=4,
                                            min_frames_per_block=30)
    quality = compute_quality_score(m2_fail, m3_fail, m4_fail)
    report = build_report(dataset, m2_fail, m3_fail, m4_fail, quality)
    assert report["m2_spatial_analysis"] is None
    html = render_html_report(report)
    assert "Depth / Spatial Analysis" not in html


def test_html_report_shows_spatial_analysis_image_when_present():
    dataset, m2, m3, m4, quality = _full_pipeline([0.0] * 40, depth_jump_threshold_m=1.0)
    report = build_report(dataset, m2, m3, m4, quality)
    fake_png = b"\x89PNG\r\n\x1a\n" + b"\x00" * 20
    html = render_html_report(report, visuals={"spatial_analysis_png": fake_png})
    assert "Depth / Spatial Analysis" in html
    assert "data:image/png;base64," in html


def test_html_report_omits_motion_deskew_section_when_absent():
    dataset, m2, m3, m4, quality = _full_pipeline([0.0] * 40, depth_jump_threshold_m=1.0)
    report = build_report(dataset, m2, m3, m4, quality)
    html = render_html_report(report)
    assert "Motion Deskew" not in html


def test_html_report_shows_motion_deskew_when_present():
    from motion.deskew import deskew_points_constant_velocity, compare_before_after
    dataset, m2, m3, m4, quality = _full_pipeline([0.0] * 40, depth_jump_threshold_m=1.0)
    points = dataset.frames[0].lidar_frame.load()
    result = deskew_points_constant_velocity(
        points, scan_period_s=0.1,
        linear_velocity_mps=np.array([5.0, 0.0, 0.0]), angular_velocity_rps=np.zeros(3),
    )
    compare = compare_before_after(points, result)
    report = build_report(dataset, m2, m3, m4, quality, deskew_compare=compare)
    fake_png = b"\x89PNG\r\n\x1a\n" + b"\x00" * 20
    html = render_html_report(report, visuals={"deskew_comparison_png": fake_png})
    assert "Motion Deskew" in html
    assert f"{compare['mean_correction_m']:.3f}" in html or str(compare["num_points"]) in html


def test_html_report_omits_dynamic_filter_section_when_absent():
    dataset, m2, m3, m4, quality = _full_pipeline([0.0] * 40, depth_jump_threshold_m=1.0)
    report = build_report(dataset, m2, m3, m4, quality)
    html = render_html_report(report)
    assert "Dynamic Object Filtering" not in html


def test_html_report_shows_dynamic_filter_when_present():
    from evaluation.dynamic_filter import DynamicFilteringComparison
    dataset, m2, m3, m4, quality = _full_pipeline([0.0] * 40, depth_jump_threshold_m=1.0)
    comparison = DynamicFilteringComparison(
        overall_mean_px=3.1, overall_classification="BAD", overall_num_edge_points=500,
        static_only_mean_px=1.2, static_only_classification="GOOD", static_only_num_edge_points=310,
        dynamic_contamination_ratio=0.38, num_dynamic_points_removed=190,
    )
    report = build_report(dataset, m2, m3, m4, quality, dynamic_filter_comparison=comparison)
    fake_png = b"\x89PNG\r\n\x1a\n" + b"\x00" * 20
    html = render_html_report(report, visuals={"dynamic_filter_overlay_png": fake_png})
    assert "Dynamic Object Filtering" in html
    assert "38.0%" in html


def test_write_html_report_writes_file():
    dataset, m2, m3, m4, quality = _full_pipeline([0.0] * 40, depth_jump_threshold_m=1.0)
    report = build_report(dataset, m2, m3, m4, quality)
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "report.html")
        write_html_report(report, path)
        assert os.path.exists(path)
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        assert content == render_html_report(report)


def test_build_report_advanced_defaults_to_none():
    dataset, m2, m3, m4, quality = _full_pipeline([0.0] * 40, depth_jump_threshold_m=1.0)
    report = build_report(dataset, m2, m3, m4, quality)
    assert report["advanced"] is None


def test_build_report_advanced_section_with_all_three_metrics():
    from evaluation.plane_consistency import evaluate_plane_consistency
    from evaluation.perturbation import evaluate_perturbation_sensitivity
    from evaluation.temporal_drift import evaluate_temporal_drift

    dataset, m2, m3, m4, quality = _full_pipeline([0.0] * 40, depth_jump_threshold_m=1.0)
    sf0 = dataset.frames[0]
    image = sf0.camera_frame.load()
    points = sf0.lidar_frame.load()

    plane_result = evaluate_plane_consistency(image=image, points_lidar=points, T_CL=dataset.extrinsic.T_CL,
                                               camera=dataset.camera, lidar_spec=_make_lidar_spec())
    perturbation_result = evaluate_perturbation_sensitivity(
        image=image, points_lidar=points, T_CL=dataset.extrinsic.T_CL, camera=dataset.camera,
        lidar_spec=_make_lidar_spec(), translation_deltas_m=(0.01,), rotation_deltas_deg=(0.1,),
        edge_alignment_kwargs={"depth_jump_threshold_m": 1.0},
    )
    temporal_drift_result = evaluate_temporal_drift(m4)

    report = build_report(dataset, m2, m3, m4, quality,
                           plane_result=plane_result, perturbation_result=perturbation_result,
                           temporal_drift_result=temporal_drift_result)

    assert report["advanced"] is not None
    assert set(report["advanced"].keys()) == {"plane_consistency", "perturbation", "temporal_drift"}
    assert report["advanced"]["plane_consistency"]["metric"] == "Plane Consistency"
    assert report["advanced"]["perturbation"]["metric"] == "Perturbation Sensitivity"
    assert report["advanced"]["temporal_drift"]["metric"] == "Temporal Drift"

    # must still serialize and render cleanly with the advanced section present
    s = to_json_string(report)
    parsed = _json.loads(s)
    assert parsed["advanced"]["plane_consistency"]["classification"] in ("GOOD", "WARNING", "BAD", "FAIL")

    html = render_html_report(report)
    assert "Advanced Diagnostics" in html
    assert "Plane Consistency" in html
    assert "Perturbation Sensitivity" in html
    assert "Temporal Drift" in html


def test_html_report_perturbation_shows_step11_sensitivity_ranking():
    from evaluation.perturbation import evaluate_perturbation_sensitivity

    dataset, m2, m3, m4, quality = _full_pipeline([0.0] * 40, depth_jump_threshold_m=1.0)
    sf0 = dataset.frames[0]
    image = sf0.camera_frame.load()
    points = sf0.lidar_frame.load()

    perturbation_result = evaluate_perturbation_sensitivity(
        image=image, points_lidar=points, T_CL=dataset.extrinsic.T_CL, camera=dataset.camera,
        lidar_spec=_make_lidar_spec(), translation_deltas_m=(0.001, 0.02), rotation_deltas_deg=(0.05, 1.0),
        edge_alignment_kwargs={"depth_jump_threshold_m": 1.0},
    )
    assert len(perturbation_result.axis_sensitivities) > 0
    report = build_report(dataset, m2, m3, m4, quality, perturbation_result=perturbation_result)
    html = render_html_report(report)
    assert "STEP11" in html
    assert any(c in html for c in ("HIGH", "MEDIUM", "LOW"))
    assert "Timestamp sensitivity not computed" in html  # no velocity supplied


def test_html_report_perturbation_shows_sensitivity_image_when_present():
    from evaluation.perturbation import evaluate_perturbation_sensitivity

    dataset, m2, m3, m4, quality = _full_pipeline([0.0] * 40, depth_jump_threshold_m=1.0)
    sf0 = dataset.frames[0]
    image = sf0.camera_frame.load()
    points = sf0.lidar_frame.load()
    perturbation_result = evaluate_perturbation_sensitivity(
        image=image, points_lidar=points, T_CL=dataset.extrinsic.T_CL, camera=dataset.camera,
        lidar_spec=_make_lidar_spec(), translation_deltas_m=(0.01,), rotation_deltas_deg=(0.1,),
        edge_alignment_kwargs={"depth_jump_threshold_m": 1.0},
    )
    report = build_report(dataset, m2, m3, m4, quality, perturbation_result=perturbation_result)
    fake_png = b"\x89PNG\r\n\x1a\n" + b"\x00" * 20
    html = render_html_report(report, visuals={"sensitivity_png": fake_png})
    assert "data:image/png;base64," in html


def test_build_report_always_includes_root_cause_diagnosis_key():
    """STEP12's root_cause_diagnosis is always computed (never None) --
    even with no opt-in diagnostics supplied, it's an empty-candidates
    result, not a missing key."""
    dataset, m2, m3, m4, quality = _full_pipeline([0.0] * 40, depth_jump_threshold_m=1.0)
    report = build_report(dataset, m2, m3, m4, quality)
    assert report["root_cause_diagnosis"] is not None
    assert "candidates" in report["root_cause_diagnosis"]


def test_build_report_always_includes_quality_confidence_coverage():
    dataset, m2, m3, m4, quality = _full_pipeline([0.0] * 40, depth_jump_threshold_m=1.0)
    report = build_report(dataset, m2, m3, m4, quality, n_blocks=4)
    qcc = report["quality_confidence_coverage"]
    assert qcc is not None
    assert qcc["quality"]["score"] == quality.overall_score
    assert qcc["confidence"]["score"] is not None
    assert qcc["coverage"]["score"] is not None


def test_html_report_shows_confidence_coverage_section():
    dataset, m2, m3, m4, quality = _full_pipeline([0.0] * 40, depth_jump_threshold_m=1.0)
    report = build_report(dataset, m2, m3, m4, quality, n_blocks=4)
    html = render_html_report(report)
    assert "Quality / Confidence / Coverage" in html
    assert "Confidence" in html
    assert "Coverage" in html


def test_html_report_shows_confirmations_only_when_no_candidates():
    """STEP14: a clean scene has no PROBLEMS, but the checks that ran and
    came back clean (e.g. GOOD sync) still show up as confirmations --
    the section is no longer omitted just because there's nothing wrong,
    since 'nothing wrong, and here's what we actually checked' is itself
    useful information (the spec's own mixed 🔴🟠🟢 panel)."""
    dataset, m2, m3, m4, quality = _full_pipeline([0.0] * 40, depth_jump_threshold_m=1.0)
    report = build_report(dataset, m2, m3, m4, quality)
    assert report["root_cause_diagnosis"]["candidates"] == []  # clean scene, nothing to diagnose
    assert len(report["root_cause_diagnosis"]["confirmations"]) > 0
    html = render_html_report(report)
    assert "Root Cause Diagnosis" in html
    assert "OK" in html


def test_html_report_omits_root_cause_section_when_truly_nothing_to_say():
    """Only when there are NEITHER candidates NOR confirmations (nothing
    was even checked) does the section disappear entirely."""
    dataset, m2, m3, m4, quality = _full_pipeline([0.0] * 40, depth_jump_threshold_m=1.0)
    report = build_report(dataset, m2, m3, m4, quality)
    report["root_cause_diagnosis"] = {"candidates": [], "confirmations": []}
    html = render_html_report(report)
    assert "Root Cause Diagnosis" not in html


def test_html_report_shows_root_cause_section_when_candidates_present():
    dataset, m2, m3, m4, quality = _full_pipeline([0.0] * 40, depth_jump_threshold_m=1.0)
    from evaluation.dynamic_filter import DynamicFilteringComparison
    fake_dynamic = DynamicFilteringComparison(
        overall_mean_px=3.1, overall_classification="BAD", overall_num_edge_points=500,
        static_only_mean_px=1.2, static_only_classification="GOOD", static_only_num_edge_points=300,
        dynamic_contamination_ratio=0.38, num_dynamic_points_removed=200,
    )
    report = build_report(dataset, m2, m3, m4, quality, dynamic_filter_comparison=fake_dynamic)
    assert len(report["root_cause_diagnosis"]["candidates"]) > 0
    html = render_html_report(report)
    assert "Root Cause Diagnosis" in html
    assert "Dynamic object contamination" in html
    assert "HIGH" in html


def test_html_report_omits_advanced_section_when_not_provided():
    dataset, m2, m3, m4, quality = _full_pipeline([0.0] * 40, depth_jump_threshold_m=1.0)
    report = build_report(dataset, m2, m3, m4, quality)
    html = render_html_report(report)
    assert "Advanced Diagnostics" not in html


def test_html_report_embeds_visuals_as_base64():
    dataset, m2, m3, m4, quality = _full_pipeline([0.0] * 40, depth_jump_threshold_m=1.0)
    report = build_report(dataset, m2, m3, m4, quality)

    from visualization.overlay import render_overlay, encode_png
    from visualization.trajectory import render_m4_trajectory_png
    from visualization.histogram import render_error_histogram_png

    sf0 = dataset.frames[0]
    overlay_img = render_overlay(sf0.camera_frame.load(), m2.edge_point_pixels,
                                  m2.edge_point_errors_px, m2.floor_px)
    overlay_png = encode_png(overlay_img)
    histogram_png = render_error_histogram_png(m2.edge_point_errors_px, m2.floor_px)
    trajectory_png = render_m4_trajectory_png(m4)

    html_without = render_html_report(report)
    html_with = render_html_report(report, visuals={
        "overlay_png": overlay_png, "histogram_png": histogram_png, "trajectory_png": trajectory_png,
    })

    assert "data:image/png;base64," in html_with
    assert "data:image/png;base64," not in html_without
    assert len(html_with) > len(html_without)


def test_html_report_visuals_missing_key_omitted_gracefully():
    dataset, m2, m3, m4, quality = _full_pipeline([0.0] * 40, depth_jump_threshold_m=1.0)
    report = build_report(dataset, m2, m3, m4, quality)
    html = render_html_report(report, visuals={"overlay_png": None})  # explicit None
    assert "data:image/png;base64," not in html


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
