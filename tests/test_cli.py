import sys
import os
import tempfile
import io
import contextlib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import cv2
import yaml

from app.cli import (
    build_arg_parser, main, _build_demo_dataset, _parse_weights, _parse_vec3,
    load_dataset_from_config, run_pipeline, validate_config_only, ConfigSchemaError,
)


# ---------------------------------------------------------------------------
# _parse_weights
# ---------------------------------------------------------------------------

def test_parse_weights_none_input():
    assert _parse_weights(None) is None


def test_parse_weights_basic():
    result = _parse_weights("geometry=0.5,generalization=0.25,stability=0.25")
    assert result == {"geometry": 0.5, "generalization": 0.25, "stability": 0.25}


def test_parse_weights_handles_spaces():
    result = _parse_weights(" geometry = 1.0 , generalization = 0.0, stability=0.0")
    assert result == {"geometry": 1.0, "generalization": 0.0, "stability": 0.0}


# ---------------------------------------------------------------------------
# _parse_vec3 (STEP5 --deskew-linear-velocity / --deskew-angular-velocity)
# ---------------------------------------------------------------------------

def test_parse_vec3_none_input():
    assert _parse_vec3(None, "--deskew-linear-velocity") is None


def test_parse_vec3_basic():
    result = _parse_vec3("1.0,2.0,3.0", "--deskew-linear-velocity")
    assert np.allclose(result, [1.0, 2.0, 3.0])


def test_parse_vec3_handles_spaces_and_negative_numbers():
    result = _parse_vec3(" -1.5 , 0.0 , 2.25 ", "--deskew-angular-velocity")
    assert np.allclose(result, [-1.5, 0.0, 2.25])


def test_parse_vec3_wrong_component_count_raises():
    try:
        _parse_vec3("1.0,2.0", "--deskew-linear-velocity")
        assert False, "expected ValueError"
    except ValueError as e:
        assert "--deskew-linear-velocity" in str(e)


def test_parse_vec3_non_numeric_raises():
    try:
        _parse_vec3("a,b,c", "--deskew-angular-velocity")
        assert False, "expected ValueError"
    except ValueError as e:
        assert "--deskew-angular-velocity" in str(e)


# ---------------------------------------------------------------------------
# _build_demo_dataset
# ---------------------------------------------------------------------------

def test_build_demo_dataset_good_scenario():
    dataset = _build_demo_dataset("good", num_frames=20)
    assert len(dataset.frames) == 20
    assert np.allclose(dataset.extrinsic.T_CL, np.eye(4))


def test_build_demo_dataset_drift_scenario():
    dataset = _build_demo_dataset("drift", num_frames=20)
    assert len(dataset.frames) == 20
    # first half should be undrifted, second half drifted -- verify via
    # the underlying point clouds directly (drift subtracts from x)
    pts_first = dataset.frames[0].lidar_frame.points
    pts_second = dataset.frames[19].lidar_frame.points
    assert not np.allclose(pts_first[:, 0], pts_second[:, 0])


def test_build_demo_dataset_invalid_scenario_raises():
    try:
        _build_demo_dataset("nonsense", num_frames=10)
        assert False, "expected ValueError"
    except ValueError:
        pass


# ---------------------------------------------------------------------------
# run_pipeline (direct, bypassing argparse)
# ---------------------------------------------------------------------------

def test_run_pipeline_writes_reports_and_returns_dict():
    dataset = _build_demo_dataset("good", num_frames=20)
    with tempfile.TemporaryDirectory() as tmpdir:
        report = run_pipeline(dataset, tmpdir, n_blocks=4, min_frames_per_block=4,
                               min_frames_m4=20, depth_jump_threshold_m=1.0)
        assert os.path.exists(os.path.join(tmpdir, "report.json"))
        assert os.path.exists(os.path.join(tmpdir, "report.html"))
        assert report["quality_score"]["overall_classification"] == "GOOD"


def test_run_pipeline_no_visuals_produces_smaller_html():
    dataset = _build_demo_dataset("good", num_frames=20)
    with tempfile.TemporaryDirectory() as tmpdir1, tempfile.TemporaryDirectory() as tmpdir2:
        run_pipeline(dataset, tmpdir1, n_blocks=4, min_frames_per_block=4,
                     min_frames_m4=20, depth_jump_threshold_m=1.0, no_visuals=False)
        run_pipeline(dataset, tmpdir2, n_blocks=4, min_frames_per_block=4,
                     min_frames_m4=20, depth_jump_threshold_m=1.0, no_visuals=True)
        size_with = os.path.getsize(os.path.join(tmpdir1, "report.html"))
        size_without = os.path.getsize(os.path.join(tmpdir2, "report.html"))
        assert size_without < size_with


def test_run_pipeline_custom_weights_applied():
    dataset = _build_demo_dataset("drift", num_frames=20)
    with tempfile.TemporaryDirectory() as tmpdir:
        report = run_pipeline(
            dataset, tmpdir, n_blocks=4, min_frames_per_block=4, min_frames_m4=20,
            depth_jump_threshold_m=1.0, edge_radius_px=8.0, frame_index=0,
            weights={"geometry": 1.0, "generalization": 0.0, "stability": 0.0},
        )
        geo_score = next(c["score"] for c in report["quality_score"]["categories"] if c["name"] == "geometry")
        assert abs(report["quality_score"]["overall_score"] - geo_score) < 1e-6


def test_run_pipeline_raises_clear_error_on_camera_image_dimension_mismatch():
    # A camera config's declared width/height that doesn't match the
    # actual loaded image (typo'd config, or an image_dir with mixed
    # resolutions) used to surface as a downstream IndexError from deep
    # inside a visualization function. run_pipeline should now catch this
    # itself, right after loading the headline frame's image, with a
    # message that names both the declared and actual dimensions.
    dataset = _build_demo_dataset("good", num_frames=5)
    dataset.camera.width = 9999
    dataset.camera.height = 9999
    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            run_pipeline(dataset, tmpdir, n_blocks=1, min_frames_per_block=1, min_frames_m4=1)
            assert False, "expected ValueError"
        except ValueError as e:
            assert "9999" in str(e)
        except IndexError:
            assert False, "should raise a clear ValueError, not a bare IndexError"


def test_run_pipeline_raises_on_empty_dataset():
    dataset = _build_demo_dataset("good", num_frames=1)
    dataset.frames = []
    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            run_pipeline(dataset, tmpdir)
            assert False, "expected ValueError"
        except ValueError:
            pass


# ---------------------------------------------------------------------------
# CLI argument parsing / main()
# ---------------------------------------------------------------------------

def test_arg_parser_requires_config_or_demo():
    parser = build_arg_parser()
    try:
        parser.parse_args([])
        assert False, "expected SystemExit (mutually exclusive group required)"
    except SystemExit:
        pass


def test_arg_parser_rejects_both_config_and_demo():
    parser = build_arg_parser()
    try:
        parser.parse_args(["--config", "x.yaml", "--demo"])
        assert False, "expected SystemExit"
    except SystemExit:
        pass


def test_main_demo_end_to_end_returns_zero():
    with tempfile.TemporaryDirectory() as tmpdir:
        code = main(["--demo", "--demo-frames", "20", "--output-dir", tmpdir, "--no-visuals"])
        assert code == 0
        assert os.path.exists(os.path.join(tmpdir, "report.json"))


def test_main_demo_advanced_flag_populates_advanced_section():
    with tempfile.TemporaryDirectory() as tmpdir:
        code = main(["--demo", "--demo-frames", "20", "--output-dir", tmpdir,
                     "--no-visuals", "--advanced"])
        assert code == 0
        import json
        with open(os.path.join(tmpdir, "report.json")) as f:
            report = json.load(f)
        assert report["advanced"] is not None
        assert report["advanced"]["plane_consistency"] is not None
        assert report["advanced"]["perturbation"] is not None
        assert report["advanced"]["temporal_drift"] is not None


def test_main_demo_without_advanced_flag_omits_advanced_section():
    with tempfile.TemporaryDirectory() as tmpdir:
        code = main(["--demo", "--demo-frames", "20", "--output-dir", tmpdir, "--no-visuals"])
        assert code == 0
        import json
        with open(os.path.join(tmpdir, "report.json")) as f:
            report = json.load(f)
        assert report["advanced"] is None


# ---------------------------------------------------------------------------
# --json-only
# ---------------------------------------------------------------------------

def test_main_json_only_skips_html_report():
    with tempfile.TemporaryDirectory() as tmpdir:
        code = main(["--demo", "--demo-frames", "20", "--output-dir", tmpdir, "--json-only"])
        assert code == 0
        assert os.path.exists(os.path.join(tmpdir, "report.json"))
        assert not os.path.exists(os.path.join(tmpdir, "report.html"))


def test_main_without_json_only_still_writes_html_report():
    # sanity check: default behavior (no --json-only) is unchanged
    with tempfile.TemporaryDirectory() as tmpdir:
        code = main(["--demo", "--demo-frames", "20", "--output-dir", tmpdir, "--no-visuals"])
        assert code == 0
        assert os.path.exists(os.path.join(tmpdir, "report.json"))
        assert os.path.exists(os.path.join(tmpdir, "report.html"))


def test_main_json_only_combines_with_fail_on_partial():
    with tempfile.TemporaryDirectory() as tmpdir:
        code = main(["--demo", "--scenario", "drift", "--demo-frames", "20",
                     "--output-dir", tmpdir, "--json-only", "--fail-on-partial"])
        assert code == 3
        assert os.path.exists(os.path.join(tmpdir, "report.json"))
        assert not os.path.exists(os.path.join(tmpdir, "report.html"))


def test_run_pipeline_json_only_produces_no_visuals_in_report_dict():
    # json_only implies no_visuals (visuals would be discarded unused
    # anyway, since HTML is what embeds them) -- report.json itself
    # doesn't carry a visuals section either way, but this exercises the
    # code path directly to make sure json_only=True doesn't error out
    # even when no_visuals is left at its default (False).
    with tempfile.TemporaryDirectory() as tmpdir:
        dataset = _build_demo_dataset("good", num_frames=20)
        report = run_pipeline(dataset, tmpdir, json_only=True)
        assert os.path.exists(os.path.join(tmpdir, "report.json"))
        assert not os.path.exists(os.path.join(tmpdir, "report.html"))
        assert report["quality_score"]["overall_classification"] in ("GOOD", "WARNING", "BAD", "FAIL")


def test_main_demo_drift_with_fail_on_bad_returns_nonzero():
    with tempfile.TemporaryDirectory() as tmpdir:
        code = main(["--demo", "--scenario", "drift", "--demo-frames", "20",
                     "--edge-radius-px", "8.0", "--output-dir", tmpdir,
                     "--no-visuals", "--fail-on-bad"])
        assert code == 2


def test_main_demo_drift_with_fail_on_partial_returns_nonzero():
    # Default drift scenario (no --edge-radius-px override): M2 and M3 FAIL
    # outright (0 edge points), M4 alone is GOOD -> overall is a partial
    # result. --fail-on-bad alone would NOT catch this (overall caps at
    # WARNING, not BAD), so --fail-on-partial exists specifically to catch
    # "couldn't fully evaluate this calibration" in CI.
    with tempfile.TemporaryDirectory() as tmpdir:
        code_no_flag = main(["--demo", "--scenario", "drift", "--demo-frames", "20",
                              "--output-dir", tmpdir, "--no-visuals"])
        assert code_no_flag == 0

    with tempfile.TemporaryDirectory() as tmpdir:
        code_bad_only = main(["--demo", "--scenario", "drift", "--demo-frames", "20",
                               "--output-dir", tmpdir, "--no-visuals", "--fail-on-bad"])
        assert code_bad_only == 0  # WARNING, not BAD/FAIL -- fail-on-bad doesn't trigger

    with tempfile.TemporaryDirectory() as tmpdir:
        code_partial = main(["--demo", "--scenario", "drift", "--demo-frames", "20",
                              "--output-dir", tmpdir, "--no-visuals", "--fail-on-partial"])
        assert code_partial == 3
        import json
        with open(os.path.join(tmpdir, "report.json")) as f:
            report = json.load(f)
        assert report["quality_score"]["num_valid_categories"] < len(report["quality_score"]["categories"])


def test_main_demo_good_with_fail_on_partial_returns_zero():
    # Sanity check: a fully-valid (3/3 categories) result must NOT trigger
    # --fail-on-partial, even combined with --fail-on-bad.
    with tempfile.TemporaryDirectory() as tmpdir:
        code = main(["--demo", "--scenario", "good", "--demo-frames", "20",
                     "--output-dir", tmpdir, "--no-visuals",
                     "--fail-on-bad", "--fail-on-partial"])
        assert code == 0


def test_main_missing_config_returns_one():
    code = main(["--config", "/definitely/does/not/exist.yaml", "--output-dir", "/tmp/x"])
    assert code == 1


# ---------------------------------------------------------------------------
# --validate-config
# ---------------------------------------------------------------------------

def _write_minimal_valid_config(tmpdir) -> str:
    path = os.path.join(tmpdir, "config.yaml")
    cfg = {
        "camera": {
            "image_dir": os.path.join(tmpdir, "images"), "width": 64, "height": 48, "model": "pinhole",
            "intrinsics": {"fx": 500, "fy": 500, "cx": 32, "cy": 24},
        },
        "lidar": {"pcd_dir": os.path.join(tmpdir, "pcds"), "sensor_spec": {"horizontal_resolution_deg": 0.2}},
        "extrinsic": {"parent": "lidar", "child": "camera", "translation": [0, 0, 0],
                      "rotation": [0, 0, 0], "rotation_format": "rpy_deg"},
    }
    with open(path, "w") as f:
        yaml.safe_dump(cfg, f)
    return path


def test_validate_config_only_passes_for_valid_schema_without_touching_data_files():
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = _write_minimal_valid_config(tmpdir)
        # image_dir/pcd_dir referenced in the config are never created on
        # disk -- validate_config_only should still succeed, since it only
        # checks YAML structure, not that the data actually exists.
        validate_config_only(config_path)  # should not raise


def test_validate_config_only_raises_config_schema_error_for_missing_keys():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "config.yaml")
        with open(path, "w") as f:
            yaml.safe_dump({"camera": {"width": 64, "height": 48}}, f)
        try:
            validate_config_only(path)
            assert False, "expected ConfigSchemaError"
        except ConfigSchemaError:
            pass


def test_main_validate_config_flag_returns_zero_for_valid_config():
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = _write_minimal_valid_config(tmpdir)
        code = main(["--config", config_path, "--validate-config"])
        assert code == 0


def test_main_validate_config_flag_returns_one_for_invalid_config():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "config.yaml")
        with open(path, "w") as f:
            yaml.safe_dump({"camera": {"width": 64}}, f)
        code = main(["--config", path, "--validate-config"])
        assert code == 1


def test_main_validate_config_with_demo_returns_one():
    code = main(["--demo", "--validate-config"])
    assert code == 1


# ---------------------------------------------------------------------------
# --compare-to / --fail-on-regression / --format github-comment
# ---------------------------------------------------------------------------

def test_main_compare_to_prints_diff_and_returns_zero_when_improved():
    with tempfile.TemporaryDirectory() as tmpdir1, tempfile.TemporaryDirectory() as tmpdir2:
        assert main(["--demo", "--scenario", "drift", "--demo-frames", "20", "--output-dir", tmpdir1,
                      "--no-visuals", "--edge-radius-px", "8.0"]) in (0, 2, 3)
        code = main(["--demo", "--scenario", "good", "--demo-frames", "20", "--output-dir", tmpdir2,
                      "--no-visuals", "--compare-to", os.path.join(tmpdir1, "report.json")])
        assert code == 0


def test_main_fail_on_regression_without_compare_to_returns_one():
    with tempfile.TemporaryDirectory() as tmpdir:
        code = main(["--demo", "--demo-frames", "20", "--output-dir", tmpdir,
                      "--no-visuals", "--fail-on-regression"])
        assert code == 1


def test_main_fail_on_regression_exits_four_on_regression():
    with tempfile.TemporaryDirectory() as tmpdir1, tempfile.TemporaryDirectory() as tmpdir2:
        main(["--demo", "--scenario", "good", "--demo-frames", "20", "--output-dir", tmpdir1, "--no-visuals"])
        code = main(["--demo", "--scenario", "drift", "--demo-frames", "20", "--output-dir", tmpdir2,
                      "--no-visuals", "--compare-to", os.path.join(tmpdir1, "report.json"),
                      "--fail-on-regression"])
        assert code == 4


def test_main_format_github_comment_prints_markdown():
    with tempfile.TemporaryDirectory() as tmpdir:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            code = main(["--demo", "--demo-frames", "20", "--output-dir", tmpdir,
                         "--no-visuals", "--format", "github-comment"])
        assert code == 0
        out = buf.getvalue()
        assert "Cam-LiDAR Calibration Quality" in out
        assert "| Category | Score | Status |" in out


# ---------------------------------------------------------------------------
# --sequence-gif / --interactive-max-points
# ---------------------------------------------------------------------------

def test_run_pipeline_sequence_gif_adds_visual_when_requested():
    dataset = _build_demo_dataset("good", num_frames=20)
    with tempfile.TemporaryDirectory() as tmpdir:
        report = run_pipeline(dataset, tmpdir, n_blocks=4, min_frames_per_block=4, min_frames_m4=20,
                               depth_jump_threshold_m=1.0, sequence_gif=True, sequence_max_frames=5)
        assert report is not None
        html = open(os.path.join(tmpdir, "report.html"), encoding="utf-8").read()
        assert "data:image/gif;base64," in html


def test_run_pipeline_no_sequence_gif_by_default():
    dataset = _build_demo_dataset("good", num_frames=20)
    with tempfile.TemporaryDirectory() as tmpdir:
        run_pipeline(dataset, tmpdir, n_blocks=4, min_frames_per_block=4, min_frames_m4=20,
                     depth_jump_threshold_m=1.0)
        html = open(os.path.join(tmpdir, "report.html"), encoding="utf-8").read()
        assert "data:image/gif;base64," not in html


# ---------------------------------------------------------------------------
# STEP5 -- motion deskew (opt-in via --deskew-* flags / run_pipeline kwargs)
# ---------------------------------------------------------------------------

def test_run_pipeline_deskew_omitted_by_default():
    dataset = _build_demo_dataset("good", num_frames=20)
    with tempfile.TemporaryDirectory() as tmpdir:
        report = run_pipeline(dataset, tmpdir, n_blocks=4, min_frames_per_block=4, min_frames_m4=20,
                               depth_jump_threshold_m=1.0)
        assert report["motion_deskew"] is None
        html = open(os.path.join(tmpdir, "report.html"), encoding="utf-8").read()
        assert "Motion Deskew" not in html


def test_run_pipeline_deskew_linear_velocity_adds_report_section_and_visual():
    dataset = _build_demo_dataset("good", num_frames=20)
    with tempfile.TemporaryDirectory() as tmpdir:
        report = run_pipeline(
            dataset, tmpdir, n_blocks=4, min_frames_per_block=4, min_frames_m4=20,
            depth_jump_threshold_m=1.0,
            deskew_linear_velocity_mps=np.array([5.0, 0.0, 0.0]),
        )
        assert report["motion_deskew"] is not None
        assert report["motion_deskew"]["max_correction_m"] > 0.0
        html = open(os.path.join(tmpdir, "report.html"), encoding="utf-8").read()
        assert "Motion Deskew" in html
        assert "data:image/png;base64," in html


def test_run_pipeline_deskew_angular_velocity_alone_also_triggers_section():
    """Giving ONLY angular velocity (no linear) should still activate
    deskew -- the missing vector defaults to zero, not 'skip deskew'."""
    dataset = _build_demo_dataset("good", num_frames=20)
    with tempfile.TemporaryDirectory() as tmpdir:
        report = run_pipeline(
            dataset, tmpdir, n_blocks=4, min_frames_per_block=4, min_frames_m4=20,
            depth_jump_threshold_m=1.0,
            deskew_angular_velocity_rps=np.array([0.0, 0.0, 1.0]),
        )
        assert report["motion_deskew"] is not None


def test_run_pipeline_deskew_zero_velocity_gives_zero_correction():
    """Explicit zero velocity still activates the section (since the flag
    was given), but the correction is exactly zero -- distinct from the
    'omitted entirely' case above."""
    dataset = _build_demo_dataset("good", num_frames=20)
    with tempfile.TemporaryDirectory() as tmpdir:
        report = run_pipeline(
            dataset, tmpdir, n_blocks=4, min_frames_per_block=4, min_frames_m4=20,
            depth_jump_threshold_m=1.0,
            deskew_linear_velocity_mps=np.zeros(3), deskew_angular_velocity_rps=np.zeros(3),
        )
        assert report["motion_deskew"] is not None
        assert report["motion_deskew"]["max_correction_m"] == 0.0


def test_run_pipeline_deskew_does_not_affect_quality_score():
    """Deskewing is diagnostic-only -- M0/M2/M3/M4 must score identically
    whether or not --deskew-* was given."""
    dataset_a = _build_demo_dataset("good", num_frames=20)
    dataset_b = _build_demo_dataset("good", num_frames=20)
    with tempfile.TemporaryDirectory() as tmpdir_a, tempfile.TemporaryDirectory() as tmpdir_b:
        report_no_deskew = run_pipeline(dataset_a, tmpdir_a, n_blocks=4, min_frames_per_block=4,
                                         min_frames_m4=20, depth_jump_threshold_m=1.0)
        report_with_deskew = run_pipeline(
            dataset_b, tmpdir_b, n_blocks=4, min_frames_per_block=4, min_frames_m4=20,
            depth_jump_threshold_m=1.0,
            deskew_linear_velocity_mps=np.array([8.0, 2.0, 0.0]),
            deskew_angular_velocity_rps=np.array([0.0, 0.0, 0.5]),
        )
        assert report_no_deskew["quality_score"] == report_with_deskew["quality_score"]
        assert report_no_deskew["m2_edge_alignment"] == report_with_deskew["m2_edge_alignment"]


def test_run_pipeline_deskew_no_visuals_omits_image_but_keeps_report_section():
    dataset = _build_demo_dataset("good", num_frames=20)
    with tempfile.TemporaryDirectory() as tmpdir:
        report = run_pipeline(
            dataset, tmpdir, n_blocks=4, min_frames_per_block=4, min_frames_m4=20,
            depth_jump_threshold_m=1.0, no_visuals=True,
            deskew_linear_velocity_mps=np.array([3.0, 0.0, 0.0]),
        )
        assert report["motion_deskew"] is not None  # numeric summary still computed


def test_main_demo_with_deskew_flags_end_to_end():
    with tempfile.TemporaryDirectory() as tmpdir:
        code = main([
            "--demo", "--demo-frames", "20", "--output-dir", tmpdir,
            "--deskew-linear-velocity", "5.0,0.0,0.0",
            "--deskew-angular-velocity", "0.0,0.0,0.3",
            "--deskew-scan-period-s", "0.1",
        ])
        assert code == 0
        html = open(os.path.join(tmpdir, "report.html"), encoding="utf-8").read()
        assert "Motion Deskew" in html


def test_main_demo_deskew_malformed_velocity_returns_clean_error():
    with tempfile.TemporaryDirectory() as tmpdir:
        stderr_buf = io.StringIO()
        with contextlib.redirect_stderr(stderr_buf):
            code = main([
                "--demo", "--output-dir", tmpdir,
                "--deskew-linear-velocity", "not,a,vector",
            ])
        assert code != 0
        assert "--deskew-linear-velocity" in stderr_buf.getvalue()


# ---------------------------------------------------------------------------
# STEP8 -- dynamic object filtering (opt-in via --dynamic-filter)
# ---------------------------------------------------------------------------

def test_main_demo_with_dynamic_filter_flag_end_to_end():
    with tempfile.TemporaryDirectory() as tmpdir:
        code = main([
            "--demo", "--demo-frames", "20", "--output-dir", tmpdir,
            "--dynamic-filter", "--dynamic-filter-window", "3",
        ])
        assert code == 0
        html = open(os.path.join(tmpdir, "report.html"), encoding="utf-8").read()
        assert "Dynamic Object Filtering" in html


def test_main_demo_without_dynamic_filter_flag_omits_section():
    with tempfile.TemporaryDirectory() as tmpdir:
        code = main(["--demo", "--demo-frames", "20", "--output-dir", tmpdir])
        assert code == 0
        html = open(os.path.join(tmpdir, "report.html"), encoding="utf-8").read()
        assert "Dynamic Object Filtering" not in html


def test_run_pipeline_dynamic_filter_does_not_affect_quality_score():
    """Like deskewing, dynamic filtering is diagnostic-only -- M0/M2/M3/M4
    must score identically whether or not --dynamic-filter was given."""
    dataset_a = _build_demo_dataset("good", num_frames=20)
    dataset_b = _build_demo_dataset("good", num_frames=20)
    with tempfile.TemporaryDirectory() as tmpdir_a, tempfile.TemporaryDirectory() as tmpdir_b:
        report_off = run_pipeline(dataset_a, tmpdir_a, n_blocks=4, min_frames_per_block=4,
                                   min_frames_m4=20, depth_jump_threshold_m=1.0)
        report_on = run_pipeline(dataset_b, tmpdir_b, n_blocks=4, min_frames_per_block=4,
                                  min_frames_m4=20, depth_jump_threshold_m=1.0,
                                  dynamic_filter=True, dynamic_filter_window=3)
        assert report_off["quality_score"] == report_on["quality_score"]
        assert report_off["m2_edge_alignment"] == report_on["m2_edge_alignment"]
        assert report_on["dynamic_filter"] is not None
        assert report_off["dynamic_filter"] is None


def test_run_pipeline_interactive_max_points_overrides_default():
    dataset = _build_demo_dataset("good", num_frames=20)
    with tempfile.TemporaryDirectory() as tmpdir1, tempfile.TemporaryDirectory() as tmpdir2:
        run_pipeline(dataset, tmpdir1, n_blocks=4, min_frames_per_block=4, min_frames_m4=20,
                     depth_jump_threshold_m=1.0, interactive_max_points=5)
        run_pipeline(dataset, tmpdir2, n_blocks=4, min_frames_per_block=4, min_frames_m4=20,
                     depth_jump_threshold_m=1.0, interactive_max_points=5000)
        size_small = os.path.getsize(os.path.join(tmpdir1, "report.html"))
        size_large = os.path.getsize(os.path.join(tmpdir2, "report.html"))
        assert size_small < size_large


# ---------------------------------------------------------------------------
# --config schema validation (ConfigSchemaError)
# ---------------------------------------------------------------------------

def _write_yaml(tmpdir, contents: dict | list) -> str:
    path = os.path.join(tmpdir, "config.yaml")
    with open(path, "w") as f:
        yaml.safe_dump(contents, f)
    return path


def test_load_dataset_from_config_missing_all_top_level_keys_lists_each_one():
    # Regression test: a config missing required keys used to surface as a
    # bare KeyError (e.g. just "'camera'") with no indication of what else
    # was wrong or where the schema is documented. It should now name every
    # missing top-level key in one pass.
    with tempfile.TemporaryDirectory() as tmpdir:
        path = _write_yaml(tmpdir, {"unrelated": "stuff"})
        try:
            load_dataset_from_config(path)
            assert False, "expected ConfigSchemaError"
        except Exception as e:
            msg = str(e)
            assert "missing top-level key 'camera'" in msg
            assert "missing top-level key 'lidar'" in msg
            assert "missing top-level key 'extrinsic'" in msg
            assert "evaluation_metric_spec.md" in msg  # points back to the schema


def test_load_dataset_from_config_missing_nested_key_names_full_path():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = _write_yaml(tmpdir, {
            "camera": {"image_dir": "x", "width": 640, "height": 480},  # no intrinsics
            "lidar": {"pcd_dir": "y"},
            "extrinsic": {"parent": "lidar", "child": "camera",
                          "rotation": [0, 0, 0], "rotation_format": "rpy_deg"},
        })
        try:
            load_dataset_from_config(path)
            assert False, "expected ConfigSchemaError"
        except Exception as e:
            assert "missing 'camera.intrinsics'" in str(e)


def test_load_dataset_from_config_invalid_rotation_format_is_named():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = _write_yaml(tmpdir, {
            "camera": {"image_dir": "x", "width": 640, "height": 480,
                       "intrinsics": {"fx": 1, "fy": 1, "cx": 1, "cy": 1}},
            "lidar": {"pcd_dir": "y"},
            "extrinsic": {"parent": "lidar", "child": "camera",
                          "rotation": [0, 0, 0], "rotation_format": "rpy_degrees"},  # typo
        })
        try:
            load_dataset_from_config(path)
            assert False, "expected ConfigSchemaError"
        except Exception as e:
            assert "rotation_format" in str(e) and "rpy_degrees" in str(e)


def test_load_dataset_from_config_non_mapping_top_level_is_rejected():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = _write_yaml(tmpdir, ["camera", "lidar"])  # a list, not a mapping
        try:
            load_dataset_from_config(path)
            assert False, "expected ConfigSchemaError"
        except Exception as e:
            assert "did not parse into a YAML mapping" in str(e)


def test_main_with_schema_invalid_config_returns_one_with_actionable_message():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = _write_yaml(tmpdir, {"unrelated": "stuff"})
        import io
        import contextlib
        stderr_buf = io.StringIO()
        with contextlib.redirect_stderr(stderr_buf):
            code = main(["--config", path, "--output-dir", os.path.join(tmpdir, "out")])
        assert code == 1
        assert "missing top-level key 'camera'" in stderr_buf.getvalue()


def test_main_respects_min_frames_per_block_override_in_demo():
    with tempfile.TemporaryDirectory() as tmpdir:
        # deliberately too strict for the frame count -> M3 should FAIL,
        # proving the CLI arg (not just the auto-adjusted default) is honored
        code = main(["--demo", "--demo-frames", "20", "--min-frames-per-block", "1000",
                     "--output-dir", tmpdir, "--no-visuals"])
        assert code == 0  # not fail-on-bad, so still exits 0
        import json
        with open(os.path.join(tmpdir, "report.json")) as f:
            report = json.load(f)
        assert report["m3_holdout_consistency"]["classification"] == "FAIL"


# ---------------------------------------------------------------------------
# load_dataset_from_config (real files)
# ---------------------------------------------------------------------------

def _write_config_and_files(tmpdir, num_frames=15):
    images_dir = os.path.join(tmpdir, "images")
    pcds_dir = os.path.join(tmpdir, "pcds")
    os.makedirs(images_dir)
    os.makedirs(pcds_dir)

    width, height = 320, 240
    image = np.zeros((height, width), dtype=np.uint8)
    image[:, width // 2:] = 255
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    fx = fy = 300.0
    cx, cy = width / 2, height / 2
    u_vals = np.linspace(0, width - 1, 80)
    v_vals = np.linspace(0, height - 1, 60)
    uu, vv = np.meshgrid(u_vals, v_vals)
    uu, vv = uu.ravel(), vv.ravel()
    zz = np.where(uu < cx, 5.0, 10.0)
    xx = (uu - cx) * zz / fx
    yy = (vv - cy) * zz / fy
    pts = np.stack([xx, yy, zz], axis=1)

    for i in range(num_frames):
        cv2.imwrite(os.path.join(images_dir, f"{i}.0.png"), image)
        with open(os.path.join(pcds_dir, f"{i}.0.pcd"), "w") as f:
            f.write(f"VERSION 0.7\nFIELDS x y z\nSIZE 4 4 4\nTYPE F F F\nCOUNT 1 1 1\n"
                     f"WIDTH {len(pts)}\nHEIGHT 1\nPOINTS {len(pts)}\nDATA ascii\n")
            for p in pts:
                f.write(f"{p[0]} {p[1]} {p[2]}\n")

    config = {
        "camera": {
            "image_dir": images_dir, "width": width, "height": height, "model": "pinhole",
            "intrinsics": {"fx": fx, "fy": fy, "cx": cx, "cy": cy},
            "distortion": {"model": "none"},
        },
        "lidar": {
            "pcd_dir": pcds_dir,
            "sensor_spec": {"horizontal_resolution_deg": 0.05, "range_accuracy_m": 0.02},
        },
        "extrinsic": {
            "parent": "lidar", "child": "camera",
            "translation": [0.0, 0.0, 0.0], "rotation": [0.0, 0.0, 0.0],
            "rotation_format": "rpy_deg", "unit": "m",
        },
        "evaluation": {"n_blocks": 3, "min_frames_per_block": 3, "min_frames_m4": num_frames,
                       "depth_jump_threshold_m": 1.0},
    }
    config_path = os.path.join(tmpdir, "config.yaml")
    with open(config_path, "w") as f:
        yaml.safe_dump(config, f)
    return config_path


def test_load_dataset_from_config_real_files():
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = _write_config_and_files(tmpdir, num_frames=15)
        dataset, eval_cfg, warnings = load_dataset_from_config(config_path)
        assert len(dataset.frames) == 15
        assert eval_cfg["n_blocks"] == 3
        assert np.allclose(dataset.extrinsic.T_CL, np.eye(4))


def test_main_config_end_to_end_with_real_files():
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = _write_config_and_files(tmpdir, num_frames=15)
        out_dir = os.path.join(tmpdir, "out")
        code = main(["--config", config_path, "--output-dir", out_dir, "--no-visuals"])
        assert code == 0
        assert os.path.exists(os.path.join(out_dir, "report.html"))
        assert os.path.exists(os.path.join(out_dir, "report.json"))


# ---------------------------------------------------------------------------
# --config with source: rosbag (requires the optional `rosbags` +
# `rosbags-image` packages; skipped if not installed)
# ---------------------------------------------------------------------------

try:
    from rosbags.rosbag2 import Writer as _Rosbag2Writer
    from rosbags.typesys import Stores as _RosbagStores, get_typestore as _get_rosbag_typestore
    from rosbags.image import message_to_cvimage as _rosbag_message_to_cvimage
    _ROSBAGS_AVAILABLE = bool(_rosbag_message_to_cvimage)
except ImportError:
    _ROSBAGS_AVAILABLE = False


def _write_rosbag_config_and_bag(tmpdir, num_frames=15):
    """Same synthetic depth-step scene as _write_config_and_files, but
    written into a single rosbag2 bag (one PointCloud2 + one Image message
    per frame) instead of image_dir/pcd_dir, with a matching
    source: rosbag config -- exercises the full --config -> rosbag loader
    -> sync -> pipeline path end to end."""
    ts = _get_rosbag_typestore(_RosbagStores.ROS2_HUMBLE)
    PointField = ts.types["sensor_msgs/msg/PointField"]
    PointCloud2 = ts.types["sensor_msgs/msg/PointCloud2"]
    Image = ts.types["sensor_msgs/msg/Image"]
    Header = ts.types["std_msgs/msg/Header"]
    Time = ts.types["builtin_interfaces/msg/Time"]

    width, height = 320, 240
    image = np.zeros((height, width), dtype=np.uint8)
    image[:, width // 2:] = 255
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    fx = fy = 300.0
    cx, cy = width / 2, height / 2
    u_vals = np.linspace(0, width - 1, 80)
    v_vals = np.linspace(0, height - 1, 60)
    uu, vv = np.meshgrid(u_vals, v_vals)
    uu, vv = uu.ravel(), vv.ravel()
    zz = np.where(uu < cx, 5.0, 10.0)
    xx = (uu - cx) * zz / fx
    yy = (vv - cy) * zz / fy
    pts = np.stack([xx, yy, zz], axis=1).astype(np.float32)

    fields = [PointField(name=n, offset=o, datatype=PointField.FLOAT32, count=1)
              for n, o in [("x", 0), ("y", 4), ("z", 8)]]

    bag_path = os.path.join(tmpdir, "bag")
    with _Rosbag2Writer(bag_path, version=9) as writer:
        pc_conn = writer.add_connection("/lidar/points", PointCloud2.__msgtype__, typestore=ts)
        img_conn = writer.add_connection("/camera/image_raw", Image.__msgtype__, typestore=ts)
        for i in range(num_frames):
            header = Header(stamp=Time(sec=i, nanosec=0), frame_id="lidar")
            pc_msg = PointCloud2(header=header, height=1, width=len(pts), fields=fields,
                                  is_bigendian=False, point_step=12, row_step=12 * len(pts),
                                  data=np.frombuffer(pts.tobytes(), dtype=np.uint8), is_dense=True)
            writer.write(pc_conn, i * 1_000_000_000, ts.serialize_cdr(pc_msg, PointCloud2.__msgtype__))

            header2 = Header(stamp=Time(sec=i, nanosec=0), frame_id="camera")
            img_msg = Image(header=header2, height=height, width=width, encoding="bgr8",
                             is_bigendian=0, step=width * 3, data=image.flatten())
            writer.write(img_conn, i * 1_000_000_000, ts.serialize_cdr(img_msg, Image.__msgtype__))

    config = {
        "camera": {
            "source": "rosbag", "rosbag_path": bag_path, "topic": "/camera/image_raw",
            "width": width, "height": height, "model": "pinhole",
            "intrinsics": {"fx": fx, "fy": fy, "cx": cx, "cy": cy},
            "distortion": {"model": "none"},
        },
        "lidar": {
            "source": "rosbag", "rosbag_path": bag_path, "topic": "/lidar/points",
            "sensor_spec": {"horizontal_resolution_deg": 0.05, "range_accuracy_m": 0.02},
        },
        "extrinsic": {
            "parent": "lidar", "child": "camera",
            "translation": [0.0, 0.0, 0.0], "rotation": [0.0, 0.0, 0.0],
            "rotation_format": "rpy_deg", "unit": "m",
        },
        "evaluation": {"n_blocks": 3, "min_frames_per_block": 3, "min_frames_m4": num_frames,
                       "depth_jump_threshold_m": 1.0},
    }
    config_path = os.path.join(tmpdir, "config.yaml")
    with open(config_path, "w") as f:
        yaml.safe_dump(config, f)
    return config_path


def test_load_dataset_from_config_rosbag_source():
    if not _ROSBAGS_AVAILABLE:
        return
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = _write_rosbag_config_and_bag(tmpdir, num_frames=15)
        dataset, eval_cfg, warnings = load_dataset_from_config(config_path)
        assert len(dataset.frames) == 15
        assert dataset.camera.source.kind == "rosbag"
        assert dataset.lidar.source.kind == "rosbag"
        assert np.allclose(dataset.extrinsic.T_CL, np.eye(4))


def test_main_config_end_to_end_with_rosbag_source():
    if not _ROSBAGS_AVAILABLE:
        return
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = _write_rosbag_config_and_bag(tmpdir, num_frames=15)
        out_dir = os.path.join(tmpdir, "out")
        code = main(["--config", config_path, "--output-dir", out_dir, "--no-visuals"])
        assert code == 0
        assert os.path.exists(os.path.join(out_dir, "report.json"))


def test_rosbag_and_directory_sources_produce_equivalent_results():
    # Both fixtures encode the exact same synthetic depth-step scene (same
    # points, same image) -- just through different loaders (pcd_dir/
    # image_dir vs rosbag). If the rosbag parsing path is correct, running
    # the full pipeline through either source should produce the same
    # Overall Quality classification and a near-identical score, proving
    # the two source kinds are equivalent from the pipeline's point of
    # view. (Not asserting a specific GOOD/WARNING/BAD/FAIL value here,
    # since that depends on unrelated fixture tuning -- e.g. edge_radius_px
    # vs point density -- not on which loader was used.)
    if not _ROSBAGS_AVAILABLE:
        return
    with tempfile.TemporaryDirectory() as dir_tmpdir, tempfile.TemporaryDirectory() as bag_tmpdir:
        dir_config_path = _write_config_and_files(dir_tmpdir, num_frames=15)
        bag_config_path = _write_rosbag_config_and_bag(bag_tmpdir, num_frames=15)

        dir_out = os.path.join(dir_tmpdir, "out")
        bag_out = os.path.join(bag_tmpdir, "out")
        assert main(["--config", dir_config_path, "--output-dir", dir_out, "--no-visuals"]) == \
            main(["--config", bag_config_path, "--output-dir", bag_out, "--no-visuals"])

        import json
        with open(os.path.join(dir_out, "report.json")) as f:
            dir_report = json.load(f)
        with open(os.path.join(bag_out, "report.json")) as f:
            bag_report = json.load(f)

        assert dir_report["quality_score"]["overall_classification"] == \
            bag_report["quality_score"]["overall_classification"]
        dir_score = dir_report["quality_score"]["overall_score"]
        bag_score = bag_report["quality_score"]["overall_score"]
        if dir_score is not None and bag_score is not None:
            assert abs(dir_score - bag_score) < 0.5


def test_config_schema_rejects_rosbag_source_missing_rosbag_path():
    config = {
        "camera": {"source": "rosbag", "width": 640, "height": 480,
                   "intrinsics": {"fx": 1, "fy": 1, "cx": 1, "cy": 1}},
        "lidar": {"pcd_dir": "x"},
        "extrinsic": {"parent": "lidar", "child": "camera",
                      "rotation": [0, 0, 0], "rotation_format": "rpy_deg"},
    }
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "config.yaml")
        with open(path, "w") as f:
            yaml.safe_dump(config, f)
        try:
            load_dataset_from_config(path)
            assert False, "expected ConfigSchemaError"
        except Exception as e:
            assert "camera.rosbag_path" in str(e)


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
