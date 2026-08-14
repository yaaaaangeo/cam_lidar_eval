import sys
import os
import tempfile
import shutil

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import cv2
import yaml

from app.cli import (
    build_arg_parser, main, _build_demo_dataset, _parse_weights,
    load_dataset_from_config, run_pipeline,
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


def test_run_pipeline_raises_on_empty_dataset():
    from input.dataset import EvaluationDataset, SyncConfig
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


def test_main_demo_drift_with_fail_on_bad_returns_nonzero():
    with tempfile.TemporaryDirectory() as tmpdir:
        code = main(["--demo", "--scenario", "drift", "--demo-frames", "20",
                     "--edge-radius-px", "8.0", "--output-dir", tmpdir,
                     "--no-visuals", "--fail-on-bad"])
        assert code == 2


def test_main_missing_config_returns_one():
    code = main(["--config", "/definitely/does/not/exist.yaml", "--output-dir", "/tmp/x"])
    assert code == 1


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
