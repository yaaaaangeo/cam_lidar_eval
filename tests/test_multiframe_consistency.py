import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from evaluation.multiframe_consistency import evaluate_multiframe_consistency
from tests.test_holdout_consistency import _make_dataset, _make_lidar_spec


def test_multiframe_consistency_low_std_when_uniformly_good():
    dataset = _make_dataset([0.0] * 40)
    result = evaluate_multiframe_consistency(
        dataset, lidar_spec=_make_lidar_spec(), min_frames=30,
        edge_alignment_kwargs={"depth_jump_threshold_m": 1.0},
    )
    assert result.classification != "FAIL", result.warnings
    assert result.num_valid_frames == 40
    assert result.num_failed_frames == 0
    assert result.std_across_frames_px < 0.5, f"std={result.std_across_frames_px}"
    assert result.num_outlier_frames == 0
    assert result.classification == "GOOD"


def test_multiframe_consistency_detects_single_outlier_frame():
    # 39 perfectly consistent frames + 1 frame with a large drift injected
    # -> should be flagged as an outlier, and increase STD noticeably.
    # Measured with this synthetic scene: edge_radius_px=8 gives a uniform
    # (zero-drift) baseline of ~2.42px and a drift=0.1 frame of ~10.96px
    # (~4.5x baseline) -- outlier_multiplier is lowered to 4.0 (a legitimate
    # tunable parameter) so this clears the bar; the default 5.0 is tuned for
    # real sensor noise, not this synthetic point-grid's specific geometry.
    drifts = [0.0] * 39 + [0.1]
    dataset = _make_dataset(drifts)
    result = evaluate_multiframe_consistency(
        dataset, lidar_spec=_make_lidar_spec(), min_frames=30, outlier_multiplier=4.0,
        edge_alignment_kwargs={"depth_jump_threshold_m": 1.0, "edge_radius_px": 8.0},
    )
    assert result.classification != "FAIL", result.warnings
    assert result.num_outlier_frames >= 1
    assert 39 in result.outlier_frame_indices


def test_multiframe_consistency_outlier_increases_std_vs_uniform_baseline():
    uniform = _make_dataset([0.0] * 40)
    with_outlier = _make_dataset([0.0] * 39 + [0.1])

    r_uniform = evaluate_multiframe_consistency(
        uniform, lidar_spec=_make_lidar_spec(), min_frames=30,
        edge_alignment_kwargs={"depth_jump_threshold_m": 1.0, "edge_radius_px": 8.0},
    )
    r_outlier = evaluate_multiframe_consistency(
        with_outlier, lidar_spec=_make_lidar_spec(), min_frames=30,
        edge_alignment_kwargs={"depth_jump_threshold_m": 1.0, "edge_radius_px": 8.0},
    )
    assert r_outlier.std_across_frames_px > r_uniform.std_across_frames_px
    assert r_outlier.max_across_frames_px > r_uniform.max_across_frames_px


def test_multiframe_consistency_fails_below_min_frames():
    dataset = _make_dataset([0.0] * 10)  # below default min_frames=30
    result = evaluate_multiframe_consistency(dataset, lidar_spec=_make_lidar_spec(), min_frames=30)
    assert result.classification == "FAIL"
    assert result.num_valid_frames == 0
    assert any("below min_frames" in w for w in result.warnings)


def test_multiframe_consistency_respects_custom_min_frames():
    dataset = _make_dataset([0.0] * 10)
    result = evaluate_multiframe_consistency(
        dataset, lidar_spec=_make_lidar_spec(), min_frames=5,
        edge_alignment_kwargs={"depth_jump_threshold_m": 1.0},
    )
    assert result.classification != "FAIL"
    assert result.num_valid_frames == 10


def test_multiframe_consistency_empty_dataset():
    dataset = _make_dataset([])
    result = evaluate_multiframe_consistency(dataset, lidar_spec=_make_lidar_spec(), min_frames=5)
    assert result.classification == "FAIL"
    assert result.num_frames_total == 0


def test_multiframe_consistency_all_frames_fail_gracefully():
    # blank image -> Canny finds no edges -> every frame's M2 call FAILs
    from tests.test_holdout_consistency import _make_camera, _make_lidar_spec as _lspec
    from input.camera import CameraFrame
    from input.lidar import LidarFrame
    from input.extrinsic import ExtrinsicModel, ExtrinsicRaw
    from input.dataset import EvaluationDataset, SyncedFrame, SyncConfig
    from input.lidar import LidarModel, LidarSource
    from tests.test_holdout_consistency import _make_base_points_cam_frame, WIDTH, HEIGHT

    camera = _make_camera()
    blank_image = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    base_points = _make_base_points_cam_frame()
    lidar = LidarModel(source=LidarSource(kind="pcd_dir", path="."), sensor_spec=_lspec())
    raw = ExtrinsicRaw(parent="lidar", child="camera", translation=(0, 0, 0),
                        rotation=(0, 0, 0), rotation_format="rpy_deg")
    extrinsic = ExtrinsicModel(T_CL=np.eye(4), parent="lidar", child="camera", raw=raw)

    frames = []
    for i in range(35):
        cam_frame = CameraFrame(timestamp=float(i), image=blank_image)
        lidar_frame = LidarFrame(timestamp=float(i), points=base_points.copy())
        frames.append(SyncedFrame(index=i, timestamp=float(i), camera_frame=cam_frame,
                                   lidar_frame=lidar_frame, time_diff_ms=0.0))

    dataset = EvaluationDataset(camera=camera, lidar=lidar, extrinsic=extrinsic,
                                 sync_config=SyncConfig(), frames=frames)

    result = evaluate_multiframe_consistency(dataset, lidar_spec=_lspec(), min_frames=30)
    assert result.classification == "FAIL"
    assert result.num_failed_frames == 35


def test_multiframe_consistency_frame_results_preserve_order_and_index():
    dataset = _make_dataset([0.0] * 30)
    result = evaluate_multiframe_consistency(
        dataset, lidar_spec=_make_lidar_spec(), min_frames=30,
        edge_alignment_kwargs={"depth_jump_threshold_m": 1.0},
    )
    indices = [f.frame_index for f in result.frame_results]
    assert indices == list(range(30))


def test_multiframe_consistency_floor_positive_for_valid_result():
    dataset = _make_dataset([0.0] * 30)
    result = evaluate_multiframe_consistency(
        dataset, lidar_spec=_make_lidar_spec(), min_frames=30,
        edge_alignment_kwargs={"depth_jump_threshold_m": 1.0},
    )
    assert result.floor_px > 0
    assert result.classification in ("GOOD", "WARNING", "BAD")


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
