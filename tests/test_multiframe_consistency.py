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


def test_multiframe_consistency_detects_single_outlier_frame_multiplier_method():
    # 39 perfectly consistent frames + 1 frame with a large drift injected
    # -> should be flagged as an outlier, and increase STD noticeably.
    # Measured with this synthetic scene: edge_radius_px=8 gives a uniform
    # (zero-drift) baseline of ~2.42px and a drift=0.1 frame of ~10.96px
    # (~4.5x baseline) -- outlier_multiplier is lowered to 4.0 (a legitimate
    # tunable parameter) so this clears the bar; the default 5.0 is tuned for
    # real sensor noise, not this synthetic point-grid's specific geometry.
    # outlier_method explicitly set to "multiplier" (STEP10's new default is
    # "hampel" -- see test_multiframe_consistency_detects_single_outlier_frame_hampel_default
    # for the same scenario under the default method) so this test still
    # actually exercises the multiplier rule its parameters imply.
    drifts = [0.0] * 39 + [0.1]
    dataset = _make_dataset(drifts)
    result = evaluate_multiframe_consistency(
        dataset, lidar_spec=_make_lidar_spec(), min_frames=30,
        outlier_method="multiplier", outlier_multiplier=4.0,
        edge_alignment_kwargs={"depth_jump_threshold_m": 1.0, "edge_radius_px": 8.0},
    )
    assert result.classification != "FAIL", result.warnings
    assert result.num_outlier_frames >= 1
    assert 39 in result.outlier_frame_indices


def test_multiframe_consistency_detects_single_outlier_frame_hampel_default():
    """Same scenario as the multiplier-method test above, but under
    STEP10's new default (outlier_method left unset -> 'hampel') -- the
    single drifted frame should still be caught without needing any
    tuned multiplier parameter at all."""
    drifts = [0.0] * 39 + [0.1]
    dataset = _make_dataset(drifts)
    result = evaluate_multiframe_consistency(
        dataset, lidar_spec=_make_lidar_spec(), min_frames=30,
        edge_alignment_kwargs={"depth_jump_threshold_m": 1.0, "edge_radius_px": 8.0},
    )
    assert result.outlier_method == "hampel"
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


# ---------------------------------------------------------------------------
# STEP10 -- robust statistics (MAD/IQR/Hampel/robust z-score) + separated
# valid/failure/outlier ratios
# ---------------------------------------------------------------------------

def test_multiframe_consistency_reports_mad_iqr_and_robust_z_scores():
    # Small per-frame drift variation (not all-identical) so MAD is
    # nonzero and robust z-scores are well-defined -- an all-but-one-
    # identical scene (MAD=0) is a genuine degenerate case for MAD-based
    # stats (see test_multiframe_consistency_mad_zero_degenerate_case),
    # not representative of what this test wants to check.
    rng_drifts = [0.001 * i for i in range(39)] + [0.1]
    dataset = _make_dataset(rng_drifts)
    result = evaluate_multiframe_consistency(
        dataset, lidar_spec=_make_lidar_spec(), min_frames=30,
        edge_alignment_kwargs={"depth_jump_threshold_m": 1.0, "edge_radius_px": 8.0},
    )
    assert np.isfinite(result.mad_px) and result.mad_px > 0
    assert np.isfinite(result.iqr_px) and result.iqr_px >= 0
    assert result.q3_px >= result.q1_px
    # the drifted frame's robust z-score should be the largest in magnitude
    valid_frames = [f for f in result.frame_results if f.classification != "FAIL"]
    z_scores = {f.frame_index: f.robust_z_score for f in valid_frames}
    assert max(z_scores, key=lambda idx: abs(z_scores[idx])) == 39


def test_multiframe_consistency_mad_zero_degenerate_case_gives_nan_z_scores():
    """When >=half the frames agree EXACTLY (a real property of MAD, not
    a bug here), MAD is 0 and per-frame robust z-scores are mathematically
    undefined -- they should come back as NaN rather than a divide-by-zero
    inf or a silently wrong number, and the Hampel outlier flag should
    fall back to its documented absolute-epsilon path instead of crashing."""
    dataset = _make_dataset([0.0] * 39 + [0.1])  # 39 IDENTICAL frames + 1 outlier -> MAD=0
    result = evaluate_multiframe_consistency(
        dataset, lidar_spec=_make_lidar_spec(), min_frames=30,
        edge_alignment_kwargs={"depth_jump_threshold_m": 1.0, "edge_radius_px": 8.0},
    )
    assert result.mad_px == 0.0
    valid_frames = [f for f in result.frame_results if f.classification != "FAIL"]
    assert all(np.isnan(f.robust_z_score) for f in valid_frames)
    assert any("MAD" in w for w in result.warnings)
    # outlier detection still works via the documented absolute-epsilon fallback
    assert 39 in result.outlier_frame_indices


def test_multiframe_consistency_valid_failure_outlier_ratios_sum_correctly():
    dataset = _make_dataset([0.0] * 39 + [0.1])
    result = evaluate_multiframe_consistency(
        dataset, lidar_spec=_make_lidar_spec(), min_frames=30,
        edge_alignment_kwargs={"depth_jump_threshold_m": 1.0, "edge_radius_px": 8.0},
    )
    assert np.isclose(result.valid_ratio + result.failure_ratio, 1.0)
    assert np.isclose(result.valid_ratio, result.num_valid_frames / result.num_frames_total)
    assert np.isclose(result.failure_ratio, result.num_failed_frames / result.num_frames_total)
    assert np.isclose(result.outlier_ratio, result.num_outlier_frames / result.num_valid_frames)


def test_multiframe_consistency_zero_failures_gives_full_valid_ratio():
    dataset = _make_dataset([0.0] * 40)
    result = evaluate_multiframe_consistency(
        dataset, lidar_spec=_make_lidar_spec(), min_frames=30,
        edge_alignment_kwargs={"depth_jump_threshold_m": 1.0},
    )
    assert result.valid_ratio == 1.0
    assert result.failure_ratio == 0.0
    assert result.outlier_ratio == 0.0


def test_multiframe_consistency_iqr_outlier_method():
    dataset = _make_dataset([0.0] * 39 + [0.1])
    result = evaluate_multiframe_consistency(
        dataset, lidar_spec=_make_lidar_spec(), min_frames=30, outlier_method="iqr",
        edge_alignment_kwargs={"depth_jump_threshold_m": 1.0, "edge_radius_px": 8.0},
    )
    assert result.outlier_method == "iqr"
    assert 39 in result.outlier_frame_indices


def test_multiframe_consistency_rejects_unknown_outlier_method():
    dataset = _make_dataset([0.0] * 30)
    try:
        evaluate_multiframe_consistency(
            dataset, lidar_spec=_make_lidar_spec(), min_frames=30, outlier_method="nonsense",
            edge_alignment_kwargs={"depth_jump_threshold_m": 1.0},
        )
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_compute_robust_stats_matches_manual_calculation():
    from evaluation.multiframe_consistency import compute_robust_stats
    values = np.array([1.0, 2.0, 3.0, 4.0, 100.0])  # one big outlier
    stats = compute_robust_stats(values)
    assert stats["median"] == 3.0
    manual_mad_raw = np.median(np.abs(values - 3.0))  # median([2,1,0,1,97]) = 1.0
    assert np.isclose(stats["mad"], 1.4826 * manual_mad_raw)
    assert np.isclose(stats["q1"], np.percentile(values, 25))
    assert np.isclose(stats["q3"], np.percentile(values, 75))


def test_compute_robust_stats_empty_input():
    from evaluation.multiframe_consistency import compute_robust_stats
    stats = compute_robust_stats(np.zeros(0))
    assert all(np.isnan(v) for v in stats.values())


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
