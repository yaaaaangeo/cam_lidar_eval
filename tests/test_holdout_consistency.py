import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import cv2

from evaluation.holdout_consistency import (
    evaluate_holdout_consistency, MIN_VALID_BLOCKS, diagnose_instability, BlockResult,
)
from input.camera import CameraModel, CameraIntrinsics, CameraDistortion, CameraSource, CameraFrame
from input.lidar import LidarSensorSpec, LidarSource, LidarModel, LidarFrame
from input.extrinsic import ExtrinsicModel, ExtrinsicRaw
from input.dataset import EvaluationDataset, SyncedFrame, SyncConfig


# ---------------------------------------------------------------------------
# Synthetic scene builder (same construction as test_edge_alignment.py's
# _make_synthetic_scene, generalized to take a per-frame world drift so we
# can simulate a fixed T_CL that is well-matched for some time blocks and
# poorly-matched for others).
# ---------------------------------------------------------------------------

WIDTH, HEIGHT = 640, 480
FX = FY = 500.0
CX, CY = 320.0, 240.0
Z_NEAR, Z_FAR = 5.0, 10.0


def _make_camera():
    return CameraModel(
        width=WIDTH, height=HEIGHT, model="pinhole",
        intrinsics=CameraIntrinsics(FX, FY, CX, CY),
        distortion=CameraDistortion(model="none"),
        source=CameraSource(kind="image_dir", path="."),
    )


def _make_image():
    image = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
    image[:, int(CX):] = 255
    return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)


def _make_base_points_cam_frame():
    u_vals = np.linspace(0, WIDTH - 1, 220)
    v_vals = np.linspace(0, HEIGHT - 1, 140)
    uu, vv = np.meshgrid(u_vals, v_vals)
    uu = uu.ravel()
    vv = vv.ravel()
    zz = np.where(uu < CX, Z_NEAR, Z_FAR)
    xx = (uu - CX) * zz / FX
    yy = (vv - CY) * zz / FY
    return np.stack([xx, yy, zz], axis=1)


def _make_lidar_spec():
    return LidarSensorSpec(horizontal_resolution_deg=0.05, vertical_resolution_deg=0.05, range_accuracy_m=0.02)


def _make_dataset(x_drifts_per_frame: list[float]) -> EvaluationDataset:
    """
    Build an EvaluationDataset with len(x_drifts_per_frame) synced frames.
    Evaluation always uses T_CL = identity (the "existing calibration" the
    tool is checking). For frame i, the LiDAR points are the base scene
    points MINUS x_drifts_per_frame[i] -- i.e. this frame's *true* world
    alignment is offset by that drift relative to what T_CL=identity
    assumes. A drift of 0.0 means this frame is perfectly consistent with
    the fixed T_CL; a nonzero drift simulates a time window where the
    existing calibration doesn't hold as well (e.g. rig flex, vibration).
    """
    camera = _make_camera()
    image = _make_image()
    base_points = _make_base_points_cam_frame()

    lidar = LidarModel(source=LidarSource(kind="pcd_dir", path="."), sensor_spec=_make_lidar_spec())
    raw = ExtrinsicRaw(parent="lidar", child="camera", translation=(0, 0, 0),
                        rotation=(0, 0, 0), rotation_format="rpy_deg")
    extrinsic = ExtrinsicModel(T_CL=np.eye(4), parent="lidar", child="camera", raw=raw)

    frames = []
    for i, drift in enumerate(x_drifts_per_frame):
        pts = base_points.copy()
        pts[:, 0] -= drift  # simulate this frame's true offset from T_CL's assumption
        cam_frame = CameraFrame(timestamp=float(i), image=image)
        lidar_frame = LidarFrame(timestamp=float(i), points=pts)
        frames.append(SyncedFrame(index=i, timestamp=float(i), camera_frame=cam_frame,
                                   lidar_frame=lidar_frame, time_diff_ms=0.0))

    return EvaluationDataset(
        camera=camera, lidar=lidar, extrinsic=extrinsic,
        sync_config=SyncConfig(), frames=frames,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_holdout_consistency_low_std_when_calibration_uniformly_good():
    # All 40 frames have zero drift -> T_CL=identity is equally (well)
    # matched everywhere -> block means should be nearly identical -> low STD.
    dataset = _make_dataset([0.0] * 40)
    result = evaluate_holdout_consistency(
        dataset, lidar_spec=_make_lidar_spec(), n_blocks=4, min_frames_per_block=5,
        edge_alignment_kwargs={"depth_jump_threshold_m": 1.0},
    )
    assert result.classification != "FAIL", result.warnings
    assert result.num_valid_blocks == 4
    assert result.std_across_blocks_px < 0.5, f"std={result.std_across_blocks_px}"
    assert result.classification == "GOOD"


def test_holdout_consistency_high_std_when_calibration_drifts_across_blocks():
    # First half of frames: zero drift (matches T_CL). Second half: drift
    # (T_CL no longer matches well there) -> block means should differ a
    # lot between early and late blocks -> high STD.
    # edge_radius_px is widened here because the drift itself shifts the
    # near/far point regions by different pixel amounts (perspective), which
    # opens a gap between them; too small a radius would miss the boundary
    # entirely (a real effect of M2's neighbor-based edge detection, not a
    # test artifact) rather than reflect it as increased error.
    drifts = [0.0] * 20 + [0.08] * 20
    dataset = _make_dataset(drifts)
    result = evaluate_holdout_consistency(
        dataset, lidar_spec=_make_lidar_spec(), n_blocks=4, min_frames_per_block=5,
        edge_alignment_kwargs={"depth_jump_threshold_m": 1.0, "edge_radius_px": 8.0},
    )
    assert result.classification != "FAIL", result.warnings
    assert result.num_valid_blocks == 4
    assert result.std_across_blocks_px > 1.0, f"std={result.std_across_blocks_px}"


def test_holdout_consistency_drifting_case_has_higher_std_than_uniform_case():
    uniform_dataset = _make_dataset([0.0] * 40)
    drifting_dataset = _make_dataset([0.0] * 20 + [0.08] * 20)

    result_uniform = evaluate_holdout_consistency(
        uniform_dataset, lidar_spec=_make_lidar_spec(), n_blocks=4, min_frames_per_block=5,
        edge_alignment_kwargs={"depth_jump_threshold_m": 1.0, "edge_radius_px": 8.0},
    )
    result_drifting = evaluate_holdout_consistency(
        drifting_dataset, lidar_spec=_make_lidar_spec(), n_blocks=4, min_frames_per_block=5,
        edge_alignment_kwargs={"depth_jump_threshold_m": 1.0, "edge_radius_px": 8.0},
    )
    assert result_drifting.std_across_blocks_px > result_uniform.std_across_blocks_px


def test_holdout_consistency_blocks_are_contiguous_time_windows():
    drifts = [0.0] * 20 + [0.08] * 20
    dataset = _make_dataset(drifts)
    result = evaluate_holdout_consistency(
        dataset, lidar_spec=_make_lidar_spec(), n_blocks=4, min_frames_per_block=5,
        edge_alignment_kwargs={"depth_jump_threshold_m": 1.0, "edge_radius_px": 8.0},
    )
    # frame_indices per block must be contiguous and in increasing order
    for b in result.block_results:
        idxs = b.frame_indices
        assert idxs == sorted(idxs)
        assert idxs == list(range(idxs[0], idxs[-1] + 1))
    # concatenation of all blocks' indices must cover 0..39 with no gaps/overlaps
    all_idxs = [i for b in result.block_results for i in b.frame_indices]
    assert all_idxs == list(range(40))


def test_holdout_consistency_excludes_blocks_below_min_frames():
    # 10 frames split into 4 blocks -> 2-3 frames per block, below min_frames_per_block=5
    dataset = _make_dataset([0.0] * 10)
    result = evaluate_holdout_consistency(
        dataset, lidar_spec=_make_lidar_spec(), n_blocks=4, min_frames_per_block=5,
        edge_alignment_kwargs={"depth_jump_threshold_m": 1.0},
    )
    assert all(b.classification == "EXCLUDED" for b in result.block_results)
    assert result.classification == "FAIL"
    assert result.num_valid_blocks == 0


def test_holdout_consistency_fails_with_fewer_than_min_valid_blocks():
    # n_blocks=2 -> can never reach MIN_VALID_BLOCKS=3
    dataset = _make_dataset([0.0] * 40)
    result = evaluate_holdout_consistency(
        dataset, lidar_spec=_make_lidar_spec(), n_blocks=2, min_frames_per_block=5,
        edge_alignment_kwargs={"depth_jump_threshold_m": 1.0},
    )
    assert result.classification == "FAIL"
    assert result.num_valid_blocks < MIN_VALID_BLOCKS
    assert any("valid block" in w for w in result.warnings)


def test_holdout_consistency_empty_dataset():
    dataset = _make_dataset([])
    result = evaluate_holdout_consistency(
        dataset, lidar_spec=_make_lidar_spec(), n_blocks=4, min_frames_per_block=5,
    )
    assert result.classification == "FAIL"
    assert result.num_valid_blocks == 0


def test_holdout_consistency_block_result_floor_positive_for_valid_blocks():
    dataset = _make_dataset([0.0] * 40)
    result = evaluate_holdout_consistency(
        dataset, lidar_spec=_make_lidar_spec(), n_blocks=4, min_frames_per_block=5,
        edge_alignment_kwargs={"depth_jump_threshold_m": 1.0},
    )
    for b in result.block_results:
        assert b.floor_px > 0
        assert b.classification in ("GOOD", "WARNING", "BAD")


# ---------------------------------------------------------------------------
# STEP10 -- scene metadata + instability diagnosis
# ---------------------------------------------------------------------------

def _make_block_result(index, mean_px, depth=15.0, edge_density=50.0, num_points_avg=5000.0,
                        fov_coverage=0.5, classification="GOOD"):
    return BlockResult(
        block_index=index, frame_indices=list(range(index * 10, index * 10 + 10)),
        num_frames_total=10, num_frames_valid=10, num_frames_failed=0,
        mean_px=mean_px, median_px=mean_px, p95_px=mean_px * 1.2,
        num_edge_points=int(edge_density * 10), representative_depth_m=depth,
        floor_px=1.0, classification=classification,
        edge_density=edge_density, num_points_avg=num_points_avg, fov_coverage=fov_coverage,
    )


def test_diagnose_instability_identifies_long_range_scenes():
    blocks = [
        _make_block_result(0, mean_px=1.0, depth=15.0),
        _make_block_result(1, mean_px=1.1, depth=14.0),
        _make_block_result(2, mean_px=1.2, depth=16.0),
        _make_block_result(3, mean_px=5.0, depth=45.0),  # worst block, also much farther
    ]
    diagnosis = diagnose_instability(blocks)
    assert diagnosis is not None
    assert diagnosis["worst_block_index"] == 3
    assert diagnosis["candidates"][0]["metric"] == "representative_depth_m"
    assert diagnosis["candidates"][0]["explanation"] == "Long-range scenes"
    assert diagnosis["candidates"][0]["relative_diff"] > 0.5


def test_diagnose_instability_identifies_sparse_edge_structure():
    blocks = [
        _make_block_result(0, mean_px=1.0, edge_density=200.0),
        _make_block_result(1, mean_px=1.1, edge_density=210.0),
        _make_block_result(2, mean_px=1.2, edge_density=190.0),
        _make_block_result(3, mean_px=5.0, edge_density=20.0),  # worst block, sparse edges
    ]
    diagnosis = diagnose_instability(blocks)
    assert diagnosis is not None
    top_metrics = [c["metric"] for c in diagnosis["candidates"]]
    assert "edge_density" in top_metrics
    edge_candidate = next(c for c in diagnosis["candidates"] if c["metric"] == "edge_density")
    assert edge_candidate["explanation"] == "Sparse edge structure"


def test_diagnose_instability_no_candidates_when_scenes_are_similar():
    blocks = [
        _make_block_result(0, mean_px=1.0, depth=15.0, edge_density=100.0),
        _make_block_result(1, mean_px=1.1, depth=15.5, edge_density=98.0),
        _make_block_result(2, mean_px=1.2, depth=14.5, edge_density=102.0),
        _make_block_result(3, mean_px=1.3, depth=15.2, edge_density=99.0),  # worst, but scene is similar
    ]
    diagnosis = diagnose_instability(blocks)
    assert diagnosis is not None
    assert diagnosis["candidates"] == []


def test_diagnose_instability_none_with_too_few_valid_blocks():
    blocks = [
        _make_block_result(0, mean_px=1.0),
        _make_block_result(1, mean_px=5.0),
    ]
    assert diagnose_instability(blocks) is None


def test_diagnose_instability_ignores_excluded_and_failed_blocks():
    blocks = [
        _make_block_result(0, mean_px=1.0, depth=15.0),
        _make_block_result(1, mean_px=1.1, depth=14.0),
        _make_block_result(2, mean_px=1.2, depth=16.0),
        _make_block_result(3, mean_px=999.0, depth=999.0, classification="EXCLUDED"),  # must be ignored
        _make_block_result(4, mean_px=5.0, depth=45.0),
    ]
    diagnosis = diagnose_instability(blocks)
    assert diagnosis is not None
    assert diagnosis["worst_block_index"] == 4  # not the EXCLUDED block, despite its huge mean_px


def test_holdout_consistency_integration_populates_scene_metadata():
    drifts = [0.0] * 20 + [0.08] * 20
    dataset = _make_dataset(drifts)
    result = evaluate_holdout_consistency(
        dataset, lidar_spec=_make_lidar_spec(), n_blocks=4, min_frames_per_block=5,
        edge_alignment_kwargs={"depth_jump_threshold_m": 1.0, "edge_radius_px": 8.0},
    )
    for b in result.block_results:
        if b.classification in ("GOOD", "WARNING", "BAD"):
            assert np.isfinite(b.edge_density) and b.edge_density > 0
            assert np.isfinite(b.num_points_avg) and b.num_points_avg > 0
            assert np.isfinite(b.fov_coverage) and b.fov_coverage >= 0
            assert b.dynamic_ratio is None  # no dynamic_masks supplied


def test_holdout_consistency_result_carries_instability_diagnosis():
    drifts = [0.0] * 20 + [0.08] * 20
    dataset = _make_dataset(drifts)
    result = evaluate_holdout_consistency(
        dataset, lidar_spec=_make_lidar_spec(), n_blocks=4, min_frames_per_block=5,
        edge_alignment_kwargs={"depth_jump_threshold_m": 1.0, "edge_radius_px": 8.0},
    )
    assert result.instability_diagnosis is not None
    assert "worst_block_index" in result.instability_diagnosis
    assert "candidates" in result.instability_diagnosis


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
