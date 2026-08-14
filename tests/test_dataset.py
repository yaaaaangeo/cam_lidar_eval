import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from input.camera import CameraFrame, CameraModel, CameraIntrinsics, CameraDistortion, CameraSource
from input.lidar import LidarFrame, LidarModel, LidarSensorSpec, LidarSource
from input.extrinsic import ExtrinsicModel, ExtrinsicRaw
from input.dataset import SyncConfig, build_dataset
import numpy as np


def _dummy_camera_model():
    return CameraModel(
        width=640, height=480, model="pinhole",
        intrinsics=CameraIntrinsics(500, 500, 320, 240),
        distortion=CameraDistortion(model="none"),
        source=CameraSource(kind="image_dir", path="."),
    )


def _dummy_lidar_model():
    return LidarModel(
        source=LidarSource(kind="pcd_dir", path="."),
        sensor_spec=LidarSensorSpec(horizontal_resolution_deg=0.2, range_accuracy_m=0.02),
    )


def _dummy_extrinsic_model():
    raw = ExtrinsicRaw(parent="lidar", child="camera", translation=(0, 0, 0),
                        rotation=(0, 0, 0), rotation_format="rpy_deg")
    return ExtrinsicModel(T_CL=np.eye(4), parent="lidar", child="camera", raw=raw)


def test_build_dataset_exact_matches():
    cam_frames = [CameraFrame(timestamp=t) for t in [0.0, 1.0, 2.0]]
    lid_frames = [LidarFrame(timestamp=t) for t in [0.0, 1.0, 2.0]]
    ds = build_dataset(_dummy_camera_model(), cam_frames, _dummy_lidar_model(), lid_frames,
                        _dummy_extrinsic_model(), SyncConfig(max_time_diff_ms=50))
    assert len(ds.frames) == 3
    assert ds.sync_stats.num_matched == 3
    assert ds.sync_stats.num_camera_dropped == 0
    for sf in ds.frames:
        assert sf.time_diff_ms == 0.0


def test_build_dataset_within_tolerance_matches():
    cam_frames = [CameraFrame(timestamp=t) for t in [0.0, 1.0, 2.0]]
    lid_frames = [LidarFrame(timestamp=t) for t in [0.02, 1.03, 2.01]]  # 20-30ms offsets
    ds = build_dataset(_dummy_camera_model(), cam_frames, _dummy_lidar_model(), lid_frames,
                        _dummy_extrinsic_model(), SyncConfig(max_time_diff_ms=50))
    assert len(ds.frames) == 3
    assert all(sf.time_diff_ms <= 50 for sf in ds.frames)


def test_build_dataset_outside_tolerance_dropped():
    cam_frames = [CameraFrame(timestamp=t) for t in [0.0, 1.0, 2.0]]
    lid_frames = [LidarFrame(timestamp=t) for t in [0.5, 1.5, 2.5]]  # 500ms off, way outside 50ms
    ds = build_dataset(_dummy_camera_model(), cam_frames, _dummy_lidar_model(), lid_frames,
                        _dummy_extrinsic_model(), SyncConfig(max_time_diff_ms=50))
    assert len(ds.frames) == 0
    assert ds.sync_stats.num_matched == 0
    assert len(ds.warnings) == 1
    assert "No camera-lidar frame pairs matched" in ds.warnings[0]


def test_build_dataset_lidar_frame_not_double_matched():
    """Two camera frames close to the same single lidar frame: only the
    closer camera frame should claim it."""
    cam_frames = [CameraFrame(timestamp=t) for t in [0.0, 0.01]]
    lid_frames = [LidarFrame(timestamp=0.005)]
    ds = build_dataset(_dummy_camera_model(), cam_frames, _dummy_lidar_model(), lid_frames,
                        _dummy_extrinsic_model(), SyncConfig(max_time_diff_ms=50))
    assert len(ds.frames) == 1
    # camera frame at 0.0 is 5ms away, camera frame at 0.01 is 5ms away too (tie)
    # either is acceptable since it's a genuine tie, but exactly one match must exist
    assert ds.sync_stats.num_matched == 1


def test_build_dataset_warns_on_majority_drop():
    cam_frames = [CameraFrame(timestamp=float(i)) for i in range(10)]
    lid_frames = [LidarFrame(timestamp=float(i)) for i in range(2)]  # only 2 lidar frames
    ds = build_dataset(_dummy_camera_model(), cam_frames, _dummy_lidar_model(), lid_frames,
                        _dummy_extrinsic_model(), SyncConfig(max_time_diff_ms=50))
    assert len(ds.frames) == 2
    assert any("poor" in w for w in ds.warnings)


def test_time_blocks_even_split():
    cam_frames = [CameraFrame(timestamp=float(i)) for i in range(9)]
    lid_frames = [LidarFrame(timestamp=float(i)) for i in range(9)]
    ds = build_dataset(_dummy_camera_model(), cam_frames, _dummy_lidar_model(), lid_frames,
                        _dummy_extrinsic_model(), SyncConfig(max_time_diff_ms=50))
    blocks = ds.time_blocks(3)
    assert len(blocks) == 3
    assert all(len(b) == 3 for b in blocks)
    # contiguous: first block should have the earliest timestamps
    assert [f.timestamp for f in blocks[0]] == [0.0, 1.0, 2.0]
    assert [f.timestamp for f in blocks[2]] == [6.0, 7.0, 8.0]


def test_time_blocks_uneven_split_stays_contiguous():
    cam_frames = [CameraFrame(timestamp=float(i)) for i in range(10)]
    lid_frames = [LidarFrame(timestamp=float(i)) for i in range(10)]
    ds = build_dataset(_dummy_camera_model(), cam_frames, _dummy_lidar_model(), lid_frames,
                        _dummy_extrinsic_model(), SyncConfig(max_time_diff_ms=50))
    blocks = ds.time_blocks(3)
    assert sum(len(b) for b in blocks) == 10
    # verify contiguity: concatenated timestamps should be sorted and equal to 0..9
    all_ts = [f.timestamp for b in blocks for f in b]
    assert all_ts == sorted(all_ts)
    assert all_ts == [float(i) for i in range(10)]


def test_time_blocks_on_empty_dataset():
    ds = build_dataset(_dummy_camera_model(), [], _dummy_lidar_model(), [],
                        _dummy_extrinsic_model(), SyncConfig())
    blocks = ds.time_blocks(3)
    assert len(blocks) == 3
    assert all(b == [] for b in blocks)


def test_time_blocks_rejects_zero_or_negative_n():
    ds = build_dataset(_dummy_camera_model(), [CameraFrame(timestamp=0.0)],
                        _dummy_lidar_model(), [LidarFrame(timestamp=0.0)],
                        _dummy_extrinsic_model(), SyncConfig())
    try:
        ds.time_blocks(0)
        assert False, "expected ValueError"
    except ValueError:
        pass


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
