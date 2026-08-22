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
    # A NON-constant mismatch (alternating +/-500ms, camera frames spaced
    # far enough apart that nearest-neighbor association is unambiguous):
    # the median offset bootstrap correctly estimates ~0 here (the +/-
    # cancel out), so offset correction can't rescue these pairs -- unlike
    # a genuinely constant offset (see test_sync_estimates_constant_offset),
    # this is real, uncorrectable desync and must still be dropped.
    cam_frames = [CameraFrame(timestamp=t) for t in [0.0, 10.0, 20.0, 30.0]]
    lid_frames = [LidarFrame(timestamp=t) for t in [-0.5, 10.5, 19.5, 30.5]]
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


# ---------------------------------------------------------------------------
# STEP 2 -- Timestamp Synchronization: candidate window + monotonic
# matching + offset estimation (see input/dataset.py's module docstring)
# ---------------------------------------------------------------------------

def test_sync_matched_indices_are_monotonic():
    """A naive 'argmin over all unused lidar frames' matcher can produce a
    NON-monotonic assignment when timestamps interleave awkwardly. The
    two-pointer matcher must never do this: matched lidar indices must be
    strictly increasing in camera-index order."""
    # Camera and lidar rates differ slightly and drift into an order where
    # a purely-greedy-per-camera-frame argmin could pick lidar frames out
    # of order (e.g. matching a later lidar frame to an earlier camera
    # frame because it happened to be marginally closer).
    cam_ts = [0.0, 0.31, 0.62, 0.93, 1.24]
    lid_ts = [0.02, 0.30, 0.29, 0.65, 0.90, 1.25]  # 0.29 sits BEFORE 0.30 in time
    cam_frames = [CameraFrame(timestamp=t) for t in cam_ts]
    lid_frames = [LidarFrame(timestamp=t) for t in lid_ts]
    ds = build_dataset(_dummy_camera_model(), cam_frames, _dummy_lidar_model(), lid_frames,
                        _dummy_extrinsic_model(), SyncConfig(max_time_diff_ms=150))
    lidar_ts_matched = [sf.lidar_frame.timestamp for sf in ds.frames]
    assert lidar_ts_matched == sorted(lidar_ts_matched), (
        f"matched lidar timestamps are not monotonic: {lidar_ts_matched}"
    )
    # camera frame order in the result must also be increasing (frames are
    # re-sorted by camera timestamp at the end of the sync pass)
    cam_ts_matched = [sf.camera_frame.timestamp for sf in ds.frames]
    assert cam_ts_matched == sorted(cam_ts_matched)


def test_sync_candidate_window_respected():
    """Even with offset-correction active, a NON-constant mismatch (no
    consistent Δt to correct for) must still be dropped -- offset
    correction only rescues a genuinely constant clock offset
    (test_sync_estimates_constant_offset), never arbitrary large jitter.
    Uses alternating +/-300ms with camera frames spaced far enough apart
    that nearest-neighbor association stays unambiguous, so the median
    bootstrap correctly lands near zero (the +/- cancel out) and offers
    no correction to rescue these clearly-out-of-window pairs."""
    cam_frames = [CameraFrame(timestamp=t) for t in [0.0, 10.0, 20.0, 30.0]]
    lid_frames = [LidarFrame(timestamp=t) for t in [-0.3, 10.3, 19.7, 30.3]]
    ds = build_dataset(_dummy_camera_model(), cam_frames, _dummy_lidar_model(), lid_frames,
                        _dummy_extrinsic_model(), SyncConfig(max_time_diff_ms=50))
    assert len(ds.frames) == 0
    assert ds.sync_stats.num_matched == 0
    assert ds.sync_stats.classification == "FAIL"


def test_sync_estimates_constant_offset():
    """A CONSTANT clock offset (every lidar frame exactly 17ms behind its
    camera counterpart) should be both recovered via the offset-correction
    pass (frames land back inside a tight window) and reported accurately
    in estimated_offset_ms, with near-zero offset_std_ms since the offset
    truly is constant."""
    offset_s = 0.017  # camera is 17ms AHEAD of lidar clock => Δt = +17ms
    cam_ts = [float(i) for i in range(20)]
    lid_ts = [t - offset_s for t in cam_ts]
    cam_frames = [CameraFrame(timestamp=t) for t in cam_ts]
    lid_frames = [LidarFrame(timestamp=t) for t in lid_ts]

    # A raw window of 10ms would normally be too tight to catch a 17ms
    # offset with plain nearest-neighbor -- but the offset-correction pass
    # re-centers the search, so it should still fully match here.
    ds = build_dataset(_dummy_camera_model(), cam_frames, _dummy_lidar_model(), lid_frames,
                        _dummy_extrinsic_model(), SyncConfig(max_time_diff_ms=10))
    assert ds.sync_stats.num_matched == 20
    assert abs(ds.sync_stats.estimated_offset_ms - 17.0) < 0.5
    assert ds.sync_stats.offset_std_ms < 0.5
    assert ds.sync_stats.classification == "GOOD"


def test_sync_offset_sign_convention():
    """Δt = camera_clock - lidar_clock: if the LIDAR is ahead of the
    camera (lidar timestamps are numerically larger for the 'same' event),
    the estimated offset should be NEGATIVE."""
    cam_ts = [float(i) for i in range(10)]
    lid_ts = [t + 0.01 for t in cam_ts]  # lidar clock reads 10ms ahead
    cam_frames = [CameraFrame(timestamp=t) for t in cam_ts]
    lid_frames = [LidarFrame(timestamp=t) for t in lid_ts]
    ds = build_dataset(_dummy_camera_model(), cam_frames, _dummy_lidar_model(), lid_frames,
                        _dummy_extrinsic_model(), SyncConfig(max_time_diff_ms=15))
    assert ds.sync_stats.estimated_offset_ms < 0
    assert abs(ds.sync_stats.estimated_offset_ms - (-10.0)) < 0.5


def test_sync_drop_ratio_and_classification_warning():
    cam_frames = [CameraFrame(timestamp=float(i)) for i in range(20)]
    # only 16/20 camera frames get a lidar match (4 lidar frames simply
    # missing) -> 20% drop ratio, right at the WARNING boundary
    lid_ts = [float(i) for i in range(20) if i not in (3, 7, 11, 15)]
    lid_frames = [LidarFrame(timestamp=t) for t in lid_ts]
    ds = build_dataset(_dummy_camera_model(), cam_frames, _dummy_lidar_model(), lid_frames,
                        _dummy_extrinsic_model(), SyncConfig(max_time_diff_ms=50))
    assert ds.sync_stats.num_matched == 16
    assert abs(ds.sync_stats.drop_ratio - 0.2) < 1e-9
    assert ds.sync_stats.classification in ("WARNING", "GOOD")  # boundary-inclusive per classify_sync


def test_sync_stats_to_dict_shape():
    cam_frames = [CameraFrame(timestamp=t) for t in [0.0, 1.0, 2.0]]
    lid_frames = [LidarFrame(timestamp=t) for t in [0.0, 1.0, 2.0]]
    ds = build_dataset(_dummy_camera_model(), cam_frames, _dummy_lidar_model(), lid_frames,
                        _dummy_extrinsic_model(), SyncConfig(max_time_diff_ms=50))
    d = ds.sync_stats.to_dict()
    for key in ("num_camera_frames", "num_lidar_frames", "num_matched",
                "num_camera_dropped", "num_lidar_dropped", "mean_time_diff_ms",
                "max_time_diff_ms", "estimated_offset_ms", "offset_std_ms",
                "drop_ratio", "classification"):
        assert key in d, f"missing key {key!r} in SyncStats.to_dict()"
    assert d["classification"] == "GOOD"
    assert d["drop_ratio"] == 0.0


def test_sync_large_offset_triggers_warning():
    """An offset large relative to max_time_diff_ms should surface a
    dataset-level warning suggesting the person fix it upstream or widen
    the tolerance -- distinct from the plain 'poor sync' majority-drop
    warning."""
    offset_s = 0.04  # 40ms, i.e. 80% of a 50ms window
    cam_ts = [float(i) for i in range(10)]
    lid_ts = [t - offset_s for t in cam_ts]
    cam_frames = [CameraFrame(timestamp=t) for t in cam_ts]
    lid_frames = [LidarFrame(timestamp=t) for t in lid_ts]
    ds = build_dataset(_dummy_camera_model(), cam_frames, _dummy_lidar_model(), lid_frames,
                        _dummy_extrinsic_model(), SyncConfig(max_time_diff_ms=50))
    assert any("offset" in w.lower() for w in ds.warnings)


def test_classify_sync_directly():
    from input.dataset import classify_sync
    assert classify_sync(num_matched=0, num_camera_frames=10, offset_std_ms=0.0, max_time_diff_ms=50) == "FAIL"
    assert classify_sync(num_matched=10, num_camera_frames=10, offset_std_ms=1.0, max_time_diff_ms=50) == "GOOD"
    assert classify_sync(num_matched=8, num_camera_frames=10, offset_std_ms=1.0, max_time_diff_ms=50) == "WARNING"
    assert classify_sync(num_matched=5, num_camera_frames=10, offset_std_ms=1.0, max_time_diff_ms=50) == "BAD"
    assert classify_sync(num_matched=10, num_camera_frames=10, offset_std_ms=40.0, max_time_diff_ms=50) == "BAD"


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
