import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from input.camera import (
    CameraIntrinsics, CameraDistortion, CameraModel, CameraSource, CameraFrame,
)
from input.lidar import LidarSensorSpec, LidarModel, LidarSource, LidarFrame
from input.validation import (
    ValidationStatus,
    ValidationReport,
    InputValidationError,
    validate_camera,
    validate_lidar,
    validate_dataset,
    validate_input,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _make_camera(fx=500.0, fy=500.0, cx=32.0, cy=24.0, width=64, height=48,
                  dist_model="none", dist_coeffs=None):
    intr = CameraIntrinsics(fx=fx, fy=fy, cx=cx, cy=cy)
    dist = CameraDistortion(model=dist_model, coeffs=dist_coeffs or {})
    source = CameraSource(kind="image_dir", path="/fake")
    return CameraModel(width=width, height=height, model="pinhole",
                        intrinsics=intr, distortion=dist, source=source)


def _cam_frames(timestamps):
    return [CameraFrame(timestamp=t, path=f"/fake/{i}.png") for i, t in enumerate(timestamps)]


def _make_lidar(min_range_m=0.0, max_range_m=200.0):
    spec = LidarSensorSpec(min_range_m=min_range_m, max_range_m=max_range_m)
    source = LidarSource(kind="pcd_dir", path="/fake")
    return LidarModel(source=source, sensor_spec=spec)


def _lidar_frames_from_points(timestamps, points_list):
    return [LidarFrame(timestamp=t, points=p) for t, p in zip(timestamps, points_list)]


def _good_points(n=200, seed=0):
    rng = np.random.default_rng(seed)
    return rng.uniform(-5, 5, size=(n, 3)).astype(np.float32)


# ---------------------------------------------------------------------------
# validate_camera
# ---------------------------------------------------------------------------

def test_validate_camera_all_good():
    camera = _make_camera()
    frames = _cam_frames([0.0, 0.033, 0.066, 0.1])
    report = validate_camera(camera, frames)
    assert report.status == ValidationStatus.VALID
    assert report.failed_items() == []


def test_validate_camera_negative_fx_is_invalid():
    camera = _make_camera(fx=-500.0)
    frames = _cam_frames([0.0, 0.1])
    report = validate_camera(camera, frames)
    assert report.status == ValidationStatus.INVALID
    names = [i.name for i in report.failed_items()]
    assert "camera.fx_positive" in names


def test_validate_camera_zero_fy_is_invalid():
    camera = _make_camera(fy=0.0)
    frames = _cam_frames([0.0, 0.1])
    report = validate_camera(camera, frames)
    assert report.status == ValidationStatus.INVALID
    names = [i.name for i in report.failed_items()]
    assert "camera.fy_positive" in names


def test_validate_camera_cx_out_of_bounds_is_warning_not_invalid():
    camera = _make_camera(cx=99999.0)
    frames = _cam_frames([0.0, 0.1])
    report = validate_camera(camera, frames)
    assert report.status == ValidationStatus.WARNING
    warn_names = [i.name for i in report.warning_items()]
    assert "camera.cx_in_image_bounds" in warn_names


def test_validate_camera_bad_image_size_is_invalid():
    camera = _make_camera(width=0)
    frames = _cam_frames([0.0, 0.1])
    report = validate_camera(camera, frames)
    assert report.status == ValidationStatus.INVALID


def test_validate_camera_no_frames_is_invalid():
    camera = _make_camera()
    report = validate_camera(camera, [])
    assert report.status == ValidationStatus.INVALID
    names = [i.name for i in report.failed_items()]
    assert "camera.frames_present" in names


def test_validate_camera_non_monotonic_timestamps_is_invalid():
    camera = _make_camera()
    frames = _cam_frames([0.0, 0.1, 0.05, 0.2])  # out of order
    report = validate_camera(camera, frames)
    assert report.status == ValidationStatus.INVALID
    reasons = report.reasons(ValidationStatus.INVALID)
    assert any("monotonic" in r.lower() for r in reasons)


def test_validate_camera_duplicate_timestamps_is_invalid():
    camera = _make_camera()
    frames = _cam_frames([0.0, 0.1, 0.1, 0.2])  # duplicate -> not strictly increasing
    report = validate_camera(camera, frames)
    assert report.status == ValidationStatus.INVALID


def test_validate_camera_nan_timestamp_is_invalid():
    camera = _make_camera()
    frames = _cam_frames([0.0, float("nan"), 0.2])
    report = validate_camera(camera, frames)
    assert report.status == ValidationStatus.INVALID
    names = [i.name for i in report.failed_items()]
    assert "camera.timestamps_finite" in names


def test_validate_camera_plumb_bob_valid_coeffs_is_valid():
    # CameraDistortion.as_array() fills missing plumb_bob keys with 0.0, so
    # a partial coeffs dict still resolves to a well-formed 5-element array
    # -- this just checks the distortion check doesn't false-positive on it.
    camera = _make_camera(dist_model="plumb_bob", dist_coeffs={"k1": 0.1, "k2": 0.0})
    frames = _cam_frames([0.0, 0.1])
    report = validate_camera(camera, frames)
    assert report.status == ValidationStatus.VALID


def test_validate_camera_nonfinite_distortion_coeffs_is_invalid():
    camera = _make_camera(dist_model="plumb_bob", dist_coeffs={"k1": float("nan")})
    frames = _cam_frames([0.0, 0.1])
    report = validate_camera(camera, frames)
    assert report.status == ValidationStatus.INVALID
    names = [i.name for i in report.failed_items()]
    assert "camera.distortion_params_valid" in names


# ---------------------------------------------------------------------------
# validate_lidar
# ---------------------------------------------------------------------------

def test_validate_lidar_all_good():
    lidar = _make_lidar()
    ts = [0.0, 0.1, 0.2, 0.3, 0.4]
    frames = _lidar_frames_from_points(ts, [_good_points(seed=i) for i in range(len(ts))])
    report = validate_lidar(lidar, frames)
    assert report.status == ValidationStatus.VALID


def test_validate_lidar_no_frames_is_invalid():
    lidar = _make_lidar()
    report = validate_lidar(lidar, [])
    assert report.status == ValidationStatus.INVALID


def test_validate_lidar_empty_pointcloud_is_invalid():
    lidar = _make_lidar()
    ts = [0.0, 0.1]
    frames = _lidar_frames_from_points(ts, [np.zeros((0, 3), dtype=np.float32), _good_points()])
    report = validate_lidar(lidar, frames, sample_frames=2)
    assert report.status == ValidationStatus.INVALID
    names = [i.name for i in report.failed_items()]
    assert "lidar.point_count_nonzero" in names


def test_validate_lidar_sparse_frame_is_warning():
    lidar = _make_lidar()
    ts = [0.0, 0.1]
    sparse = _good_points(n=5)
    frames = _lidar_frames_from_points(ts, [sparse, _good_points()])
    report = validate_lidar(lidar, frames, sample_frames=2)
    assert report.status == ValidationStatus.WARNING
    names = [i.name for i in report.warning_items()]
    assert "lidar.point_count_sufficient" in names


def test_validate_lidar_nan_points_severe_is_invalid():
    lidar = _make_lidar()
    pts = _good_points(n=200)
    pts[:150] = np.nan  # 75% NaN -- above the INVALID threshold
    frames = _lidar_frames_from_points([0.0], [pts])
    report = validate_lidar(lidar, frames, sample_frames=1)
    assert report.status == ValidationStatus.INVALID
    names = [i.name for i in report.failed_items()]
    assert "lidar.xyz_finite" in names


def test_validate_lidar_nan_points_mild_is_warning():
    lidar = _make_lidar()
    pts = _good_points(n=200)
    pts[:5] = np.nan  # 2.5% NaN -- above warning threshold, below invalid
    frames = _lidar_frames_from_points([0.0], [pts])
    report = validate_lidar(lidar, frames, sample_frames=1)
    assert report.status == ValidationStatus.WARNING


def test_validate_lidar_non_monotonic_timestamps_is_invalid():
    lidar = _make_lidar()
    ts = [0.0, 0.2, 0.1]
    frames = _lidar_frames_from_points(ts, [_good_points(seed=i) for i in range(3)])
    report = validate_lidar(lidar, frames, sample_frames=3)
    assert report.status == ValidationStatus.INVALID


def test_validate_lidar_out_of_range_points_is_warning():
    lidar = _make_lidar(min_range_m=0.0, max_range_m=10.0)
    far_pts = np.full((200, 3), 500.0, dtype=np.float32)  # way beyond max_range_m
    frames = _lidar_frames_from_points([0.0], [far_pts])
    report = validate_lidar(lidar, frames, sample_frames=1)
    assert report.status == ValidationStatus.WARNING
    names = [i.name for i in report.warning_items()]
    assert "lidar.range_within_sensor_spec" in names


# ---------------------------------------------------------------------------
# validate_dataset
# ---------------------------------------------------------------------------

def test_validate_dataset_good_overlap():
    # span must exceed MIN_OVERLAP_S_WARNING (1.0s) to land as fully VALID
    cam_frames = _cam_frames([0.0, 1.0, 2.0, 3.0])
    lidar_frames = _lidar_frames_from_points([0.0, 1.0, 2.0, 3.0], [_good_points()] * 4)
    report = validate_dataset(cam_frames, lidar_frames)
    assert report.status == ValidationStatus.VALID


def test_validate_dataset_no_overlap_is_invalid():
    cam_frames = _cam_frames([0.0, 0.1, 0.2])
    lidar_frames = _lidar_frames_from_points([100.0, 100.1, 100.2], [_good_points()] * 3)
    report = validate_dataset(cam_frames, lidar_frames)
    assert report.status == ValidationStatus.INVALID
    names = [i.name for i in report.failed_items()]
    assert "dataset.overlap_duration" in names


def test_validate_dataset_short_overlap_is_warning():
    cam_frames = _cam_frames([0.0, 0.05])
    lidar_frames = _lidar_frames_from_points([0.0, 0.05], [_good_points()] * 2)
    report = validate_dataset(cam_frames, lidar_frames)
    assert report.status == ValidationStatus.WARNING


def test_validate_dataset_empty_streams_is_invalid():
    report = validate_dataset([], [])
    assert report.status == ValidationStatus.INVALID


# ---------------------------------------------------------------------------
# validate_input (combined) + report/error mechanics
# ---------------------------------------------------------------------------

def test_validate_input_combined_valid():
    camera = _make_camera()
    lidar = _make_lidar()
    cam_frames = _cam_frames([0.0, 1.0, 2.0, 3.0])
    lidar_frames = _lidar_frames_from_points([0.0, 1.0, 2.0, 3.0], [_good_points(seed=i) for i in range(4)])
    report = validate_input(camera, cam_frames, lidar, lidar_frames)
    assert report.status == ValidationStatus.VALID
    assert report.reasons() == []


def test_validate_input_combined_reports_worst_status():
    camera = _make_camera(fx=-1.0)  # INVALID
    lidar = _make_lidar()
    cam_frames = _cam_frames([0.0, 0.1])
    lidar_frames = _lidar_frames_from_points([0.0, 0.1], [_good_points(seed=i) for i in range(2)])
    report = validate_input(camera, cam_frames, lidar, lidar_frames)
    assert report.status == ValidationStatus.INVALID
    assert any("fx" in r for r in report.reasons(ValidationStatus.INVALID))


def test_validate_input_raise_on_invalid():
    camera = _make_camera(fx=-1.0)
    lidar = _make_lidar()
    cam_frames = _cam_frames([0.0, 0.1])
    lidar_frames = _lidar_frames_from_points([0.0, 0.1], [_good_points(seed=i) for i in range(2)])
    try:
        validate_input(camera, cam_frames, lidar, lidar_frames, raise_on_invalid=True)
        assert False, "expected InputValidationError"
    except InputValidationError as e:
        assert e.report.status == ValidationStatus.INVALID
        assert "INPUT INVALID" in str(e)


def test_validate_input_does_not_raise_when_valid():
    camera = _make_camera()
    lidar = _make_lidar()
    cam_frames = _cam_frames([0.0, 1.0])
    lidar_frames = _lidar_frames_from_points([0.0, 1.0], [_good_points(seed=i) for i in range(2)])
    # should not raise even with raise_on_invalid=True, since status is VALID
    report = validate_input(camera, cam_frames, lidar, lidar_frames, raise_on_invalid=True)
    assert report.status == ValidationStatus.VALID


def test_validation_report_to_dict_shape():
    report = ValidationReport()
    from input.validation import ValidationCheckItem
    report.add(ValidationCheckItem("x.check", ValidationStatus.WARNING, "some detail", value=1.0))
    d = report.to_dict()
    assert d["status"] == "INPUT_WARNING"
    assert d["checks"][0]["name"] == "x.check"
    assert d["reasons"] == ["x.check: some detail"]


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
