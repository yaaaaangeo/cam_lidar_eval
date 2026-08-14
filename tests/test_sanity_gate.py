import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from evaluation.sanity_gate import run_sanity_gate, DEFAULT_MIN_FOV_COVERAGE
from input.camera import CameraModel, CameraIntrinsics, CameraDistortion, CameraSource
from input.lidar import LidarSensorSpec


def _camera():
    return CameraModel(
        width=640, height=480, model="pinhole",
        intrinsics=CameraIntrinsics(500, 500, 320, 240),
        distortion=CameraDistortion(model="none"),
        source=CameraSource(kind="image_dir", path="."),
    )


def _well_behaved_points(n=2000, seed=0):
    """A cluster of points comfortably in front of and within the camera's
    FOV, at a single sane depth, with basic occlusion structure (no two
    surfaces folded onto each other)."""
    rng = np.random.RandomState(seed)
    x = rng.uniform(-1.5, 1.5, n)
    y = rng.uniform(-1.0, 1.0, n)
    z = np.full(n, 5.0) + rng.uniform(-0.05, 0.05, n)
    return np.stack([x, y, z], axis=1)


def test_sanity_gate_passes_for_well_behaved_scene():
    camera = _camera()
    lidar_spec = LidarSensorSpec(horizontal_resolution_deg=0.2, range_accuracy_m=0.02,
                                  min_range_m=0.1, max_range_m=100.0)
    points = _well_behaved_points()
    result = run_sanity_gate(points, T_CL=np.eye(4), camera=camera, lidar_spec=lidar_spec)
    assert result.passed, result.failed_items()
    assert all(item.passed for item in result.items)
    assert result.fov_coverage_ratio > DEFAULT_MIN_FOV_COVERAGE
    assert result.num_valid_points > 0


def test_sanity_gate_fails_on_low_fov_coverage_when_extrinsic_is_way_off():
    camera = _camera()
    lidar_spec = LidarSensorSpec(horizontal_resolution_deg=0.2, range_accuracy_m=0.02)
    points = _well_behaved_points()
    T_way_off = np.eye(4)
    T_way_off[0, 3] = 500.0  # shifts everything far outside the image
    result = run_sanity_gate(points, T_CL=T_way_off, camera=camera, lidar_spec=lidar_spec)
    assert not result.passed
    fov_item = next(i for i in result.items if i.name == "fov_coverage_sufficient")
    assert not fov_item.passed
    assert any("FOV coverage" in w for w in result.warnings)


def test_sanity_gate_fails_on_empty_point_cloud():
    camera = _camera()
    lidar_spec = LidarSensorSpec(horizontal_resolution_deg=0.2, range_accuracy_m=0.02)
    result = run_sanity_gate(np.zeros((0, 3)), T_CL=np.eye(4), camera=camera, lidar_spec=lidar_spec)
    assert not result.passed
    depth_item = next(i for i in result.items if i.name == "depth_distribution_valid")
    assert not depth_item.passed


def test_sanity_gate_fails_on_non_finite_points():
    camera = _camera()
    lidar_spec = LidarSensorSpec(horizontal_resolution_deg=0.2, range_accuracy_m=0.02)
    points = _well_behaved_points(n=600)
    points[0, 0] = float("nan")
    points[1, 2] = float("inf")
    result = run_sanity_gate(points, T_CL=np.eye(4), camera=camera, lidar_spec=lidar_spec)
    depth_item = next(i for i in result.items if i.name == "depth_distribution_valid")
    assert not depth_item.passed
    assert not result.passed


def test_sanity_gate_fails_on_points_outside_sensor_range():
    camera = _camera()
    # sensor only rated for up to 3m, but points are at ~5m
    lidar_spec = LidarSensorSpec(horizontal_resolution_deg=0.2, range_accuracy_m=0.02,
                                  min_range_m=0.1, max_range_m=3.0)
    points = _well_behaved_points()
    result = run_sanity_gate(points, T_CL=np.eye(4), camera=camera, lidar_spec=lidar_spec)
    depth_item = next(i for i in result.items if i.name == "depth_distribution_valid")
    assert not depth_item.passed
    assert not result.passed


def test_sanity_gate_fails_below_min_valid_points():
    camera = _camera()
    lidar_spec = LidarSensorSpec(horizontal_resolution_deg=0.2, range_accuracy_m=0.02)
    points = _well_behaved_points(n=50)  # below default min_valid_points=500
    result = run_sanity_gate(points, T_CL=np.eye(4), camera=camera, lidar_spec=lidar_spec)
    count_item = next(i for i in result.items if i.name == "sufficient_valid_points")
    assert not count_item.passed
    assert not result.passed


def test_sanity_gate_detects_occlusion_violation():
    """
    Construct a scene where two clusters of points -- at very different
    depths -- project into the SAME small image region. Real depth-first
    visibility would only ever show the near surface there; a far surface
    "showing through" it at 10x the depth is exactly the kind of
    structurally-broken projection this check exists to catch (in practice
    this is what a badly wrong T_CL folding unrelated geometry onto the
    same viewing rays looks like).
    """
    camera = _camera()
    lidar_spec = LidarSensorSpec(horizontal_resolution_deg=0.2, range_accuracy_m=0.02, max_range_m=200.0)

    rng = np.random.RandomState(1)
    near = np.stack([
        rng.uniform(-0.3, 0.3, 800),
        rng.uniform(-0.3, 0.3, 800),
        np.full(800, 5.0),
    ], axis=1)
    far = np.stack([
        rng.uniform(-0.3, 0.3, 800),   # same x,y footprint as `near`
        rng.uniform(-0.3, 0.3, 800),
        np.full(800, 50.0),            # but 10x farther
    ], axis=1)
    points = np.vstack([near, far])

    result = run_sanity_gate(points, T_CL=np.eye(4), camera=camera, lidar_spec=lidar_spec)
    occlusion_item = next(i for i in result.items if i.name == "occlusion_plausible")
    assert not occlusion_item.passed
    assert result.occlusion_violation_ratio > 0.1
    assert not result.passed


def test_sanity_gate_occlusion_check_passes_for_single_surface():
    camera = _camera()
    lidar_spec = LidarSensorSpec(horizontal_resolution_deg=0.2, range_accuracy_m=0.02)
    points = _well_behaved_points()
    result = run_sanity_gate(points, T_CL=np.eye(4), camera=camera, lidar_spec=lidar_spec)
    occlusion_item = next(i for i in result.items if i.name == "occlusion_plausible")
    assert occlusion_item.passed
    assert result.occlusion_violation_ratio < 0.05


def test_sanity_gate_to_dict_structure():
    camera = _camera()
    lidar_spec = LidarSensorSpec(horizontal_resolution_deg=0.2, range_accuracy_m=0.02)
    points = _well_behaved_points()
    result = run_sanity_gate(points, T_CL=np.eye(4), camera=camera, lidar_spec=lidar_spec)
    d = result.to_dict()
    assert set(d.keys()) == {"passed", "num_input_points", "num_valid_points",
                              "fov_coverage_ratio", "occlusion_violation_ratio",
                              "checks", "warnings"}
    assert isinstance(d["checks"], list)
    assert all({"name", "passed", "detail", "value"} <= set(c.keys()) for c in d["checks"])


def test_sanity_gate_to_dict_sanitizes_nan():
    camera = _camera()
    lidar_spec = LidarSensorSpec(horizontal_resolution_deg=0.2, range_accuracy_m=0.02)
    result = run_sanity_gate(np.zeros((0, 3)), T_CL=np.eye(4), camera=camera, lidar_spec=lidar_spec)
    d = result.to_dict()
    assert d["fov_coverage_ratio"] is None or isinstance(d["fov_coverage_ratio"], float)
    import json
    json.dumps(d, allow_nan=False)  # must not raise -- no raw NaN/Inf left


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
