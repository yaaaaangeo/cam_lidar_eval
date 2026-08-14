import sys
import os
import tempfile
import shutil
import struct

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from input.lidar import (
    LidarSensorSpec,
    read_pcd,
    read_ply_ascii,
    load_lidar_from_pcd_dir,
    load_lidar_from_rosbag,
)


def _write_ascii_pcd(path, points, with_intensity=False):
    n = len(points)
    fields = "x y z intensity" if with_intensity else "x y z"
    size = "4 4 4 4" if with_intensity else "4 4 4"
    typ = "F F F F" if with_intensity else "F F F"
    count = "1 1 1 1" if with_intensity else "1 1 1"
    header = f"""# .PCD v0.7
VERSION 0.7
FIELDS {fields}
SIZE {size}
TYPE {typ}
COUNT {count}
WIDTH {n}
HEIGHT 1
VIEWPOINT 0 0 0 1 0 0 0
POINTS {n}
DATA ascii
"""
    with open(path, "w") as f:
        f.write(header)
        for p in points:
            f.write(" ".join(str(v) for v in p) + "\n")


def _write_binary_pcd(path, points, with_intensity=False):
    n = len(points)
    fields = "x y z intensity" if with_intensity else "x y z"
    size = "4 4 4 4" if with_intensity else "4 4 4"
    typ = "F F F F" if with_intensity else "F F F"
    count = "1 1 1 1" if with_intensity else "1 1 1"
    header = f"""# .PCD v0.7
VERSION 0.7
FIELDS {fields}
SIZE {size}
TYPE {typ}
COUNT {count}
WIDTH {n}
HEIGHT 1
VIEWPOINT 0 0 0 1 0 0 0
POINTS {n}
DATA binary
"""
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        for p in points:
            f.write(struct.pack(f"<{len(p)}f", *p))


def _write_ascii_ply(path, points, with_intensity=False):
    n = len(points)
    with open(path, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {n}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        if with_intensity:
            f.write("property float intensity\n")
        f.write("end_header\n")
        for p in points:
            f.write(" ".join(str(v) for v in p) + "\n")


def test_read_pcd_ascii_xyz():
    tmpdir = tempfile.mkdtemp()
    try:
        path = os.path.join(tmpdir, "test.pcd")
        pts = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        _write_ascii_pcd(path, pts)
        arr = read_pcd(path)
        assert arr.shape == (2, 3)
        assert np.allclose(arr, pts, atol=1e-5)
    finally:
        shutil.rmtree(tmpdir)


def test_read_pcd_ascii_with_intensity():
    tmpdir = tempfile.mkdtemp()
    try:
        path = os.path.join(tmpdir, "test.pcd")
        pts = [[1.0, 2.0, 3.0, 0.5], [4.0, 5.0, 6.0, 0.9]]
        _write_ascii_pcd(path, pts, with_intensity=True)
        arr = read_pcd(path)
        assert arr.shape == (2, 4)
        assert np.allclose(arr, pts, atol=1e-5)
    finally:
        shutil.rmtree(tmpdir)


def test_read_pcd_binary_xyz():
    tmpdir = tempfile.mkdtemp()
    try:
        path = os.path.join(tmpdir, "test.pcd")
        pts = [[1.5, -2.5, 3.5], [0.0, 0.0, 0.0], [10.1, 20.2, 30.3]]
        _write_binary_pcd(path, pts)
        arr = read_pcd(path)
        assert arr.shape == (3, 3)
        assert np.allclose(arr, pts, atol=1e-4)
    finally:
        shutil.rmtree(tmpdir)


def test_read_pcd_binary_with_intensity():
    tmpdir = tempfile.mkdtemp()
    try:
        path = os.path.join(tmpdir, "test.pcd")
        pts = [[1.0, 2.0, 3.0, 0.42], [4.0, 5.0, 6.0, 0.77]]
        _write_binary_pcd(path, pts, with_intensity=True)
        arr = read_pcd(path)
        assert arr.shape == (2, 4)
        assert np.allclose(arr, pts, atol=1e-4)
    finally:
        shutil.rmtree(tmpdir)


def test_read_pcd_missing_xyz_raises():
    tmpdir = tempfile.mkdtemp()
    try:
        path = os.path.join(tmpdir, "bad.pcd")
        header = """VERSION 0.7
FIELDS intensity
SIZE 4
TYPE F
COUNT 1
WIDTH 1
HEIGHT 1
POINTS 1
DATA ascii
0.5
"""
        with open(path, "w") as f:
            f.write(header)
        try:
            read_pcd(path)
            assert False, "expected ValueError"
        except ValueError:
            pass
    finally:
        shutil.rmtree(tmpdir)


def test_read_ply_ascii_xyz():
    tmpdir = tempfile.mkdtemp()
    try:
        path = os.path.join(tmpdir, "test.ply")
        pts = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        _write_ascii_ply(path, pts)
        arr = read_ply_ascii(path)
        assert arr.shape == (2, 3)
        assert np.allclose(arr, pts, atol=1e-5)
    finally:
        shutil.rmtree(tmpdir)


def test_read_ply_ascii_with_intensity():
    tmpdir = tempfile.mkdtemp()
    try:
        path = os.path.join(tmpdir, "test.ply")
        pts = [[1.0, 2.0, 3.0, 0.3]]
        _write_ascii_ply(path, pts, with_intensity=True)
        arr = read_ply_ascii(path)
        assert arr.shape == (1, 4)
        assert np.allclose(arr, pts, atol=1e-5)
    finally:
        shutil.rmtree(tmpdir)


def test_load_lidar_from_pcd_dir_basic():
    tmpdir = tempfile.mkdtemp()
    try:
        _write_ascii_pcd(os.path.join(tmpdir, "10.0.pcd"), [[1, 2, 3]])
        _write_ascii_pcd(os.path.join(tmpdir, "10.1.pcd"), [[4, 5, 6]])
        spec = LidarSensorSpec(horizontal_resolution_deg=0.2, range_accuracy_m=0.02)
        result = load_lidar_from_pcd_dir(tmpdir, spec)
        assert len(result.frames) == 2
        assert set(f.timestamp for f in result.frames) == {10.0, 10.1}
        assert result.warnings == []  # spec fully provided
    finally:
        shutil.rmtree(tmpdir)


def test_load_lidar_from_pcd_dir_warns_on_missing_sensor_spec():
    tmpdir = tempfile.mkdtemp()
    try:
        _write_ascii_pcd(os.path.join(tmpdir, "1.pcd"), [[1, 2, 3]])
        spec = LidarSensorSpec()  # nothing provided
        result = load_lidar_from_pcd_dir(tmpdir, spec)
        assert len(result.warnings) == 2  # angular + range accuracy warnings
    finally:
        shutil.rmtree(tmpdir)


def test_load_lidar_from_pcd_dir_lazy_vs_eager():
    tmpdir = tempfile.mkdtemp()
    try:
        _write_ascii_pcd(os.path.join(tmpdir, "1.pcd"), [[1, 2, 3]])
        spec = LidarSensorSpec(horizontal_resolution_deg=0.2, range_accuracy_m=0.02)
        lazy_result = load_lidar_from_pcd_dir(tmpdir, spec, lazy=True)
        assert lazy_result.frames[0].points is None

        eager_result = load_lidar_from_pcd_dir(tmpdir, spec, lazy=False)
        assert eager_result.frames[0].points is not None
    finally:
        shutil.rmtree(tmpdir)


def test_load_lidar_from_pcd_dir_mixed_pcd_ply():
    tmpdir = tempfile.mkdtemp()
    try:
        _write_ascii_pcd(os.path.join(tmpdir, "1.pcd"), [[1, 2, 3]])
        _write_ascii_ply(os.path.join(tmpdir, "2.ply"), [[4, 5, 6]])
        spec = LidarSensorSpec(horizontal_resolution_deg=0.2, range_accuracy_m=0.02)
        result = load_lidar_from_pcd_dir(tmpdir, spec, lazy=False)
        assert len(result.frames) == 2
        pts = np.vstack([f.points for f in result.frames])
        assert np.allclose(sorted(pts[:, 0].tolist()), [1.0, 4.0])
    finally:
        shutil.rmtree(tmpdir)


def test_lidar_frame_load_dispatches_by_extension():
    tmpdir = tempfile.mkdtemp()
    try:
        pcd_path = os.path.join(tmpdir, "1.pcd")
        _write_ascii_pcd(pcd_path, [[1, 2, 3]])
        from input.lidar import LidarFrame
        frame = LidarFrame(timestamp=1.0, path=pcd_path)
        pts = frame.load()
        assert pts.shape == (1, 3)
        assert frame.points is not None  # cached
    finally:
        shutil.rmtree(tmpdir)


def test_rosbag_lidar_loader_raises_not_implemented():
    try:
        load_lidar_from_rosbag()
        assert False
    except NotImplementedError:
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
