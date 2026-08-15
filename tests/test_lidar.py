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


# ---------------------------------------------------------------------------
# rosbag loader (requires the optional `rosbags` package -- if it isn't
# installed, these tests are skipped rather than failing, since rosbag
# support is opt-in: `pip install "cam-lidar-eval[rosbag]"`)
# ---------------------------------------------------------------------------

try:
    from rosbags.rosbag2 import Writer as _Rosbag2Writer
    from rosbags.typesys import Stores as _RosbagStores, get_typestore as _get_rosbag_typestore
    _ROSBAGS_AVAILABLE = True
except ImportError:
    _ROSBAGS_AVAILABLE = False


def _write_pointcloud2_bag(bag_path, frames, topic="/lidar/points", with_intensity=False):
    """
    Write a synthetic rosbag2 containing one sensor_msgs/msg/PointCloud2
    message per frame. `frames` is a list of (points, stamp_sec,
    stamp_nanosec, bag_timestamp_ns) tuples; points is an (N,3) or (N,4)
    array (x,y,z[,intensity]), used to build the corresponding
    PointField layout so this exercises the same field-offset-driven
    parsing path _pointcloud2_to_array() uses for real sensor data.
    """
    ts = _get_rosbag_typestore(_RosbagStores.ROS2_HUMBLE)
    PointField = ts.types["sensor_msgs/msg/PointField"]
    PointCloud2 = ts.types["sensor_msgs/msg/PointCloud2"]
    Header = ts.types["std_msgs/msg/Header"]
    Time = ts.types["builtin_interfaces/msg/Time"]

    if with_intensity:
        fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name="intensity", offset=12, datatype=PointField.FLOAT32, count=1),
        ]
        point_step = 16
    else:
        fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
        ]
        point_step = 12

    with _Rosbag2Writer(bag_path, version=9) as writer:
        conn = writer.add_connection(topic, PointCloud2.__msgtype__, typestore=ts)
        for points, stamp_sec, stamp_nanosec, bag_ts_ns in frames:
            points = np.asarray(points, dtype=np.float32)
            n = points.shape[0]
            header = Header(stamp=Time(sec=stamp_sec, nanosec=stamp_nanosec), frame_id="lidar")
            msg = PointCloud2(
                header=header, height=1, width=n, fields=fields,
                is_bigendian=False, point_step=point_step, row_step=point_step * n,
                data=np.frombuffer(points.tobytes(), dtype=np.uint8), is_dense=True,
            )
            raw = ts.serialize_cdr(msg, PointCloud2.__msgtype__)
            writer.write(conn, bag_ts_ns, raw)


def test_rosbag_lidar_loader_reads_points_and_timestamps():
    if not _ROSBAGS_AVAILABLE:
        return  # opt-in dependency not installed; nothing to test here
    tmpdir = tempfile.mkdtemp()
    try:
        bag_path = os.path.join(tmpdir, "bag")
        pts = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        _write_pointcloud2_bag(bag_path, [(pts, 10, 500_000_000, 10_500_000_000)])

        spec = LidarSensorSpec(horizontal_resolution_deg=0.2, range_accuracy_m=0.02)
        result = load_lidar_from_rosbag(bag_path, spec)

        assert result.lidar.source.kind == "rosbag"
        assert result.lidar.source.topic == "/lidar/points"
        assert len(result.frames) == 1
        assert result.frames[0].timestamp == 10.5  # sec + nanosec/1e9, from header.stamp
        np.testing.assert_allclose(result.frames[0].points, pts)
    finally:
        shutil.rmtree(tmpdir)


def test_rosbag_lidar_loader_reads_intensity_field():
    if not _ROSBAGS_AVAILABLE:
        return
    tmpdir = tempfile.mkdtemp()
    try:
        bag_path = os.path.join(tmpdir, "bag")
        pts = [[1.0, 2.0, 3.0, 0.75]]
        _write_pointcloud2_bag(bag_path, [(pts, 1, 0, 1_000_000_000)], with_intensity=True)

        spec = LidarSensorSpec()
        result = load_lidar_from_rosbag(bag_path, spec)
        assert result.frames[0].points.shape == (1, 4)
        np.testing.assert_allclose(result.frames[0].points, pts)
    finally:
        shutil.rmtree(tmpdir)


def test_rosbag_lidar_loader_multiple_frames_sorted_by_timestamp():
    if not _ROSBAGS_AVAILABLE:
        return
    tmpdir = tempfile.mkdtemp()
    try:
        bag_path = os.path.join(tmpdir, "bag")
        # write out of chronological order -- loader should sort by timestamp
        frames = [
            ([[3.0, 3.0, 3.0]], 3, 0, 3_000_000_000),
            ([[1.0, 1.0, 1.0]], 1, 0, 1_000_000_000),
            ([[2.0, 2.0, 2.0]], 2, 0, 2_000_000_000),
        ]
        _write_pointcloud2_bag(bag_path, frames)

        spec = LidarSensorSpec()
        result = load_lidar_from_rosbag(bag_path, spec)
        assert [fr.timestamp for fr in result.frames] == [1.0, 2.0, 3.0]
    finally:
        shutil.rmtree(tmpdir)


def test_rosbag_lidar_loader_unstamped_header_falls_back_to_bag_time_with_warning():
    if not _ROSBAGS_AVAILABLE:
        return
    tmpdir = tempfile.mkdtemp()
    try:
        bag_path = os.path.join(tmpdir, "bag")
        # stamp_sec=0, stamp_nanosec=0 -> unstamped; bag_ts_ns=7.25e9 should be used instead
        _write_pointcloud2_bag(bag_path, [([[0.0, 0.0, 0.0]], 0, 0, 7_250_000_000)])

        spec = LidarSensorSpec()
        result = load_lidar_from_rosbag(bag_path, spec)
        assert abs(result.frames[0].timestamp - 7.25) < 1e-6
        assert any("unstamped header" in w for w in result.warnings)
    finally:
        shutil.rmtree(tmpdir)


def test_rosbag_lidar_loader_missing_topic_raises_value_error():
    if not _ROSBAGS_AVAILABLE:
        return
    tmpdir = tempfile.mkdtemp()
    try:
        bag_path = os.path.join(tmpdir, "bag")
        _write_pointcloud2_bag(bag_path, [([[1.0, 2.0, 3.0]], 1, 0, 1_000_000_000)],
                                topic="/lidar/points")
        spec = LidarSensorSpec()
        try:
            load_lidar_from_rosbag(bag_path, spec, topic="/wrong/topic")
            assert False, "expected ValueError"
        except ValueError as e:
            assert "/wrong/topic" in str(e)
    finally:
        shutil.rmtree(tmpdir)


def test_rosbag_lidar_loader_ambiguous_topic_without_selection_raises():
    if not _ROSBAGS_AVAILABLE:
        return
    tmpdir = tempfile.mkdtemp()
    try:
        bag_path = os.path.join(tmpdir, "bag")
        ts = _get_rosbag_typestore(_RosbagStores.ROS2_HUMBLE)
        PointField = ts.types["sensor_msgs/msg/PointField"]
        PointCloud2 = ts.types["sensor_msgs/msg/PointCloud2"]
        Header = ts.types["std_msgs/msg/Header"]
        Time = ts.types["builtin_interfaces/msg/Time"]
        fields = [PointField(name=n, offset=o, datatype=PointField.FLOAT32, count=1)
                  for n, o in [("x", 0), ("y", 4), ("z", 8)]]
        pts = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
        with _Rosbag2Writer(bag_path, version=9) as writer:
            for topic in ("/lidar/front", "/lidar/rear"):
                conn = writer.add_connection(topic, PointCloud2.__msgtype__, typestore=ts)
                header = Header(stamp=Time(sec=1, nanosec=0), frame_id="lidar")
                msg = PointCloud2(header=header, height=1, width=1, fields=fields,
                                   is_bigendian=False, point_step=12, row_step=12,
                                   data=np.frombuffer(pts.tobytes(), dtype=np.uint8), is_dense=True)
                writer.write(conn, 1_000_000_000, ts.serialize_cdr(msg, PointCloud2.__msgtype__))

        spec = LidarSensorSpec()
        try:
            load_lidar_from_rosbag(bag_path, spec)  # no topic specified -> ambiguous
            assert False, "expected ValueError"
        except ValueError as e:
            assert "/lidar/front" in str(e) and "/lidar/rear" in str(e)
    finally:
        shutil.rmtree(tmpdir)


def test_rosbag_lidar_loader_missing_path_raises_file_not_found():
    if not _ROSBAGS_AVAILABLE:
        return
    try:
        load_lidar_from_rosbag("/definitely/does/not/exist", LidarSensorSpec())
        assert False, "expected FileNotFoundError"
    except FileNotFoundError:
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
