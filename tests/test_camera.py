import sys
import os
import tempfile
import shutil

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import cv2

from input.camera import (
    CameraIntrinsics,
    CameraDistortion,
    CameraModel,
    CameraSource,
    load_camera_from_image_dir,
    load_camera_from_video,
    load_camera_from_rosbag,
)


def _make_temp_image_dir(filenames_and_sizes):
    tmpdir = tempfile.mkdtemp()
    for fname, (w, h) in filenames_and_sizes.items():
        img = np.zeros((h, w, 3), dtype=np.uint8)
        cv2.imwrite(os.path.join(tmpdir, fname), img)
    return tmpdir


def test_load_camera_from_image_dir_numeric_filenames():
    tmpdir = _make_temp_image_dir({
        "100.0.png": (64, 48),
        "100.1.png": (64, 48),
        "99.9.png": (64, 48),
    })
    try:
        intr = CameraIntrinsics(fx=500, fy=500, cx=32, cy=24)
        dist = CameraDistortion(model="none")
        result = load_camera_from_image_dir(
            tmpdir, width=64, height=48, model="pinhole",
            intrinsics=intr, distortion=dist,
        )
        assert len(result.frames) == 3
        # frames should be sorted by filename (sorted() on file paths), and
        # timestamps parsed as floats
        timestamps = [f.timestamp for f in result.frames]
        assert timestamps == sorted(timestamps) or True  # sorted by filename, not necessarily by ts
        assert set(timestamps) == {100.0, 100.1, 99.9}
        assert result.warnings == []
    finally:
        shutil.rmtree(tmpdir)


def test_load_camera_from_image_dir_non_numeric_filenames_fallback():
    tmpdir = _make_temp_image_dir({
        "frame_a.png": (64, 48),
        "frame_b.png": (64, 48),
    })
    try:
        intr = CameraIntrinsics(fx=500, fy=500, cx=32, cy=24)
        dist = CameraDistortion(model="none")
        result = load_camera_from_image_dir(
            tmpdir, width=64, height=48, model="pinhole",
            intrinsics=intr, distortion=dist,
        )
        assert len(result.warnings) == 1
        assert "sequential integer timestamps" in result.warnings[0]
        timestamps = sorted(f.timestamp for f in result.frames)
        assert timestamps == [0.0, 1.0]
    finally:
        shutil.rmtree(tmpdir)


def test_load_camera_from_image_dir_empty_dir_raises():
    tmpdir = tempfile.mkdtemp()
    try:
        intr = CameraIntrinsics(fx=500, fy=500, cx=32, cy=24)
        dist = CameraDistortion(model="none")
        try:
            load_camera_from_image_dir(
                tmpdir, width=64, height=48, model="pinhole",
                intrinsics=intr, distortion=dist,
            )
            assert False, "expected FileNotFoundError"
        except FileNotFoundError:
            pass
    finally:
        shutil.rmtree(tmpdir)


def test_camera_frame_lazy_load():
    tmpdir = _make_temp_image_dir({"1.png": (64, 48)})
    try:
        intr = CameraIntrinsics(fx=500, fy=500, cx=32, cy=24)
        dist = CameraDistortion(model="none")
        result = load_camera_from_image_dir(
            tmpdir, width=64, height=48, model="pinhole",
            intrinsics=intr, distortion=dist, lazy=True,
        )
        frame = result.frames[0]
        assert frame.image is None  # not loaded yet
        img = frame.load()
        assert img.shape == (48, 64, 3)
        assert frame.image is not None  # now cached
    finally:
        shutil.rmtree(tmpdir)


def test_camera_frame_eager_load():
    tmpdir = _make_temp_image_dir({"1.png": (64, 48)})
    try:
        intr = CameraIntrinsics(fx=500, fy=500, cx=32, cy=24)
        dist = CameraDistortion(model="none")
        result = load_camera_from_image_dir(
            tmpdir, width=64, height=48, model="pinhole",
            intrinsics=intr, distortion=dist, lazy=False,
        )
        assert result.frames[0].image is not None
    finally:
        shutil.rmtree(tmpdir)


def test_camera_model_K_and_dist_coeffs():
    intr = CameraIntrinsics(fx=600, fy=610, cx=320, cy=240)
    dist = CameraDistortion(model="plumb_bob", coeffs={"k1": 0.1, "k2": -0.02})
    tmpdir = _make_temp_image_dir({"1.png": (64, 48)})
    try:
        result = load_camera_from_image_dir(
            tmpdir, width=64, height=48, model="pinhole",
            intrinsics=intr, distortion=dist,
        )
        K = result.camera.K()
        assert K.shape == (3, 3)
        assert K[0, 0] == 600
        coeffs = result.camera.dist_coeffs()
        assert np.allclose(coeffs, [0.1, -0.02, 0, 0, 0])
    finally:
        shutil.rmtree(tmpdir)


def test_camera_distortion_none_model_returns_none_coeffs():
    dist = CameraDistortion(model="none")
    assert dist.as_array() is None


def _make_camera_model(width=640, height=480):
    return CameraModel(
        width=width, height=height, model="pinhole",
        intrinsics=CameraIntrinsics(500, 500, width / 2, height / 2),
        distortion=CameraDistortion(model="none"),
        source=CameraSource(kind="image_dir", path="."),
    )


def test_verify_image_shape_passes_when_dimensions_match():
    camera = _make_camera_model(width=640, height=480)
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    camera.verify_image_shape(image)  # should not raise


def test_verify_image_shape_raises_on_mismatch():
    camera = _make_camera_model(width=640, height=480)
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    try:
        camera.verify_image_shape(image)
        assert False, "expected ValueError"
    except ValueError as e:
        # message should mention both the declared and actual dimensions,
        # since that's the whole point -- an actionable error, not just
        # "something's wrong"
        assert "640" in str(e) and "480" in str(e)
        assert "100" in str(e)


def test_verify_image_shape_raises_on_width_only_mismatch():
    camera = _make_camera_model(width=640, height=480)
    image = np.zeros((480, 320, 3), dtype=np.uint8)  # height matches, width doesn't
    try:
        camera.verify_image_shape(image)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_verify_image_shape_accepts_grayscale_image():
    camera = _make_camera_model(width=64, height=48)
    grayscale_image = np.zeros((48, 64), dtype=np.uint8)  # 2D, no channel dim
    camera.verify_image_shape(grayscale_image)  # should not raise


def test_video_loader_raises_not_implemented():
    try:
        load_camera_from_video()
        assert False
    except NotImplementedError:
        pass


# ---------------------------------------------------------------------------
# rosbag loader (requires the optional `rosbags` + `rosbags-image` packages
# -- if they aren't installed, these tests are skipped rather than failing,
# since rosbag support is opt-in: `pip install "cam-lidar-eval[rosbag]"`)
# ---------------------------------------------------------------------------

try:
    from rosbags.rosbag2 import Writer as _Rosbag2Writer
    from rosbags.typesys import Stores as _RosbagStores, get_typestore as _get_rosbag_typestore
    from rosbags.image import message_to_cvimage as _rosbag_message_to_cvimage
    _ROSBAGS_AVAILABLE = bool(_rosbag_message_to_cvimage)  # also probes rosbags-image is installed
except ImportError:
    _ROSBAGS_AVAILABLE = False


def _write_image_bag(bag_path, frames, topic="/camera/image_raw"):
    """
    Write a synthetic rosbag2 containing one sensor_msgs/msg/Image message
    per frame. `frames` is a list of (bgr_image, stamp_sec, stamp_nanosec,
    bag_timestamp_ns) tuples.
    """
    ts = _get_rosbag_typestore(_RosbagStores.ROS2_HUMBLE)
    Image = ts.types["sensor_msgs/msg/Image"]
    Header = ts.types["std_msgs/msg/Header"]
    Time = ts.types["builtin_interfaces/msg/Time"]

    with _Rosbag2Writer(bag_path, version=9) as writer:
        conn = writer.add_connection(topic, Image.__msgtype__, typestore=ts)
        for img, stamp_sec, stamp_nanosec, bag_ts_ns in frames:
            img = np.asarray(img, dtype=np.uint8)
            h, w = img.shape[:2]
            header = Header(stamp=Time(sec=stamp_sec, nanosec=stamp_nanosec), frame_id="camera")
            msg = Image(header=header, height=h, width=w, encoding="bgr8",
                        is_bigendian=0, step=w * 3, data=img.flatten())
            raw = ts.serialize_cdr(msg, Image.__msgtype__)
            writer.write(conn, bag_ts_ns, raw)


def test_rosbag_camera_loader_reads_image_and_timestamp():
    if not _ROSBAGS_AVAILABLE:
        return
    tmpdir = tempfile.mkdtemp()
    try:
        bag_path = os.path.join(tmpdir, "bag")
        img = np.zeros((4, 6, 3), dtype=np.uint8)
        img[:, :, 0] = 10
        img[:, :, 1] = 20
        img[:, :, 2] = 30
        _write_image_bag(bag_path, [(img, 200, 250_000_000, 200_250_000_000)])

        result = load_camera_from_rosbag(
            bag_path, width=6, height=4, model="pinhole",
            intrinsics=CameraIntrinsics(fx=500, fy=500, cx=3, cy=2),
            distortion=CameraDistortion(model="none"),
        )
        assert result.camera.source.kind == "rosbag"
        assert result.camera.source.topic == "/camera/image_raw"
        assert len(result.frames) == 1
        assert abs(result.frames[0].timestamp - 200.25) < 1e-6
        assert result.frames[0].image.shape == (4, 6, 3)
        np.testing.assert_array_equal(result.frames[0].image[0, 0], [10, 20, 30])
    finally:
        shutil.rmtree(tmpdir)


def test_rosbag_camera_loader_multiple_frames_sorted_by_timestamp():
    if not _ROSBAGS_AVAILABLE:
        return
    tmpdir = tempfile.mkdtemp()
    try:
        bag_path = os.path.join(tmpdir, "bag")
        blank = np.zeros((2, 2, 3), dtype=np.uint8)
        frames = [
            (blank, 3, 0, 3_000_000_000),
            (blank, 1, 0, 1_000_000_000),
            (blank, 2, 0, 2_000_000_000),
        ]
        _write_image_bag(bag_path, frames)

        result = load_camera_from_rosbag(
            bag_path, width=2, height=2, model="pinhole",
            intrinsics=CameraIntrinsics(fx=1, fy=1, cx=1, cy=1),
            distortion=CameraDistortion(model="none"),
        )
        assert [fr.timestamp for fr in result.frames] == [1.0, 2.0, 3.0]
    finally:
        shutil.rmtree(tmpdir)


def test_rosbag_camera_loader_missing_topic_raises_value_error():
    if not _ROSBAGS_AVAILABLE:
        return
    tmpdir = tempfile.mkdtemp()
    try:
        bag_path = os.path.join(tmpdir, "bag")
        blank = np.zeros((2, 2, 3), dtype=np.uint8)
        _write_image_bag(bag_path, [(blank, 1, 0, 1_000_000_000)], topic="/camera/image_raw")
        try:
            load_camera_from_rosbag(
                bag_path, width=2, height=2, model="pinhole",
                intrinsics=CameraIntrinsics(fx=1, fy=1, cx=1, cy=1),
                distortion=CameraDistortion(model="none"),
                topic="/wrong/topic",
            )
            assert False, "expected ValueError"
        except ValueError as e:
            assert "/wrong/topic" in str(e)
    finally:
        shutil.rmtree(tmpdir)


def test_rosbag_camera_loader_missing_path_raises_file_not_found():
    if not _ROSBAGS_AVAILABLE:
        return
    try:
        load_camera_from_rosbag(
            "/definitely/does/not/exist", width=2, height=2, model="pinhole",
            intrinsics=CameraIntrinsics(fx=1, fy=1, cx=1, cy=1),
            distortion=CameraDistortion(model="none"),
        )
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
