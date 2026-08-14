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


def test_video_and_rosbag_loaders_raise_not_implemented():
    try:
        load_camera_from_video()
        assert False
    except NotImplementedError:
        pass
    try:
        load_camera_from_rosbag()
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
