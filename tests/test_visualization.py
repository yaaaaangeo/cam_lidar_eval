import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from visualization.overlay import render_overlay, render_overlay_from_result, encode_png, save_overlay_png
from visualization.trajectory import render_m4_trajectory_png
from visualization.histogram import render_error_histogram_png

from evaluation.edge_alignment import evaluate_edge_alignment
from evaluation.multiframe_consistency import evaluate_multiframe_consistency
from tests.test_holdout_consistency import _make_dataset, _make_lidar_spec, _make_camera, _make_image, _make_base_points_cam_frame


# ---------------------------------------------------------------------------
# overlay.py
# ---------------------------------------------------------------------------

def test_render_overlay_returns_same_shape_bgr_image():
    camera = _make_camera()
    image = _make_image()
    points = _make_base_points_cam_frame()
    result = evaluate_edge_alignment(image=image, points_lidar=points, T_CL=np.eye(4),
                                      camera=camera, lidar_spec=_make_lidar_spec(),
                                      depth_jump_threshold_m=1.0)
    assert result.classification != "FAIL"
    overlay = render_overlay(image, result.edge_point_pixels, result.edge_point_errors_px, result.floor_px)
    assert overlay.shape == image.shape
    assert overlay.dtype == image.dtype


def test_render_overlay_does_not_mutate_input():
    camera = _make_camera()
    image = _make_image()
    points = _make_base_points_cam_frame()
    result = evaluate_edge_alignment(image=image, points_lidar=points, T_CL=np.eye(4),
                                      camera=camera, lidar_spec=_make_lidar_spec(),
                                      depth_jump_threshold_m=1.0)
    original = image.copy()
    render_overlay(image, result.edge_point_pixels, result.edge_point_errors_px, result.floor_px)
    assert np.array_equal(image, original)


def test_render_overlay_actually_draws_something():
    camera = _make_camera()
    image = _make_image()
    points = _make_base_points_cam_frame()
    result = evaluate_edge_alignment(image=image, points_lidar=points, T_CL=np.eye(4),
                                      camera=camera, lidar_spec=_make_lidar_spec(),
                                      depth_jump_threshold_m=1.0)
    overlay = render_overlay(image, result.edge_point_pixels, result.edge_point_errors_px,
                              result.floor_px, draw_edge_map=False)
    assert not np.array_equal(overlay, image)  # points must have been drawn


def test_render_overlay_from_result_convenience_wrapper():
    camera = _make_camera()
    image = _make_image()
    points = _make_base_points_cam_frame()
    result = evaluate_edge_alignment(image=image, points_lidar=points, T_CL=np.eye(4),
                                      camera=camera, lidar_spec=_make_lidar_spec(),
                                      depth_jump_threshold_m=1.0)
    overlay = render_overlay_from_result(image, result)
    assert overlay is not None
    assert overlay.shape == image.shape


def test_render_overlay_from_result_returns_none_on_fail():
    camera = _make_camera()
    blank = np.zeros((camera.height, camera.width, 3), dtype=np.uint8)
    points = _make_base_points_cam_frame()
    result = evaluate_edge_alignment(image=blank, points_lidar=points, T_CL=np.eye(4),
                                      camera=camera, lidar_spec=_make_lidar_spec(),
                                      depth_jump_threshold_m=1.0)
    assert result.classification == "FAIL"
    overlay = render_overlay_from_result(blank, result)
    assert overlay is None


def test_encode_png_roundtrip():
    img = np.zeros((10, 10, 3), dtype=np.uint8)
    img[5, 5] = [255, 0, 0]
    png_bytes = encode_png(img)
    assert isinstance(png_bytes, bytes)
    assert png_bytes[:8] == b"\x89PNG\r\n\x1a\n"  # PNG magic number


def test_save_overlay_png_writes_file():
    import tempfile
    img = np.zeros((10, 10, 3), dtype=np.uint8)
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "test.png")
        save_overlay_png(img, path)
        assert os.path.exists(path)
        assert os.path.getsize(path) > 0


# ---------------------------------------------------------------------------
# trajectory.py
# ---------------------------------------------------------------------------

def test_render_m4_trajectory_png_returns_valid_png():
    dataset = _make_dataset([0.0] * 35)
    m4 = evaluate_multiframe_consistency(dataset, lidar_spec=_make_lidar_spec(), min_frames=30,
                                          edge_alignment_kwargs={"depth_jump_threshold_m": 1.0})
    png_bytes = render_m4_trajectory_png(m4)
    assert png_bytes is not None
    assert png_bytes[:8] == b"\x89PNG\r\n\x1a\n"


def test_render_m4_trajectory_png_none_when_no_valid_frames():
    dataset = _make_dataset([0.0] * 5)  # below min_frames -> FAIL, no frame_results
    m4 = evaluate_multiframe_consistency(dataset, lidar_spec=_make_lidar_spec(), min_frames=30)
    assert m4.classification == "FAIL"
    png_bytes = render_m4_trajectory_png(m4)
    assert png_bytes is None


def test_render_m4_trajectory_png_with_outliers():
    dataset = _make_dataset([0.0] * 39 + [0.1])
    m4 = evaluate_multiframe_consistency(dataset, lidar_spec=_make_lidar_spec(), min_frames=30,
                                          outlier_multiplier=4.0,
                                          edge_alignment_kwargs={"depth_jump_threshold_m": 1.0,
                                                                  "edge_radius_px": 8.0})
    assert m4.num_outlier_frames >= 1
    png_bytes = render_m4_trajectory_png(m4)
    assert png_bytes is not None


# ---------------------------------------------------------------------------
# histogram.py
# ---------------------------------------------------------------------------

def test_render_error_histogram_png_returns_valid_png():
    camera = _make_camera()
    image = _make_image()
    points = _make_base_points_cam_frame()
    result = evaluate_edge_alignment(image=image, points_lidar=points, T_CL=np.eye(4),
                                      camera=camera, lidar_spec=_make_lidar_spec(),
                                      depth_jump_threshold_m=1.0)
    png_bytes = render_error_histogram_png(result.edge_point_errors_px, result.floor_px)
    assert png_bytes is not None
    assert png_bytes[:8] == b"\x89PNG\r\n\x1a\n"


def test_render_error_histogram_png_none_on_empty_input():
    png_bytes = render_error_histogram_png(np.array([]), floor_px=1.0)
    assert png_bytes is None


def test_render_error_histogram_png_handles_nan_floor_gracefully():
    errors = np.array([1.0, 2.0, 3.0])
    png_bytes = render_error_histogram_png(errors, floor_px=float("nan"))
    assert png_bytes is not None  # should still render the histogram, just skip reference lines


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
