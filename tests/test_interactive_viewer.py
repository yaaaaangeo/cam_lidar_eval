import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from visualization.interactive_viewer import build_interactive_scene, build_interactive_scene_from_dataset
from tests.test_holdout_consistency import _make_camera, _make_image, _make_base_points_cam_frame, _make_dataset


def test_build_interactive_scene_returns_json_serializable_dict():
    camera = _make_camera()
    image = _make_image()
    points = _make_base_points_cam_frame()
    scene = build_interactive_scene(image, points, np.eye(4), camera)
    # Must round-trip through json.dumps without error -- this is the
    # actual contract report/html.py depends on.
    encoded = json.dumps(scene)
    assert isinstance(encoded, str)
    assert scene["num_points"] > 0


def test_build_interactive_scene_contains_expected_traces():
    camera = _make_camera()
    image = _make_image()
    points = _make_base_points_cam_frame()
    scene = build_interactive_scene(image, points, np.eye(4), camera)
    names = {trace["name"] for trace in scene["data"]}
    assert "LiDAR origin" in names
    assert "Camera origin" in names
    assert "Camera frustum" in names
    assert "Frustum far plane" in names
    assert "Colorized LiDAR points" in names


def test_build_interactive_scene_lidar_origin_at_zero():
    camera = _make_camera()
    image = _make_image()
    points = _make_base_points_cam_frame()
    scene = build_interactive_scene(image, points, np.eye(4), camera)
    lidar_trace = next(t for t in scene["data"] if t["name"] == "LiDAR origin")
    assert lidar_trace["x"] == [0.0]
    assert lidar_trace["y"] == [0.0]
    assert lidar_trace["z"] == [0.0]


def test_build_interactive_scene_camera_origin_reflects_translation():
    camera = _make_camera()
    image = _make_image()
    points = _make_base_points_cam_frame()
    T_CL = np.eye(4)
    T_CL[:3, 3] = [1.0, 2.0, 3.0]
    scene = build_interactive_scene(image, points, T_CL, camera)
    cam_trace = next(t for t in scene["data"] if t["name"] == "Camera origin")
    assert np.allclose([cam_trace["x"][0], cam_trace["y"][0], cam_trace["z"][0]], [-1.0, -2.0, -3.0])


def test_build_interactive_scene_respects_colorize_max_points():
    camera = _make_camera()
    image = _make_image()
    points = _make_base_points_cam_frame()
    scene = build_interactive_scene(image, points, np.eye(4), camera, colorize_max_points=10)
    points_trace = next(t for t in scene["data"] if t["name"] == "Colorized LiDAR points")
    assert len(points_trace["x"]) == 10
    assert scene["num_points"] == 10


def test_build_interactive_scene_no_colorized_trace_when_no_valid_points():
    camera = _make_camera()
    image = _make_image()
    points = _make_base_points_cam_frame()
    behind = points.copy()
    behind[:, 2] *= -1  # everything behind the camera -> nothing colorized
    scene = build_interactive_scene(image, behind, np.eye(4), camera)
    names = {trace["name"] for trace in scene["data"]}
    assert "Colorized LiDAR points" not in names
    # frustum/origins should still be present regardless
    assert "Camera frustum" in names
    assert scene["num_points"] == 0


def test_build_interactive_scene_from_dataset_convenience_wrapper():
    dataset = _make_dataset([0.0] * 10)
    scene = build_interactive_scene_from_dataset(dataset)
    json.dumps(scene)  # round-trip check
    names = {trace["name"] for trace in scene["data"]}
    assert "Camera frustum" in names


def test_build_interactive_scene_from_dataset_raises_on_empty_dataset():
    dataset = _make_dataset([])
    try:
        build_interactive_scene_from_dataset(dataset)
        assert False, "expected ValueError for empty dataset"
    except ValueError:
        pass
