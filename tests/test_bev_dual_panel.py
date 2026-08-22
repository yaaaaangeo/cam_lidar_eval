import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from visualization.bev_dual_panel import (
    render_bev_dual_panel,
    render_bev_dual_panel_from_result,
    _recompute_edge_points_lidar,
)
from evaluation.edge_alignment import evaluate_edge_alignment
from tests.test_holdout_consistency import _make_camera, _make_image, _make_base_points_cam_frame, _make_lidar_spec


def _make_result():
    camera = _make_camera()
    image = _make_image()
    points = _make_base_points_cam_frame()
    result = evaluate_edge_alignment(
        image=image, points_lidar=points, T_CL=np.eye(4),
        camera=camera, lidar_spec=_make_lidar_spec(), depth_jump_threshold_m=1.0,
    )
    assert result.classification != "FAIL"
    return camera, image, points, result


def test_recompute_edge_points_lidar_matches_result_count():
    camera, image, points, result = _make_result()
    edge_points_lidar, edge_pixels = _recompute_edge_points_lidar(
        points, np.eye(4), camera, edge_radius_px=3.0, depth_jump_threshold_m=1.0, min_neighbors=3,
    )
    assert edge_points_lidar.shape[0] == result.edge_point_errors_px.shape[0]
    assert edge_pixels.shape[0] == result.edge_point_errors_px.shape[0]
    # pixels recomputed here should match the ones on the result (same computation)
    assert np.allclose(edge_pixels, result.edge_point_pixels)


def test_recompute_edge_points_lidar_positions_are_plausible():
    # Points should be actual LiDAR-frame points from the input cloud, not
    # some placeholder -- check they're a subset (row-wise) of the input.
    camera, image, points, result = _make_result()
    edge_points_lidar, _ = _recompute_edge_points_lidar(
        points, np.eye(4), camera, edge_radius_px=3.0, depth_jump_threshold_m=1.0, min_neighbors=3,
    )
    assert edge_points_lidar.shape[0] > 0
    for p in edge_points_lidar[:5]:
        assert np.any(np.all(np.isclose(points, p), axis=1))


def test_render_bev_dual_panel_returns_valid_png_bytes():
    camera, image, points, result = _make_result()
    png = render_bev_dual_panel(image, points, np.eye(4), camera, result, depth_jump_threshold_m=1.0)
    assert png is not None
    assert isinstance(png, bytes)
    assert png[:8] == b"\x89PNG\r\n\x1a\n"


def test_render_bev_dual_panel_returns_none_on_fail_result():
    camera, image, points, result = _make_result()
    result.classification = "FAIL"
    assert render_bev_dual_panel(image, points, np.eye(4), camera, result, depth_jump_threshold_m=1.0) is None


def test_render_bev_dual_panel_returns_none_on_mismatched_params():
    # Deliberately using a different depth_jump_threshold_m than the one
    # used to produce `result` should (usually) change which points count
    # as edge points, tripping the point-count mismatch guard. Use a very
    # different threshold to make the mismatch reliable.
    camera, image, points, result = _make_result()
    out = render_bev_dual_panel(image, points, np.eye(4), camera, result, depth_jump_threshold_m=50.0)
    assert out is None


def test_render_bev_dual_panel_from_result_convenience_wrapper():
    camera, image, points, result = _make_result()
    edge_kwargs = {"depth_jump_threshold_m": 1.0}
    png = render_bev_dual_panel_from_result(image, points, np.eye(4), camera, result, edge_kwargs=edge_kwargs)
    assert png is not None
    assert png[:8] == b"\x89PNG\r\n\x1a\n"


def test_render_bev_dual_panel_from_result_none_when_fail():
    camera, image, points, result = _make_result()
    result.classification = "FAIL"
    assert render_bev_dual_panel_from_result(image, points, np.eye(4), camera, result) is None


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
