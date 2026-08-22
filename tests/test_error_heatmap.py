import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from visualization.error_heatmap import (
    compute_error_grid,
    render_error_heatmap,
    render_error_heatmap_from_result,
    ErrorGridResult,
)
from evaluation.edge_alignment import evaluate_edge_alignment
from tests.test_holdout_consistency import _make_camera, _make_image, _make_base_points_cam_frame, _make_lidar_spec


WIDTH, HEIGHT = 640, 480


def _make_result():
    camera = _make_camera()
    image = _make_image()
    points = _make_base_points_cam_frame()
    result = evaluate_edge_alignment(
        image=image, points_lidar=points, T_CL=np.eye(4),
        camera=camera, lidar_spec=_make_lidar_spec(), depth_jump_threshold_m=1.0,
    )
    assert result.classification != "FAIL"
    return image, result


def test_compute_error_grid_basic_shape_and_counts():
    pixels = np.array([[10.0, 10.0], [15.0, 12.0], [600.0, 470.0]])
    errors = np.array([1.0, 2.0, 5.0])
    grid = compute_error_grid(HEIGHT, WIDTH, pixels, errors, floor_px=1.0,
                               grid_rows=4, grid_cols=4, min_points_per_cell=1)
    assert grid.mean_err_px.shape == (4, 4)
    assert grid.counts.shape == (4, 4)
    assert grid.num_points == 3
    # top-left cell should hold the two near-(10,10)/(15,12) points -> mean 1.5
    assert np.isclose(grid.mean_err_px[0, 0], 1.5)
    # bottom-right cell should hold the (600, 470) point -> mean 5.0
    assert np.isclose(grid.mean_err_px[-1, -1], 5.0)


def test_compute_error_grid_excludes_sparse_cells():
    pixels = np.array([[10.0, 10.0], [600.0, 470.0], [605.0, 475.0], [610.0, 460.0]])
    errors = np.array([1.0, 2.0, 3.0, 4.0])
    grid = compute_error_grid(HEIGHT, WIDTH, pixels, errors, floor_px=1.0,
                               grid_rows=4, grid_cols=4, min_points_per_cell=2)
    # top-left cell only has 1 point -> excluded (NaN)
    assert np.isnan(grid.mean_err_px[0, 0])
    # bottom-right cell has 3 points -> included
    assert not np.isnan(grid.mean_err_px[-1, -1])
    assert grid.num_populated_cells == 1


def test_compute_error_grid_empty_points():
    grid = compute_error_grid(HEIGHT, WIDTH, np.zeros((0, 2)), np.zeros((0,)), floor_px=1.0)
    assert grid.num_points == 0
    assert grid.num_populated_cells == 0
    assert np.all(np.isnan(grid.mean_err_px))


def test_render_error_heatmap_returns_same_shape_bgr_image():
    image, result = _make_result()
    grid = compute_error_grid(image.shape[0], image.shape[1], result.edge_point_pixels,
                               result.edge_point_errors_px, result.floor_px, min_points_per_cell=1)
    heatmap = render_error_heatmap(image, grid)
    assert heatmap is not None
    assert heatmap.shape == image.shape
    assert heatmap.dtype == image.dtype


def test_render_error_heatmap_does_not_mutate_input():
    image, result = _make_result()
    original = image.copy()
    grid = compute_error_grid(image.shape[0], image.shape[1], result.edge_point_pixels,
                               result.edge_point_errors_px, result.floor_px, min_points_per_cell=1)
    render_error_heatmap(image, grid)
    assert np.array_equal(image, original)


def test_render_error_heatmap_returns_none_when_no_populated_cells():
    empty_grid = ErrorGridResult(
        mean_err_px=np.full((4, 4), np.nan), counts=np.zeros((4, 4), dtype=int),
        grid_rows=4, grid_cols=4, cell_height=HEIGHT / 4, cell_width=WIDTH / 4,
        floor_px=1.0, min_points_per_cell=3, num_points=0, num_populated_cells=0,
    )
    image = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    assert render_error_heatmap(image, empty_grid) is None


def test_render_error_heatmap_returns_none_when_floor_px_invalid():
    grid = ErrorGridResult(
        mean_err_px=np.array([[1.0]]), counts=np.array([[5]]),
        grid_rows=1, grid_cols=1, cell_height=HEIGHT, cell_width=WIDTH,
        floor_px=0.0, min_points_per_cell=3, num_points=5, num_populated_cells=1,
    )
    image = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    assert render_error_heatmap(image, grid) is None


def test_render_error_heatmap_low_error_cell_is_greenish():
    # A cell with error well below the GOOD threshold should render close
    # to the GOOD color (green-dominant in BGR: G channel highest).
    pixels = np.tile(np.array([[50.0, 50.0]]), (5, 1))
    errors = np.full(5, 0.01)  # tiny error relative to floor
    image = np.full((HEIGHT, WIDTH, 3), 128, dtype=np.uint8)  # mid-gray canvas
    grid = compute_error_grid(HEIGHT, WIDTH, pixels, errors, floor_px=1.0,
                               grid_rows=4, grid_cols=4, min_points_per_cell=1)
    heatmap = render_error_heatmap(image, grid, alpha=1.0, draw_grid_lines=False)
    # cell (0,0) covers roughly rows/cols 0..HEIGHT/4, 0..WIDTH/4
    sample = heatmap[10, 10]  # B, G, R
    assert sample[1] > sample[2]  # green channel should dominate over red


def test_render_error_heatmap_high_error_cell_is_reddish():
    pixels = np.tile(np.array([[50.0, 50.0]]), (5, 1))
    errors = np.full(5, 100.0)  # huge error relative to floor
    image = np.full((HEIGHT, WIDTH, 3), 128, dtype=np.uint8)
    grid = compute_error_grid(HEIGHT, WIDTH, pixels, errors, floor_px=1.0,
                               grid_rows=4, grid_cols=4, min_points_per_cell=1)
    heatmap = render_error_heatmap(image, grid, alpha=1.0, draw_grid_lines=False)
    sample = heatmap[10, 10]  # B, G, R
    assert sample[2] > sample[1]  # red channel should dominate over green


def test_render_error_heatmap_from_result_convenience_wrapper():
    image, result = _make_result()
    heatmap = render_error_heatmap_from_result(image, result, min_points_per_cell=1)
    assert heatmap is not None
    assert heatmap.shape == image.shape


def test_render_error_heatmap_from_result_none_when_fail():
    image, result = _make_result()
    result.classification = "FAIL"
    assert render_error_heatmap_from_result(image, result) is None


# ---------------------------------------------------------------------------
# Vectorized compute_error_grid vs. a naive reference implementation
# ---------------------------------------------------------------------------

def _compute_error_grid_naive(image_height, image_width, edge_pixels, errors_px, floor_px,
                               grid_rows, grid_cols, min_points_per_cell):
    """Reference implementation matching the ORIGINAL (pre-vectorization)
    compute_error_grid: a plain double loop over grid cells, re-masking
    the full point array on every iteration. Kept only for the
    equivalence test below -- compute_error_grid itself now uses
    np.bincount instead."""
    edge_pixels = np.asarray(edge_pixels, dtype=np.float64)
    errors_px = np.asarray(errors_px, dtype=np.float64)
    cell_h = image_height / grid_rows
    cell_w = image_width / grid_cols
    mean_grid = np.full((grid_rows, grid_cols), np.nan)
    counts = np.zeros((grid_rows, grid_cols), dtype=int)
    if edge_pixels.shape[0] > 0:
        col_idx = np.clip((edge_pixels[:, 0] / cell_w).astype(int), 0, grid_cols - 1)
        row_idx = np.clip((edge_pixels[:, 1] / cell_h).astype(int), 0, grid_rows - 1)
        in_bounds = (
            (edge_pixels[:, 0] >= 0) & (edge_pixels[:, 0] < image_width) &
            (edge_pixels[:, 1] >= 0) & (edge_pixels[:, 1] < image_height)
        )
        for r in range(grid_rows):
            for c in range(grid_cols):
                mask = in_bounds & (row_idx == r) & (col_idx == c)
                n = int(mask.sum())
                counts[r, c] = n
                if n >= min_points_per_cell:
                    mean_grid[r, c] = float(errors_px[mask].mean())
    return mean_grid, counts


def test_compute_error_grid_matches_naive_reference_random():
    # Randomized equivalence test (same discipline as the project's other
    # performance rewrites -- see extract_lidar_edge_points): the
    # vectorized np.bincount version must produce bit-for-bit-equivalent
    # counts and means to the original nested-loop version across many
    # random configurations, not just the hand-picked cases above.
    rng = np.random.default_rng(0)
    for trial in range(20):
        h, w = int(rng.integers(50, 500)), int(rng.integers(50, 500))
        grid_rows, grid_cols = int(rng.integers(2, 12)), int(rng.integers(2, 12))
        n_points = int(rng.integers(0, 300))
        min_points_per_cell = int(rng.integers(1, 5))

        pixels = np.column_stack([
            rng.uniform(-10, w + 10, n_points),  # include some out-of-bounds points
            rng.uniform(-10, h + 10, n_points),
        ])
        errors = rng.uniform(0, 20, n_points)

        grid = compute_error_grid(h, w, pixels, errors, floor_px=1.0,
                                   grid_rows=grid_rows, grid_cols=grid_cols,
                                   min_points_per_cell=min_points_per_cell)
        naive_mean, naive_counts = _compute_error_grid_naive(
            h, w, pixels, errors, floor_px=1.0,
            grid_rows=grid_rows, grid_cols=grid_cols, min_points_per_cell=min_points_per_cell,
        )

        assert np.array_equal(grid.counts, naive_counts), f"trial {trial}: counts mismatch"
        assert np.allclose(grid.mean_err_px, naive_mean, equal_nan=True, rtol=1e-9), \
            f"trial {trial}: mean_err_px mismatch"


def test_compute_error_grid_vectorized_is_faster_at_fine_grid_resolution():
    # The whole point of the rewrite: a fine grid (e.g. 40x40) used to
    # scale as O(rows*cols*N) with a Python-level double loop; confirm
    # the vectorized version is now meaningfully faster on a large point
    # set + fine grid, rather than just "not obviously slower".
    import time
    rng = np.random.default_rng(1)
    n_points = 20_000
    h, w = 1080, 1920
    pixels = np.column_stack([rng.uniform(0, w, n_points), rng.uniform(0, h, n_points)])
    errors = rng.uniform(0, 20, n_points)

    t0 = time.perf_counter()
    compute_error_grid(h, w, pixels, errors, floor_px=1.0, grid_rows=40, grid_cols=40, min_points_per_cell=1)
    vectorized_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    _compute_error_grid_naive(h, w, pixels, errors, floor_px=1.0, grid_rows=40, grid_cols=40, min_points_per_cell=1)
    naive_time = time.perf_counter() - t0

    assert vectorized_time < naive_time * 0.5, (
        f"expected the vectorized version to be meaningfully faster: "
        f"vectorized={vectorized_time:.4f}s, naive={naive_time:.4f}s"
    )


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
