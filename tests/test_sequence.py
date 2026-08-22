import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from visualization.sequence import render_sequence_gif, select_frame_indices
from tests.test_holdout_consistency import _make_dataset, _make_lidar_spec


def test_select_frame_indices_returns_all_when_fewer_than_max():
    assert select_frame_indices(5, 16) == [0, 1, 2, 3, 4]


def test_select_frame_indices_caps_at_max_and_includes_endpoints():
    indices = select_frame_indices(100, 10)
    assert len(indices) <= 10
    assert indices[0] == 0
    assert indices[-1] == 99
    assert indices == sorted(indices)


def test_select_frame_indices_empty_when_zero_frames():
    assert select_frame_indices(0, 10) == []


def test_render_sequence_gif_returns_valid_gif_bytes():
    dataset = _make_dataset([0.0] * 20)
    gif = render_sequence_gif(dataset, _make_lidar_spec(), edge_kwargs={"depth_jump_threshold_m": 1.0},
                               max_frames=5)
    assert gif is not None
    assert isinstance(gif, bytes)
    assert gif[:6] in (b"GIF87a", b"GIF89a")


def test_render_sequence_gif_respects_max_frames():
    dataset = _make_dataset([0.0] * 20)
    gif_small = render_sequence_gif(dataset, _make_lidar_spec(), edge_kwargs={"depth_jump_threshold_m": 1.0},
                                     max_frames=3)
    gif_large = render_sequence_gif(dataset, _make_lidar_spec(), edge_kwargs={"depth_jump_threshold_m": 1.0},
                                     max_frames=10)
    assert gif_small is not None and gif_large is not None
    # more frames -> larger (or at least not smaller) encoded GIF
    assert len(gif_large) >= len(gif_small)


def test_render_sequence_gif_returns_none_for_empty_dataset():
    dataset = _make_dataset([])
    assert render_sequence_gif(dataset, _make_lidar_spec()) is None


def test_render_sequence_gif_explicit_frame_indices():
    dataset = _make_dataset([0.0] * 20)
    gif = render_sequence_gif(dataset, _make_lidar_spec(), edge_kwargs={"depth_jump_threshold_m": 1.0},
                               frame_indices=[0, 5, 10])
    assert gif is not None
    assert gif[:6] in (b"GIF87a", b"GIF89a")


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
