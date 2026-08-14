import sys
import os
import json as _json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from evaluation.sanity_gate import run_sanity_gate
from evaluation.edge_alignment import evaluate_edge_alignment
from evaluation.holdout_consistency import evaluate_holdout_consistency
from evaluation.multiframe_consistency import evaluate_multiframe_consistency
from quality.quality_score import compute_quality_score
from report.builder import build_report
from report.json import to_json_string
from report.html import render_html_report

from tests.test_holdout_consistency import _make_dataset, _make_lidar_spec


def test_m0_result_flows_through_full_report_pipeline():
    dataset = _make_dataset([0.0] * 40)
    sf0 = dataset.frames[0]
    points = sf0.lidar_frame.load()

    m0 = run_sanity_gate(points, T_CL=dataset.extrinsic.T_CL, camera=dataset.camera,
                          lidar_spec=_make_lidar_spec())
    assert m0.passed

    m2 = evaluate_edge_alignment(
        image=sf0.camera_frame.load(), points_lidar=points, T_CL=dataset.extrinsic.T_CL,
        camera=dataset.camera, lidar_spec=_make_lidar_spec(), depth_jump_threshold_m=1.0,
    )
    m3 = evaluate_holdout_consistency(dataset, lidar_spec=_make_lidar_spec(), n_blocks=4,
                                       min_frames_per_block=5,
                                       edge_alignment_kwargs={"depth_jump_threshold_m": 1.0})
    m4 = evaluate_multiframe_consistency(dataset, lidar_spec=_make_lidar_spec(), min_frames=30,
                                          edge_alignment_kwargs={"depth_jump_threshold_m": 1.0})
    quality = compute_quality_score(m2, m3, m4)

    report = build_report(dataset, m2, m3, m4, quality, m0_result=m0.to_dict(), n_blocks=4, min_frames_m4=30)

    assert report["m0_sanity_gate"] is not None
    assert report["m0_sanity_gate"]["passed"] is True
    assert len(report["m0_sanity_gate"]["checks"]) == 4

    # must still serialize cleanly and render cleanly with M0 present
    s = to_json_string(report)
    parsed = _json.loads(s)
    assert parsed["m0_sanity_gate"]["passed"] is True

    html = render_html_report(report)
    assert "<!DOCTYPE html>" in html


def test_m0_failure_is_visible_in_report_dict():
    dataset = _make_dataset([0.0] * 40)
    sf0 = dataset.frames[0]
    points = sf0.lidar_frame.load()

    T_way_off = np.eye(4)
    T_way_off[0, 3] = 500.0
    m0_fail = run_sanity_gate(points, T_CL=T_way_off, camera=dataset.camera, lidar_spec=_make_lidar_spec())
    assert not m0_fail.passed

    m2 = evaluate_edge_alignment(
        image=sf0.camera_frame.load(), points_lidar=points, T_CL=dataset.extrinsic.T_CL,
        camera=dataset.camera, lidar_spec=_make_lidar_spec(), depth_jump_threshold_m=1.0,
    )
    m3 = evaluate_holdout_consistency(dataset, lidar_spec=_make_lidar_spec(), n_blocks=4,
                                       min_frames_per_block=5,
                                       edge_alignment_kwargs={"depth_jump_threshold_m": 1.0})
    m4 = evaluate_multiframe_consistency(dataset, lidar_spec=_make_lidar_spec(), min_frames=30,
                                          edge_alignment_kwargs={"depth_jump_threshold_m": 1.0})
    quality = compute_quality_score(m2, m3, m4)

    report = build_report(dataset, m2, m3, m4, quality, m0_result=m0_fail.to_dict())
    assert report["m0_sanity_gate"]["passed"] is False
    failed_checks = [c for c in report["m0_sanity_gate"]["checks"] if not c["passed"]]
    assert len(failed_checks) > 0

    s = to_json_string(report)  # must not raise even with M0 failures/warnings present
    _json.loads(s)


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
