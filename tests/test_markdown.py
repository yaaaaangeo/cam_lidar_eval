import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from report.markdown import render_github_comment
from report.diff import compute_report_diff


def _report(overall_score, overall_cls, num_valid=3, warnings=None,
            geo=(86.7, "GOOD"), gen=(100.0, "GOOD"), stab=(100.0, "GOOD"), m0_passed=True):
    return {
        "metadata": {"tool_version": "0.2.0", "generated_at": "2026-08-16T00:00:00+00:00"},
        "m0_sanity_gate": {"passed": m0_passed},
        "quality_score": {
            "overall_score": overall_score, "overall_classification": overall_cls,
            "num_valid_categories": num_valid,
            "categories": [
                {"name": "geometry", "score": geo[0], "classification": geo[1]},
                {"name": "generalization", "score": gen[0], "classification": gen[1]},
                {"name": "stability", "score": stab[0], "classification": stab[1]},
            ],
        },
        "warnings": warnings or [],
    }


def test_render_github_comment_is_valid_utf8_for_every_classification():
    # Regression test: the FAIL emoji was originally a lone UTF-16
    # surrogate pair ("\ud83d\udeab"), which Python does NOT combine into
    # a single astral character -- encoding it raised UnicodeEncodeError.
    # Every classification's rendering must survive an actual UTF-8 encode.
    for cls in ("GOOD", "WARNING", "BAD", "FAIL"):
        report = _report(50.0, cls, geo=(50.0, cls))
        md = render_github_comment(report)
        md.encode("utf-8")  # raises UnicodeEncodeError if broken


def test_render_github_comment_includes_overall_score_and_classification():
    report = _report(95.6, "GOOD")
    md = render_github_comment(report)
    assert "GOOD" in md
    assert "95.6" in md


def test_render_github_comment_includes_all_three_categories():
    report = _report(95.6, "GOOD")
    md = render_github_comment(report)
    assert "Geometry (M2)" in md
    assert "Generalization (M3)" in md
    assert "Stability (M4)" in md


def test_render_github_comment_shows_m0_status():
    passed = render_github_comment(_report(95.6, "GOOD", m0_passed=True))
    failed = render_github_comment(_report(95.6, "GOOD", m0_passed=False))
    assert "PASS" in passed
    assert "FAIL" in failed


def test_render_github_comment_includes_partial_note_when_categories_missing():
    report = _report(50.0, "WARNING", num_valid=2)
    md = render_github_comment(report)
    assert "2/3" in md


def test_render_github_comment_includes_warnings_when_present():
    report = _report(95.6, "GOOD", warnings=["something worth flagging"])
    md = render_github_comment(report)
    assert "something worth flagging" in md
    assert "1 warning" in md


def test_render_github_comment_omits_warnings_section_when_none():
    report = _report(95.6, "GOOD", warnings=[])
    md = render_github_comment(report)
    assert "warning(s)" not in md


def test_render_github_comment_with_diff_shows_delta_column_and_arrow():
    old = _report(70.0, "WARNING", geo=(60.0, "WARNING"))
    new = _report(95.6, "GOOD", geo=(86.7, "GOOD"))
    diff = compute_report_diff(old, new)
    md = render_github_comment(new, diff=diff)
    assert "vs previous" in md
    assert "WARNING" in md and "GOOD" in md


def test_render_github_comment_with_diff_marks_regression():
    old = _report(95.6, "GOOD")
    new = _report(40.0, "BAD", geo=(20.0, "FAIL"))
    diff = compute_report_diff(old, new)
    md = render_github_comment(new, diff=diff)
    assert "regressed" in md
    md.encode("utf-8")


def test_render_github_comment_without_diff_omits_delta_column():
    report = _report(95.6, "GOOD")
    md = render_github_comment(report, diff=None)
    assert "vs previous" not in md


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
