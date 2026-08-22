import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from report.diff import compute_report_diff, render_diff_console


def _report(overall_score, overall_cls, geo=None, gen=None, stab=None):
    categories = []
    if geo is not None:
        categories.append({"name": "geometry", "score": geo[0], "classification": geo[1]})
    if gen is not None:
        categories.append({"name": "generalization", "score": gen[0], "classification": gen[1]})
    if stab is not None:
        categories.append({"name": "stability", "score": stab[0], "classification": stab[1]})
    return {"quality_score": {
        "overall_score": overall_score, "overall_classification": overall_cls,
        "categories": categories,
    }}


def test_compute_report_diff_no_change():
    r = _report(95.6, "GOOD", geo=(86.7, "GOOD"), gen=(100.0, "GOOD"), stab=(100.0, "GOOD"))
    diff = compute_report_diff(r, r)
    assert diff["overall"]["delta_score"] == 0.0
    assert diff["overall"]["regressed"] is False
    assert diff["any_regressed"] is False


def test_compute_report_diff_detects_improvement():
    old = _report(80.0, "WARNING", geo=(70.0, "WARNING"), gen=(100.0, "GOOD"), stab=(90.0, "GOOD"))
    new = _report(95.6, "GOOD", geo=(86.7, "GOOD"), gen=(100.0, "GOOD"), stab=(100.0, "GOOD"))
    diff = compute_report_diff(old, new)
    assert diff["overall"]["delta_score"] > 0
    assert diff["overall"]["regressed"] is False
    assert diff["any_regressed"] is False


def test_compute_report_diff_detects_classification_regression():
    old = _report(95.6, "GOOD", geo=(86.7, "GOOD"), gen=(100.0, "GOOD"), stab=(100.0, "GOOD"))
    new = _report(60.0, "WARNING", geo=(40.0, "BAD"), gen=(100.0, "GOOD"), stab=(100.0, "GOOD"))
    diff = compute_report_diff(old, new)
    assert diff["overall"]["regressed"] is True
    assert diff["categories"]["geometry"]["regressed"] is True
    assert diff["categories"]["generalization"]["regressed"] is False
    assert diff["any_regressed"] is True


def test_compute_report_diff_detects_same_classification_score_drop():
    # Classification unchanged (GOOD -> GOOD) but score dropped -- should
    # still count as a regression, not be masked by the unchanged label.
    old = _report(99.0, "GOOD", geo=(99.0, "GOOD"), gen=(100.0, "GOOD"), stab=(100.0, "GOOD"))
    new = _report(90.0, "GOOD", geo=(90.0, "GOOD"), gen=(100.0, "GOOD"), stab=(100.0, "GOOD"))
    diff = compute_report_diff(old, new)
    assert diff["overall"]["old_classification"] == diff["overall"]["new_classification"] == "GOOD"
    assert diff["overall"]["regressed"] is True
    assert diff["any_regressed"] is True


def test_compute_report_diff_handles_missing_category():
    # A category that FAILed outright is omitted from "categories" in the
    # real report shape -- diff should treat it as worst-case, not crash.
    old = _report(95.6, "GOOD", geo=(86.7, "GOOD"), gen=(100.0, "GOOD"), stab=(100.0, "GOOD"))
    new = _report(50.0, "WARNING", gen=(100.0, "GOOD"), stab=(100.0, "GOOD"))  # geometry missing = FAILed
    diff = compute_report_diff(old, new)
    assert diff["categories"]["geometry"]["new_score"] is None
    assert diff["categories"]["geometry"]["new_classification"] is None
    assert diff["categories"]["geometry"]["regressed"] is True
    assert diff["any_regressed"] is True


def test_render_diff_console_marks_regressions():
    old = _report(95.6, "GOOD", geo=(86.7, "GOOD"), gen=(100.0, "GOOD"), stab=(100.0, "GOOD"))
    new = _report(60.0, "WARNING", geo=(40.0, "BAD"), gen=(100.0, "GOOD"), stab=(100.0, "GOOD"))
    diff = compute_report_diff(old, new)
    text = render_diff_console(diff)
    assert "REGRESSED" in text
    assert "GOOD -> WARNING" in text
    assert "GOOD -> BAD" in text


def test_render_diff_console_no_regression_marker_when_improved():
    old = _report(80.0, "WARNING", geo=(70.0, "WARNING"), gen=(100.0, "GOOD"), stab=(90.0, "GOOD"))
    new = _report(95.6, "GOOD", geo=(86.7, "GOOD"), gen=(100.0, "GOOD"), stab=(100.0, "GOOD"))
    diff = compute_report_diff(old, new)
    text = render_diff_console(diff)
    assert "REGRESSED" not in text


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
