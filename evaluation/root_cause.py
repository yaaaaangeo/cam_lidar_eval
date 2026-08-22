"""
evaluation/root_cause.py

STEP 12 -- Root Cause Diagnosis Engine (see evaluation_metric_spec.md's
STEP12, "⭐ 프로젝트의 최종 핵심 -- 이제야 자동 진단을 만듭니다").

This is the FIRST module in the whole pipeline that CROSS-REFERENCES every
other diagnostic this tool has built (STEP1-11) instead of reporting each
one in isolation. Every input listed in the spec is already computed
somewhere else in this codebase -- this module's only job is to combine
them with simple, explainable IF/THEN rules (per the spec's own explicit
"처음부터 AI를 사용하지 않는 것을 추천합니다" -- rule-based first, not ML)
into a ranked list of plausible root causes:

    input                          -> where it already comes from
    -----                             ---------------------------
    Timestamp metrics               input/dataset.py's SyncStats (STEP2)
    M2                               evaluation/edge_alignment.py
    M3 (+ scene metadata)           evaluation/holdout_consistency.py (STEP10)
    M4 (+ robust stats)             evaluation/multiframe_consistency.py (STEP10)
    Depth / Spatial analysis        evaluation/spatial_analysis.py (STEP9)
    Dynamic ratio                   evaluation/dynamic_filter.py (STEP8)
    Uncertainty (normalized_error)  evaluation/edge_alignment.py (STEP7)
    Sensitivity                     evaluation/perturbation.py (STEP11)
    Coverage                        evaluation/holdout_consistency.py's
                                     BlockResult.fov_coverage (STEP10)

Every rule below is a direct, named implementation of one of the spec's
own worked examples, or a natural extension using the same style of
evidence this tool already has on hand:

  - TEMPORAL_OFFSET: spec's own example ("timestamp_offset > threshold AND
    M2 improves after synchronization") is adapted to the data actually
    available -- STEP2's sync classification ALREADY folds together
    offset magnitude, residual jitter, and drop ratio into one GOOD/
    WARNING/BAD/FAIL judgement, so this rule reuses that directly instead
    of re-deriving its own threshold.
  - YAW_MISALIGNMENT / PITCH_MISALIGNMENT: spec's own example ("right-side
    error >> left-side error AND yaw sensitivity = HIGH") generalized to
    both spatial axes STEP9 already reports (LEFT/RIGHT for yaw,
    TOP/BOTTOM for pitch), cross-referenced against STEP11's per-axis
    sensitivity ranking.
  - TX_MISALIGNMENT: a translation offset along the baseline direction
    produces a parallax-style error that shrinks with distance (unlike a
    rotation offset, which is roughly depth-independent) -- so STEP9's
    depth_trend == "decreases_with_depth" (worse up close, better far) is
    used as tx's spatial signature, the same way LEFT/RIGHT asymmetry is
    yaw's.
  - DYNAMIC_CONTAMINATION: spec's own example, implemented directly
    against STEP8's DynamicFilteringComparison.
  - SCENE_DEPENDENT_INSTABILITY: reuses STEP10's own
    diagnose_instability(...) output (e.g. "Long-range scenes") as a
    root-cause candidate in its own right, rather than only surfacing it
    inside M3's section.
  - UNEXPLAINED_SENSITIVITY: a catch-all LOW-confidence candidate for any
    axis STEP11 flags HIGH/MEDIUM that none of the specific rules above
    already claimed (e.g. roll, ty, tz, or timestamp) -- so a real signal
    is never silently dropped just because this module doesn't have a
    crisp spatial signature for that particular axis yet.

Every input is Optional: this module accepts whatever subset of
STEP1-11's results the caller actually has (many are opt-in --
dynamic_filter_comparison and perturbation_result in particular need
extra data this tool can't always get on its own) and simply skips any
rule whose required inputs are missing, rather than requiring the full
set to run at all.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


_CONFIDENCE_SCORE = {"HIGH": 3, "MEDIUM": 2, "LOW": 1}

# How much worse the max side must be than BOTH the min side and the
# center region before spatial asymmetry counts as a real signal (not
# just uniformly-elevated error) -- e.g. 2.0 means "at least 2x".
_ASYMMETRY_RATIO_THRESHOLD = 2.0
_DYNAMIC_CONTAMINATION_HIGH = 0.30   # matches the spec's own ">30%" example exactly
_DYNAMIC_CONTAMINATION_MEDIUM = 0.15
_DYNAMIC_CONTAMINATION_LOW = 0.05


@dataclass
class RootCauseCandidate:
    cause: str              # short machine-readable code, e.g. "YAW_MISALIGNMENT"
    label: str              # human-readable, e.g. "Yaw misalignment"
    confidence: str         # "HIGH" | "MEDIUM" | "LOW"
    evidence: list[str] = field(default_factory=list)
    _tiebreak: float = 0.0  # internal: magnitude of the underlying signal, for ranking within a confidence tier

    def to_dict(self) -> dict:
        return {"cause": self.cause, "label": self.label, "confidence": self.confidence, "evidence": list(self.evidence)}


@dataclass
class RootCauseConfirmation:
    """A specific thing that WAS checked and came back clean -- the
    spec's own example output mixes problems (🔴🟠) with confirmations
    (🟢 'Timestamp OK', 🟢 'Sensor quality OK') in the SAME list, not just
    a bare list of problems; this is what supplies the 🟢 half."""
    label: str
    detail: str

    def to_dict(self) -> dict:
        return {"label": self.label, "detail": self.detail}


@dataclass
class RootCauseDiagnosisResult:
    candidates: list[RootCauseCandidate]  # ranked, highest confidence (then magnitude) first
    confirmations: list[RootCauseConfirmation] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "candidates": [c.to_dict() for c in self.candidates],
            "confirmations": [c.to_dict() for c in self.confirmations],
        }

    def summary_lines(self) -> list[str]:
        """Plain-text ranking, matching the spec's own output format:
        '1. Yaw misalignment       HIGH'"""
        return [f"{i}. {c.label:<24} {c.confidence}" for i, c in enumerate(self.candidates, start=1)]


def _axis_sensitivity_lookup(perturbation_result) -> dict:
    if perturbation_result is None:
        return {}
    return {a.axis: a.classification for a in perturbation_result.axis_sensitivities}


def _check_temporal_offset(sync_stats) -> Optional[RootCauseCandidate]:
    """Spec's own worked example, adapted to reuse STEP2's own sync
    classification (which already combines offset magnitude, residual
    jitter, and drop ratio) rather than re-deriving separate thresholds."""
    if sync_stats is None or sync_stats.classification not in ("WARNING", "BAD"):
        return None
    confidence = "HIGH" if sync_stats.classification == "BAD" else "MEDIUM"
    evidence = [
        f"Synchronization is {sync_stats.classification} "
        f"(estimated offset {sync_stats.estimated_offset_ms:+.1f}ms, "
        f"std {sync_stats.offset_std_ms:.1f}ms, drop ratio {sync_stats.drop_ratio:.1%}).",
    ]
    return RootCauseCandidate(
        cause="TEMPORAL_OFFSET", label="Timestamp offset", confidence=confidence,
        evidence=evidence, _tiebreak=abs(sync_stats.estimated_offset_ms),
    )


def _region_asymmetry(regions: dict, low_key: str, high_key: str, center_key: str = "CENTER"):
    """Returns (worse_key, worse_value, other_value, center_value) if the
    worse side is at least _ASYMMETRY_RATIO_THRESHOLD times BOTH the
    other side and the center region (a genuine one-sided pattern, not
    just uniformly elevated error) -- else None."""
    low = regions.get(low_key)
    high = regions.get(high_key)
    center = regions.get(center_key)
    if low is None or high is None or center is None:
        return None
    low_v, high_v, center_v = low.mean_px, high.mean_px, center.mean_px
    if not all(v == v and v is not None for v in (low_v, high_v, center_v)):  # NaN check
        return None
    if low_v <= 0 or high_v <= 0 or center_v <= 0:
        return None

    worse_key, worse_v, other_v = (high_key, high_v, low_v) if high_v >= low_v else (low_key, low_v, high_v)
    if worse_v >= _ASYMMETRY_RATIO_THRESHOLD * max(other_v, 1e-9) and worse_v >= _ASYMMETRY_RATIO_THRESHOLD * max(center_v, 1e-9):
        return worse_key, worse_v, other_v, center_v
    return None


def _check_rotation_misalignment(
    axis_label: str, cause_code: str, sensitivity_axis: str,
    spatial_analysis, region_dict_name: str, low_key: str, high_key: str,
    axis_sensitivity: dict,
) -> Optional[RootCauseCandidate]:
    """Shared logic behind YAW_MISALIGNMENT (horizontal regions) and
    PITCH_MISALIGNMENT (vertical regions): spatial asymmetry cross-
    referenced against STEP11's sensitivity ranking for that axis."""
    if spatial_analysis is None:
        return None
    regions = getattr(spatial_analysis, region_dict_name, None)
    if not regions:
        return None
    asymmetry = _region_asymmetry(regions, low_key, high_key)
    if asymmetry is None:
        return None
    worse_key, worse_v, other_v, center_v = asymmetry
    ratio = worse_v / max(other_v, 1e-9)

    sensitivity = axis_sensitivity.get(sensitivity_axis)
    if sensitivity == "HIGH":
        confidence = "HIGH"
    elif sensitivity == "MEDIUM":
        confidence = "MEDIUM"
    else:
        confidence = "LOW"  # spatial asymmetry alone, unconfirmed by (or missing) sensitivity data

    evidence = [
        f"{worse_key} region error ({worse_v:.2f}px) is {ratio:.1f}x the opposite side "
        f"({other_v:.2f}px) and {worse_v / max(center_v, 1e-9):.1f}x the center region ({center_v:.2f}px).",
    ]
    if sensitivity is not None:
        evidence.append(f"{axis_label} perturbation sensitivity is {sensitivity}.")
    else:
        evidence.append(f"{axis_label} perturbation sensitivity not available -- run with --advanced to confirm.")

    return RootCauseCandidate(cause=cause_code, label=f"{axis_label} misalignment", confidence=confidence,
                               evidence=evidence, _tiebreak=ratio)


def _check_tx_misalignment(spatial_analysis, axis_sensitivity: dict) -> Optional[RootCauseCandidate]:
    """A translation offset along the baseline (tx) direction produces a
    parallax-style error that SHRINKS with distance -- the opposite
    depth signature a rotation offset (roughly depth-independent)
    produces. STEP9's depth_trend == 'decreases_with_depth' is used as
    tx's spatial fingerprint, the same role LEFT/RIGHT asymmetry plays
    for yaw."""
    if spatial_analysis is None or spatial_analysis.depth_trend != "decreases_with_depth":
        return None

    sensitivity = axis_sensitivity.get("tx")
    if sensitivity == "HIGH":
        confidence = "HIGH"
    elif sensitivity == "MEDIUM":
        confidence = "MEDIUM"
    else:
        confidence = "LOW"

    near = spatial_analysis.depth_bins.get("0-10m")
    far_candidates = [spatial_analysis.depth_bins.get(k) for k in ("30-50m", "50m+", "20-30m")]
    far = next((b for b in far_candidates if b is not None and (b.valid_count + b.failure_count) > 0), None)
    evidence = ["Error decreases with depth (worse up close, better far) -- a parallax-style signature "
                "consistent with a translation offset along the camera-LiDAR baseline."]
    if near is not None and far is not None and near.mean_px == near.mean_px and far.mean_px == far.mean_px:
        evidence.append(f"Near-range mean error {near.mean_px:.2f}px vs far-range {far.mean_px:.2f}px.")
    if sensitivity is not None:
        evidence.append(f"Tx perturbation sensitivity is {sensitivity}.")
    else:
        evidence.append("Tx perturbation sensitivity not available -- run with --advanced to confirm.")

    return RootCauseCandidate(cause="TX_MISALIGNMENT", label="Tx misalignment", confidence=confidence,
                               evidence=evidence, _tiebreak=1.0)


def _check_dynamic_contamination(dynamic_filter_comparison) -> Optional[RootCauseCandidate]:
    """Spec's own worked example, implemented directly against STEP8's
    DynamicFilteringComparison."""
    if dynamic_filter_comparison is None:
        return None
    ratio = dynamic_filter_comparison.dynamic_contamination_ratio
    if ratio != ratio or ratio < _DYNAMIC_CONTAMINATION_LOW:  # NaN or below the lowest bar
        return None
    if dynamic_filter_comparison.static_only_classification not in ("GOOD", "WARNING"):
        return None  # static-only ALSO looks bad -- contamination isn't the (whole) story

    if ratio >= _DYNAMIC_CONTAMINATION_HIGH:
        confidence = "HIGH"
    elif ratio >= _DYNAMIC_CONTAMINATION_MEDIUM:
        confidence = "MEDIUM"
    else:
        confidence = "LOW"

    evidence = [
        f"{ratio:.0%} of edge points were classified as dynamic and excluded.",
        f"Overall M2 is {dynamic_filter_comparison.overall_classification} "
        f"({dynamic_filter_comparison.overall_mean_px:.2f}px) vs static-only "
        f"{dynamic_filter_comparison.static_only_classification} "
        f"({dynamic_filter_comparison.static_only_mean_px:.2f}px).",
    ]
    return RootCauseCandidate(cause="DYNAMIC_CONTAMINATION", label="Dynamic object contamination",
                               confidence=confidence, evidence=evidence, _tiebreak=ratio)


def _check_scene_dependent_instability(m3) -> Optional[RootCauseCandidate]:
    """Reuses STEP10's own diagnose_instability(...) output as a root-
    cause candidate in its own right, instead of only surfacing it inside
    M3's own section."""
    if m3 is None or m3.classification not in ("WARNING", "BAD"):
        return None
    diagnosis = getattr(m3, "instability_diagnosis", None)
    if not diagnosis or not diagnosis.get("candidates"):
        return None
    top = diagnosis["candidates"][0]
    relative_diff = abs(top["relative_diff"])
    confidence = "HIGH" if relative_diff >= 1.0 else "MEDIUM"
    evidence = [
        f"Block {diagnosis['worst_block_index']} has the highest error "
        f"({diagnosis['worst_block_mean_px']:.2f}px) and its {top['metric']} differs from "
        f"other blocks by {top['relative_diff']:+.0%}.",
    ]
    return RootCauseCandidate(
        cause="SCENE_DEPENDENT_INSTABILITY", label=top["explanation"], confidence=confidence,
        evidence=evidence, _tiebreak=relative_diff,
    )


def _check_unexplained_sensitivity(axis_sensitivity: dict, claimed_axes: set) -> list[RootCauseCandidate]:
    """Catch-all: any axis STEP11 flags HIGH/MEDIUM that none of the
    specific rules above already claimed (e.g. roll, ty, tz, timestamp)
    -- reported as LOW confidence since there's no corroborating spatial
    signature, but never silently dropped."""
    labels = {"roll_deg": "Roll", "pitch_deg": "Pitch", "yaw_deg": "Yaw",
              "tx": "Tx", "ty": "Ty", "tz": "Tz", "timestamp": "Timestamp sync"}
    out = []
    for axis, classification in axis_sensitivity.items():
        if axis in claimed_axes or classification not in ("HIGH", "MEDIUM"):
            continue
        label = labels.get(axis, axis)
        out.append(RootCauseCandidate(
            cause=f"UNEXPLAINED_SENSITIVITY_{axis.upper()}",
            label=f"{label} sensitivity ({classification.lower()}, unconfirmed)",
            confidence="LOW",
            evidence=[f"{label} perturbation sensitivity is {classification}, but no corroborating "
                      f"spatial/depth pattern was found to confirm this as the cause."],
            _tiebreak=0.0,
        ))
    return out


def _check_confirmations(
    sync_stats, m3, dynamic_filter_comparison, axis_sensitivity: dict,
) -> list[RootCauseConfirmation]:
    """
    STEP14: the 🟢 half of the spec's own diagnosis panel example --
    things that were actually CHECKED and came back clean, not just
    silence. Each confirmation only fires when its corresponding check
    ran AND found nothing (mirrors each candidate rule's own "only
    problems clear this bar" logic, inverted)."""
    confirmations = []

    if sync_stats is not None and sync_stats.classification == "GOOD":
        confirmations.append(RootCauseConfirmation(
            "Timestamp sync",
            f"No significant clock offset detected (offset {sync_stats.estimated_offset_ms:+.1f}ms, GOOD).",
        ))

    if dynamic_filter_comparison is not None:
        ratio = dynamic_filter_comparison.dynamic_contamination_ratio
        if ratio == ratio and ratio < _DYNAMIC_CONTAMINATION_LOW:  # not NaN, below the lowest bar
            confirmations.append(RootCauseConfirmation(
                "Dynamic object contamination",
                f"Only {ratio:.0%} of edge points were classified as dynamic -- negligible.",
            ))

    if m3 is not None and m3.classification == "GOOD":
        confirmations.append(RootCauseConfirmation(
            "Block-to-block consistency (M3)",
            "No scene-dependent instability detected across time blocks (GOOD).",
        ))

    if axis_sensitivity and all(c == "LOW" for c in axis_sensitivity.values()):
        confirmations.append(RootCauseConfirmation(
            "Calibration parameter sensitivity",
            f"All {len(axis_sensitivity)} checked axes show LOW sensitivity to small perturbations.",
        ))

    return confirmations


def diagnose_root_cause(
    sync_stats=None,
    m2=None,
    m3=None,
    m4=None,
    spatial_analysis=None,
    dynamic_filter_comparison=None,
    perturbation_result=None,
) -> RootCauseDiagnosisResult:
    """
    STEP12: cross-reference every diagnostic STEP1-11 already computed
    into a ranked list of plausible root causes. Every argument is
    Optional and independently skippable -- pass whatever subset of
    STEP1-11's results you have; rules whose required inputs are missing
    simply don't fire (see this module's docstring for exactly which
    input feeds which rule).

    Candidates are ranked by confidence (HIGH > MEDIUM > LOW), then by
    the magnitude of each rule's own underlying evidence within a tier.
    """
    axis_sensitivity = _axis_sensitivity_lookup(perturbation_result)
    candidates: list[RootCauseCandidate] = []
    claimed_axes: set = set()

    c = _check_temporal_offset(sync_stats)
    if c is not None:
        candidates.append(c)
        claimed_axes.add("timestamp")

    c = _check_rotation_misalignment(
        "Yaw", "YAW_MISALIGNMENT", "yaw_deg", spatial_analysis, "horizontal_regions", "LEFT", "RIGHT",
        axis_sensitivity,
    )
    if c is not None:
        candidates.append(c)
        claimed_axes.add("yaw_deg")

    c = _check_rotation_misalignment(
        "Pitch", "PITCH_MISALIGNMENT", "pitch_deg", spatial_analysis, "vertical_regions", "TOP", "BOTTOM",
        axis_sensitivity,
    )
    if c is not None:
        candidates.append(c)
        claimed_axes.add("pitch_deg")

    c = _check_tx_misalignment(spatial_analysis, axis_sensitivity)
    if c is not None:
        candidates.append(c)
        claimed_axes.add("tx")

    c = _check_dynamic_contamination(dynamic_filter_comparison)
    if c is not None:
        candidates.append(c)

    c = _check_scene_dependent_instability(m3)
    if c is not None:
        candidates.append(c)

    candidates.extend(_check_unexplained_sensitivity(axis_sensitivity, claimed_axes))

    candidates.sort(key=lambda c: (_CONFIDENCE_SCORE[c.confidence], c._tiebreak), reverse=True)
    confirmations = _check_confirmations(sync_stats, m3, dynamic_filter_comparison, axis_sensitivity)
    return RootCauseDiagnosisResult(candidates=candidates, confirmations=confirmations)
