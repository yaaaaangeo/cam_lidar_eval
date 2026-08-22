"""
quality/confidence_coverage.py

STEP 13 -- Quality / Confidence / Coverage separation (see
evaluation_metric_spec.md's STEP13, "Quality와 Confidence를 분리한다").

quality.quality_score's Overall Quality answers "how good does the
calibration LOOK, given what we measured" -- but that single number can't
distinguish two very different situations that look identical on paper:

    Quality: 82/100     Quality: 82/100
    Confidence: 94/100  Confidence: 42/100
    Coverage: 97/100    Coverage: 51/100

The LEFT case is a trustworthy 82: measured thoroughly, across a wide
range of conditions, with plenty of matched points and stable blocks/
frames. The RIGHT case is the SAME 82, but from a thin, narrow, partially-
failed measurement -- the calibration might genuinely be fine, or it might
not be; this number simply doesn't know yet. Reporting only "82" collapses
these into the same statement, which is exactly what this module fixes by
reporting all three:

  - Quality (unchanged): quality.quality_score.QualityScoreResult.overall_score,
    passed through as-is. STEP13 doesn't change how Quality is computed --
    it adds two NEW, independent numbers alongside it.

  - Confidence: how much this run's numbers can be TRUSTED, i.e. how
    reliable the MEASUREMENT PROCESS was, regardless of what it measured.
    Built from signals already computed elsewhere in this pipeline that
    are specifically about measurement reliability:
      - STEP2 sync quality (a badly-synced dataset undermines everything
        downstream, however good M2/M3/M4 individually look)
      - STEP6 match rate (how much of M2's point set found a genuine
        correspondence vs was penalized/excluded)
      - STEP10 M4 valid_ratio (how much of the frame sequence M2 could
        even evaluate)
      - M3 valid block ratio (how many of the configured time blocks
        actually had enough data)
      - STEP1 input validation status (WARNING here means some part of
        the raw input itself was already suspect)

  - Coverage: how much of the sensor's OPERATING ENVELOPE was actually
    exercised by this dataset -- a measurement can be highly RELIABLE
    (high Confidence) while still only having tested a narrow slice of
    real-world conditions (low Coverage). Built from:
      - STEP9 depth-bin coverage (how many of the 5 depth bins had data)
      - STEP9 spatial-region coverage (how many of the LEFT/CENTER/RIGHT
        + TOP/CENTER/BOTTOM regions had data)
      - STEP10 M3 FOV coverage (how much of the image area itself was
        actually spanned by edge points, averaged across blocks)

Both Confidence and Coverage are 0-100 scores using the SAME GOOD/WARNING/
BAD boundaries (80/50) quality.normalization.score_to_classification
already uses for Quality, so all three numbers read on the same scale and
share one classification vocabulary.

Every component is independently optional: this module accepts whatever
subset of STEP1-11's results the caller has (mirroring
evaluation.root_cause's same design) and averages whichever components
are actually available, rather than requiring the full set.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from quality.normalization import score_to_classification


_SYNC_CLASSIFICATION_SCORE = {"GOOD": 1.0, "WARNING": 0.6, "BAD": 0.2, "FAIL": 0.0}
_INPUT_VALIDATION_SCORE = {"INPUT_VALID": 1.0, "INPUT_WARNING": 0.5, "INPUT_INVALID": 0.0}


@dataclass
class ComponentScore:
    name: str
    value: float  # 0-1
    detail: str


@dataclass
class ConfidenceCoverageAxis:
    """One axis (Confidence or Coverage): a 0-100 score, its GOOD/WARNING/
    BAD classification (same 80/50 boundaries as Quality), and the
    individual components that were averaged to produce it -- so a
    reader can see WHY confidence or coverage is low, not just that it is."""
    score: float
    classification: str
    components: list[ComponentScore] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "score": self.score if self.score == self.score else None,  # NaN -> None
            "classification": self.classification,
            "components": [{"name": c.name, "value": c.value, "detail": c.detail} for c in self.components],
        }


def _combine(components: list[ComponentScore]) -> ConfidenceCoverageAxis:
    if not components:
        return ConfidenceCoverageAxis(score=float("nan"), classification="FAIL", components=[])
    score = 100.0 * sum(c.value for c in components) / len(components)
    return ConfidenceCoverageAxis(score=score, classification=score_to_classification(score), components=components)


def compute_confidence(
    sync_stats=None,
    m2=None,
    m3=None,
    m4=None,
    input_validation: Optional[dict] = None,
    n_blocks: Optional[int] = None,
) -> ConfidenceCoverageAxis:
    """
    STEP13 Confidence axis: how much this run's measurement PROCESS can
    be trusted -- see this module's docstring for exactly which signal
    feeds which component. Every argument is Optional; components whose
    required input is missing are simply skipped (not penalized), so
    Confidence never drops just because the caller didn't run an opt-in
    diagnostic -- it reflects only what was actually checked.
    """
    components: list[ComponentScore] = []

    if sync_stats is not None:
        value = _SYNC_CLASSIFICATION_SCORE.get(sync_stats.classification, 0.0)
        components.append(ComponentScore(
            "sync_quality", value,
            f"Timestamp sync is {sync_stats.classification} "
            f"(offset {sync_stats.estimated_offset_ms:+.1f}ms, drop ratio {sync_stats.drop_ratio:.1%}).",
        ))

    if input_validation is not None:
        status = input_validation.get("status")
        value = _INPUT_VALIDATION_SCORE.get(status, 0.5)
        components.append(ComponentScore("input_validation", value, f"Input validation status is {status}."))

    if m2 is not None and getattr(m2, "match_rate", None) is not None:
        components.append(ComponentScore(
            "m2_match_rate", m2.match_rate,
            f"{m2.match_rate:.0%} of M2 edge points found a genuine STEP6 correspondence.",
        ))

    if m3 is not None and n_blocks is not None and n_blocks > 0:
        ratio = min(1.0, m3.num_valid_blocks / n_blocks)
        components.append(ComponentScore(
            "m3_valid_blocks", ratio,
            f"{m3.num_valid_blocks}/{n_blocks} configured time blocks had enough data to evaluate.",
        ))

    if m4 is not None and getattr(m4, "valid_ratio", None) is not None and m4.valid_ratio == m4.valid_ratio:  # not NaN
        components.append(ComponentScore(
            "m4_valid_ratio", m4.valid_ratio,
            f"{m4.valid_ratio:.0%} of frames produced a usable M2 result (rest FAILed).",
        ))

    return _combine(components)


def compute_coverage(
    spatial_analysis=None,
    m3=None,
) -> ConfidenceCoverageAxis:
    """
    STEP13 Coverage axis: how much of the sensor's operating envelope
    (depth range, field of view) was actually exercised by this dataset
    -- see this module's docstring for exactly which signal feeds which
    component. Every argument is Optional; components whose required
    input is missing are simply skipped.
    """
    components: list[ComponentScore] = []

    if spatial_analysis is not None:
        depth_bins = getattr(spatial_analysis, "depth_bins", None) or {}
        if depth_bins:
            populated = sum(1 for b in depth_bins.values() if (b.valid_count + b.failure_count) > 0)
            ratio = populated / len(depth_bins)
            components.append(ComponentScore(
                "depth_bin_coverage", ratio,
                f"{populated}/{len(depth_bins)} depth bins (0-10m..50m+) had data.",
            ))

        h_regions = getattr(spatial_analysis, "horizontal_regions", None) or {}
        v_regions = getattr(spatial_analysis, "vertical_regions", None) or {}
        all_regions = {**h_regions, **v_regions}
        if all_regions:
            populated = sum(1 for b in all_regions.values() if (b.valid_count + b.failure_count) > 0)
            ratio = populated / len(all_regions)
            components.append(ComponentScore(
                "spatial_region_coverage", ratio,
                f"{populated}/{len(all_regions)} camera regions (LEFT/CENTER/RIGHT, TOP/CENTER/BOTTOM) had data.",
            ))

    if m3 is not None:
        fov_values = [b.fov_coverage for b in m3.block_results
                      if b.classification in ("GOOD", "WARNING", "BAD") and b.fov_coverage == b.fov_coverage]
        if fov_values:
            mean_fov = sum(fov_values) / len(fov_values)
            components.append(ComponentScore(
                "fov_coverage", min(1.0, mean_fov),
                f"Edge points spanned {mean_fov:.0%} of the image area on average across valid blocks.",
            ))

    return _combine(components)


@dataclass
class QualityConfidenceCoverageResult:
    quality_score: float
    quality_classification: str
    confidence: ConfidenceCoverageAxis
    coverage: ConfidenceCoverageAxis

    def to_dict(self) -> dict:
        return {
            "quality": {
                "score": self.quality_score if self.quality_score == self.quality_score else None,
                "classification": self.quality_classification,
            },
            "confidence": self.confidence.to_dict(),
            "coverage": self.coverage.to_dict(),
        }

    def summary_line(self) -> str:
        def _fmt(score):
            return f"{score:.0f}/100" if score == score else "N/A"
        return (f"Quality: {_fmt(self.quality_score)} ({self.quality_classification})  "
                f"Confidence: {_fmt(self.confidence.score)} ({self.confidence.classification})  "
                f"Coverage: {_fmt(self.coverage.score)} ({self.coverage.classification})")


def compute_quality_confidence_coverage(
    quality_result,
    sync_stats=None,
    m2=None,
    m3=None,
    m4=None,
    spatial_analysis=None,
    input_validation: Optional[dict] = None,
    n_blocks: Optional[int] = None,
) -> QualityConfidenceCoverageResult:
    """
    STEP13 top-level entry point: Quality passed through unchanged from
    quality.quality_score.QualityScoreResult, plus the new Confidence and
    Coverage axes computed from whatever subset of STEP1-11's results the
    caller provides.
    """
    confidence = compute_confidence(sync_stats=sync_stats, m2=m2, m3=m3, m4=m4,
                                     input_validation=input_validation, n_blocks=n_blocks)
    coverage = compute_coverage(spatial_analysis=spatial_analysis, m3=m3)
    return QualityConfidenceCoverageResult(
        quality_score=quality_result.overall_score,
        quality_classification=quality_result.overall_classification,
        confidence=confidence,
        coverage=coverage,
    )
