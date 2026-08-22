"""
report/html.py

Renders a report dict (from report/builder.py) as a single self-contained
static HTML document -- an "instrument panel" for an existing calibration:
one glanceable Overall Quality reading up top, then Geometry/Generalization/
Stability category tiles, then per-metric detail tables, then any warnings.

Design intent (see design tokens below): this is a measurement instrument's
readout, not a marketing page. Dark console background, semantic
GOOD/WARNING/BAD/FAIL color coding used consistently everywhere (badges,
the overall-score gauge ring, category tiles), and monospace type for every
numeric readout to reinforce that these are precise sensor measurements.
No JavaScript is required for the core report -- the overall-score ring
is pure CSS (conic-gradient), so the file works as a plain
double-clickable static document with no server and no network dependency
beyond optional web fonts (which degrade gracefully to system fonts if
unavailable). The one opt-in exception is the interactive 3D viewer: when
an interactive scene is supplied, this module inlines the vendored
plotly.js gl3d bundle (report/vendor/, no CDN) plus the scene's JSON data
directly into the document, so the file still works fully offline -- just
no longer JS-free in that case.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional
from html import escape
from functools import lru_cache
import base64
import json
import os


# ---------------------------------------------------------------------------
# Design tokens
# ---------------------------------------------------------------------------

_STATUS_COLORS = {
    "GOOD": "#3FB950",
    "WARNING": "#D29922",
    "BAD": "#F85149",
    "FAIL": "#6E7681",
    # STEP11 -- Perturbation Sensitivity's HIGH/MEDIUM/LOW classification
    # reuses the same badge rendering as GOOD/WARNING/BAD; HIGH sensitivity
    # is the one worth worrying about, so it maps to the same red as BAD.
    "HIGH": "#F85149",
    "MEDIUM": "#D29922",
    "LOW": "#3FB950",
}

_CSS = """
:root {
  --bg: #0D1117;
  --surface: #161B22;
  --surface-alt: #1C2333;
  --border: #2A3244;
  --text-primary: #E6EDF3;
  --text-secondary: #8B98AC;
  --accent: #7DD3FC;
  --good: #3FB950;
  --warning: #D29922;
  --bad: #F85149;
  --fail: #6E7681;
  --font-display: 'Space Grotesk', 'Segoe UI', system-ui, sans-serif;
  --font-body: 'Inter', 'Segoe UI', system-ui, sans-serif;
  --font-mono: 'JetBrains Mono', 'SFMono-Regular', Consolas, monospace;
}

* { box-sizing: border-box; }

body {
  margin: 0;
  background: var(--bg);
  color: var(--text-primary);
  font-family: var(--font-body);
  line-height: 1.5;
  padding: 0 0 4rem 0;
}

@media (prefers-reduced-motion: no-preference) {
  body { animation: fade-in 0.4s ease-out; }
}
@keyframes fade-in { from { opacity: 0; } to { opacity: 1; } }

.container {
  max-width: 960px;
  margin: 0 auto;
  padding: 2.5rem 1.5rem;
}

header.report-header {
  display: flex;
  justify-content: space-between;
  align-items: baseline;
  flex-wrap: wrap;
  gap: 0.5rem;
  border-bottom: 1px solid var(--border);
  padding-bottom: 1.25rem;
  margin-bottom: 2rem;
}

header.report-header h1 {
  font-family: var(--font-display);
  font-size: clamp(1.4rem, 2.5vw, 1.9rem);
  font-weight: 600;
  letter-spacing: -0.01em;
  margin: 0;
}

header.report-header .meta {
  font-family: var(--font-mono);
  font-size: 0.8rem;
  color: var(--text-secondary);
}

.hero {
  display: flex;
  align-items: center;
  gap: 2.5rem;
  flex-wrap: wrap;
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 16px;
  padding: 2rem;
  margin-bottom: 2rem;
}

.gauge {
  width: 148px;
  height: 148px;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  flex-shrink: 0;
  background: conic-gradient(var(--gauge-color) calc(var(--gauge-pct) * 1%), var(--surface-alt) 0);
}

.gauge-inner {
  width: 116px;
  height: 116px;
  border-radius: 50%;
  background: var(--surface);
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
}

.gauge-inner .score {
  font-family: var(--font-mono);
  font-size: 2rem;
  font-weight: 700;
  line-height: 1;
}

.gauge-inner .score-label {
  font-size: 0.65rem;
  color: var(--text-secondary);
  letter-spacing: 0.08em;
  text-transform: uppercase;
  margin-top: 0.2rem;
}

.hero-text h2 {
  font-family: var(--font-display);
  font-size: 1.1rem;
  margin: 0 0 0.4rem 0;
  color: var(--text-secondary);
  font-weight: 500;
  text-transform: uppercase;
  letter-spacing: 0.06em;
}

.hero-text .verdict {
  font-family: var(--font-display);
  font-size: 1.6rem;
  font-weight: 600;
  margin: 0;
}

.badge {
  display: inline-flex;
  align-items: center;
  gap: 0.4rem;
  padding: 0.2rem 0.65rem;
  border-radius: 999px;
  font-family: var(--font-mono);
  font-size: 0.72rem;
  font-weight: 600;
  letter-spacing: 0.04em;
  text-transform: uppercase;
  border: 1px solid currentColor;
}
.badge::before {
  content: '';
  width: 6px;
  height: 6px;
  border-radius: 50%;
  background: currentColor;
}

.category-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
  gap: 1rem;
  margin-bottom: 2.5rem;
}

.category-card {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 1.25rem;
  border-top: 3px solid var(--card-color, var(--border));
}

.category-card .cat-label {
  font-size: 0.75rem;
  color: var(--text-secondary);
  text-transform: uppercase;
  letter-spacing: 0.06em;
  margin-bottom: 0.3rem;
}

.category-card .cat-metric {
  font-family: var(--font-mono);
  font-size: 0.7rem;
  color: var(--text-secondary);
  margin-bottom: 0.6rem;
}

.category-card .cat-score {
  font-family: var(--font-mono);
  font-size: 1.9rem;
  font-weight: 700;
  margin-bottom: 0.5rem;
}

section.metric-section {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 1.5rem 1.75rem;
  margin-bottom: 1.5rem;
}

section.metric-section h3 {
  font-family: var(--font-display);
  font-size: 1.05rem;
  margin: 0 0 0.2rem 0;
  display: flex;
  align-items: center;
  gap: 0.6rem;
}

section.metric-section .metric-subtitle {
  color: var(--text-secondary);
  font-size: 0.85rem;
  margin: 0 0 1rem 0;
}

.stat-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
  gap: 0.75rem;
  margin-bottom: 1rem;
}

.stat {
  background: var(--surface-alt);
  border-radius: 8px;
  padding: 0.6rem 0.8rem;
}

.stat .stat-label {
  font-size: 0.68rem;
  color: var(--text-secondary);
  text-transform: uppercase;
  letter-spacing: 0.05em;
}

.stat .stat-value {
  font-family: var(--font-mono);
  font-size: 1.1rem;
  font-weight: 600;
  margin-top: 0.15rem;
}

table.data-table {
  width: 100%;
  border-collapse: collapse;
  font-family: var(--font-mono);
  font-size: 0.82rem;
  margin-top: 0.5rem;
}

table.data-table th, table.data-table td {
  text-align: left;
  padding: 0.45rem 0.6rem;
  border-bottom: 1px solid var(--border);
}

table.data-table th {
  color: var(--text-secondary);
  font-weight: 500;
  text-transform: uppercase;
  font-size: 0.68rem;
  letter-spacing: 0.05em;
}

.warnings-section {
  border-radius: 12px;
  overflow: hidden;
}

.warning-item {
  background: var(--surface);
  border-left: 3px solid var(--warning);
  padding: 0.7rem 1rem;
  margin-bottom: 0.5rem;
  font-size: 0.85rem;
  color: var(--text-secondary);
  border-radius: 0 8px 8px 0;
}

.matrix {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 2px;
  font-family: var(--font-mono);
  font-size: 0.75rem;
  max-width: 320px;
  margin-top: 0.4rem;
}
.matrix span {
  background: var(--surface-alt);
  padding: 0.3rem 0.4rem;
  border-radius: 4px;
  text-align: right;
}

footer {
  text-align: center;
  color: var(--text-secondary);
  font-size: 0.75rem;
  font-family: var(--font-mono);
  margin-top: 2.5rem;
}

.view-toggle {
  display: inline-flex;
  gap: 0.25rem;
  background: var(--surface-alt);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 0.2rem;
  margin-top: 0.6rem;
}
.view-toggle-btn {
  font-family: var(--font-mono);
  font-size: 0.75rem;
  color: var(--text-secondary);
  background: transparent;
  border: none;
  border-radius: 6px;
  padding: 0.35rem 0.8rem;
  cursor: pointer;
}
.view-toggle-btn.active {
  background: var(--surface);
  color: var(--text-primary);
}
.view-panel {
  display: none;
}
.view-panel.active {
  display: block;
}
"""

_FONT_LINKS = (
    '<link rel="preconnect" href="https://fonts.googleapis.com">'
    '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
    '<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&'
    'family=Inter:wght@400;500&family=JetBrains+Mono:wght@400;600;700&display=swap" '
    'rel="stylesheet">'
)


def _color_for(classification: str) -> str:
    return _STATUS_COLORS.get(classification, _STATUS_COLORS["FAIL"])


def _img_tag(image_bytes: Optional[bytes], alt: str, mime: str = "image/png") -> str:
    """Embed an image (PNG or GIF) as a base64 data URI so the HTML report
    stays a single self-contained file (no sibling image files to lose
    track of when shared). Returns an empty string if image_bytes is
    None, so callers can unconditionally splice this into a template
    without an if/else."""
    if not image_bytes:
        return ""
    b64 = base64.b64encode(image_bytes).decode("ascii")
    return (
        f'<img src="data:{mime};base64,{b64}" alt="{escape(alt)}" '
        f'style="width:100%; border-radius:10px; margin-top:0.75rem; display:block;">'
    )


def _badge(classification: str) -> str:
    color = _color_for(classification)
    return f'<span class="badge" style="color:{color}">{escape(classification)}</span>'


def _fmt(value, unit: str = "", digits: int = 3) -> str:
    if value is None:
        return "&mdash;"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (int,)):
        return f"{value}{unit}"
    if isinstance(value, float):
        return f"{value:.{digits}f}{unit}"
    return escape(str(value))


def _stat(label: str, value: str) -> str:
    return (
        f'<div class="stat"><div class="stat-label">{escape(label)}</div>'
        f'<div class="stat-value">{value}</div></div>'
    )


def _render_hero(quality: dict) -> str:
    score = quality["overall_score"]
    classification = quality["overall_classification"]
    color = _color_for(classification)
    pct = max(0.0, min(100.0, score)) if score is not None else 0.0
    score_display = f"{score:.1f}" if score is not None else "&mdash;"

    verdict_text = {
        "GOOD": "Calibration looks solid across all measured metrics.",
        "WARNING": "Calibration shows some inconsistency worth reviewing.",
        "BAD": "Calibration does not hold up well under evaluation.",
        "FAIL": "Not enough valid data to assess this calibration.",
    }.get(classification, "")

    return f"""
    <div class="hero">
      <div class="gauge" style="--gauge-pct:{pct}; --gauge-color:{color};">
        <div class="gauge-inner">
          <div class="score" style="color:{color}">{score_display}</div>
          <div class="score-label">/ 100</div>
        </div>
      </div>
      <div class="hero-text">
        <h2>Overall Calibration Quality</h2>
        <p class="verdict">{_badge(classification)}</p>
        <p style="color:var(--text-secondary); margin-top:0.6rem; max-width:32rem;">{escape(verdict_text)}</p>
      </div>
    </div>
    """


def _render_categories(quality: dict) -> str:
    labels = {
        "geometry": ("Geometry", "How precisely LiDAR structure lines up with image edges, right now."),
        "generalization": ("Generalization", "Whether this calibration holds up across different time windows."),
        "stability": ("Stability", "Whether error stays consistent frame-to-frame, or spikes unpredictably."),
    }
    cards = []
    for cat in quality["categories"]:
        name = cat["name"]
        title, subtitle = labels.get(name, (name.title(), ""))
        color = _color_for(cat["classification"])
        score_display = f"{cat['score']:.1f}" if cat["score"] is not None else "&mdash;"
        cards.append(f"""
        <div class="category-card" style="--card-color:{color}">
          <div class="cat-label">{escape(title)}</div>
          <div class="cat-metric">{escape(cat["metric"])} &middot; {escape(subtitle)}</div>
          <div class="cat-score" style="color:{color}">{score_display}</div>
          {_badge(cat["classification"])}
        </div>
        """)
    return f'<div class="category-grid">{"".join(cards)}</div>'


def _render_confidence_coverage(qcc: Optional[dict]) -> str:
    """
    STEP 13 -- Quality / Confidence / Coverage separation: renders
    Confidence and Coverage as two more cards alongside the existing
    Overall Quality gauge, using the identical card style
    _render_categories already established -- these three numbers are
    meant to be read side by side (see this module's docstring for why
    the same Quality score can mean very different things depending on
    Confidence/Coverage).
    """
    if not qcc:
        return ""

    def _card(title, subtitle, axis):
        color = _color_for(axis["classification"])
        score_display = f'{axis["score"]:.1f}' if axis["score"] is not None else "&mdash;"
        components_html = "".join(
            f'<li>{escape(c["detail"])}</li>' for c in axis["components"]
        )
        components_block = (
            f'<details style="margin-top:0.4rem;"><summary style="cursor:pointer; color:var(--text-secondary); font-size:0.78rem;">why?</summary>'
            f'<ul style="margin:0.3rem 0 0; padding-left:1.1rem; font-size:0.78rem; color:var(--text-secondary);">{components_html}</ul></details>'
            if axis["components"] else ""
        )
        return f"""
        <div class="category-card" style="--card-color:{color}">
          <div class="cat-label">{escape(title)}</div>
          <div class="cat-metric">{escape(subtitle)}</div>
          <div class="cat-score" style="color:{color}">{score_display}</div>
          {_badge(axis["classification"])}
          {components_block}
        </div>
        """

    cards = (
        _card("Confidence", "How much this run's own measurement process can be trusted.", qcc["confidence"])
        + _card("Coverage", "How much of the sensor's depth/FOV range was actually exercised.", qcc["coverage"])
    )
    return f"""
    <section class="metric-section">
      <h3>STEP13 &middot; Quality / Confidence / Coverage</h3>
      <p class="metric-subtitle">The same Overall Quality score above can mean very different things depending on these two: a high score from a thorough, well-synced, fully-matched measurement across the sensor's whole range (high Confidence + Coverage) is trustworthy; the same score from a thin, narrow, partially-failed one is not yet a verdict.</p>
      <div class="category-grid">{cards}</div>
    </section>
    """


def _render_input_validation(input_validation: Optional[dict]) -> str:
    """
    STEP 1 -- Input Validation section. Rendered near the top of the
    report, before any calibration-quality metric, so a broken input reads
    as exactly that -- not folded into (or mistaken for) a calibration
    verdict. Omitted entirely if input_validation is None (e.g. --demo).
    """
    if input_validation is None:
        return ""
    status = input_validation["status"]
    reasons = input_validation.get("reasons") or []
    reasons_html = (
        "<ul class='data-table' style='padding-left:1.25rem; margin-top:0.5rem;'>"
        + "".join(f"<li style='color:var(--text-secondary); margin:0.25rem 0;'>{escape(r)}</li>" for r in reasons)
        + "</ul>"
    ) if reasons else ""
    return f"""
    <section class="metric-section">
      <h3>Input Validation {_badge(status)}</h3>
      <p class="metric-subtitle">STEP1 -- does the raw camera/LiDAR/dataset input itself hold up, before any calibration judgement is made?</p>
      {reasons_html}
    </section>
    """


def _render_synchronization(sync: Optional[dict]) -> str:
    """
    STEP 2 -- Timestamp Synchronization section: matched frame count,
    estimated clock offset (Δt) and its residual jitter (std), drop
    ratio, and GOOD/WARNING/BAD/FAIL classification. Omitted if sync
    never ran (sync is None).
    """
    if sync is None:
        return ""
    return f"""
    <section class="metric-section">
      <h3>Synchronization {_badge(sync["classification"])}</h3>
      <p class="metric-subtitle">STEP2 -- candidate-window + monotonic camera&harr;LiDAR frame matching, with clock offset (&Delta;t) estimation.</p>
      <div class="stat-grid">
        {_stat("Matched frames", f'{sync["num_matched"]} / {sync["num_camera_frames"]}')}
        {_stat("Mean &Delta;t", _fmt(sync["estimated_offset_ms"], " ms", digits=1))}
        {_stat("Offset std", _fmt(sync["offset_std_ms"], " ms", digits=1))}
        {_stat("Drop ratio", _fmt(sync["drop_ratio"] * 100 if sync["drop_ratio"] is not None else None, "%", digits=1))}
        {_stat("Camera dropped", _fmt(sync["num_camera_dropped"]))}
        {_stat("LiDAR dropped", _fmt(sync["num_lidar_dropped"]))}
      </div>
    </section>
    """


def _render_projection_overlay(projection_overlay_png: Optional[bytes]) -> str:
    """
    STEP 3 -- raw projection sanity-check section: every valid projected
    LiDAR point drawn on the plain camera image, colored by depth. Unlike
    the M2 section's overlay (GOOD/WARNING/BAD per-point, matched edge
    points only), this needs no edge-matching to have run at all -- it's
    meant as the first, cheapest "does this even look right" check.
    Omitted if the image wasn't generated (e.g. --no-visuals, or a broken
    3D/plotting environment for OTHER visuals -- this one is plain OpenCV
    2D drawing, so it doesn't share that particular failure mode).
    """
    if not projection_overlay_png:
        return ""
    return f"""
    <section class="metric-section">
      <h3>Projection Sanity Check</h3>
      <p class="metric-subtitle">STEP3 -- every valid LiDAR point projected onto the raw camera image, colored by depth (near&rarr;warm, far&rarr;cool). No edge-matching involved -- just "does the projected point cloud's shape follow the image's real geometry?"</p>
      {_img_tag(projection_overlay_png, "All valid projected LiDAR points over the raw camera image, colored by depth")}
    </section>
    """


def _render_range_image(range_image_png: Optional[bytes]) -> str:
    """
    STEP 4 -- LiDAR Ring/Topology section: the range image (ring rows x
    azimuth columns, colored by range) with LiDAR-native depth-
    discontinuity cells marked. Independent of camera projection/
    extrinsic entirely -- this is purely "what does the LiDAR's own scan
    structure look like". Omitted if the image wasn't generated (e.g.
    --no-visuals, or a broken plotting environment for OTHER visuals --
    this one shares camera_frustum/colorized_pointcloud's matplotlib
    degrade-to-None pattern, so it can also be silently absent there).
    """
    if not range_image_png:
        return ""
    return f"""
    <section class="metric-section">
      <h3>LiDAR Range Image</h3>
      <p class="metric-subtitle">STEP4 -- the LiDAR's own scan structure (ring &times; azimuth), colored by range. Highlighted points are LiDAR-native depth discontinuities -- detected from adjacent laser returns in the actual scan, not from where points happen to land after projection.</p>
      {_img_tag(range_image_png, "LiDAR range image: ring rows by azimuth columns, colored by range, with native depth-discontinuity cells highlighted")}
    </section>
    """


def _render_motion_deskew(motion_deskew: Optional[dict], deskew_comparison_png: Optional[bytes]) -> str:
    """
    STEP 5 -- Motion Deskew section: the platform-velocity-driven
    before/after comparison (motion.deskew.compare_before_after's
    summary dict + visualization.deskew_comparison's BEV/histogram
    image). Opt-in (app.cli's --deskew-* flags) -- omitted entirely if
    the caller never ran deskewing (motion_deskew is None), since
    deskewing needs an external platform-velocity input this tool has no
    way to measure on its own.
    """
    if motion_deskew is None:
        return ""
    return f"""
    <section class="metric-section">
      <h3>Motion Deskew</h3>
      <p class="metric-subtitle">STEP5 -- per-point correction for platform motion during the LiDAR scan (constant-velocity model). At zero platform velocity this is exactly a no-op; the numbers below reflect the velocity actually supplied for this run.</p>
      <div class="stat-grid">
        {_stat("Points", _fmt(motion_deskew.get("num_points")))}
        {_stat("Scan period", _fmt(motion_deskew.get("scan_period_s"), " s"))}
        {_stat("Reference time", _fmt(motion_deskew.get("reference_time_s"), " s"))}
        {_stat("Mean correction", _fmt(motion_deskew.get("mean_correction_m"), " m"))}
        {_stat("P95 correction", _fmt(motion_deskew.get("p95_correction_m"), " m"))}
        {_stat("Max correction", _fmt(motion_deskew.get("max_correction_m"), " m"))}
      </div>
      {_img_tag(deskew_comparison_png, "Bird's-eye view of LiDAR points before vs after motion deskew, plus a histogram of per-point correction magnitude")}
    </section>
    """


def _render_dynamic_filter(dynamic_filter: Optional[dict], dynamic_filter_overlay_png: Optional[bytes]) -> str:
    """
    STEP 8 -- Dynamic Object Filtering section: M2 computed with vs
    without moving-object points (evaluation.dynamic_filter.
    DynamicFilteringComparison), plus the static/dynamic/unknown overlay
    image. Opt-in (app.cli's --dynamic-filter flag) -- omitted entirely
    if the caller never ran it. Diagnostic only: never changes
    quality_score, which is always computed on the unfiltered point set.
    """
    if dynamic_filter is None:
        return ""
    contamination = dynamic_filter.get("dynamic_contamination_ratio")
    contamination_str = _fmt(contamination * 100 if contamination is not None else None, "%", digits=1)
    return f"""
    <section class="metric-section">
      <h3>Dynamic Object Filtering</h3>
      <p class="metric-subtitle">STEP8 -- M2 computed with every edge point ("overall") vs. with points on likely-moving objects removed ("static only"), so apparent misalignment caused by a moving object (rather than the calibration itself) can be told apart. Classification comes from multi-frame motion consistency and assumes the platform was approximately stationary across the frame window used -- see evaluation/dynamic_filter.py for that caveat.</p>
      <div class="stat-grid">
        {_stat("Overall mean error", _fmt(dynamic_filter.get("overall_mean_px"), " px"))}
        {_badge(dynamic_filter.get("overall_classification", "FAIL"))}
        {_stat("Static-only mean error", _fmt(dynamic_filter.get("static_only_mean_px"), " px"))}
        {_badge(dynamic_filter.get("static_only_classification", "FAIL"))}
        {_stat("Dynamic contamination", contamination_str)}
        {_stat("Points removed", _fmt(dynamic_filter.get("num_dynamic_points_removed")))}
      </div>
      {_img_tag(dynamic_filter_overlay_png, "Projected LiDAR points colored by static (green) / dynamic (red) / unknown (gray) classification")}
    </section>
    """


def _render_root_cause(root_cause: Optional[dict]) -> str:
    """
    STEP 12 -- Root Cause Diagnosis Engine: the ranked list of plausible
    causes evaluation.root_cause.diagnose_root_cause produces by cross-
    referencing every other diagnostic this report already contains
    (sync, M2/M3/M4, spatial analysis, dynamic filtering, sensitivity).

    STEP 14 -- confirmations (the spec's own 🟢 "Timestamp OK" / "Sensor
    quality OK" half of its diagnosis panel example) are rendered as
    additional rows in the SAME table, right after the candidates --
    exactly the spec's mixed panel, not a separate "everything is fine"
    box competing for attention elsewhere in the report.

    Always present in the report (root_cause_diagnosis is never None),
    but this section renders nothing when there's neither a candidate
    NOR a confirmation to show, rather than an empty box.
    """
    candidates = (root_cause or {}).get("candidates") or []
    confirmations = (root_cause or {}).get("confirmations") or []
    if not candidates and not confirmations:
        return ""

    rows = "".join(
        f'<tr><td>{i}</td><td>{escape(c["label"])}</td><td>{_badge(c["confidence"])}</td>'
        f'<td><ul style="margin:0; padding-left:1.1rem;">'
        + "".join(f"<li>{escape(e)}</li>" for e in c["evidence"])
        + '</ul></td></tr>'
        for i, c in enumerate(candidates, start=1)
    )
    confirmation_rows = "".join(
        f'<tr style="opacity:0.85;"><td>{len(candidates) + i}</td><td>{escape(c["label"])}</td>'
        f'<td><span class="badge" style="color:{_color_for("GOOD")}">OK</span></td>'
        f'<td>{escape(c["detail"])}</td></tr>'
        for i, c in enumerate(confirmations, start=1)
    )
    return f"""
    <section class="metric-section" style="border-color:var(--accent, #58A6FF);">
      <h3>&#11088; Root Cause Diagnosis</h3>
      <p class="metric-subtitle">STEP12 -- a ranked, rule-based ("if X and Y, then Z") diagnosis combining every signal in this report (timestamp sync, M2/M3/M4, depth/spatial analysis, dynamic object filtering, and calibration sensitivity) into plausible explanations for what's actually wrong, not just how wrong it is. STEP14 -- confirmed-clean checks (&#128994;) are listed alongside flagged problems, not just the problems.</p>
      <table class="data-table">
        <thead><tr><th>#</th><th>Cause</th><th>Confidence</th><th>Evidence</th></tr></thead>
        <tbody>{rows}{confirmation_rows}</tbody>
      </table>
    </section>
    """


def _render_m2(
    m2: dict,
    overlay_png: Optional[bytes] = None,
    histogram_png: Optional[bytes] = None,
    colorized_pointcloud_png: Optional[bytes] = None,
    error_heatmap_png: Optional[bytes] = None,
    bev_dual_panel_png: Optional[bytes] = None,
    uncertainty_plot_png: Optional[bytes] = None,
) -> str:
    match_rate_stat = ""
    if m2.get("match_rate") is not None:
        match_rate_stat = _stat("Match rate (STEP6)", _fmt(m2["match_rate"] * 100, "%", digits=1))
    normalized_error_stat = ""
    if m2.get("mean_normalized_error") is not None:
        normalized_error_stat = _stat("Mean normalized error", _fmt(m2["mean_normalized_error"], "x floor"))
    return f"""
    <section class="metric-section">
      <h3>M2 &middot; Edge Alignment {_badge(m2["classification"])}</h3>
      <p class="metric-subtitle">How closely projected LiDAR depth-discontinuity points land on actual image edges.</p>
      <div class="stat-grid">
        {_stat("Mean error", _fmt(m2["mean_px"], " px"))}
        {_stat("Median error", _fmt(m2["median_px"], " px"))}
        {_stat("P95 error", _fmt(m2["p95_px"], " px"))}
        {_stat("Max error", _fmt(m2["max_px"], " px"))}
        {_stat("Noise floor", _fmt(m2["floor_px"], " px"))}
        {_stat("Edge points", _fmt(m2["num_edge_points"]))}
        {match_rate_stat}
        {normalized_error_stat}
      </div>
      {_img_tag(overlay_png, "Projected LiDAR edge points over the camera image, colored GOOD/WARNING/BAD")}
      {_img_tag(histogram_png, "Histogram of per-point alignment error")}
      {_render_warning_list(m2["warnings"])}
      <p class="metric-subtitle" style="margin-top:1.25rem;">Spatial error heatmap: image split into a grid, with each cell's average error shown as a translucent GOOD/WARNING/BAD color. Errors concentrated at edges/corners or one side of the frame point at a specific cause (e.g. distortion, a small rotation offset) rather than uniform miscalibration.</p>
      {_img_tag(error_heatmap_png, "Grid heatmap of spatially-aggregated alignment error, colored GOOD/WARNING/BAD")}
      <p class="metric-subtitle" style="margin-top:1.25rem;">Bird's-eye view: the same edge points shown in both the camera image and a top-down view (X vs depth), colored identically in each. Makes it easy to see whether error grows with distance or concentrates on one side.</p>
      {_img_tag(bev_dual_panel_png, "Camera image and bird's-eye view side by side, with the same edge points highlighted and colored to match in both")}
      <p class="metric-subtitle" style="margin-top:1.25rem;">Fused view: LiDAR points colorized by the camera pixel they project onto. Color bleed or smearing at object edges is a visual sign of extrinsic misalignment.</p>
      {_img_tag(colorized_pointcloud_png, "LiDAR point cloud colorized by projected camera pixel, shown from a 3D angle and bird's-eye view")}
      <p class="metric-subtitle" style="margin-top:1.25rem;">STEP7 -- per-point error vs. the sensor-relative noise floor at EACH point's own depth: points hugging the floor(Z) curve are consistent with ordinary sensor noise (however large their raw pixel error looks in isolation); points well above it are the ones actually worth investigating as a calibration problem.</p>
      {_img_tag(uncertainty_plot_png, "Scatter of per-point error against depth, with the sensor-relative noise floor curve and GOOD/WARNING/BAD bands overlaid")}
    </section>
    """


def _render_bin_table(bins: dict, order: list) -> str:
    rows = ""
    for label in order:
        b = bins.get(label)
        if b is None:
            continue
        total = b.get("total_count", 0)
        failed = b.get("failure_count", 0)
        rows += f"""
          <tr>
            <td>{escape(label)}</td>
            <td>{_fmt(b.get("mean_px"), " px")}</td>
            <td>{_fmt(b.get("median_px"), " px")}</td>
            <td>{_fmt(b.get("p95_px"), " px")}</td>
            <td>{_fmt(b.get("std_px"), " px")}</td>
            <td>{total}</td>
            <td>{failed}</td>
          </tr>"""
    return f"""
      <table class="data-table">
        <thead><tr><th>Region</th><th>Mean</th><th>Median</th><th>P95</th><th>Std</th><th>n</th><th>Failed</th></tr></thead>
        <tbody>{rows}
        </tbody>
      </table>
    """


def _render_spatial_analysis(spatial: Optional[dict], spatial_analysis_png: Optional[bytes]) -> str:
    """
    STEP 9 -- Depth/Spatial Error Analysis section: mean/median/P95/std +
    valid/failure counts per depth bin (0-10m/10-20m/20-30m/30-50m/50m+)
    and per camera region (LEFT/CENTER/RIGHT, TOP/CENTER/BOTTOM), plus the
    bar-chart visualization and a plain-language depth-trend line. Always
    present when M2 succeeded (no extra opt-in input needed, unlike
    STEP5/STEP8) -- omitted only if M2 itself FAILed.
    """
    if spatial is None:
        return ""
    trend = spatial.get("depth_trend")
    trend_text = {
        "increases_with_depth": "Error increases with depth -- worth checking whether this is within sensor-relative expectations (see the STEP7 uncertainty plot above) or a genuine calibration issue that only shows up at range.",
        "decreases_with_depth": "Error decreases with depth.",
        "stable": "No clear depth trend -- error doesn't consistently grow or shrink with range.",
        None: "Not enough populated depth bins to determine a trend.",
    }.get(trend, "")
    return f"""
    <section class="metric-section">
      <h3>M2 &middot; Depth / Spatial Analysis</h3>
      <p class="metric-subtitle">STEP9 -- M2's per-point errors broken down by depth bin and by camera region, instead of collapsed into one mean. {escape(trend_text)}</p>
      {_img_tag(spatial_analysis_png, "Bar charts of mean error (with std error bars and P95 markers) by depth bin, horizontal region, and vertical region")}
      <div style="display:grid; grid-template-columns: 1fr 1fr 1fr; gap:1rem; margin-top:1rem;">
        <div><h4 style="color:var(--text-secondary); font-size:0.8rem; margin-bottom:0.5rem;">By depth</h4>{_render_bin_table(spatial.get("depth_bins", {}), ["0-10m", "10-20m", "20-30m", "30-50m", "50m+"])}</div>
        <div><h4 style="color:var(--text-secondary); font-size:0.8rem; margin-bottom:0.5rem;">By horizontal region</h4>{_render_bin_table(spatial.get("horizontal_regions", {}), ["LEFT", "CENTER", "RIGHT"])}</div>
        <div><h4 style="color:var(--text-secondary); font-size:0.8rem; margin-bottom:0.5rem;">By vertical region</h4>{_render_bin_table(spatial.get("vertical_regions", {}), ["TOP", "CENTER", "BOTTOM"])}</div>
      </div>
    </section>
    """


def _render_m3(m3: dict) -> str:
    rows = "".join(
        f'<tr><td>{b["block_index"]}</td><td>{b["num_frames_valid"]}/{b["num_frames_total"]}</td>'
        f'<td>{_fmt(b["mean_px"], " px")}</td><td>{_fmt(b["p95_px"], " px")}</td>'
        f'<td>{_fmt(b["representative_depth_m"], " m")}</td><td>{_fmt(b["edge_density"], "/frame")}</td>'
        f'<td>{_fmt(b["fov_coverage"] * 100 if b["fov_coverage"] is not None else None, "%", digits=0)}</td>'
        f'<td>{_badge(b["classification"])}</td></tr>'
        for b in m3["blocks"]
    )
    table = f"""
    <table class="data-table">
      <thead><tr><th>Block</th><th>Frames</th><th>Mean</th><th>P95</th><th>Depth</th><th>Edge density</th><th>FOV coverage</th><th>Status</th></tr></thead>
      <tbody>{rows}</tbody>
    </table>
    """ if m3["blocks"] else "<p style='color:var(--text-secondary)'>No blocks evaluated.</p>"

    diagnosis = m3.get("instability_diagnosis")
    diagnosis_html = ""
    if diagnosis and diagnosis.get("candidates"):
        top = diagnosis["candidates"][0]
        diagnosis_html = f"""
        <div class="warning-item" style="margin-top:0.75rem;">
          <strong>STEP10 instability diagnosis:</strong> Block {diagnosis["worst_block_index"]}
          (mean {_fmt(diagnosis["worst_block_mean_px"], " px")}) differs from the other blocks'
          {escape(top["metric"])} by {top["relative_diff"]:+.0%} &mdash; possible cause:
          <em>{escape(top["explanation"])}</em>.
        </div>
        """

    return f"""
    <section class="metric-section">
      <h3>M3 &middot; Hold-out Consistency {_badge(m3["classification"])}</h3>
      <p class="metric-subtitle">Whether this fixed calibration performs consistently across different contiguous time windows.</p>
      <div class="stat-grid">
        {_stat("Mean across blocks", _fmt(m3["mean_across_blocks_px"], " px"))}
        {_stat("STD across blocks", _fmt(m3["std_across_blocks_px"], " px"))}
        {_stat("Range", _fmt(m3["range_px"], " px"))}
        {_stat("Noise floor", _fmt(m3["floor_px"], " px"))}
        {_stat("Valid blocks", _fmt(m3["num_valid_blocks"]))}
      </div>
      {table}
      {diagnosis_html}
      {_render_warning_list(m3["warnings"])}
    </section>
    """


def _render_m4(m4: dict, trajectory_png: Optional[bytes] = None) -> str:
    trajectory = m4["frame_trajectory"]
    outlier_idx = set(m4["outlier_frame_indices"])
    # Keep the table readable on long sequences: show all outliers plus a
    # bounded sample of the rest, rather than every frame.
    sample = [f for f in trajectory if f["frame_index"] in outlier_idx]
    non_outliers = [f for f in trajectory if f["frame_index"] not in outlier_idx]
    sample += non_outliers[:5]
    if len(non_outliers) > 5:
        sample += non_outliers[-5:]
    sample.sort(key=lambda f: f["frame_index"])

    rows = "".join(
        f'<tr style="{"color:var(--bad)" if f["is_outlier"] else ""}">'
        f'<td>{f["frame_index"]}</td><td>{_fmt(f["mean_px"], " px")}</td>'
        f'<td>{_fmt(f["robust_z_score"])}</td>'
        f'<td>{_badge(f["classification"])}</td>'
        f'<td>{"outlier" if f["is_outlier"] else ""}</td></tr>'
        for f in sample
    )
    note = (
        f"<p style='color:var(--text-secondary); font-size:0.8rem;'>"
        f"Showing {len(sample)} of {len(trajectory)} frames (all outliers, plus a sample).</p>"
        if len(sample) < len(trajectory) else ""
    )
    table = f"""
    <table class="data-table">
      <thead><tr><th>Frame</th><th>Mean</th><th>Robust z</th><th>Status</th><th></th></tr></thead>
      <tbody>{rows}</tbody>
    </table>
    {note}
    """ if trajectory else "<p style='color:var(--text-secondary)'>No frames evaluated.</p>"

    return f"""
    <section class="metric-section">
      <h3>M4 &middot; Multi-frame Consistency {_badge(m4["classification"])}</h3>
      <p class="metric-subtitle">Whether error stays stable frame-to-frame, or specific frames spike. Outlier detection: {escape(m4.get("outlier_method", "multiplier"))} method (STEP10 -- MAD/IQR-based robust statistics, replacing a plain multiple-of-median rule).</p>
      <div class="stat-grid">
        {_stat("Mean", _fmt(m4["mean_across_frames_px"], " px"))}
        {_stat("STD", _fmt(m4["std_across_frames_px"], " px"))}
        {_stat("P95", _fmt(m4["p95_across_frames_px"], " px"))}
        {_stat("Max", _fmt(m4["max_across_frames_px"], " px"))}
        {_stat("MAD", _fmt(m4.get("mad_px"), " px"))}
        {_stat("IQR", _fmt(m4.get("iqr_px"), " px"))}
        {_stat("Valid ratio", _fmt(m4.get("valid_ratio") * 100 if m4.get("valid_ratio") is not None else None, "%", digits=1))}
        {_stat("Failure ratio", _fmt(m4.get("failure_ratio") * 100 if m4.get("failure_ratio") is not None else None, "%", digits=1))}
        {_stat("Outlier ratio", _fmt(m4.get("outlier_ratio") * 100 if m4.get("outlier_ratio") is not None else None, "%", digits=1))}
      </div>
      {table}
      {_img_tag(trajectory_png, "Per-frame error trajectory with outliers marked")}
      {_render_warning_list(m4["warnings"])}
    </section>
    """


def _render_sequence_gif(gif_bytes: Optional[bytes]) -> str:
    """Render the (opt-in, --sequence-gif) animated overlay section.
    Returns "" if no GIF was generated, so callers can splice this in
    unconditionally."""
    if not gif_bytes:
        return ""
    return f"""
    <section class="metric-section">
      <h3>Sequence Overlay</h3>
      <p class="metric-subtitle">M2's overlay, sampled across the sequence and animated -- shows whether alignment quality holds steady over time or drifts, rather than a single snapshot.</p>
      {_img_tag(gif_bytes, "Animated GIF of the M2 overlay across sampled frames in the sequence", mime="image/gif")}
    </section>
    """


def _render_warning_list(warnings: list[str]) -> str:
    if not warnings:
        return ""
    items = "".join(f'<div class="warning-item">{escape(w)}</div>' for w in warnings)
    return f'<div style="margin-top:1rem;">{items}</div>'


_VENDOR_DIR = os.path.join(os.path.dirname(__file__), "vendor")


@lru_cache(maxsize=1)
def _load_plotly_js() -> str:
    """Read the vendored plotly.js gl3d bundle off disk (see
    report/vendor/README.md). Cached so repeated report generation in a
    single process only pays the ~1.7MB read once."""
    path = os.path.join(_VENDOR_DIR, "plotly-gl3d.min.js")
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _render_rig_geometry(
    metadata: dict,
    frustum_png: Optional[bytes] = None,
    interactive_scene: Optional[dict] = None,
    div_id: str = "cam-lidar-interactive-3d",
) -> str:
    """
    Render the top-of-report "Rig Geometry" section. If both a static
    frustum PNG and interactive scene data are available, shows a
    Static/Interactive toggle (defaulting to Static, since it costs
    nothing to display) with the interactive Plotly scene lazily
    initialized on first switch -- not on page load -- so opening the
    report doesn't pay Plotly's render cost unless someone actually asks
    for it. Falls back to whichever single view is available, or "" if
    neither is.
    """
    if not frustum_png and not interactive_scene:
        return ""

    ext = metadata["extrinsic"]
    stat_grid = f"""
      <div class="stat-grid">
        {_stat("Baseline", _fmt(ext["baseline_m"], " m"))}
        {_stat("Parent &rarr; child", f'{escape(ext["parent"])} &rarr; {escape(ext["child"])}')}
      </div>
    """

    if frustum_png and interactive_scene:
        scene_json = json.dumps(interactive_scene, separators=(",", ":"))
        body = f"""
      <div class="view-toggle">
        <button type="button" class="view-toggle-btn active" data-target="rig-static-view">Static</button>
        <button type="button" class="view-toggle-btn" data-target="rig-interactive-view">Interactive</button>
      </div>
      <div id="rig-static-view" class="view-panel active">
        {_img_tag(frustum_png, "3D view of the camera's position and viewing frustum in the LiDAR coordinate frame")}
      </div>
      <div id="rig-interactive-view" class="view-panel">
        <div id="{div_id}" style="width:100%; height:520px; margin-top:0.75rem; border-radius:10px; overflow:hidden;"></div>
      </div>
      <script>
        (function() {{
          var scene = {scene_json};
          var rendered = false;
          var panels = document.querySelectorAll('#rig-geometry-section .view-panel');
          var buttons = document.querySelectorAll('#rig-geometry-section .view-toggle-btn');
          buttons.forEach(function(btn) {{
            btn.addEventListener('click', function() {{
              var target = btn.getAttribute('data-target');
              panels.forEach(function(p) {{ p.classList.toggle('active', p.id === target); }});
              buttons.forEach(function(b) {{ b.classList.toggle('active', b === btn); }});
              if (target === 'rig-interactive-view' && !rendered) {{
                Plotly.newPlot({div_id!r}, scene.data, scene.layout, scene.config);
                rendered = true;
              }}
            }});
          }});
        }})();
      </script>
    """
    elif interactive_scene:
        scene_json = json.dumps(interactive_scene, separators=(",", ":"))
        body = f"""
      <div id="{div_id}" style="width:100%; height:520px; margin-top:0.75rem; border-radius:10px; overflow:hidden;"></div>
      <script>
        (function() {{
          var scene = {scene_json};
          Plotly.newPlot({div_id!r}, scene.data, scene.layout, scene.config);
        }})();
      </script>
    """
    else:
        body = _img_tag(frustum_png, "3D view of the camera's position and viewing frustum in the LiDAR coordinate frame")

    return f"""
    <section class="metric-section" id="rig-geometry-section">
      <h3>Rig Geometry</h3>
      <p class="metric-subtitle">Camera position and viewing frustum placed in the LiDAR frame, from the extrinsic under evaluation &mdash; a physical sanity check that's easier to read at a glance than raw translation/rotation numbers. The interactive view also overlays the colorized point cloud; drag to orbit, scroll to zoom.</p>
      {stat_grid}
      {body}
    </section>
    """


def _render_metadata(metadata: dict) -> str:
    cam = metadata["camera"]
    T = metadata["extrinsic"]["T_CL"]
    matrix_cells = "".join(f"<span>{v:.4f}</span>" for row in T for v in row)
    return f"""
    <section class="metric-section">
      <h3>Configuration</h3>
      <div class="stat-grid">
        {_stat("Camera model", escape(cam["model"]))}
        {_stat("Resolution", f'{cam["width"]}&times;{cam["height"]}')}
        {_stat("fx / fy", f'{cam["fx"]:.1f} / {cam["fy"]:.1f}')}
        {_stat("LiDAR source", escape(metadata["lidar"]["source_kind"]))}
        {_stat("Baseline", _fmt(metadata["extrinsic"]["baseline_m"], " m"))}
        {_stat("Synced frames", _fmt(metadata["dataset"]["num_synced_frames"]))}
      </div>
      <div class="stat-label" style="margin-top:0.8rem; margin-bottom:0.2rem;">T_CL (camera_from_lidar)</div>
      <div class="matrix">{matrix_cells}</div>
    </section>
    """


def _render_advanced(advanced: Optional[dict], sensitivity_png: Optional[bytes] = None) -> str:
    if not advanced:
        return ""
    parts = []

    plane = advanced.get("plane_consistency")
    if plane:
        parts.append(f"""
        <section class="metric-section">
          <h3>Plane Consistency {_badge(plane["classification"])}</h3>
          <p class="metric-subtitle">Advanced diagnostic: how well the dominant flat surface (ground/wall) lines up with its image silhouette.</p>
          <div class="stat-grid">
            {_stat("Plane found", _fmt(plane["plane_found"]))}
            {_stat("Inlier ratio", f'{plane["inlier_ratio"]*100:.1f}%' if plane["inlier_ratio"] is not None else "&mdash;")}
            {_stat("Boundary points", _fmt(plane["num_boundary_points"]))}
            {_stat("Mean error", _fmt(plane["mean_px"], " px"))}
          </div>
          {_render_warning_list(plane["warnings"])}
        </section>
        """)

    perturbation = advanced.get("perturbation")
    if perturbation:
        best = perturbation.get("best_sample")
        best_str = (f'{best["axis"]} {best["direction"]}{best["delta"]} &rarr; {best["mean_px"]:.3f} px'
                    if best else "&mdash;")
        axis_sensitivities = perturbation.get("axis_sensitivities") or []
        sensitivity_rows = "".join(
            f'<tr><td>{escape(a["axis"])}</td><td>{_badge(a["classification"])}</td>'
            f'<td>{_fmt(a["small_delta_effect_px"], " px")}</td><td>{_fmt(a["large_delta_effect_px"], " px")}</td></tr>'
            for a in axis_sensitivities
        )
        sensitivity_table = f"""
        <table class="data-table">
          <thead><tr><th>Axis</th><th>Sensitivity</th><th>Effect @ smallest &Delta;</th><th>Effect @ largest &Delta;</th></tr></thead>
          <tbody>{sensitivity_rows}</tbody>
        </table>
        """ if axis_sensitivities else ""
        timestamp_note = (
            "" if perturbation.get("timestamp_sensitivity_computed")
            else "<p class='metric-subtitle' style='margin-top:0.5rem;'>Timestamp sensitivity not computed -- "
                 "requires an assumed platform velocity (this tool has no independent way to measure one).</p>"
        )
        parts.append(f"""
        <section class="metric-section">
          <h3>Perturbation Sensitivity {_badge(perturbation["classification"])}</h3>
          <p class="metric-subtitle">Advanced diagnostic: does a small nudge to T_CL find a better alignment nearby?</p>
          <div class="stat-grid">
            {_stat("Baseline", _fmt(perturbation["baseline_mean_px"], " px"))}
            {_stat("At local minimum", _fmt(perturbation["is_local_minimum"]))}
            {_stat("Improvement margin", _fmt(perturbation["improvement_margin_px"], " px"))}
            {_stat("Best nudge", best_str)}
          </div>
          {_render_warning_list(perturbation["warnings"])}
          <p class="metric-subtitle" style="margin-top:1.25rem;">STEP11 -- Calibration Sensitivity Analysis: how much each parameter's error changes as it's nudged by increasing amounts, ranked HIGH/MEDIUM/LOW relative to the sensor-relative noise floor. This is the data a future Root Cause Diagnosis Engine would use (e.g. "yaw sensitivity is HIGH and error concentrates on one side of the frame" -> yaw misalignment is a plausible cause).</p>
          {_img_tag(sensitivity_png, "Horizontal bar chart ranking each calibration parameter's sensitivity (HIGH/MEDIUM/LOW)")}
          {sensitivity_table}
          {timestamp_note}
        </section>
        """)

    drift = advanced.get("temporal_drift")
    if drift:
        parts.append(f"""
        <section class="metric-section">
          <h3>Temporal Drift {_badge(drift["classification"])}</h3>
          <p class="metric-subtitle">Advanced diagnostic: does per-frame error trend up or down over the sequence?</p>
          <div class="stat-grid">
            {_stat("Slope", _fmt(drift["slope_px_per_frame"], " px/frame", digits=5))}
            {_stat("Significant", _fmt(drift["is_statistically_significant"]))}
            {_stat("p-value", _fmt(drift["p_value"], digits=4))}
            {_stat("Total drift", _fmt(drift["total_drift_px"], " px"))}
          </div>
          {_render_warning_list(drift["warnings"])}
        </section>
        """)

    if not parts:
        return ""
    return '<h2 style="font-family:var(--font-display); font-size:1rem; color:var(--text-secondary); text-transform:uppercase; letter-spacing:0.06em; margin: 2rem 0 1rem;">Advanced Diagnostics</h2>' + "".join(parts)


def render_html_report(report: dict, visuals: Optional[dict] = None) -> str:
    """
    Render a full self-contained HTML document string from a report dict
    built by report/builder.py.build_report().

    visuals: optional dict of PNG bytes to embed as base64 data URIs (so
    the HTML stays a single shareable file). Recognized keys:
      "overlay_png"              -- from visualization.overlay.render_overlay(...)
      "histogram_png"            -- from visualization.histogram.render_error_histogram_png(...)
      "trajectory_png"           -- from visualization.trajectory.render_m4_trajectory_png(...)
      "colorized_pointcloud_png" -- from visualization.colorized_pointcloud.render_colorized_pointcloud_from_frame(...)
      "error_heatmap_png"        -- from visualization.error_heatmap.render_error_heatmap_from_result(...)
      "camera_frustum_png"       -- from visualization.camera_frustum.render_camera_frustum_from_dataset(...)
      "bev_dual_panel_png"       -- from visualization.bev_dual_panel.render_bev_dual_panel_from_result(...)
      "uncertainty_plot_png"     -- from visualization.uncertainty_plot.render_uncertainty_plot_from_result(...)
                                     (STEP7: per-point error vs. sensor-relative noise floor at each point's
                                     own depth -- distinguishes sensor noise from real calibration error)
      "spatial_analysis_png"     -- from visualization.spatial_analysis_plot.render_spatial_analysis_from_result(...)
                                     (STEP9: M2 error broken down by depth bin and camera region instead of
                                     one mean -- see report["m2_spatial_analysis"] for the paired numeric tables)
      "projection_overlay_png"   -- from visualization.projection_overlay.render_projection_overlay_from_frame(...)
                                     (STEP3: raw depth-colored LiDAR->image projection, independent of M2's
                                     edge-matching -- a basic "does projection look sane" sanity check)
      "range_image_png"          -- from visualization.range_image.render_range_image_from_points(...)
                                     (STEP4: LiDAR-native ring x azimuth range image with native depth-
                                     discontinuity cells highlighted -- independent of camera projection entirely)
      "deskew_comparison_png"    -- from visualization.deskew_comparison.render_deskew_comparison_from_points(...)
                                     (STEP5: before/after BEV overlay + correction-magnitude histogram for
                                     motion deskew -- only present if the caller opted in with a nonzero
                                     platform velocity; see report["motion_deskew"] for the paired numeric summary)
      "dynamic_filter_overlay_png" -- from visualization.dynamic_filter_overlay.render_dynamic_filter_overlay_from_frame(...)
                                     (STEP8: projected points colored static/dynamic/unknown -- only present if
                                     the caller opted in with --dynamic-filter; see report["dynamic_filter"] for
                                     the paired overall-vs-static-only numeric comparison)
      "interactive_scene"        -- a dict from visualization.interactive_viewer.build_interactive_scene(...)
                                     (NOT bytes -- raw JSON-serializable scene data, embedded + rendered
                                     client-side via the vendored plotly.js gl3d bundle)
      "sequence_gif"             -- GIF bytes from visualization.sequence.render_sequence_gif(...)
                                     (opt-in via app.cli's --sequence-gif; embedded as image/gif, not image/png)
    Any missing/None key simply omits that image -- visualization is
    optional and the report renders fine without it (see report/json.py's
    counterpart: the JSON report never carries images, only this HTML one).
    """
    visuals = visuals or {}
    metadata = report["metadata"]
    quality = report["quality_score"]

    body = (
        _render_input_validation(report.get("input_validation"))
        + _render_synchronization(report.get("synchronization"))
        + _render_hero(quality)
        + _render_categories(quality)
        + _render_confidence_coverage(report.get("quality_confidence_coverage"))
        + _render_root_cause(report.get("root_cause_diagnosis"))
        + _render_rig_geometry(metadata, visuals.get("camera_frustum_png"), visuals.get("interactive_scene"))
        + _render_projection_overlay(visuals.get("projection_overlay_png"))
        + _render_range_image(visuals.get("range_image_png"))
        + _render_motion_deskew(report.get("motion_deskew"), visuals.get("deskew_comparison_png"))
        + _render_dynamic_filter(report.get("dynamic_filter"), visuals.get("dynamic_filter_overlay_png"))
        + _render_m2(report["m2_edge_alignment"], visuals.get("overlay_png"), visuals.get("histogram_png"),
                     visuals.get("colorized_pointcloud_png"), visuals.get("error_heatmap_png"),
                     visuals.get("bev_dual_panel_png"), visuals.get("uncertainty_plot_png"))
        + _render_spatial_analysis(report.get("m2_spatial_analysis"), visuals.get("spatial_analysis_png"))
        + _render_m3(report["m3_holdout_consistency"])
        + _render_m4(report["m4_multiframe_consistency"], visuals.get("trajectory_png"))
        + _render_sequence_gif(visuals.get("sequence_gif"))
        + _render_advanced(report.get("advanced"), visuals.get("sensitivity_png"))
        + _render_metadata(metadata)
        + _render_warning_list(report.get("warnings", []))
    )

    plotly_script_tag = f"<script>{_load_plotly_js()}</script>" if visuals.get("interactive_scene") else ""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Cam-LiDAR Calibration Quality Report</title>
{_FONT_LINKS}
<style>{_CSS}</style>
{plotly_script_tag}
</head>
<body>
  <div class="container">
    <header class="report-header">
      <h1>Cam&ndash;LiDAR Calibration Quality</h1>
      <div class="meta">generated {escape(metadata["generated_at"])} &middot; v{escape(metadata["tool_version"])}</div>
    </header>
    {body}
    <footer>GT-free calibration evaluation &middot; not a substitute for target-based validation</footer>
  </div>
</body>
</html>"""


def write_html_report(report: dict, path: str, visuals: Optional[dict] = None) -> None:
    Path(path).write_text(render_html_report(report, visuals), encoding="utf-8")
