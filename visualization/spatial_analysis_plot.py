"""
visualization/spatial_analysis_plot.py

STEP 9 -- Depth/Spatial Error Analysis visualization: bar charts of mean
error (with std as an error bar, and P95 as a marker) per depth bin and
per camera region (evaluation.spatial_analysis.SpatialAnalysisResult).
This is the direct visual form of the spec's own worked example --

    0-10m      0.8 px
    10-20m     1.0 px
    20-30m     1.8 px
    30-50m     3.9 px

-- made into a chart instead of a table, with a bar's height immediately
showing whether error grows with depth (STEP9's headline diagnosis) or
concentrates on one side of the frame (the horizontal/region charts).
"""

from __future__ import annotations

from io import BytesIO
from typing import Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless: no display server needed, safe for CLI/report generation
import matplotlib.pyplot as plt

from evaluation.spatial_analysis import SpatialAnalysisResult, DEPTH_BIN_LABELS, HORIZONTAL_REGIONS, VERTICAL_REGIONS


_BG = "#0D1117"
_SURFACE = "#161B22"
_TEXT = "#8B98AC"
_GRID = "#2A3244"
_BAR_COLOR = "#58A6FF"
_P95_COLOR = "#F0883E"
_FAILURE_COLOR = "#F85149"


def _bar_panel(ax, bin_stats_dict: dict, labels: list, title: str):
    means = [bin_stats_dict[l].mean_px for l in labels]
    stds = [bin_stats_dict[l].std_px for l in labels]
    p95s = [bin_stats_dict[l].p95_px for l in labels]
    valid = [bin_stats_dict[l].valid_count for l in labels]
    failed = [bin_stats_dict[l].failure_count for l in labels]

    x = np.arange(len(labels))
    means_plot = [m if np.isfinite(m) else 0.0 for m in means]
    stds_plot = [s if np.isfinite(s) else 0.0 for s in stds]

    ax.bar(x, means_plot, yerr=stds_plot, color=_BAR_COLOR, capsize=3, width=0.55,
           edgecolor=_BG, linewidth=0.5, zorder=2, label="mean \u00b1 std")
    p95_x = [xi for xi, p in zip(x, p95s) if np.isfinite(p)]
    p95_y = [p for p in p95s if np.isfinite(p)]
    if p95_y:
        ax.scatter(p95_x, p95_y, color=_P95_COLOR, marker="D", s=22, zorder=3, label="P95")

    top_of_data = max(
        [m + s for m, s in zip(means_plot, stds_plot)] + p95_y + [0.1],
    )
    for xi, (v, f) in enumerate(zip(valid, failed)):
        total = v + f
        label = f"n={total}" + (f"\n({f} failed)" if f > 0 else "")
        y_pos = max(means_plot[xi] + stds_plot[xi], p95s[xi] if np.isfinite(p95s[xi]) else 0.0)
        ax.text(xi, y_pos + top_of_data * 0.08, label, ha="center", va="bottom", color=_TEXT, fontsize=6.5)

    ax.set_ylim(0, top_of_data * 1.35)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, color=_TEXT, fontsize=8)
    ax.set_title(title, color=_TEXT, fontsize=9, pad=10)
    ax.tick_params(colors=_TEXT, labelsize=7)
    for spine in ax.spines.values():
        spine.set_color(_GRID)
    ax.set_facecolor(_SURFACE)
    ax.grid(True, axis="y", color=_GRID, linewidth=0.5, alpha=0.6)


def render_spatial_analysis_png(
    result: SpatialAnalysisResult,
    dpi: int = 130,
) -> Optional[bytes]:
    """
    Three-panel bar chart: depth bins, horizontal regions (LEFT/CENTER/
    RIGHT), vertical regions (TOP/CENTER/BOTTOM). Each bar shows mean
    error with a std error bar and a P95 marker; bins/regions with zero
    points are shown as an empty (zero-height) bar with "n=0" rather than
    omitted, so a reader can see which bins had no data at all versus
    which had data and a small error.

    Returns None if every bin/region across all three panels is empty
    (nothing at all to show).
    """
    total_points = sum(
        s.valid_count + s.failure_count
        for group in (result.depth_bins, result.horizontal_regions, result.vertical_regions)
        for s in group.values()
    )
    if total_points == 0:
        return None

    fig = plt.figure(figsize=(11.5, 3.6), dpi=dpi)
    try:
        fig.patch.set_facecolor(_BG)
        ax1 = fig.add_subplot(1, 3, 1)
        ax2 = fig.add_subplot(1, 3, 2)
        ax3 = fig.add_subplot(1, 3, 3)

        _bar_panel(ax1, result.depth_bins, DEPTH_BIN_LABELS, "Error by depth")
        _bar_panel(ax2, result.horizontal_regions, HORIZONTAL_REGIONS, "Error by horizontal region")
        _bar_panel(ax3, result.vertical_regions, VERTICAL_REGIONS, "Error by vertical region")

        ax1.set_ylabel("Error (px)", color=_TEXT, fontsize=8)
        handles, legend_labels = ax1.get_legend_handles_labels()
        if handles:
            fig.legend(handles, legend_labels, loc="upper center", ncol=2, frameon=False,
                       labelcolor=_TEXT, fontsize=7, bbox_to_anchor=(0.5, 1.06))

        trend = result.depth_trend
        if trend is not None:
            trend_label = {
                "increases_with_depth": "error increases with depth",
                "decreases_with_depth": "error decreases with depth",
                "stable": "no clear depth trend",
            }[trend]
            fig.text(0.01, 0.01, f"Depth trend: {trend_label}", color=_TEXT, fontsize=7, ha="left", va="bottom")

        fig.tight_layout(rect=(0, 0.04, 1, 0.94))
        buf = BytesIO()
        fig.savefig(buf, format="png", facecolor=fig.get_facecolor())
        return buf.getvalue()
    except Exception:
        # Consistent with the other visualization modules: a broken/
        # partial matplotlib install shouldn't crash the caller over one
        # optional diagnostic image.
        return None
    finally:
        plt.close(fig)


def render_spatial_analysis_from_result(edge_alignment_result, image_width: int, image_height: int, **kwargs) -> Optional[bytes]:
    """
    Convenience wrapper: runs evaluation.spatial_analysis.
    analyze_depth_and_spatial_from_result and renders the result in one
    call, mirroring the other visualization modules' *_from_result
    helpers.
    """
    from evaluation.spatial_analysis import analyze_depth_and_spatial_from_result
    analysis = analyze_depth_and_spatial_from_result(edge_alignment_result, image_width, image_height)
    if analysis is None:
        return None
    return render_spatial_analysis_png(analysis, **kwargs)
