"""
visualization/sensitivity_plot.py

STEP 11 -- Calibration Sensitivity Analysis visualization: a horizontal
bar chart of evaluation.perturbation's per-axis sensitivity ranking, the
direct visual form of the spec's own worked example:

    Parameter   Sensitivity
    Yaw         ██████████ HIGH
    Tx          █████████  HIGH
    Pitch       ████       MEDIUM
    Ty          ██         LOW
    Roll        █          LOW
    Tz          █          LOW

Bar length encodes large_delta_effect_px (how much mean_px moved at the
LARGEST configured perturbation for that axis); bar color encodes the
HIGH/MEDIUM/LOW classification, matching the same GOOD/WARNING/BAD-style
color convention used throughout this project's other visuals.
"""

from __future__ import annotations

from io import BytesIO
from typing import Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless: no display server needed, safe for CLI/report generation
import matplotlib.pyplot as plt


_BG = "#0D1117"
_SURFACE = "#161B22"
_TEXT = "#8B98AC"
_GRID = "#2A3244"
_COLORS = {"HIGH": "#F85149", "MEDIUM": "#D29922", "LOW": "#3FB950"}

_AXIS_DISPLAY_NAMES = {
    "roll_deg": "Roll", "pitch_deg": "Pitch", "yaw_deg": "Yaw",
    "tx": "Tx", "ty": "Ty", "tz": "Tz", "timestamp": "Timestamp",
}


def render_sensitivity_png(
    axis_sensitivities: list,
    dpi: int = 130,
) -> Optional[bytes]:
    """
    axis_sensitivities: list of evaluation.perturbation.AxisSensitivity
    (or equivalent objects/dicts with .axis/.classification/
    .large_delta_effect_px attributes) -- typically
    PerturbationResult.axis_sensitivities directly.

    Bars are shown in the SAME order given (evaluate_perturbation_sensitivity
    already sorts HIGH->LOW, largest effect first within each tier), so
    the highest-sensitivity axis reads at the top, matching the spec's
    own example layout. Returns None if the list is empty.
    """
    if not axis_sensitivities:
        return None

    def _get(item, attr):
        return getattr(item, attr) if hasattr(item, attr) else item[attr]

    labels = [_AXIS_DISPLAY_NAMES.get(_get(a, "axis"), _get(a, "axis")) for a in axis_sensitivities]
    classifications = [_get(a, "classification") for a in axis_sensitivities]
    values = [_get(a, "large_delta_effect_px") for a in axis_sensitivities]
    colors = [_COLORS.get(c, _TEXT) for c in classifications]

    # Reverse so the FIRST (highest-sensitivity) item plots at the TOP of
    # a horizontal barh chart (matplotlib's barh draws bottom-to-top).
    labels = labels[::-1]
    classifications = classifications[::-1]
    values = values[::-1]
    colors = colors[::-1]

    fig_h = max(2.0, 0.5 * len(labels) + 1.0)
    fig = plt.figure(figsize=(7.5, fig_h), dpi=dpi)
    try:
        fig.patch.set_facecolor(_BG)
        ax = fig.add_subplot(111)
        ax.set_facecolor(_SURFACE)

        y = np.arange(len(labels))
        bars = ax.barh(y, values, color=colors, height=0.6, edgecolor=_BG, linewidth=0.5)

        max_val = max(values) if values and max(values) > 0 else 1.0
        for yi, (val, cls) in enumerate(zip(values, classifications)):
            ax.text(val + max_val * 0.02, yi, cls, va="center", ha="left", color=_TEXT, fontsize=8)

        ax.set_yticks(y)
        ax.set_yticklabels(labels, color=_TEXT, fontsize=9)
        ax.set_xlabel("Error change at largest configured perturbation (px)", color=_TEXT, fontsize=8)
        ax.set_title("Calibration parameter sensitivity", color=_TEXT, fontsize=10)
        ax.set_xlim(0, max_val * 1.3)
        ax.tick_params(colors=_TEXT, labelsize=8)
        for spine in ax.spines.values():
            spine.set_color(_GRID)
        ax.grid(True, axis="x", color=_GRID, linewidth=0.5, alpha=0.6)

        fig.tight_layout()
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


def render_sensitivity_from_result(perturbation_result, **kwargs) -> Optional[bytes]:
    """Convenience wrapper taking a PerturbationResult directly."""
    return render_sensitivity_png(perturbation_result.axis_sensitivities, **kwargs)
