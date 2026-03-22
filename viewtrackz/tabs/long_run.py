"""
viewtrackz.tabs.long_run
~~~~~~~~~~~~~~~~~~~~~~~~
Long Run analysis tab.

Displays two sections:

1. **Run summary** (from LongRunStats)
   - Pacing strategy, cardiac drift, first/second-half pace comparison
   - Thirds breakdown table (pace + HR per third)

2. **Form resilience** (from FormResilienceStats)
   - One scatter plot per available channel (stride length, cadence, leg spring)
   - X axis: distance into run (km)
   - Y axis: drift from early baseline (%)
   - Points coloured by pace zone (green = fast, red = slow)
   - Dashed reference at 0 % (baseline) and −5 % (breakdown threshold)
   - Summary line showing at which km each metric first broke down

Reactivity
----------
The tab holds ``long_run_stats`` and ``form_resilience`` as ``param.Parameter``
objects.  Call ``tab.update(...)`` after analysis completes to push new data and
trigger a re-render.  The ``panel()`` method returns a layout that watches those
parameters and rebuilds whenever they change.
"""

from __future__ import annotations

from typing import Optional

import panel as pn
import param

# Bokeh imports — used only when there is data to plot
from bokeh.plotting import figure
from bokeh.models import Span, ColumnDataSource


# ── Pace-zone colour palette ──────────────────────────────────────────────────
#   Zone 1 = fastest quartile  →  green
#   Zone 2 = second quartile   →  blue
#   Zone 3 = third quartile    →  orange
#   Zone 4 = slowest quartile  →  red
_ZONE_COLORS = {1: "#2ecc71", 2: "#3498db", 3: "#f39c12", 4: "#e74c3c"}
_ZONE_LABELS = {1: "Fast", 2: "Mod-Fast", 3: "Mod-Slow", 4: "Slow"}


class LongRunTab(param.Parameterized):
    """Panel component for the Long Run analysis tab."""

    long_run_stats  = param.Parameter(default=None)
    form_resilience = param.Parameter(default=None)

    def __init__(self, state: dict, **params):
        super().__init__(**params)
        self._state = state

    # ── Public interface ──────────────────────────────────────────────────────

    def update(
        self,
        long_run_stats=None,
        form_resilience=None,
    ) -> None:
        """
        Push new analysis results and trigger a reactive re-render.

        Called by LoadSmoothTab after a long-run analysis completes.
        Both arguments are optional — pass only what has changed.
        """
        self.long_run_stats  = long_run_stats
        self.form_resilience = form_resilience

    def panel(self) -> pn.viewable.Viewable:
        """Return the Panel layout for this tab (reactive)."""
        return pn.Column(
            pn.pane.Markdown("## Long Run", margin=(10, 0, 0, 0)),
            self._content,
            sizing_mode="stretch_width",
        )

    # ── Reactive content ──────────────────────────────────────────────────────

    @param.depends("long_run_stats", "form_resilience")
    def _content(self) -> pn.viewable.Viewable:
        if self.long_run_stats is None:
            return pn.pane.Markdown(
                "_Load and analyse a long run to see results here._",
                margin=(20, 10),
            )

        sections = []

        # ── 1. Run summary ────────────────────────────────────────────────
        sections.append(_build_summary(self.long_run_stats))

        # ── 2. Thirds breakdown table ─────────────────────────────────────
        if self.long_run_stats.thirds:
            sections.append(_build_thirds_table(self.long_run_stats.thirds))

        # ── 3. Form resilience plots ──────────────────────────────────────
        if self.form_resilience is not None and self.form_resilience.windows:
            sections.append(_build_resilience_section(self.form_resilience))
        else:
            sections.append(pn.pane.Markdown(
                "_Form resilience data is not available for this activity._  \n"
                "Requires `stride_length`, `cadence`, or `leg_spring_stiffness` "
                "channels from a Coros or Stryd-equipped device.",
                margin=(10, 0),
            ))

        return pn.Column(*sections, sizing_mode="stretch_width")


# ── Summary card ──────────────────────────────────────────────────────────────

def _fmt_pace(pace_min_km: Optional[float]) -> str:
    """Format a decimal min/km value as ``M:SS /km``."""
    if pace_min_km is None:
        return "—"
    m = int(pace_min_km)
    s = int(round((pace_min_km - m) * 60))
    return f"{m}:{s:02d} /km"


def _build_summary(stats) -> pn.viewable.Viewable:
    """Single-row summary strip with pacing, cardiac drift, and pace splits."""

    strategy_icon = {
        "negative": "🟢 Negative split",
        "even":     "🟡 Even split",
        "positive": "🔴 Positive split",
    }.get(stats.pacing_strategy or "", stats.pacing_strategy or "—")

    if stats.cardiac_drift_pct is not None:
        drift_str = f"{stats.cardiac_drift_pct:+.1f} %"
        drift_note = " (✓ good)" if abs(stats.cardiac_drift_pct) < 5 else ""
        cardiac_str = drift_str + drift_note
    else:
        cardiac_str = "—"

    md = (
        "### Run Summary\n"
        f"| | |\n"
        f"|---|---|\n"
        f"| **Pacing strategy** | {strategy_icon} |\n"
        f"| **First half pace** | {_fmt_pace(stats.first_half_pace_min_km)} |\n"
        f"| **Second half pace** | {_fmt_pace(stats.second_half_pace_min_km)} |\n"
        f"| **Cardiac drift** | {cardiac_str} |\n"
    )
    return pn.pane.Markdown(md, margin=(0, 0, 10, 0))


# ── Thirds table ──────────────────────────────────────────────────────────────

def _build_thirds_table(thirds) -> pn.viewable.Viewable:
    rows = "\n".join(
        f"| {t.label.capitalize()} | {t.pace_str} /km | {t.avg_hr:.0f} bpm |"
        for t in thirds
    )
    md = (
        "### Thirds Breakdown\n"
        "| Third | Avg pace | Avg HR |\n"
        "|---|---|---|\n"
        f"{rows}\n"
    )
    return pn.pane.Markdown(md, margin=(0, 0, 15, 0))


# ── Form resilience section ───────────────────────────────────────────────────

def _build_resilience_section(fr) -> pn.viewable.Viewable:
    """Build the full form resilience section: header + plots."""
    sections: list = []

    # ── Breakdown summary header ───────────────────────────────────────────
    breakdown_str = fr.breakdown_summary()
    md = (
        f"### Form Resilience\n"
        f"Baseline: first {fr.baseline_end_km:.1f} km  ·  "
        f"Window: 500 m rolling  ·  "
        f"Breakdown threshold: −5 %\n\n"
        f"**Form breakdown detected:** {breakdown_str}"
    )
    sections.append(pn.pane.Markdown(md, margin=(0, 0, 5, 0)))

    # ── Pace zone legend ──────────────────────────────────────────────────
    if fr.pace_zone_thresholds:
        p25, p50, p75 = fr.pace_zone_thresholds
        legend_md = (
            "**Pace zones** (relative to this run)  "
            f"🟢 Fast (< {_fmt_pace(p25)})  "
            f"🔵 Mod-Fast ({_fmt_pace(p25)}–{_fmt_pace(p50)})  "
            f"🟠 Mod-Slow ({_fmt_pace(p50)}–{_fmt_pace(p75)})  "
            f"🔴 Slow (> {_fmt_pace(p75)})"
        )
        sections.append(pn.pane.Markdown(legend_md, margin=(0, 0, 10, 0)))

    # ── One plot per available metric ─────────────────────────────────────
    plots = []

    has_stride     = any(w.stride_length_drift_pct is not None for w in fr.windows)
    has_cadence    = any(w.cadence_drift_pct       is not None for w in fr.windows)
    has_leg_spring = any(w.leg_spring_drift_pct    is not None for w in fr.windows)

    if has_stride:
        plots.append(_make_form_plot(
            windows=fr.windows,
            drift_attr="stride_length_drift_pct",
            title="Stride Length Drift",
            y_label="Drift from baseline (%)",
            breakdown_km=fr.stride_breakdown_km,
        ))
    if has_cadence:
        plots.append(_make_form_plot(
            windows=fr.windows,
            drift_attr="cadence_drift_pct",
            title="Cadence Drift",
            y_label="Drift from baseline (%)",
            breakdown_km=fr.cadence_breakdown_km,
        ))
    if has_leg_spring:
        plots.append(_make_form_plot(
            windows=fr.windows,
            drift_attr="leg_spring_drift_pct",
            title="Leg Spring Stiffness Drift",
            y_label="Drift from baseline (%)",
            breakdown_km=fr.leg_spring_breakdown_km,
        ))

    if plots:
        # Lay out up to 2 plots per row
        rows = []
        for i in range(0, len(plots), 2):
            row_plots = plots[i:i + 2]
            rows.append(pn.Row(*[pn.pane.Bokeh(p) for p in row_plots]))
        sections.extend(rows)

    return pn.Column(*sections, sizing_mode="stretch_width")


def _make_form_plot(
    windows,
    drift_attr: str,
    title: str,
    y_label: str,
    breakdown_km: Optional[float],
    breakdown_threshold: float = 5.0,
) -> "figure":
    """
    Return a Bokeh scatter figure showing rolling drift coloured by pace zone.

    Parameters
    ----------
    windows : list[FormWindow]
    drift_attr : str
        Which attribute to read from each FormWindow for the y-axis.
    title : str
        Plot title.
    y_label : str
        Y-axis label.
    breakdown_km : float or None
        If set, a vertical dashed line is drawn at this x position.
    breakdown_threshold : float
        Percentage below which a breakdown is flagged (drawn as a dotted line).
    """
    # Collect data per zone
    zone_data: dict[int, tuple[list, list]] = {z: ([], []) for z in range(1, 5)}
    for w in windows:
        val = getattr(w, drift_attr)
        if val is not None:
            zone_data[w.pace_zone][0].append(w.distance_km)
            zone_data[w.pace_zone][1].append(val)

    # Y range: ±max_drift with headroom, at minimum ±8 %
    all_vals = [v for xs, ys in zone_data.values() for v in ys]
    if all_vals:
        ymax = max(abs(v) for v in all_vals) * 1.2
        ymax = max(ymax, 8.0)
    else:
        ymax = 10.0

    x_max = max((w.distance_km for w in windows), default=1.0) + 0.5

    p = figure(
        title=title,
        x_axis_label="Distance (km)",
        y_axis_label=y_label,
        width=580,
        height=300,
        y_range=(-ymax, ymax),
        x_range=(0, x_max),
        toolbar_location="above",
        tools="pan,wheel_zoom,box_zoom,reset,save",
    )
    p.title.text_font_size = "13px"

    # ── Reference lines ───────────────────────────────────────────────────

    # Baseline at y = 0
    baseline_span = Span(
        location=0, dimension="width",
        line_color="#555555", line_dash="dashed", line_width=1.5
    )
    p.add_layout(baseline_span)

    # Breakdown threshold
    threshold_span = Span(
        location=-breakdown_threshold, dimension="width",
        line_color="#e74c3c", line_dash="dotted", line_width=1.2
    )
    p.add_layout(threshold_span)

    # Breakdown km marker
    if breakdown_km is not None:
        bkm_span = Span(
            location=breakdown_km, dimension="height",
            line_color="#e74c3c", line_dash="dashed", line_width=1.5
        )
        p.add_layout(bkm_span)

    # ── Scatter points per pace zone ──────────────────────────────────────
    for zone in [1, 2, 3, 4]:
        xs, ys = zone_data[zone]
        if not xs:
            continue
        source = ColumnDataSource({"x": xs, "y": ys})
        p.circle(
            x="x", y="y",
            source=source,
            color=_ZONE_COLORS[zone],
            legend_label=_ZONE_LABELS[zone],
            size=7,
            alpha=0.8,
            line_color=None,
        )

    # ── Annotations ───────────────────────────────────────────────────────
    if breakdown_km is not None:
        from bokeh.models import Label
        lbl = Label(
            x=breakdown_km + 0.1,
            y=-ymax * 0.9,
            text=f"↑ {breakdown_km:.1f} km",
            text_font_size="10px",
            text_color="#e74c3c",
        )
        p.add_layout(lbl)

    # ── Legend styling ────────────────────────────────────────────────────
    p.legend.location = "upper right"
    p.legend.label_text_font_size = "10px"
    p.legend.click_policy = "hide"
    p.legend.background_fill_alpha = 0.7

    p.xgrid.grid_line_color = "#eeeeee"
    p.ygrid.grid_line_color = "#eeeeee"

    return p
