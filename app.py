"""
ViewTrackz — main dashboard
-----------------------------
Serves the full ViewTrackz UI.  Prototype tabs (Tempo, Intervals, etc.) still
use generated fake data; the Long Run tab uses the real LongRunTab component
from viewtrackz.tabs.long_run and runs the full RunTrackz analysis pipeline
when a .fit file is uploaded and Analyse is clicked.

Run:
    panel serve app.py --show
    panel serve app.py --show --port 5007 --autoreload
"""

from __future__ import annotations

import pathlib
import sys
import tempfile

import numpy as np
import pandas as pd
import panel as pn
from bokeh.plotting import figure
from bokeh.models import Span, BoxAnnotation

pn.extension(sizing_mode="stretch_width", notifications=True)

# ── Wire in the real LongRunTab from the viewtrackz package ──────────────────
_HERE = pathlib.Path(__file__).parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from viewtrackz.tabs.long_run import LongRunTab as _LongRunTabClass  # noqa: E402

_long_run_tab_obj = _LongRunTabClass({})

# ── Palette ───────────────────────────────────────────────────────────────────

ACCENT     = "#E8491E"
HR_COLOR   = "#E74C3C"
PACE_COLOR = "#2980B9"
ZONE_COLS  = ["#2ECC71", "#27AE60", "#F39C12", "#E67E22", "#E74C3C"]

CHANNEL_COLORS = {
    "Heart Rate (bpm)":             "#E74C3C",
    "Speed (m/s)":                  "#2980B9",
    "Cadence (rpm)":                "#27AE60",
    "Altitude (m)":                 "#1ABC9C",
    "Power (W)":                    "#E67E22",
    "Stride Length (m)":            "#9B59B6",
    "Vertical Oscillation (cm)":    "#E91E8C",
    "Stance Time (ms)":             "#00BCD4",
}

CHANNEL_UNITS = {
    "Heart Rate (bpm)":             "bpm",
    "Speed (m/s)":                  "m/s",
    "Cadence (rpm)":                "rpm",
    "Altitude (m)":                 "m",
    "Power (W)":                    "W",
    "Stride Length (m)":            "m",
    "Vertical Oscillation (cm)":    "cm",
    "Stance Time (ms)":             "ms",
}

# ── Fake time-series data (all channels) ─────────────────────────────────────
# ~1 hour run sampled at 1 Hz.  First/last 45 s are warmup/cooldown
# (speed < 2 m/s) so the min_speed filter effect is visible.

np.random.seed(42)
N    = 1800                           # 30 min at 2-second intervals
t_s  = np.arange(N) * 2.0            # elapsed seconds
t_m  = t_s / 60.0                    # elapsed minutes

# Speed: ramp up → steady → ramp down
_speed_base = np.ones(N) * 3.2
_speed_base[:45]  = np.linspace(0.5, 3.2, 45)
_speed_base[-45:] = np.linspace(3.2, 0.5, 45)
FAKE_SPEED = np.clip(_speed_base + np.random.randn(N) * 0.18, 0, 6.0)

FAKE_CHANNELS = {
    "Heart Rate (bpm)": (
        np.clip(
            138 + 22 * np.sin(t_s / 900) + np.cumsum(np.random.randn(N) * 0.12),
            95, 185,
        )
    ),
    "Speed (m/s)":              FAKE_SPEED,
    "Cadence (rpm)":            np.clip(88 + 4 * np.sin(t_s / 600) + np.random.randn(N) * 2.5, 70, 105),
    "Altitude (m)":             50 + 25 * np.sin(t_s / 1200) + np.random.randn(N) * 1.2,
    "Power (W)":                np.clip(230 + 40 * np.sin(t_s / 800) + np.random.randn(N) * 12, 80, 400),
    "Stride Length (m)":        np.clip(1.18 + 0.08 * (FAKE_SPEED - 3.0) + np.random.randn(N) * 0.03, 0.7, 1.8),
    "Vertical Oscillation (cm)":np.clip(8.5  + 1.5  * np.sin(t_s / 700) + np.random.randn(N) * 0.5, 5, 14),
    "Stance Time (ms)":         np.clip(248  - 12   * np.sin(t_s / 600) + np.random.randn(N) * 4,  180, 320),
}

CHANNEL_NAMES = list(FAKE_CHANNELS.keys())

# ── Smoothing (actual SMA / EMA with segment restart at min_speed) ────────────

def apply_smoother(
    channel: str,
    method:  str,
    param:   float,
    min_speed: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return (raw, smoothed) arrays.

    Smoothing is applied only to active segments (speed >= min_speed).
    Below-threshold regions remain NaN in both arrays.
    """
    signal = FAKE_CHANNELS[channel].copy().astype(float)
    speed  = FAKE_SPEED

    is_active = speed >= min_speed

    # Mark inactive samples as NaN in the raw signal too
    raw = np.where(is_active, signal, np.nan)

    if method == "None":
        return raw, raw.copy()

    smoothed = np.full(N, np.nan)

    # Find contiguous active segments and smooth each independently
    padded = np.concatenate([[False], is_active, [False]])
    starts = np.where(np.diff(padded.astype(int)) ==  1)[0]
    ends   = np.where(np.diff(padded.astype(int)) == -1)[0]

    for s, e in zip(starts, ends):
        seg = signal[s:e]
        if method == "SMA":
            w = max(1, int(round(param)))
            smoothed[s:e] = (
                pd.Series(seg)
                .rolling(w, center=True, min_periods=1)
                .mean()
                .values
            )
        elif method == "EMA":
            alpha = float(np.clip(param, 1e-4, 1.0))
            smoothed[s:e] = (
                pd.Series(seg)
                .ewm(alpha=alpha, adjust=False)
                .mean()
                .values
            )

    return raw, smoothed


# ══════════════════════════════════════════════════════════════════════════════
# LOAD & SMOOTH CONTROLS
# ══════════════════════════════════════════════════════════════════════════════

PRESETS: dict[str, tuple[str, float]] = {
    "SMA  window = 5":   ("SMA", 5.0),
    "SMA  window = 10":  ("SMA", 10.0),
    "SMA  window = 20":  ("SMA", 20.0),
    "SMA  window = 30":  ("SMA", 30.0),
    "EMA  α = 0.5":      ("EMA", 0.5),
    "EMA  α = 0.3":      ("EMA", 0.3),
    "EMA  α = 0.1":      ("EMA", 0.1),
    "None (raw)":        ("None", 0.0),
}

# — channel selectors
ch_a_w = pn.widgets.Select(
    name="Channel A", options=CHANNEL_NAMES, value="Heart Rate (bpm)", width=220,
)
ch_b_w = pn.widgets.Select(
    name="Channel B", options=CHANNEL_NAMES, value="Cadence (rpm)", width=220,
)

# — preset dropdown
preset_w = pn.widgets.Select(
    name="Preset", options=list(PRESETS.keys()), value="SMA  window = 10", width=200,
)

# — manual fine-tune controls (populated from preset, can be overridden)
smoother_type_w = pn.widgets.Select(
    name="Smoother",
    options=["SMA", "EMA", "None"],
    value="SMA",
    width=100,
)
# SMA uses an integer window; EMA uses a float alpha — both always visible
sma_window_w = pn.widgets.IntInput(
    name="SMA Window",
    value=10, step=1, start=1, end=500,
    width=110,
)
ema_alpha_w = pn.widgets.FloatInput(
    name="EMA Alpha (0–1)",
    value=0.3, step=0.05, start=0.001, end=1.0,
    width=120,
)
min_speed_w = pn.widgets.FloatInput(
    name="Min Speed (m/s)",
    value=2.0, step=0.1, start=0.0, end=10.0,
    width=130,
)

# Sync preset → manual fields
def _on_preset(event):
    method, param = PRESETS[event.new]
    smoother_type_w.value = method
    if method == "SMA":
        sma_window_w.value = int(param)
    elif method == "EMA":
        ema_alpha_w.value = float(param)

preset_w.param.watch(_on_preset, "value")

# ── Reactive chart builders ───────────────────────────────────────────────────

def _raw_smooth_fig(channel: str, method: str, sma_window: int, ema_alpha: float,
                    min_speed: float, height: int = 330) -> pn.pane.Bokeh:
    param = sma_window if method == "SMA" else ema_alpha
    raw, smoothed = apply_smoother(channel, method, param, min_speed)
    color = CHANNEL_COLORS[channel]
    unit  = CHANNEL_UNITS[channel]

    f = figure(
        title=f"{channel}  —  raw vs smoothed",
        height=height,
        x_axis_label="elapsed (min)",
        y_axis_label=unit,
        toolbar_location="above",
        sizing_mode="stretch_width",
    )
    # shaded region where speed < min_speed
    shade = BoxAnnotation(
        left=t_m[0], right=t_m[44],
        fill_color="#cccccc", fill_alpha=0.25, line_color=None,
    )
    shade2 = BoxAnnotation(
        left=t_m[-45], right=t_m[-1],
        fill_color="#cccccc", fill_alpha=0.25, line_color=None,
    )
    f.add_layout(shade)
    f.add_layout(shade2)

    f.line(t_m, raw,      color="#cccccc",  line_width=1.5, legend_label="Raw",      alpha=0.9)
    f.line(t_m, smoothed, color=color,      line_width=2.5, legend_label="Smoothed", alpha=0.95)
    f.legend.location = "top_left"
    f.legend.click_policy = "hide"
    return pn.pane.Bokeh(f, sizing_mode="stretch_width")


def _residuals_fig(channel: str, method: str, sma_window: int, ema_alpha: float,
                   min_speed: float, height: int = 200) -> pn.pane.Bokeh:
    param = sma_window if method == "SMA" else ema_alpha
    raw, smoothed = apply_smoother(channel, method, param, min_speed)
    residuals = raw - smoothed
    color = CHANNEL_COLORS[channel]
    unit  = CHANNEL_UNITS[channel]

    f = figure(
        title=f"Residuals (raw − smoothed)  —  {channel}",
        height=height,
        x_axis_label="elapsed (min)",
        y_axis_label=unit,
        toolbar_location="above",
        sizing_mode="stretch_width",
    )
    zero = Span(location=0, dimension="width", line_color="#999999",
                line_dash="dashed", line_width=1)
    f.add_layout(zero)

    # Fill positive/negative residuals with the channel colour at low opacity
    pos = np.where(~np.isnan(residuals) & (residuals >= 0), residuals, np.nan)
    neg = np.where(~np.isnan(residuals) & (residuals <  0), residuals, np.nan)
    zeros_ = np.zeros(N)

    f.varea(x=t_m, y1=zeros_, y2=np.nan_to_num(pos), color=color,   alpha=0.35)
    f.varea(x=t_m, y1=zeros_, y2=np.nan_to_num(neg), color="#E74C3C", alpha=0.25)
    f.line(t_m, np.nan_to_num(residuals), color=color, line_width=1.2, alpha=0.7)

    return pn.pane.Bokeh(f, sizing_mode="stretch_width")


# Bind reactive plots to widgets
plot_raw_a   = pn.bind(_raw_smooth_fig, ch_a_w, smoother_type_w, sma_window_w, ema_alpha_w, min_speed_w)
plot_raw_b   = pn.bind(_raw_smooth_fig, ch_b_w, smoother_type_w, sma_window_w, ema_alpha_w, min_speed_w)
plot_resid_a = pn.bind(_residuals_fig,  ch_a_w, smoother_type_w, sma_window_w, ema_alpha_w, min_speed_w)
plot_resid_b = pn.bind(_residuals_fig,  ch_b_w, smoother_type_w, sma_window_w, ema_alpha_w, min_speed_w)

# ── Divider label helper ──────────────────────────────────────────────────────

def section(title: str) -> pn.Column:
    return pn.Column(
        pn.pane.HTML(
            f'<div style="font-size:13px;font-weight:600;color:#555;'
            f'text-transform:uppercase;letter-spacing:0.6px;'
            f'border-bottom:2px solid {ACCENT};padding-bottom:4px;'
            f'margin-top:12px;">{title}</div>'
        ),
        margin=(0, 0, 6, 0),
    )


# ── Load & Smooth tab ─────────────────────────────────────────────────────────

type_sel = pn.widgets.Select(
    name="Activity Type",
    options=["Long Run", "Tempo", "Intervals", "Treadmill", "Normal Run"],
    value="Long Run", width=180,
)
analyse_btn = pn.widgets.Button(name="⚡  Analyse", button_type="success",  width=150)
save_btn    = pn.widgets.Button(name="💾  Save",    button_type="warning",  width=130)


def _on_analyse_click(event):
    """Run the RunTrackz analysis pipeline for the uploaded .fit file."""
    if fit_file_input.value is None or not fit_file_input.filename:
        pn.state.notifications.error(
            "No .fit file uploaded — use the sidebar to upload a file first.",
            duration=5000,
        )
        return

    act_type = type_sel.value  # e.g. "Long Run"

    analyse_btn.loading = True
    analyse_btn.name    = "Analysing…"
    tmp_path = None

    try:
        # ── Save bytes to a temp file ────────────────────────────────────
        suffix = pathlib.Path(fit_file_input.filename).suffix or ".fit"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(fit_file_input.value)
            tmp_path = pathlib.Path(tmp.name)

        # ── Load via FitTrackz adapter ───────────────────────────────────
        from viewtrackz.fittrackz_adapter import load as _fit_load
        from runtrackz import hr_analysis, pace_analysis
        from runtrackz.long_run_analysis import (
            analyze as _lr_analyze,
            analyze_form_resilience as _lr_form,
        )

        run        = _fit_load(tmp_path)
        hr_stats   = hr_analysis.analyze(run)
        pace_stats = pace_analysis.analyze(run)

        if act_type == "Long Run":
            long_run_stats  = _lr_analyze(run, hr_stats=hr_stats, pace_stats=pace_stats)
            form_resilience = _lr_form(run)
            _long_run_tab_obj.update(
                long_run_stats=long_run_stats,
                form_resilience=form_resilience,
            )
            pn.state.notifications.success(
                f"Long run analysis complete — switch to the Long Run tab.",
                duration=5000,
            )
        else:
            pn.state.notifications.info(
                f"Analysis for '{act_type}' is not yet wired up in this prototype.",
                duration=5000,
            )

    except Exception as exc:
        pn.state.notifications.error(
            f"Analysis failed: {exc}",
            duration=10_000,
        )
    finally:
        analyse_btn.loading = False
        analyse_btn.name    = "⚡  Analyse"
        if tmp_path is not None:
            try:
                tmp_path.unlink()
            except Exception:
                pass


analyse_btn.on_click(_on_analyse_click)

tab_smooth = pn.Column(
    # ── Step 1: Channel & smoother controls ────────────────────────────────
    section("1 · Channel Selection & Smoothing"),
    pn.Row(
        # Channel selectors
        pn.Column(ch_a_w, ch_b_w, width=240),

        # Preset + manual type + both param inputs
        pn.Column(
            preset_w,
            pn.Row(
                smoother_type_w,
                sma_window_w,
                ema_alpha_w,
            ),
            width=460,
        ),

        # Min speed + note
        pn.Column(
            min_speed_w,
            pn.pane.HTML(
                '<span style="font-size:11px;color:#888;">'
                "Records below this speed are excluded<br>"
                "and the smoother resets at each gap."
                "</span>"
            ),
            width=200,
        ),
        align="start",
    ),

    # ── Charts ─────────────────────────────────────────────────────────────
    pn.Row(plot_raw_a, plot_raw_b, sizing_mode="stretch_width"),
    pn.Row(plot_resid_a, plot_resid_b, sizing_mode="stretch_width"),

    # ── Step 2: Classify & Analyse ─────────────────────────────────────────
    section("2 · Classify & Analyse"),
    pn.Row(type_sel, pn.Column(pn.pane.Markdown(""), analyse_btn), align="end"),

    # ── Step 3: Save ───────────────────────────────────────────────────────
    section("3 · Save to Database"),
    pn.Row(save_btn),

    sizing_mode="stretch_width",
)


# ══════════════════════════════════════════════════════════════════════════════
# REMAINING TABS (static placeholder content)
# ══════════════════════════════════════════════════════════════════════════════

# ── Chart helpers for other tabs ──────────────────────────────────────────────

def _static_hr(height=280):
    raw, sm = apply_smoother("Heart Rate (bpm)", "SMA", 10, 2.0)
    f = figure(title="Heart Rate", height=height,
               x_axis_label="min", y_axis_label="bpm",
               toolbar_location="above", sizing_mode="stretch_width")
    f.line(t_m, sm, color=HR_COLOR, line_width=2)
    return pn.pane.Bokeh(f, sizing_mode="stretch_width")

def _static_pace(height=280):
    _, spd = apply_smoother("Speed (m/s)", "SMA", 10, 2.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        pace = np.where(spd > 0, 1000 / spd / 60, np.nan)
    f = figure(title="Pace (min/km)", height=height,
               x_axis_label="min", y_axis_label="min/km",
               toolbar_location="above", sizing_mode="stretch_width")
    f.line(t_m, pace, color=PACE_COLOR, line_width=2)
    f.y_range.flipped = True
    return pn.pane.Bokeh(f, sizing_mode="stretch_width")

def _zone_bar(height=240):
    zones = ["Z1", "Z2", "Z3", "Z4", "Z5"]
    times = [8.1, 22.4, 18.6, 9.3, 1.6]
    f = figure(title="HR Zone Distribution", height=height,
               x_range=zones, y_axis_label="min",
               toolbar_location=None, sizing_mode="stretch_width")
    f.vbar(x=zones, top=times, width=0.6, color=ZONE_COLS)
    f.xgrid.grid_line_color = None
    return pn.pane.Bokeh(f, sizing_mode="stretch_width")

def _splits_bar(height=240):
    km  = [str(i+1) for i in range(12)]
    spd = [5.05, 5.00, 5.10, 5.08, 5.12, 5.18, 5.22, 5.25, 5.20, 5.30, 5.35, 5.45]
    f = figure(title="Km Splits", height=height,
               x_range=km, y_axis_label="min/km",
               toolbar_location=None, sizing_mode="stretch_width")
    f.vbar(x=km, top=spd, width=0.6, color=PACE_COLOR)
    f.xgrid.grid_line_color = None
    f.y_range.flipped = True
    return pn.pane.Bokeh(f, sizing_mode="stretch_width")

def _thirds_bar(height=240):
    thirds = ["First", "Middle", "Last"]
    paces  = [5.05, 5.20, 5.45]
    f = figure(title="Pace by Thirds", height=height,
               x_range=thirds, y_axis_label="min/km",
               toolbar_location=None, sizing_mode="stretch_width")
    f.vbar(x=thirds, top=paces, width=0.5, color=PACE_COLOR, alpha=0.8)
    f.xgrid.grid_line_color = None
    f.y_range.flipped = True
    return pn.pane.Bokeh(f, sizing_mode="stretch_width")

def _gap_chart(height=240):
    segs = ["Seg 1 (1%)", "Seg 2 (3%)", "Seg 3 (6%)", "Seg 4 (1%)"]
    gaps = [5.10, 4.85, 4.48, 5.08]
    f = figure(title="Grade-Adjusted Pace by Segment", height=height,
               x_range=segs, y_axis_label="min/km",
               toolbar_location=None, sizing_mode="stretch_width")
    f.vbar(x=segs, top=gaps, width=0.6, color=PACE_COLOR, alpha=0.8)
    f.xgrid.grid_line_color = None
    f.y_range.flipped = True
    return pn.pane.Bokeh(f, sizing_mode="stretch_width")

def _mileage_chart(height=250):
    months   = pd.date_range("2025-10", periods=6, freq="MS")
    labels   = [m.strftime("%b %Y") for m in months]
    mileages = [42, 55, 48, 61, 58, 63]
    f = figure(title="Monthly Mileage (km)", height=height,
               x_range=labels, y_axis_label="km",
               toolbar_location=None, sizing_mode="stretch_width")
    f.vbar(x=labels, top=mileages, width=0.6, color=ACCENT, alpha=0.85)
    f.xgrid.grid_line_color = None
    return pn.pane.Bokeh(f, sizing_mode="stretch_width")

def _trimp_chart(height=250):
    trimps = [280, 340, 310, 410, 390, 430]
    x = list(range(6))
    cumtrimp = list(np.cumsum(trimps))
    f = figure(title="Cumulative TRIMP", height=height,
               y_axis_label="TRIMP",
               toolbar_location=None, sizing_mode="stretch_width")
    f.line(x, cumtrimp, color="#9B59B6", line_width=2)
    f.scatter(x, cumtrimp, color="#9B59B6", size=7)
    f.xaxis.ticker = x
    f.xaxis.major_label_overrides = dict(enumerate(["Oct","Nov","Dec","Jan","Feb","Mar"]))
    return pn.pane.Bokeh(f, sizing_mode="stretch_width")

# ── Stat card ─────────────────────────────────────────────────────────────────

def stat_card(label: str, value: str) -> pn.pane.HTML:
    return pn.pane.HTML(
        f'<div style="background:#f8f9fa;border-radius:8px;padding:14px 18px;'
        f'text-align:center;border:1px solid #dee2e6;min-width:110px;">'
        f'<div style="font-size:22px;font-weight:700;color:#2c3e50;margin-bottom:4px;">{value}</div>'
        f'<div style="font-size:11px;color:#7f8c8d;text-transform:uppercase;letter-spacing:0.5px;">{label}</div>'
        f'</div>',
        sizing_mode="stretch_width",
    )

def stat_row(*cards):
    return pn.Row(*cards, sizing_mode="stretch_width")

# ── Long Run ──────────────────────────────────────────────────────────────────
# Uses the real LongRunTab component.  Upload a .fit file and click Analyse
# to populate it; until then it shows a placeholder message.

tab_long_run = _long_run_tab_obj.panel()

# ── Tempo ─────────────────────────────────────────────────────────────────────

tab_tempo = pn.Column(
    pn.pane.Markdown("## Tempo Run"),
    stat_row(
        stat_card("Avg Pace",         "4:21 min/km"),
        stat_card("Pace CV",          "2.1 %"),
        stat_card("HR Drift",         "3.8 %"),
        stat_card("Time @ Threshold", "78 %"),
        stat_card("HR % of Max",      "86 %"),
    ),
    pn.layout.Divider(),
    pn.Row(_static_pace(300), _static_hr(300)),
    sizing_mode="stretch_width",
)

# ── Intervals ─────────────────────────────────────────────────────────────────

_intervals_df = pd.DataFrame({
    "Rep":      range(1, 7),
    "Pace":     ["3:52", "3:55", "3:50", "3:58", "3:53", "3:56"],
    "Avg HR":   [172, 174, 171, 175, 173, 176],
    "Dist (m)": [1002, 998, 1005, 997, 1001, 1003],
    "HR Drop":  [28, 25, 30, 22, 27, 24],
})

tab_intervals = pn.Column(
    pn.pane.Markdown("## Intervals"),
    stat_row(
        stat_card("Reps",             "6"),
        stat_card("Avg Rep Pace",     "3:54 min/km"),
        stat_card("Pace Consistency", "CV 0.8 %"),
        stat_card("HR Consistency",   "CV 1.2 %"),
    ),
    pn.layout.Divider(),
    pn.Row(
        pn.Column(
            pn.pane.Markdown("**Per-Rep Summary**"),
            pn.widgets.DataFrame(_intervals_df, sizing_mode="stretch_width", height=230),
        ),
        _static_hr(260),
    ),
    sizing_mode="stretch_width",
)

# ── Treadmill ─────────────────────────────────────────────────────────────────

_gradient_df = pd.DataFrame({
    "Start (s)":    [0, 600, 1200, 1800],
    "Gradient (%)": [1.0, 3.0, 6.0, 1.0],
})

tab_treadmill = pn.Column(
    pn.pane.Markdown("## Treadmill"),
    pn.pane.Alert("Enter the gradient schedule before running the analysis.", alert_type="warning"),
    pn.Row(
        pn.Column(
            pn.pane.Markdown("**Gradient Schedule**"),
            pn.widgets.DataFrame(_gradient_df, sizing_mode="fixed", width=290, height=190),
            pn.widgets.Button(name="Run Treadmill Analysis", button_type="primary", width=210),
        ),
        pn.Column(
            stat_row(
                stat_card("Avg GAP",     "4:29 min/km"),
                stat_card("GAP Factor",  "1.08"),
                stat_card("Flat Equiv.", "10.8 km"),
            ),
            _gap_chart(240),
        ),
        sizing_mode="stretch_width",
    ),
    sizing_mode="stretch_width",
)

# ── Normal Run ────────────────────────────────────────────────────────────────

tab_normal = pn.Column(
    pn.pane.Markdown("## Normal Run"),
    stat_row(
        stat_card("Distance", "8.4 km"),
        stat_card("Duration", "48:12"),
        stat_card("Avg HR",   "142 bpm"),
        stat_card("Avg Pace", "5:44 min/km"),
    ),
    pn.layout.Divider(),
    pn.Row(_static_hr(), _static_pace()),
    pn.layout.Divider(),
    pn.Row(_zone_bar(), _splits_bar()),
    sizing_mode="stretch_width",
)

# ── Comparisons ───────────────────────────────────────────────────────────────
# Fake set of stored activities to compare across
_COMP_ACTIVITIES = [
    "2026-03-21  Long Run  12.5 km",
    "2026-03-18  Tempo      9.1 km",
    "2026-03-15  Intervals  8.2 km",
    "2026-03-10  Long Run  14.1 km",
    "2026-03-04  Tempo      9.5 km",
    "2026-02-25  Long Run  13.8 km",
]

# Activity-level fake summary used for the trend scatter
_COMP_SUMMARY_DF = pd.DataFrame({
    "Date":           ["2026-02-25", "2026-03-04", "2026-03-10", "2026-03-18", "2026-03-21"],
    "Type":           ["Long Run",  "Tempo",     "Long Run",   "Tempo",      "Long Run"],
    "Avg HR (bpm)":   [152, 165, 148, 163, 145],
    "Avg Pace":       ["5:31", "4:22", "5:38", "4:19", "5:25"],
    "Stride Len (m)": [1.15, 1.29, 1.14, 1.31, 1.18],
    "TRIMP":          [96, 70, 92, 68, 84],
})

# Fake per-activity time-series overlay (5 activities, same time axis)
np.random.seed(7)
_OVERLAY_T = np.linspace(0, 30, 300)          # 30-min window

def _fake_overlay_hr(seed_offset: int, base: float, drift: float) -> np.ndarray:
    np.random.seed(seed_offset)
    return np.clip(
        base + drift * np.sin(_OVERLAY_T / 10) + np.cumsum(np.random.randn(300) * 0.15),
        100, 190,
    )

_OVERLAY_SERIES = {
    "2026-03-21  Long Run":  _fake_overlay_hr(10, 145, 12),
    "2026-03-10  Long Run":  _fake_overlay_hr(11, 148, 14),
    "2026-02-25  Long Run":  _fake_overlay_hr(12, 152, 16),
    "2026-03-18  Tempo":     _fake_overlay_hr(13, 162,  8),
    "2026-03-04  Tempo":     _fake_overlay_hr(14, 165,  9),
}

_OVERLAY_COLORS = ["#E74C3C", "#2980B9", "#27AE60", "#E67E22", "#9B59B6"]

# ─ Comparison controls ────────────────────────────────────────────────────────

comp_multi_w = pn.widgets.MultiSelect(
    name="Activities to compare",
    options=list(_OVERLAY_SERIES.keys()),
    value=list(_OVERLAY_SERIES.keys())[:3],
    size=6,
    width=260,
)
comp_channel_w = pn.widgets.Select(
    name="Overlay channel",
    options=["Heart Rate (bpm)", "Pace (min/km)", "Stride Length (m)", "Cadence (rpm)"],
    value="Heart Rate (bpm)",
    width=200,
)

def _comp_overlay_fig(selected: list, channel: str) -> pn.pane.Bokeh:
    """Multi-activity overlay line chart for the selected channel."""
    y_label = channel
    f = figure(
        title=f"{channel} — activity overlay",
        height=320,
        x_axis_label="elapsed (min)",
        y_axis_label=y_label,
        toolbar_location="above",
        sizing_mode="stretch_width",
    )
    if not selected:
        return pn.pane.Bokeh(f, sizing_mode="stretch_width")

    for i, act in enumerate(selected):
        color = _OVERLAY_COLORS[i % len(_OVERLAY_COLORS)]
        series = _OVERLAY_SERIES.get(act)
        if series is None:
            continue
        # For non-HR channels apply a simple transform to fake it
        if channel == "Pace (min/km)":
            y = np.clip(16.67 / np.where(series / 60 > 0, series / 60, 1), 4.0, 8.0)
        elif channel == "Stride Length (m)":
            y = np.clip(1.1 + (series - 140) / 300, 0.9, 1.5)
        elif channel == "Cadence (rpm)":
            y = np.clip(85 + (series - 145) / 5, 75, 100)
        else:
            y = series
        f.line(_OVERLAY_T, y, color=color, line_width=2.2, alpha=0.85,
               legend_label=act.split("  ")[0])
    f.legend.location = "top_left"
    f.legend.click_policy = "hide"
    if channel == "Pace (min/km)":
        f.y_range.flipped = True
    return pn.pane.Bokeh(f, sizing_mode="stretch_width")


def _comp_trend_fig(channel: str) -> pn.pane.Bokeh:
    """Scatter showing metric trend over calendar dates."""
    f = figure(
        title=f"{channel} trend over time",
        height=260,
        x_axis_label="date",
        y_axis_label=channel,
        toolbar_location="above",
        sizing_mode="stretch_width",
        x_axis_type="datetime",
    )
    dates = pd.to_datetime(_COMP_SUMMARY_DF["Date"]).astype(np.int64) / 1e6

    col_map = {
        "Heart Rate (bpm)":   "Avg HR (bpm)",
        "Pace (min/km)":      "Avg HR (bpm)",   # use HR as stand-in pace proxy
        "Stride Length (m)":  "Stride Len (m)",
        "Cadence (rpm)":      "Avg HR (bpm)",
    }
    col = col_map.get(channel, "Avg HR (bpm)")
    vals = _COMP_SUMMARY_DF[col].values.astype(float)

    run_mask  = _COMP_SUMMARY_DF["Type"] == "Long Run"
    tempo_mask = _COMP_SUMMARY_DF["Type"] == "Tempo"

    f.scatter(dates[run_mask],   vals[run_mask],   size=10, color="#2980B9",
              legend_label="Long Run",  marker="circle")
    f.scatter(dates[tempo_mask], vals[tempo_mask], size=10, color="#E67E22",
              legend_label="Tempo",     marker="triangle")
    f.legend.location = "top_right"
    return pn.pane.Bokeh(f, sizing_mode="stretch_width")


comp_overlay_plot = pn.bind(_comp_overlay_fig, comp_multi_w, comp_channel_w)
comp_trend_plot   = pn.bind(_comp_trend_fig,   comp_channel_w)

tab_comparisons = pn.Column(
    pn.pane.Markdown("## Comparisons"),
    pn.pane.HTML(
        '<p style="color:#666;font-size:13px;margin-top:0;">'
        "Select saved activities to overlay their channels and spot improvement trends."
        "</p>"
    ),
    pn.Row(
        pn.Column(
            section("Activity Selection"),
            comp_multi_w,
            width=280,
        ),
        pn.Column(
            section("Overlay Channel"),
            comp_channel_w,
            pn.pane.HTML(
                '<p style="font-size:11px;color:#888;margin-top:6px;">'
                "All activities are aligned to elapsed time (min 0 = start of active segment)."
                "</p>"
            ),
            width=240,
        ),
        align="start",
    ),
    pn.layout.Divider(),
    section("Channel Overlay"),
    comp_overlay_plot,
    pn.layout.Divider(),
    pn.Row(
        pn.Column(
            section("Metric Trend"),
            comp_trend_plot,
        ),
        pn.Column(
            section("Summary Table"),
            pn.widgets.DataFrame(
                _COMP_SUMMARY_DF,
                sizing_mode="stretch_width",
                height=230,
            ),
        ),
        sizing_mode="stretch_width",
    ),
    sizing_mode="stretch_width",
)

# ── Race ──────────────────────────────────────────────────────────────────────

_RACE_DATA = {
    "2026-02-08  Stavanger Half (21.1 km)": {
        "time": "1:41:22", "avg_pace": "4:48", "avg_hr": 171,
        "first_half_pace": 4.73, "second_half_pace": 4.83,
        "km_paces": [4.65, 4.60, 4.62, 4.68, 4.70, 4.72, 4.75, 4.78, 4.74,
                     4.76, 4.80, 4.82, 4.85, 4.88, 4.83, 4.86, 4.90, 4.88,
                     4.85, 4.82, 4.75],
        "km_hr":   [162, 164, 165, 167, 168, 169, 170, 171, 171, 172, 173,
                    174, 174, 175, 175, 176, 177, 177, 176, 175, 178],
    },
    "2025-10-12  Oslo 10K": {
        "time": "43:52", "avg_pace": "4:23", "avg_hr": 177,
        "first_half_pace": 4.30, "second_half_pace": 4.16,
        "km_paces": [4.41, 4.35, 4.28, 4.25, 4.22, 4.19, 4.15, 4.12, 4.08, 4.07],
        "km_hr":    [170, 173, 175, 177, 178, 178, 179, 180, 181, 182],
    },
    "2025-06-22  Bergen Half (21.1 km)": {
        "time": "1:44:55", "avg_pace": "4:58", "avg_hr": 169,
        "first_half_pace": 4.88, "second_half_pace": 5.08,
        "km_paces": [4.75, 4.78, 4.80, 4.82, 4.85, 4.88, 4.90, 4.95, 4.95,
                     4.98, 5.00, 5.05, 5.08, 5.10, 5.12, 5.10, 5.15, 5.18,
                     5.15, 5.10, 5.02],
        "km_hr":   [160, 162, 164, 165, 166, 167, 168, 169, 170, 170, 171,
                    172, 172, 173, 173, 174, 174, 175, 175, 174, 177],
    },
}

_RACE_NAMES = list(_RACE_DATA.keys())
_RACE_COLORS = ["#E74C3C", "#2980B9", "#27AE60"]

race_multi_w = pn.widgets.MultiSelect(
    name="Races",
    options=_RACE_NAMES,
    value=_RACE_NAMES[:2],
    size=4,
    width=310,
)


def _race_pace_fig(selected: list) -> pn.pane.Bokeh:
    f = figure(
        title="Pace per km  (↑ = slower)",
        height=300,
        x_axis_label="km",
        y_axis_label="min/km",
        toolbar_location="above",
        sizing_mode="stretch_width",
    )
    f.y_range.flipped = True
    for i, name in enumerate(selected):
        data = _RACE_DATA[name]
        paces = data["km_paces"]
        xs = list(range(1, len(paces) + 1))
        color = _RACE_COLORS[i % len(_RACE_COLORS)]
        label = name.split("  ")[0]          # date only
        f.line(xs, paces, color=color, line_width=2.4, legend_label=label, alpha=0.9)
        f.scatter(xs, paces, color=color, size=6, alpha=0.85)
        # Midpoint marker
        mid = len(paces) // 2
        half_span = Span(location=mid + 0.5, dimension="height",
                         line_color=color, line_dash="dashed", line_width=1.2, line_alpha=0.5)
        f.add_layout(half_span)
    f.legend.location = "top_left"
    f.legend.click_policy = "hide"
    return pn.pane.Bokeh(f, sizing_mode="stretch_width")


def _race_hr_fig(selected: list) -> pn.pane.Bokeh:
    f = figure(
        title="HR profile per km",
        height=300,
        x_axis_label="km",
        y_axis_label="bpm",
        toolbar_location="above",
        sizing_mode="stretch_width",
    )
    for i, name in enumerate(selected):
        data = _RACE_DATA[name]
        hrs = data["km_hr"]
        xs  = list(range(1, len(hrs) + 1))
        color = _RACE_COLORS[i % len(_RACE_COLORS)]
        label = name.split("  ")[0]
        f.line(xs, hrs, color=color, line_width=2.4, legend_label=label, alpha=0.9)
        f.scatter(xs, hrs, color=color, size=6, alpha=0.85)
    f.legend.location = "top_left"
    f.legend.click_policy = "hide"
    return pn.pane.Bokeh(f, sizing_mode="stretch_width")


def _race_split_fig(selected: list) -> pn.pane.Bokeh:
    """Grouped bar: first vs second half pace for each race."""
    if not selected:
        return pn.pane.Bokeh(
            figure(height=260, sizing_mode="stretch_width"), sizing_mode="stretch_width"
        )
    labels  = [n.split("  ")[0] for n in selected]  # date labels
    first_h = [_RACE_DATA[n]["first_half_pace"]  for n in selected]
    second_h= [_RACE_DATA[n]["second_half_pace"] for n in selected]

    from bokeh.transform import dodge
    from bokeh.models import ColumnDataSource, FactorRange

    src = ColumnDataSource({"label": labels, "first": first_h, "second": second_h})
    f = figure(
        title="Pacing split: first vs second half",
        height=260,
        x_range=FactorRange(*labels),
        y_axis_label="min/km",
        toolbar_location=None,
        sizing_mode="stretch_width",
    )
    f.y_range.flipped = True
    f.vbar(x=dodge("label", -0.2, range=f.x_range), top="first",  width=0.35,
           source=src, color="#2980B9", alpha=0.85, legend_label="First half")
    f.vbar(x=dodge("label",  0.2, range=f.x_range), top="second", width=0.35,
           source=src, color="#E74C3C", alpha=0.85, legend_label="Second half")
    f.xgrid.grid_line_color = None
    f.legend.location = "top_right"
    return pn.pane.Bokeh(f, sizing_mode="stretch_width")


def _race_summary_df(selected: list) -> pd.DataFrame:
    rows = []
    for name in selected:
        d = _RACE_DATA[name]
        rows.append({
            "Race":         name.split("  ", 1)[1],
            "Date":         name.split("  ")[0],
            "Time":         d["time"],
            "Avg Pace":     d["avg_pace"],
            "Avg HR":       d["avg_hr"],
            "1st ½ pace":   f"{d['first_half_pace']:.2f}",
            "2nd ½ pace":   f"{d['second_half_pace']:.2f}",
            "Split diff":   f"+{d['second_half_pace'] - d['first_half_pace']:.2f}",
        })
    return pd.DataFrame(rows) if rows else pd.DataFrame()


race_pace_plot  = pn.bind(_race_pace_fig,  race_multi_w)
race_hr_plot    = pn.bind(_race_hr_fig,    race_multi_w)
race_split_plot = pn.bind(_race_split_fig, race_multi_w)


def _race_summary_widget(selected):
    df = _race_summary_df(selected)
    if df.empty:
        return pn.pane.Markdown("_No races selected._")
    return pn.widgets.DataFrame(df, sizing_mode="stretch_width", height=160)


race_summary_widget = pn.bind(_race_summary_widget, race_multi_w)

tab_race = pn.Column(
    pn.pane.Markdown("## Race Analysis"),
    pn.pane.HTML(
        '<p style="color:#666;font-size:13px;margin-top:0;">'
        "Compare pacing strategy, HR trajectory, and positive/negative splits across races."
        "</p>"
    ),
    pn.Row(
        pn.Column(
            section("Select Races"),
            race_multi_w,
            width=330,
        ),
        align="start",
    ),
    pn.layout.Divider(),
    section("Race Summary"),
    race_summary_widget,
    pn.layout.Divider(),
    pn.Row(race_pace_plot, race_hr_plot, sizing_mode="stretch_width"),
    pn.layout.Divider(),
    race_split_plot,
    sizing_mode="stretch_width",
)

# ── Aggregates ────────────────────────────────────────────────────────────────

_recent_df = pd.DataFrame({
    "Date":      ["2026-03-21", "2026-03-18", "2026-03-15", "2026-03-12", "2026-03-10"],
    "Type":      ["Long Run",   "Tempo",      "Intervals",  "Normal",     "Long Run"],
    "Dist (km)": [12.5,         9.1,          8.2,          7.8,          14.1],
    "Avg HR":    [148,          162,          171,          141,          145],
    "TRIMP":     [84,           72,           68,           54,           96],
})

tab_aggregates = pn.Column(
    pn.pane.Markdown("## Aggregates"),
    stat_row(
        stat_card("Total Distance", "327 km"),
        stat_card("Activities",     "42"),
        stat_card("Total TRIMP",    "2 360"),
        stat_card("Avg Weekly",     "52 km"),
    ),
    pn.layout.Divider(),
    pn.Row(_mileage_chart(), _trimp_chart()),
    pn.layout.Divider(),
    pn.pane.Markdown("**Recent Activities**"),
    pn.widgets.DataFrame(_recent_df, sizing_mode="stretch_width", height=210),
    sizing_mode="stretch_width",
)

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR  (file upload only)
# ══════════════════════════════════════════════════════════════════════════════

fit_file_input = pn.widgets.FileInput(accept=".fit", sizing_mode="stretch_width")

sidebar = pn.Column(
    pn.pane.Markdown("### Upload Activity"),
    fit_file_input,
    pn.pane.HTML(
        '<p style="font-size:11px;color:#999;margin-top:6px;">'
        "Upload a .fit file, then select<br>"
        "<em>Long Run</em> and click ⚡ Analyse."
        "</p>"
    ),
)

# ══════════════════════════════════════════════════════════════════════════════
# TEMPLATE
# ══════════════════════════════════════════════════════════════════════════════

tabs = pn.Tabs(
    ("Smoothing",      tab_smooth),
    ("Long Run",       tab_long_run),
    ("Tempo",          tab_tempo),
    ("Intervals",      tab_intervals),
    ("Treadmill",      tab_treadmill),
    ("Normal Run",     tab_normal),
    ("Comparisons",    tab_comparisons),
    ("Race",           tab_race),
    ("Aggregates",     tab_aggregates),
    sizing_mode="stretch_width",
)

template = pn.template.FastListTemplate(
    title="ViewTrackz",
    sidebar=[sidebar],
    main=[tabs],
    accent=ACCENT,
    sidebar_width=240,
    collapsed_sidebar=False,
)

template.servable()
