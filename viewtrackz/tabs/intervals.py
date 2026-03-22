"""
viewtrackz.tabs.intervals
~~~~~~~~~~~~~~~~~~~~~~~~~
Intervals (workout) analysis tab.

Workflow
--------
1. User enters the number of intervals (IntInput).
2. A Tabulator is pre-populated with *n* editable rows:
       Rep | Start (s) | Stop (s) | Rest-stop (s)
3. User fills/edits the interval boundaries (seconds from run start).
4. "Apply" button computes a 2×2 grid of Bokeh plots:

   ┌──────────────────────────────┬──────────────────────────────┐
   │ 1. Avg HR & Pace ± std       │ 2. HR recovery in rest       │
   │    (dual y-axis, vs time     │    periods (per-rest colour)  │
   │     into interval)           │                              │
   ├──────────────────────────────┼──────────────────────────────┤
   │ 3. Running dynamics vs rep   │ 4. Time in HR zones          │
   │    (normalised to rep 1,     │    (bar chart, workout        │
   │     ± std errorbars)         │     portion of the run)      │
   └──────────────────────────────┴──────────────────────────────┘

HR zones are loaded from  <repo_root>/config/hr_zones.toml.
When no real RunData is attached the tab renders with clearly-labelled
synthetic workout data so the layout is visible immediately.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import panel as pn
import param
from bokeh.models import (
    ColumnDataSource,
    HoverTool,
    LinearAxis,
    Range1d,
    Span,
)
from bokeh.plotting import figure

# ── Paths ─────────────────────────────────────────────────────────────────────

_HERE      = Path(__file__).parent
_TOML_PATH = _HERE.parent.parent / "config" / "hr_zones.toml"

# ── Colour palette ────────────────────────────────────────────────────────────

_HR_COLOR   = "#E74C3C"
_PACE_COLOR = "#2980B9"

# Up to 8 rest-period colours (cycled if more reps than colours)
_REST_COLORS = [
    "#3498DB", "#9B59B6", "#1ABC9C", "#E67E22",
    "#E74C3C", "#2ECC71", "#F39C12", "#E91E8C",
]

# Running-dynamics metric colours and display names.
# Each entry is (candidate_column_names, colour, label).
# Real data uses suffixed names; demo data uses the bare names.
_DYN_META = [
    (("stride_length_m",      "stride_length"),        "#9B59B6", "Stride Length"),
    (("cadence",),                                      "#27AE60", "Cadence"),
    (("leg_spring_stiffness",),                         "#E67E22", "Spring Stiffness"),
]

# ── Default HR zones (fallback when TOML is absent / unreadable) ──────────────

_DEFAULT_MAX_HR = 185
_DEFAULT_ZONES: list[dict] = [
    {"name": "Z1 Recovery",  "min_pct":  0, "max_pct": 60, "color": "#2ECC71"},
    {"name": "Z2 Aerobic",   "min_pct": 60, "max_pct": 70, "color": "#27AE60"},
    {"name": "Z3 Tempo",     "min_pct": 70, "max_pct": 80, "color": "#F39C12"},
    {"name": "Z4 Threshold", "min_pct": 80, "max_pct": 90, "color": "#E67E22"},
    {"name": "Z5 Max",       "min_pct": 90, "max_pct": 100,"color": "#E74C3C"},
]


# ═══════════════════════════════════════════════════════════════════════════════
# Helper utilities
# ═══════════════════════════════════════════════════════════════════════════════

def _load_hr_zones() -> tuple[float, list[dict]]:
    """Return ``(max_hr, zones_list)`` from TOML or from built-in defaults."""
    if _TOML_PATH.exists():
        try:
            try:
                import tomllib                          # Python ≥ 3.11
            except ImportError:
                import tomli as tomllib                 # type: ignore[no-redef]
            data   = tomllib.loads(_TOML_PATH.read_text(encoding="utf-8"))
            max_hr = float(data.get("max_hr", _DEFAULT_MAX_HR))
            zones  = data.get("zones", _DEFAULT_ZONES)
            return max_hr, zones
        except Exception:
            pass
    return _DEFAULT_MAX_HR, _DEFAULT_ZONES


def _get_elapsed(df: pd.DataFrame) -> np.ndarray:
    """
    Return the elapsed-time column as a float array of seconds.

    Tries common column names in order; converts datetime columns to seconds
    relative to the first sample.  Falls back to integer row index.
    """
    for col in ("elapsed_s", "t_s", "time_s", "time", "timestamp"):
        if col in df.columns:
            vals = df[col].values
            if pd.api.types.is_datetime64_any_dtype(vals):
                vals = (vals - vals[0]).astype("timedelta64[s]").astype(float)
            return vals.astype(float)
    return np.arange(len(df), dtype=float)


def _get_col(df: pd.DataFrame, *names: str) -> np.ndarray:
    """
    Return the first matching column as a float array.

    Accepts multiple candidate names so callers handle both real-data column
    names (e.g. ``speed_ms``, ``stride_length_m``, ``pace_min_km``) and the
    shorter names used by the synthetic demo data (e.g. ``speed``,
    ``stride_length``).  Returns an all-NaN array when none are found.
    """
    for name in names:
        if name in df.columns:
            return df[name].values.astype(float)
    return np.full(len(df), np.nan)


def _assign_hr_zones(hr: np.ndarray, max_hr: float, zones: list[dict]) -> np.ndarray:
    """Return 0-based zone index for each sample in *hr*."""
    idx = np.zeros(len(hr), dtype=int)
    for i, z in enumerate(zones):
        lo = z["min_pct"] / 100.0 * max_hr
        hi = z["max_pct"] / 100.0 * max_hr
        idx[(hr >= lo) & (hr < hi)] = i
    # anything ≥ last zone's ceiling → last zone
    last_hi = zones[-1]["max_pct"] / 100.0 * max_hr
    idx[hr >= last_hi] = len(zones) - 1
    return idx


# ── Fake workout data (used when no RunData is loaded) ────────────────────────

def _make_fake_df(n_intervals: int = 6) -> pd.DataFrame:
    """
    Simulate a workout run DataFrame.

    Layout:  5 min warm-up  +  n × (3 min hard + 2 min rest)  +  5 min cool-down
    Sample rate: 1 Hz.
    """
    rng         = np.random.default_rng(42)
    warmup_s    = 300
    interval_s  = 180
    rest_s      = 120
    cooldown_s  = 300
    n           = warmup_s + n_intervals * (interval_s + rest_s) + cooldown_s

    elapsed = np.arange(n, dtype=float)
    speed   = np.zeros(n)
    hr      = np.zeros(n)
    cadence = np.zeros(n)
    stride  = np.zeros(n)
    spring  = np.zeros(n)

    # Warm-up
    speed[:warmup_s]   = rng.normal(2.8, 0.10, warmup_s)
    hr[:warmup_s]      = rng.normal(130, 3.0, warmup_s)
    cadence[:warmup_s] = rng.normal(84,  2.0, warmup_s)
    stride[:warmup_s]  = rng.normal(1.10, 0.02, warmup_s)
    spring[:warmup_s]  = rng.normal(10.0, 0.30, warmup_s)

    t = warmup_s
    for rep in range(n_intervals):
        fatigue = 1.0 - rep * 0.015          # 1.5 % per rep
        # Hard interval
        speed[t:t+interval_s]   = rng.normal(4.5 * fatigue, 0.15, interval_s)
        hr[t:t+interval_s]      = rng.normal(170 + rep * 0.5, 4.0, interval_s)
        cadence[t:t+interval_s] = rng.normal(92 * fatigue, 2.0, interval_s)
        stride[t:t+interval_s]  = rng.normal(1.35 * fatigue, 0.03, interval_s)
        spring[t:t+interval_s]  = rng.normal(12.5 * fatigue, 0.40, interval_s)
        t += interval_s
        # Rest / recovery
        speed[t:t+rest_s]  = rng.normal(1.5, 0.20, rest_s)
        hr_start           = hr[t - 1]
        hr_drop            = np.linspace(hr_start, hr_start - 38, rest_s)
        hr[t:t+rest_s]     = hr_drop + rng.normal(0, 2.0, rest_s)
        cadence[t:t+rest_s]= rng.normal(76, 3.0, rest_s)
        stride[t:t+rest_s] = rng.normal(0.95, 0.04, rest_s)
        spring[t:t+rest_s] = rng.normal(8.5,  0.50, rest_s)
        t += rest_s

    # Cool-down
    speed[t:]   = rng.normal(2.5, 0.10, n - t)
    hr[t:]      = rng.normal(128, 5.0, n - t)
    cadence[t:] = rng.normal(82,  2.0, n - t)
    stride[t:]  = rng.normal(1.08, 0.02, n - t)
    spring[t:]  = rng.normal(9.5,  0.30, n - t)

    return pd.DataFrame({
        "elapsed_s":            elapsed,
        "heart_rate":           np.clip(hr,      80, 200),
        "speed":                np.clip(speed,  0.5,   7),
        "cadence":              np.clip(cadence, 60, 110),
        "stride_length":        np.clip(stride,  0.7, 1.9),
        "leg_spring_stiffness": np.clip(spring,  5.0,  20),
    })


def _default_interval_table(
    n:           int,
    warmup_s:    int = 300,
    interval_s:  int = 180,
    rest_s:      int = 120,
) -> pd.DataFrame:
    """Return a pre-filled interval-definition table (times in minutes)."""
    rows = []
    t = warmup_s
    for rep in range(1, n + 1):
        rows.append({
            "Rep":             rep,
            "Start (min)":     round(t                          / 60.0, 2),
            "Stop (min)":      round((t + interval_s)           / 60.0, 2),
            "Rest-stop (min)": round((t + interval_s + rest_s)  / 60.0, 2),
        })
        t += interval_s + rest_s
    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════════════
# Plot builders
# ═══════════════════════════════════════════════════════════════════════════════

_FIG_W = 680
_FIG_H = 390
_TOOLS = "pan,wheel_zoom,box_zoom,reset,save"


def _table_height(n: int) -> int:
    """Pixel height for the interval-definition Tabulator."""
    return max(240, min(420, 75 + n * 32))


def _pace_height(n: int) -> int:
    """Pixel height for the pace-overview plot (taller than the table)."""
    return max(420, min(620, 180 + n * 38))


def _build_avg_plot(df: pd.DataFrame, table_df: pd.DataFrame) -> figure:
    """
    Plot 1 — Dual y-axis: average HR (left) and average pace (right) across all
    interval reps, plotted vs time into the interval.  ±1 std shown as shaded band.
    """
    elapsed = _get_elapsed(df)
    hr      = _get_col(df, "heart_rate")

    # Use pace_min_km directly if available (real data from FitTrackz/RunTrackz),
    # otherwise derive from speed_ms / speed (demo data).
    pace = _get_col(df, "pace_min_km")
    if np.all(np.isnan(pace)):
        speed = _get_col(df, "speed_ms", "speed")
        speed = np.where(speed > 0.3, speed, np.nan)
        pace  = np.where(~np.isnan(speed), 1000.0 / (speed * 60.0), np.nan)

    reps_hr, reps_pace = [], []
    max_dur = 0.0
    for _, row in table_df.iterrows():
        s, e = float(row["Start (min)"]) * 60.0, float(row["Stop (min)"]) * 60.0
        if e <= s:
            continue
        mask = (elapsed >= s) & (elapsed <= e)
        if mask.sum() < 2:
            continue
        t_rep = elapsed[mask] - s
        reps_hr.append((t_rep,  hr[mask]))
        reps_pace.append((t_rep, pace[mask]))
        max_dur = max(max_dur, t_rep[-1])

    if not reps_hr:
        p = figure(title="Avg HR & Pace (no interval data)", width=_FIG_W, height=_FIG_H)
        p.text(x=[0], y=[0], text=["No valid interval rows — check Start/Stop times."],
               text_color="#888888")
        return p

    t_grid = np.arange(0.0, max_dur + 1.0, 1.0)

    def _interp_rep(reps):
        out = []
        for t_rep, y_rep in reps:
            # Remove NaN before interp
            valid = ~np.isnan(y_rep)
            if valid.sum() < 2:
                out.append(np.full(len(t_grid), np.nan))
            else:
                out.append(np.interp(t_grid, t_rep[valid], y_rep[valid]))
        return np.array(out)

    hr_mat   = _interp_rep(reps_hr)
    pace_mat = _interp_rep(reps_pace)

    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        hr_mean   = np.nanmean(hr_mat,   axis=0)
        hr_std    = np.nanstd (hr_mat,   axis=0)
        pace_mean = np.nanmean(pace_mat, axis=0)
        pace_std  = np.nanstd (pace_mat, axis=0)

    # NaN std means only one rep contributed at that time point — treat as 0 spread
    hr_std   = np.where(np.isnan(hr_std),   0.0, hr_std)
    pace_std = np.where(np.isnan(pace_std), 0.0, pace_std)

    has_pace = bool(np.any(~np.isnan(pace_mean)))
    if not has_pace:
        print(
            f"[intervals] pace still all-NaN after column fallbacks. "
            f"df columns: {list(df.columns)!r}"
        )

    # ── Y-axis ranges ────────────────────────────────────────────────────────
    hr_lo  = max(40,  float(np.nanmin(hr_mean - hr_std)) - 5)
    hr_hi  =          float(np.nanmax(hr_mean + hr_std)) + 5

    t_min = t_grid / 60.0   # display in minutes

    p = figure(
        title="Avg HR & Pace per interval rep  (mean ± 1 std)",
        width=_FIG_W, height=_FIG_H,
        x_axis_label="Time into interval (min)",
        y_axis_label="Heart Rate (bpm)",
        y_range=(hr_lo, hr_hi),
        toolbar_location="above",
        tools=_TOOLS,
    )
    p.title.text_font_size = "13px"

    # ── HR varea + line ──────────────────────────────────────────────────────
    # Note: varea is used instead of Band because Band ignores y_range_name
    # on extra axes — varea is a proper glyph and respects it correctly.
    hr_src = ColumnDataSource(dict(
        t=t_min, mean=hr_mean,
        upper=hr_mean + hr_std, lower=hr_mean - hr_std,
    ))
    p.varea("t", "lower", "upper", source=hr_src,
            fill_color=_HR_COLOR, fill_alpha=0.20)
    p.line("t", "mean", source=hr_src, color=_HR_COLOR,
           line_width=2.5, legend_label="HR")
    p.add_tools(HoverTool(
        tooltips=[("Time", "@t{0.0} min"), ("HR", "@mean{0} ± @{lower}{0} bpm")],
        mode="vline", renderers=[p.renderers[-1]],
    ))

    # ── Pace varea + line (only if speed data is available) ──────────────────
    if has_pace:
        p_lo = float(np.nanmin(pace_mean - pace_std)) - 0.15
        p_hi = float(np.nanmax(pace_mean + pace_std)) + 0.15
        # Inverted range: lower min/km value = faster = higher on plot
        p.extra_y_ranges = {"pace": Range1d(start=p_hi, end=p_lo)}
        p.add_layout(LinearAxis(y_range_name="pace", axis_label="Pace (min/km)"), "right")
        pace_src = ColumnDataSource(dict(
            t=t_min, mean=pace_mean,
            upper=pace_mean + pace_std, lower=pace_mean - pace_std,
        ))
        p.varea("t", "lower", "upper", source=pace_src,
                fill_color=_PACE_COLOR, fill_alpha=0.20, y_range_name="pace")
        p.line("t", "mean", source=pace_src, color=_PACE_COLOR,
               line_width=2.5, y_range_name="pace", legend_label="Pace",
               line_dash="dashed")
    else:
        from bokeh.models import Title
        p.add_layout(Title(
            text="⚠ pace unavailable — check server console for column info",
            text_font_size="11px", text_color="#F39C12",
        ), "above")

    p.legend.location     = "top_right"
    p.legend.label_text_font_size = "10px"
    p.legend.click_policy = "hide"
    p.legend.background_fill_alpha = 0.6
    p.xgrid.grid_line_color = "#444444"
    p.ygrid.grid_line_color = "#444444"
    return p


def _build_rest_hr_plot(df: pd.DataFrame, table_df: pd.DataFrame) -> figure:
    """
    Plot 2 — HR evolution during each rest / recovery period.
    Each rest gets a unique colour; time axis is offset from rest start.
    """
    elapsed = _get_elapsed(df)
    hr      = _get_col(df, "heart_rate")

    p = figure(
        title="HR Recovery in Rest Periods",
        width=_FIG_W, height=_FIG_H,
        x_axis_label="Time into rest (min)",
        y_axis_label="Heart Rate (bpm)",
        toolbar_location="above",
        tools=_TOOLS,
    )
    p.title.text_font_size = "13px"

    any_data = False
    for i, (_, row) in enumerate(table_df.iterrows()):
        rest_start = float(row["Stop (min)"])      * 60.0
        rest_end   = float(row["Rest-stop (min)"]) * 60.0
        if rest_end <= rest_start:
            continue
        mask = (elapsed >= rest_start) & (elapsed <= rest_end)
        if mask.sum() < 2:
            continue
        t_rel = (elapsed[mask] - rest_start) / 60.0   # minutes
        color = _REST_COLORS[i % len(_REST_COLORS)]
        label = f"Rest {i + 1}"
        src   = ColumnDataSource(dict(t=t_rel, hr=hr[mask]))
        p.line("t", "hr", source=src, color=color,
               line_width=2, alpha=0.85, legend_label=label)
        p.scatter("t", "hr", source=src, color=color, marker="circle",
                  size=4, alpha=0.5, legend_label=label)
        any_data = True

    if not any_data:
        p.text(x=[0], y=[150],
               text=["No rest data — check Stop / Rest-stop columns."],
               text_color="#888888")
    else:
        p.legend.location     = "top_right"
        p.legend.label_text_font_size = "10px"
        p.legend.click_policy = "hide"
        p.legend.background_fill_alpha = 0.6

    p.xgrid.grid_line_color = "#444444"
    p.ygrid.grid_line_color = "#444444"
    return p


def _build_dynamics_plot(df: pd.DataFrame, table_df: pd.DataFrame) -> figure:
    """
    Plot 3 — Median stride length, cadence, and spring stiffness per interval rep,
    normalised to rep 1 (100 %).  Std shown as error bars.
    """
    elapsed = _get_elapsed(df)
    reps    = []

    for _, row in table_df.iterrows():
        s, e = float(row["Start (min)"]) * 60.0, float(row["Stop (min)"]) * 60.0
        if e <= s:
            continue
        mask = (elapsed >= s) & (elapsed <= e)
        if mask.sum() < 2:
            continue
        entry = {}
        for cols, _, _ in _DYN_META:
            key  = cols[0]                       # canonical name used as dict key
            vals = _get_col(df, *cols)[mask]
            valid = vals[~np.isnan(vals)]
            entry[key] = (np.median(valid) if len(valid) else np.nan,
                          np.std(valid)    if len(valid) else np.nan)
        reps.append(entry)

    n_reps = len(reps)

    p = figure(
        title="Running Dynamics vs Rep  (normalised to Rep 1 = 100 %)",
        width=_FIG_W, height=_FIG_H,
        x_axis_label="Rep",
        y_axis_label="% of Rep 1",
        toolbar_location="above",
        tools=_TOOLS,
    )
    p.title.text_font_size = "13px"

    if n_reps == 0:
        p.text(x=[0], y=[100],
               text=["No interval data — check Start/Stop times."],
               text_color="#888888")
        p.xgrid.grid_line_color = "#444444"
        p.ygrid.grid_line_color = "#444444"
        return p

    x = np.arange(1, n_reps + 1, dtype=float)
    cap_w = 0.18   # half-width of error-bar caps in rep units

    for cols, color, label in _DYN_META:
        key     = cols[0]
        medians = np.array([r[key][0] for r in reps])
        stds    = np.array([r[key][1] for r in reps])

        # Normalise to rep-1 median (skip metric if rep-1 is NaN/zero).
        # When the entire column is absent (e.g. Spring Stiffness without a
        # Stryd pod) all medians are NaN.  Still add a greyed-out legend entry
        # so the user knows the metric was looked for but unavailable.
        ref = medians[0] if (not np.isnan(medians[0]) and medians[0] != 0) else None
        if ref is None:
            p.line(
                [np.nan], [np.nan], color=color, line_width=1.5,
                line_dash="dashed", alpha=0.45,
                legend_label=f"{label} (no data)",
            )
            continue

        med_pct = medians / ref * 100.0
        std_pct = stds    / ref * 100.0

        upper = med_pct + std_pct
        lower = med_pct - std_pct

        src = ColumnDataSource(dict(
            x=x, y=med_pct, upper=upper, lower=lower,
        ))

        # Vertical error bar
        p.segment("x", "lower", "x", "upper", source=src,
                  line_color=color, line_width=2)
        # Caps — one horizontal segment per upper cap, one per lower cap.
        # np.concatenate([x, x]) pairs each rep's x with its upper then lower
        # endpoint.  np.repeat(x, 2) would interleave reps, mismatching cap_y.
        cap_x  = np.concatenate([x, x])
        cap_y  = np.concatenate([upper, lower])
        cap_x0 = cap_x - cap_w
        cap_x1 = cap_x + cap_w
        p.segment(x0=cap_x0, y0=cap_y, x1=cap_x1, y1=cap_y,
                  line_color=color, line_width=2)
        # Central markers
        p.scatter("x", "y", source=src, size=9, marker="circle",
                  color=color, alpha=0.9, legend_label=label)
        p.line("x", "y", source=src, color=color,
               line_width=1.8, alpha=0.7)

    # 100 % reference line
    p.line([0.5, n_reps + 0.5], [100, 100],
           line_color="#666666", line_dash="dashed", line_width=1.2)

    p.x_range = Range1d(0.5, n_reps + 0.5)
    p.xaxis.ticker = list(range(1, n_reps + 1))

    if p.renderers:
        p.legend.location     = "top_right"
        p.legend.label_text_font_size = "10px"
        p.legend.click_policy = "hide"
        p.legend.background_fill_alpha = 0.6

    p.xgrid.grid_line_color = "#444444"
    p.ygrid.grid_line_color = "#444444"
    return p


def _build_zone_bar(
    df:        pd.DataFrame,
    table_df:  pd.DataFrame,
    max_hr:    float,
    zones:     list[dict],
) -> figure:
    """
    Plot 4 — Bar chart of time spent in each HR zone.

    Time range: from the first interval start to the last rest-stop (the workout
    portion of the run, excluding warm-up and cool-down).
    """
    elapsed = _get_elapsed(df)
    hr_all  = _get_col(df, "heart_rate")

    # Restrict to workout portion if table has data
    if len(table_df) > 0:
        t_lo = table_df["Start (min)"].min()     * 60.0
        t_hi = table_df["Rest-stop (min)"].max() * 60.0
        if t_hi > t_lo:
            mask   = (elapsed >= t_lo) & (elapsed <= t_hi)
            elapsed = elapsed[mask]
            hr_all  = hr_all[mask]

    # Estimate sample period from median dt
    dt_arr    = np.diff(elapsed)
    sample_dt = float(np.median(dt_arr)) if len(dt_arr) > 0 else 1.0

    zone_idx = _assign_hr_zones(hr_all[~np.isnan(hr_all)], max_hr, zones)
    names    = [z["name"]  for z in zones]
    colors   = [z["color"] for z in zones]

    time_min = []
    for i in range(len(zones)):
        count = int((zone_idx == i).sum())
        time_min.append(count * sample_dt / 60.0)

    src = ColumnDataSource(dict(names=names, time=time_min, color=colors))

    p = figure(
        title="Time in HR Zones  (workout portion)",
        x_range=names,
        width=_FIG_W, height=_FIG_H,
        x_axis_label="HR Zone",
        y_axis_label="Time (min)",
        toolbar_location="above",
        tools="save",
    )
    p.title.text_font_size = "13px"

    p.vbar("names", top="time", width=0.7, color="color",
           source=src, alpha=0.88)
    p.add_tools(HoverTool(
        tooltips=[("Zone", "@names"), ("Time", "@time{0.1} min")],
    ))
    p.xaxis.major_label_orientation = 0.45
    p.xgrid.grid_line_color = None
    p.ygrid.grid_line_color = "#444444"
    return p


def _build_pace_overview_plot(
    df:       pd.DataFrame,
    table_df: pd.DataFrame,
    height:   int = 300,
) -> figure:
    """
    Full-run pace trace with vertical split lines drawn from *table_df*.

    The three line styles make it easy to confirm that the numbers in the
    interval table line up with the visible pace changes:
      • green dashed  — interval start
      • red dashed    — interval stop
      • blue dotted   — rest stop
    """
    elapsed = _get_elapsed(df)

    # Prefer the pre-computed pace column (real data); fall back to deriving
    # from speed (demo data).  Always apply a light rolling smooth to reduce
    # GPS noise.  pandas rolling is used (not np.convolve) because convolve
    # propagates NaN outward — a single stopped sample would blank 14 neighbours.
    pace_raw = _get_col(df, "pace_min_km")
    if np.any(~np.isnan(pace_raw)):
        pace = (
            pd.Series(pace_raw)
            .rolling(15, center=True, min_periods=3)
            .mean()
            .values
        )
    else:
        speed_raw   = _get_col(df, "speed_ms", "speed")
        speed_clean = np.where(speed_raw > 0.3, speed_raw, np.nan)
        speed_sm    = (
            pd.Series(speed_clean)
            .rolling(15, center=True, min_periods=3)
            .mean()
            .values
        )
        pace = np.where(speed_sm > 0.3, 1000.0 / (speed_sm * 60.0), np.nan)
    t_min = elapsed / 60.0

    # Y-range: inverted so *faster* pace sits higher on the plot
    valid = pace[~np.isnan(pace)]
    p_lo  = max(1.5,  float(np.percentile(valid, 1))  - 0.3) if len(valid) else 3.0
    p_hi  = min(15.0, float(np.percentile(valid, 99)) + 0.3) if len(valid) else 8.0

    from bokeh.models import BoxZoomTool, PanTool, ResetTool, SaveTool
    box_zoom = BoxZoomTool()
    p = figure(
        title="Pace overview — verify interval splits",
        width=_FIG_W, height=height,
        x_axis_label="Time (min)",
        y_axis_label="Pace (min/km)  ← faster",
        y_range=Range1d(start=p_hi, end=p_lo),   # inverted axis
        toolbar_location="above",
        tools=[PanTool(), box_zoom, ResetTool(), SaveTool()],
    )
    p.toolbar.active_drag = box_zoom   # box-select zoom is the default gesture
    p.title.text_font_size = "12px"

    # ── Pace trace ────────────────────────────────────────────────────────────
    src = ColumnDataSource(dict(t=t_min, pace=pace))
    p.line("t", "pace", source=src,
           color=_PACE_COLOR, line_width=1.8, alpha=0.9)
    p.add_tools(HoverTool(
        tooltips=[("Time", "@t{0.1} min"), ("Pace", "@pace{0.00} min/km")],
        mode="vline",
        renderers=[p.renderers[-1]],
    ))

    # ── Vertical split lines ──────────────────────────────────────────────────
    # Table stores minutes; pace plot x-axis is also minutes — use directly.
    for _, row in table_df.iterrows():
        start     = float(row["Start (min)"])
        stop      = float(row["Stop (min)"])
        rest_stop = float(row["Rest-stop (min)"])

        if start > 0:
            p.add_layout(Span(
                location=start, dimension="height",
                line_color="#2ECC71", line_width=1.8, line_dash="dashed",
            ))
        if stop > start:
            p.add_layout(Span(
                location=stop, dimension="height",
                line_color="#E74C3C", line_width=1.8, line_dash="dashed",
            ))
        if rest_stop > stop:
            p.add_layout(Span(
                location=rest_stop, dimension="height",
                line_color="#2980B9", line_width=1.5, line_dash="dotted",
            ))

    # ── Legend (invisible dummy lines) ───────────────────────────────────────
    nan2 = [np.nan, np.nan]
    t2   = [0.0, 0.0]
    p.line(t2, nan2, color="#2ECC71", line_width=2.0, line_dash="dashed",
           legend_label="Interval start")
    p.line(t2, nan2, color="#E74C3C", line_width=2.0, line_dash="dashed",
           legend_label="Interval stop")
    p.line(t2, nan2, color="#2980B9", line_width=1.5, line_dash="dotted",
           legend_label="Rest stop")

    p.legend.location              = "top_right"
    p.legend.label_text_font_size  = "10px"
    p.legend.background_fill_alpha = 0.6
    p.legend.click_policy          = "hide"
    p.xgrid.grid_line_color        = "#444444"
    p.ygrid.grid_line_color        = "#444444"
    return p


# ═══════════════════════════════════════════════════════════════════════════════
# Panel component
# ═══════════════════════════════════════════════════════════════════════════════

class IntervalsTab(param.Parameterized):
    """
    Panel component for the Intervals analysis tab.

    Usage
    -----
    tab = IntervalsTab({})
    tab.update(run_data=my_run_data)   # push real RunData after analysis
    layout = tab.panel()
    """

    run_data = param.Parameter(default=None, doc="RunData instance from RunTrackz, or None")

    def __init__(self, state: dict, **params):
        super().__init__(**params)
        self._state = state

        # ── Controls ────────────────────────────────────────────────────────
        self._n_reps_w = pn.widgets.IntInput(
            name="Number of intervals", value=6, start=1, end=30, width=200,
        )
        self._apply_btn = pn.widgets.Button(
            name="▶  Apply", button_type="primary", width=130,
        )

        # ── Table container (replaced when n_reps changes) ──────────────────
        self._table_container = pn.Column(sizing_mode="stretch_width")
        self._pace_pane = None          # sentinel — set below after table exists
        self._rebuild_table(self._n_reps_w.value)

        # ── Pace overview pane (live-updates as the table is edited) ─────────
        n0  = self._n_reps_w.value
        h0  = _pace_height(n0)
        self._pace_pane = pn.pane.Bokeh(
            _build_pace_overview_plot(
                _make_fake_df(n0), self._current_table_df, height=h0,
            ),
            sizing_mode="stretch_width",
            min_height=240,
        )

        # ── Plot area ───────────────────────────────────────────────────────
        self._plot_container = pn.Column(
            pn.pane.Markdown(
                "_Click **▶ Apply** above to generate plots._",
                margin=(20, 10),
            ),
            sizing_mode="stretch_width",
        )

        # ── Wiring ──────────────────────────────────────────────────────────
        self._n_reps_w.param.watch(self._on_n_reps, "value")
        self._apply_btn.on_click(self._on_apply)

    # ── Public interface ───────────────────────────────────────────────────────

    def update(self, run_data=None) -> None:
        """Push new RunData after analysis completes."""
        self.run_data = run_data

    @param.depends("run_data")
    def _banner(self) -> pn.viewable.Viewable:
        """Reactive banner that reflects whether real data is loaded."""
        if self.run_data is None:
            return pn.pane.Markdown(
                "⚠️ **Demo data** — no workout loaded yet.  "
                "Load a .fit file and run an intervals analysis to use real data.",
                styles={"background": "#2c2c1e", "border-left": "3px solid #F39C12",
                        "padding": "6px 12px", "border-radius": "4px"},
                margin=(0, 0, 8, 0),
            )
        return pn.pane.Markdown(
            "✅ **Real RunData loaded** — "
            "edit the interval table below then click **▶ Apply**.",
            styles={"background": "#1e2c1e", "border-left": "3px solid #2ECC71",
                    "padding": "6px 12px", "border-radius": "4px"},
            margin=(0, 0, 8, 0),
        )

    def panel(self) -> pn.viewable.Viewable:
        """Return the full Panel layout for this tab."""
        controls = pn.Column(
            pn.Row(
                self._n_reps_w,
                pn.Spacer(width=12),
                self._apply_btn,
                align="end",
            ),
            pn.pane.Markdown(
                "_Enter all times in **minutes from the start of the run**.  "
                "Rest-stop is the end of the recovery jog after each interval.  "
                "The pace plot updates live as you edit cells._",
                margin=(2, 0, 4, 0),
                styles={"font-size": "12px", "color": "#aaaaaa"},
            ),
            pn.Row(
                # 30 / 70 width split.  Wrapping each side in a Column with a
                # CSS flex weight lets the browser distribute space correctly
                # regardless of the exact viewport width.
                pn.Column(
                    self._table_container,
                    sizing_mode="stretch_width",
                    styles={"flex": "3", "min-width": "0"},
                ),
                pn.Column(
                    self._pace_pane,
                    sizing_mode="stretch_width",
                    styles={"flex": "7", "min-width": "0"},
                ),
                sizing_mode="stretch_width",
                styles={"display": "flex"},
            ),
            sizing_mode="stretch_width",
        )

        return pn.Column(
            pn.pane.Markdown("## Intervals", margin=(10, 0, 0, 0)),
            self._banner,
            controls,
            pn.layout.Divider(margin=(8, 0, 8, 0)),
            self._plot_container,
            sizing_mode="stretch_width",
        )

    # ── Private helpers ────────────────────────────────────────────────────────

    def _rebuild_table(self, n: int) -> None:
        """Replace the table widget with a fresh one sized for *n* reps."""
        df = _default_interval_table(n)
        table = pn.widgets.Tabulator(
            df,
            editors={
                "Start (min)":     "number",
                "Stop (min)":      "number",
                "Rest-stop (min)": "number",
            },
            configuration={"headerSort": False, "columnDefaults": {"headerSort": False}},
            show_index=False,
            height=_table_height(n),
            sizing_mode="stretch_width",
        )
        # Live-update the pace plot whenever the user edits a cell
        table.param.watch(self._on_table_change, "value")
        self._table_container.clear()
        self._table_container.append(table)
        # Resize pace pane to keep it aligned with the table
        self._rebuild_pace_plot()

    def _rebuild_pace_plot(self) -> None:
        """Rebuild the pace overview Bokeh figure and push it to the pane."""
        if self._pace_pane is None:
            return     # called before pane is initialised — skip
        n   = self._n_reps_w.value
        fig = _build_pace_overview_plot(
            self._get_df(), self._current_table_df, height=_pace_height(n),
        )
        self._pace_pane.object = fig

    def _on_table_change(self, _event) -> None:
        """Called every time a cell in the Tabulator is committed."""
        self._rebuild_pace_plot()

    @param.depends("run_data", watch=True)
    def _on_run_data_change(self) -> None:
        """Rebuild the pace overview when a new workout is loaded."""
        self._rebuild_pace_plot()

    @property
    def _current_table_df(self) -> pd.DataFrame:
        """Read current table widget value."""
        widgets = [w for w in self._table_container if isinstance(w, pn.widgets.Tabulator)]
        if widgets:
            return widgets[0].value
        return _default_interval_table(self._n_reps_w.value)

    def _get_df(self) -> pd.DataFrame:
        """Return the active time-series DataFrame (real or fake)."""
        if self.run_data is not None and hasattr(self.run_data, "df"):
            return self.run_data.df
        return _make_fake_df(self._n_reps_w.value)

    def _on_n_reps(self, event) -> None:
        self._rebuild_table(event.new)

    def _on_apply(self, _event) -> None:
        df       = self._get_df()
        tdf      = self._current_table_df
        max_hr, zones = _load_hr_zones()

        try:
            p1 = _build_avg_plot(df, tdf)
            p2 = _build_rest_hr_plot(df, tdf)
            p3 = _build_dynamics_plot(df, tdf)
            p4 = _build_zone_bar(df, tdf, max_hr, zones)

            grid = pn.Column(
                pn.Row(
                    pn.pane.Bokeh(p1, sizing_mode="stretch_width"),
                    pn.pane.Bokeh(p2, sizing_mode="stretch_width"),
                ),
                pn.Row(
                    pn.pane.Bokeh(p3, sizing_mode="stretch_width"),
                    pn.pane.Bokeh(p4, sizing_mode="stretch_width"),
                ),
                sizing_mode="stretch_width",
            )
        except Exception as exc:
            grid = pn.pane.Markdown(
                f"⚠️ Error building plots: `{exc}`\n\n"
                "Check that Start < Stop < Rest-stop for every row.",
                margin=(10, 0),
            )

        self._plot_container.clear()
        self._plot_container.append(grid)
