"""
viewtrackz.app
~~~~~~~~~~~~~~
Panel application entry point.

Run with:
    panel serve viewtrackz/app.py --show
    panel serve viewtrackz/app.py --port 5007 --autoreload

Layout
------
    ┌────────────┬─────────────────────────────────────────────┐
    │            │  [ Load & Smooth ] [ Long Run ] [ Tempo ]   │
    │  Activity  │  [ Intervals ] [ Treadmill ] [ Normal ]     │
    │  Browser   │  [ Aggregates ]                             │
    │            │─────────────────────────────────────────────│
    │  [Upload   │                                             │
    │   .FIT]    │   Active tab content                        │
    │            │                                             │
    └────────────┴─────────────────────────────────────────────┘
"""

from __future__ import annotations

import panel as pn

from viewtrackz.storage import ensure_schema
from viewtrackz.components.activity_browser import ActivityBrowser
from viewtrackz.tabs.load_smooth import LoadSmoothTab
from viewtrackz.tabs.long_run import LongRunTab
from viewtrackz.tabs.tempo import TempoTab
from viewtrackz.tabs.intervals import IntervalsTab
from viewtrackz.tabs.treadmill import TreadmillTab
from viewtrackz.tabs.normal_run import NormalRunTab
from viewtrackz.tabs.aggregates import AggregatesTab

pn.extension(sizing_mode="stretch_width", notifications=True)

# ── Ensure DB schema exists on startup ────────────────────────────────────────
ensure_schema()

# ── Shared state ──────────────────────────────────────────────────────────────
# Tabs share a mutable state dict so that data flows naturally through the
# pipeline:  Load & Smooth → analysis tabs → save.
# This is intentionally simple for v1 — reactive state management can be
# introduced later if the app grows more complex.

state: dict = {
    "run":        None,   # runtrackz.RunData, set after FitTrackz parses
    "hr_stats":   None,   # HRStats, set after analysis
    "pace_stats": None,   # PaceStats, set after analysis
    "type_stats": None,   # type-specific stats, set after analysis
    "activity_type": None,
}

# ── Instantiate tabs ──────────────────────────────────────────────────────────

load_smooth_tab  = LoadSmoothTab(state)
long_run_tab     = LongRunTab(state)
tempo_tab        = TempoTab(state)
intervals_tab    = IntervalsTab(state)
treadmill_tab    = TreadmillTab(state)
normal_run_tab   = NormalRunTab(state)
aggregates_tab   = AggregatesTab(state)

tabs = pn.Tabs(
    ("Load & Smooth", load_smooth_tab.panel()),
    ("Long Run",      long_run_tab.panel()),
    ("Tempo",         tempo_tab.panel()),
    ("Intervals",     intervals_tab.panel()),
    ("Treadmill",     treadmill_tab.panel()),
    ("Normal Run",    normal_run_tab.panel()),
    ("Aggregates",    aggregates_tab.panel()),
    dynamic=True,
)

# ── Sidebar ───────────────────────────────────────────────────────────────────

activity_browser = ActivityBrowser(on_select=load_smooth_tab.load_from_parquet)

sidebar = pn.Column(
    pn.pane.Markdown("## ViewTrackz", margin=(10, 10, 5, 10)),
    activity_browser.panel(),
    width=260,
    styles={"border-right": "1px solid #e0e0e0"},
)

# ── Layout ────────────────────────────────────────────────────────────────────

layout = pn.Row(
    sidebar,
    pn.Column(tabs, sizing_mode="stretch_both"),
    sizing_mode="stretch_both",
)

layout.servable(title="ViewTrackz")
