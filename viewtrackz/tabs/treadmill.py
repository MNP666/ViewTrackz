"""
viewtrackz.tabs.treadmill
~~~~~~~~~~~~~~~~~~~~~~~~~
Treadmill analysis tab.

⚠️  This tab requires user input before analysis can run:
    The user must supply a gradient schedule — a list of
    (start_time_s, gradient_pct) pairs — via a table widget.
    This is used by runtrackz.treadmill_analysis.analyze(gradient=...).

Displays results from runtrackz.treadmill_analysis.analyze():
  - TreadmillStats.avg_gap_min_km         (grade-adjusted pace)
  - TreadmillStats.flat_equivalent_distance_m
  - TreadmillStats.gap_factor             (>1 means net uphill effort)
  - TreadmillStats.segments               (per-gradient segment breakdown)

TODO: implement gradient schedule input widget and Panel layout.
"""

from __future__ import annotations

import panel as pn
import param


class TreadmillTab(param.Parameterized):
    """Panel component for the Treadmill analysis tab."""

    def __init__(self, state: dict, **params):
        super().__init__(**params)
        self._state = state

    def panel(self) -> pn.viewable.Viewable:
        return pn.Column(
            pn.pane.Markdown(
                "## Treadmill\n"
                "*Load a treadmill activity and enter a gradient schedule to see results.*"
            ),
            pn.pane.Markdown("_Not yet implemented._"),
        )
