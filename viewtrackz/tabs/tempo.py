"""
viewtrackz.tabs.tempo
~~~~~~~~~~~~~~~~~~~~~
Tempo Run analysis tab.

Displays results from runtrackz.tempo_analysis.analyze():
  - TempoStats.avg_pace_min_km
  - TempoStats.pace_variability_cv    (lower = more consistent)
  - TempoStats.hr_drift_pct           (HR rise from first to last quarter)
  - TempoStats.time_at_threshold_s / pct_at_threshold
  - TempoStats.hr_pct_of_max

TODO: implement Panel widgets and chart layout.
"""

from __future__ import annotations

import panel as pn
import param


class TempoTab(param.Parameterized):
    """Panel component for the Tempo Run analysis tab."""

    def __init__(self, state: dict, **params):
        super().__init__(**params)
        self._state = state

    def panel(self) -> pn.viewable.Viewable:
        return pn.Column(
            pn.pane.Markdown("## Tempo Run\n*Run a tempo analysis to see results here.*"),
            pn.pane.Markdown("_Not yet implemented._"),
        )
