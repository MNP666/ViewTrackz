"""
viewtrackz.tabs.long_run
~~~~~~~~~~~~~~~~~~~~~~~~
Long Run analysis tab.

Displays results from runtrackz.long_run_analysis.analyze():
  - LongRunStats.cardiac_drift_pct
  - LongRunStats.pacing_strategy  (even / negative / positive)
  - LongRunStats.thirds           (first / middle / last third breakdown)
  - LongRunStats.first_half_pace_min_km vs second_half_pace_min_km

Also shows core HR and pace charts from RunTrackz charts.py
(embedded via pn.pane.Matplotlib).

TODO: implement Panel widgets and chart layout.
"""

from __future__ import annotations

import panel as pn
import param


class LongRunTab(param.Parameterized):
    """Panel component for the Long Run analysis tab."""

    def __init__(self, state: dict, **params):
        super().__init__(**params)
        self._state = state

    def panel(self) -> pn.viewable.Viewable:
        return pn.Column(
            pn.pane.Markdown("## Long Run\n*Run a long run analysis to see results here.*"),
            pn.pane.Markdown("_Not yet implemented._"),
        )
