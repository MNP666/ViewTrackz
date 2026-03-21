"""
viewtrackz.tabs.aggregates
~~~~~~~~~~~~~~~~~~~~~~~~~~
Aggregates tab — always accessible, not gated on a loaded file.

Reads from the DuckDB ``aggregates`` table (via storage.monthly_aggregates)
and the ``activities`` table (via storage.all_activities).

Shows:
  - Monthly mileage bar chart
  - Cumulative TRIMP trend line
  - Recent activities summary table (last N runs: date, type, distance, HR, TRIMP)

TODO: implement Panel widgets and chart layout.
"""

from __future__ import annotations

import panel as pn
import param


class AggregatesTab(param.Parameterized):
    """Panel component for the Aggregates tab."""

    def __init__(self, state: dict, **params):
        super().__init__(**params)
        self._state = state

    def panel(self) -> pn.viewable.Viewable:
        return pn.Column(
            pn.pane.Markdown("## Aggregates\n*Save activities to see trends here.*"),
            pn.pane.Markdown("_Not yet implemented._"),
        )
