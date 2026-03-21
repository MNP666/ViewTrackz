"""
viewtrackz.tabs.normal_run
~~~~~~~~~~~~~~~~~~~~~~~~~~
Normal Run tab.

No deep analysis for normal runs — just a summary card derived from
hr_analysis and pace_analysis, which always run regardless of activity type.

Shows:
  - Distance, duration, avg HR, avg pace
  - HR zone distribution (bar chart)
  - Kilometre splits table

TODO: implement Panel widgets.
"""

from __future__ import annotations

import panel as pn
import param


class NormalRunTab(param.Parameterized):
    """Panel component for the Normal Run tab."""

    def __init__(self, state: dict, **params):
        super().__init__(**params)
        self._state = state

    def panel(self) -> pn.viewable.Viewable:
        return pn.Column(
            pn.pane.Markdown("## Normal Run\n*Load a normal run to see a summary here.*"),
            pn.pane.Markdown("_Not yet implemented._"),
        )
