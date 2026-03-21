"""
viewtrackz.tabs.intervals
~~~~~~~~~~~~~~~~~~~~~~~~~
Intervals (workout) analysis tab.

Displays results from runtrackz.workout_analysis.analyze():
  - WorkoutStats.num_intervals
  - WorkoutStats.intervals          — per-rep table: pace, HR, duration, distance
  - WorkoutStats.recoveries         — recovery HR drop per rep
  - WorkoutStats.avg_interval_pace_min_km
  - WorkoutStats.pace_consistency_cv / hr_consistency_cv

TODO: implement Panel widgets and chart layout.
"""

from __future__ import annotations

import panel as pn
import param


class IntervalsTab(param.Parameterized):
    """Panel component for the Intervals analysis tab."""

    def __init__(self, state: dict, **params):
        super().__init__(**params)
        self._state = state

    def panel(self) -> pn.viewable.Viewable:
        return pn.Column(
            pn.pane.Markdown("## Intervals\n*Run an interval analysis to see results here.*"),
            pn.pane.Markdown("_Not yet implemented._"),
        )
