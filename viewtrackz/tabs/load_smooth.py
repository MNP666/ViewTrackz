"""
viewtrackz.tabs.load_smooth
~~~~~~~~~~~~~~~~~~~~~~~~~~~
"Load & Smooth" tab — the entry point for every new activity.

Workflow
--------
1. User uploads a .fit file via FileInput widget.
2. FitTrackz parses the file (fittrackz_adapter.load).
3. A quick preview chart (HR and pace over time) is shown.
4. User selects a smoother from the dropdown and clicks "Apply Smoother"
   to re-parse and update the preview — repeat until satisfied.
5. User selects the activity type and clicks "Analyse".
6. The tab triggers the appropriate RunTrackz analysis functions,
   populates shared state, and signals the correct analysis tab.

TODO: implement Panel widgets and callbacks.
"""

from __future__ import annotations

import panel as pn
import param


class LoadSmoothTab(param.Parameterized):
    """Panel component for the Load & Smooth tab."""

    def __init__(self, state: dict, **params):
        super().__init__(**params)
        self._state = state

    # ── Public interface ──────────────────────────────────────────────────

    def panel(self) -> pn.viewable.Viewable:
        """Return the Panel layout for this tab."""
        return pn.Column(
            pn.pane.Markdown("## Load & Smooth\n*Upload a `.fit` file to get started.*"),
            pn.pane.Markdown("_Not yet implemented._"),
        )

    def load_from_parquet(self, parquet_path: str) -> None:
        """
        Load a previously saved activity from Parquet (called by ActivityBrowser).
        Populates shared state and switches to the appropriate analysis tab.

        TODO: implement.
        """
        pass
