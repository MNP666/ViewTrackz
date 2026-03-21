"""
viewtrackz.components.activity_browser
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Sidebar activity browser — persistent across all tabs.

Shows a list of stored activities (date, type, distance) loaded from DuckDB.
Clicking one calls the provided ``on_select`` callback with the activity's
Parquet path, which triggers LoadSmoothTab.load_from_parquet().

TODO: implement Panel widget (pn.widgets.DataFrame or a custom list widget).
"""

from __future__ import annotations

from typing import Callable, Optional

import panel as pn
import param


class ActivityBrowser(param.Parameterized):
    """Sidebar component listing stored activities."""

    def __init__(self, on_select: Optional[Callable[[str], None]] = None, **params):
        super().__init__(**params)
        self._on_select = on_select

    def panel(self) -> pn.viewable.Viewable:
        return pn.Column(
            pn.pane.Markdown("### Activity History"),
            pn.pane.Markdown("_No activities saved yet._"),
        )

    def refresh(self) -> None:
        """Reload the activity list from DuckDB.  Call after saving a new activity."""
        pass
