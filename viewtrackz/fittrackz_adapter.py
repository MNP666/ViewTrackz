"""
viewtrackz.fittrackz_adapter
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The single seam between FitTrackz and RunTrackz.

This module is the ONLY place in ViewTrackz that knows about FitTrackz
internals.  When FitTrackz eventually exposes a PyO3 Python binding, only
this file changes — nothing else in the app needs to.

Responsibilities
----------------
1. Call the FitTrackz binary via subprocess (delegates to FitTrackz's own
   ``analysis/utils.run_fit`` wrapper).
2. Map FitTrackz output column names  →  RunTrackz ``DATAFRAME_SCHEMA``.
3. Build and return a ``RunData`` object ready for analysis.

FitTrackz output columns
------------------------
    timestamp      FIT epoch seconds (int)
    time           UTC datetime (pandas Timestamp, already converted)
    elapsed_min    minutes from start
    distance_m     cumulative distance
    raw_<ch>       unsmoothed channel values
    smoothed_<ch>  smoothed channel values   ← we use these

RunTrackz DATAFRAME_SCHEMA (required columns)
---------------------------------------------
    index          UTC-aware DatetimeIndex
    heart_rate     bpm
    speed_ms       m/s
    speed_kmh      km/h   (derived automatically by RunData.from_dataframe)
    pace_min_km    min/km (derived automatically)
    distance_m     m
    elapsed_s      s      (derived automatically)
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import pandas as pd

from viewtrackz.config import FITTRACKZ_DIR, RUNTRACKZ_DIR

# Make FitTrackz and RunTrackz importable
if str(FITTRACKZ_DIR) not in sys.path:
    sys.path.insert(0, str(FITTRACKZ_DIR))
if str(RUNTRACKZ_DIR) not in sys.path:
    sys.path.insert(0, str(RUNTRACKZ_DIR))

from analysis.utils import run_fit  # FitTrackz subprocess wrapper  # noqa: E402
import runtrackz                    # RunTrackz analysis library     # noqa: E402

# ── Column mapping ────────────────────────────────────────────────────────────
# Maps FitTrackz "smoothed_<channel>" names → RunTrackz DATAFRAME_SCHEMA names.
# Only channels that map 1-to-1 are listed here; derived columns (speed_kmh,
# pace_min_km, elapsed_s) are handled automatically by RunData.from_dataframe.

_CHANNEL_MAP: dict[str, str] = {
    "smoothed_heart_rate": "heart_rate",
    "smoothed_speed":      "speed_ms",
    "smoothed_altitude":   "altitude_m",
    "smoothed_cadence":    "cadence",
    "smoothed_power":      "power_w",
    # running dynamics (Coros-specific channels)
    "smoothed_vertical_oscillation": "vertical_oscillation_cm",
    "smoothed_stance_time":          "stance_time_ms",
}


def load(
    fit_path: Path | str,
    smoother: str = "sma",
    param: float = 10,
    channels: Optional[list[str]] = None,
    min_speed: Optional[float] = None,
) -> runtrackz.RunData:
    """
    Parse a ``.fit`` file with FitTrackz and return a ``RunData`` ready for
    RunTrackz analysis.

    Parameters
    ----------
    fit_path : Path or str
        Path to the ``.fit`` file.
    smoother : str
        Smoothing algorithm: ``"sma"``, ``"ema"``, or ``"none"``.
    param : float
        Smoother parameter: window size for SMA, alpha (0–1) for EMA.
    channels : list[str], optional
        FitTrackz channel names to request.  Defaults to the channels listed
        in ``viewtrackz.config.FIT_CHANNELS``.
    min_speed : float, optional
        Records with speed below this threshold (m/s) are excluded.
        Defaults to FitTrackz's ``config.toml`` value (2.0 m/s).

    Returns
    -------
    runtrackz.RunData
        Wrapped and validated run, ready for ``hr_analysis.analyze()`` etc.
    """
    from viewtrackz.config import FIT_CHANNELS

    fit_path = Path(fit_path)
    if not fit_path.exists():
        raise FileNotFoundError(f".fit file not found: {fit_path}")

    # ── 1. Call FitTrackz ─────────────────────────────────────────────────
    df_raw = run_fit(
        fit_file=fit_path,
        channels=channels or FIT_CHANNELS,
        smoother=smoother,
        param=param,
        min_speed=min_speed,
    )

    if df_raw.empty:
        raise ValueError(f"FitTrackz returned an empty DataFrame for: {fit_path}")

    # ── 2. Map columns ────────────────────────────────────────────────────
    df = _map_columns(df_raw)

    # ── 3. Build a minimal session dict from available metadata ───────────
    session = _build_session(df, fit_path)

    # ── 4. Construct RunData ──────────────────────────────────────────────
    smoother_label = "none" if smoother == "none" else f"{smoother}_{param}"
    run = runtrackz.RunData.from_dataframe(
        df,
        session=session,
        source_file=fit_path,
        is_smoothed=(smoother != "none"),
    )
    # Stash the smoother label so storage.py can persist it
    run._smoother_used = smoother_label   # type: ignore[attr-defined]

    return run


def _map_columns(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Rename FitTrackz output columns to RunTrackz DATAFRAME_SCHEMA names,
    set the UTC DatetimeIndex, and drop FitTrackz-specific housekeeping
    columns.

    Parameters
    ----------
    df_raw : pd.DataFrame
        Raw DataFrame from ``run_fit()``.

    Returns
    -------
    pd.DataFrame
        Schema-conformant DataFrame ready for ``RunData.from_dataframe()``.
    """
    df = df_raw.copy()

    # Set UTC DatetimeIndex from FitTrackz's pre-converted 'time' column
    if "time" in df.columns:
        df.index = pd.to_datetime(df["time"], utc=True)
    elif "timestamp" in df.columns:
        # Fallback: FIT epoch (seconds since 1989-12-31)
        fit_epoch = pd.Timestamp("1989-12-31", tz="UTC")
        df.index = fit_epoch + pd.to_timedelta(df["timestamp"], unit="s")
    else:
        raise ValueError("FitTrackz output has neither 'time' nor 'timestamp' column.")

    df.index.name = None

    # Rename smoothed channels → schema names
    rename = {k: v for k, v in _CHANNEL_MAP.items() if k in df.columns}
    df = df.rename(columns=rename)

    # Keep distance_m as-is (same name in both projects)
    # Drop FitTrackz housekeeping columns that RunTrackz doesn't need
    drop = ["timestamp", "time", "elapsed_min"]
    # Also drop raw_* columns — we only use the smoothed versions
    drop += [c for c in df.columns if c.startswith("raw_")]
    # Drop any remaining smoothed_* columns that weren't in our mapping
    drop += [c for c in df.columns if c.startswith("smoothed_")]
    df = df.drop(columns=[c for c in drop if c in df.columns])

    return df


def _build_session(df: pd.DataFrame, fit_path: Path) -> dict:
    """
    Build a minimal session summary dict from the time-series DataFrame.
    FitTrackz's subprocess interface doesn't return session-level metadata,
    so we derive the basics ourselves.
    """
    session: dict = {
        "sport":    "running",
        "filename": fit_path.name,
    }
    if "distance_m" in df.columns and not df["distance_m"].isna().all():
        session["total_distance"] = float(df["distance_m"].max())
    if not df.empty:
        elapsed = (df.index[-1] - df.index[0]).total_seconds()
        session["total_elapsed_time"] = float(elapsed)
    if "heart_rate" in df.columns and not df["heart_rate"].isna().all():
        session["avg_heart_rate"] = float(df["heart_rate"].mean())
    return session
