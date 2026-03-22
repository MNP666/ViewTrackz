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
   ``analysis/utils.run_fit`` / ``run_fit_metadata`` wrappers).
2. Map FitTrackz output column names  →  RunTrackz ``DATAFRAME_SCHEMA``.
3. Build and return a ``RunData`` object ready for analysis.

Two entry points
----------------
``load_metadata(fit_path)``
    Lightweight — calls ``fit-cli <path> metadata``, returns a plain dict
    with device info, session totals, and sport from the FIT file header.
    Use this first when a file is uploaded to populate the database.

``load(fit_path, smoother, ...)``
    Full parse — calls ``fit-cli`` for all requested channels, applies
    smoothing, maps columns to RunTrackz schema, returns a ``RunData``.
    Called when the user clicks "Analyse".

FitTrackz output columns  (from ``run_fit()``)
-----------------------------------------------
    timestamp      FIT epoch seconds (int)
    time           UTC datetime (pandas Timestamp, already converted)
    elapsed_min    minutes from start
    distance_m     cumulative distance
    raw_<ch>       unsmoothed channel values
    smoothed_<ch>  smoothed channel values   ← used by RunTrackz

RunTrackz DATAFRAME_SCHEMA (required columns)
---------------------------------------------
    index          UTC-aware DatetimeIndex
    heart_rate     bpm
    speed_ms       m/s
    speed_kmh      km/h   (derived automatically by RunData.from_dataframe)
    pace_min_km    min/km (derived automatically)
    distance_m     m
    elapsed_s      s      (derived automatically)

FIT epoch conversion
--------------------
    ``time_created`` and ``start_time`` in the metadata dict are raw FIT
    epoch seconds (since 1989-12-31 UTC).  Convert with:
        pd.to_datetime(value + 631_065_600, unit='s', utc=True)
"""

from __future__ import annotations

import json
import subprocess
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
    "smoothed_stride_length":        "stride_length_m",
    "smoothed_vertical_oscillation": "vertical_oscillation_cm",
    "smoothed_stance_time":          "stance_time_ms",
}


def load_metadata(fit_path: Path | str) -> dict:
    """
    Extract activity-level metadata from a ``.fit`` file without decoding
    per-second records.

    Calls ``fit-cli <path> metadata`` which reads only the ``file_id``,
    ``session``, and ``device_info`` FIT messages and returns JSON.

    Parameters
    ----------
    fit_path : Path or str
        Path to the ``.fit`` file.

    Returns
    -------
    dict
        Keys (all values may be ``None`` if the device did not record them):

        Device
            ``manufacturer``      str   — "garmin", "coros", "suunto", …
            ``product_name``      str   — device model, e.g. "VERTIX 2S"
            ``serial_number``     int
            ``firmware_version``  str   — e.g. "4.20"
            ``time_created``      int   — FIT epoch seconds

        Session
            ``sport``             str   — "running", "cycling", …
            ``sub_sport``         str   — "generic", "trail", "treadmill", …
            ``start_time``        int   — FIT epoch seconds
            ``total_elapsed_s``   float — wall-clock duration incl. pauses
            ``total_timer_s``     float — active time only
            ``total_distance_m``  float
            ``total_ascent_m``    float
            ``total_descent_m``   float
            ``total_calories``    int
            ``avg_speed_ms``      float
            ``max_speed_ms``      float
            ``avg_heart_rate``    int
            ``max_heart_rate``    int
            ``avg_cadence``       int
            ``max_cadence``       int
            ``avg_power_w``       int
            ``max_power_w``       int
            ``training_stress_score`` float

    Notes
    -----
    Convert ``time_created`` or ``start_time`` to a pandas Timestamp with::

        pd.to_datetime(meta["start_time"] + 631_065_600, unit="s", utc=True)
    """
    from analysis.utils import REPO_ROOT  # FitTrackz repo root for cwd

    fit_path = Path(fit_path)
    if not fit_path.exists():
        raise FileNotFoundError(f".fit file not found: {fit_path}")

    cmd = [
        "cargo", "run", "--release", "--bin", "fit-cli", "--",
        str(fit_path),
        "metadata",
    ]
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=True,
        cwd=REPO_ROOT,
    )
    if not result.stdout.strip():
        raise RuntimeError(
            f"fit-cli metadata produced no output for {fit_path}.\n"
            f"stderr:\n{result.stderr}"
        )
    return json.loads(result.stdout)


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
