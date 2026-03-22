"""
viewtrackz.config
~~~~~~~~~~~~~~~~~
Central configuration: paths, smoother options, and runtime settings.

All path resolution lives here so the rest of the app uses symbolic names
rather than constructing paths inline.
"""

from __future__ import annotations

import os
from pathlib import Path

# ── Project roots ─────────────────────────────────────────────────────────────

# Root of the ViewTrackz repo (one level above this file's package dir)
APP_DIR = Path(__file__).parent.parent

# Sibling projects — override via environment variables if your layout differs
FITTRACKZ_DIR = Path(os.getenv("FITTRACKZ_DIR", str(APP_DIR.parent / "FitTrackz"))).resolve()
RUNTRACKZ_DIR = Path(os.getenv("RUNTRACKZ_DIR", str(APP_DIR.parent / "RunTrackz"))).resolve()

# ── Data directories ──────────────────────────────────────────────────────────

DATA_DIR     = APP_DIR / "data"
DATABASE_DIR = DATA_DIR / "database"
PARQUET_DIR  = DATA_DIR / "parquet"

DATABASE_PATH = DATABASE_DIR / "viewtrackz.db"

# Ensure data directories exist at import time (safe to call repeatedly)
DATABASE_DIR.mkdir(parents=True, exist_ok=True)
PARQUET_DIR.mkdir(parents=True, exist_ok=True)

# ── FitTrackz smoother options ────────────────────────────────────────────────
# Used to populate the smoother selector in the Load & Smooth tab.

SMOOTHERS = {
    "SMA (window=5)":  ("sma", 5),
    "SMA (window=10)": ("sma", 10),
    "SMA (window=20)": ("sma", 20),
    "EMA (α=0.3)":     ("ema", 0.3),
    "EMA (α=0.1)":     ("ema", 0.1),
    "None (raw)":      ("none", 0),
}

DEFAULT_SMOOTHER = "SMA (window=10)"

# Channels to request from FitTrackz (subset of what the device may record).
# Running dynamics channels (stride_length, vertical_oscillation, stance_time)
# are native Coros fields; leg_spring_stiffness is a Stryd developer field.
# FitTrackz outputs NaN columns for channels absent from the device, so it
# is safe to always request the full list.
FIT_CHANNELS = [
    "heart_rate",
    "speed",
    "altitude",
    "cadence",
    "power",
    # Coros running dynamics
    "stride_length",
    "vertical_oscillation",
    "stance_time",
    # Stryd developer fields (NaN if no Stryd pod paired)
    "leg_spring_stiffness",
    "form_power",
    "air_power",
    "impact_loading_rate",
]

# ── Activity types ────────────────────────────────────────────────────────────

ACTIVITY_TYPES = ["long_run", "tempo", "intervals", "treadmill", "normal"]

ACTIVITY_TYPE_LABELS = {
    "long_run":  "Long Run",
    "tempo":     "Tempo",
    "intervals": "Intervals",
    "treadmill": "Treadmill",
    "normal":    "Normal Run",
}
