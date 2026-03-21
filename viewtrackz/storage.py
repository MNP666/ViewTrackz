"""
viewtrackz.storage
~~~~~~~~~~~~~~~~~~
All persistence for ViewTrackz: DuckDB tables and Parquet files.

This is the ONLY module that touches the database or the filesystem for
writing.  Nothing in the UI or analysis layer should import duckdb directly.

Schema
------
    activities          — one row per stored activity (the index)
    analysis_long_run   — long run results, FK → activities.id
    analysis_tempo      — tempo results,    FK → activities.id
    analysis_intervals  — interval results, FK → activities.id
    analysis_treadmill  — treadmill results,FK → activities.id
    aggregates          — monthly rollups, updated on every save

Parquet files are stored in data/parquet/ using the naming convention
produced by runtrackz.make_parquet_path(): ``DDMMYYYY_run_NN.parquet``.
"""

from __future__ import annotations

import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

import duckdb
import pandas as pd

from viewtrackz.config import DATABASE_PATH, PARQUET_DIR


# ── Schema DDL ────────────────────────────────────────────────────────────────

_DDL = """
CREATE TABLE IF NOT EXISTS activities (
    id              UUID        DEFAULT gen_random_uuid() PRIMARY KEY,
    filename        VARCHAR,
    parquet_path    VARCHAR,
    date            DATE,
    activity_type   VARCHAR,
    distance_km     FLOAT,
    duration_s      INTEGER,
    avg_hr          FLOAT,
    trimp           FLOAT,
    smoother_used   VARCHAR,
    notes           VARCHAR,
    created_at      TIMESTAMP   DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS analysis_long_run (
    id                      UUID    DEFAULT gen_random_uuid() PRIMARY KEY,
    activity_id             UUID    REFERENCES activities(id),
    cardiac_drift_pct       FLOAT,
    pacing_strategy         VARCHAR,
    first_half_pace_min_km  FLOAT,
    second_half_pace_min_km FLOAT,
    thirds_json             VARCHAR     -- JSON-encoded list of RunThird dicts
);

CREATE TABLE IF NOT EXISTS analysis_tempo (
    id                      UUID    DEFAULT gen_random_uuid() PRIMARY KEY,
    activity_id             UUID    REFERENCES activities(id),
    avg_pace_min_km         FLOAT,
    pace_variability_cv     FLOAT,
    avg_hr                  FLOAT,
    hr_drift_pct            FLOAT,
    time_at_threshold_s     FLOAT,
    pct_at_threshold        FLOAT,
    hr_pct_of_max           FLOAT
);

CREATE TABLE IF NOT EXISTS analysis_intervals (
    id                      UUID    DEFAULT gen_random_uuid() PRIMARY KEY,
    activity_id             UUID    REFERENCES activities(id),
    num_intervals           INTEGER,
    avg_interval_pace_min_km FLOAT,
    pace_consistency_cv     FLOAT,
    hr_consistency_cv       FLOAT,
    intervals_json          VARCHAR,   -- JSON-encoded list of Interval dicts
    recoveries_json         VARCHAR    -- JSON-encoded list of Recovery dicts
);

CREATE TABLE IF NOT EXISTS analysis_treadmill (
    id                      UUID    DEFAULT gen_random_uuid() PRIMARY KEY,
    activity_id             UUID    REFERENCES activities(id),
    avg_gap_min_km          FLOAT,
    flat_equivalent_dist_m  FLOAT,
    gap_factor              FLOAT,
    segments_json           VARCHAR    -- JSON-encoded list of GradientSegment dicts
);

CREATE TABLE IF NOT EXISTS aggregates (
    month               DATE    PRIMARY KEY,
    total_distance_km   FLOAT   DEFAULT 0,
    total_duration_s    INTEGER DEFAULT 0,
    total_trimp         FLOAT   DEFAULT 0,
    n_activities        INTEGER DEFAULT 0
);
"""


# ── Connection context manager ────────────────────────────────────────────────

@contextmanager
def get_db(path: Path = DATABASE_PATH):
    """Yield a DuckDB connection, ensuring it's closed on exit."""
    conn = duckdb.connect(str(path))
    try:
        yield conn
    finally:
        conn.close()


def ensure_schema(path: Path = DATABASE_PATH) -> None:
    """Create all tables if they don't exist yet.  Safe to call on every startup."""
    with get_db(path) as conn:
        conn.execute(_DDL)


# ── Save ──────────────────────────────────────────────────────────────────────

def save_activity(
    run,                      # runtrackz.RunData
    hr_stats,                 # runtrackz.hr_analysis.HRStats
    pace_stats,               # runtrackz.pace_analysis.PaceStats
    activity_type: str,
    type_stats=None,          # type-specific stats dataclass, or None for normal runs
    notes: Optional[str] = None,
    db_path: Path = DATABASE_PATH,
    parquet_dir: Path = PARQUET_DIR,
) -> uuid.UUID:
    """
    Persist a completed activity to DuckDB and Parquet.

    Steps
    -----
    1. Save the time-series DataFrame as a Parquet file.
    2. Insert a row into ``activities``.
    3. Insert type-specific analysis results (if provided).
    4. Upsert the ``aggregates`` row for the activity's month.

    Parameters
    ----------
    run : RunData
        The parsed and smoothed run.
    hr_stats : HRStats
        Heart rate analysis results.
    pace_stats : PaceStats
        Pace analysis results.
    activity_type : str
        One of ``long_run``, ``tempo``, ``intervals``, ``treadmill``, ``normal``.
    type_stats : dataclass, optional
        Output from the type-specific RunTrackz analysis function.
        Pass ``None`` for normal runs.
    notes : str, optional
        Free-text notes to store with the activity.
    db_path : Path
        Override the default database path (useful for testing).
    parquet_dir : Path
        Override the default Parquet directory (useful for testing).

    Returns
    -------
    uuid.UUID
        The newly created activity ID.
    """
    import runtrackz
    import json

    ensure_schema(db_path)

    # ── 1. Save Parquet ───────────────────────────────────────────────────
    parquet_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = runtrackz.make_parquet_path(run, parquet_dir)
    run.save_parquet(parquet_path)

    # ── 2. Insert into activities ─────────────────────────────────────────
    activity_id = uuid.uuid4()
    smoother_used = getattr(run, "_smoother_used", None)

    with get_db(db_path) as conn:
        conn.execute("""
            INSERT INTO activities (
                id, filename, parquet_path, date, activity_type,
                distance_km, duration_s, avg_hr, trimp, smoother_used, notes
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, [
            str(activity_id),
            run.source_file.name,
            str(parquet_path),
            run.df.index[0].date(),
            activity_type,
            round(pace_stats.total_distance_km, 3),
            int(pace_stats.total_time_s),
            round(hr_stats.avg_hr, 1),
            round(hr_stats.trimp, 2),
            smoother_used,
            notes,
        ])

        # ── 3. Insert type-specific analysis ─────────────────────────────
        if type_stats is not None:
            _insert_type_stats(conn, activity_id, activity_type, type_stats)

        # ── 4. Upsert monthly aggregates ──────────────────────────────────
        _upsert_aggregates(
            conn,
            month=run.df.index[0].date().replace(day=1),
            distance_km=pace_stats.total_distance_km,
            duration_s=int(pace_stats.total_time_s),
            trimp=hr_stats.trimp,
        )

    return activity_id


def _insert_type_stats(conn, activity_id: uuid.UUID, activity_type: str, stats) -> None:
    """Insert type-specific analysis results into the appropriate table."""
    import json
    aid = str(activity_id)

    if activity_type == "long_run":
        thirds_json = json.dumps([
            {"label": t.label, "avg_pace_min_km": t.avg_pace_min_km,
             "avg_hr": t.avg_hr, "distance_m": t.distance_m}
            for t in (stats.thirds or [])
        ])
        conn.execute("""
            INSERT INTO analysis_long_run (
                activity_id, cardiac_drift_pct, pacing_strategy,
                first_half_pace_min_km, second_half_pace_min_km, thirds_json
            ) VALUES (?, ?, ?, ?, ?, ?)
        """, [aid, stats.cardiac_drift_pct, stats.pacing_strategy,
              stats.first_half_pace_min_km, stats.second_half_pace_min_km,
              thirds_json])

    elif activity_type == "tempo":
        conn.execute("""
            INSERT INTO analysis_tempo (
                activity_id, avg_pace_min_km, pace_variability_cv, avg_hr,
                hr_drift_pct, time_at_threshold_s, pct_at_threshold, hr_pct_of_max
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, [aid, stats.avg_pace_min_km, stats.pace_variability_cv,
              stats.avg_hr, stats.hr_drift_pct, stats.time_at_threshold_s,
              stats.pct_at_threshold, stats.hr_pct_of_max])

    elif activity_type == "intervals":
        intervals_json = json.dumps([
            {"index": iv.index, "start_s": iv.start_s, "end_s": iv.end_s,
             "duration_s": iv.duration_s, "distance_m": iv.distance_m,
             "avg_pace_min_km": iv.avg_pace_min_km, "avg_hr": iv.avg_hr}
            for iv in (stats.intervals or [])
        ])
        recoveries_json = json.dumps([
            {"index": r.index, "duration_s": r.duration_s,
             "hr_drop_bpm": r.hr_drop_bpm, "avg_hr": r.avg_hr}
            for r in (stats.recoveries or [])
        ])
        conn.execute("""
            INSERT INTO analysis_intervals (
                activity_id, num_intervals, avg_interval_pace_min_km,
                pace_consistency_cv, hr_consistency_cv,
                intervals_json, recoveries_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """, [aid, stats.num_intervals, stats.avg_interval_pace_min_km,
              stats.pace_consistency_cv, stats.hr_consistency_cv,
              intervals_json, recoveries_json])

    elif activity_type == "treadmill":
        segments_json = json.dumps([
            {"index": s.index, "gradient_pct": s.gradient_pct,
             "avg_gap_min_km": s.avg_gap_min_km, "duration_s": s.duration_s}
            for s in (stats.segments or [])
        ])
        conn.execute("""
            INSERT INTO analysis_treadmill (
                activity_id, avg_gap_min_km, flat_equivalent_dist_m,
                gap_factor, segments_json
            ) VALUES (?, ?, ?, ?, ?)
        """, [aid, stats.avg_gap_min_km, stats.flat_equivalent_distance_m,
              stats.gap_factor, segments_json])


def _upsert_aggregates(
    conn,
    month,
    distance_km: float,
    duration_s: int,
    trimp: float,
) -> None:
    """Insert or update the monthly aggregates row."""
    conn.execute("""
        INSERT INTO aggregates (month, total_distance_km, total_duration_s, total_trimp, n_activities)
        VALUES (?, ?, ?, ?, 1)
        ON CONFLICT (month) DO UPDATE SET
            total_distance_km = aggregates.total_distance_km + excluded.total_distance_km,
            total_duration_s  = aggregates.total_duration_s  + excluded.total_duration_s,
            total_trimp       = aggregates.total_trimp        + excluded.total_trimp,
            n_activities      = aggregates.n_activities       + 1
    """, [month, round(distance_km, 3), duration_s, round(trimp, 2)])


# ── Query helpers ─────────────────────────────────────────────────────────────

def all_activities(db_path: Path = DATABASE_PATH) -> pd.DataFrame:
    """Return all activities ordered by date descending."""
    ensure_schema(db_path)
    with get_db(db_path) as conn:
        return conn.execute("""
            SELECT id, filename, date, activity_type,
                   distance_km, duration_s, avg_hr, trimp, notes
            FROM activities
            ORDER BY date DESC, created_at DESC
        """).df()


def get_activity(activity_id: str, db_path: Path = DATABASE_PATH) -> dict:
    """Return a single activity row as a dict."""
    with get_db(db_path) as conn:
        row = conn.execute(
            "SELECT * FROM activities WHERE id = ?", [activity_id]
        ).df()
    if row.empty:
        raise KeyError(f"No activity found with id={activity_id}")
    return row.iloc[0].to_dict()


def monthly_aggregates(db_path: Path = DATABASE_PATH) -> pd.DataFrame:
    """Return all monthly aggregate rows ordered by month ascending."""
    ensure_schema(db_path)
    with get_db(db_path) as conn:
        return conn.execute(
            "SELECT * FROM aggregates ORDER BY month ASC"
        ).df()
