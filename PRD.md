# ViewTrackz — Product Requirements Document

> **Status:** Draft v0.2 · **Type:** Hobby / Learning project
> **Stack:** Python · Panel · DuckDB · Parquet · FitTrackz (Rust/subprocess) · RunTrackz (Python)

---

## 1. Purpose & Goals

ViewTrackz is a local, single-user dashboard for loading, analyzing, and storing running activity data from Coros or Garmin `.FIT` files. It acts as the orchestration, presentation, and persistence layer on top of two existing backend projects:

- **FitTrackz** — Rust binary (called via subprocess) that parses `.FIT` files and applies configurable smoothing to raw sensor data.
- **RunTrackz** — Pure Python analysis library. Takes a DataFrame, returns structured analysis results (dataclasses). Has no knowledge of files, databases, or the outside world.

The project has two goals that reinforce each other:

1. **Personal utility** — a real tool for reviewing training data.
2. **Learning** — practice designing a non-trivial Panel dashboard, structuring a Python project with a clear data pipeline, and making deliberate decisions about storage and UI layout.

### Non-goals (v1)

- PyO3 integration (subprocess is sufficient for now)
- Multi-user support or web deployment
- GPS map visualisation
- Export to PDF / report generation
- Social or sharing features

---

## 2. Project Responsibilities

This section is the architectural boundary. Each project does exactly one thing. If a feature doesn't fit inside a project's one-liner, it belongs elsewhere.

| Project | Responsibility | Does NOT do |
|---|---|---|
| **FitTrackz** | Parse binary `.FIT` files into a DataFrame; apply smoothing (SMA/EMA/none) | Analysis, storage, UI |
| **RunTrackz** | Accept a DataFrame, return analysis results as dataclasses | Parsing files, writing to disk, managing databases |
| **ViewTrackz** | Orchestrate the pipeline; own all persistence (DuckDB + Parquet); render the UI | Any analysis logic; any parsing logic |

The single seam between FitTrackz and RunTrackz is a column-mapping step that lives in ViewTrackz's `fittrackz_adapter.py`. FitTrackz outputs `smoothed_heart_rate`, `smoothed_speed`, etc. — the adapter renames these to match `RunTrackz.DATAFRAME_SCHEMA` (`heart_rate`, `speed_ms`, etc.) before constructing a `RunData` object.

---

## 3. User Stories

1. **Upload & inspect** — Drop a `.FIT` file into the dashboard and immediately see the raw parsed data to confirm the file loaded correctly.
2. **Compare smoothers** — Try different FitTrackz smoothing options and see the effect visually before committing to one.
3. **Choose analysis type** — Label an activity (Long Run, Tempo, Intervals, Treadmill, or Normal Run) and have the right analysis run automatically.
4. **Explore results** — See the RunTrackz analysis output in a well-organised tab with relevant charts and metrics.
5. **Persist the activity** — Save the activity (metadata, analysis results, time-series data) to revisit later without re-uploading.
6. **Track aggregates** — View TRIMP, monthly mileage, and similar aggregate metrics across all stored activities, including Normal Runs.
7. **Browse history** — Select a previously stored activity and reload its results in the dashboard.

---

## 4. System Architecture

```
.FIT file (upload)
      │
      ▼
┌──────────────────────────────────────────────────────────────┐
│  ViewTrackz  (Panel UI)                                      │
│                                                              │
│  fittrackz_adapter.py                                        │
│    └─ subprocess → FitTrackz binary → raw DataFrame         │
│    └─ column mapping: smoothed_* → DATAFRAME_SCHEMA names   │
│                                                              │
│  RunData.from_dataframe(df, session, is_smoothed=True)       │
│    └─ hr_analysis.analyze()   → HRStats                     │
│    └─ pace_analysis.analyze() → PaceStats                   │
│    └─ <type>_analysis.analyze() → type-specific stats       │
│                                                              │
│  storage.py                                                  │
│    └─ DuckDB: activities, analysis_*, aggregates tables     │
│    └─ Parquet: one file per activity (time-series)          │
└──────────────────────────────────────────────────────────────┘
```

### FitTrackz subprocess interface

The `fittrackz_adapter.py` wraps `FitTrackz/analysis/utils.run_fit()`:

```python
run_fit(fit_file, channels=[...], smoother="sma", param=10, min_speed=2.0)
# → DataFrame with columns: timestamp, distance_m, time (UTC datetime),
#   raw_<channel>, smoothed_<channel>  for each requested channel
```

Available smoothers: `"sma"` (window size), `"ema"` (alpha 0–1), `"none"`.

### Column mapping (adapter responsibility)

| FitTrackz output | RunTrackz schema |
|---|---|
| `smoothed_heart_rate` | `heart_rate` |
| `smoothed_speed` | `speed_ms` |
| `distance_m` | `distance_m` |
| `time` (UTC datetime) | DataFrame index |
| `smoothed_altitude` | `altitude_m` |
| `smoothed_cadence` | `cadence` |
| `smoothed_power` | `power_w` |

`speed_kmh`, `pace_min_km`, and `elapsed_s` are derived automatically by `RunData.from_dataframe()`.

---

## 5. Data Model

### DuckDB Tables

**`activities`** — one row per stored activity, acts as the index.

| Column | Type | Notes |
|---|---|---|
| `id` | UUID | Primary key |
| `filename` | VARCHAR | Original .FIT filename |
| `parquet_path` | VARCHAR | Path to the corresponding Parquet file |
| `date` | DATE | Activity date |
| `activity_type` | VARCHAR | `long_run`, `tempo`, `intervals`, `treadmill`, `normal` |
| `distance_km` | FLOAT | |
| `duration_s` | INTEGER | |
| `avg_hr` | FLOAT | |
| `trimp` | FLOAT | From `HRStats.trimp` |
| `smoother_used` | VARCHAR | e.g. `sma_10` or `ema_0.2` |
| `notes` | VARCHAR | Optional free-text |
| `created_at` | TIMESTAMP | Insertion timestamp |

**`analysis_long_run`**, **`analysis_tempo`**, **`analysis_intervals`**, **`analysis_treadmill`** — one row per analysed activity of that type, foreign-keyed to `activities.id`. Columns map directly to the fields of the corresponding RunTrackz result dataclass (formalise once RunTrackz stabilises).

**`aggregates`** — one row per calendar month, updated on every save.

| Column | Type | Notes |
|---|---|---|
| `month` | DATE | First day of month |
| `total_distance_km` | FLOAT | |
| `total_duration_s` | INTEGER | |
| `total_trimp` | FLOAT | |
| `n_activities` | INTEGER | |

### Parquet Files

One file per activity, named `DDMMYYYY_run_NN.parquet` (using `RunData.make_parquet_path()`), stored in a configurable data directory. Contains the full time-series in `DATAFRAME_SCHEMA` format. This is the source of truth for all charts — analysis results in DuckDB are derived from it.

---

## 6. Feature Scope (v1)

### 6.1 Load & Smooth (Entry point)

- File upload widget accepting `.FIT` files
- Call FitTrackz via `fittrackz_adapter.py`, display a loading indicator
- Show a quick preview of the parsed time-series (HR and pace over time)
- Allow the user to select from available smoothers (SMA / EMA / none) and adjust the parameter — re-runs the preview for visual comparison
- Confirm smoother selection before proceeding to analysis

### 6.2 Activity Classification & Analysis

- Auto-suggest activity type using `run_type.classify()` (user can override)
- Dropdown or button group to select final type
- "Analyse" button triggers `hr_analysis`, `pace_analysis`, and the type-specific analysis function
- Results are displayed in the correct analysis tab (see §6.3)
- Normal Runs skip deep analysis but still proceed to storage

### 6.3 Analysis Tabs

Each tab is only active after the corresponding analysis has been run.

| Tab | Key things to show |
|---|---|
| **Long Run** | `LongRunStats`: cardiac drift, pacing strategy, per-third breakdown |
| **Tempo** | `TempoStats`: pace variability CV, HR drift, time at threshold |
| **Intervals** | `WorkoutStats`: per-rep table (pace, HR, duration), recovery HR drop |
| **Treadmill** | `TreadmillStats`: GAP, per-segment metrics. *Requires gradient schedule input from user.* |
| **Normal Run** | Basic summary card — distance, time, avg HR, avg pace |

### 6.4 Persist Activity

- Explicit "Save activity" button — no auto-save
- Writes metadata to `activities`, analysis results to the type-specific table, time-series to Parquet
- Updates `aggregates`
- Shows a confirmation with the assigned activity ID

### 6.5 Aggregates View

- Always-accessible tab (not gated on a loaded file)
- Monthly mileage bar chart
- Cumulative TRIMP trend line
- Summary table: last N activities with type, distance, HR, TRIMP

### 6.6 Activity Browser

- Sidebar listing stored activities (date, type, distance)
- Clicking one loads its Parquet via `RunData.load_parquet()` and repopulates the relevant analysis tab

---

## 7. UI Layout (High-level)

```
┌──────────────────────────────────────────────────────────────┐
│  ViewTrackz                                                  │
├────────────┬─────────────────────────────────────────────────┤
│            │                                                 │
│  Activity  │  [ Load & Smooth ] [ Long Run ] [ Tempo ]       │
│  Browser   │  [ Intervals ] [ Treadmill ] [ Normal ]         │
│  (list of  │  [ Aggregates ]                                 │
│  stored    │─────────────────────────────────────────────────│
│  runs)     │                                                 │
│            │   Active tab content                            │
│  [Upload   │   (charts, tables, metrics widgets)             │
│   .FIT]    │                                                 │
│            │                                                 │
└────────────┴─────────────────────────────────────────────────┘
```

The sidebar is persistent. The "Load & Smooth" tab is always the entry point for a new file. The Aggregates tab is always accessible regardless of whether a file is loaded.

---

## 8. Open Questions / Future Considerations

- **PyO3 migration** — when FitTrackz exposes a Python API, replace the subprocess call in `fittrackz_adapter.py` with a direct import. No other file in ViewTrackz should need to change.
- **GPS map** — future tab using `hvplot` or `folium` once the core workflow is stable.
- **Treadmill gradient input** — the UI needs a small table widget where the user enters `(start_time_s, gradient_pct)` pairs before analysis can run. Worth designing carefully.
- **RunTrackz analysis table columns** — intentionally loose for now. Formalise the DuckDB schema for each analysis type once the RunTrackz dataclasses stabilise.
- **charts.py reuse** — RunTrackz already produces matplotlib `Figure` objects. Panel embeds these via `pn.pane.Matplotlib`. Use them as-is for v1 rather than rebuilding in HoloViews.
