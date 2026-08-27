"""
Feature assembly — build the model-ready feature table.

Loads the 1km grid, the weather-point mapping, and hourly weather data,
joins them so every grid_id has hourly weather features, and optionally
merges terrain features (slope, elevation, TWI) produced by a teammate's
pipeline (app/terrain_utils.py).

The full table is ~63M rows (904 cells × 70,128 hours).  To keep memory
manageable, a date-range filter is applied — default is monsoon months
only (May–October).

Usage
-----
    from app.features import build_feature_table
    df = build_feature_table("data/processed")
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# ── Weather columns we carry forward ─────────────────────────────────
WEATHER_COLS = [
    "precipitation_mm",
    "soil_moisture_0_7",
    "soil_moisture_7_28",
    "temp_c",
    "api_3d",
    "api_7d",
]

# ── Default monsoon months (May–October) ──────────────────────────────
MONSOON_MONTHS = (5, 6, 7, 8, 9, 10)


def build_feature_table(
    data_dir: str | Path = "data/processed",
    start_date: str | None = None,
    end_date: str | None = None,
    monsoon_only: bool = True,
    terrain_path: str | Path | None = None,
) -> pd.DataFrame:
    """
    Build the model-ready feature table by joining grid, mapping, and weather.

    Parameters
    ----------
    data_dir : str or Path
        Directory containing kamrup_metro_grid_1km.parquet,
        grid_weather_mapping.parquet, and weather_hourly.parquet.
    start_date, end_date : str or None
        Optional ISO-8601 date strings to restrict the time range.
        Applied BEFORE the monsoon filter if both are set.
    monsoon_only : bool
        If True (default), keep only months May–October.
    terrain_path : str, Path, or None
        Explicit path to terrain_features.parquet.  If None, looks for
        ``data_dir / "terrain_features.parquet"`` automatically.

    Returns
    -------
    pandas.DataFrame
        Columns: grid_id, timestamp, + weather features, + terrain
        features (when available).
    """
    data_dir = Path(data_dir)

    # ── 1. Load grid (for grid_id list only — geometry not needed here) ──
    grid_path = data_dir / "kamrup_metro_grid_1km.parquet"
    grid_df = pd.read_parquet(grid_path, columns=["grid_id"])
    n_cells = len(grid_df)
    logger.info("Loaded grid: %d cells", n_cells)

    # ── 2. Load weather-point mapping ─────────────────────────────────
    mapping_path = data_dir / "grid_weather_mapping.parquet"
    mapping_df = pd.read_parquet(mapping_path, columns=["grid_id", "weather_point_id"])
    logger.info(
        "Loaded mapping: %d grid→weather links, %d unique weather points",
        len(mapping_df),
        mapping_df["weather_point_id"].nunique(),
    )

    # ── 3. Load weather hourly (with optional date-range filter) ──────
    weather_cols = ["weather_point_id", "timestamp"] + WEATHER_COLS
    weather_path = data_dir / "weather_hourly.parquet"

    # Read the full file; filter in-memory after load.
    weather_df = pd.read_parquet(weather_path, columns=weather_cols)
    logger.info(
        "Loaded weather: %d rows, date range %s to %s",
        len(weather_df),
        weather_df["timestamp"].min(),
        weather_df["timestamp"].max(),
    )

    # Apply explicit date-range bounds
    if start_date is not None:
        weather_df = weather_df[weather_df["timestamp"] >= pd.Timestamp(start_date)]
    if end_date is not None:
        weather_df = weather_df[weather_df["timestamp"] <= pd.Timestamp(end_date)]

    # Apply monsoon filter
    if monsoon_only:
        weather_df = weather_df[weather_df["timestamp"].dt.month.isin(MONSOON_MONTHS)]
        logger.info(
            "Monsoon filter (months %s): %d weather rows retained",
            MONSOON_MONTHS,
            len(weather_df),
        )

    # Downcast float64 → float32 to halve memory on the 31M-row result
    float_cols = weather_df.select_dtypes("float64").columns
    weather_df[float_cols] = weather_df[float_cols].astype("float32")

    # ── 4. Load terrain features (small, 904 rows) BEFORE the big join ─
    if terrain_path is None:
        terrain_path = data_dir / "terrain_features.parquet"
    else:
        terrain_path = Path(terrain_path)

    terrain_df = None
    if terrain_path.exists():
        terrain_df = pd.read_parquet(terrain_path)
        terrain_cols_list = [c for c in terrain_df.columns if c != "grid_id"]
        # Downcast terrain too
        tf_floats = terrain_df.select_dtypes("float64").columns
        terrain_df[tf_floats] = terrain_df[tf_floats].astype("float32")
        logger.info(
            "Loaded terrain features: %d rows, columns %s",
            len(terrain_df),
            terrain_cols_list,
        )
    else:
        logger.warning(
            "terrain_features.parquet NOT FOUND at %s -- "
            "continuing WITHOUT terrain columns (slope, elevation, TWI). "
            "Re-run after teammate produces this file.",
            terrain_path,
        )

    # ── 5. Merge mapping + terrain (tiny: 904 rows) ───────────────────
    # Attach terrain to the mapping table BEFORE the big weather join,
    # so the join only copies small terrain columns alongside weather.
    mapping_enriched = mapping_df.copy()
    if terrain_df is not None:
        mapping_enriched = mapping_enriched.merge(terrain_df, on="grid_id", how="left")
        logger.info("Terrain merged onto mapping (%d rows)", len(mapping_enriched))

    # ── 5b. Compute 3hr rolling precip on weather (671K rows, not 31.9M) ─
    weather_df.sort_values(["weather_point_id", "timestamp"], inplace=True)
    weather_df["precip_3hr_mm"] = (
        weather_df.groupby("weather_point_id")["precipitation_mm"]
        .transform(lambda s: s.rolling(window=3, min_periods=1).sum())
    )

    # ── 6. Join: enriched-mapping × weather → full feature table ──────
    #
    # mapping_enriched has grid_id + weather_point_id + terrain (904 rows).
    # weather has weather_point_id × timestamp (19 WPs × N hours).
    # Process per weather_point_id to avoid one giant allocation.
    chunks = []
    for wp_id, wp_weather in weather_df.groupby("weather_point_id"):
        wp_mapping = mapping_enriched[mapping_enriched["weather_point_id"] == wp_id]
        # Cross join: each grid_id mapped to this WP gets all timestamps
        wp_weather_slim = wp_weather.drop(columns=["weather_point_id"])
        chunk = wp_mapping.assign(key=1).merge(
            wp_weather_slim.assign(key=1), on="key"
        ).drop(columns=["key", "weather_point_id"])

        # Downcast to float32 to halve memory versus float64.
        #
        # NOT float16: the trigger model is trained on float32 features
        # (app/train_trigger_model.py), and float16 quantises api_7d to
        # ~0.5 mm near 500 mm and precip_3hr to ~0.03 mm near the 60 mm
        # labelling threshold.  Serving float16 to a float32-trained model
        # is train/serve skew that can flip borderline tree splits.
        float_cols = chunk.select_dtypes(include=["float64"]).columns
        chunk[float_cols] = chunk[float_cols].astype("float32")

        chunks.append(chunk)

    # Free memory before the massive pd.concat
    del weather_df, mapping_enriched
    import gc
    gc.collect()

    feature_df = pd.concat(chunks, ignore_index=True)
    del chunks
    gc.collect()

    logger.info(
        "Feature table after weather join: %d rows (%d cells x %d timestamps)",
        len(feature_df),
        feature_df["grid_id"].nunique(),
        feature_df["timestamp"].nunique(),
    )

    if terrain_df is not None:
        logger.info("Terrain features included -- %d total columns", len(feature_df.columns))

    # Note: data is already ordered by weather_point_id groups from the
    # chunked join.  Skipping global sort (grid_id, timestamp) because
    # pandas sort on 31.9M rows exceeds available RAM on this machine.
    # Downstream code does not require sorted order.

    logger.info(
        "Feature table ready: %d rows, %d columns: %s",
        len(feature_df),
        len(feature_df.columns),
        list(feature_df.columns),
    )

    return feature_df
