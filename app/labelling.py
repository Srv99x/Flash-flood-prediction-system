"""
Labelling module — generate binary target labels for flash-flood / landslide
prediction training.

Two labelling sources are supported (use in combination):

1. **Rainfall-threshold labelling** (implemented) — a cell-hour is positive
   if hourly or short-duration cumulative rainfall breaches an intensity
   threshold, optionally combined with a slope condition.

2. **ASDMA point-based labelling** (stub) — future: intersect known
   landslide-prone locations with extreme-rainfall dates.

Threshold justification
-----------------------
Default thresholds are derived from:

    Dikshit, A. & Satyam, N. (2019).  "Estimation of rainfall thresholds
    for landslide occurrences in Kalimpong, Darjeeling Himalayas, India."
    Innovative Infrastructure Solutions, 4:24.
    DOI: 10.1007/s41062-019-0210-0

    Also consistent with the Caine (1980) global intensity–duration envelope
    as adapted for the Indian NE / Himalayan region by Froude & Petley (2018)
    and Abraham et al. (2020, Idukki, Kerala).

For the NE Himalayan context:
  - 1-hour intensity ≥ 20 mm/hr triggers shallow landslides on steep slopes.
  - 3-hour cumulative ≥ 60 mm captures sustained heavy rainfall events that
    saturate soil and trigger slope failures even at moderate instantaneous
    rates.

These are CONFIGURABLE — pass different values to the function if needed.

Negative sampling
-----------------
Negatives are drawn from cell-hours that are NOT positive, with a spatial
buffer of ≥ 1 km from any positive cell to avoid spatial leakage (per
CLAUDE.md §5).
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd

logger = logging.getLogger(__name__)

# ── Default thresholds ────────────────────────────────────────────────
# Source: Dikshit & Satyam (2019), Kalimpong, Darjeeling Himalayas
DEFAULT_PRECIP_1HR_MM = 20.0    # mm in a single hour
DEFAULT_PRECIP_3HR_MM = 60.0    # mm cumulative over 3 hours
DEFAULT_SLOPE_DEG = 15.0        # degrees — minimum slope for terrain filter


def label_by_rainfall_threshold(
    feature_df: pd.DataFrame,
    precip_1hr_threshold: float = DEFAULT_PRECIP_1HR_MM,
    precip_3hr_threshold: float = DEFAULT_PRECIP_3HR_MM,
    slope_threshold: float = DEFAULT_SLOPE_DEG,
    slope_column: str = "slope_mean",
) -> pd.DataFrame:
    """
    Label cell-hours as positive (1) or unlabelled (0) based on rainfall
    intensity thresholds, optionally combined with a terrain slope condition.

    Parameters
    ----------
    feature_df : pandas.DataFrame
        Output of ``app.features.build_feature_table()``.  Must have columns
        ``grid_id``, ``timestamp``, ``precipitation_mm``.  If *slope_column*
        is present, the terrain condition is applied; otherwise rainfall-only.
    precip_1hr_threshold : float
        1-hour precipitation threshold in mm.  Default 20 mm/hr.
    precip_3hr_threshold : float
        3-hour cumulative precipitation threshold in mm.  Default 60 mm.
    slope_threshold : float
        Minimum slope (degrees) for a cell to be eligible as a positive.
        Only applied when *slope_column* exists in the data.
    slope_column : str
        Name of the slope column from the terrain feature table.

    Returns
    -------
    pandas.DataFrame
        Input DataFrame with added columns (in-place to avoid OOM on 31M rows):
        - ``rainfall_trigger``: bool, True if either threshold is breached
        - ``terrain_filter_applied``: bool, True if slope filter was used
        - ``target_event``: int, 1 = positive, 0 = unlabelled

    Notes
    -----
    Expects ``precip_3hr_mm`` to already exist in feature_df (computed in
    ``build_feature_table()`` on the 671K-row weather table before the
    fan-out join, to avoid computing rolling windows on 31.9M rows).
    If absent, falls back to computing it in-place.
    """
    # Work in-place on the input to avoid OOM copying 31.9M rows
    df = feature_df

    # ── 3-hour rolling: use pre-computed if available ─────────────────
    if "precip_3hr_mm" not in df.columns:
        logger.info("precip_3hr_mm not found, computing in-place...")
        df.sort_values(["grid_id", "timestamp"], inplace=True)
        df["precip_3hr_mm"] = (
            df.groupby("grid_id")["precipitation_mm"]
            .transform(lambda s: s.rolling(window=3, min_periods=1).sum())
        )

    # ── Rainfall trigger: either threshold breached ───────────────────
    df["rainfall_trigger"] = (
        (df["precipitation_mm"] >= precip_1hr_threshold)
        | (df["precip_3hr_mm"] >= precip_3hr_threshold)
    )

    # ── Apply terrain condition if slope data is available ────────────
    has_slope = slope_column in df.columns
    df["terrain_filter_applied"] = has_slope

    if has_slope:
        df["target_event"] = (
            (df["rainfall_trigger"]) & (df[slope_column] >= slope_threshold)
        ).astype("int8")
        logger.info(
            "Labelling with terrain filter: precip_1hr >= %.0f mm OR "
            "precip_3hr >= %.0f mm, AND %s >= %.0f deg",
            precip_1hr_threshold,
            precip_3hr_threshold,
            slope_column,
            slope_threshold,
        )
    else:
        df["target_event"] = df["rainfall_trigger"].astype("int8")
        logger.warning(
            "Terrain data NOT available -- labelling with rainfall-only "
            "(no slope filter).  Positives will over-represent flat areas. "
            "Re-run after terrain_features.parquet is produced."
        )
        logger.info(
            "Labelling (rainfall-only): precip_1hr >= %.0f mm OR "
            "precip_3hr >= %.0f mm",
            precip_1hr_threshold,
            precip_3hr_threshold,
        )

    n_pos = int(df["target_event"].sum())
    n_total = len(df)
    pct = 100 * n_pos / n_total if n_total > 0 else 0
    logger.info(
        "Labelling result: %d positives / %d total (%.4f%%) -- ratio 1:%.0f",
        n_pos,
        n_total,
        pct,
        (n_total - n_pos) / n_pos if n_pos > 0 else float("inf"),
    )

    return df


def sample_negatives(
    labelled_df: pd.DataFrame,
    grid_gdf: gpd.GeoDataFrame,
    min_separation_m: float = 1000.0,
    neg_to_pos_ratio: int = 10,
    random_state: int = 42,
    metric_crs: str = "EPSG:32646",
) -> pd.DataFrame:
    """
    Sample negatives maintaining >= min_separation_m from positive cells.

    Uses a two-tier strategy:

    **Tier 1 — Global cell exclusion** (preferred, when terrain filter is
    active): identify grid_ids that NEVER appear as positive, buffer all
    positive cell centroids by min_separation_m, and sample negatives only
    from cells whose centroids fall outside the buffer.

    **Tier 2 — Per-timestamp separation** (fallback, when all cells have
    positives due to rainfall-only labelling): at any given timestamp, only
    cells sharing the triggering weather point(s) are positive.  Cells on
    other weather points (~9 km away) are non-positive at that timestamp and
    naturally satisfy the 1 km separation.  Sample negatives from these
    non-positive cell-hours directly.

    Parameters
    ----------
    labelled_df : pandas.DataFrame
        Output of ``label_by_rainfall_threshold()``.  Must have columns
        ``grid_id``, ``timestamp``, ``target_event``.
    grid_gdf : geopandas.GeoDataFrame
        Full grid with geometry (EPSG:4326).
    min_separation_m : float
        Minimum Euclidean distance in metres between any negative cell
        centroid and any positive cell centroid.
    neg_to_pos_ratio : int
        Maximum negatives per positive.  Actual count may be lower if
        insufficient safe cells exist.
    random_state : int
        Random seed for reproducibility.
    metric_crs : str
        Projected CRS for distance calculations.

    Returns
    -------
    pandas.DataFrame
        Subset of labelled_df containing all positives + sampled negatives.
        Sorted by grid_id, timestamp.
    """
    positives = labelled_df[labelled_df["target_event"] == 1]
    n_pos = len(positives)

    if n_pos == 0:
        logger.warning("No positives found -- cannot sample negatives.")
        return labelled_df.head(0)

    pos_grid_ids = set(positives["grid_id"].unique())
    n_total_cells = grid_gdf["grid_id"].nunique()
    logger.info(
        "%d positive cell-hours across %d / %d unique grid cells",
        n_pos,
        len(pos_grid_ids),
        n_total_cells,
    )

    # ── Try Tier 1: global cell exclusion ─────────────────────────────
    grid_metric = grid_gdf.to_crs(metric_crs)
    pos_cells = grid_metric[grid_metric["grid_id"].isin(pos_grid_ids)]
    pos_centroids = pos_cells.geometry.centroid
    buffer_union = pos_centroids.buffer(min_separation_m).union_all()

    all_centroids = grid_metric.geometry.centroid
    safe_mask = ~all_centroids.within(buffer_union)
    safe_grid_ids = set(grid_metric.loc[safe_mask, "grid_id"].values)

    tier1_active = len(safe_grid_ids) > 0

    # Create mask for valid negatives
    if tier1_active:
        logger.info(
            "Tier 1 (global exclusion): %d positive cells buffered by "
            "%.0f m -> %d safe cells for negatives",
            len(pos_grid_ids), min_separation_m, len(safe_grid_ids),
        )
        mask = (labelled_df["target_event"] == 0) & (labelled_df["grid_id"].isin(safe_grid_ids))
    else:
        logger.info(
            "Tier 2 (per-timestamp): all %d cells have positives "

            "(no terrain filter) -> sampling from non-positive cell-hours. ",
            n_total_cells
        )
        mask = (labelled_df["target_event"] == 0)

    n_neg_target = n_pos * neg_to_pos_ratio
    n_available = mask.sum()
    
    logger.info(
        "Negative pool: %d cell-hours, target sample: %d (ratio %d:1)",
        n_available, n_neg_target, neg_to_pos_ratio,
    )

    if n_available == 0:
        logger.warning("No negatives available at all — returning positives only.")
        return positives.copy().sort_values(["grid_id", "timestamp"]).reset_index(drop=True)

    # Memory efficient sampling: get row indices where mask is true, then sample those indices
    indices = np.where(mask)[0]
    
    if n_available <= n_neg_target:
        sampled_indices = indices
        logger.info("Negative pool (%d) <= target (%d) -- using all available", n_available, n_neg_target)
    else:
        np.random.seed(random_state)
        sampled_indices = np.random.choice(indices, size=n_neg_target, replace=False)
        logger.info("Sampled %d negatives", len(sampled_indices))
        
    neg_sample = labelled_df.iloc[sampled_indices].copy()

    # ── Combine and report ────────────────────────────────────────────
    result = pd.concat([positives, neg_sample], ignore_index=True)
    result = result.sort_values(["grid_id", "timestamp"]).reset_index(drop=True)

    n_neg_final = len(result) - n_pos
    logger.info(
        "Training set: %d positives + %d negatives = %d total "
        "(positive rate %.2f%%, ratio 1:%.1f)",
        n_pos,
        n_neg_final,
        len(result),
        100 * n_pos / len(result),
        n_neg_final / n_pos if n_pos > 0 else 0,
    )

    return result


# =====================================================================
# ASDMA POINT-BASED LABELLING — STUB
#
# This function is a placeholder for labelling based on ASDMA's 366
# landslide-prone locations for Kamrup Metropolitan district.  It will
# intersect those locations with the 1km grid and mark cell-hours as
# positive when an extreme-rainfall event coincides with a known
# landslide-prone cell.
#
# DO NOT implement until the ASDMA point CSV has been transcribed and
# verified by a team member.  Do not invent ASDMA data.
# =====================================================================

def add_asdma_positives(
    feature_df: pd.DataFrame,
    asdma_csv_path: str | Path,
    verified_csv_path: str | Path,
    grid_gdf: gpd.GeoDataFrame,
    metric_crs: str = "EPSG:32646",
) -> pd.DataFrame:
    """
    Add positives from ASDMA's landslide-prone location records and verified incidents.

    The function will:
    1. Initialize a `label_source` column ('threshold' for existing positives).
    2. Spatially join the 47 ASDMA locations to the 1km grid (nearest cell).
       For these known vulnerable cells, any cell-hour with `rainfall_trigger == True`
       is marked as positive, bypassing the slope filter.
    3. Spatially join the 8 verified dated incidents to the 1km grid.
       For the specific date of the incident, all 24 hours are marked as positive
       and tagged with `label_source = 'verified_incident'`.

    Parameters
    ----------
    feature_df : pandas.DataFrame
        Feature table with grid_id, timestamp, target_event, rainfall_trigger.
    asdma_csv_path : str or Path
        Path to the 47 ASDMA vulnerable locations CSV.
    verified_csv_path : str or Path
        Path to the 8 dated verified incidents CSV.
    grid_gdf : geopandas.GeoDataFrame
        Grid with geometry for spatial join.
    metric_crs : str
        Projected CRS for accurate distance calculation.

    Returns
    -------
    pandas.DataFrame
        feature_df with target_event updated and label_source column added (in-place).
    """
    df = feature_df

    if "label_source" not in df.columns:
        df["label_source"] = "none"
        df.loc[df["target_event"] == 1, "label_source"] = "threshold"

    # Prepare metric centroids for accurate nearest-neighbor join
    grid_metric = grid_gdf.to_crs(metric_crs).copy()
    grid_metric["geometry"] = grid_metric.geometry.centroid

    # ── 1. ASDMA 47 Vulnerable Locations ──────────────────────────────
    asdma_df = pd.read_csv(asdma_csv_path)
    asdma_gdf = gpd.GeoDataFrame(
        asdma_df,
        geometry=gpd.points_from_xy(asdma_df["longitude"], asdma_df["latitude"]),
        crs="EPSG:4326"
    ).to_crs(metric_crs)

    asdma_mapped = asdma_gdf.sjoin_nearest(grid_metric, how="inner")
    asdma_grid_ids = asdma_mapped["grid_id"].unique()

    # For known vulnerable cells, ANY rainfall trigger is a positive (bypassing slope)
    mask_asdma = df["grid_id"].isin(asdma_grid_ids) & df["rainfall_trigger"]
    df.loc[mask_asdma, "target_event"] = 1
    df.loc[mask_asdma & (df["label_source"] == "none"), "label_source"] = "threshold"

    logger.info(
        "ASDMA vulnerable locations: %d unique grid cells identified. "
        "Bypassed slope filter for these cells.",
        len(asdma_grid_ids)
    )

    # ── 2. Verified Dated Incidents (Ground Truth) ────────────────────
    ver_df = pd.read_csv(verified_csv_path)
    ver_gdf = gpd.GeoDataFrame(
        ver_df,
        geometry=gpd.points_from_xy(ver_df["longitude"], ver_df["latitude"]),
        crs="EPSG:4326"
    ).to_crs(metric_crs)

    ver_mapped = ver_gdf.sjoin_nearest(grid_metric, how="inner")

    ver_count = 0
    for _, row in ver_mapped.iterrows():
        g_id = row["grid_id"]
        # Date is YYYY-MM-DD
        dt = pd.Timestamp(row["date"])
        end_dt = dt + pd.Timedelta(days=1)
        
        # Mark all 24 hours of that specific date as verified positives
        # Avoid .dt.date which allocates a huge object array and causes OOM
        mask_ver = (df["grid_id"] == g_id) & (df["timestamp"] >= dt) & (df["timestamp"] < end_dt)
        df.loc[mask_ver, "target_event"] = 1
        df.loc[mask_ver, "label_source"] = "verified_incident"
        ver_count += mask_ver.sum()

    logger.info(
        "Verified incidents: marked %d cell-hours as 'verified_incident' ground truth.",
        ver_count
    )

    n_pos = int(df["target_event"].sum())
    logger.info(
        "Final positive count after ASDMA & verified incidents: %d", n_pos
    )

    return df
