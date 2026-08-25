"""
weather_fetch.py — Open-Meteo ERA5-Land archive fetcher for Kamrup Metropolitan.

Public API, keyless.  Endpoint: https://archive-api.open-meteo.com/v1/archive
ERA5-Land resolution: 0.1° (~11 km); temporal: hourly from 1950-01-01.

Usage example
-------------
    from app.weather_fetch import fetch_weather_for_grid

    grid_gdf = gpd.read_parquet("data/processed/grid.parquet")
    df = fetch_weather_for_grid(
        grid_gdf,
        start_date="2024-06-01",
        end_date="2024-06-30",
        cache_dir="data/processed/weather",
    )
    # df columns: grid_id, time, precip_mm, soil_moisture_0_7cm
    # (plus the API raw columns renamed for consistency)

Notes
-----
* ERA5-Land is ~0.1° grid so many 1 km cells share the same ERA5 pixel.
  We deduplicate by (rounded_lat, rounded_lon) before hitting the API and
  then broadcast back to the full grid — avoids redundant calls on the
  ~870-cell district.
* The API allows one coordinate per call.  We loop over unique ERA5 pixels
  with a small sleep to stay comfortably under the 10,000 call/day free limit.
* If a cached parquet exists for a (lat, lon, start_date, end_date) key it is
  reused without a network call (idempotent re-runs).
* API variables fetched:
    precipitation          → precip_mm       (mm h⁻¹)
    soil_moisture_0_to_7cm → soil_moisture_0_7cm (m³/m³)
  Both are available in ERA5-Land hourly.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import pandas as pd
import requests

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

# ERA5-Land is 0.1° — round coordinates to nearest 0.1° to deduplicate calls.
ERA5_RESOLUTION = 0.1  # degrees

# API variable names  →  our column names
VARIABLE_MAP: dict[str, str] = {
    "precipitation": "precip_mm",
    "soil_moisture_0_to_7cm": "soil_moisture_0_7cm",
}

REQUEST_SLEEP_S = 0.5   # seconds between API calls (conservative)
REQUEST_TIMEOUT = 30    # seconds


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _round_to_era5(val: float) -> float:
    """Round a coordinate to the nearest ERA5-Land 0.1° grid point."""
    return round(round(val / ERA5_RESOLUTION) * ERA5_RESOLUTION, 6)


def _cache_path(cache_dir: Path, lat: float, lon: float,
                start_date: str, end_date: str) -> Path:
    """Deterministic parquet filename for a single ERA5 pixel + date range."""
    lat_s = f"{lat:.1f}".replace("-", "S").replace(".", "p")
    lon_s = f"{lon:.1f}".replace("-", "W").replace(".", "p")
    fname = f"era5_{lat_s}_{lon_s}_{start_date}_{end_date}.parquet"
    return cache_dir / fname


def _fetch_single_pixel(lat: float, lon: float,
                         start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetch hourly ERA5-Land precipitation + soil moisture for one pixel.

    Returns a DataFrame with columns [time, precip_mm, soil_moisture_0_7cm].
    Raises requests.HTTPError on non-200 responses.
    """
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": ",".join(VARIABLE_MAP.keys()),
        "timezone": "Asia/Kolkata",   # IST — consistent with ASDMA records
    }
    resp = requests.get(ARCHIVE_URL, params=params, timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()
    payload = resp.json()

    hourly = payload.get("hourly", {})
    if not hourly or "time" not in hourly:
        raise ValueError(f"Unexpected API response for ({lat}, {lon}): {list(hourly.keys())}")

    df = pd.DataFrame({"time": pd.to_datetime(hourly["time"])})
    for api_var, col_name in VARIABLE_MAP.items():
        df[col_name] = hourly.get(api_var)  # None → NaN if missing variable

    return df


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fetch_weather_for_grid(
    grid_gdf,
    start_date: str,
    end_date: str,
    cache_dir: str | Path = "data/processed/weather",
    sleep_s: float = REQUEST_SLEEP_S,
) -> pd.DataFrame:
    """
    Fetch Open-Meteo ERA5-Land weather for every cell in *grid_gdf*.

    Parameters
    ----------
    grid_gdf : geopandas.GeoDataFrame
        Must have columns: grid_id, centroid_lat, centroid_lon.
        (Output of app.grid_utils.generate_grid.)
    start_date, end_date : str
        ISO-8601 dates, e.g. ``"2023-06-01"``.
    cache_dir : str or Path
        Directory where per-pixel parquet files are cached.  Created if absent.
    sleep_s : float
        Seconds to wait between API calls (default 0.5 s).

    Returns
    -------
    pandas.DataFrame
        Long-format table with columns:
        ``grid_id, time, era5_lat, era5_lon, precip_mm, soil_moisture_0_7cm``

        *precip_mm* is hourly precipitation in mm.
        *soil_moisture_0_7cm* is volumetric soil moisture in m³/m³.

    Notes
    -----
    * ERA5 resolution is 0.1° so multiple grid cells may share one ERA5 pixel.
      The function deduplicates, fetches once, then fans out to all matching cells.
    * Re-running is safe: cached pixels are not re-fetched.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    required_cols = {"grid_id", "centroid_lat", "centroid_lon"}
    if not required_cols.issubset(grid_gdf.columns):
        raise ValueError(f"grid_gdf must have columns {required_cols}; got {list(grid_gdf.columns)}")

    # ------------------------------------------------------------------
    # 1. Build mapping: ERA5 pixel (rounded lat/lon) → list of grid_ids
    # ------------------------------------------------------------------
    grid_df = grid_gdf[["grid_id", "centroid_lat", "centroid_lon"]].copy()
    grid_df["era5_lat"] = grid_df["centroid_lat"].apply(_round_to_era5)
    grid_df["era5_lon"] = grid_df["centroid_lon"].apply(_round_to_era5)

    pixel_to_cells: dict[tuple[float, float], list[str]] = {}
    for _, row in grid_df.iterrows():
        key = (row["era5_lat"], row["era5_lon"])
        pixel_to_cells.setdefault(key, []).append(row["grid_id"])

    n_unique = len(pixel_to_cells)
    logger.info(
        "Grid has %d cells → %d unique ERA5-Land pixels to fetch "
        "(date range: %s to %s)",
        len(grid_df), n_unique, start_date, end_date,
    )

    # ------------------------------------------------------------------
    # 2. Fetch / load each unique ERA5 pixel
    # ------------------------------------------------------------------
    pixel_frames: list[pd.DataFrame] = []

    for i, ((era5_lat, era5_lon), cell_ids) in enumerate(pixel_to_cells.items(), start=1):
        cache_file = _cache_path(cache_dir, era5_lat, era5_lon, start_date, end_date)

        if cache_file.exists():
            logger.debug("[%d/%d] Cache hit: %s", i, n_unique, cache_file.name)
            pixel_df = pd.read_parquet(cache_file)
        else:
            logger.info("[%d/%d] Fetching ERA5 pixel (%.1f, %.1f) …",
                        i, n_unique, era5_lat, era5_lon)
            try:
                pixel_df = _fetch_single_pixel(era5_lat, era5_lon, start_date, end_date)
            except Exception as exc:
                logger.error(
                    "  Failed for (%.1f, %.1f): %s — skipping pixel.",
                    era5_lat, era5_lon, exc,
                )
                continue

            pixel_df["era5_lat"] = era5_lat
            pixel_df["era5_lon"] = era5_lon
            pixel_df.to_parquet(cache_file, index=False)

            if i < n_unique:
                time.sleep(sleep_s)

        # Fan out to every grid cell that maps to this pixel
        for gid in cell_ids:
            cell_df = pixel_df.copy()
            cell_df.insert(0, "grid_id", gid)
            pixel_frames.append(cell_df)

    if not pixel_frames:
        raise RuntimeError(
            "No weather data fetched. Check your internet connection and date range."
        )

    result = pd.concat(pixel_frames, ignore_index=True)

    # Ensure consistent column order
    cols = ["grid_id", "time", "era5_lat", "era5_lon", "precip_mm", "soil_moisture_0_7cm"]
    result = result[[c for c in cols if c in result.columns]]

    logger.info(
        "fetch_weather_for_grid complete: %d rows, %d grid cells, date range %s–%s",
        len(result), result["grid_id"].nunique(), start_date, end_date,
    )
    return result


def add_antecedent_precip(df: pd.DataFrame,
                           windows_days: tuple[int, ...] = (3, 7)) -> pd.DataFrame:
    """
    Add Antecedent Precipitation Index (API) columns to a weather DataFrame.

    API is the rolling cumulative precipitation over the prior N days,
    per grid cell.  This is the highest-value feature in the model (CLAUDE.md §4).

    Parameters
    ----------
    df : pandas.DataFrame
        Output of fetch_weather_for_grid.  Must have columns
        [grid_id, time, precip_mm].  *time* must be hourly and monotonically
        increasing within each grid_id.
    windows_days : tuple of int
        Rolling windows in days.  Default: (3, 7) → api_3d, api_7d.

    Returns
    -------
    pandas.DataFrame
        Input DataFrame with additional columns ``api_{n}d`` for each window,
        in mm cumulative precipitation over the prior N days.
        The first N*24 hours of each cell will have NaN (insufficient history).

    Notes
    -----
    * Rolling is computed in hours (window * 24) with min_periods=1 to avoid
      all-NaN at the start of the series.  For a true API you want at least
      N days of prior data before your event window — ensure start_date is
      set N days earlier than needed.
    * Sort by grid_id + time is enforced defensively.
    """
    df = df.sort_values(["grid_id", "time"]).copy()

    for days in windows_days:
        window_h = days * 24
        col_name = f"api_{days}d"
        df[col_name] = (
            df.groupby("grid_id")["precip_mm"]
            .transform(lambda s: s.rolling(window=window_h, min_periods=1).sum())
        )

    return df
