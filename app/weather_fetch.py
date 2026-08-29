"""
Open-Meteo ERA5-Land weather fetcher.

IMPORTANT RESOLUTION NOTE:
Open-Meteo's ERA5-Land archive is natively ~9km resolution. Our analysis grid is
1km. Fetching weather per 1km grid cell would issue ~80x more API calls than
necessary and would misrepresent 9km data as 1km-resolution weather.

Instead: a coarse set of weather sample points (~9km spacing) is generated over
the district's bounding box, fetched once each, and every 1km grid cell is
assigned to its nearest weather point (see assign_grid_to_weather_points).
Downstream users of this data must treat precipitation/soil-moisture/temperature
as ~9km-resolution fields, nearest-neighbour-downscaled onto the 1km grid — not
as genuine 1km-resolution weather.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import requests

logger = logging.getLogger(__name__)

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
HOURLY_VARS = "precipitation,soil_moisture_0_to_7cm,soil_moisture_7_to_28cm,temperature_2m"
TIMEZONE = "Asia/Kolkata"

# Every hourly field the model / feature pipeline consumes downstream.
# "time" is the index; the other four are the raw inputs to
# app/features.py -> RandomForest in build_trigger_cache.py.
REQUIRED_HOURLY_KEYS = (
    "time",
    "precipitation",
    "soil_moisture_0_to_7cm",
    "soil_moisture_7_to_28cm",
    "temperature_2m",
)


class WeatherDataError(RuntimeError):
    """
    Raised when an Open-Meteo archive response is missing a required hourly
    field or contains null values in one.

    WHY THIS EXISTS
    ---------------
    fetch_weather_point() builds a DataFrame directly from the response.
    If Open-Meteo returns HTTP 200 but omits a requested variable, or
    returns a value array with None entries (which happens at the archive
    edge and for occasional soil-moisture gaps), the resulting column is
    silently NaN. That NaN then flows unchecked through
    app/features.build_feature_table() into RandomForest.predict_proba()
    in build_trigger_cache.py and yields a confident-looking but
    meaningless trigger probability — and, from there, a wrong risk score
    on the demo map.

    Silent NaN propagation into the RandomForest predictions was flagged
    as a demo risk, so we validate the payload and fail loudly here.
    Do NOT catch this except at a UI boundary (show it via st.error()).
    """


def _validate_hourly_payload(data, weather_point_id, lat, lon, start_date, end_date):
    """
    Check that an Open-Meteo archive response carries every required hourly
    field, fully populated: no missing keys, no None entries, and every
    value array the same length as the ``time`` axis.

    Raises
    ------
    WeatherDataError
        If any required field is missing or contains null values. The
        message names the exact field(s) and the weather point / coords /
        date range being fetched, so a bad fetch can be traced immediately
        rather than surfacing as a NaN column much later in the pipeline.
    """
    where = (
        f"weather point {weather_point_id} ({lat:.4f}, {lon:.4f}), "
        f"{start_date}..{end_date}"
    )

    hourly = data.get("hourly") if isinstance(data, dict) else None
    if not isinstance(hourly, dict):
        raise WeatherDataError(
            f"Open-Meteo response for {where} has no 'hourly' block "
            f"(top-level keys: {sorted(data) if isinstance(data, dict) else type(data)})."
        )

    missing_keys = [k for k in REQUIRED_HOURLY_KEYS if k not in hourly]
    if missing_keys:
        raise WeatherDataError(
            f"Open-Meteo response for {where} is missing required hourly "
            f"field(s): {', '.join(missing_keys)}."
        )

    time_axis = hourly["time"]
    n_time = len(time_axis) if time_axis is not None else 0
    if n_time == 0:
        raise WeatherDataError(
            f"Open-Meteo response for {where} has an empty 'time' axis."
        )

    problems = []
    for key in REQUIRED_HOURLY_KEYS:
        values = hourly[key]
        if values is None:
            problems.append(f"{key} (entire array is null)")
            continue
        if len(values) != n_time:
            problems.append(
                f"{key} (length {len(values)} != {n_time} timestamps)"
            )
            continue
        n_null = sum(1 for v in values if v is None)
        if n_null:
            problems.append(f"{key} ({n_null}/{n_time} values null)")

    if problems:
        raise WeatherDataError(
            f"Open-Meteo response for {where} has missing/null values in: "
            + "; ".join(problems)
            + ". Refusing to return NaN weather into the feature pipeline."
        )


def generate_weather_points(boundary_gdf, spacing_km=9, metric_crs="EPSG:32646"):
    """
    Generate a coarse grid of weather sample points spaced ~spacing_km apart,
    covering the bounding box of boundary_gdf.

    Parameters
    ----------
    boundary_gdf : geopandas.GeoDataFrame
        District boundary (or any geometry) whose bounding box the points
        should cover. Must have a CRS set.
    spacing_km : float
        Spacing between sample points, in kilometres.
    metric_crs : str
        Projected CRS used to compute spacing accurately. Default EPSG:32646
        (UTM zone 46N).

    Returns
    -------
    geopandas.GeoDataFrame
        Columns: weather_point_id, lat, lon, geometry (Point, EPSG:4326).
    """
    if boundary_gdf.crs is None:
        raise ValueError("boundary_gdf must have a CRS set")

    spacing_m = spacing_km * 1000
    boundary_metric = boundary_gdf.to_crs(metric_crs)
    minx, miny, maxx, maxy = boundary_metric.total_bounds

    n_cols = int(np.ceil((maxx - minx) / spacing_m))
    n_rows = int(np.ceil((maxy - miny) / spacing_m))
    n_cols = max(n_cols, 1)
    n_rows = max(n_rows, 1)

    point_ids = []
    xs = []
    ys = []
    for row in range(n_rows):
        y = miny + (row + 0.5) * spacing_m
        for col in range(n_cols):
            x = minx + (col + 0.5) * spacing_m
            point_ids.append(f"WP_R{row:02d}_C{col:02d}")
            xs.append(x)
            ys.append(y)

    points_metric = gpd.GeoDataFrame(
        {"weather_point_id": point_ids},
        geometry=gpd.points_from_xy(xs, ys),
        crs=metric_crs,
    )
    points_4326 = points_metric.to_crs("EPSG:4326")
    points_4326["lat"] = points_4326.geometry.y
    points_4326["lon"] = points_4326.geometry.x

    return points_4326[["weather_point_id", "lat", "lon", "geometry"]]


def fetch_weather_point(
    weather_point_id,
    lat,
    lon,
    start_date,
    end_date,
    cache_dir,
    max_retries=5,
    backoff_base=2.0,
    timeout_s=60,
    request_delay_s=1.5,
):
    """
    Fetch hourly precipitation, soil moisture, and temperature for one point
    over [start_date, end_date] from the Open-Meteo archive API.

    Uses a local parquet cache keyed by weather_point_id: if cached data
    already exists, the API is not called again.

    Returns
    -------
    (pandas.DataFrame, bool)
        Tidy dataframe with columns weather_point_id, lat, lon, timestamp,
        precipitation_mm, soil_moisture_0_7, soil_moisture_7_28, temp_c,
        and a bool indicating whether an API call was actually made
        (False = served from cache).
    """
    cache_dir = Path(cache_dir)
    cache_path = cache_dir / f"{weather_point_id}.parquet"

    if cache_path.exists():
        return pd.read_parquet(cache_path), False

    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": HOURLY_VARS,
        "timezone": TIMEZONE,
    }

    last_exc = None
    data = None
    for attempt in range(max_retries):
        try:
            resp = requests.get(ARCHIVE_URL, params=params, timeout=timeout_s)
            resp.raise_for_status()
            data = resp.json()
            break
        except requests.RequestException as exc:
            last_exc = exc
            if attempt < max_retries - 1:
                time.sleep(backoff_base ** attempt)

    if data is None:
        raise RuntimeError(
            f"Failed to fetch weather point {weather_point_id} after {max_retries} attempts"
        ) from last_exc

    # Validate the response BEFORE parsing / caching it. A 200 response can
    # still be missing a variable or carry null entries; parsed naively those
    # become silent NaN columns that poison the RandomForest trigger
    # predictions downstream (see WeatherDataError docstring). Raise loudly
    # instead, naming the field(s) and the point/date. This does not touch
    # retry, caching, or the API call itself — only what happens after the
    # payload is in hand — and a failed response is never written to cache.
    _validate_hourly_payload(
        data, weather_point_id, lat, lon, start_date, end_date
    )

    hourly = data["hourly"]
    df = pd.DataFrame(
        {
            "weather_point_id": weather_point_id,
            "lat": lat,
            "lon": lon,
            "timestamp": pd.to_datetime(hourly["time"]),
            "precipitation_mm": hourly["precipitation"],
            "soil_moisture_0_7": hourly["soil_moisture_0_to_7cm"],
            "soil_moisture_7_28": hourly["soil_moisture_7_to_28cm"],
            "temp_c": hourly["temperature_2m"],
        }
    )

    cache_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache_path)

    time.sleep(request_delay_s)

    return df, True


def fetch_all_weather_points(points_gdf, start_date, end_date, cache_dir):
    """
    Fetch (or load from cache) hourly weather for every point in points_gdf.

    Returns
    -------
    (pandas.DataFrame, dict)
        Concatenated tidy long-format weather table, and a stats dict with
        n_points, n_api_calls, n_cache_hits, total_rows, wall_time_seconds.
    """
    start_time = time.time()
    frames = []
    n_api_calls = 0
    n_cache_hits = 0

    for row in points_gdf.itertuples():
        df, called_api = fetch_weather_point(
            row.weather_point_id, row.lat, row.lon, start_date, end_date, cache_dir
        )
        frames.append(df)
        if called_api:
            n_api_calls += 1
        else:
            n_cache_hits += 1

    weather_df = pd.concat(frames, ignore_index=True)

    stats = {
        "n_points": len(points_gdf),
        "n_api_calls": n_api_calls,
        "n_cache_hits": n_cache_hits,
        "total_rows": len(weather_df),
        "wall_time_seconds": time.time() - start_time,
    }

    return weather_df, stats


def add_antecedent_precip(df, windows_days=(3, 7)):
    """
    Add Antecedent Precipitation Index (API) columns to the weather-point table.

    Computed per weather_point_id, on the ~9km weather-point table BEFORE the
    grid join — each weather point has ~40+ grid cells assigned to it (see
    assign_grid_to_weather_points), so computing the rolling sum after joining
    onto grid_id would recompute identical values that many times over.

    Parameters
    ----------
    df : pandas.DataFrame
        Output of fetch_all_weather_points. Must have columns
        [weather_point_id, timestamp, precipitation_mm].
    windows_days : tuple of int
        Rolling windows in days. Default: (3, 7) -> api_3d, api_7d.

    Returns
    -------
    pandas.DataFrame
        Input DataFrame with additional columns ``api_{n}d`` for each window,
        in mm cumulative precipitation over the prior N days per weather point.
        The first N*24 hours of each point's series will reflect a
        shorter-than-N-day window (min_periods=1), not NaN.
    """
    df = df.sort_values(["weather_point_id", "timestamp"]).copy()

    for days in windows_days:
        window_h = days * 24
        col_name = f"api_{days}d"
        df[col_name] = (
            df.groupby("weather_point_id")["precipitation_mm"]
            .transform(lambda s: s.rolling(window=window_h, min_periods=1).sum())
        )

    return df


def assign_grid_to_weather_points(grid_gdf, points_gdf, metric_crs="EPSG:32646"):
    """
    Assign each 1km grid cell to its nearest weather sample point.

    Parameters
    ----------
    grid_gdf : geopandas.GeoDataFrame
        Must have grid_id, centroid_lat, centroid_lon columns.
    points_gdf : geopandas.GeoDataFrame
        Output of generate_weather_points (weather_point_id, lat, lon, geometry).
    metric_crs : str
        Projected CRS used for accurate nearest-neighbour distance.

    Returns
    -------
    pandas.DataFrame
        Columns: grid_id, weather_point_id, distance_m.
    """
    grid_points = gpd.GeoDataFrame(
        {"grid_id": grid_gdf["grid_id"].values},
        geometry=gpd.points_from_xy(grid_gdf["centroid_lon"], grid_gdf["centroid_lat"]),
        crs="EPSG:4326",
    ).to_crs(metric_crs)

    points_metric = points_gdf.to_crs(metric_crs)

    joined = gpd.sjoin_nearest(
        grid_points, points_metric[["weather_point_id", "geometry"]], distance_col="distance_m"
    )

    return joined[["grid_id", "weather_point_id", "distance_m"]].reset_index(drop=True)


# =============================================================================
# DEPRECATED — superseded 2026-08-26 by the weather-point design above.
#
# This was the original per-grid-cell fetcher (dedup by rounding each grid
# cell's centroid to the nearest 0.1 degree ERA5-Land pixel, then fanning the
# fetched series back out to every grid_id that shares a pixel). It is kept
# here, unused, because the team may need to explain to judges why the design
# changed: ERA5-Land is natively ~9-11km resolution, and fanning identical
# weather values out across all ~904 1km grid cells overstates how much
# independent information the weather source actually provides at 1km.
# Do not import or call these — use fetch_all_weather_points() above instead.
# =============================================================================

# ERA5-Land is 0.1 degree - round coordinates to nearest 0.1 degree grid point.
ERA5_RESOLUTION_DEPRECATED = 0.1  # degrees

# API variable names -> our column names
VARIABLE_MAP_DEPRECATED: dict[str, str] = {
    "precipitation": "precip_mm",
    "soil_moisture_0_to_7cm": "soil_moisture_0_7cm",
}

REQUEST_SLEEP_S_DEPRECATED = 0.5   # seconds between API calls (conservative)
REQUEST_TIMEOUT_DEPRECATED = 30    # seconds


def _round_to_era5_deprecated(val: float) -> float:
    """Round a coordinate to the nearest ERA5-Land 0.1 degree grid point."""
    return round(round(val / ERA5_RESOLUTION_DEPRECATED) * ERA5_RESOLUTION_DEPRECATED, 6)


def _cache_path_deprecated(cache_dir: Path, lat: float, lon: float,
                            start_date: str, end_date: str) -> Path:
    """Deterministic parquet filename for a single ERA5 pixel + date range."""
    lat_s = f"{lat:.1f}".replace("-", "S").replace(".", "p")
    lon_s = f"{lon:.1f}".replace("-", "W").replace(".", "p")
    fname = f"era5_{lat_s}_{lon_s}_{start_date}_{end_date}.parquet"
    return cache_dir / fname


def _fetch_single_pixel_deprecated(lat: float, lon: float,
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
        "hourly": ",".join(VARIABLE_MAP_DEPRECATED.keys()),
        "timezone": "Asia/Kolkata",   # IST - consistent with ASDMA records
    }
    resp = requests.get(ARCHIVE_URL, params=params, timeout=REQUEST_TIMEOUT_DEPRECATED)
    resp.raise_for_status()
    payload = resp.json()

    hourly = payload.get("hourly", {})
    if not hourly or "time" not in hourly:
        raise ValueError(f"Unexpected API response for ({lat}, {lon}): {list(hourly.keys())}")

    # NOTE: this DEPRECATED path still uses hourly.get(api_var), which turns a
    # missing variable into a silent NaN column. It is intentionally left as-is
    # because nothing imports or calls it (see the module banner above). The
    # ACTIVE fetcher, fetch_weather_point(), runs _validate_hourly_payload()
    # and raises WeatherDataError instead. If this path is ever revived, route
    # it through that validator too.
    df = pd.DataFrame({"time": pd.to_datetime(hourly["time"])})
    for api_var, col_name in VARIABLE_MAP_DEPRECATED.items():
        df[col_name] = hourly.get(api_var)  # None -> NaN if missing variable

    return df


def fetch_weather_for_grid_deprecated(
    grid_gdf,
    start_date: str,
    end_date: str,
    cache_dir: str | Path = "data/processed/weather",
    sleep_s: float = REQUEST_SLEEP_S_DEPRECATED,
) -> pd.DataFrame:
    """
    DEPRECATED — see module note above. Use fetch_all_weather_points() instead.

    Fetch Open-Meteo ERA5-Land weather for every cell in *grid_gdf*.

    Parameters
    ----------
    grid_gdf : geopandas.GeoDataFrame
        Must have columns: grid_id, centroid_lat, centroid_lon.
        (Output of app.grid_utils.generate_grid.)
    start_date, end_date : str
        ISO-8601 dates, e.g. ``"2023-06-01"``.
    cache_dir : str or Path
        Directory where per-pixel parquet files are cached. Created if absent.
    sleep_s : float
        Seconds to wait between API calls (default 0.5 s).

    Returns
    -------
    pandas.DataFrame
        Long-format table with columns:
        ``grid_id, time, era5_lat, era5_lon, precip_mm, soil_moisture_0_7cm``

    Notes
    -----
    * ERA5 resolution is 0.1 degree so multiple grid cells may share one ERA5
      pixel. The function deduplicates, fetches once, then fans out to all
      matching cells.
    * Re-running is safe: cached pixels are not re-fetched.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    required_cols = {"grid_id", "centroid_lat", "centroid_lon"}
    if not required_cols.issubset(grid_gdf.columns):
        raise ValueError(f"grid_gdf must have columns {required_cols}; got {list(grid_gdf.columns)}")

    # ------------------------------------------------------------------
    # 1. Build mapping: ERA5 pixel (rounded lat/lon) -> list of grid_ids
    # ------------------------------------------------------------------
    grid_df = grid_gdf[["grid_id", "centroid_lat", "centroid_lon"]].copy()
    grid_df["era5_lat"] = grid_df["centroid_lat"].apply(_round_to_era5_deprecated)
    grid_df["era5_lon"] = grid_df["centroid_lon"].apply(_round_to_era5_deprecated)

    pixel_to_cells: dict[tuple[float, float], list[str]] = {}
    for _, row in grid_df.iterrows():
        key = (row["era5_lat"], row["era5_lon"])
        pixel_to_cells.setdefault(key, []).append(row["grid_id"])

    n_unique = len(pixel_to_cells)
    logger.info(
        "Grid has %d cells -> %d unique ERA5-Land pixels to fetch "
        "(date range: %s to %s)",
        len(grid_df), n_unique, start_date, end_date,
    )

    # ------------------------------------------------------------------
    # 2. Fetch / load each unique ERA5 pixel
    # ------------------------------------------------------------------
    pixel_frames: list[pd.DataFrame] = []

    for i, ((era5_lat, era5_lon), cell_ids) in enumerate(pixel_to_cells.items(), start=1):
        cache_file = _cache_path_deprecated(cache_dir, era5_lat, era5_lon, start_date, end_date)

        if cache_file.exists():
            logger.debug("[%d/%d] Cache hit: %s", i, n_unique, cache_file.name)
            pixel_df = pd.read_parquet(cache_file)
        else:
            logger.info("[%d/%d] Fetching ERA5 pixel (%.1f, %.1f) ...",
                        i, n_unique, era5_lat, era5_lon)
            try:
                pixel_df = _fetch_single_pixel_deprecated(era5_lat, era5_lon, start_date, end_date)
            except Exception as exc:
                logger.error(
                    "  Failed for (%.1f, %.1f): %s - skipping pixel.",
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

    cols = ["grid_id", "time", "era5_lat", "era5_lon", "precip_mm", "soil_moisture_0_7cm"]
    result = result[[c for c in cols if c in result.columns]]

    logger.info(
        "fetch_weather_for_grid_deprecated complete: %d rows, %d grid cells, date range %s-%s",
        len(result), result["grid_id"].nunique(), start_date, end_date,
    )
    return result


def add_antecedent_precip_per_cell_deprecated(df: pd.DataFrame,
                                               windows_days: tuple[int, ...] = (3, 7)) -> pd.DataFrame:
    """
    DEPRECATED — see module note above. Use add_antecedent_precip() instead,
    which computes the same rolling sums once per weather_point_id instead of
    once per grid_id (avoiding ~40x redundant computation on identical data).

    Add Antecedent Precipitation Index (API) columns to a weather DataFrame.

    API is the rolling cumulative precipitation over the prior N days,
    per grid cell. This is the highest-value feature in the model (CLAUDE.md §4).

    Parameters
    ----------
    df : pandas.DataFrame
        Output of fetch_weather_for_grid_deprecated. Must have columns
        [grid_id, time, precip_mm]. *time* must be hourly and monotonically
        increasing within each grid_id.
    windows_days : tuple of int
        Rolling windows in days. Default: (3, 7) -> api_3d, api_7d.

    Returns
    -------
    pandas.DataFrame
        Input DataFrame with additional columns ``api_{n}d`` for each window,
        in mm cumulative precipitation over the prior N days.
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
