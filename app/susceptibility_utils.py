import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

# Ordered weakest -> strongest. Used by the official-hazard floor below.
CLASS_ORDER = ["Low", "Moderate", "High", "Very High"]

# Floor applied to cells containing an officially identified hazard location.
OFFICIAL_HAZARD_FLOOR = "High"


# ── Slope-class cutoffs, in degrees ───────────────────────────────────
# These are TEAM-ASSIGNED and calibrated to Kamrup Metropolitan's own
# terrain. They are NOT taken from a cited study.
#
# Why they were changed: the previous cutoffs (15 / 30 / 45 deg) were
# sized for Himalayan terrain. Measured over the district's 811 cells
# with DEM coverage, slope_mean runs 0.00 - 31.91 deg (median 9.41,
# 95th pct 18.25). Under the old cutoffs "Very High" (>45 deg) was
# structurally unreachable, "High" (30-45 deg) caught exactly 1 cell,
# and 81% of the district collapsed into "Low" -- the classes carried
# almost no information.
#
# The cutoffs below are anchored to that measured distribution:
#   8 deg   ~ the district median (9.41) rounded down; separates the
#             Brahmaputra floodplain and central Guwahati from the
#             foothill fringe
#   15 deg  ~ the shallow-landslide slope threshold already used for
#             labelling in app/labelling.py, so "High" begins exactly
#             where a cell becomes eligible to be a positive
#   20 deg  ~ the steepest ~5% of the district (95th pct 18.25,
#             99th pct 22.39)
#
# These rank cells RELATIVE TO EACH OTHER within one district. They are
# not absolute physical hazard thresholds and should not be presented as
# transferable to other districts.
SLOPE_CUTOFF_LOW_MODERATE = 8.0
SLOPE_CUTOFF_MODERATE_HIGH = 15.0
SLOPE_CUTOFF_HIGH_VERYHIGH = 20.0


def classify_slope_susceptibility(slope_mean):
    """
    Assign a landslide-susceptibility proxy class from mean slope.

    This is NOT the official GSI NLSM classification, and the degree
    cutoffs are NOT taken from a specific cited study. They are
    team-assigned bands calibrated to this district's measured slope
    distribution -- see the module-level comment for the derivation.

    Classes (Kamrup Metropolitan calibration):
        < 8 degrees    -> Low         (floodplain / central Guwahati)
        8-15 degrees   -> Moderate    (foothill fringe)
        15-20 degrees  -> High        (hillside, landslide-eligible)
        >= 20 degrees  -> Very High   (steepest ~5% of the district)

    Known limitation: slope is a proxy for LANDSLIDE susceptibility. It
    systematically under-weights flat urban areas that flood badly
    (e.g. Anil Nagar / Zoo Road, which appear in the verified-incident
    record as urban flooding). A flood-specific layer such as TWI or
    distance-to-stream would be the correct complement.

    Parameters
    ----------
    slope_mean : float
        Mean slope of a grid cell in degrees.

    Returns
    -------
    str or None
        Proxy susceptibility class, or None for missing values.
    """
    if slope_mean is None or not np.isfinite(slope_mean):
        return None

    if slope_mean < SLOPE_CUTOFF_LOW_MODERATE:
        return "Low"
    elif slope_mean < SLOPE_CUTOFF_MODERATE_HIGH:
        return "Moderate"
    elif slope_mean < SLOPE_CUTOFF_HIGH_VERYHIGH:
        return "High"
    else:
        return "Very High"


# ══════════════════════════════════════════════════════════════════════
# OFFICIAL HAZARD FLOOR
# ══════════════════════════════════════════════════════════════════════
#
# Why this exists
# ---------------
# Mean slope over a 1 km cell is a poor proxy for landslide susceptibility
# in this district, and we can show it rather than assert it. Measured
# against the only ground truth available:
#
#     6 cells containing a documented incident : slope_mean median 6.0 deg
#     34 cells containing an ASDMA location    : slope_mean median 4.5 deg
#     the district as a whole                  : slope_mean median 9.4 deg
#
# Every documented landslide location sits in a cell that is FLATTER than
# the district average on mean slope. These are settlements at the foot of
# steep cuts: the failure happens on a local face (Bonda slope_max 29.9 deg,
# Dhirenpara 45.2 deg) that a 1 km average erases.
#
# Switching to slope_max does not fix it -- district slope_max median is
# 30.9 deg versus 30.4 deg for ASDMA cells, so it discriminates no better.
#
# So terrain alone cannot rank these cells correctly. The floor below adds
# the hazard information terrain is missing, from ASDMA's officially
# identified vulnerable locations and the verified incident record.
#
# DISCLOSURE: this means a cell can be High because it appears on an
# official hazard list, not because the model or the DEM inferred it. That
# must be stated wherever risk is presented -- it is in the Streamlit
# sidebar and in docs/model_training_log.md section 8.


def cells_containing_points(
    points_df: pd.DataFrame,
    grid_gdf: gpd.GeoDataFrame,
    lat_col: str = "latitude",
    lon_col: str = "longitude",
) -> set:
    """
    Return the set of grid_ids whose polygon contains each point.

    Parameters
    ----------
    points_df : pandas.DataFrame
        Must have latitude and longitude columns in EPSG:4326.
    grid_gdf : geopandas.GeoDataFrame
        The 1 km grid, EPSG:4326, with grid_id and geometry.

    Returns
    -------
    set of str
        grid_ids containing at least one point. Points falling outside the
        district are silently skipped -- check the returned size against
        the input size if that matters.
    """
    found = set()
    for _, row in points_df.iterrows():
        hit = grid_gdf[grid_gdf.geometry.contains(Point(row[lon_col], row[lat_col]))]
        if len(hit):
            found.add(hit.iloc[0]["grid_id"])
    return found


def apply_official_hazard_floor(
    classes: pd.Series,
    grid_ids: pd.Series,
    flagged_grid_ids: set,
    floor: str = OFFICIAL_HAZARD_FLOOR,
) -> pd.Series:
    """
    Raise the susceptibility class of officially flagged cells to *floor*.

    Cells already at or above *floor* keep their terrain-derived class, so
    the floor only ever raises, never lowers. Cells with no DEM coverage
    (class None) are also raised, since the flag is independent evidence
    that does not depend on having terrain data.

    Parameters
    ----------
    classes : pandas.Series
        Terrain-derived class per row (may contain None/NaN).
    grid_ids : pandas.Series
        grid_id per row, aligned with *classes*.
    flagged_grid_ids : set
        Cells containing an ASDMA location or a verified incident.
    floor : str
        Minimum class for flagged cells.

    Returns
    -------
    pandas.Series
        Class per row after the floor is applied.
    """
    floor_rank = CLASS_ORDER.index(floor)

    def _lift(cls, gid):
        if gid not in flagged_grid_ids:
            return cls
        if not isinstance(cls, str):          # None / NaN -> no DEM coverage
            return floor
        return cls if CLASS_ORDER.index(cls) >= floor_rank else floor

    return pd.Series(
        [_lift(c, g) for c, g in zip(classes, grid_ids)], index=classes.index
    )
