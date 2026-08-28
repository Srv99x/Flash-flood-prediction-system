"""
Per-cell plain-English explanation for the Flash Flood Risk app.

This module explains the two components of the production risk score:

    final_risk = trigger_probability * susceptibility_multiplier

The dynamic trigger comes from the existing RandomForest model and the
static susceptibility comes from the existing susceptibility parquet.

No LLM or external API is used.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import geopandas as gpd
import pandas as pd
import streamlit as st

from app.config import SUSCEPTIBILITY_MULTIPLIERS
from app.susceptibility_utils import cells_containing_points


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROCESSED_DIR = Path("data/processed")
RAW_DIR = Path("data/raw")
MODEL_PATH = Path("models/random_forest_trigger_model.pkl")


# ---------------------------------------------------------------------------
# Production susceptibility multipliers live in app/config.py (imported above).
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Production severity bands from app/streamlit_app.py
# ---------------------------------------------------------------------------

RISK_BINS = [0.0, 0.25, 0.50, 0.75, 1.01]
RISK_LABELS = ["Low", "Medium", "High", "Severe"]


MODEL_FEATURES = [
    "soil_moisture_0_7",
    "soil_moisture_7_28",
    "temp_c",
    "api_3d",
    "api_7d",
]


FEATURE_LABELS = {
    "soil_moisture_0_7": "Soil moisture at 0–7 cm",
    "soil_moisture_7_28": "Soil moisture at 7–28 cm",
    "temp_c": "Temperature",
    "api_3d": "3-day antecedent rainfall",
    "api_7d": "7-day antecedent rainfall",
}


FEATURE_UNITS = {
    "soil_moisture_0_7": "m³/m³",
    "soil_moisture_7_28": "m³/m³",
    "temp_c": "°C",
    "api_3d": "mm",
    "api_7d": "mm",
}


def get_severity(risk_score: float) -> str:
    """Return the same severity label used by the Streamlit map."""
    for upper_bound, label in zip(RISK_BINS[1:], RISK_LABELS):
        if risk_score < upper_bound:
            return label
    return RISK_LABELS[-1]


@st.cache_data(show_spinner=False)
def load_static_data():
    """
    Load static grid/mapping/susceptibility/terrain data once.

    Returns
    -------
    tuple
        missing terrain cell IDs, grid-weather mapping, susceptibility data,
        terrain data, and grid geometry.
    """
    missing_path = PROCESSED_DIR / "missing_terrain_cells.json"
    with missing_path.open("r", encoding="utf-8") as fh:
        missing_cells = set(json.load(fh))

    mapping_df = pd.read_parquet(
        PROCESSED_DIR / "grid_weather_mapping.parquet"
    ).set_index("grid_id")

    susceptibility_df = pd.read_parquet(
        PROCESSED_DIR / "susceptibility_features.parquet"
    ).set_index("grid_id")

    terrain_df = pd.read_parquet(
        PROCESSED_DIR / "terrain_features.parquet"
    ).set_index("grid_id")

    grid_gdf = gpd.read_parquet(
        PROCESSED_DIR / "kamrup_metro_grid_1km.parquet"
    )

    return (
        missing_cells,
        mapping_df,
        susceptibility_df,
        terrain_df,
        grid_gdf,
    )


@st.cache_data(show_spinner=False)
def load_hazard_cells():
    """
    Reproduce the project's existing hazard-source spatial mapping.

    Returns
    -------
    tuple[set, set]
        ASDMA grid IDs and verified-incident grid IDs.
    """
    grid_gdf = gpd.read_parquet(
        PROCESSED_DIR / "kamrup_metro_grid_1km.parquet"
    )

    asdma_df = pd.read_csv(
        RAW_DIR / "asdma_vulnerable_locations.csv"
    )

    incidents_df = pd.read_csv(
        RAW_DIR / "verified_incidents.csv"
    )

    asdma_cells = cells_containing_points(asdma_df, grid_gdf)
    incident_cells = cells_containing_points(incidents_df, grid_gdf)

    return asdma_cells, incident_cells


def get_floor_source(grid_id: str) -> str | None:
    """
    Identify the actual source of the hazard floor for a grid cell.

    Returns
    -------
    str or None
        'asdma', 'incident', 'both', or None.
    """
    asdma_cells, incident_cells = load_hazard_cells()

    in_asdma = grid_id in asdma_cells
    in_incident = grid_id in incident_cells

    if in_asdma and in_incident:
        return "both"
    if in_asdma:
        return "asdma"
    if in_incident:
        return "incident"
    return None


@st.cache_data(show_spinner=False)
def load_weather_point(weather_point_id: str) -> pd.DataFrame:
    """Load hourly history for exactly one weather point."""
    weather_path = PROCESSED_DIR / "weather_hourly.parquet"

    return pd.read_parquet(
        weather_path,
        filters=[("weather_point_id", "==", weather_point_id)],
    )


@st.cache_data(show_spinner=False)
def load_model():
    """Load the existing production RandomForest model."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}"
        )

    with MODEL_PATH.open("rb") as fh:
        return pickle.load(fh)


@st.cache_data(show_spinner=False)
def get_historical_baselines(
    weather_point_id: str,
    month: int,
) -> dict[str, float]:
    """
    Compute feature baselines from the actual 2018–2025 history.

    Baseline definition:
        same weather point
        same calendar month
        years 2018–2025
        median of hourly observations
    """
    weather_df = load_weather_point(weather_point_id).copy()

    weather_df["timestamp"] = pd.to_datetime(
        weather_df["timestamp"]
    )

    historical = weather_df[
        weather_df["timestamp"].dt.year.between(2018, 2025)
        & (weather_df["timestamp"].dt.month == month)
    ]

    if historical.empty:
        raise ValueError(
            f"No historical observations found for weather point "
            f"{weather_point_id} in calendar month {month}."
        )

    missing_features = [
        feature
        for feature in MODEL_FEATURES
        if feature not in historical.columns
    ]

    if missing_features:
        raise ValueError(
            "Missing required historical features: "
            + ", ".join(missing_features)
        )

    return {
        feature: float(historical[feature].median())
        for feature in MODEL_FEATURES
    }


def get_peak_hour_features(
    weather_point_id: str,
    date_value: pd.Timestamp,
) -> tuple[float, dict[str, float], pd.Timestamp]:
    """
    Find the exact hourly row that produced the daily peak trigger.

    The production trigger cache takes the MAX RandomForest probability for
    each weather point/day. When probabilities tie, the earliest timestamp
    is chosen deterministically here.

    Returns
    -------
    trigger_probability, feature_values, peak_timestamp
    """
    weather_df = load_weather_point(weather_point_id).copy()

    weather_df["timestamp"] = pd.to_datetime(
        weather_df["timestamp"]
    )

    day_start = date_value.normalize()
    day_end = day_start + pd.Timedelta(days=1)

    day_weather = weather_df[
        (weather_df["timestamp"] >= day_start)
        & (weather_df["timestamp"] < day_end)
    ].sort_values("timestamp")

    if day_weather.empty:
        raise ValueError(
            f"No hourly weather data for weather point "
            f"{weather_point_id} on {day_start.date()}."
        )

    missing_features = [
        feature
        for feature in MODEL_FEATURES
        if feature not in day_weather.columns
    ]

    if missing_features:
        raise ValueError(
            "Missing required model features: "
            + ", ".join(missing_features)
        )

    model = load_model()

    X = day_weather[MODEL_FEATURES]

    hourly_probabilities = model.predict_proba(X)[:, 1]

    peak_position = int(hourly_probabilities.argmax())

    peak_probability = float(
        hourly_probabilities[peak_position]
    )

    peak_row = day_weather.iloc[peak_position]

    actual_features = {
        feature: float(peak_row[feature])
        for feature in MODEL_FEATURES
    }

    peak_timestamp = pd.Timestamp(
        peak_row["timestamp"]
    )

    return (
        peak_probability,
        actual_features,
        peak_timestamp,
    )


@st.cache_data(show_spinner=False)
def get_cached_trigger_probability(
    weather_point_id: str,
    date_value: pd.Timestamp,
) -> float:
    """Read the production daily trigger probability for one WP/date."""
    cache_path = PROCESSED_DIR / "trigger_prob_daily.parquet"

    trigger_df = pd.read_parquet(
        cache_path,
        filters=[("weather_point_id", "==", weather_point_id)],
    )

    if trigger_df.empty:
        raise ValueError(
            f"No trigger-cache rows for weather point "
            f"{weather_point_id}."
        )

    trigger_df["date"] = pd.to_datetime(trigger_df["date"])

    match = trigger_df[
        trigger_df["date"] == date_value.normalize()
    ]

    if match.empty:
        raise ValueError(
            f"No cached trigger probability for weather point "
            f"{weather_point_id} on {date_value.date()}."
        )

    return float(match.iloc[0]["trigger_prob"])


def compare_features(
    actual_features: dict[str, float],
    baselines: dict[str, float],
) -> dict[str, dict]:
    """Build data-backed feature-vs-baseline comparisons."""
    comparisons = {}

    for feature in MODEL_FEATURES:
        actual = actual_features[feature]
        baseline = baselines[feature]

        difference = actual - baseline

        if difference > 0:
            direction = "higher"
        elif difference < 0:
            direction = "lower"
        else:
            direction = "equal"

        comparisons[feature] = {
            "actual": actual,
            "baseline": baseline,
            "difference": difference,
            "direction": direction,
            "label": FEATURE_LABELS[feature],
            "unit": FEATURE_UNITS[feature],
        }

    return comparisons


def format_feature_line(
    feature: str,
    comparison: dict,
) -> str:
    """Format one feature comparison in judge-friendly language."""
    actual = comparison["actual"]
    baseline = comparison["baseline"]
    direction = comparison["direction"]
    label = comparison["label"]
    unit = comparison["unit"]

    if direction == "equal":
        comparison_text = (
            f"equal to the historical monthly median "
            f"of {baseline:.2f} {unit}"
        )
    else:
        comparison_text = (
            f"{abs(comparison['difference']):.2f} {unit} "
            f"{direction} than the historical monthly median "
            f"of {baseline:.2f} {unit}"
        )

    return (
        f"{label}: {actual:.2f} {unit}; "
        f"{comparison_text}."
    )


def build_trigger_text(
    comparisons: dict[str, dict],
) -> str:
    """Build the full trigger explanation text."""
    lines = [
        "The dynamic trigger is based on these observed values "
        "at the model's peak-trigger hour:"
    ]

    for feature in MODEL_FEATURES:
        lines.append(
            format_feature_line(
                feature,
                comparisons[feature],
            )
        )

    return "\n".join(lines)


def build_summary(
    severity: str,
    trigger_probability: float,
    susceptibility_class: str,
    floor_source: str | None,
) -> str:
    """Build a concise plain-English summary."""
    risk_percent = trigger_probability * 100

    if floor_source == "asdma":
        susceptibility_reason = (
            "an officially identified ASDMA vulnerable location"
        )
    elif floor_source == "incident":
        susceptibility_reason = (
            "a verified historical incident location"
        )
    elif floor_source == "both":
        susceptibility_reason = (
            "an ASDMA vulnerable location and a verified historical incident"
        )
    else:
        susceptibility_reason = (
            f"{susceptibility_class.lower()} terrain susceptibility"
        )

    if trigger_probability == 0.0:
        return (
            "No elevated risk: the model's trigger probability is 0.0% "
            "for this date, so the final risk is 0.0."
        )

    return (
        f"{severity.upper()} RISK: This cell is affected by "
        f"{susceptibility_reason} and has a {risk_percent:.1f}% "
        f"dynamic trigger probability."
    )


def explain_cell(
    grid_id: str,
    date: str,
) -> dict:
    """
    Explain the production risk for one grid cell and date.

    Parameters
    ----------
    grid_id:
        Existing grid identifier.
    date:
        ISO date string, e.g. '2025-05-30'.

    Returns
    -------
    dict
        Structured explanation suitable for Streamlit or notebook use.
    """
    date_value = pd.Timestamp(date).normalize()

    (
        missing_cells,
        mapping_df,
        susceptibility_df,
        terrain_df,
        _grid_gdf,
    ) = load_static_data()

    # ---------------------------------------------------------------
    # No-DEM edge case
    # ---------------------------------------------------------------
    if grid_id in missing_cells:
        return {
            "grid_id": grid_id,
            "date": date_value.strftime("%Y-%m-%d"),
            "error_state": "no_dem",
            "summary": (
                "Terrain data unavailable for this cell — "
                "risk cannot be assessed."
            ),
        }

    # ---------------------------------------------------------------
    # Static mapping
    # ---------------------------------------------------------------
    if grid_id not in mapping_df.index:
        raise ValueError(
            f"Grid ID {grid_id} not found in grid-weather mapping."
        )

    if grid_id not in susceptibility_df.index:
        raise ValueError(
            f"Grid ID {grid_id} not found in susceptibility features."
        )

    if grid_id not in terrain_df.index:
        raise ValueError(
            f"Grid ID {grid_id} not found in terrain features."
        )

    weather_point_id = mapping_df.loc[
        grid_id,
        "weather_point_id",
    ]

    susceptibility_row = susceptibility_df.loc[grid_id]

    susceptibility_class = susceptibility_row[
        "gsi_susceptibility_class"
    ]

    if pd.isna(susceptibility_class):
        raise ValueError(
            f"Susceptibility class is missing for {grid_id}."
        )

    susceptibility_class = str(
        susceptibility_class
    )

    if susceptibility_class not in SUSCEPTIBILITY_MULTIPLIERS:
        raise ValueError(
            f"Unknown susceptibility class: "
            f"{susceptibility_class}"
        )

    susceptibility_multiplier = SUSCEPTIBILITY_MULTIPLIERS[
        susceptibility_class
    ]

    hazard_floor_applied = bool(
        susceptibility_row["hazard_floor_applied"]
    )

    slope_mean = float(
        terrain_df.loc[grid_id, "slope_mean"]
    )

    # ---------------------------------------------------------------
    # Trigger probability + exact peak-hour features
    # ---------------------------------------------------------------
    cached_trigger_probability = get_cached_trigger_probability(
        str(weather_point_id),
        date_value,
    )

    (
        peak_trigger_probability,
        actual_features,
        peak_timestamp,
    ) = get_peak_hour_features(
        str(weather_point_id),
        date_value,
    )

    # This must agree with the production cache.
    if abs(
        peak_trigger_probability - cached_trigger_probability
    ) > 1e-6:
        raise ValueError(
            "Production trigger mismatch: "
            f"hourly max={peak_trigger_probability:.10f}, "
            f"cached={cached_trigger_probability:.10f}."
        )

    trigger_probability = cached_trigger_probability

    # ---------------------------------------------------------------
    # Historical baselines
    # ---------------------------------------------------------------
    baselines = get_historical_baselines(
        str(weather_point_id),
        date_value.month,
    )

    comparisons = compare_features(
        actual_features,
        baselines,
    )

    # ---------------------------------------------------------------
    # Final risk
    # ---------------------------------------------------------------
    final_risk_score = (
        trigger_probability
        * susceptibility_multiplier
    )

    severity = get_severity(
        final_risk_score
    )

    # ---------------------------------------------------------------
    # Hazard-floor source
    # ---------------------------------------------------------------
    floor_source = (
        get_floor_source(grid_id)
        if hazard_floor_applied
        else None
    )

    if hazard_floor_applied and floor_source is None:
        raise ValueError(
            f"{grid_id} is marked hazard_floor_applied=True, "
            "but no ASDMA/incident spatial source was found."
        )

    # ---------------------------------------------------------------
    # Susceptibility explanation
    # ---------------------------------------------------------------
    if hazard_floor_applied:
        if floor_source == "asdma":
            source_text = (
                "an officially identified ASDMA vulnerable location"
            )
        elif floor_source == "incident":
            source_text = (
                "a verified historical incident"
            )
        elif floor_source == "both":
            source_text = (
                "an officially identified ASDMA vulnerable location "
                "and a verified historical incident"
            )
        else:
            source_text = "an identified hazard source"

        susceptibility_text = (
            f"This cell contains {source_text}, so its susceptibility "
            f"is floored to {susceptibility_class} regardless of "
            f"average slope."
        )
    else:
        susceptibility_text = (
            f"This cell averages {slope_mean:.1f} degrees slope, "
            f"placing it in the {susceptibility_class} "
            "susceptibility band."
        )

    # ---------------------------------------------------------------
    # Trigger explanation
    # ---------------------------------------------------------------
    trigger_text = build_trigger_text(
        comparisons
    )

    summary = build_summary(
        severity=severity,
        trigger_probability=trigger_probability,
        susceptibility_class=susceptibility_class,
        floor_source=floor_source,
    )

    return {
        "grid_id": grid_id,
        "date": date_value.strftime("%Y-%m-%d"),
        "error_state": None,
        "weather_point_id": str(weather_point_id),
        "peak_timestamp": peak_timestamp.isoformat(),
        "trigger_prob": trigger_probability,
        "susceptibility_class": susceptibility_class,
        "susceptibility_multiplier": susceptibility_multiplier,
        "hazard_floor_applied": hazard_floor_applied,
        "floor_source": floor_source,
        "slope_mean": slope_mean,
        "final_risk_score": final_risk_score,
        "severity": severity,
        "actual_features": actual_features,
        "baselines": baselines,
        "feature_comparisons": comparisons,
        "susceptibility_text": susceptibility_text,
        "trigger_text": trigger_text,
        "summary": summary,
    }