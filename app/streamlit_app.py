"""
app/streamlit_app.py
────────────────────
Flash Flood Early Warning System — Interactive Demo
SIH 2026 | Kamrup Metropolitan District, Assam

Run with:
    streamlit run app/streamlit_app.py

What is real and what is simulated
─────────────────────────────────
REAL       Risk scores, for any date from 2018-01-01 to 2025-12-31.
             risk[cell, date] = trigger_prob[weather_point(cell), date]
                                * susceptibility_multiplier[cell]
           Trigger probabilities come from the RandomForest model via the
           precomputed cache (build_trigger_cache.py, 55,518 rows); the
           susceptibility layer is terrain-derived and floored by ASDMA's
           officially identified vulnerable locations
           (build_susceptibility.py). The cache builder asserts this
           matches app/predict.py to 1e-6 on all 904 cells.
SIMULATED  IoT sensor telemetry (app/mqtt_sim.py). No public
           village-level sensor network exists; the panel demonstrates
           the ingestion interface a real feed would drop into. This is
           disclosed in the UI and must stay disclosed.

Regenerating after a model or multiplier change
───────────────────────────────────────────────
    python build_susceptibility.py     # susceptibility classes
    python build_trigger_cache.py      # trigger probability cache

Files owned by this module:
    app/streamlit_app.py  ← this file
    app/mqtt_sim.py       ← simulated sensor telemetry

Do NOT import or modify:
    app/grid_utils.py, app/weather_fetch.py, app/terrain_utils.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.explain import explain_cell

import json
import time
import os
import sys

from datetime import date

import numpy as np
import pandas as pd
import geopandas as gpd
import folium
import streamlit as st
from streamlit_folium import st_folium

# mqtt_sim lives in the same app/ directory
sys.path.insert(0, os.path.dirname(__file__))
from mqtt_sim import get_sensor_readings  # noqa: E402

# ══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="PRAVAH — Kamrup Metro Early Warning",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════
GUWAHATI_LAT = 26.05
GUWAHATI_LON = 91.70
DEFAULT_ZOOM = 11
GRID_PARQUET = os.path.join(
    os.path.dirname(__file__), "..", "data", "processed",
    "kamrup_metro_grid_1km.parquet"
)

RISK_BINS   = [0.0,  0.25,  0.50,  0.75, 1.01]
RISK_COLORS = ["#C8F7C5", "#FFF176", "#FF8C00", "#C62828"]
RISK_LABELS = ["Low", "Medium", "High", "Severe"]

# Opens on the deadliest documented event in the record.
DEFAULT_DATE = date(2025, 5, 30)

# One-click jumps. Each is a documented event or a reference condition —
# see data/raw/verified_incidents.csv.
DEMO_DATES = [
    ("30 May 2025 — Bonda landslide (5 deaths)", date(2025, 5, 30)),
    ("17 Jun 2023 — Dhirenpara landslide (1 death)", date(2023, 6, 17)),
    ("26 May 2020 — wettest hour on record", date(2020, 5, 26)),
    ("15 Jan 2020 — dry season (contrast)", date(2020, 1, 15)),
]


# ══════════════════════════════════════════════════════════════════════════════
# RISK LOADING
# ══════════════════════════════════════════════════════════════════════════════

# Minimum fraction of grid cells that must resolve to a risk score.
# A synthetic placeholder grid was once committed whose grid_id scheme
# ("KM_0001") did not match the rest of the pipeline ("KM_R000_C028"), so the
# join silently matched 0 of 904 rows and fillna() painted the whole map 0.0
# risk for days. Fail loudly instead of rendering a lie.
MIN_RISK_MATCH_FRACTION = 0.50

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
TRIGGER_CACHE = os.path.join(DATA_DIR, "trigger_prob_daily.parquet")
MAPPING_PARQUET = os.path.join(DATA_DIR, "grid_weather_mapping.parquet")
SUSCEPTIBILITY_PARQUET = os.path.join(DATA_DIR, "susceptibility_features.parquet")
MISSING_CELLS_JSON = os.path.join(DATA_DIR, "missing_terrain_cells.json")

# Susceptibility class -> static multiplier.  Imported from app.predict so the
# app and the offline pipeline can never disagree about the risk scale.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.predict import SUSCEPTIBILITY_MULTIPLIERS  # noqa: E402


# ══════════════════════════════════════════════════════════════════════════════
# STATIC LAYER — grid geometry + per-cell susceptibility.  Loaded once.
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def load_static() -> gpd.GeoDataFrame:
    """
    Grid geometry joined to weather point and susceptibility multiplier.

    Everything here is date-independent, so it is loaded exactly once per
    session; only the trigger probability changes when the date changes.
    """
    gdf = gpd.read_parquet(GRID_PARQUET)
    mapping = pd.read_parquet(MAPPING_PARQUET, columns=["grid_id", "weather_point_id"])
    sus = pd.read_parquet(SUSCEPTIBILITY_PARQUET)

    n_before = len(gdf)
    gdf = gdf.merge(mapping, on="grid_id", how="left")
    gdf = gdf.merge(sus, on="grid_id", how="left")

    matched = int(gdf["weather_point_id"].notna().sum())
    if matched / n_before < MIN_RISK_MATCH_FRACTION:
        msg = "\n".join([
            f"GRID/WEATHER JOIN FAILED — only {matched} of {n_before} cells "
            f"({matched/n_before:.1%}) matched a weather point.",
            "",
            f"  grid_id in grid file:    {list(gdf['grid_id'].head(2))}",
            f"  grid_id in mapping file: {list(mapping['grid_id'].head(2))}",
            "",
            "The files use different grid_id schemes. Regenerate the grid with "
            "app.grid_utils.generate_grid() against "
            "data/raw/boundaries/kamrup_metropolitan.geojson.",
        ])
        st.error(msg)
        raise RuntimeError(msg)

    gdf["susceptibility_mult"] = (
        gdf["gsi_susceptibility_class"].map(SUSCEPTIBILITY_MULTIPLIERS).fillna(0.0)
    )

    with open(MISSING_CELLS_JSON, "r") as fh:
        gdf["no_dem"] = gdf["grid_id"].isin(json.load(fh))

    return gdf


@st.cache_data(show_spinner=False)
def load_trigger_cache() -> pd.DataFrame:
    """
    Daily-peak trigger probability per (weather_point_id, date).

    55,518 rows covering 2018-2025 — small enough to hold in memory, so any
    date in range renders as a lookup and a multiply with no model call.
    Built by build_trigger_cache.py.
    """
    df = pd.read_parquet(TRIGGER_CACHE)
    df["date"] = pd.to_datetime(df["date"])
    return df


@st.cache_data(show_spinner=False)
def available_date_range() -> tuple:
    """First and last date present in the trigger cache."""
    cache = load_trigger_cache()
    return cache["date"].min().date(), cache["date"].max().date()


# ══════════════════════════════════════════════════════════════════════════════
# RISK FOR A GIVEN DATE
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def risk_for_date(target_date) -> pd.DataFrame:
    """
    Per-cell risk for one date:  trigger probability x susceptibility.

    Identical to app.predict.predict_risk() on that date's peak hour — the
    cache builder asserts the two agree to 1e-6 across all 904 cells.
    Cached per date, so revisiting a date is instant.
    """
    static = load_static()
    cache = load_trigger_cache()

    day = cache[cache["date"] == pd.Timestamp(target_date)]
    if day.empty:
        return pd.DataFrame(
            {"grid_id": static["grid_id"], "trigger_prob": np.nan, "risk": np.nan}
        )

    trig = static["weather_point_id"].map(
        day.set_index("weather_point_id")["trigger_prob"]
    )
    return pd.DataFrame({
        "grid_id": static["grid_id"],
        "trigger_prob": trig.to_numpy(),
        "risk": (trig * static["susceptibility_mult"]).to_numpy(),
    })


def build_display_gdf(target_date) -> gpd.GeoDataFrame:
    """Static grid + this date's risk, with severity bands and No-Data cells."""
    gdf = load_static().copy()
    gdf["risk_probability"] = risk_for_date(target_date)["risk"].to_numpy()
    gdf["risk_pct"] = (gdf["risk_probability"] * 100).round(1)

    gdf["severity"] = pd.cut(
        gdf["risk_probability"], bins=RISK_BINS, labels=RISK_LABELS, right=False
    )
    gdf["severity"] = gdf["severity"].cat.add_categories(["No Data"])

    # Cells with no DEM coverage have no susceptibility class, so their risk is
    # undefined rather than low — render grey, not green.
    gdf.loc[gdf["no_dem"], "severity"] = "No Data"
    gdf.loc[gdf["no_dem"], ["risk_probability", "risk_pct"]] = np.nan

    return gdf


def get_risk_color(risk: float) -> str:
    """Map a risk value [0,1] to its display hex colour."""
    if pd.isna(risk):
        return "#808080" # 5th category for No Data
    for i, threshold in enumerate(RISK_BINS[1:]):
        if risk < threshold:
            return RISK_COLORS[i]
    return RISK_COLORS[-1]


# ══════════════════════════════════════════════════════════════════════════════
# MAP BUILDER
# Not cached with @st.cache_data — folium.Map contains lambdas that
# can't be pickled by Streamlit's cache. The heavy work (grid I/O + risk
# scoring) is cached in load_static(); map build is ~1 s from memory.
# ══════════════════════════════════════════════════════════════════════════════

def build_folium_map(gdf: gpd.GeoDataFrame, threshold: float) -> folium.Map:
    """
    Build the Folium choropleth map of all 904 grid cells.

    Uses a single GeoJson FeatureCollection — style properties are stored
    inside each feature's properties dict so the style_function lambda
    captures nothing from the outer scope (no closure = no pickle issue).
    """
    m = folium.Map(
        location=[GUWAHATI_LAT, GUWAHATI_LON],
        zoom_start=DEFAULT_ZOOM,
        tiles=None,
        control_scale=True,
    )

    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Dark_Gray_Base/MapServer/tile/{z}/{y}/{x}",
        attr=(
            'Tiles &copy; Esri &mdash; Esri, HERE, Garmin, FAO, NOAA, USGS'
        ),
        name="Esri Dark Gray",
        max_zoom=16,
    ).add_to(m)

    # Build a FeatureCollection with style props embedded in each feature
    features = []
    for _, row in gdf.iterrows():
        risk          = float(row["risk_probability"])
        fill_color    = get_risk_color(risk)
        fill_opacity  = 0.50
        border_color  = "#FF4444" if risk >= threshold else "#555555"
        border_weight = 2.0 if risk >= threshold else 0.3

        features.append({
            "type": "Feature",
            "geometry": row["geometry"].__geo_interface__,
            "properties": {
                "grid_id":      row["grid_id"],
                "risk_pct":     float(row["risk_pct"]),
                "severity":     str(row["severity"]),
                "lat":          float(row["centroid_lat"]),
                "lon":          float(row["centroid_lon"]),
                # Pre-computed style — lambda below reads these, captures nothing
                "fillColor":    fill_color,
                "fillOpacity":  fill_opacity,
                "color":        border_color,
                "weight":       border_weight,
            },
        })

    folium.GeoJson(
        {"type": "FeatureCollection", "features": features},
        style_function=lambda feat: {
            "fillColor":   feat["properties"]["fillColor"],
            "color":       feat["properties"]["color"],
            "weight":      feat["properties"]["weight"],
            "fillOpacity": feat["properties"]["fillOpacity"],
        },
        tooltip=folium.GeoJsonTooltip(
            fields=["grid_id", "risk_pct", "severity"],
            aliases=["Grid ID", "Risk %", "Severity"],
            localize=True,
            sticky=True,
        ),
        popup=folium.GeoJsonPopup(
            fields=["grid_id", "risk_pct", "severity", "lat", "lon"],
            aliases=["Grid ID", "Risk %", "Severity", "Lat", "Lon"],
            max_width=240,
        ),
        name="Risk Grid",
    ).add_to(m)

    # Legend overlay
    m.get_root().html.add_child(folium.Element("""
    <div style="position:fixed;bottom:30px;left:30px;z-index:9999;
                background:rgba(20,20,30,0.88);padding:12px 16px;
                border-radius:8px;border:1px solid #334;
                font-family:monospace;font-size:12px;color:#eee;">
      <b style="color:#4fc3f7;">RISK LEVEL</b><br>
      <span style="background:#C8F7C5;padding:2px 8px;">&nbsp;</span>&nbsp;&lt;25% &mdash; Low<br>
      <span style="background:#FFF176;padding:2px 8px;">&nbsp;</span>&nbsp;25&ndash;50% &mdash; Medium<br>
      <span style="background:#FF8C00;padding:2px 8px;">&nbsp;</span>&nbsp;50&ndash;75% &mdash; High<br>
      <span style="background:#C62828;padding:2px 8px;">&nbsp;</span>&nbsp;&gt;75% &mdash; Severe<br>
      <span style="background:#808080;padding:2px 8px;">&nbsp;</span>&nbsp;No DEM data<br>
      <hr style="border-color:#445;margin:6px 0;">
      <span style="color:#888;font-size:10px;">RandomForest trigger &times; susceptibility<br>
      (terrain + ASDMA official hazard list)</span>
    </div>
    """))

    return m


# ══════════════════════════════════════════════════════════════════════════════
# CUSTOM CSS
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #0d1117 0%, #0f1a2e 60%, #0d1117 100%);
}
[data-testid="stSidebar"] {
    background: rgba(10, 20, 40, 0.95) !important;
    border-right: 1px solid #1e3a5f;
}
[data-testid="stMetric"] {
    background: rgba(14, 40, 80, 0.6);
    border: 1px solid #1e4a80;
    border-radius: 10px;
    padding: 12px 16px;
}
[data-testid="stMetricValue"] { color: #4fc3f7 !important; font-weight: 700; }
[data-testid="stMetricLabel"] { color: #90caf9 !important; }
h2 { color: #4fc3f7 !important; border-bottom: 1px solid #1e4a80; padding-bottom: 6px; }
h3 { color: #81d4fa !important; }
.sidebar-caption { color: #607d8b; font-size: 11px; font-style: italic; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## 🌊 PRAVAH")
    st.markdown("**Kamrup Metro District Early Warning**")
    st.markdown("*SIH 2026 — Flash Flood Prediction System*")
    st.divider()

    st.markdown("### 📅 Forecast Date")
    _min_date, _max_date = available_date_range()

    # A jump button on the previous run leaves a pending date here.  It must
    # be applied BEFORE the date_input widget is created — Streamlit forbids
    # writing a widget's session_state key after the widget exists.
    if "forecast_date" not in st.session_state:
        st.session_state.forecast_date = DEFAULT_DATE
    if "_pending_date" in st.session_state:
        st.session_state.forecast_date = st.session_state.pop("_pending_date")

    forecast_date = st.date_input(
        "Select date",
        min_value=_min_date,
        max_value=_max_date,
        key="forecast_date",
        label_visibility="collapsed",
    )
    st.caption(
        f"Any date from {_min_date:%d %b %Y} to {_max_date:%d %b %Y}. "
        "The map recomputes on change."
    )

    st.markdown("**Jump to a documented event**")
    for _label, _d in DEMO_DATES:
        if st.button(_label, use_container_width=True, key=f"demo_{_d.isoformat()}"):
            st.session_state["_pending_date"] = _d
            st.rerun()

    st.divider()

    st.markdown("### ⚠️ Alert Threshold")
    threshold = st.slider(
        "Risk threshold for alerts",
        min_value=0.0, max_value=1.0, value=0.60, step=0.05, format="%.2f",
        help="Cells with risk above this value appear in the warning table.",
    )
    st.caption(f"Cells with risk ≥ **{int(threshold*100)}%** trigger warnings.")

    st.divider()

    st.markdown("### 📡 IoT Telemetry")
    iot_active = st.toggle(
        "Ingest Live IoT Telemetry", value=False,
        help="Show live sensor readings from 5 simulated field stations.",
    )
    st.markdown(
        '<p class="sidebar-caption">⚠ Sensor data is simulated for this demonstration. '
        "A live MQTT ingestion pipeline will connect to real sensors when deployed.</p>",
        unsafe_allow_html=True,
    )

    st.divider()
    st.markdown("### 📖 How to read this map")
    st.markdown(
        '<p class="sidebar-caption">'
        "<b>Risk = trigger × susceptibility.</b><br><br>"
        "<b>Trigger</b> is real model output — a RandomForest trained on "
        "ERA5-Land soil moisture and antecedent precipitation, labelled from "
        "rainfall intensity–duration thresholds plus 7 verified "
        "landslide/flood incidents (2022–2025).<br><br>"
        "<b>Susceptibility is terrain-derived, floored by ASDMA's officially "
        "identified vulnerable locations.</b> That means a cell can show high "
        "risk because it appears on Assam's official vulnerable-locations "
        "list, not only because the terrain model inferred it. We do this "
        "because 1 km mean slope alone ranks these cells wrongly: every "
        "documented landslide site in the district sits in a cell that is "
        "flatter than the district average, since the failure happens on a "
        "local hill cut a 1 km average erases.<br><br>"
        "Susceptibility multipliers (0.20 / 0.45 / 0.70 / 0.90) are "
        "team-assigned weights calibrated to this district's slope "
        "distribution — not values from a published study."
        "</p>",
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# MAIN PANE — HEADER
# ══════════════════════════════════════════════════════════════════════════════

col_title, col_date = st.columns([3, 1])
with col_title:
    st.markdown("## 🗺️ Flash Flood Risk Map — Kamrup Metro")
with col_date:
    st.markdown(f"<br><b>Forecast:</b> {forecast_date}", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# LOAD DATA
# ══════════════════════════════════════════════════════════════════════════════

with st.spinner("Loading grid and computing risk …"):
    gdf = build_display_gdf(forecast_date)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN PANE — MAP
# ══════════════════════════════════════════════════════════════════════════════

with st.spinner("Rendering risk map …"):
    flood_map = build_folium_map(gdf, threshold)

st_folium(
    flood_map,
    use_container_width=True,
    height=520,
    returned_objects=[],
)

# ══════════════════════════════════════════════════════════════════════════════
# WHY IS THIS CELL AT RISK? — EXPLAINABILITY PANEL
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("### 🔎 Why is this cell at risk?")

# The current Folium map intentionally does not return click events
# (returned_objects=[]), so use the task-approved Grid ID selector fallback.
valid_gdf = gdf[gdf["severity"] != "No Data"].copy()

if valid_gdf.empty:
    st.info("No valid terrain cells are available for explanation on this date.")
else:
    # Automatically select the highest-risk valid cell.
    highest_risk_grid = (
        valid_gdf.sort_values("risk_probability", ascending=False)
        .iloc[0]["grid_id"]
    )

    grid_options = gdf["grid_id"].tolist()

    selected_grid = st.selectbox(
        "Explain a grid cell",
        options=grid_options,
        index=(
            grid_options.index(highest_risk_grid)
            if highest_risk_grid in grid_options
            else 0
        ),
        help=(
            "Select any grid cell to see why its current risk is "
            "Low, Medium, High, or Severe."
        ),
    )

    try:
        explanation = explain_cell(
            selected_grid,
            forecast_date.strftime("%Y-%m-%d"),
        )
    except Exception as exc:
        st.error(
            f"Explanation failed for {selected_grid} on {forecast_date:%Y-%m-%d}: {exc}"
        )
        st.stop()

    if explanation.get("error_state") == "no_dem":
        st.warning(explanation["summary"])
    else:
        st.info(f"**{explanation['summary']}**")

        metric1, metric2, metric3, metric4 = st.columns(4)

        metric1.metric(
            "Final Risk",
            f"{explanation['final_risk_score']:.1%}",
        )

        metric2.metric(
            "Trigger Probability",
            f"{explanation['trigger_prob']:.1%}",
        )

        metric3.metric(
            "Susceptibility",
            explanation["susceptibility_class"],
        )

        metric4.metric(
            "Severity",
            explanation["severity"],
        )

        st.markdown(
            '<div style="color:#90caf9; font-size:18px; font-weight:700; margin-top:18px;">'
            '1. Static susceptibility — why this place is vulnerable'
            '</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            f'<div style="color:#e0e0e0; font-size:16px; line-height:1.7; margin-top:8px;">'
            f'{explanation["susceptibility_text"]}'
            f'</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            '<div style="color:#90caf9; font-size:18px; font-weight:700; margin-top:20px;">'
            '2. Dynamic trigger — what is happening now'
            '</div>',
            unsafe_allow_html=True,
        )

        trigger_html = explanation["trigger_text"].replace("\n", "<br>")

        st.markdown(
            f'<div style="color:#e0e0e0; font-size:16px; line-height:1.7; margin-top:8px;">'
            f'{trigger_html}'
            f'</div>',
            unsafe_allow_html=True,
        )

        st.caption(
            f"Risk calculation: "
            f"{explanation['trigger_prob']:.1%} trigger probability × "
            f"{explanation['susceptibility_multiplier']:.2f} susceptibility "
            f"multiplier = {explanation['final_risk_score']:.1%} final risk."
        )

st.caption(
    "**Risk = RandomForest trigger probability × susceptibility multiplier.** "
    "Susceptibility is terrain-derived (SRTM slope) and **floored by ASDMA's "
    "officially identified vulnerable locations** — 34 of 904 cells are raised "
    "to at least *High* on that basis. Multipliers are team-assigned weights "
    "calibrated to this district's slope distribution, not values from a "
    "published study.  "
    "Bands: 🟢 Low (<25%) · 🟡 Medium (25–50%) · 🟠 High (50–75%) · 🔴 Severe (>75%) · "
    "⬜ No Data (93 cells outside DEM coverage)."
)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN PANE — METRICS + WARNING TABLE
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("---")
st.markdown("## ⚠️ Active Early Warnings")

above_thresh = gdf[gdf["risk_probability"] >= threshold]
top10        = above_thresh.nlargest(10, "risk_probability")

m1, m2, m3 = st.columns(3)
m1.metric("🛰️ Total Cells Monitored", f"{len(gdf):,}")
m2.metric(
    "🚨 Cells Above Threshold",
    f"{len(above_thresh):,}",
    delta=f"≥ {int(threshold*100)}% risk",
    delta_color="inverse",
)
m3.metric("🔺 Highest Risk", f"{gdf['risk_probability'].max() * 100:.1f}%")

st.markdown(f"**Top 10 highest-risk cells** above {int(threshold*100)}% threshold")

if top10.empty:
    st.info(
        f"✅ No cells exceed the {int(threshold*100)}% threshold. "
        "Lower the slider to see warning candidates."
    )
else:
    display_df = top10[["grid_id", "centroid_lat", "centroid_lon", "risk_pct", "severity"]].copy()
    display_df.columns = ["Grid ID", "Lat", "Lon", "Risk %", "Severity"]
    display_df["Lat"] = display_df["Lat"].round(4)
    display_df["Lon"] = display_df["Lon"].round(4)

    def _sev_color(val):
        return {
            "Low":    "color: #a5d6a7",
            "Medium": "color: #fff176",
            "High":   "color: #ffb74d",
            "Severe": "color: #ef5350",
        }.get(str(val), "")

    styled = (
        display_df.style
        # Styler.applymap was removed in pandas 3.0 — .map is the replacement
        .map(_sev_color, subset=["Severity"])
        .format({"Risk %": "{:.1f}"})
        .set_properties(**{"background-color": "rgba(10,20,40,0.6)", "color": "#e0e0e0"})
        .set_table_styles([
            {"selector": "th", "props": [("background-color", "#1a3a6b"), ("color", "#90caf9")]},
        ])
    )
    st.dataframe(styled, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# IOT TELEMETRY PANEL
# ══════════════════════════════════════════════════════════════════════════════

if iot_active:
    st.markdown("---")
    st.markdown(
        "## 📡 Live Sensor Telemetry\n"
        '<span style="color:#4db6ac;font-size:12px;letter-spacing:0.08em;">'
        "⚠ SIMULATED SENSOR TELEMETRY — DEMONSTRATION OF LIVE INGESTION CAPABILITY"
        "</span>",
        unsafe_allow_html=True,
    )
    st.caption(
        "5 virtual IoT sensors placed in hilly northern grid cells. "
        "Readings drift realistically every 3 seconds. "
        "In production this panel consumes a live MQTT feed via paho-mqtt."
    )

    sensor_placeholder = st.empty()
    refresh_label      = st.empty()

    for i in range(600):   # safety cap ~30 min
        readings = get_sensor_readings()

        with sensor_placeholder.container():
            cols = st.columns(len(readings))
            for col, r in zip(cols, readings):
                with col:
                    cell_risk = gdf.loc[gdf["grid_id"] == r["grid_id"], "risk_probability"]
                    crv = float(cell_risk.values[0]) if len(cell_risk) else 0.0
                    st.markdown(
                        f"**{r['sensor_id']}**  \n<small>{r['label']}</small>",
                        unsafe_allow_html=True,
                    )
                    st.metric("🌧 Rainfall",     f"{r['rainfall_mm_hr']} mm/hr")
                    st.metric("🌱 Soil Moisture", f"{r['soil_moisture_pct']}%")
                    st.metric("💧 Water Level",   f"{r['water_level_m']} m")
                    st.caption(r["timestamp"])

        refresh_label.caption(
            f"Last refresh: {time.strftime('%H:%M:%S')}  |  Update #{i+1}"
        )
        time.sleep(3)
