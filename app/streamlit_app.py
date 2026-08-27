"""
app/streamlit_app.py
────────────────────
Flash Flood Early Warning System — Interactive Demo
SIH 2026 | Kamrup Metropolitan District, Assam

Run with:
    streamlit run app/streamlit_app.py

Architecture note
─────────────────
All risk scores are currently SYNTHETIC (see generate_dummy_risk).
When the trained model ships, replace ONLY that function with a call to
app/predict.py — nothing else in this file needs to change.

Files owned by this module:
    app/streamlit_app.py  ← this file
    app/mqtt_sim.py       ← simulated sensor telemetry

Do NOT import or modify:
    app/grid_utils.py, app/weather_fetch.py, app/terrain_utils.py
"""

import time
import os
import sys

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
    page_title="FloodSense — Kamrup Metro Early Warning",
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

# Hotspot centroids — used only by generate_dummy_risk()
_HOTSPOTS = [
    (26.40, 91.52),
    (26.38, 91.70),
    (26.35, 91.85),
    (26.42, 91.60),
]


# ══════════════════════════════════════════════════════════════════════════════
# RISK GENERATION  — TEMPORARY, clearly isolated for easy swap
# ══════════════════════════════════════════════════════════════════════════════

def generate_dummy_risk(grid_gdf: gpd.GeoDataFrame) -> np.ndarray:
    """
    Generate spatially coherent fake risk probabilities.

    # TEMPORARY — replaced by real model predictions in app/predict.py
    # To swap: return predict.predict_risk(grid_gdf) here.

    Strategy: lat gradient (north = higher risk) + hotspot falloff + noise.
    """
    np.random.seed(7)
    lats = grid_gdf["centroid_lat"].values
    lons = grid_gdf["centroid_lon"].values

    lat_min, lat_max = lats.min(), lats.max()
    lat_range = lat_max - lat_min if lat_max > lat_min else 1e-6
    lat_score = (lats - lat_min) / lat_range

    hotspot_score = np.zeros(len(grid_gdf))
    for h_lat, h_lon in _HOTSPOTS:
        dist = np.sqrt((lats - h_lat) ** 2 + (lons - h_lon) ** 2)
        sigma = 0.12
        hotspot_score += np.exp(-(dist ** 2) / (2 * sigma ** 2))
    hs_max = hotspot_score.max()
    if hs_max > 0:
        hotspot_score = 0.5 * hotspot_score / hs_max

    combined = 0.35 * lat_score + 0.50 * hotspot_score
    noise    = np.random.normal(0, 0.04, size=len(grid_gdf))
    return np.clip(combined + noise, 0.0, 1.0).astype(np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING — cached so the grid + risk scores load ONCE per session
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data
def load_grid() -> gpd.GeoDataFrame:
    """Load the Kamrup Metro grid and attach dummy risk scores. Cached once."""
    gdf = gpd.read_parquet(GRID_PARQUET)
    gdf["risk_probability"] = generate_dummy_risk(gdf)
    gdf["risk_pct"]  = (gdf["risk_probability"] * 100).round(1)
    gdf["severity"]  = pd.cut(
        gdf["risk_probability"],
        bins=RISK_BINS,
        labels=RISK_LABELS,
        right=False,
    )
    return gdf


def get_risk_color(risk: float) -> str:
    """Map a risk value [0,1] to its display hex colour."""
    for i, threshold in enumerate(RISK_BINS[1:]):
        if risk < threshold:
            return RISK_COLORS[i]
    return RISK_COLORS[-1]


# ══════════════════════════════════════════════════════════════════════════════
# MAP BUILDER
# Not cached with @st.cache_data — folium.Map contains lambdas that
# can't be pickled by Streamlit's cache. The heavy work (grid I/O + risk
# scoring) is cached in load_grid(); map build is ~1 s from memory.
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
        tiles="https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png",
        attr=(
            '&copy; <a href="https://www.openstreetmap.org/copyright">OSM</a>'
            ' &copy; <a href="https://carto.com/">CARTO</a>'
        ),
        name="Dark Mode",
        max_zoom=19,
    ).add_to(m)

    # Build a FeatureCollection with style props embedded in each feature
    features = []
    for _, row in gdf.iterrows():
        risk          = float(row["risk_probability"])
        fill_color    = get_risk_color(risk)
        fill_opacity  = 0.0 if risk < 0.25 else 0.50
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
      <hr style="border-color:#445;margin:6px 0;">
      <span style="color:#888;font-size:10px;">&#9888; Risk scores are SIMULATED</span>
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
    st.markdown("## 🌊 FloodSense")
    st.markdown("**Kamrup Metro District Early Warning**")
    st.markdown("*SIH 2026 — Flash Flood Prediction System*")
    st.divider()

    st.markdown("### 📅 Forecast Date")
    forecast_date = st.date_input("Select date", label_visibility="collapsed")
    st.caption("Date selection is ready to wire into the model pipeline.")

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
    st.markdown(
        '<p class="sidebar-caption">Risk scores are currently <b>synthetic</b>. '
        "Swap <code>generate_dummy_risk()</code> with <code>predict.predict_risk()</code> "
        "when the trained model is ready.</p>",
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

with st.spinner("Loading grid data …"):
    gdf = load_grid()


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

st.caption(
    "⚠️ **SIMULATED RISK DATA** — Synthetic risk scores used for demonstration.  "
    "Colour bands: 🟢 Low (<25%) · 🟡 Medium (25–50%) · 🟠 High (50–75%) · 🔴 Severe (>75%)"
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
        .applymap(_sev_color, subset=["Severity"])
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
