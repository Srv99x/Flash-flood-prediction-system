"""
app/mqtt_sim.py
───────────────
Simulated IoT sensor telemetry for the Flash Flood Prediction System demo.

This module DOES NOT connect to a real MQTT broker. It generates realistic,
drifting sensor readings in-process for 5 virtual sensors placed in hilly /
northern grid cells of the Kamrup Metro grid.

Why simulate?
  - The real sensor network does not yet exist.
  - Judges can see the live-ingestion architecture without infrastructure.
  - Every widget that displays this data is labelled
    "SIMULATED SENSOR TELEMETRY — demonstration of live ingestion capability"

When the real sensors and MQTT broker arrive:
  1. Replace get_sensor_readings() with a real paho-mqtt subscriber.
  2. Keep the SENSOR_NODES dict to map sensor_id → grid_id.

paho-mqtt is imported for structural credibility only.
The broker connection is mocked — we do not start a real broker.
"""

import time
import math
import random
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional

# paho-mqtt imported for structural demo; broker is MOCKED (not started)
try:
    import paho.mqtt.client as mqtt  # noqa: F401  (imported, not used in demo)
    _PAHO_AVAILABLE = True
except ImportError:
    _PAHO_AVAILABLE = False

# ── Sensor node configuration ──────────────────────────────────────────────────
# 5 virtual sensors placed on the 5 steepest well-separated grid cells in the
# "Very High" susceptibility band.
#
# grid_id, lat and lon below are REAL values read from
# data/processed/kamrup_metro_grid_1km.parquet (cell centroids) and
# data/processed/terrain_features.parquet (slope, elevation) — they are not
# invented. Labels describe the cell's measured terrain rather than naming a
# landmark: the previous version attached landmark names to fabricated
# coordinates that fell outside the district entirely.
#
# The telemetry VALUES these nodes emit are still simulated (see _BASELINES).
SENSOR_NODES: Dict[str, Dict[str, Any]] = {
    "S001": {"grid_id": "KM_R011_C021", "lat": 26.1070, "lon": 91.8501,
             "label": "Hill site · 31.9° · 179 m"},
    "S002": {"grid_id": "KM_R009_C047", "lat": 26.0866, "lon": 92.1059,
             "label": "Hill site · 23.7° · 360 m"},
    "S003": {"grid_id": "KM_R005_C043", "lat": 26.0502, "lon": 92.0662,
             "label": "Hill site · 23.3° · 443 m"},
    "S004": {"grid_id": "KM_R007_C023", "lat": 26.0634, "lon": 91.8710,
             "label": "Hill site · 23.2° · 126 m"},
    "S005": {"grid_id": "KM_R019_C022", "lat": 26.1751, "lon": 91.8551,
             "label": "Hill site · 22.7° · 231 m"},
}

# ── Baseline environmental parameters per sensor ───────────────────────────────
# Realistic July–August monsoon baselines for Assam.
_BASELINES: Dict[str, Dict[str, float]] = {
    "S001": {"rainfall": 12.5, "soil_moisture": 72.0, "water_level": 1.8},
    "S002": {"rainfall":  8.0, "soil_moisture": 68.0, "water_level": 1.2},
    "S003": {"rainfall": 15.0, "soil_moisture": 78.0, "water_level": 2.1},
    "S004": {"rainfall":  6.5, "soil_moisture": 65.0, "water_level": 0.9},
    "S005": {"rainfall": 11.0, "soil_moisture": 74.0, "water_level": 1.6},
}

# Period (seconds) for the slow sinusoidal drift — makes readings feel alive
_DRIFT_PERIOD = 120  # 2-minute cycle


def _drift(base: float, amplitude: float, t: float, phase: float, noise_scale: float) -> float:
    """
    Return a realistically drifting value.

    Uses a slow sinusoidal trend to simulate gradual environmental change,
    plus small Gaussian noise to avoid perfectly smooth curves.

    Args:
        base:        Centre value (physical baseline).
        amplitude:   Peak swing around the base.
        t:           Current time in seconds (time.time()).
        phase:       Per-sensor phase offset to de-correlate sensors.
        noise_scale: Standard deviation of additive Gaussian noise.

    Returns:
        Clipped float ≥ 0.
    """
    trend = amplitude * math.sin(2 * math.pi * t / _DRIFT_PERIOD + phase)
    noise = random.gauss(0, noise_scale)
    return max(0.0, base + trend + noise)


def get_sensor_readings(t: Optional[float] = None) -> List[Dict[str, Any]]:
    """
    Return the latest simulated sensor reading for all 5 sensor nodes.

    TEMPORARY — demonstration of live ingestion capability.
    Replace the body of this function with a real paho-mqtt subscriber
    when the physical sensor network is operational.

    Args:
        t: Unix timestamp to evaluate readings at. Defaults to now.

    Returns:
        List of dicts, one per sensor:
            sensor_id, grid_id, label, lat, lon,
            timestamp (ISO-8601 UTC),
            rainfall_mm_hr, soil_moisture_pct, water_level_m
    """
    if t is None:
        t = time.time()

    readings = []
    for i, (sid, node) in enumerate(SENSOR_NODES.items()):
        phase = i * (2 * math.pi / len(SENSOR_NODES))  # spread sensors across the cycle
        bl = _BASELINES[sid]

        rainfall = round(_drift(bl["rainfall"],    amplitude=4.0, t=t, phase=phase,       noise_scale=0.3), 2)
        soil     = round(_drift(bl["soil_moisture"], amplitude=5.0, t=t, phase=phase + 1.0, noise_scale=0.5), 1)
        water    = round(_drift(bl["water_level"],  amplitude=0.4, t=t, phase=phase + 2.0, noise_scale=0.05), 3)

        # Clamp physical limits
        soil = min(soil, 100.0)

        readings.append({
            "sensor_id":        sid,
            "grid_id":          node["grid_id"],
            "label":            node["label"],
            "lat":              node["lat"],
            "lon":              node["lon"],
            "timestamp":        datetime.fromtimestamp(t, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
            "rainfall_mm_hr":   rainfall,
            "soil_moisture_pct": soil,
            "water_level_m":    water,
        })

    return readings


# ── Mock MQTT publisher (structural demo) ──────────────────────────────────────
# This section demonstrates the publish/subscribe architecture that would be used
# with a real broker (e.g., Eclipse Mosquitto on the server).
# The broker is NOT started; these functions are never called by the app.

MQTT_BROKER   = "localhost"
MQTT_PORT     = 1883
MQTT_TOPIC    = "guwahati/flood/sensors/#"


def _on_connect(client, userdata, flags, rc):  # noqa: D401
    """MQTT on-connect callback (structural demo — not executed)."""
    client.subscribe(MQTT_TOPIC)


def _on_message(client, userdata, msg):  # noqa: D401
    """MQTT on-message callback (structural demo — not executed)."""
    # In production: parse msg.payload, push to database / state store
    pass


def build_mock_mqtt_client():
    """
    Build a paho-mqtt client wired to the demo callbacks.

    STRUCTURAL DEMO — do not call .connect() in demo mode.
    When a real broker is available:
        client = build_mock_mqtt_client()
        client.connect(MQTT_BROKER, MQTT_PORT, 60)
        client.loop_start()
    """
    if not _PAHO_AVAILABLE:
        return None
    client = mqtt.Client(client_id="flood-demo-subscriber")
    client.on_connect = _on_connect
    client.on_message = _on_message
    return client
