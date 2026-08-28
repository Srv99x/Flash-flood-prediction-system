# PRAVAH

**Flash Flood Prediction System for Hilly Regions using Multi-Source Data**
Team Luit · Smart India Hackathon 2026 · Problem Statement PS26192 (Ministry of Home Affairs, Disaster Management)

## Overview

PRAVAH predicts rainfall-triggered flash-flood and slope-failure risk at 1 km grid
resolution for the **Kamrup Metropolitan district** of Assam (Guwahati), and surfaces it
as an interactive risk map with a ranked early-warning list. It replaces the current
district-level, after-the-fact warning paradigm with hyper-local, proactive alerts. The
pilot grid covers **904 cells at 1 km resolution**, and risk can be rendered for any date
from 2018-01-01 to 2025-12-31.

## Architecture

Risk for each cell and date is:

```
risk[cell, date] = trigger_prob[weather_point(cell), date] * susceptibility_multiplier[cell]
```

- **Static susceptibility** is terrain-derived from 1 km mean slope, then *floored* to at
  least "High" for any cell containing an ASDMA officially identified vulnerable location
  or a verified historical incident. It is mapped to a team-assigned multiplier
  (Low 0.20 / Moderate 0.45 / High 0.70 / Very High 0.90).
- **Dynamic trigger** is a scikit-learn RandomForest (`n_estimators=100`,
  `class_weight='balanced'`) trained on five weather-point-level features —
  `soil_moisture_0_7`, `soil_moisture_7_28`, `temp_c`, `api_3d`, `api_7d` (soil moisture,
  temperature, and 3-day / 7-day Antecedent Precipitation Index). Labels come from
  rainfall intensity–duration thresholds plus verified landslide/flood incidents.
- Trigger probabilities are precomputed by `build_trigger_cache.py` so the app renders
  any date as a lookup and a multiply, with no model call at demo time.

## Setup

```bash
pip install -r requirements.txt
```

## Regenerating the data

Run from the repository root, in order:

```bash
python build_susceptibility.py     # data/processed/susceptibility_features.parquet
python build_trigger_cache.py      # data/processed/trigger_prob_daily.parquet
```

## Running the app

```bash
streamlit run app/streamlit_app.py
```

The sidebar provides a date selector, a risk-threshold slider for the warning list, and an
"Ingest Live IoT Telemetry" toggle. The main pane shows the full-width 1 km risk map and,
below it, the ranked early-warning list.

## Validated metrics

Episode-grouped train/test split (`StratifiedGroupKFold`, seed 42); see
`docs/model_training_log.md` for the split method and full results.

| Metric | RandomForest (shipped) |
|---|---|
| PR-AUC | 0.8257 |
| F1-macro | 0.7640 |

ROC-AUC is deliberately not reported — it is misleading at ~9% positive class.

## Known limitations

- **Weather resolution.** Open-Meteo's ERA5-Land archive is natively ~9 km resolution;
  weather fields are nearest-neighbour-downscaled onto the 1 km grid, not genuine 1 km
  weather.
- **Simulated IoT telemetry.** No public village-level sensor network exists in India. The
  IoT panel demonstrates the ingestion interface a real MQTT feed would drop into; the
  data is simulated and disclosed as such in the UI.
- **Team-assigned susceptibility multipliers.** The 0.20 / 0.45 / 0.70 / 0.90 weights are
  calibrated to this district's slope distribution, not taken from a published study.
- **Partial DEM coverage.** 93 of the 904 cells lack DEM coverage; they get multiplier 0.0
  and are rendered as grey "No Data", not as genuine low risk.
- **Single-district scope.** The system is built and validated for Kamrup Metropolitan
  only.
