# CLAUDE.md — Project Context

> This file is the single source of truth for this repository.
> Read it fully before any task. Do not contradict the LOCKED DECISIONS section.

---

## 1. What we are building

**Smart India Hackathon 2026 — Problem Statement PS26192**
"Flash Flood Prediction System for Hilly Regions using Multi-Source Data"
Ministry of Home Affairs · Software category · Disaster Management theme.

**One-line description:** A machine-learning system that predicts flash-flood / rainfall-triggered slope-failure risk at 1km grid resolution for Kamrup Metropolitan district (Guwahati, Assam), surfaced as an interactive risk map with an early-warning list — replacing the current district-level, too-late warning paradigm with hyper-local, proactive alerts.

**Pilot district (LOCKED):** Kamrup Metropolitan, Assam. Approx bounding box `[91.50, 26.00, 92.10, 26.30]`, centre ~26.14°N, 91.74°E.
⚠️ **Kamrup Metropolitan ≠ Kamrup (rural).** These are two different districts. Every filter, query and download must target **Kamrup Metropolitan**.

## 2. Hard deadline and judging constraints

- **Internal college round: 31 August 2026.** Code freeze 30 Aug, 12:00 PM.
- Max **10 minutes** pitch. Max **12 slides**.
- A working prototype/demo carries **explicit added advantage**.
- Judged on: feasibility, innovation, technical implementation, scalability, real-world impact.
- Judges are mixed-discipline college faculty, **not** geospatial ML specialists. Optimise explanations accordingly.

## 3. Team reality (this constrains everything)

- **3 members build.** Python/ML capable.
- **3 members assist only.** Tech background but not tool-fluent. Suitable for: data transcription, source verification, slide drafting, rehearsal. **Not** for independent coding or debugging.
- Therefore: **prefer the boring, working solution over the elegant one every single time.** Any task that requires learning a new framework mid-week must be rejected or downgraded.

## 4. LOCKED DECISIONS — do not re-litigate these

| Area | Decision | Why |
|---|---|---|
| Language | Python 3 | — |
| Core libs | pandas, numpy, geopandas, rasterio, scikit-learn, xgboost, shap, streamlit, folium, matplotlib | Already in `requirements.txt` |
| Model family | scikit-learn RandomForest (baseline) → XGBoost (final) | Small tabular data + need for explainability. **Deep learning is rejected** — would overfit and destroy interpretability |
| Class imbalance | XGBoost `scale_pos_weight` **only** | SMOTE/ADASYN rejected: synthesising fake landslide locations on spatially autocorrelated geodata is methodologically weak and hard to defend |
| Metrics | **PR-AUC + F1-macro.** Explicitly reject ROC-AUC | ~2% positive class makes ROC-AUC misleading. This is a credibility point in the pitch |
| Validation | Spatial block CV. **Fallback: strict temporal split** (train earlier years / test most recent) | Random k-fold leaks via spatial autocorrelation. Do not burn >3 hours on spatial-kfold; take the fallback |
| Rainfall + soil moisture | **Open-Meteo ERA5-Land archive API** | Keyless, no login, hourly, verifiably covers 26.14°N. Collapses two data sources into one HTTP call |
| Terrain / DEM | **Local SRTM/CartoDEM + rasterio** (already working) | Do NOT switch to Google Earth Engine. GEE adds GCP auth + new mental model for zero necessary gain |
| Landslide susceptibility | GSI NLSM, treated as **1:50,000 macro-scale** | See §6 — meso-scale coverage for Guwahati is UNVERIFIED |
| Common grid | 1km × 1km vector grid over Kamrup Metro (~870 cells) | Balance of hyper-local relevance vs compute |
| Common timestep | 1 hour | Flash floods are sub-daily events |
| Real-time IoT | **Simulated** via MQTT (paho-mqtt), openly declared as simulated | No public village-level sensor network exists in India. Architected so real sensors drop into the same ingestion interface |
| Output | Per-grid-cell risk class + interactive Folium map + early-warning list | — |

### Target feature table

```
grid_id, timestamp,
precip_1hr, precip_24h_cum, api_3d, api_7d,   # Open-Meteo
soil_moisture_0_7cm,                           # Open-Meteo
slope_mean, elevation_mean, twi_max,           # SRTM DEM derived
dist_to_stream,                                # optional
gsi_susceptibility,                            # GSI NLSM, categorical
target_event                                   # binary label
```

### Feature priority (build in this order, cut from the bottom)
1. Antecedent Precipitation Index (3d, 7d rolling) — highest value, lowest effort
2. Slope + elevation — high value, already built
3. TWI — high value, medium effort
4. Soil moisture — medium value, low effort (free with the rainfall call)
5. Distance to stream — medium value, medium effort
6. ~~Plan/profile curvature~~ — **CUT**. Low value, high effort

## 5. Labelling strategy (hybrid — this is the crux)

Historical event records for one district are sparse. Use a hybrid:

1. **Empirical positives:** ASDMA's 366 landslide-prone locations for Kamrup Metro, intersected with documented extreme-rainfall dates.
2. **Threshold-derived synthetic positives:** grid cells breaching a rainfall intensity-duration threshold **and** slope > 15°, during known monsoon peaks.
3. **Negatives:** stratified random sampling across the district, maintaining ≥1000m Euclidean distance from any positive point to avoid spatial leakage.

Expect severe imbalance. Handle with `scale_pos_weight` = (negatives / positives).

## 6. VERIFICATION STATUS — critical, read before citing anything

### ✅ Verified true
- Open-Meteo archive API: real, keyless, ERA5-Land 0.1° from 1950, hourly, includes soil moisture at depth layers. Endpoint `https://archive-api.open-meteo.com/v1/archive`. Free tier ~10,000 calls/day, one coordinate per call — **batch coordinates, don't loop naively**.
- GSI NLSM programme: 1:50,000 baseline covering ~4.3 lakh sq km across 19 states/UTs including the NE Tertiary belt. Assam is covered.
- GSI meso-scale upscaling: 160 of 200 target sectors completed by field season 2024-25.
- GSI national inventory: ~91,000 historical landslides, 33,904 field-validated.
- `bhusanket.gsi.gov.in` is live. `bhukosh.gsi.gov.in` was timing out as of 25 Aug 2026 — retry periodically.
- GitHub repo `sunruijie0506-ai/lsm-stacking-framework` exists, MIT licensed — but see below.

### ❌ False / fabricated — do not use
- `bhusanket.gsi.gov.in/NLSM_10K_Map.html` — **this URL does not exist.** Fabricated by a research tool.
- MOSDAC SMAP soil moisture — bounding box is 5–24°N, **excludes Guwahati at 26.14°N**. Do not use.

### 🗺️ Known limitation — noted, not blocking

- **GADM boundary area vs census figure:** GADM gives ~904 km² (904 × 1 km cells), while some official sources cite 1,528 km² for Kamrup Metropolitan. The discrepancy likely reflects GADM omitting peri-urban/fringe areas or census measuring a different administrative boundary vintage. Visual inspection of `docs/img/02_grid_preview.png` confirms the GADM polygon covers all operationally critical areas: central Guwahati, Fatasil/Kalapahar/Narengi hill zones, and the Brahmaputra riverfront — i.e., every area where ASDMA landslide points and the pitch narrative are situated. Any territory GADM may omit is peripheral and low-risk. **Do not investigate further.** Acknowledge as a limitation in the final report if asked.

### ⚠️ Unverified — must not be stated as fact
- **"Guwahati is one of GSI's 160 meso-scale sectors."** Not confirmed anywhere. Assume 1:50,000 macro-scale only until someone visually confirms a 1:10,000 Guwahati map.
- **April 2026 Guwahati flood casualty figures.** Every incident cited in the pitch must be re-verified against a primary news source before it goes on a slide.

### 🚫 Explicitly rejected (do not suggest these again)
- Google Earth Engine as geospatial backend — auth overhead, no necessary gain
- SMOTE / SMOTE-Tomek / ADASYN — use `scale_pos_weight`
- `sunruijie0506-ai/lsm-stacking-framework` reuse — it's a Three Gorges (China) InSAR corridor study; adapting it costs more time than it saves
- Bayesian/Optuna hyperparameter tuning — defaults are fine, nobody wins on hyperparameters
- Deep learning (PyTorch/TensorFlow) as the core model
- Switching DEM source to Copernicus GLO-30 mid-build

## 7. Repository structure

```
data/raw/          # gitignored — DEM, boundaries, downloaded data
data/processed/    # gitignored — feature tables (.parquet)
notebooks/         # exploration only, clear outputs before commit
app/               # reusable .py modules + streamlit app
docs/              # research findings, verification records
requirements.txt
```

**Convention:** explore in `notebooks/*.ipynb`, then extract working logic into `app/*.py` as importable functions. Never let the Streamlit app import from a notebook.

**Git:** fork-and-PR workflow. Main repo is `Srv99x/PRAVAH`. Always clear notebook outputs before committing.

## 8. Current state as of 25 Aug 2026

**Done and merged:**
- Repo structure, `requirements.txt`, `.gitignore`
- `app/terrain_utils.py` — `compute_slope(dem_array, cellsize)` using numpy.gradient, returns degrees. Working.
- `notebooks/01_terrain.ipynb` — DEM loading + slope + plotting. **HAS A BUG: cell execution order is broken.** Cell 2 plots `slope` before cell 3 defines it; cell 3 references `bounds`, `pixel_height_deg`, `pixel_width_deg` which are never defined; the `sys.path` fix is in cell 4 but must run first. **Fix this before building on it.**
- `docs/rainfall_soil_sources_verified.md` — good, contains the MOSDAC 24°N catch
- `docs/gsi_bhukosh_findings.md` — **currently the empty TODO template on main.** The completed version exists locally with a GO verdict; needs pushing.

**Not started:**
- 1km grid generation
- Open-Meteo data pipeline
- Village/admin boundaries (GADM `gadm.org` → India → sub-divisions)
- ASDMA 366-point transcription
- Labelling, model training, validation
- Streamlit + Folium app
- MQTT simulator
- Slides

## 9. Build order (dependency-correct)

1. Fix `01_terrain.ipynb` cell order
2. Download GADM boundaries for Kamrup Metropolitan → `data/raw/boundaries/`
3. Generate 1km × 1km grid clipped to the district → `app/grid_utils.py`
4. Open-Meteo fetcher → `app/weather_fetch.py` (batched, cached to parquet)
5. Terrain zonal stats onto grid (slope, elevation, TWI) → `app/terrain_utils.py`
6. Join met + terrain + susceptibility → feature table
7. Labelling → `app/labelling.py`
8. Train RandomForest baseline → then XGBoost with `scale_pos_weight`
9. Validation (spatial blocks, fallback temporal) + PR-AUC/F1 report
10. Static SHAP plots → `docs/img/`
11. Streamlit + Folium app → `app/streamlit_app.py`
12. MQTT simulator + UI toggle → `app/mqtt_sim.py`

**Cut order if behind schedule:** MQTT live toggle → interactive SHAP (use static images) → spatial CV (use temporal split) → soil moisture feature → distance-to-stream. **Never cut:** the map, the risk scores, the narrative.

## 10. Demo requirements

- Streamlit: sidebar = date selector, risk threshold slider, "Ingest Live IoT Telemetry" toggle. Main pane = full-width Folium map with 1km grid coloured transparent→yellow→orange→red by predicted probability.
- Below map: early-warning list — top 10 highest-risk cells, descending, with a notional impacted-structures count.
- SHAP: pre-generated static waterfall plots labelled in plain language ("Why is this area at high risk today?"), red = raises risk, blue = lowers risk. Do **not** build interactive SHAP popups first.
- Must be clickable-through by a judge in under 3 minutes.
- A backup demo video must exist by 30 Aug.

## 11. How to work with me (Claude Code)

- **One task at a time.** Do not attempt multiple build-order steps in one go.
- Before writing code: state what you're about to do and which files you'll touch. Wait for confirmation on anything that changes existing working code.
- After each task: report (a) what was built, (b) how to verify it works, (c) anything that needs a human decision, (d) the exact next task.
- **Never invent URLs, dataset IDs, API endpoints, or file paths.** If uncertain, say so explicitly. This project has already been burned by fabricated sources twice.
- Prefer explicit, readable code over clever code. Teammates with limited tooling experience must be able to read it.
- If a step is taking longer than its value justifies given the 31 Aug deadline, say so and propose the fallback from §9.
- Flag clearly when a task needs **me** (the human) to do something: download a file behind a login, verify a portal visually, make a scope call.
