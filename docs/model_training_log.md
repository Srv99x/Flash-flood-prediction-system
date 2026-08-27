# Model Training Log — Dynamic Trigger Model

**Generated:** 28 Aug 2026, 01:40
**Script:** `app/train_trigger_model.py` (rerun it to reproduce exactly)
**Seed:** `42` (numpy RNG, StratifiedGroupKFold, and both models)

> **Superseded numbers.** An earlier run reported PR-AUC figures for
> RandomForest and XGBoost that were produced by training code which was
> never saved to a file and cannot be reproduced. Those figures are void
> and must not appear in slides, the report, or any pitch material. The
> metrics in this document are the only citable ones.

---

## 1. Features (leak-safe — 5 columns)

```
soil_moisture_0_7
soil_moisture_7_28
temp_c
api_3d
api_7d
```

Slope and elevation are loaded because the **label rule** needs them, but
they are never passed to the model. `train_both()` asserts `X.columns`
equals exactly the list above, so the leak cannot silently return.

## 2. Labels

| source | positive cell-hours |
|---|---|
| `threshold` | 2,459 |
| `verified_incident` | 168 |
| **total positives** | **2,627** |
| negatives (sampled 10:1) | 26,270 |
| **training set** | **28,897** (9.09% positive) |

Rainfall threshold: `precip_1hr >= 20 mm` OR
`precip_3hr >= 60 mm`, AND `slope_mean >= 15°`
(Dikshit & Satyam 2019, Kalimpong — see `app/labelling.py`).
ASDMA vulnerable cells bypass the slope filter; verified incidents are
marked positive for all 24 hours of the incident date.

### Verified-incident coverage

**7 of 8 verified incidents used**; the following were excluded
because they fall outside the weather record (2018-01-01 → 2025-12-31):

- **16 Jul 2026 — Lal Ganesh** — excluded, post-dates weather record

Specifically: 7 of 8 verified incidents used; 16 Jul 2026 Lal Ganesh
excluded, post-dates weather record.

## 3. Split method — episode-grouped, NOT a temporal cutoff

Rainfall-threshold labels make temporally adjacent hours near-duplicates,
so a random split leaks across storms.

1. An hour is **active** if ANY of the 19 weather points breaches the
   rainfall trigger.
2. Runs of active hours separated by **≤ 6 h** merge into one storm
   episode; a longer gap closes it. → **61 storm episodes**.
3. Quiet intervals between storms form their own groups.
4. Group key = `(year, episode_id)`. → **127 groups** in the training set.
5. `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)`,
   fold 0 taken as the held-out test set.

| | rows | episodes | positive rate |
|---|---|---|---|
| train | 23,120 | 103 | 9.09% |
| test | 5,777 | 24 | 9.09% |

**Leakage check: 0 episode groups appear on both sides** (asserted in code —
the script raises rather than training if any group straddles the split).

## 4. Results

| model | PR-AUC | F1-macro |
|---|---|---|
| RandomForest (`n_estimators=100`, `class_weight='balanced'`) | **0.8257** | 0.7640 |
| XGBoost (`n_estimators=150`, `max_depth=6`, `lr=0.05`, `scale_pos_weight=9.9990`) | **0.7504** | 0.8307 |

Both models trained on **identical** train/test indices.
Metrics are PR-AUC + F1-macro. ROC-AUC is deliberately not reported
(CLAUDE.md §4 — misleading under this class imbalance).

**Winner on PR-AUC: `random_forest`** → saved to `models/random_forest_trigger_model.pkl`
and wired into `app/predict.py`.

## 5. Memory-safe execution

The full fan-out (904 cells × 35,328 monsoon hours = 31,936,512 rows)
peaks around 2.4 GB at `pd.concat`, against ~3.7 GB free on the build
machine. This script never materialises it:

- **Pass 1** — label one weather point at a time (~1.7M rows, ~90 MB), keep positives only.
- **Between** — compute the global 1 km safe-cell set from all positives.
- **Pass 2** — re-walk chunks restricted to safe cells, draw negatives by proportional quota.

The 3-hour rolling sum is computed pre-join on the 1.3M-row weather table.
Float16 downcasting is applied to terrain columns only; weather and model
columns stay float32, because float16 spacing near the 60 mm threshold
(0.03 mm) and near `api_7d` ≈ 500 mm (0.5 mm) would corrupt labels and inputs.

---

## 6. KNOWN INCIDENT — synthetic placeholder grid shadowed the real grid

**Window:** commit `011d17e` ("Demo: Streamlit risk map, early warning table,
simulated IoT panel") until 28 Aug 2026.

`app/generate_synthetic_grid.py` wrote a **synthetic placeholder** grid to
`data/processed/kamrup_metro_grid_1km.parquet` and that file was committed.
It used grid_ids of the form `KM_0001`, while `grid_weather_mapping`,
`terrain_features`, `susceptibility_features` and `current_risk_scores` all
used `KM_R000_C028` — produced by the real generator, `app/grid_utils.py`,
whose output was never committed. **The two id sets had zero overlap.**

The placeholder was also geographically wrong: a plain rectangle spanning
91.40–92.00 °E / 25.98–26.11 °N, versus the real GADM-clipped grid at
91.63–92.18 °E / 26.00–26.26 °N — shifted ~20 km west with the northern
third clipped off.

**Everything below was invalid for the whole window:**

- **Map output.** `get_real_risk()` in `app/streamlit_app.py` merged risk
  scores onto the grid by `grid_id` and matched **0 of 904** rows; the
  subsequent `.fillna(0.0)` painted every cell 0.0 risk. The Streamlit map
  never displayed a real prediction once it was wired to real scores.
- **ASDMA and verified-incident labels.** `add_asdma_positives()` spatially
  joins against the grid, so it returned placeholder ids that matched no
  feature rows. **ASDMA vulnerable cells and all verified incidents
  contributed ZERO positives.** Any model trained in this window saw
  rainfall-threshold labels only, regardless of what its log claimed.
- **Any PR-AUC / F1 figure produced in this window**, and any episode or
  group count derived from it, is not comparable to the metrics in §4 above
  and must not be cited.

**Resolution:** the grid was regenerated with
`app.grid_utils.generate_grid()` against
`data/raw/boundaries/kamrup_metropolitan.geojson` — 904 cells whose id set
is exactly equal to the mapping / terrain / susceptibility id sets.
`app/generate_synthetic_grid.py` was deleted (recoverable at
`git show 011d17e:app/generate_synthetic_grid.py` if ever needed).
`get_real_risk()` now raises if fewer than 50% of cells match on the join,
so a silent 0-of-904 merge cannot recur.

**Still outstanding:** `app/mqtt_sim.py` hardcodes placeholder sensor
grid_ids (`KM_0042`, …) and lat/lons (e.g. 26.38 °N) that fall outside the
real grid. Currently harmless — the looked-up value is unused in rendering
and the telemetry itself is simulated — but the 5 sensor cells should be
remapped to real grid_ids with verified coordinates before the demo.

---

## 7. Susceptibility calibration (28 Aug 2026)

### 7.1 The trigger model is NOT miscalibrated — the demo snapshot was extreme

Suspicion was that `dynamic_trigger_prob` averaging 0.922 meant the model
had learned the 10:1 sampled training ratio rather than real conditions.
Measured across all 904 cells on three dates:

| date | condition | min | median | mean | max |
|---|---|---|---|---|---|
| 2020-05-26 18:00 | wettest hour in the 8-year record (28.2 mm/hr district mean) | 0.469 | 1.000 | 0.922 | 1.000 |
| 2018-05-19 12:00 | monsoon, API-3d at the monsoon median | 0.000 | 0.000 | 0.0022 | 0.090 |
| 2020-01-15 12:00 | dry season | 0.000 | 0.000 | 0.000 | 0.000 |

The model discriminates sharply: **exactly 0.0 on a dry day, near 1.0 on the
most extreme rainfall hour in eight years.** The 0.922 figure came from
`generate_current_risk.py` using that extreme hour as its snapshot, not from
miscalibration. **No probability calibration was applied** — isotonic
calibration on this would have been a fix for a problem that does not exist.

The response is close to a step function rather than a gentle ramp (median
1.000 vs 0.000 between the two monsoon dates). For the demo this means date
selection matters: a mid-range monsoon date shows an all-green map.

### 7.2 Slope cutoffs were sized for the wrong terrain

Measured `slope_mean` over the 811 cells with DEM coverage:

```
min 0.00   median 9.41   95th pct 18.25   99th pct 22.39   max 31.91  (degrees)
```

The district's steepest 1 km cell averages **31.9°**. The previous cutoffs
(15 / 30 / 45°) were Himalayan-scale, so:

- **"Very High" (>45°) was structurally unreachable** — 0 cells, always.
- **"High" (30-45°)** caught exactly **1** cell.
- **81% of the district** collapsed into "Low".

New cutoffs, anchored to the measured distribution (see
`app/susceptibility_utils.py` for the full derivation):

| class | old | new | cells (of 904) |
|---|---|---|---|
| Low | < 15° | **< 8°** | 373 (41.3%) |
| Moderate | 15-30° | **8-15°** | 283 (31.3%) |
| High | 30-45° | **15-20°** | 135 (14.9%) |
| Very High | > 45° | **>= 20°** | 20 (2.2%) |
| No DEM data | — | — | 93 (10.3%) |

15° is not arbitrary: it is the shallow-landslide slope threshold already
used for labelling in `app/labelling.py`, so "High" begins exactly where a
cell becomes eligible to be a positive.

### 7.3 Multipliers

Still **team-assigned, not from a cited study.** Raised from 0.1 / 0.3 /
0.7 / 1.0 to **0.20 / 0.45 / 0.70 / 0.90**.

Because trigger probability saturates near 1.0 on extreme dates,
`final_risk ≈ multiplier` there — so the multipliers effectively *are* the
risk scale on the dates that matter. They were chosen so each susceptibility
class lands in its own severity band (Low <0.25, Medium 0.25-0.50, High
0.50-0.75, Severe >=0.75).

Resulting distribution across the 904 cells:

| date | Low | Medium | High | Severe | max |
|---|---|---|---|---|---|
| **extreme** — old scheme | 770 | 133 | 1 | 0 | 0.700 |
| **extreme** — new scheme | 481 | 272 | **132** | **19** | 0.900 |
| moderate monsoon — new | 904 | 0 | 0 | 0 | 0.081 |
| dry season — new | 904 | 0 | 0 | 0 | 0.000 |

An extreme date now produces visible orange and red; a moderate or dry date
produces none. Ordering (steeper = higher risk) is preserved throughout.

### 7.4 Known limitation — slope is a landslide proxy, not a flood proxy

The susceptibility layer ranks cells by slope, which encodes **landslide**
susceptibility. It therefore systematically under-weights flat urban areas
that flood badly — Anil Nagar / Zoo Road appear in the verified-incident
record explicitly as *urban flooding*, and sit in the Low band. A
flood-specific layer (TWI, or distance-to-stream) would be the correct
complement. Not addressed before the 31 Aug freeze; state it as a limitation
rather than claiming the current layer captures flood risk.

Cells with no DEM coverage (93) get multiplier 0.0 and are rendered as grey
"No Data" by the app, not as genuine low risk. Verified: that set is exactly
`missing_terrain_cells.json`.

---

## 8. Official hazard floor — why, and the circularity, stated plainly

### 8.1 Terrain alone ranks the real incident cells wrongly

Measured against the only ground truth available:

| group | cells | slope_mean (median) | slope_max (median) |
|---|---|---|---|
| containing a documented incident | 6 | **6.0°** | 27.6° |
| containing an ASDMA vulnerable location | 34 | **4.5°** | 30.4° |
| the district as a whole | 904 | **9.4°** | 30.9° |

**Every documented landslide location sits in a cell that is flatter than the
district average on mean slope.** The slope-mean proxy is not merely
uninformative here — it is mildly *anti*-correlated with the ground truth.

The reason is aggregation: these are settlements at the foot of steep cuts.
Bonda's cell has slope_mean 7.09° but slope_max 29.90°; Dhirenpara's has
slope_mean 10.52° but slope_max 45.16°. The failure happens on a local hill
cut or guard wall that a 1 km average erases.

Switching to `slope_max` does **not** fix it and was rejected on the evidence:
district-wide slope_max median is 30.9° versus 30.4° for ASDMA cells, so it
discriminates no better — it just shifts every cell upward.

Consequence before the fix, with the slope-only layer:

| incident | date | trigger | multiplier | risk | band |
|---|---|---|---|---|---|
| Bonda, 5 deaths | 2025-05-30 | 1.000 | 0.20 | 0.200 | **Low** |
| Dhirenpara, 1 death | 2023-06-17 | 1.000 | 0.45 | 0.450 | Medium |

The trigger model fired at maximum on both. The susceptibility layer alone
rendered a fatal landslide site green on the day five people died.

### 8.2 The floor

    susceptibility = max(terrain-derived class,
                         "High" if the cell contains an ASDMA vulnerable
                                   location or a verified incident)

**34 of 904 cells (3.8%) are flagged; 31 are actually raised** (3 were already
High or above). The floor only ever raises, never lowers.

| class | terrain only | after floor |
|---|---|---|
| Low | 373 | 351 |
| Moderate | 283 | 274 |
| High | 135 | **166** |
| Very High | 20 | 20 |
| No DEM | 93 | 93 |

After the floor, both incident cells render **High (0.700)** on their own
incident dates.

### 8.3 The circularity — NOT metric leakage

ASDMA locations are used in two places:

1. **Labelling** (`app/labelling.py`) — ASDMA cells bypass the slope filter,
   so a rainfall trigger there produces a positive.
2. **The susceptibility floor** (this section).

A sharp reader will spot that and should be given the honest answer:

**This does not affect PR-AUC 0.8257 or F1-macro 0.7640.** Susceptibility is a
*static display-layer gate* applied to the model's output — it is not a model
input. The five model features are `soil_moisture_0_7`, `soil_moisture_7_28`,
`temp_c`, `api_3d`, `api_7d`, all weather-point-level, and
`train_both()` asserts `X.columns` equals exactly that list. The model never
sees susceptibility, ASDMA membership, slope, or elevation. Changing the
multipliers or the floor changes the rendered map and changes nothing about
the reported metrics.

**What it does mean, and what must be disclosed:** a cell can display high risk
because it is on Assam's official vulnerable-locations list, not because the
model or the DEM inferred it. That is stated in the Streamlit sidebar under
"How to read this map" and in the map caption. It is not buried.

The defensible framing: the system combines a *learned dynamic trigger* with a
*static hazard layer built from official government hazard identification plus
terrain*. That is how operational early-warning systems are normally built —
the objection would be if we claimed the model discovered these locations. We
do not.

### 8.4 Verified validation claim (checked, not assumed)

All 8 verified incident points fall in **6 unique grid cells**. All 6 are
inside ASDMA's 34 cells:

```
verified cells : KM_R013_C004  KM_R016_C013  KM_R017_C008
                 KM_R018_C012  KM_R020_C016  KM_R020_C018
subset of ASDMA cells : True
```

Restricting to the 6 incidents with an explicit death count gives **5 unique
cells** (`KM_R013_C004`, `KM_R017_C008`, `KM_R018_C012`, `KM_R020_C016`,
`KM_R020_C018`) — also a strict subset.

**Precise wording for the pitch** (the loose version conflates two counts):

> All 6 grid cells containing a documented landslide or flood incident —
> including all 5 cells where a fatal landslide occurred — were already on
> ASDMA's officially identified vulnerable-locations list.

Note `KM_R017_C008` alone hosted three separate documented incidents
(Dhirenpara 2023-06-17, Krishnanagar/Lal Ganesh 2024-05-29, Lal Ganesh
2026-07-16), which is why 8 points collapse to 6 cells.

### 8.5 Date-responsive map and the trigger cache

`build_trigger_cache.py` precomputes daily-peak trigger probability per
`(weather_point_id, date)` — **55,518 rows, 82 KB**, covering 2018-01-01 to
2025-12-31.

This is exact rather than an approximation because all five model features are
weather-point-level, so trigger probability is identical across every cell
sharing a weather point. That reduces 904 × 70,128 = 63.4M model evaluations
to 19 × 2,922 cached values. The app renders any date as a lookup and a
multiply, with no model call on stage.

The builder asserts the cached path matches `app.predict.predict_risk()` to
within 1e-6 across all 904 cells; measured difference on 2020-05-26 was
**0.0000000000**.

Risk distribution on the four demo dates:

| date | Low | Medium | High | Severe | max |
|---|---|---|---|---|---|
| 2025-05-30 Bonda | 465 | 297 | 133 | 9 | 0.900 |
| 2023-06-17 Dhirenpara | 626 | 214 | 64 | 0 | 0.700 |
| 2020-05-26 wettest | 444 | 274 | 166 | 20 | 0.900 |
| 2020-01-15 dry | 904 | 0 | 0 | 0 | 0.000 |
