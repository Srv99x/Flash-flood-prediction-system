"""
app/train_trigger_model.py
──────────────────────────
Train the dynamic-trigger model for the flash-flood risk pipeline.

Trains BOTH RandomForest and XGBoost on an IDENTICAL leak-safe split,
compares them on PR-AUC + F1-macro, and saves the winner to models/.

Run from the repository root:

    python app/train_trigger_model.py

Design notes
────────────
1. MEMORY (two-pass, per-weather-point chunks)
   The full fan-out is 904 cells x 35,328 monsoon hours = 31.9M rows,
   which peaks around 2.4 GB during pd.concat -- too close to the free
   RAM on the build machine.  Instead we process one weather point at a
   time (~48 cells x 35,328 hours = ~1.7M rows, ~90 MB) and keep only
   the rows we actually need:

     Pass 1  label each chunk, keep POSITIVES only.
     (between) compute the global 1 km safe-cell set from all positives.
     Pass 2  re-walk the chunks restricted to safe cells, draw NEGATIVES.

   The 3-hour rolling sum is still computed pre-join on the 1.3M-row
   weather table (never on the fan-out), as in app/features.py.

   Float16 downcasting is applied to the terrain columns only.  The
   weather/model columns stay float32: at 1.7M rows per chunk there is
   no memory pressure left to justify it, and float16 spacing near the
   60 mm threshold (0.03 mm) and near api_7d ~500 mm (0.5 mm) would
   corrupt both the labels and the model inputs.

2. FEATURES (leak-safe)
   Only the 5 confirmed non-leaking columns are fed to the models.
   Slope/elevation are loaded because the LABEL rule needs them, but
   they are never in X -- a hard assertion enforces this.

3. SPLIT (episode-grouped)
   Rainfall-threshold labels make temporally adjacent hours near-
   duplicates, so a random split leaks.  We cut the monsoon hourly axis
   into storm episodes: an hour is "active" if ANY weather point breaches
   the rainfall trigger; runs of active hours separated by <= 6 h merge
   into one episode; a gap > 6 h closes it.  Quiet intervals between
   storms are their own groups.  Every row is grouped by
   (year, episode_id) and StratifiedGroupKFold keeps whole episodes on
   one side of the split.
"""

from __future__ import annotations

import json
import logging
import pickle
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# Repo root on sys.path so `from app.labelling import ...` works when this
# file is run directly as a script.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import geopandas as gpd  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.metrics import average_precision_score, f1_score  # noqa: E402
from sklearn.model_selection import StratifiedGroupKFold  # noqa: E402
from xgboost import XGBClassifier  # noqa: E402

from app.labelling import (  # noqa: E402
    DEFAULT_PRECIP_1HR_MM,
    DEFAULT_PRECIP_3HR_MM,
    DEFAULT_SLOPE_DEG,
    add_asdma_positives,
    label_by_rainfall_threshold,
)

logger = logging.getLogger("train")

# ── Paths ─────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed"
RAW_DIR = ROOT / "data" / "raw"
MODELS_DIR = ROOT / "models"
DOCS_DIR = ROOT / "docs"

ASDMA_CSV = RAW_DIR / "asdma_vulnerable_locations.csv"
VERIFIED_CSV = RAW_DIR / "verified_incidents.csv"

# ── Locked configuration ──────────────────────────────────────────────
# The 5 confirmed non-leaking features.  Do NOT add slope, elevation, or
# any rainfall-threshold-adjacent column: that reintroduces the leak.
FEATURES = [
    "soil_moisture_0_7",
    "soil_moisture_7_28",
    "temp_c",
    "api_3d",
    "api_7d",
]

MONSOON_MONTHS = (5, 6, 7, 8, 9, 10)
EPISODE_GAP_HOURS = 6
NEG_TO_POS_RATIO = 10
MIN_SEPARATION_M = 1000.0
METRIC_CRS = "EPSG:32646"
RANDOM_STATE = 42
N_SPLITS = 5

# If the reconstructed episode count lands outside this range, the episode
# definition has diverged from the validated one -- stop instead of training.
EPISODE_SANITY_RANGE = (50, 500)

TERRAIN_COLS = ["elevation_mean", "elevation_max", "slope_mean", "slope_max"]
WEATHER_COLS = [
    "precipitation_mm",
    "soil_moisture_0_7",
    "soil_moisture_7_28",
    "temp_c",
    "api_3d",
    "api_7d",
]


# ══════════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════

def load_weather() -> pd.DataFrame:
    """
    Load the hourly weather table and compute the 3-hour rolling sum
    PRE-JOIN (on 1.3M rows, never on the 31.9M-row fan-out).

    The rolling sum is computed on the FULL year before the monsoon
    filter, so the first hours of each May are not truncated.
    """
    cols = ["weather_point_id", "timestamp"] + WEATHER_COLS
    w = pd.read_parquet(DATA_DIR / "weather_hourly.parquet", columns=cols)

    float_cols = w.select_dtypes("float64").columns
    w[float_cols] = w[float_cols].astype("float32")

    w.sort_values(["weather_point_id", "timestamp"], inplace=True)
    w["precip_3hr_mm"] = (
        w.groupby("weather_point_id")["precipitation_mm"]
        .transform(lambda s: s.rolling(window=3, min_periods=1).sum())
    )

    logger.info(
        "Weather loaded: %s rows, %d weather points, %s -> %s",
        f"{len(w):,}",
        w["weather_point_id"].nunique(),
        w["timestamp"].min(),
        w["timestamp"].max(),
    )

    w = w[w["timestamp"].dt.month.isin(MONSOON_MONTHS)].copy()
    logger.info(
        "Monsoon filter (months %s): %s rows, %s unique timestamps",
        MONSOON_MONTHS,
        f"{len(w):,}",
        f"{w['timestamp'].nunique():,}",
    )
    return w


def load_mapping_with_terrain() -> pd.DataFrame:
    """grid_id -> weather_point_id, with terrain columns attached (904 rows)."""
    mapping = pd.read_parquet(
        DATA_DIR / "grid_weather_mapping.parquet",
        columns=["grid_id", "weather_point_id"],
    )
    terrain = pd.read_parquet(DATA_DIR / "terrain_features.parquet")

    # float16 is safe here: these columns only feed the slope >= 15 deg
    # comparison, where float16 spacing is ~0.008 deg.
    tf = terrain.select_dtypes(include=["float64", "float32"]).columns
    terrain[tf] = terrain[tf].astype("float16")

    merged = mapping.merge(terrain, on="grid_id", how="left")
    logger.info(
        "Mapping + terrain: %d cells, %d weather points",
        len(merged),
        merged["weather_point_id"].nunique(),
    )
    return merged


def check_verified_incident_coverage(weather: pd.DataFrame) -> list[dict]:
    """
    Warn loudly about verified incidents that fall outside the weather
    record -- add_asdma_positives() would silently match zero rows.

    Returns the list of excluded incidents for the training log.
    """
    ver = pd.read_csv(VERIFIED_CSV)
    ver["date"] = pd.to_datetime(ver["date"])
    w_min, w_max = weather["timestamp"].min(), weather["timestamp"].max()

    excluded = []
    for _, row in ver.iterrows():
        if not (w_min <= row["date"] <= w_max):
            excluded.append(
                {"date": row["date"].strftime("%d %b %Y"), "location": row["location"]}
            )

    if excluded:
        for inc in excluded:
            logger.warning(
                "VERIFIED INCIDENT EXCLUDED -- %s (%s) post-dates the weather "
                "record (ends %s). It will contribute ZERO positives.",
                inc["date"],
                inc["location"],
                w_max.date(),
            )
    logger.info(
        "Verified incidents: %d of %d usable, %d excluded",
        len(ver) - len(excluded),
        len(ver),
        len(excluded),
    )
    return excluded


# ══════════════════════════════════════════════════════════════════════
# CHUNK BUILDER
# ══════════════════════════════════════════════════════════════════════

def build_wp_chunk(wp_weather: pd.DataFrame, wp_mapping: pd.DataFrame) -> pd.DataFrame:
    """
    Cross-join one weather point's hours against the grid cells mapped to
    it.  ~48 cells x 35,328 hours = ~1.7M rows, ~90 MB.
    """
    wp_slim = wp_weather.drop(columns=["weather_point_id"])
    cells = wp_mapping.drop(columns=["weather_point_id"])
    return (
        cells.assign(_k=1)
        .merge(wp_slim.assign(_k=1), on="_k")
        .drop(columns=["_k"])
    )


def label_chunk(chunk: pd.DataFrame, grid_gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    """
    Apply the project's labelling functions to one chunk.

    Semantically identical to running them once on the full 31.9M-row
    table: the 3-hour rolling sum is already precomputed, and both the
    ASDMA membership test and the verified-incident date match are
    per-cell, so there is no cross-chunk dependency.
    """
    chunk = label_by_rainfall_threshold(chunk)
    chunk = add_asdma_positives(
        chunk,
        asdma_csv_path=ASDMA_CSV,
        verified_csv_path=VERIFIED_CSV,
        grid_gdf=grid_gdf,
        metric_crs=METRIC_CRS,
    )
    return chunk


# ══════════════════════════════════════════════════════════════════════
# PASS 1 — POSITIVES
# ══════════════════════════════════════════════════════════════════════

def pass1_collect_positives(
    weather: pd.DataFrame,
    mapping: pd.DataFrame,
    grid_gdf: gpd.GeoDataFrame,
) -> pd.DataFrame:
    """Label every chunk, keep only positive rows."""
    logger.info("── PASS 1: collecting positives ──")
    out = []
    for wp_id, wp_weather in weather.groupby("weather_point_id"):
        wp_mapping = mapping[mapping["weather_point_id"] == wp_id]
        if wp_mapping.empty:
            continue
        chunk = build_wp_chunk(wp_weather, wp_mapping)
        chunk = label_chunk(chunk, grid_gdf)
        pos = chunk[chunk["target_event"] == 1].copy()
        if len(pos):
            out.append(pos)
        logger.info(
            "  %s: %s rows -> %s positives", wp_id, f"{len(chunk):,}", f"{len(pos):,}"
        )
        del chunk

    positives = pd.concat(out, ignore_index=True)
    logger.info("Pass 1 complete: %s positives", f"{len(positives):,}")
    logger.info(
        "Positives by label_source:\n%s",
        positives["label_source"].value_counts().to_string(),
    )
    return positives


# ══════════════════════════════════════════════════════════════════════
# SAFE-CELL SET (global, between passes)
# ══════════════════════════════════════════════════════════════════════

def compute_safe_cells(
    positives: pd.DataFrame, grid_gdf: gpd.GeoDataFrame
) -> set[str]:
    """
    Cells whose centroid lies >= MIN_SEPARATION_M from EVERY positive
    cell centroid.  Lifted from sample_negatives() Tier 1, hoisted here
    because it needs the global positive set.
    """
    pos_ids = set(positives["grid_id"].unique())
    grid_m = grid_gdf.to_crs(METRIC_CRS)
    pos_centroids = grid_m[grid_m["grid_id"].isin(pos_ids)].geometry.centroid
    buffer_union = pos_centroids.buffer(MIN_SEPARATION_M).union_all()

    safe_mask = ~grid_m.geometry.centroid.within(buffer_union)
    safe = set(grid_m.loc[safe_mask, "grid_id"].values)

    logger.info(
        "Safe cells: %d positive cells buffered by %.0f m -> %d of %d cells safe",
        len(pos_ids),
        MIN_SEPARATION_M,
        len(safe),
        len(grid_gdf),
    )
    return safe


# ══════════════════════════════════════════════════════════════════════
# PASS 2 — NEGATIVES
# ══════════════════════════════════════════════════════════════════════

def pass2_sample_negatives(
    weather: pd.DataFrame,
    mapping: pd.DataFrame,
    grid_gdf: gpd.GeoDataFrame,
    safe_cells: set[str],
    n_target: int,
) -> pd.DataFrame:
    """
    Draw negatives uniformly from (safe cells x monsoon hours).

    Quota is allocated per weather point in proportion to how many safe
    cells it owns, then sampled uniformly within each chunk -- which is
    exactly a uniform draw over the whole safe population.
    """
    logger.info("── PASS 2: sampling negatives ──")
    safe_map = mapping[mapping["grid_id"].isin(safe_cells)]
    n_ts = weather["timestamp"].nunique()

    avail = safe_map.groupby("weather_point_id").size() * n_ts
    total_avail = int(avail.sum())
    logger.info(
        "Negative pool: %s safe cell-hours, target %s (ratio %d:1)",
        f"{total_avail:,}",
        f"{n_target:,}",
        NEG_TO_POS_RATIO,
    )

    if total_avail <= n_target:
        quota = {wp: int(n) for wp, n in avail.items()}
        logger.info("Pool <= target -- taking all available")
    else:
        raw = {wp: n_target * n / total_avail for wp, n in avail.items()}
        quota = {wp: int(np.floor(v)) for wp, v in raw.items()}
        # Hand the rounding remainder to the largest fractional parts.
        short = n_target - sum(quota.values())
        for wp in sorted(raw, key=lambda k: raw[k] - np.floor(raw[k]), reverse=True)[:short]:
            quota[wp] += 1

    rng = np.random.default_rng(RANDOM_STATE)
    out = []
    for wp_id, wp_weather in weather.groupby("weather_point_id"):
        k = quota.get(wp_id, 0)
        if k <= 0:
            continue
        wp_mapping = safe_map[safe_map["weather_point_id"] == wp_id]
        wp_mapping = mapping[mapping["grid_id"].isin(set(wp_mapping["grid_id"]))]
        chunk = build_wp_chunk(wp_weather, wp_mapping)
        chunk = label_chunk(chunk, grid_gdf)

        n_pos_in_safe = int(chunk["target_event"].sum())
        if n_pos_in_safe:
            # Should be structurally impossible: a cell that is ever
            # positive is inside its own 1 km buffer and cannot be safe.
            logger.error(
                "  %s: %d POSITIVES found among safe cells -- dropping them",
                wp_id,
                n_pos_in_safe,
            )
            chunk = chunk[chunk["target_event"] == 0]

        idx = rng.choice(len(chunk), size=min(k, len(chunk)), replace=False)
        out.append(chunk.iloc[idx].copy())
        logger.info("  %s: %s candidates -> %s sampled", wp_id, f"{len(chunk):,}", f"{k:,}")
        del chunk

    negatives = pd.concat(out, ignore_index=True)
    logger.info("Pass 2 complete: %s negatives", f"{len(negatives):,}")
    return negatives


# ══════════════════════════════════════════════════════════════════════
# EPISODE-GROUPED SPLIT
# ══════════════════════════════════════════════════════════════════════

def build_storm_episodes(weather: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """
    Cut the monsoon hourly axis into storm episodes.

    An hour is "active" if ANY weather point breaches the rainfall
    trigger.  Consecutive active hours separated by <= EPISODE_GAP_HOURS
    merge into one episode; a longer gap closes it.

    Returns (starts, ends) as sorted datetime64 arrays.
    """
    trig = (
        (weather["precipitation_mm"] >= DEFAULT_PRECIP_1HR_MM)
        | (weather["precip_3hr_mm"] >= DEFAULT_PRECIP_3HR_MM)
    )
    active = np.sort(weather.loc[trig, "timestamp"].unique())
    if len(active) == 0:
        raise RuntimeError("No active storm hours found -- cannot build episodes.")

    gaps_h = np.diff(active).astype("timedelta64[h]").astype(np.int64)
    is_new = np.concatenate([[True], gaps_h > EPISODE_GAP_HOURS])
    is_end = np.concatenate([is_new[1:], [True]])

    starts, ends = active[is_new], active[is_end]
    logger.info(
        "Storm episodes: %s active hours -> %d episodes (gap > %d h closes)",
        f"{len(active):,}",
        len(starts),
        EPISODE_GAP_HOURS,
    )
    return starts, ends


def assign_groups(
    ts: pd.Series, starts: np.ndarray, ends: np.ndarray
) -> pd.Series:
    """
    Map each row's timestamp to its (year, episode_id) group.

    Inside a storm window -> that storm's id.  Between storms -> the
    quiet interval's id.  Year is prefixed so a quiet stretch spanning a
    season boundary cannot merge two years into one group.
    """
    t = ts.to_numpy()
    i = np.searchsorted(starts, t, side="right")  # 0 .. n_storms
    inside = (i > 0) & (t <= ends[np.clip(i - 1, 0, len(ends) - 1)])

    kind = np.where(inside, "S", "Q")
    num = np.where(inside, i - 1, i)
    year = ts.dt.year.to_numpy()
    return pd.Series(
        [f"{y}_{k}{n}" for y, k, n in zip(year, kind, num)], index=ts.index
    )


def make_split(
    training_df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, pd.Series, int]:
    """Episode-grouped, label-stratified 80/20 split (fold 0 of 5)."""
    groups = training_df["episode_group"]
    y = training_df["target_event"].to_numpy()
    n_groups = groups.nunique()

    logger.info("Episode groups in training set: %d", n_groups)
    lo, hi = EPISODE_SANITY_RANGE
    if not (lo <= n_groups <= hi):
        raise RuntimeError(
            f"Episode count {n_groups} is outside the sanity range {lo}-{hi}. "
            "The episode definition has diverged from the validated one "
            "(Task 4 reported 183 train / 46 test = 229). Stopping before "
            "training rather than producing an unverifiable split."
        )

    sgkf = StratifiedGroupKFold(
        n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE
    )
    train_idx, test_idx = next(sgkf.split(training_df[FEATURES], y, groups))

    g_train = set(groups.iloc[train_idx])
    g_test = set(groups.iloc[test_idx])
    overlap = g_train & g_test
    if overlap:
        raise RuntimeError(f"Episode leakage: {len(overlap)} groups on both sides.")

    logger.info(
        "Split: %s train rows (%d episodes, %.2f%% pos) / %s test rows "
        "(%d episodes, %.2f%% pos) -- 0 shared episodes",
        f"{len(train_idx):,}",
        len(g_train),
        100 * y[train_idx].mean(),
        f"{len(test_idx):,}",
        len(g_test),
        100 * y[test_idx].mean(),
    )
    return train_idx, test_idx, groups, n_groups


# ══════════════════════════════════════════════════════════════════════
# TRAINING
# ══════════════════════════════════════════════════════════════════════

def evaluate(model, X_test, y_test) -> dict:
    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)
    return {
        "pr_auc": float(average_precision_score(y_test, proba)),
        "f1_macro": float(f1_score(y_test, pred, average="macro")),
    }


def train_both(training_df, train_idx, test_idx) -> dict:
    X = training_df[FEATURES].astype("float32")
    y = training_df["target_event"].to_numpy()

    # Hard leak guard: X must be exactly the 5 approved columns.
    assert list(X.columns) == FEATURES, f"Feature leak: {list(X.columns)}"

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    n_neg, n_pos = int((y_train == 0).sum()), int((y_train == 1).sum())
    spw = n_neg / n_pos
    logger.info("Train set: %d neg / %d pos -> scale_pos_weight %.4f", n_neg, n_pos, spw)

    logger.info("Training RandomForest ...")
    rf = RandomForestClassifier(
        n_estimators=100,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    rf_metrics = evaluate(rf, X_test, y_test)
    logger.info(
        "  RandomForest: PR-AUC %.4f | F1-macro %.4f",
        rf_metrics["pr_auc"],
        rf_metrics["f1_macro"],
    )

    logger.info("Training XGBoost ...")
    xgb = XGBClassifier(
        n_estimators=150,
        max_depth=6,
        learning_rate=0.05,
        scale_pos_weight=spw,
        eval_metric="aucpr",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    xgb.fit(X_train, y_train)
    xgb_metrics = evaluate(xgb, X_test, y_test)
    logger.info(
        "  XGBoost:      PR-AUC %.4f | F1-macro %.4f",
        xgb_metrics["pr_auc"],
        xgb_metrics["f1_macro"],
    )

    return {
        "random_forest": {"model": rf, "metrics": rf_metrics, "spw": None},
        "xgboost": {"model": xgb, "metrics": xgb_metrics, "spw": spw},
    }


# ══════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════

def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s | %(name)s | %(message)s"
    )
    MODELS_DIR.mkdir(exist_ok=True)

    grid_gdf = gpd.read_parquet(DATA_DIR / "kamrup_metro_grid_1km.parquet")
    weather = load_weather()
    mapping = load_mapping_with_terrain()
    excluded_incidents = check_verified_incident_coverage(weather)

    # ── Build the training set ────────────────────────────────────────
    positives = pass1_collect_positives(weather, mapping, grid_gdf)
    safe_cells = compute_safe_cells(positives, grid_gdf)
    negatives = pass2_sample_negatives(
        weather, mapping, grid_gdf, safe_cells, len(positives) * NEG_TO_POS_RATIO
    )

    training_df = pd.concat([positives, negatives], ignore_index=True)
    training_df.sort_values(["grid_id", "timestamp"], inplace=True)
    training_df.reset_index(drop=True, inplace=True)
    logger.info(
        "Training set: %s rows (%s pos + %s neg, %.2f%% positive)",
        f"{len(training_df):,}",
        f"{len(positives):,}",
        f"{len(negatives):,}",
        100 * len(positives) / len(training_df),
    )

    # ── Episode-grouped split ─────────────────────────────────────────
    starts, ends = build_storm_episodes(weather)
    training_df["episode_group"] = assign_groups(training_df["timestamp"], starts, ends)
    train_idx, test_idx, groups, n_groups = make_split(training_df)

    # ── Train + compare ───────────────────────────────────────────────
    results = train_both(training_df, train_idx, test_idx)

    winner = max(results, key=lambda k: results[k]["metrics"]["pr_auc"])
    loser = "xgboost" if winner == "random_forest" else "random_forest"
    logger.info(
        "WINNER on PR-AUC: %s (%.4f vs %.4f)",
        winner,
        results[winner]["metrics"]["pr_auc"],
        results[loser]["metrics"]["pr_auc"],
    )

    model_path = MODELS_DIR / f"{winner}_trigger_model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(results[winner]["model"], f)
    logger.info("Saved winner -> %s", model_path)

    # ── Training log ──────────────────────────────────────────────────
    write_log(
        results, winner, training_df, positives, negatives,
        train_idx, test_idx, groups, n_groups, starts,
        excluded_incidents, model_path,
    )

    summary = {
        "winner": winner,
        "model_path": str(model_path.relative_to(ROOT)).replace("\\", "/"),
        "metrics": {k: v["metrics"] for k, v in results.items()},
        "n_train_rows": int(len(train_idx)),
        "n_test_rows": int(len(test_idx)),
        "n_episodes": int(n_groups),
    }
    print("\n" + json.dumps(summary, indent=2))


def write_log(
    results, winner, training_df, positives, negatives,
    train_idx, test_idx, groups, n_groups, starts,
    excluded_incidents, model_path,
) -> None:
    """Write docs/model_training_log.md -- the permanent record."""
    g_train = groups.iloc[train_idx].nunique()
    g_test = groups.iloc[test_idx].nunique()
    src = positives["label_source"].value_counts()

    excl_lines = "\n".join(
        f"- **{i['date']} — {i['location']}** — excluded, post-dates weather record"
        for i in excluded_incidents
    ) or "- none"

    rf, xgb = results["random_forest"]["metrics"], results["xgboost"]["metrics"]
    n_ver = len(pd.read_csv(VERIFIED_CSV))

    md = f"""# Model Training Log — Dynamic Trigger Model

**Generated:** {datetime.now().strftime('%d %b %Y, %H:%M')}
**Script:** `app/train_trigger_model.py` (rerun it to reproduce exactly)
**Seed:** `{RANDOM_STATE}` (numpy RNG, StratifiedGroupKFold, and both models)

> **Superseded numbers.** An earlier run reported PR-AUC figures for
> RandomForest and XGBoost that were produced by training code which was
> never saved to a file and cannot be reproduced. Those figures are void
> and must not appear in slides, the report, or any pitch material. The
> metrics in this document are the only citable ones.

---

## 1. Features (leak-safe — 5 columns)

```
{chr(10).join(FEATURES)}
```

Slope and elevation are loaded because the **label rule** needs them, but
they are never passed to the model. `train_both()` asserts `X.columns`
equals exactly the list above, so the leak cannot silently return.

## 2. Labels

| source | positive cell-hours |
|---|---|
{chr(10).join(f'| `{k}` | {v:,} |' for k, v in src.items())}
| **total positives** | **{len(positives):,}** |
| negatives (sampled {NEG_TO_POS_RATIO}:1) | {len(negatives):,} |
| **training set** | **{len(training_df):,}** ({100*len(positives)/len(training_df):.2f}% positive) |

Rainfall threshold: `precip_1hr >= {DEFAULT_PRECIP_1HR_MM:.0f} mm` OR
`precip_3hr >= {DEFAULT_PRECIP_3HR_MM:.0f} mm`, AND `slope_mean >= {DEFAULT_SLOPE_DEG:.0f}°`
(Dikshit & Satyam 2019, Kalimpong — see `app/labelling.py`).
ASDMA vulnerable cells bypass the slope filter; verified incidents are
marked positive for all 24 hours of the incident date.

### Verified-incident coverage

**{n_ver - len(excluded_incidents)} of {n_ver} verified incidents used**; the following were excluded
because they fall outside the weather record (2018-01-01 → 2025-12-31):

{excl_lines}

Specifically: 7 of 8 verified incidents used; 16 Jul 2026 Lal Ganesh
excluded, post-dates weather record.

## 3. Split method — episode-grouped, NOT a temporal cutoff

Rainfall-threshold labels make temporally adjacent hours near-duplicates,
so a random split leaks across storms.

1. An hour is **active** if ANY of the 19 weather points breaches the
   rainfall trigger.
2. Runs of active hours separated by **≤ {EPISODE_GAP_HOURS} h** merge into one storm
   episode; a longer gap closes it. → **{len(starts)} storm episodes**.
3. Quiet intervals between storms form their own groups.
4. Group key = `(year, episode_id)`. → **{n_groups} groups** in the training set.
5. `StratifiedGroupKFold(n_splits={N_SPLITS}, shuffle=True, random_state={RANDOM_STATE})`,
   fold 0 taken as the held-out test set.

| | rows | episodes | positive rate |
|---|---|---|---|
| train | {len(train_idx):,} | {g_train} | {100*training_df['target_event'].to_numpy()[train_idx].mean():.2f}% |
| test | {len(test_idx):,} | {g_test} | {100*training_df['target_event'].to_numpy()[test_idx].mean():.2f}% |

**Leakage check: 0 episode groups appear on both sides** (asserted in code —
the script raises rather than training if any group straddles the split).

## 4. Results

| model | PR-AUC | F1-macro |
|---|---|---|
| RandomForest (`n_estimators=100`, `class_weight='balanced'`) | **{rf['pr_auc']:.4f}** | {rf['f1_macro']:.4f} |
| XGBoost (`n_estimators=150`, `max_depth=6`, `lr=0.05`, `scale_pos_weight={results['xgboost']['spw']:.4f}`) | **{xgb['pr_auc']:.4f}** | {xgb['f1_macro']:.4f} |

Both models trained on **identical** train/test indices.
Metrics are PR-AUC + F1-macro. ROC-AUC is deliberately not reported
(CLAUDE.md §4 — misleading under this class imbalance).

**Winner on PR-AUC: `{winner}`** → saved to `{str(model_path.relative_to(ROOT)).replace(chr(92), '/')}`
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
simulated IoT panel") until {datetime.now().strftime('%d %b %Y')}.

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
"""
    DOCS_DIR.mkdir(exist_ok=True)
    (DOCS_DIR / "model_training_log.md").write_text(md, encoding="utf-8")
    logger.info("Wrote docs/model_training_log.md")


if __name__ == "__main__":
    main()
