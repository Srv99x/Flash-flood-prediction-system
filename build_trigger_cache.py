"""
build_trigger_cache.py
──────────────────────
Precompute daily-peak trigger probability for every weather point and every
date in the record, so the Streamlit date picker renders instantly.

    python build_trigger_cache.py

Output: data/processed/trigger_prob_daily.parquet
        columns: weather_point_id, date, trigger_prob

Why this collapses so far
─────────────────────────
All five model features are WEATHER-POINT level, not cell level:

    soil_moisture_0_7, soil_moisture_7_28, temp_c, api_3d, api_7d

No terrain feature reaches the model (that is the leak fix). So the trigger
probability is identical for every grid cell sharing a weather point, and
the whole 2018-2025 range reduces from

    904 cells x 70,128 hours = 63.4M model evaluations

to 19 weather points x 2,922 days = 55,518 cached rows, under 1 MB.

The app then renders any date as a lookup and a multiply:

    final_risk[cell] = trigger_prob[weather_point(cell), date]
                       * susceptibility_multiplier[cell]

which is exactly what app/predict.py computes, so the cache cannot drift
from the live path as long as both use the same model and multipliers.
The consistency check at the end of this script asserts that.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from app.predict import EXPECTED_FEATURES, load_model

DATA = Path("data/processed")
OUT = DATA / "trigger_prob_daily.parquet"


def main() -> None:
    model = load_model()

    weather = pd.read_parquet(
        DATA / "weather_hourly.parquet",
        columns=["weather_point_id", "timestamp"] + EXPECTED_FEATURES,
    )
    # float32 to match training (app/train_trigger_model.py)
    weather[EXPECTED_FEATURES] = weather[EXPECTED_FEATURES].astype("float32")
    print(
        f"Weather: {len(weather):,} rows, "
        f"{weather.weather_point_id.nunique()} weather points, "
        f"{weather.timestamp.min().date()} -> {weather.timestamp.max().date()}"
    )

    print("Scoring hourly rows ...")
    weather["trigger_prob"] = model.predict_proba(weather[EXPECTED_FEATURES])[:, 1]

    # Daily peak: what a daily warning product would issue for that date.
    weather["date"] = weather["timestamp"].dt.normalize()
    daily = (
        weather.groupby(["weather_point_id", "date"], as_index=False)["trigger_prob"]
        .max()
    )
    daily.to_parquet(OUT, index=False)

    print(f"\nSaved {OUT}")
    print(f"  rows: {len(daily):,}  ({daily.weather_point_id.nunique()} WPs "
          f"x {daily.date.nunique():,} dates)")
    print(f"  size: {OUT.stat().st_size/1024:.0f} KB")
    print(f"  trigger_prob: min {daily.trigger_prob.min():.4f} "
          f"mean {daily.trigger_prob.mean():.4f} max {daily.trigger_prob.max():.4f}")

    nonzero = (daily.trigger_prob > 0.25).groupby(daily.date.dt.year).sum()
    print("\nDates with trigger_prob > 0.25, per year (any weather point):")
    print(nonzero.to_string())

    # ── Consistency check against the live predict path ───────────────
    print("\nConsistency check vs app.predict.predict_risk() ...")
    from app.features import build_feature_table
    from app.predict import predict_risk

    probe_date = "2020-05-26"
    feats = build_feature_table(
        DATA, start_date=f"{probe_date} 00:00:00",
        end_date=f"{probe_date} 23:00:00", monsoon_only=False,
    )
    live = (
        predict_risk(feats)
        .groupby("grid_id", as_index=False)["final_risk_score"].max()
        .sort_values("grid_id")
        .reset_index(drop=True)
    )

    mapping = pd.read_parquet(DATA / "grid_weather_mapping.parquet",
                              columns=["grid_id", "weather_point_id"])
    sus = pd.read_parquet(DATA / "susceptibility_features.parquet")
    from app.predict import SUSCEPTIBILITY_MULTIPLIERS

    cached = mapping.merge(
        daily[daily.date == pd.Timestamp(probe_date)], on="weather_point_id", how="left"
    ).merge(sus[["grid_id", "gsi_susceptibility_class"]], on="grid_id", how="left")
    cached["mult"] = cached["gsi_susceptibility_class"].map(
        SUSCEPTIBILITY_MULTIPLIERS
    ).fillna(0.0)
    cached["final_risk_score"] = cached["trigger_prob"] * cached["mult"]
    cached = cached.sort_values("grid_id").reset_index(drop=True)

    delta = np.abs(
        cached["final_risk_score"].to_numpy() - live["final_risk_score"].to_numpy()
    ).max()
    print(f"  max |cached - live| over 904 cells on {probe_date}: {delta:.10f}")
    assert delta < 1e-6, f"Cache diverges from live predict path by {delta}"
    print("  PASS -- cache matches the live predict path")


if __name__ == "__main__":
    main()
