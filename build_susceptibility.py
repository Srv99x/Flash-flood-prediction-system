"""
build_susceptibility.py
───────────────────────
Regenerate data/processed/susceptibility_features.parquet.

    python build_susceptibility.py

Two layers:
  1. Terrain-derived class from slope_mean (app/susceptibility_utils.py)
  2. An official-hazard FLOOR raising cells that contain an ASDMA
     vulnerable location or a verified incident to at least "High"

See app/susceptibility_utils.py for why layer 2 is necessary, and
docs/model_training_log.md section 8 for the disclosure and the
circularity note.
"""

from pathlib import Path

import pandas as pd
import geopandas as gpd

from app.susceptibility_utils import (
    apply_official_hazard_floor,
    cells_containing_points,
    classify_slope_susceptibility,
)

DATA = Path("data/processed")
RAW = Path("data/raw")
OUT = DATA / "susceptibility_features.parquet"


def main() -> None:
    grid = gpd.read_parquet(DATA / "kamrup_metro_grid_1km.parquet")
    terrain = pd.read_parquet(DATA / "terrain_features.parquet")

    # ── Layer 1: terrain ──────────────────────────────────────────────
    terrain["slope_mean"] = terrain["slope_mean"].astype("float64")
    terrain["terrain_class"] = terrain["slope_mean"].apply(classify_slope_susceptibility)

    # ── Layer 2: official hazard floor ────────────────────────────────
    asdma = pd.read_csv(RAW / "asdma_vulnerable_locations.csv")
    verified = pd.read_csv(RAW / "verified_incidents.csv")

    asdma_cells = cells_containing_points(asdma, grid)
    verified_cells = cells_containing_points(verified, grid)
    flagged = asdma_cells | verified_cells

    print(f"ASDMA:    {len(asdma)} points -> {len(asdma_cells)} cells")
    print(f"Verified: {len(verified)} points -> {len(verified_cells)} cells")
    print(f"Flagged union: {len(flagged)} cells ({100*len(flagged)/len(grid):.1f}% of district)")
    print(f"Verified cells subset of ASDMA cells: {verified_cells <= asdma_cells}")

    terrain["gsi_susceptibility_class"] = apply_official_hazard_floor(
        terrain["terrain_class"], terrain["grid_id"], flagged
    )
    terrain["is_proxy"] = True
    terrain["hazard_floor_applied"] = terrain["grid_id"].isin(flagged)

    out = terrain[
        ["grid_id", "gsi_susceptibility_class", "is_proxy", "hazard_floor_applied"]
    ].copy()
    out.to_parquet(OUT, index=False)

    # Compare on filled values: NaN != NaN would count all 93 no-DEM cells
    # as "changed" even when the floor left them untouched.
    n_lifted = int(
        (
            terrain["terrain_class"].fillna("__none__")
            != terrain["gsi_susceptibility_class"].fillna("__none__")
        ).sum()
    )
    print(f"\nSaved {OUT} ({len(out)} rows)")
    print(f"Cells raised by the hazard floor: {n_lifted}")
    print("\nTerrain-only class distribution:")
    print(terrain["terrain_class"].value_counts(dropna=False).to_string())
    print("\nFinal class distribution (after floor):")
    print(out["gsi_susceptibility_class"].value_counts(dropna=False).to_string())


if __name__ == "__main__":
    main()
