"""
generate_synthetic_grid.py
──────────────────────────
One-off script to create a synthetic 904-cell 1 km grid over Kamrup
Metropolitan District, Assam.

Run once:
    python3 app/generate_synthetic_grid.py

Output:
    data/processed/kamrup_metro_grid_1km.parquet

NOTE: This synthetic grid is a geographic placeholder.
      Replace with the real grid file from the shared Google Drive when available.
      The real file has identical columns: grid_id, geometry, centroid_lat, centroid_lon
      and CRS EPSG:4326.
"""

import numpy as np
import geopandas as gpd
from shapely.geometry import box
import os

# ── Bounding box for Kamrup Metro District (approx.) ──────────────────────────
# Guwahati city is roughly 26.0–26.25°N, 91.5–91.95°E
# We extend a bit to reach ~904 cells at 0.009° ≈ 1 km resolution
LAT_MIN, LAT_MAX = 25.98, 26.42
LON_MIN, LON_MAX = 91.40, 92.00

# Cell size in degrees (≈ 1 km at this latitude)
CELL_DEG = 0.009  # ~1 km

TARGET_ROWS = 904

np.random.seed(42)


def generate_grid(lat_min, lat_max, lon_min, lon_max, cell_deg, target=TARGET_ROWS):
    """Generate a rectangular 1-km grid of polygons over the bounding box."""
    lats = np.arange(lat_min, lat_max, cell_deg)
    lons = np.arange(lon_min, lon_max, cell_deg)

    rows = []
    gid = 1
    for lat in lats:
        for lon in lons:
            geom = box(lon, lat, lon + cell_deg, lat + cell_deg)
            centroid = geom.centroid
            rows.append({
                "grid_id": f"KM_{gid:04d}",
                "geometry": geom,
                "centroid_lat": round(centroid.y, 6),
                "centroid_lon": round(centroid.x, 6),
            })
            gid += 1
            if gid > target:
                break
        if gid > target:
            break

    return gpd.GeoDataFrame(rows, crs="EPSG:4326")


if __name__ == "__main__":
    out_path = os.path.join(
        os.path.dirname(__file__), "..", "data", "processed",
        "kamrup_metro_grid_1km.parquet"
    )
    out_path = os.path.abspath(out_path)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    print("Generating synthetic 1 km grid …")
    gdf = generate_grid(LAT_MIN, LAT_MAX, LON_MIN, LON_MAX, CELL_DEG)
    print(f"  → {len(gdf)} cells generated")
    gdf.to_parquet(out_path, index=False)
    print(f"  → Saved to {out_path}")
    print("Done.")
