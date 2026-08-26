import numpy as np
import geopandas as gpd
from shapely.geometry import box


def generate_grid(boundary_gdf, cell_size_m=1000, metric_crs="EPSG:32646"):
    """
    Generate a square grid over a district boundary, clipped to the district polygon.

    Parameters
    ----------
    boundary_gdf : geopandas.GeoDataFrame
        District boundary (one or more polygons). Must have a CRS set.
    cell_size_m : float
        Grid cell size in metres.
    metric_crs : str
        Projected CRS used to build the grid so cell_size_m is accurate metres.
        Default EPSG:32646 (UTM zone 46N), appropriate for Kamrup Metropolitan
        (~91.7 deg E).

    Returns
    -------
    geopandas.GeoDataFrame
        Columns: grid_id, geometry, centroid_lat, centroid_lon.
        geometry is in EPSG:4326.
    """
    if boundary_gdf.crs is None:
        raise ValueError("boundary_gdf must have a CRS set")

    boundary_metric = boundary_gdf.to_crs(metric_crs)
    district_union = boundary_metric.union_all()

    minx, miny, maxx, maxy = district_union.bounds
    n_cols = int(np.ceil((maxx - minx) / cell_size_m))
    n_rows = int(np.ceil((maxy - miny) / cell_size_m))

    grid_ids = []
    cells = []
    for row in range(n_rows):
        y0 = miny + row * cell_size_m
        for col in range(n_cols):
            x0 = minx + col * cell_size_m
            cell = box(x0, y0, x0 + cell_size_m, y0 + cell_size_m)
            if cell.intersects(district_union):
                grid_ids.append(f"KM_R{row:03d}_C{col:03d}")
                cells.append(cell)

    grid_metric = gpd.GeoDataFrame({"grid_id": grid_ids}, geometry=cells, crs=metric_crs)

    # Clip each cell to the actual district shape (edge cells become partial polygons)
    grid_metric["geometry"] = grid_metric.geometry.intersection(district_union)
    grid_metric = grid_metric[~grid_metric.geometry.is_empty].reset_index(drop=True)

    # Centroid computed in metric CRS (true planar centroid), then reprojected as a point
    centroids_metric = gpd.GeoSeries(grid_metric.geometry.centroid, crs=metric_crs)
    centroids_4326 = centroids_metric.to_crs("EPSG:4326")

    result = gpd.GeoDataFrame(
        {
            "grid_id": grid_metric["grid_id"].values,
            "centroid_lat": centroids_4326.y.values,
            "centroid_lon": centroids_4326.x.values,
        },
        geometry=grid_metric.to_crs("EPSG:4326").geometry.values,
        crs="EPSG:4326",
    )

    return result[["grid_id", "geometry", "centroid_lat", "centroid_lon"]]
