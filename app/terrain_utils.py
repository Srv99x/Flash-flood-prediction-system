import numpy as np


def compute_slope(dem_array, cellsize):
    """
    Compute terrain slope from a DEM elevation array.

    Parameters
    ----------
    dem_array : numpy.ndarray
        Elevation values in metres. NoData values should be represented
        as np.nan.
    cellsize : float or tuple
        Cell size in metres. If a single value is provided,
        it is used for both X and Y directions.

    Returns
    -------
    numpy.ndarray
        Slope values in degrees.
    """

    if isinstance(cellsize, tuple):
        cellsize_y, cellsize_x = cellsize
    else:
        cellsize_y = cellsize_x = cellsize

    dem = dem_array.astype(float)

    gradient_y, gradient_x = np.gradient(
        dem,
        cellsize_y,
        cellsize_x
    )

    gradient_magnitude = np.sqrt(
        gradient_x**2 + gradient_y**2
    )

    slope = np.degrees(
        np.arctan(gradient_magnitude)
    )

    return slope