import numpy as np


def classify_slope_susceptibility(slope_mean):
    """
    Assign a landslide-susceptibility proxy class from mean slope.

    This is NOT the official GSI NLSM classification.
    It is a documented proxy derived from published slope-risk
    relationships for Northeast India / Guwahati terrain.

    Classes:
        < 15 degrees   -> Low
        15-30 degrees  -> Moderate
        30-45 degrees  -> High
        > 45 degrees   -> Very High

    Parameters
    ----------
    slope_mean : float
        Mean slope of a grid cell in degrees.

    Returns
    -------
    str or None
        Proxy susceptibility class, or None for missing values.
    """
    if slope_mean is None or not np.isfinite(slope_mean):
        return None

    if slope_mean < 15:
        return "Low"
    elif slope_mean <= 30:
        return "Moderate"
    elif slope_mean <= 45:
        return "High"
    else:
        return "Very High"