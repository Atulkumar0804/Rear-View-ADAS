"""
ttc_calculation.py
------------------
Compute Time-To-Collision (TTC) safely and robustly.

Definitions:
 - Z  = forward distance (meters)
 - vz = relative approach speed in ground frame (m/s)
        (positive vz means object approaching the ego vehicle from behind)

TTC = Z / vz

If vz <= 0 → TTC = +inf (no collision coming toward ego).
"""

import math


def compute_ttc(Z: float, vz: float) -> float:
    """
    Computes TTC = Z / vz (seconds).
    Returns math.inf if vz <= 0 or Z <= 0.

    Args:
        Z: forward distance (m)
        vz: relative speed (m/s), positive means object is approaching ego.

    Returns:
        TTC (float)
    """
    if vz <= 0:
        return math.inf
    if Z <= 0:
        return 0.0
    return float(Z / vz)


def classify_ttc(ttc: float,
                 critical_ttc: float = 1.5,
                 warning_ttc: float = 4.0) -> str:
    """
    Classify TTC into NONE, WARN, CRITICAL levels.

    Args:
        ttc: Time-to-Collision
        critical_ttc: threshold for critical warning (seconds)
        warning_ttc: threshold for early warning

    Returns:
        One of: "NONE", "WARN", "CRITICAL"
    """
    if ttc < critical_ttc:
        return "CRITICAL"
    if ttc < warning_ttc:
        return "WARN"
    return "NONE"
