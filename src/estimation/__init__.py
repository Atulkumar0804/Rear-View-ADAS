"""
Estimation module
-----------------
Kalman filtering, velocity estimation, and trajectory prediction.
"""

from .kalman_filter import MultiKalman
from .relative_velocity import (
    depth_difference_velocity,
    EgoMotionEstimator,
    compute_relative_velocity_from_depths
)
from .trajectory_prediction import CVPredictor, CAPredictor, UncertaintyPropagator

__all__ = [
    "MultiKalman",
    "depth_difference_velocity",
    "EgoMotionEstimator",
    "compute_relative_velocity_from_depths",
    "CVPredictor",
    "CAPredictor",
    "UncertaintyPropagator"
]
