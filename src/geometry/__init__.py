"""
Geometry module
---------------
Ground-plane projection, depth estimation, and camera calibration.
"""

from .projection import GroundProjector
from .depth_estimation import DepthEstimator
from .calibration import calibrate_checkerboard, save_intrinsics

__all__ = ["GroundProjector", "DepthEstimator", "calibrate_checkerboard", "save_intrinsics"]
