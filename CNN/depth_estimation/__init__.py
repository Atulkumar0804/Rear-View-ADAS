"""
Depth Estimation Module
Monocular depth estimation using Depth-Anything-V2
"""

from .depth_estimator import DepthEstimator, VehicleDepthTracker
from .depth_config import DepthConfig

__all__ = ['DepthEstimator', 'VehicleDepthTracker', 'DepthConfig']
