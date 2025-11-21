"""
Tracking module
---------------
Multi-object tracking with IOU+Kalman fallback and ByteTrack support.
"""

from .tracker import Tracker, Track
from .byte_tracker_utils import xyxy_to_xywh, xywh_to_xyxy, iou_xyxy, iou_matrix

__all__ = ["Tracker", "Track", "xyxy_to_xywh", "xywh_to_xyxy", "iou_xyxy", "iou_matrix"]
