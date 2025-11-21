"""
Detection module
----------------
YOLO-based object detection with fallback motion detector.
"""

from .detector import Detector, Detection
from .yolo_loader import YOLOLoader

__all__ = ["Detector", "Detection", "YOLOLoader"]
