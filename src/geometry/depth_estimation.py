"""
depth_estimation.py
-------------------
Depth estimation from bounding-box size.
Used when YOLO detection provides bounding boxes and vehicle type.

Formula:
    Z = f * H_real / h_img

Works best for mid-range (3–25 m).
"""

import yaml
import os


def load_yaml(path, default=None):
    if os.path.exists(path):
        with open(path, "r") as f:
            return yaml.safe_load(f)
    return default if default is not None else {}


DEFAULT_CLASS_HEIGHTS = {
    "car": 1.5,
    "truck": 3.0,
    "bus": 3.0,
    "motorcycle": 1.1,
    "scooter": 1.1,
    "other": 1.6
}


class DepthEstimator:
    def __init__(self, camera_path="config/camera_config.yaml"):
        cam_cfg = load_yaml(camera_path, default={
            "fy": 1000.0
        })

        self.f = float(cam_cfg.get("fy", 1000.0))
        self.class_heights = DEFAULT_CLASS_HEIGHTS

        print(f"[DepthEstimator] Using fy={self.f} px")

    def bbox_depth(self, bbox, cls_name):
        """
        Args:
            bbox: (x1,y1,x2,y2)
            cls_name: class string, e.g., 'car', 'bus', 'motorcycle'
        Returns:
            Z depth in meters (float)
        """
        x1, y1, x2, y2 = bbox
        h_img = max(2, float(y2 - y1))

        H_real = self.class_heights.get(cls_name, 1.6)
        Z = (self.f * H_real) / h_img

        return float(Z)
