"""
projection.py
-------------
Maps image bbox bottom point -> ground-plane (X, Z) coordinates
using pinhole camera geometry.

Assumptions:
 - Flat ground plane.
 - Camera height and pitch known.
 - Rotation about pitch axis only (no roll/yaw errors).

Used by pipeline.py for distance & trajectory estimation.
"""

import math
import yaml
import os


def load_yaml(path, default=None):
    if os.path.exists(path):
        with open(path, "r") as f:
            return yaml.safe_load(f)
    return default if default is not None else {}


class GroundProjector:
    def __init__(self, config_path="config/camera_config.yaml", cfg=None):
        # Load default + override
        self.cfg = load_yaml(config_path, {})
        if cfg:
            self.cfg.update(cfg)

        # Focal lengths
        self.fx = float(self.cfg.get("fx", 1000.0))
        self.fy = float(self.cfg.get("fy", 1000.0))

        # Principal point
        self.cx = float(self.cfg.get("cx", 640.0))
        self.cy = float(self.cfg.get("cy", 360.0))

        # Camera mounting parameters
        self.h = float(self.cfg.get("mounting_height", 1.0))  # meters
        self.pitch_deg = float(self.cfg.get("pitch_deg", 0.0))
        self.pitch = math.radians(self.pitch_deg)

        print(f"[GroundProjector] fx={self.fx}, fy={self.fy}, h={self.h}, pitch={self.pitch_deg} deg")

    def bottom_to_ground(self, bbox):
        """
        bbox = (x1, y1, x2, y2)
        Returns:
            X_ground (meters), Z_ground (meters)
        """

        x1, y1, x2, y2 = bbox

        # Bottom-center pixel of bbox
        u = (x1 + x2) / 2.0
        v = y2

        # Vertical ray direction (with pitch compensation)
        v_corrected = v - self.cy

        # Basic pinhole ground projection:
        denom = v_corrected
        if abs(denom) < 1e-6:
            denom = 1e-6

        # Z forward distance (meters)
        Z = abs((self.fy * self.h) / denom)

        # Simple lateral X:
        X = (u - self.cx) * Z / self.fx

        return float(X), float(Z)
