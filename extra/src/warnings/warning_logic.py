"""
warning_logic.py
-----------------
Central module for combining:
 - TTC evaluation
 - distance thresholds
 - lateral overlap test
 - predicted trajectory intersection
 - final warning decision

Compatible with:
 - trajectory_prediction.py
 - ttc_calculation.py
 - GroundProjector outputs (X, Z)
"""

import yaml
import os
import math

from .ttc_calculation import compute_ttc, classify_ttc


def load_yaml(path, default=None):
    if os.path.exists(path):
        with open(path, "r") as f:
            return yaml.safe_load(f)
    return default if default else {}


class WarningSystem:
    """
    Compute risk levels for each tracked object.

    Inputs:
      X, Z, vx, vz  → current ground-frame state
      predictions   → list of (t, X_t, Z_t)
    """

    def __init__(self, config_path="config/warning_config.yaml", cfg=None):
        # Load config
        self.cfg = load_yaml(config_path, {})
        if cfg:
            self.cfg.update(cfg)

        self.crit_ttc = float(self.cfg.get("critical_ttc", 1.5))
        self.warn_ttc = float(self.cfg.get("warn_ttc", 4.0))

        self.crit_dist = float(self.cfg.get("critical_distance", 3.0))
        self.warn_dist = float(self.cfg.get("warning_distance", 6.0))

        self.lat_overlap_thresh = float(self.cfg.get("lateral_overlap_threshold", 0.5))

        # ego assumed to be centered at X = 0
        self.ego_half_width = 0.6  # approximate scooter + lane wobble margin

    # ---------------------------------------
    # LATERAL OVERLAP CHECK
    # ---------------------------------------
    def _compute_lateral_overlap(self, obj_left, obj_right, ego_left, ego_right):
        return max(0.0, min(obj_right, ego_right) - max(obj_left, ego_left))

    def lateral_overlap(self, X: float, object_half_width=0.8):
        """
        Returns True if object laterally overlaps with ego's corridor.

        object_half_width = approx vehicle half width (car/motorcycle)
        """
        obj_left = X - object_half_width
        obj_right = X + object_half_width

        ego_left = -self.ego_half_width
        ego_right = self.ego_half_width

        overlap = self._compute_lateral_overlap(obj_left, obj_right, ego_left, ego_right)
        return overlap >= self.lat_overlap_thresh

    # ---------------------------------------
    # PATH INTERSECTION USING PREDICTED TRAJECTORY
    # ---------------------------------------
    def will_paths_intersect(self, predictions):
        """
        Returns True if predicted trajectory intersects ego corridor
        in a future time window.
        """
        ego_left = -self.ego_half_width
        ego_right = self.ego_half_width

        for (t, Xp, Zp) in predictions:
            if Zp < 0:
                continue
            # consider dangerous zone: Z small (within 0–2 meters)
            if 0 < Zp < 2.5:
                obj_left = Xp - 0.8
                obj_right = Xp + 0.8
                overlap = self._compute_lateral_overlap(obj_left, obj_right, ego_left, ego_right)
                if overlap >= self.lat_overlap_thresh:
                    return True
        return False

    # ---------------------------------------
    # FINAL DECISION LOGIC
    # ---------------------------------------
    def decide(self, X, Z, vx, vz, predictions):
        """
        Returns: (warning_level, TTC)

        warning_level ∈ {"NONE", "WARN", "CRITICAL"}
        """

        # --- TTC computation ---
        ttc = compute_ttc(Z, vz)
        ttc_level = classify_ttc(ttc, self.crit_ttc, self.warn_ttc)

        # --- Proximity distance checks ---
        if Z < self.crit_dist:
            return "CRITICAL", ttc
        if Z < self.warn_dist:
            return "WARN", ttc

        # --- Lateral overlap at current frame ---
        lat_overlap_now = self.lateral_overlap(X)

        # If object overlaps ego corridor and closing fast
        if lat_overlap_now and ttc_level != "NONE":
            return ttc_level, ttc

        # --- Predicted path intersection ---
        if self.will_paths_intersect(predictions):
            return "CRITICAL", ttc

        # --- TTC alone may still generate warning ---
        if ttc_level != "NONE":
            return ttc_level, ttc

        return "NONE", ttc
