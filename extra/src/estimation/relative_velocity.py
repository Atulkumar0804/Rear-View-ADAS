"""
relative_velocity.py

Helpers to compute relative velocity between ego-vehicle (scooter) and tracked objects.

Provided utilities:
 - v_rel = depth_difference_velocity(Z_prev, Z_now, dt)
 - EgoMotionEstimator: lightweight optical-flow-based estimator to get approximate ego forward speed (m/s)
 - compute_relative_velocity_from_track: convenience combining object velocity and ego speed

Notes on EgoMotionEstimator:
 - It uses cv2.goodFeaturesToTrack + cv2.calcOpticalFlowPyrLK to measure vertical pixel motion (dv).
 - We convert the median pixel vertical motion into delta-Z (meters) using the
   ground-projection linearization:
       Z = fy * h / (v - cy)
       dZ ≈ - fy * h / (v - cy)^2 * dv
   where dv is change in pixel row (positive downward).
 - This is approximate and assumes background points are on the road and dominant flow is from ego forward motion.
"""

import time
from typing import Optional, Tuple, List

import numpy as np
import cv2

def depth_difference_velocity(Z_prev: float, Z_now: float, dt: float) -> float:
    """
    Simple relative velocity from depth change (m/s).
    Positive if object is closing (Z decreasing -> positive approach speed means (Z_prev - Z_now)/dt > 0).
    We'll define v_rel = (Z_prev - Z_now) / dt so positive means approaching.
    """
    if dt <= 0:
        return 0.0
    return float((Z_prev - Z_now) / dt)


class EgoMotionEstimator:
    """
    Estimate ego forward motion (m/s) from optical flow between consecutive frames.

    Usage:
        e = EgoMotionEstimator(fy_px, camera_height_m, cy_px)
        ego_v = e.update(prev_gray, curr_gray, dt)  # returns m/s (positive forward), or None if failed

    Important: result is approximate. Use IMU / wheel-speed when available for production.
    """
    def __init__(self, fy_px: float, camera_height_m: float, cy_px: float,
                 max_corners: int = 800, quality_level: float = 0.01, min_distance: float = 7.0):
        self.fy = float(fy_px)
        self.h = float(camera_height_m)
        self.cy = float(cy_px)
        self.max_corners = int(max_corners)
        self.quality_level = float(quality_level)
        self.min_distance = float(min_distance)
        self.prev_gray = None
        self.prev_pts = None
        self.prev_time = None

    def reset(self):
        self.prev_gray = None
        self.prev_pts = None
        self.prev_time = None

    def _pixel_dv_to_dz(self, v_pixel: float, dv: float) -> float:
        """
        Convert pixel vertical shift dv (px) at image row v_pixel into approximate delta Z (meters)
        using derivative of Z = fy*h / (v - cy):
            dZ/dv = -fy*h / (v - cy)^2
        so dZ ≈ dZ/dv * dv
        Note dv positive downward; if dv > 0 and camera is moving forward, dv for background points tends to be positive,
        leading to negative dZ (camera moving closer to background) — we return signed dZ.
        """
        denom = (v_pixel - self.cy)
        if abs(denom) < 1e-6:
            denom = 1e-6 if denom >= 0 else -1e-6
        deriv = - (self.fy * self.h) / (denom * denom)
        return deriv * dv

    def update(self, prev_gray: np.ndarray, curr_gray: np.ndarray, dt: float) -> Optional[float]:
        """
        Estimate ego forward speed in m/s from prev and curr grayscale frames and time delta dt.
        Returns forward speed (positive = moving forward), or None if estimation failed.
        """
        if prev_gray is None or curr_gray is None:
            return None
        if dt <= 0 or prev_gray.shape != curr_gray.shape:
            return None

        # detect good features in prev frame
        pts = cv2.goodFeaturesToTrack(prev_gray,
                                      maxCorners=self.max_corners,
                                      qualityLevel=self.quality_level,
                                      minDistance=self.min_distance)
        if pts is None or len(pts) < 10:
            return None

        # compute flow
        pts = pts.reshape(-1,1,2)
        next_pts, st, err = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, pts, None,
                                                     winSize=(21,21), maxLevel=3,
                                                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
        if next_pts is None:
            return None

        # keep valid tracks
        pts = pts.squeeze()
        next_pts = next_pts.squeeze()
        st = st.squeeze()
        valid = st == 1
        if np.sum(valid) < 8:
            return None

        pts = pts[valid]
        next_pts = next_pts[valid]

        # compute dv for each tracked point (vertical displacement)
        dv = next_pts[:,1] - pts[:,1]   # positive downward
        v_pixel = pts[:,1]              # approximate pixel row where displacement measured

        # filter by small & large motions (remove outliers using median absolute deviation)
        med = np.median(dv)
        mad = np.median(np.abs(dv - med)) + 1e-6
        keep_mask = np.abs(dv - med) < (3.0 * mad)
        if np.sum(keep_mask) < 6:
            # fallback to using median directly
            dv_med = med
            v_med = float(np.median(v_pixel))
        else:
            dv_med = float(np.median(dv[keep_mask]))
            v_med = float(np.median(v_pixel[keep_mask]))

        # convert to delta Z (meters)
        dZ = self._pixel_dv_to_dz(v_med, dv_med)   # meters (signed)
        # ego forward speed: camera moves forward -> background appears to move downward (dv>0) resulting in negative dZ
        # We want positive forward speed (m/s). If dZ < 0 means camera got closer to background => forward motion positive.
        ego_speed = - dZ / max(dt, 1e-6)
        # sanity clamp to realistic scooter speeds
        ego_speed = float(np.clip(ego_speed, -40.0, 40.0))
        return ego_speed


def compute_relative_velocity_from_depths(Z_prev: float, Z_now: float, dt: float, ego_speed: Optional[float] = None) -> float:
    """
    Compute relative approach velocity (m/s) from depth change, optionally subtracting ego forward speed.

    Definition:
      v_rel = (Z_prev - Z_now) / dt - ego_speed

    If ego_speed is None we simply return (Z_prev - Z_now)/dt.
    Positive v_rel means the object is approaching (closing).
    """
    v_rel_obj = depth_difference_velocity(Z_prev, Z_now, dt) if dt > 0 else 0.0
    if ego_speed is None:
        return float(v_rel_obj)
    # If ego_speed is available (forward positive), subtract it to get relative speed of object wrt ego:
    # If object is stationary in world (v_obj_world = 0) and ego moves forward at +5 m/s, then Z will reduce -> v_rel_obj ≈ +5.
    # We want v_rel = v_obj_world - v_ego_world. Rearrangement gives approx v_rel = v_rel_obj - ego_speed
    return float(v_rel_obj - ego_speed)


# small helper
def depth_difference_velocity(Z_prev: float, Z_now: float, dt: float) -> float:
    return (Z_prev - Z_now) / (dt if dt > 1e-6 else 1e-6)
