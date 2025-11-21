"""
kalman_filter.py

Multi-track Kalman wrapper for ground-frame state [X, Z, vx, vz].

API:
    K = MultiKalman(dt=0.033)
    state = K.predict_and_update(track_id, meas_x, meas_z, dt=dt)
    # returns (x, z, vx, vz)

If filterpy is installed it uses filterpy.kalman.KalmanFilter for better numeric behavior.
Otherwise uses a simple built-in linear-KF implementation per track.

Notes:
 - State vector: [X, Z, vx, vz]^T
 - Motion model: constant velocity
 - Measurement: [X_meas, Z_meas]
"""

import time
import math
import os
from typing import Tuple, Dict, Optional

import numpy as np

try:
    from filterpy.kalman import KalmanFilter
    FILTERPY_AVAILABLE = True
except Exception:
    KalmanFilter = None
    FILTERPY_AVAILABLE = False


class MultiKalman:
    def __init__(self, dt: float = 1.0/30.0, process_q: float = 1.0, meas_var_x: float = 0.5, meas_var_z: float = 0.8):
        """
        dt: nominal timestep (s)
        process_q: process noise scale
        meas_var_x, meas_var_z: measurement variances for X and Z
        """
        self.dt = float(dt)
        self.process_q = float(process_q)
        self.meas_var_x = float(meas_var_x)
        self.meas_var_z = float(meas_var_z)

        # Map track_id -> filter object (either FilterPy KalmanFilter or dict for fallback)
        self.filters = {}  # track_id -> object
        self.last_t = {}   # track_id -> last update time

    def _create_filterpy(self, init_x: float, init_z: float) -> KalmanFilter:
        kf = KalmanFilter(dim_x=4, dim_z=2)
        dt = self.dt
        # State: [X, Z, vx, vz]
        kf.F = np.array([
            [1.0, 0.0, dt,  0.0],
            [0.0, 1.0, 0.0, dt ],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ], dtype=float)
        kf.H = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0]
        ], dtype=float)
        # initial state
        kf.x = np.array([init_x, init_z, 0.0, 0.0], dtype=float)
        # Covariances
        kf.P = np.eye(4) * 5.0
        q = self.process_q
        # simple Q
        kf.Q = q * np.array([
            [dt**4/4, 0, dt**3/2, 0],
            [0, dt**4/4, 0, dt**3/2],
            [dt**3/2, 0, dt**2, 0],
            [0, dt**3/2, 0, dt**2]
        ], dtype=float)
        kf.R = np.diag([self.meas_var_x, self.meas_var_z])
        return kf

    def _create_fallback(self, init_x: float, init_z: float):
        # Simple state dictionary for fallback KF-like smoothing
        return {
            "x": float(init_x),
            "z": float(init_z),
            "vx": 0.0,
            "vz": 0.0,
            "P": np.eye(4) * 10.0  # rough covariance
        }

    def _ensure_filter(self, track_id: int, init_x: float, init_z: float):
        if track_id in self.filters:
            return
        if FILTERPY_AVAILABLE:
            self.filters[track_id] = self._create_filterpy(init_x, init_z)
        else:
            self.filters[track_id] = self._create_fallback(init_x, init_z)
        self.last_t[track_id] = time.time()

    def predict_and_update(self, track_id: int, meas_x: float, meas_z: float, dt: Optional[float] = None) -> Tuple[float,float,float,float]:
        """
        Predict (optionally using dt) and update with measurement (meas_x, meas_z).
        Returns state tuple (x, z, vx, vz).
        """
        now = time.time()
        if dt is None:
            # compute dt from last update if available, else use nominal
            dt = now - self.last_t.get(track_id, now - self.dt)
            dt = max(1e-3, min(1.0, dt))  # clamp
        self._ensure_filter(track_id, meas_x, meas_z)

        if FILTERPY_AVAILABLE:
            kf: KalmanFilter = self.filters[track_id]
            # Update F for current dt
            kf.F[0,2] = dt
            kf.F[1,3] = dt
            # update process noise Q for dt if desired
            q = self.process_q
            kf.Q = q * np.array([
                [dt**4/4, 0, dt**3/2, 0],
                [0, dt**4/4, 0, dt**3/2],
                [dt**3/2, 0, dt**2, 0],
                [0, dt**3/2, 0, dt**2]
            ], dtype=float)
            # predict & update
            kf.predict()
            z = np.array([meas_x, meas_z])
            kf.update(z)
            x = float(kf.x[0]); zt = float(kf.x[1]); vx = float(kf.x[2]); vz = float(kf.x[3])
            self.last_t[track_id] = now
            return x, zt, vx, vz
        else:
            # fallback simple predictive update (very approximate)
            f = self.filters[track_id]
            # predict
            f["x"] += f["vx"] * dt
            f["z"] += f["vz"] * dt
            # measurement residuals
            rx = meas_x - f["x"]
            rz = meas_z - f["z"]
            # simple gain heuristics
            k_pos = 0.5
            k_vel = 0.3
            # update position
            f["x"] += k_pos * rx
            f["z"] += k_pos * rz
            # update velocity (finite diff blended)
            vx_new = rx / max(dt, 1e-3)
            vz_new = rz / max(dt, 1e-3)
            f["vx"] = (1.0 - k_vel) * f["vx"] + k_vel * vx_new
            f["vz"] = (1.0 - k_vel) * f["vz"] + k_vel * vz_new
            self.last_t[track_id] = now
            return float(f["x"]), float(f["z"]), float(f["vx"]), float(f["vz"])

    def remove(self, track_id: int):
        if track_id in self.filters:
            del self.filters[track_id]
        if track_id in self.last_t:
            del self.last_t[track_id]

    def clear(self):
        self.filters.clear()
        self.last_t.clear()
