"""
trajectory_prediction.py

Provides deterministic and simple probabilistic trajectory predictors.

Available classes/functions:
 - CVPredictor(horizon=5.0, step=0.2)  : constant velocity predictor
 - CAPredictor(horizon=5.0, step=0.2)  : constant acceleration predictor (accepts ax,az)
 - UncertaintyPropagator: propagate KF covariance under linear CV model to get predicted covariances

All predictors operate in ground coordinates (X lateral, Z forward).
Prediction returns a list of tuples (t, X_t, Z_t) or (t, mu_vec, Sigma_matrix) for uncertainty propagation.
"""

from typing import List, Tuple, Optional
import numpy as np

class CVPredictor:
    def __init__(self, horizon: float = 5.0, step: float = 0.2):
        self.horizon = float(horizon)
        self.step = float(step)

    def predict(self, X: float, Z: float, vx: float, vz: float) -> List[Tuple[float,float,float]]:
        """
        Predict using constant velocity model.
        Returns list of (t, X_t, Z_t) for t in [0..horizon] step.
        """
        out = []
        t = 0.0
        while t <= self.horizon + 1e-9:
            xp = X + vx * t
            zp = Z + vz * t
            out.append((t, float(xp), float(zp)))
            t += self.step
        return out


class CAPredictor:
    def __init__(self, horizon: float = 5.0, step: float = 0.2):
        self.horizon = float(horizon)
        self.step = float(step)

    def predict(self, X: float, Z: float, vx: float, vz: float, ax: float = 0.0, az: float = 0.0) -> List[Tuple[float,float,float]]:
        """
        Predict using constant acceleration model.
        ax, az are lateral / longitudinal accelerations (m/s^2)
        """
        out = []
        t = 0.0
        while t <= self.horizon + 1e-9:
            xp = X + vx * t + 0.5 * ax * t * t
            zp = Z + vz * t + 0.5 * az * t * t
            out.append((t, float(xp), float(zp)))
            t += self.step
        return out


class UncertaintyPropagator:
    """
    Propagate mean and covariance under linear CV model:
    State vector: [X, Z, vx, vz]
    F(dt) = [[1,0,dt,0],
             [0,1,0,dt],
             [0,0,1,0],
             [0,0,0,1]]
    Q(dt) = process noise matrix (tunable)
    """

    def __init__(self, process_q: float = 1.0):
        self.process_q = float(process_q)

    def _F(self, dt: float) -> np.ndarray:
        F = np.array([
            [1.0, 0.0, dt,  0.0],
            [0.0, 1.0, 0.0, dt ],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ], dtype=float)
        return F

    def _Q(self, dt: float) -> np.ndarray:
        q = self.process_q
        Q = q * np.array([
            [dt**4/4, 0, dt**3/2, 0],
            [0, dt**4/4, 0, dt**3/2],
            [dt**3/2, 0, dt**2, 0],
            [0, dt**3/2, 0, dt**2]
        ], dtype=float)
        return Q

    def propagate(self, mu0: np.ndarray, P0: np.ndarray, horizon: float = 5.0, step: float = 0.2) -> List[Tuple[float, np.ndarray, np.ndarray]]:
        """
        Propagate mean and covariance forward under CV model.
        mu0: initial mean shape (4,) = [X, Z, vx, vz]
        P0: initial covariance shape (4,4)
        Returns list of (t, mu_t, P_t)
        """
        results = []
        t = 0.0
        mu = mu0.copy()
        P = P0.copy()
        while t <= horizon + 1e-9:
            results.append((t, mu.copy(), P.copy()))
            # step forward
            F = self._F(step)
            Q = self._Q(step)
            mu = F.dot(mu)
            P = F.dot(P).dot(F.T) + Q
            t += step
        return results
