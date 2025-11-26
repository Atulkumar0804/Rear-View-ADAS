#!/usr/bin/env python3
"""
Advanced trajectory prediction and collision path detection.
Predicts if vehicles are on collision course with camera vehicle.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict
from collections import deque
import cv2


class TrajectoryPredictor:
    """Predict vehicle trajectories and collision paths."""
    
    def __init__(self, history_size=10):
        """
        Initialize trajectory predictor.
        
        Args:
            history_size: Number of frames to track for trajectory
        """
        self.history_size = history_size
        self.vehicle_tracks = {}  # track_id -> deque of positions
        
    def update_track(self, track_id: int, bbox: List[int], timestamp: float):
        """
        Update vehicle track with new position.
        
        Args:
            track_id: Unique vehicle ID
            bbox: Bounding box [x1, y1, x2, y2]
            timestamp: Frame timestamp
        """
        if track_id not in self.vehicle_tracks:
            self.vehicle_tracks[track_id] = deque(maxlen=self.history_size)
        
        # Store center point and timestamp
        center_x = (bbox[0] + bbox[2]) / 2
        center_y = (bbox[1] + bbox[3]) / 2
        
        self.vehicle_tracks[track_id].append({
            'x': center_x,
            'y': center_y,
            'timestamp': timestamp,
            'bbox': bbox
        })
    
    def predict_trajectory(self, track_id: int, predict_frames: int = 30) -> Optional[List[Tuple[float, float]]]:
        """
        Predict future trajectory of vehicle.
        
        Args:
            track_id: Vehicle ID
            predict_frames: Number of frames to predict ahead
            
        Returns:
            List of (x, y) predicted positions or None
        """
        if track_id not in self.vehicle_tracks:
            return None
        
        track = self.vehicle_tracks[track_id]
        if len(track) < 3:
            return None
        
        # Extract positions and times
        positions = np.array([[p['x'], p['y']] for p in track])
        times = np.array([p['timestamp'] for p in track])
        
        # Fit polynomial trajectory (2nd order for acceleration)
        try:
            # Fit x and y separately
            coeffs_x = np.polyfit(times, positions[:, 0], deg=min(2, len(times)-1))
            coeffs_y = np.polyfit(times, positions[:, 1], deg=min(2, len(times)-1))
            
            # Predict future positions
            last_time = times[-1]
            avg_dt = np.mean(np.diff(times)) if len(times) > 1 else 0.033
            
            future_times = last_time + np.arange(1, predict_frames + 1) * avg_dt
            future_x = np.polyval(coeffs_x, future_times)
            future_y = np.polyval(coeffs_y, future_times)
            
            return list(zip(future_x, future_y))
        except:
            return None
    
    def estimate_velocity_2d(self, track_id: int) -> Optional[Tuple[float, float]]:
        """
        Estimate 2D velocity (vx, vy) in pixels/second.
        
        Returns:
            (vx, vy) velocity vector or None
        """
        if track_id not in self.vehicle_tracks:
            return None
        
        track = self.vehicle_tracks[track_id]
        if len(track) < 2:
            return None
        
        positions = np.array([[p['x'], p['y']] for p in track])
        times = np.array([p['timestamp'] for p in track])
        
        # Linear regression for velocity
        try:
            vx = np.polyfit(times, positions[:, 0], 1)[0]
            vy = np.polyfit(times, positions[:, 1], 1)[0]
            return (vx, vy)
        except:
            return None


class CollisionPathDetector:
    """Detect if vehicles are on collision path with camera vehicle."""
    
    def __init__(self, 
                 frame_width: int,
                 frame_height: int,
                 camera_path_width: float = 0.3,
                 prediction_time: float = 3.0):
        """
        Initialize collision path detector.
        
        Args:
            frame_width: Frame width in pixels
            frame_height: Frame height in pixels
            camera_path_width: Width of camera vehicle path (0.0-1.0, fraction of frame)
            prediction_time: Time horizon for prediction (seconds)
        """
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.camera_path_width = camera_path_width
        self.prediction_time = prediction_time
        
        # Define camera vehicle's path (center lane in rear view)
        self.path_center_x = frame_width / 2
        self.path_width_pixels = frame_width * camera_path_width
        self.path_x_min = self.path_center_x - self.path_width_pixels / 2
        self.path_x_max = self.path_center_x + self.path_width_pixels / 2
        
    def is_in_camera_path(self, x: float, y: Optional[float] = None) -> bool:
        """
        Check if position is in camera vehicle's path.
        
        Args:
            x: X coordinate (pixels)
            y: Y coordinate (optional, for future use)
            
        Returns:
            True if in camera path
        """
        return self.path_x_min <= x <= self.path_x_max
    
    def calculate_lateral_deviation(self, x: float) -> float:
        """
        Calculate lateral deviation from camera path center.
        
        Args:
            x: X coordinate (pixels)
            
        Returns:
            Deviation in pixels (0 = on path center)
        """
        return abs(x - self.path_center_x)
    
    def predict_collision_risk(self,
                               current_pos: Tuple[float, float],
                               velocity_2d: Tuple[float, float],
                               depth: float,
                               depth_velocity: float,
                               bbox_width: float = None,
                               vehicle_class: str = 'car') -> Dict:
        """
        Simplified collision risk based on physics only.
        
        Key factors:
        1. Relative velocity (depth_velocity) - are they closing the gap?
        2. Current distance (depth)
        3. Time to collision (TTC)
        4. Vehicle type (car, truck, etc.)
        
        No path/lane analysis - just pure collision physics.
        
        Args:
            current_pos: Current (x, y) position in pixels
            velocity_2d: (vx, vy) velocity in pixels/second
            depth: Current depth in meters
            depth_velocity: Depth change rate in m/s (negative = approaching)
            bbox_width: Width of bounding box
            vehicle_class: Type of vehicle (car, truck, bus, etc.)
            
        Returns:
            Dict with risk assessment
        """
        x, y = current_pos
        vx, vy = velocity_2d
        
        # SIMPLIFIED LOGIC: Only depth-based collision detection
        
        # 1. Is vehicle approaching? (negative depth_velocity = closing gap)
        approaching = depth_velocity < -0.2  # Threshold for active approach
        rapidly_approaching = depth_velocity < -0.8  # Very fast approach
        
        # 2. Is this just traffic? (both vehicles same speed)
        stationary_traffic = abs(depth_velocity) < 0.15
        
        # 3. Calculate Time to Collision (TTC)
        ttc = None
        if approaching and abs(depth_velocity) > 0.1:
            ttc = depth / abs(depth_velocity)
        
        # 4. Determine risk level
        collision_risk = self._assess_collision_risk_simple(
            depth=depth,
            depth_velocity=depth_velocity,
            approaching=approaching,
            rapidly_approaching=rapidly_approaching,
            stationary_traffic=stationary_traffic,
            ttc=ttc,
            vehicle_class=vehicle_class
        )
        
        return {
            'approaching': approaching,
            'stationary_traffic': stationary_traffic,
            'ttc': ttc,
            'depth': depth,
            'depth_velocity': depth_velocity,
            'collision_risk': collision_risk['level'],
            'risk_score': collision_risk['score'],
            'warning_message': collision_risk['message'],
            'color': collision_risk['color']
        }
    
    def _assess_collision_risk_simple(self,
                                      depth: float,
                                      depth_velocity: float,
                                      approaching: bool,
                                      rapidly_approaching: bool,
                                      stationary_traffic: bool,
                                      ttc: Optional[float],
                                      vehicle_class: str) -> Dict:
        """
        Simplified risk assessment based purely on collision physics.
        
        Algorithm:
        1. If stationary traffic (depth_velocity ≈ 0) → SAFE/LOW (no relative motion)
        2. If approaching + close distance → Calculate severity
        3. If TTC < threshold → CRITICAL/HIGH
        4. Otherwise → SAFE
        
        Args:
            depth: Distance in meters
            depth_velocity: Relative velocity (m/s)
            approaching: Is vehicle closing gap?
            rapidly_approaching: Is vehicle closing gap fast?
            stationary_traffic: Are both vehicles at same speed?
            ttc: Time to collision (seconds)
            vehicle_class: Type of vehicle
            
        Returns:
            Dict with level, score, message, color
        """
        
        # CASE 1: Traffic scenario - both vehicles same speed
        if stationary_traffic:
            if depth < 2.0:
                level = 'LOW'
                color = (255, 255, 0)  # Cyan
                message = f'Traffic: {depth:.1f}m (v≈0 m/s)'
                risk_score = 15
            else:
                level = 'SAFE'
                color = (0, 255, 0)  # Green
                message = f'{depth:.1f}m (v≈0 m/s)'
                risk_score = 0
            
            return {
                'level': level,
                'score': risk_score,
                'message': message,
                'color': color
            }
        
        # CASE 2: Vehicle is receding (moving away) - always safe
        if depth_velocity > 0.1:
            level = 'SAFE'
            color = (0, 255, 0)  # Green
            message = f'{depth:.1f}m (v=+{depth_velocity:.1f} m/s) ↗'
            risk_score = 0
            
            return {
                'level': level,
                'score': risk_score,
                'message': message,
                'color': color
            }
        
        # CASE 3: Vehicle is approaching - assess danger
        if approaching:
            risk_score = 0
            
            # Factor 1: How fast is it approaching?
            if rapidly_approaching:  # > 0.8 m/s
                risk_score += 50
            else:  # 0.2 - 0.8 m/s
                risk_score += 25
            
            # Factor 2: How close is it?
            if depth < 2.0:
                risk_score += 40  # Very close
            elif depth < 4.0:
                risk_score += 30  # Close
            elif depth < 6.0:
                risk_score += 20  # Moderate
            elif depth < 10.0:
                risk_score += 10  # Far but approaching
            
            # Factor 3: Time to collision
            if ttc is not None:
                if ttc < 1.5:
                    risk_score += 30  # Imminent
                elif ttc < 3.0:
                    risk_score += 20  # Soon
                elif ttc < 5.0:
                    risk_score += 10  # Moderate time
            
            # Factor 4: Vehicle type (larger = more dangerous)
            if vehicle_class in ['truck', 'bus']:
                risk_score += 10
            
            # Determine level
            if risk_score >= 80:
                level = 'CRITICAL'
                color = (0, 0, 255)  # Red
                if ttc is not None:
                    message = f'⚠️ COLLISION! TTC:{ttc:.1f}s | {depth:.1f}m | v={abs(depth_velocity):.1f}m/s'
                else:
                    message = f'⚠️ CRITICAL! {depth:.1f}m | v={abs(depth_velocity):.1f}m/s'
            elif risk_score >= 50:
                level = 'HIGH'
                color = (0, 140, 255)  # Orange
                if ttc is not None:
                    message = f'⚠️ WARNING! TTC:{ttc:.1f}s | {depth:.1f}m | v={abs(depth_velocity):.1f}m/s'
                else:
                    message = f'⚠️ HIGH RISK! {depth:.1f}m | v={abs(depth_velocity):.1f}m/s'
            elif risk_score >= 30:
                level = 'MEDIUM'
                color = (0, 255, 255)  # Yellow
                message = f'CAUTION: {depth:.1f}m | v={abs(depth_velocity):.1f}m/s'
            else:
                level = 'LOW'
                color = (255, 255, 0)  # Cyan
                message = f'{depth:.1f}m | v={abs(depth_velocity):.1f}m/s ↘'
            
            return {
                'level': level,
                'score': risk_score,
                'message': message,
                'color': color
            }
        
        # CASE 4: No significant motion
        level = 'SAFE'
        color = (0, 255, 0)  # Green
        message = f'{depth:.1f}m'
        risk_score = 0
        
        return {
            'level': level,
            'score': risk_score,
            'message': message,
            'color': color
        }
    
    def draw_camera_path(self, frame: np.ndarray, alpha: float = 0.3) -> np.ndarray:
        """
        Draw camera vehicle's path on frame.
        
        Args:
            frame: Input frame
            alpha: Transparency (0-1)
            
        Returns:
            Frame with path overlay
        """
        overlay = frame.copy()
        
        # Draw path boundaries
        x1 = int(self.path_x_min)
        x2 = int(self.path_x_max)
        
        # Semi-transparent green rectangle
        cv2.rectangle(overlay, 
                     (x1, 0), 
                     (x2, frame.shape[0]),
                     (0, 255, 0),
                     -1)
        
        # Blend
        frame = cv2.addWeighted(frame, 1-alpha, overlay, alpha, 0)
        
        # Draw center line
        cx = int(self.path_center_x)
        cv2.line(frame, (cx, 0), (cx, frame.shape[0]), (0, 255, 0), 2, cv2.LINE_AA)
        
        # Add label
        cv2.putText(frame, 'Camera Vehicle Path', 
                   (cx - 80, 30),
                   cv2.FONT_HERSHEY_SIMPLEX,
                   0.6, (0, 255, 0), 2, cv2.LINE_AA)
        
        return frame
    
    def draw_trajectory(self, 
                       frame: np.ndarray,
                       trajectory: List[Tuple[float, float]],
                       color: Tuple[int, int, int] = (255, 0, 255)) -> np.ndarray:
        """
        Draw predicted trajectory on frame.
        
        Args:
            frame: Input frame
            trajectory: List of (x, y) points
            color: Line color
            
        Returns:
            Frame with trajectory
        """
        if not trajectory or len(trajectory) < 2:
            return frame
        
        # Draw trajectory line
        points = np.array(trajectory, dtype=np.int32)
        
        for i in range(len(points) - 1):
            # Fade color along trajectory
            alpha = 1.0 - (i / len(points)) * 0.5
            pt_color = tuple(int(c * alpha) for c in color)
            
            cv2.line(frame, 
                    tuple(points[i]), 
                    tuple(points[i+1]),
                    pt_color, 2, cv2.LINE_AA)
        
        # Draw endpoint
        cv2.circle(frame, tuple(points[-1]), 5, color, -1)
        
        return frame


def integrate_trajectory_prediction(detection: Dict,
                                   trajectory_predictor: TrajectoryPredictor,
                                   collision_detector: CollisionPathDetector,
                                   timestamp: float) -> Dict:
    """
    Integrate trajectory prediction into detection.
    
    Args:
        detection: Vehicle detection dict
        trajectory_predictor: Trajectory predictor instance
        collision_detector: Collision detector instance
        timestamp: Current timestamp
        
    Returns:
        Enhanced detection with trajectory info
    """
    track_id = detection['track_id']
    bbox = detection['bbox']
    depth = detection.get('depth', 15.0)
    depth_velocity = detection.get('velocity', 0.0)
    
    # Update trajectory
    trajectory_predictor.update_track(track_id, bbox, timestamp)
    
    # Get current position
    center_x = (bbox[0] + bbox[2]) / 2
    center_y = (bbox[1] + bbox[3]) / 2
    
    # Estimate 2D velocity
    velocity_2d = trajectory_predictor.estimate_velocity_2d(track_id)
    
    if velocity_2d is not None:
        # Predict trajectory
        trajectory = trajectory_predictor.predict_trajectory(track_id, predict_frames=30)
        
        # Assess collision risk
        risk_info = collision_detector.predict_collision_risk(
            current_pos=(center_x, center_y),
            velocity_2d=velocity_2d,
            depth=depth,
            depth_velocity=depth_velocity
        )
        
        # Add to detection
        detection['velocity_2d'] = velocity_2d
        detection['trajectory'] = trajectory
        detection['collision_risk'] = risk_info
        
        # Override status based on path collision
        if risk_info['collision_risk'] in ['CRITICAL', 'HIGH']:
            detection['status'] = f"APPROACHING PATH [{risk_info['collision_risk']}]"
            detection['color'] = risk_info['warning_message']
    
    return detection


if __name__ == "__main__":
    # Example usage
    print("Trajectory Prediction and Collision Path Detection")
    print("=" * 60)
    print("\nFeatures:")
    print("  1. 2D trajectory prediction (x, y motion)")
    print("  2. Camera vehicle path detection")
    print("  3. Lateral deviation calculation")
    print("  4. Collision path assessment")
    print("  5. Risk scoring (0-100)")
    print("\nRisk Levels:")
    print("  CRITICAL (70+): Collision imminent")
    print("  HIGH (50-70): Vehicle entering path")
    print("  MEDIUM (30-50): Monitor vehicle")
    print("  LOW (15-30): Vehicle detected")
    print("  SAFE (<15): No threat")
