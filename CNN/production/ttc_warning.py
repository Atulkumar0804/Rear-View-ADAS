"""
Time-to-Collision (TTC) Warning System
Advanced warning logic for production ADAS
"""

import numpy as np
from enum import Enum
from typing import Dict, Optional, Tuple
import time


class RiskLevel(Enum):
    """Risk levels for collision warning"""
    SAFE = 0        # TTC > 5s - Green
    CAUTION = 1     # TTC 2-5s - Yellow
    CRITICAL = 2    # TTC 1-2s - Red
    IMMINENT = 3    # TTC < 1s - Emergency


class TTCCalculator:
    """
    Calculate Time-to-Collision and generate appropriate warnings
    """
    
    def __init__(self):
        self.prev_distances = {}  # track_id -> [distances, timestamps]
        self.ttc_history = {}     # track_id -> [ttc_values]
        self.history_size = 5     # Smooth over last 5 frames
        
    def calculate_ttc(self, 
                     distance: float, 
                     relative_velocity: float,
                     object_class: str) -> Tuple[float, RiskLevel]:
        """
        Calculate Time-to-Collision
        
        Args:
            distance: Current distance in meters
            relative_velocity: Relative velocity in m/s (negative = approaching)
            object_class: Type of object
            
        Returns:
            ttc: Time to collision in seconds
            risk_level: Risk level enum
        """
        # If object is moving away or stationary, no collision risk
        if relative_velocity >= -0.1:  # Small threshold for noise
            return float('inf'), RiskLevel.SAFE
        
        # TTC = distance / |relative_velocity|
        ttc = distance / abs(relative_velocity)
        
        # Determine risk level
        if ttc > 5.0:
            risk_level = RiskLevel.SAFE
        elif ttc > 2.0:
            risk_level = RiskLevel.CAUTION
        elif ttc > 1.0:
            risk_level = RiskLevel.CRITICAL
        else:
            risk_level = RiskLevel.IMMINENT
            
        return ttc, risk_level
    
    def update_tracking(self, track_id: int, distance: float, timestamp: float):
        """
        Update distance tracking for velocity calculation
        
        Args:
            track_id: Unique ID of tracked object
            distance: Current distance in meters
            timestamp: Current timestamp
        """
        if track_id not in self.prev_distances:
            self.prev_distances[track_id] = []
        
        self.prev_distances[track_id].append((distance, timestamp))
        
        # Keep only recent history
        if len(self.prev_distances[track_id]) > self.history_size:
            self.prev_distances[track_id].pop(0)
    
    def estimate_velocity(self, track_id: int) -> Optional[float]:
        """
        Estimate relative velocity from distance history
        
        Args:
            track_id: Unique ID of tracked object
            
        Returns:
            velocity: Relative velocity in m/s (negative = approaching)
        """
        if track_id not in self.prev_distances:
            return None
        
        history = self.prev_distances[track_id]
        
        if len(history) < 2:
            return None
        
        # Use linear regression for smoother velocity estimate
        distances = np.array([d for d, _ in history])
        timestamps = np.array([t for _, t in history])
        
        # Velocity = change in distance / change in time
        # Negative velocity means approaching
        velocity = np.polyfit(timestamps, distances, 1)[0]
        
        return velocity
    
    def smooth_ttc(self, track_id: int, ttc: float) -> float:
        """
        Smooth TTC values over time to reduce jitter
        
        Args:
            track_id: Unique ID of tracked object
            ttc: Current TTC value
            
        Returns:
            smoothed_ttc: Smoothed TTC value
        """
        if track_id not in self.ttc_history:
            self.ttc_history[track_id] = []
        
        self.ttc_history[track_id].append(ttc)
        
        # Keep only recent history
        if len(self.ttc_history[track_id]) > self.history_size:
            self.ttc_history[track_id].pop(0)
        
        # Use median for robustness against outliers
        return np.median(self.ttc_history[track_id])
    
    def get_warning_message(self, 
                           risk_level: RiskLevel, 
                           ttc: float,
                           object_class: str,
                           distance: float) -> Dict:
        """
        Generate warning message based on risk level
        
        Args:
            risk_level: Current risk level
            ttc: Time to collision
            object_class: Type of object
            distance: Current distance
            
        Returns:
            warning: Dictionary with warning information
        """
        warning = {
            'level': risk_level.name,
            'ttc': ttc,
            'object_class': object_class,
            'distance': distance,
            'message': '',
            'audio_alert': False,
            'haptic_alert': False,
            'brake_assist': False
        }
        
        if risk_level == RiskLevel.SAFE:
            warning['message'] = f'{object_class.upper()} DETECTED - {distance:.1f}m'
            
        elif risk_level == RiskLevel.CAUTION:
            warning['message'] = f'⚠️ CAUTION: {object_class.upper()} - {distance:.1f}m (TTC: {ttc:.1f}s)'
            warning['audio_alert'] = True  # Single beep
            
        elif risk_level == RiskLevel.CRITICAL:
            warning['message'] = f'🔴 WARNING: {object_class.upper()} - {distance:.1f}m (TTC: {ttc:.1f}s)'
            warning['audio_alert'] = True  # Rapid beeps
            warning['haptic_alert'] = True  # Vibrate
            
        elif risk_level == RiskLevel.IMMINENT:
            warning['message'] = f'🚨 EMERGENCY: {object_class.upper()} - {distance:.1f}m (TTC: {ttc:.1f}s)'
            warning['audio_alert'] = True  # Continuous alarm
            warning['haptic_alert'] = True  # Strong vibration
            warning['brake_assist'] = True  # Trigger brake assist
            
        return warning


class MultiObjectWarningSystem:
    """
    Manage warnings for multiple objects simultaneously
    """
    
    def __init__(self):
        self.ttc_calculator = TTCCalculator()
        self.active_warnings = {}
        
    def process_detections(self, detections: list) -> list:
        """
        Process multiple detections and generate warnings
        
        Args:
            detections: List of detection dictionaries
            
        Returns:
            warnings: List of warning dictionaries sorted by priority
        """
        warnings = []
        current_time = time.time()
        
        for det in detections:
            track_id = det.get('track_id')
            distance = det.get('distance')
            object_class = det.get('class')
            
            if track_id is None or distance is None:
                continue
            
            # Update tracking
            self.ttc_calculator.update_tracking(track_id, distance, current_time)
            
            # Estimate velocity
            velocity = self.ttc_calculator.estimate_velocity(track_id)
            
            if velocity is None:
                continue
            
            # Calculate TTC
            ttc, risk_level = self.ttc_calculator.calculate_ttc(
                distance, velocity, object_class
            )
            
            # Smooth TTC
            smooth_ttc = self.ttc_calculator.smooth_ttc(track_id, ttc)
            
            # Generate warning
            warning = self.ttc_calculator.get_warning_message(
                risk_level, smooth_ttc, object_class, distance
            )
            warning['track_id'] = track_id
            warning['velocity'] = velocity
            
            warnings.append(warning)
        
        # Sort by risk level and distance (prioritize closer objects)
        warnings.sort(key=lambda w: (
            -w['level'] if isinstance(w['level'], int) else -RiskLevel[w['level']].value,
            w['distance']
        ))
        
        return warnings
    
    def get_highest_risk(self, warnings: list) -> Optional[Dict]:
        """Get the warning with highest risk level"""
        if not warnings:
            return None
        return warnings[0]  # Already sorted by priority


if __name__ == "__main__":
    # Test the TTC calculator
    print("Testing TTC Warning System...\n")
    
    ttc_calc = TTCCalculator()
    
    # Simulate approaching vehicle
    test_scenarios = [
        (20.0, -5.0, "car"),      # 20m away, approaching at 5 m/s
        (15.0, -5.0, "car"),      # Getting closer
        (10.0, -5.0, "car"),      # Critical
        (3.0, -5.0, "car"),       # Imminent
    ]
    
    for distance, velocity, obj_class in test_scenarios:
        ttc, risk = ttc_calc.calculate_ttc(distance, velocity, obj_class)
        warning = ttc_calc.get_warning_message(risk, ttc, obj_class, distance)
        
        print(f"Distance: {distance}m, Velocity: {velocity}m/s")
        print(f"TTC: {ttc:.2f}s, Risk: {risk.name}")
        print(f"Message: {warning['message']}")
        print(f"Alerts: Audio={warning['audio_alert']}, "
              f"Haptic={warning['haptic_alert']}, "
              f"Brake={warning['brake_assist']}")
        print("-" * 60)
