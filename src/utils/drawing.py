"""
drawing.py
-----------
Drawing utilities for visualization in the rear-view ADAS pipeline.

All functions modify the frame in-place.
"""

import cv2
import numpy as np


# Color presets
GREEN = (0, 255, 0)
ORANGE = (0, 165, 255)
RED = (0, 0, 255)
WHITE = (255, 255, 255)
YELLOW = (0, 255, 255)
BLUE = (255, 0, 0)


def draw_bbox(frame, bbox, color=GREEN, thickness=2):
    """Draw bounding box (x1,y1,x2,y2)."""
    x1, y1, x2, y2 = bbox
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)


def draw_label(frame, x, y, text, color=WHITE, scale=0.45):
    """Draw text label above object."""
    cv2.putText(frame, text, (x, y),
                cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def draw_warning_status(frame, bbox, warning_level):
    """
    Color bounding box based on warning level.
    """
    color = GREEN
    if warning_level == "WARN":
        color = ORANGE
    elif warning_level == "CRITICAL":
        color = RED
    draw_bbox(frame, bbox, color=color, thickness=2)


def draw_fps(frame, fps: float):
    """Display FPS on top-left."""
    cv2.putText(frame, f"FPS: {fps:.1f}",
                (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, YELLOW, 2)


def draw_trajectory(frame, trajectory, color=BLUE):
    """
    Draw predicted trajectory using image-space projection points.
    trajectory: list of (u, v)
    """
    for (u, v) in trajectory:
        cv2.circle(frame, (int(u), int(v)), 2, color, -1)


def draw_text_bottom(frame, x, y, text, color=WHITE):
    """
    Draw additional info below a bounding box.
    """
    cv2.putText(frame, text, (x, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
