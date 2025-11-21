"""
file_utils.py
--------------
File and directory utilities for ADAS project.
"""

import os
import cv2
import json
import yaml
from datetime import datetime


def ensure_dir(path):
    """Create directory if not exists."""
    os.makedirs(path, exist_ok=True)


def save_frame(frame, out_dir="output_frames/", prefix="frame"):
    """Save a single frame to disk."""
    ensure_dir(out_dir)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    fname = f"{prefix}_{ts}.jpg"
    cv2.imwrite(os.path.join(out_dir, fname), frame)
    return fname


def load_yaml(path, default=None):
    """Load YAML safely."""
    if os.path.exists(path):
        with open(path, "r") as f:
            return yaml.safe_load(f)
    return default if default else {}


def save_json(path, data):
    """Save dictionary to JSON file."""
    ensure_dir(os.path.dirname(path))
    with open(path, "w") as f:
        json.dump(data, f, indent=4)


def load_json(path, default=None):
    """Load JSON with fallback."""
    if not os.path.exists(path):
        return default
    with open(path, "r") as f:
        return json.load(f)
