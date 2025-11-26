"""
Vehicle Classes Configuration for Indian Roads
Extended classes for production ADAS system
"""

# Original 4 classes
BASIC_CLASSES = ['car', 'truck', 'bus', 'person']

# Extended classes for Indian market
INDIAN_VEHICLE_CLASSES = [
    'car',
    'truck', 
    'bus',
    'person',
    'auto_rickshaw',  # Auto-rickshaw (very common in India)
    'motorcycle',     # Motorcycle/Scooter (major collision risk)
    'bicycle',        # Bicycle
    'animal'          # Cows, dogs on roads
]

# Real-world heights for distance estimation (in meters)
REAL_HEIGHTS = {
    'car': 1.5,
    'truck': 3.0,
    'bus': 3.2,
    'person': 1.7,
    'auto_rickshaw': 1.8,
    'motorcycle': 1.4,
    'bicycle': 1.6,
    'animal': 1.0  # Average for cow/dog
}

# Color mapping for visualization
CLASS_COLORS = {
    'car': (0, 255, 0),           # Green
    'truck': (255, 165, 0),       # Orange
    'bus': (0, 165, 255),         # Blue
    'person': (255, 0, 255),      # Magenta
    'auto_rickshaw': (255, 255, 0),  # Yellow
    'motorcycle': (255, 0, 128),  # Pink
    'bicycle': (0, 255, 255),     # Cyan
    'animal': (128, 128, 128)     # Gray
}

# YOLO class mapping (standard COCO + custom)
YOLO_CLASS_MAPPING = {
    0: 'person',
    1: 'bicycle',
    2: 'car',
    3: 'motorcycle',
    5: 'bus',
    7: 'truck',
    # Custom classes (if you retrain YOLO)
    80: 'auto_rickshaw',
    81: 'animal'
}

# Risk priority (higher = more dangerous)
RISK_PRIORITY = {
    'person': 10,        # Highest priority
    'bicycle': 9,
    'motorcycle': 8,
    'auto_rickshaw': 7,
    'animal': 6,
    'car': 5,
    'bus': 4,
    'truck': 3
}
