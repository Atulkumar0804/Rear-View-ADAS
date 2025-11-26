"""
Configuration for Depth Estimation
"""

class DepthConfig:
    """Configuration for depth estimation"""
    
    # Model selection
    MODEL_SIZE = 'small'  # Options: 'small', 'base', 'large'
    
    # Model paths (will be downloaded automatically on first use)
    MODEL_CONFIGS = {
        'small': {
            'encoder': 'vits',
            'features': 64,
            'out_channels': [48, 96, 192, 384],
            'params': '24.8M',
            'speed': 'Fast (30+ FPS on RTX)',
        },
        'base': {
            'encoder': 'vitb',
            'features': 128,
            'out_channels': [96, 192, 384, 768],
            'params': '97.5M',
            'speed': 'Medium (20-25 FPS)',
        },
        'large': {
            'encoder': 'vitl',
            'features': 256,
            'out_channels': [256, 512, 1024, 1024],
            'params': '335.3M',
            'speed': 'Slow (10-15 FPS)',
        }
    }
    
    # Processing settings
    INPUT_SIZE = (518, 518)  # Depth-Anything-V2 optimal size
    DEPTH_SCALE = 1000.0     # Scale factor for depth values
    
    # Distance estimation settings
    MIN_DEPTH = 0.5          # meters - minimum valid depth
    MAX_DEPTH = 50.0         # meters - maximum valid depth
    DEPTH_PERCENTILE = 50    # Use median (50th percentile) of bbox region
    
    # Velocity estimation settings
    HISTORY_SIZE = 5         # Number of frames to track
    VELOCITY_THRESHOLD = 0.1 # m/s - threshold for stationary detection
    SMOOTHING_ALPHA = 0.7    # Exponential smoothing factor
    
    # Warning thresholds (meters)
    SAFE_DISTANCE = 10.0     # > 10m is safe
    CAUTION_DISTANCE = 5.0   # 5-10m is caution
    CRITICAL_DISTANCE = 2.0  # 2-5m is critical
    DANGER_DISTANCE = 1.0    # < 2m is danger
    
    # Vehicle height priors (for scale calibration)
    VEHICLE_HEIGHTS = {
        'car': 1.5,           # meters
        'truck': 3.0,
        'bus': 3.5,
        'person': 1.7,
    }
    
    @classmethod
    def get_model_info(cls, size='small'):
        """Get model configuration info"""
        if size not in cls.MODEL_CONFIGS:
            raise ValueError(f"Invalid model size: {size}. Choose from {list(cls.MODEL_CONFIGS.keys())}")
        return cls.MODEL_CONFIGS[size]
    
    @classmethod
    def print_config(cls):
        """Print current configuration"""
        print("\n" + "="*60)
        print("DEPTH ESTIMATION CONFIGURATION")
        print("="*60)
        print(f"Model Size: {cls.MODEL_SIZE}")
        print(f"Model Info: {cls.get_model_info(cls.MODEL_SIZE)}")
        print(f"Input Size: {cls.INPUT_SIZE}")
        print(f"History Size: {cls.HISTORY_SIZE} frames")
        print(f"Velocity Threshold: {cls.VELOCITY_THRESHOLD} m/s")
        print(f"\nDistance Thresholds:")
        print(f"  Safe: > {cls.SAFE_DISTANCE}m")
        print(f"  Caution: {cls.CAUTION_DISTANCE}-{cls.SAFE_DISTANCE}m")
        print(f"  Critical: {cls.CRITICAL_DISTANCE}-{cls.CAUTION_DISTANCE}m")
        print(f"  Danger: < {cls.CRITICAL_DISTANCE}m")
        print("="*60 + "\n")
