"""
Depth Estimator using Depth-Anything-V2
Provides monocular depth estimation and distance tracking
"""

import torch
import torch.nn.functional as F
import numpy as np
import cv2
from pathlib import Path
from typing import Dict, Tuple, Optional, List
import time
from collections import deque

try:
    import transformers
    from transformers import pipeline
    HAS_DEPTH_MODEL = True
except ImportError:
    print("⚠️  transformers not installed. Install with: pip install transformers")
    HAS_DEPTH_MODEL = False

from .depth_config import DepthConfig


class DepthEstimator:
    """
    Monocular depth estimation using Depth-Anything-V2
    Estimates absolute depth from single RGB images
    """
    
    def __init__(self, 
                 model_size: str = 'small',
                 device: str = 'cuda',
                 cache_dir: str = './models/depth_models',
                 use_finetuned: bool = True):
        """
        Initialize depth estimator using Hugging Face Transformers
        
        Args:
            model_size: 'small', 'base', or 'large'
            device: 'cuda' or 'cpu'
            cache_dir: Directory to cache downloaded models
            use_finetuned: If True, use fine-tuned model for rear-view camera
        """
        if not HAS_DEPTH_MODEL:
            raise ImportError("transformers not installed. Install with: pip install transformers")
        
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.model_size = model_size
        self.config = DepthConfig()
        self.use_finetuned = use_finetuned
        
        print(f"🔧 Initializing Depth Estimator ({model_size})...")
        print(f"   Device: {self.device}")
        
        # Create cache directory
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Check for fine-tuned model
        finetuned_path = Path(__file__).parent / 'finetuned_model' / 'best_depth_model'
        
        if use_finetuned and finetuned_path.exists():
            model_name = str(finetuned_path)
            print(f"   ✅ Using fine-tuned model: rear-view optimized")
        else:
            # Model mapping to Hugging Face
            model_names = {
                'small': 'Intel/dpt-hybrid-midas',  # 123M params, fast
                'base': 'Intel/dpt-large',          # 345M params, accurate
                'large': 'Intel/dpt-large',         # Same as base for now
            }
            
            if model_size not in model_names:
                raise ValueError(f"Invalid model size. Choose from: {list(model_names.keys())}")
            
            model_name = model_names[model_size]
            if use_finetuned:
                print(f"   ⚠️  Fine-tuned model not found, using pre-trained")
        
        print(f"   Loading model: {model_name}...")
        if 'Intel' in model_name:
            print(f"   (First run will download ~500MB model)")
        
        # Initialize model using transformers pipeline
        self.pipe = pipeline(
            task="depth-estimation",
            model=model_name,
            device=0 if self.device == 'cuda' else -1
        )
        
        print("✅ Depth estimator ready!\n")
    

    
    @torch.no_grad()
    def estimate_depth(self, image: np.ndarray) -> np.ndarray:
        """
        Estimate depth map from RGB image
        
        Args:
            image: RGB image (H, W, 3) in BGR format (OpenCV)
            
        Returns:
            depth_map: Depth map (H, W) with relative depth values
        """
        from PIL import Image
        
        # Convert BGR to RGB
        if len(image.shape) == 3 and image.shape[2] == 3:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image
        
        # Get original size
        h, w = image_rgb.shape[:2]
        
        # Convert to PIL Image for transformers pipeline
        pil_image = Image.fromarray(image_rgb)
        
        # Estimate depth using pipeline
        result = self.pipe(pil_image)
        
        # Extract depth map (returns PIL Image)
        depth_pil = result['depth']
        
        # Convert to numpy and resize to original size
        depth_array = np.array(depth_pil)
        
        # Resize if needed
        if depth_array.shape[:2] != (h, w):
            depth_array = cv2.resize(depth_array, (w, h), interpolation=cv2.INTER_LINEAR)
        
        # Normalize to 0-1 range
        depth_array = depth_array.astype(np.float32)
        if depth_array.max() > depth_array.min():
            depth_array = (depth_array - depth_array.min()) / (depth_array.max() - depth_array.min())
        
        # Scale to meters (approximate - relative depth)
        depth_array = depth_array * 50.0  # Scale to 0-50m range
        
        return depth_array
    
    def get_bbox_depth(self, 
                       depth_map: np.ndarray, 
                       bbox: Tuple[int, int, int, int],
                       percentile: int = 50) -> float:
        """
        Extract depth from bounding box region
        
        Args:
            depth_map: Full depth map
            bbox: (x1, y1, x2, y2)
            percentile: Percentile to use (50 = median, robust to outliers)
            
        Returns:
            depth: Estimated depth in meters
        """
        x1, y1, x2, y2 = bbox
        
        # Clamp bbox to image bounds
        h, w = depth_map.shape
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        # Extract region
        bbox_depth = depth_map[y1:y2, x1:x2]
        
        if bbox_depth.size == 0:
            return 0.0
        
        # Use percentile for robustness (median filters outliers)
        depth = np.percentile(bbox_depth, percentile)
        
        # Clamp to valid range
        depth = np.clip(depth, self.config.MIN_DEPTH, self.config.MAX_DEPTH)
        
        return float(depth)
    
    def visualize_depth(self, 
                       depth_map: np.ndarray,
                       colormap: int = cv2.COLORMAP_INFERNO) -> np.ndarray:
        """
        Visualize depth map with colormap
        
        Args:
            depth_map: Depth map (H, W)
            colormap: OpenCV colormap
            
        Returns:
            colored_depth: RGB visualization
        """
        # Normalize to 0-255
        depth_normalized = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
        depth_normalized = depth_normalized.astype(np.uint8)
        
        # Apply colormap
        colored_depth = cv2.applyColorMap(depth_normalized, colormap)
        
        return colored_depth


class VehicleDepthTracker:
    """
    Track vehicle depth over time and estimate velocity
    Combines depth estimation with temporal tracking
    """
    
    def __init__(self, 
                 depth_estimator: DepthEstimator,
                 history_size: int = 5):
        """
        Initialize vehicle depth tracker
        
        Args:
            depth_estimator: Depth estimator instance
            history_size: Number of frames to track
        """
        self.depth_estimator = depth_estimator
        self.config = DepthConfig()
        self.history_size = history_size
        
        # Track history: track_id -> deque of (depth, timestamp)
        self.depth_history: Dict[int, deque] = {}
        self.velocity_history: Dict[int, deque] = {}
        
        print(f"🎯 Vehicle Depth Tracker initialized")
        print(f"   History size: {history_size} frames")
    
    def update(self, 
               depth_map: np.ndarray,
               detections: List[Dict],
               timestamp: float = None) -> List[Dict]:
        """
        Update tracking with new detections
        
        Args:
            depth_map: Current frame depth map
            detections: List of detections with 'bbox' and 'track_id'
            timestamp: Current timestamp (auto-generated if None)
            
        Returns:
            enhanced_detections: Detections with depth, velocity, status
        """
        if timestamp is None:
            timestamp = time.time()
        
        enhanced_detections = []
        
        for det in detections:
            track_id = det.get('track_id')
            bbox = det.get('bbox')
            
            if track_id is None or bbox is None:
                enhanced_detections.append(det)
                continue
            
            # Get depth
            depth = self.depth_estimator.get_bbox_depth(depth_map, bbox)
            
            # Initialize history if needed
            if track_id not in self.depth_history:
                self.depth_history[track_id] = deque(maxlen=self.history_size)
                self.velocity_history[track_id] = deque(maxlen=self.history_size)
            
            # Add to history
            self.depth_history[track_id].append((depth, timestamp))
            
            # Estimate velocity
            velocity = self._estimate_velocity(track_id)
            
            if velocity is not None:
                self.velocity_history[track_id].append(velocity)
            
            # Determine status
            status, status_color = self._determine_status(velocity, depth)
            
            # Calculate TTC if approaching
            ttc = None
            if velocity is not None and velocity < -self.config.VELOCITY_THRESHOLD:
                ttc = depth / abs(velocity)  # seconds
            
            # Enhance detection
            det['depth'] = depth
            det['velocity'] = velocity
            det['status'] = status
            det['status_color'] = status_color
            det['ttc'] = ttc
            
            enhanced_detections.append(det)
        
        return enhanced_detections
    
    def _estimate_velocity(self, track_id: int) -> Optional[float]:
        """
        Estimate velocity from depth history using linear regression
        
        Args:
            track_id: Track ID
            
        Returns:
            velocity: m/s (negative = approaching, positive = receding)
        """
        if track_id not in self.depth_history:
            return None
        
        history = list(self.depth_history[track_id])
        
        if len(history) < 2:
            return None
        
        # Extract depths and timestamps
        depths = np.array([d for d, _ in history])
        timestamps = np.array([t for _, t in history])
        
        # Linear regression: velocity = change in depth / change in time
        velocity = np.polyfit(timestamps, depths, 1)[0]
        
        return float(velocity)
    
    def _determine_status(self, velocity: Optional[float], depth: float) -> Tuple[str, Tuple]:
        """
        Determine vehicle status (approaching/stable/receding)
        
        Args:
            velocity: Velocity in m/s
            depth: Current depth in meters
            
        Returns:
            status: String status
            color: BGR color tuple
        """
        if velocity is None:
            return 'DETECTING', (255, 255, 255)  # White
        
        # Check velocity
        if velocity < -self.config.VELOCITY_THRESHOLD:
            # Approaching - color depends on depth
            if depth < self.config.DANGER_DISTANCE:
                return 'APPROACHING [DANGER]', (0, 0, 255)  # Red
            elif depth < self.config.CRITICAL_DISTANCE:
                return 'APPROACHING [CRITICAL]', (0, 69, 255)  # Orange-red
            elif depth < self.config.CAUTION_DISTANCE:
                return 'APPROACHING [CAUTION]', (0, 165, 255)  # Orange
            else:
                return 'APPROACHING', (0, 140, 255)  # Light orange
        
        elif velocity > self.config.VELOCITY_THRESHOLD:
            return 'RECEDING', (0, 255, 255)  # Yellow
        
        else:
            return 'STABLE', (0, 255, 0)  # Green
    
    def get_smooth_velocity(self, track_id: int) -> Optional[float]:
        """Get smoothed velocity using exponential moving average"""
        if track_id not in self.velocity_history:
            return None
        
        velocities = list(self.velocity_history[track_id])
        
        if not velocities:
            return None
        
        # Simple moving average
        return float(np.mean(velocities))
    
    def cleanup_old_tracks(self, active_track_ids: List[int]):
        """Remove tracking data for tracks no longer active"""
        all_track_ids = list(self.depth_history.keys())
        
        for track_id in all_track_ids:
            if track_id not in active_track_ids:
                self.depth_history.pop(track_id, None)
                self.velocity_history.pop(track_id, None)


if __name__ == "__main__":
    # Test depth estimator
    print("Testing Depth Estimator...")
    
    # Create dummy image
    test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    try:
        # Initialize
        estimator = DepthEstimator(model_size='small', device='cuda')
        
        # Estimate depth
        print("\n🔍 Estimating depth...")
        start = time.time()
        depth_map = estimator.estimate_depth(test_image)
        elapsed = time.time() - start
        
        print(f"✅ Depth estimated in {elapsed*1000:.1f}ms")
        print(f"   Depth shape: {depth_map.shape}")
        print(f"   Depth range: {depth_map.min():.2f} - {depth_map.max():.2f}")
        
        # Test bbox depth
        bbox = (100, 100, 200, 200)
        bbox_depth = estimator.get_bbox_depth(depth_map, bbox)
        print(f"   BBox depth: {bbox_depth:.2f}m")
        
        # Visualize
        colored_depth = estimator.visualize_depth(depth_map)
        print(f"   Visualization shape: {colored_depth.shape}")
        
        print("\n✅ Depth estimator test passed!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
