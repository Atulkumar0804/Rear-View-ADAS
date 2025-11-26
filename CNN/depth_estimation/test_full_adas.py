"""
Complete Depth Estimation Test with YOLO Vehicle Detection + Trajectory Prediction
Detects all vehicles + depth + velocity + path prediction + collision risk
"""

import sys
from pathlib import Path
import cv2
import numpy as np
import time
import torch
from torchvision import transforms

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from depth_estimation import DepthEstimator, VehicleDepthTracker, DepthConfig
from trajectory_predictor import TrajectoryPredictor, CollisionPathDetector

# YOLO
from ultralytics import YOLO

# YOLO class mapping
YOLO_CLASS_MAPPING = {
    0: 'person',
    2: 'car',
    5: 'bus',
    7: 'truck'
}

# Color mapping
CLASS_COLORS = {
    'car': (0, 255, 0),
    'truck': (255, 165, 0),
    'bus': (0, 165, 255),
    'person': (255, 0, 255),
}


class VehicleDetectorWithDepth:
    """Complete vehicle detection + depth estimation system"""
    
    def __init__(self, yolo_model_path, depth_model_size='small', use_finetuned=True):
        """
        Initialize vehicle detector with depth estimation.
        
        Args:
            yolo_model_path: Path to YOLO model
            depth_model_size: Size of depth model ('small', 'base', 'large')
            use_finetuned: If True, use fine-tuned depth model
        """
        print("🔧 Initializing Detection System...")
        
        # Load YOLO
        print("   Loading YOLO...")
        self.yolo = YOLO(yolo_model_path)
        print("   ✅ YOLO loaded")
        
        # Load depth estimator
        print("   Loading Depth Estimator...")
        self.depth_estimator = DepthEstimator(model_size=depth_model_size, use_finetuned=use_finetuned)
        print("   ✅ Depth loaded")
        
        # Initialize tracker
        self.depth_tracker = VehicleDepthTracker(self.depth_estimator)
        
        # Initialize trajectory predictor and collision detector
        self.trajectory_predictor = None  # Will be initialized with frame size
        self.collision_detector = None
        
        # Tracking
        self.prev_detections = {}
        self.track_id_counter = 0
        
        print("✅ System ready!\n")
    
    def initialize_trajectory_system(self, frame_width: int, frame_height: int):
        """Initialize trajectory prediction system with frame dimensions."""
        if self.trajectory_predictor is None:
            self.trajectory_predictor = TrajectoryPredictor(history_size=10)
            self.collision_detector = CollisionPathDetector(
                frame_width=frame_width,
                frame_height=frame_height,
                camera_path_width=0.35,  # 35% of frame width
                prediction_time=3.0  # 3 seconds ahead
            )
            print("🎯 Trajectory prediction system initialized")
    
    def compute_iou(self, box1, box2):
        """Compute IoU"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        if x2 < x1 or y2 < y1:
            return 0.0
        
        inter = (x2 - x1) * (y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - inter
        
        return inter / union if union > 0 else 0.0
    
    def detect_vehicles(self, frame, confidence_threshold=0.4):
        """Detect vehicles using YOLO"""
        detections = []
        
        # YOLO detection
        results = self.yolo(frame, conf=confidence_threshold, verbose=False)
        
        for result in results:
            if result.boxes is None:
                continue
            
            for box in result.boxes:
                cls_id = int(box.cls[0])
                
                if cls_id not in YOLO_CLASS_MAPPING:
                    continue
                
                # Get bbox
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                
                # Filter small boxes
                if (x2 - x1) * (y2 - y1) < 1500:
                    continue
                
                conf = float(box.conf[0])
                vehicle_class = YOLO_CLASS_MAPPING[cls_id]
                
                detections.append({
                    'bbox': [x1, y1, x2, y2],
                    'class': vehicle_class,
                    'confidence': conf,
                })
        
        return detections
    
    def assign_track_ids(self, detections):
        """Assign persistent track IDs"""
        tracked = []
        
        for det in detections:
            best_iou = 0
            best_track_id = None
            
            # Match with previous frame
            for track_id, prev_det in self.prev_detections.items():
                if prev_det['class'] != det['class']:
                    continue
                
                iou = self.compute_iou(det['bbox'], prev_det['bbox'])
                if iou > 0.3 and iou > best_iou:
                    best_iou = iou
                    best_track_id = track_id
            
            if best_track_id is not None:
                det['track_id'] = best_track_id
            else:
                det['track_id'] = self.track_id_counter
                self.track_id_counter += 1
            
            tracked.append(det)
        
        # Update tracking
        self.prev_detections = {det['track_id']: det for det in tracked}
        
        return tracked
    
    def process_frame(self, frame, timestamp):
        """Process frame: detect + depth + tracking + trajectory prediction"""
        
        # Initialize trajectory system on first frame
        if self.trajectory_predictor is None:
            self.initialize_trajectory_system(frame.shape[1], frame.shape[0])
        
        # Detect vehicles
        detections = self.detect_vehicles(frame)
        
        # Assign track IDs
        detections = self.assign_track_ids(detections)
        
        # Estimate depth
        depth_map = self.depth_estimator.estimate_depth(frame)
        
        # Update depth tracking
        enhanced_detections = self.depth_tracker.update(depth_map, detections, timestamp)
        
        # Add trajectory prediction and collision risk
        for det in enhanced_detections:
            track_id = det['track_id']
            bbox = det['bbox']
            depth = det.get('depth', 15.0)
            depth_velocity = det.get('velocity', 0.0)
            vehicle_class = det.get('class', 'car')  # Get vehicle type
            
            # Update trajectory
            self.trajectory_predictor.update_track(track_id, bbox, timestamp)
            
            # Get current position
            center_x = (bbox[0] + bbox[2]) / 2
            center_y = (bbox[1] + bbox[3]) / 2
            
            # Get bbox dimensions
            bbox_width = bbox[2] - bbox[0]
            
            # Simple velocity estimation (just for collision detector compatibility)
            velocity_2d = (0.0, 0.0)  # Not used in simplified detection
            
            # Handle None velocity
            if depth_velocity is None:
                depth_velocity = 0.0
            
            # Assess collision risk (simplified - only depth, velocity, TTC)
            risk_info = self.collision_detector.predict_collision_risk(
                current_pos=(center_x, center_y),
                velocity_2d=velocity_2d,
                depth=depth,
                depth_velocity=depth_velocity,
                bbox_width=bbox_width,
                vehicle_class=vehicle_class
            )
            
            # Add to detection
            det['collision_risk'] = risk_info
            
            # Override status based on collision risk
            if risk_info['collision_risk'] in ['CRITICAL', 'HIGH']:
                det['status'] = f"{risk_info['warning_message']}"
                det['risk_color'] = risk_info.get('color', (0, 0, 255))
        
        return enhanced_detections, depth_map
        
        # Update depth tracking
        enhanced_detections = self.depth_tracker.update(depth_map, detections, timestamp)
        
        return enhanced_detections, depth_map


def test_video_with_full_detection(video_path: str, 
                                   yolo_model_path: str,
                                   depth_model_size: str = 'small',
                                   use_finetuned: bool = True,
                                   save_output: bool = True):
    """Test complete system on video"""
    
    print("\n" + "="*70)
    print("🚗 COMPLETE ADAS SYSTEM TEST")
    print("   Vehicle Detection + Depth + Velocity + TTC Warnings")
    print("="*70)
    
    # Check video exists
    if not Path(video_path).exists():
        print(f"❌ Video not found: {video_path}")
        return
    
    print(f"📹 Video: {video_path}")
    
    # Initialize system
    detector = VehicleDetectorWithDepth(yolo_model_path, depth_model_size, use_finetuned=use_finetuned)
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Failed to open video")
        return
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"\n📊 Video Info:")
    print(f"   Resolution: {width}x{height}")
    print(f"   FPS: {fps:.2f}")
    print(f"   Total frames: {total_frames}")
    print(f"   Duration: {total_frames/fps:.1f}s")
    
    # Setup output video
    writer = None
    if save_output:
        output_path = Path(video_path).stem + "_full_adas.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        print(f"💾 Output: {output_path}\n")
    
    print("🚀 Processing video...")
    print("   Press 'q' to quit early\n")
    
    frame_count = 0
    start_time = time.time()
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("\n🎬 End of video")
                break
            
            frame_count += 1
            timestamp = frame_count / fps
            
            # Process frame
            detections, depth_map = detector.process_frame(frame, timestamp)
            
            # Draw results with trajectory
            annotated = draw_full_annotations(frame, depth_map, detections, frame_count, fps, detector)
            
            # Save
            if writer:
                writer.write(annotated)
            
            # Display
            cv2.imshow('Full ADAS System - Press q to quit', annotated)
            
            # Progress
            if frame_count % 30 == 0 or frame_count == total_frames:
                progress = (frame_count / total_frames) * 100
                elapsed = time.time() - start_time
                current_fps = frame_count / elapsed if elapsed > 0 else 0
                print(f"   Progress: {progress:.1f}% ({frame_count}/{total_frames}) - "
                      f"Vehicles: {len(detections)} - FPS: {current_fps:.1f}")
            
            # Check for quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\n⚠️  Stopped by user")
                break
    
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user")
    
    finally:
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()
        
        # Final stats
        elapsed = time.time() - start_time
        avg_fps = frame_count / elapsed if elapsed > 0 else 0
        
        print("\n" + "="*70)
        print("📊 PROCESSING COMPLETE")
        print("="*70)
        print(f"Frames processed: {frame_count}/{total_frames}")
        print(f"Time elapsed: {elapsed:.2f}s")
        print(f"Average FPS: {avg_fps:.2f}")
        if save_output:
            print(f"Output saved: {output_path}")
        print("="*70 + "\n")


def draw_full_annotations(frame, depth_map, detections, frame_num, fps, detector=None):
    """Draw comprehensive annotations with collision risk (no path/trajectory visualization)"""
    
    annotated = frame.copy()
    h, w = frame.shape[:2]
    
    # Create depth visualization (small overlay)
    depth_colored = cv2.applyColorMap(
        cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8),
        cv2.COLORMAP_INFERNO
    )
    depth_small = cv2.resize(depth_colored, (320, 240))
    
    # Overlay depth map in corner
    x_offset = w - 330
    y_offset = 10
    annotated[y_offset:y_offset+240, x_offset:x_offset+320] = depth_small
    cv2.rectangle(annotated, (x_offset-2, y_offset-2), 
                 (x_offset+322, y_offset+242), (0, 255, 0), 2)
    cv2.putText(annotated, "Depth Map", (x_offset, y_offset-10),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Draw each detection
    for det in detections:
        x1, y1, x2, y2 = det['bbox']
        vehicle_class = det['class']
        conf = det['confidence']
        
        # Get depth info
        depth = det.get('depth', 0)
        velocity = det.get('velocity')
        status = det.get('status', 'DETECTING')
        status_color = det.get('status_color', (255, 255, 255))
        ttc = det.get('ttc')
        
        # Get collision risk info
        collision_risk = det.get('collision_risk', {})
        risk_level = collision_risk.get('collision_risk', 'SAFE')
        risk_color = collision_risk.get('color', (0, 255, 0)) if collision_risk else (255, 255, 255)
        
        # Draw bounding box with risk color
        box_color = risk_color if risk_level in ['CRITICAL', 'HIGH'] else CLASS_COLORS.get(vehicle_class, (255, 255, 255))
        thickness = 4 if risk_level in ['CRITICAL', 'HIGH'] else 3
        cv2.rectangle(annotated, (x1, y1), (x2, y2), box_color, thickness)
        
        # Title box
        title = f"{vehicle_class.upper()}: {conf:.2f}"
        (title_w, title_h), _ = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(annotated, (x1, y1 - title_h - 12), (x1 + title_w + 8, y1), box_color, -1)
        cv2.putText(annotated, title, (x1 + 4, y1 - 6),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        # Info panel (below bbox)
        info_y = y2 + 22
        line_height = 22
        
        # Distance
        if depth > 0:
            dist_text = f"D: {depth:.1f}m"
            cv2.putText(annotated, dist_text, (x1, info_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
            info_y += line_height
        
        # Velocity
        if velocity is not None:
            vel_text = f"V: {velocity:.2f}m/s"
            vel_color = (0, 0, 255) if velocity < 0 else (0, 255, 255)
            cv2.putText(annotated, vel_text, (x1, info_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.55, vel_color, 2)
            info_y += line_height
        
        # Collision risk status
        if collision_risk:
            risk_text = f"{risk_level}: {collision_risk.get('risk_score', 0):.0f}%"
            cv2.putText(annotated, risk_text, (x1, info_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.55, risk_color, 2)
            info_y += line_height
            
            # Path info
            if collision_risk.get('in_path_now'):
                cv2.putText(annotated, "IN CAMERA PATH!", (x1, info_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                info_y += line_height
            elif collision_risk.get('will_enter_path'):
                cv2.putText(annotated, "ENTERING PATH", (x1, info_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)
                info_y += line_height
        
        # Status (fallback)
        else:
            status_short = status.replace('APPROACHING ', 'APPR ').replace('RECEDING', 'RECED')
            cv2.putText(annotated, f"[{status_short}]", (x1, info_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 2)
            info_y += line_height
        
        # TTC warning
        if ttc is not None and ttc < 10:
            ttc_text = f"TTC: {ttc:.1f}s"
            ttc_color = (0, 0, 255) if ttc < 2 else (0, 165, 255)
            cv2.putText(annotated, ttc_text, (x1, info_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, ttc_color, 2)
            
            # Big warning if critical
            if ttc < 2 or risk_level == 'CRITICAL':
                warn_text = "⚠️ COLLISION WARNING ⚠️"
                (warn_w, warn_h), _ = cv2.getTextSize(warn_text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 3)
                warn_x = (w - warn_w) // 2
                warn_y = 50
                cv2.rectangle(annotated, (warn_x - 10, warn_y - warn_h - 10),
                             (warn_x + warn_w + 10, warn_y + 10), (0, 0, 255), -1)
                cv2.putText(annotated, warn_text, (warn_x, warn_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)
    
    # Frame info (top-left)
    cv2.putText(annotated, f"Frame: {frame_num}", (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(annotated, f"Vehicles: {len(detections)}", (10, 60),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(annotated, f"FPS: {fps:.1f}", (10, 90),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    return annotated


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Complete ADAS Test')
    parser.add_argument('--video', type=str, required=True,
                       help='Path to video file')
    parser.add_argument('--yolo', type=str, 
                       default='../models/yolo/yolov8n_RearView.pt',
                       help='Path to YOLO model')
    parser.add_argument('--model', type=str, default='small',
                       choices=['small', 'base', 'large'],
                       help='Depth model size')
    parser.add_argument('--no-save', action='store_true',
                       help='Do not save output video')
    
    args = parser.parse_args()
    
    # Print config
    DepthConfig.print_config()
    
    # Run test
    test_video_with_full_detection(
        args.video, 
        args.yolo,
        depth_model_size=args.model, 
        save_output=not args.no_save
    )
