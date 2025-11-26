"""
Camera Inference with Depth Estimation
Real-time vehicle detection + depth + velocity estimation
Enhanced version of camera_inference.py with Depth-Anything-V2
"""

import torch
import torch.nn as nn
from torchvision import transforms
import cv2
import numpy as np
import argparse
from pathlib import Path
import time
import sys

sys.path.append('.')
from models.architectures import create_model
from ultralytics import YOLO
from depth_estimation import DepthEstimator, VehicleDepthTracker, DepthConfig

# Get absolute path to YOLO model
SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent.parent
YOLO_MODEL_PATH = str(PROJECT_ROOT / "CNN" / "models" / "yolo" / "yolov8n_RearView.pt")

# Configuration
IMG_SIZE = (224, 224)
CONFIDENCE_THRESHOLD = 0.4

# Vehicle classes
VEHICLE_CLASSES = ['car', 'truck', 'bus', 'person']

# YOLO class mapping
YOLO_CLASS_MAPPING = {
    0: 'person',
    2: 'car',
    5: 'bus',
    7: 'truck'
}

# Color mapping
CLASS_COLORS = {
    'car': (0, 255, 0),       # Green
    'truck': (255, 165, 0),   # Orange
    'bus': (0, 165, 255),     # Blue
    'person': (255, 0, 255),  # Magenta
}

# CNN Transform
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


class DepthEnhancedVehicleDetector:
    """Vehicle detector with depth estimation capabilities"""
    
    def __init__(self, cnn_model_path, depth_model_size='small', device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f"🔥 Device: {self.device}\n")
        
        # Load YOLO
        print("📦 Loading YOLO...")
        self.yolo_model = YOLO(YOLO_MODEL_PATH)
        print("✅ YOLO loaded\n")
        
        # Load CNN
        print(f"📦 Loading CNN: {cnn_model_path}")
        checkpoint = torch.load(cnn_model_path, map_location=self.device)
        
        model_name = Path(cnn_model_path).parent.name
        print(f"   Model: {model_name}")
        
        self.cnn_model = create_model(model_name, num_classes=len(VEHICLE_CLASSES), pretrained=False)
        self.cnn_model.load_state_dict(checkpoint['model_state_dict'])
        self.cnn_model = self.cnn_model.to(self.device)
        self.cnn_model.eval()
        
        print("✅ CNN loaded\n")
        
        # Initialize depth estimator
        print(f"📦 Loading Depth Estimator ({depth_model_size})...")
        self.depth_estimator = DepthEstimator(model_size=depth_model_size, device=device)
        
        # Initialize depth tracker
        self.depth_tracker = VehicleDepthTracker(self.depth_estimator)
        
        print(f"✅ Depth system ready!\n")
        
        # Tracking
        self.prev_detections = {}
        self.track_id_counter = 0
        
    def classify_crop(self, crop):
        """Classify vehicle crop with CNN"""
        if crop.size == 0 or crop.shape[0] < 10 or crop.shape[1] < 10:
            return None, 0.0
        
        try:
            crop_resized = cv2.resize(crop, IMG_SIZE)
            crop_rgb = cv2.cvtColor(crop_resized, cv2.COLOR_BGR2RGB)
            crop_tensor = transform(crop_rgb).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                outputs = self.cnn_model(crop_tensor)
                probs = torch.softmax(outputs, dim=1)
                conf, pred = torch.max(probs, 1)
            
            return VEHICLE_CLASSES[pred.item()], conf.item()
        except Exception as e:
            return None, 0.0
    
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
    
    def detect_and_track(self, frame):
        """Detect vehicles and track with depth"""
        detections = []
        
        # YOLO detection
        results = self.yolo_model(frame, conf=CONFIDENCE_THRESHOLD, verbose=False)
        
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
                
                yolo_conf = float(box.conf[0])
                yolo_class = YOLO_CLASS_MAPPING[cls_id]
                
                # Crop and classify with CNN
                crop = frame[y1:y2, x1:x2]
                cnn_class, cnn_conf = self.classify_crop(crop)
                
                # Fusion logic
                if yolo_class == 'person':
                    if cnn_class == 'person' and cnn_conf > 0.7:
                        final_class = 'person'
                        final_conf = (yolo_conf + cnn_conf) / 2
                    else:
                        final_class = 'person'
                        final_conf = yolo_conf
                else:
                    if cnn_class and cnn_conf > 0.6:
                        if cnn_class == 'person' and yolo_class in ['car', 'truck', 'bus']:
                            final_class = yolo_class
                            final_conf = yolo_conf
                        else:
                            final_class = cnn_class
                            final_conf = (yolo_conf + cnn_conf) / 2
                    else:
                        final_class = yolo_class
                        final_conf = yolo_conf
                
                detections.append({
                    'bbox': [x1, y1, x2, y2],
                    'class': final_class,
                    'confidence': final_conf,
                })
        
        # Track vehicles (assign IDs)
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
    
    def process_frame_with_depth(self, frame):
        """Process frame with detection + depth estimation"""
        
        # Detect and track vehicles
        detections = self.detect_and_track(frame)
        
        # Estimate depth map
        depth_map = self.depth_estimator.estimate_depth(frame)
        
        # Update depth tracking
        enhanced_detections = self.depth_tracker.update(depth_map, detections)
        
        return enhanced_detections, depth_map
    
    def draw_detections(self, frame, detections, fps=0, show_depth_map=False, depth_map=None):
        """Draw enhanced annotations with depth info"""
        annotated = frame.copy()
        
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            class_name = det['class']
            conf = det['confidence']
            
            # Depth info
            depth = det.get('depth')
            velocity = det.get('velocity')
            status = det.get('status', 'DETECTING')
            status_color = det.get('status_color', (255, 255, 255))
            ttc = det.get('ttc')
            
            # Draw bounding box
            color = CLASS_COLORS.get(class_name, (255, 255, 255))
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label = f"{class_name}: {conf:.2f}"
            (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(annotated, (x1, y1 - label_h - 10), (x1 + label_w, y1), color, -1)
            cv2.putText(annotated, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            
            # Draw depth info
            if depth is not None:
                depth_label = f"D: {depth:.1f}m"
                cv2.putText(annotated, depth_label, (x1, y2 + 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            # Draw velocity
            if velocity is not None:
                vel_label = f"V: {velocity:.2f}m/s"
                cv2.putText(annotated, vel_label, (x1, y2 + 40),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            # Draw status
            cv2.putText(annotated, f"[{status}]", (x1, y2 + 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 2)
            
            # Draw TTC if approaching
            if ttc is not None and ttc < 10:
                ttc_label = f"TTC: {ttc:.1f}s"
                cv2.putText(annotated, ttc_label, (x1, y2 + 80),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        # Draw FPS and info
        info_y = 30
        cv2.putText(annotated, f"FPS: {fps:.1f}", (10, info_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(annotated, f"Vehicles: {len(detections)}", (10, info_y + 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Show depth map overlay
        if show_depth_map and depth_map is not None:
            depth_colored = self.depth_estimator.visualize_depth(depth_map)
            depth_resized = cv2.resize(depth_colored, (320, 240))
            
            # Overlay in corner
            h, w = annotated.shape[:2]
            x_offset = w - 330
            y_offset = 10
            annotated[y_offset:y_offset+240, x_offset:x_offset+320] = depth_resized
            
            cv2.rectangle(annotated, (x_offset-2, y_offset-2), 
                         (x_offset+322, y_offset+242), (0, 255, 0), 2)
            cv2.putText(annotated, "Depth Map", (x_offset, y_offset-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        return annotated


def main():
    parser = argparse.ArgumentParser(description='Camera Vehicle Detection with Depth')
    parser.add_argument('--model', type=str, 
                       default='checkpoints/transfer_resnet18/best_model.pth',
                       help='Path to CNN model')
    parser.add_argument('--camera', type=str, default='0',
                       help='Camera ID or video file')
    parser.add_argument('--depth-model', type=str, default='small',
                       choices=['small', 'base', 'large'],
                       help='Depth model size')
    parser.add_argument('--width', type=int, default=1280)
    parser.add_argument('--height', type=int, default=720)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--save', type=str, help='Save output video')
    parser.add_argument('--show-depth', action='store_true',
                       help='Show depth map overlay')
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("🚗 DEPTH-ENHANCED VEHICLE DETECTION")
    print("="*60 + "\n")
    
    # Check CNN model
    if not Path(args.model).exists():
        print(f"❌ Model not found: {args.model}")
        return
    
    # Initialize detector
    detector = DepthEnhancedVehicleDetector(
        args.model, 
        depth_model_size=args.depth_model,
        device=args.device
    )
    
    # Open camera
    try:
        camera_id = int(args.camera)
        is_file = False
    except ValueError:
        camera_id = args.camera
        is_file = True
    
    print(f"📹 Opening {'video' if is_file else 'camera'}: {camera_id}...")
    cap = cv2.VideoCapture(camera_id)
    
    if not cap.isOpened():
        print(f"❌ Failed to open")
        return
    
    if not is_file:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"✅ Opened: {actual_width}x{actual_height}\n")
    
    # Video writer
    writer = None
    if args.save:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(args.save, fourcc, 20, (actual_width, actual_height))
        print(f"💾 Saving to: {args.save}\n")
    
    print("🚀 Starting detection with depth estimation...")
    print("   Press 'q' to quit")
    print("   Press 's' to screenshot")
    print("   Press 'd' to toggle depth overlay\n")
    
    frame_count = 0
    start_time = time.time()
    screenshot_count = 0
    show_depth = args.show_depth
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                if is_file:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                break
            
            frame_count += 1
            
            # Process with depth
            detections, depth_map = detector.process_frame_with_depth(frame)
            
            # Calculate FPS
            elapsed = time.time() - start_time
            fps = frame_count / elapsed if elapsed > 0 else 0
            
            # Draw
            annotated = detector.draw_detections(
                frame, detections, fps=fps,
                show_depth_map=show_depth, depth_map=depth_map
            )
            
            # Display
            cv2.imshow('Depth-Enhanced Detection - Press q to quit', annotated)
            
            if writer:
                writer.write(annotated)
            
            # Keys
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                path = f"screenshot_depth_{screenshot_count:03d}.jpg"
                cv2.imwrite(path, annotated)
                print(f"📸 Screenshot: {path}")
                screenshot_count += 1
            elif key == ord('d'):
                show_depth = not show_depth
                print(f"Depth overlay: {'ON' if show_depth else 'OFF'}")
    
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted")
    
    finally:
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()
        
        elapsed = time.time() - start_time
        avg_fps = frame_count / elapsed if elapsed > 0 else 0
        
        print("\n" + "="*60)
        print("📊 STATISTICS")
        print("="*60)
        print(f"Frames: {frame_count}")
        print(f"Time: {elapsed:.2f}s")
        print(f"Avg FPS: {avg_fps:.2f}")
        print("="*60 + "\n")


if __name__ == "__main__":
    main()
