"""
Real-time Camera Inference for Vehicle Detection
Simplified version for easy camera-based predictions

Usage:
    # Using best model (transfer_resnet18)
    python camera_inference.py
    
    # Using specific model
    python camera_inference.py --model checkpoints/mobilenet_inspired/best_model.pth
    
    # Using external camera (camera ID 1)
    python camera_inference.py --camera 1
    
    # Lower resolution for faster FPS
    python camera_inference.py --width 640 --height 480
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
import os

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


class CameraVehicleDetector:
    """Real-time camera-based vehicle detector"""
    
    def __init__(self, cnn_model_path, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f"🔥 Device: {self.device}")
        
        # Load YOLO
        print("📦 Loading YOLO...")
        self.yolo_model = YOLO(YOLO_MODEL_PATH)
        print("✅ YOLO loaded")
        
        # Load CNN
        print(f"📦 Loading CNN: {cnn_model_path}")
        checkpoint = torch.load(cnn_model_path, map_location=self.device)
        
        model_name = Path(cnn_model_path).parent.name
        print(f"   Model: {model_name}")
        
        self.cnn_model = create_model(model_name, num_classes=len(VEHICLE_CLASSES), pretrained=False)
        self.cnn_model.load_state_dict(checkpoint['model_state_dict'])
        self.cnn_model = self.cnn_model.to(self.device)
        self.cnn_model.eval()
        
        print("✅ CNN loaded")
        print(f"   Classes: {VEHICLE_CLASSES}\n")
        
        # Tracking
        self.prev_detections = {}
        self.track_id_counter = 0
        
    def classify_crop(self, crop):
        """Classify vehicle crop with CNN"""
        if crop.size == 0 or crop.shape[0] < 10 or crop.shape[1] < 10:
            return None, 0.0
        
        try:
            # Resize and transform
            crop_resized = cv2.resize(crop, IMG_SIZE)
            crop_rgb = cv2.cvtColor(crop_resized, cv2.COLOR_BGR2RGB)
            crop_tensor = transform(crop_rgb).unsqueeze(0).to(self.device)
            
            # Predict
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
        """Detect vehicles and track them"""
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
                
                # Smart fusion: Trust YOLO for person detection, use CNN for vehicle refinement
                if yolo_class == 'person':
                    # YOLO is very good at detecting persons, trust it
                    # Only override if CNN also says person with high confidence
                    if cnn_class == 'person' and cnn_conf > 0.7:
                        final_class = 'person'
                        final_conf = (yolo_conf + cnn_conf) / 2
                    else:
                        # Trust YOLO for person detection
                        final_class = 'person'
                        final_conf = yolo_conf
                else:
                    # For vehicles, use CNN to refine classification
                    if cnn_class and cnn_conf > 0.6:
                        # CNN is confident, but check consistency
                        if cnn_class == 'person' and yolo_class in ['car', 'truck', 'bus']:
                            # CNN says person but YOLO says vehicle - trust YOLO
                            final_class = yolo_class
                            final_conf = yolo_conf
                        else:
                            # Use CNN classification
                            final_class = cnn_class
                            final_conf = (yolo_conf + cnn_conf) / 2
                    else:
                        # CNN not confident, use YOLO
                        final_class = yolo_class
                        final_conf = yolo_conf
                
                detections.append({
                    'bbox': [x1, y1, x2, y2],
                    'class': final_class,
                    'confidence': final_conf,
                    'yolo_class': yolo_class,
                    'cnn_class': cnn_class,
                    'cnn_conf': cnn_conf,
                })
        
        # Track vehicles
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
                # Matched - compute distance change
                prev_area = (self.prev_detections[best_track_id]['bbox'][2] - 
                           self.prev_detections[best_track_id]['bbox'][0]) * \
                          (self.prev_detections[best_track_id]['bbox'][3] - 
                           self.prev_detections[best_track_id]['bbox'][1])
                
                curr_area = (det['bbox'][2] - det['bbox'][0]) * \
                           (det['bbox'][3] - det['bbox'][1])
                
                area_change = (curr_area - prev_area) / prev_area if prev_area > 0 else 0
                
                if area_change > 0.15:
                    status = 'APPROACHING'
                    color = (0, 0, 255)  # Red
                elif area_change < -0.15:
                    status = 'RECEDING'
                    color = (0, 255, 255)  # Yellow
                else:
                    status = 'STABLE'
                    color = (0, 255, 0)  # Green
                
                det['track_id'] = best_track_id
                det['status'] = status
                det['status_color'] = color
            else:
                # New detection
                det['track_id'] = self.track_id_counter
                det['status'] = 'NEW'
                det['status_color'] = (255, 255, 255)  # White
                self.track_id_counter += 1
            
            tracked.append(det)
        
        # Update tracking
        self.prev_detections = {det['track_id']: det for det in tracked}
        
        return tracked
    
    def draw_detections(self, frame, detections, fps=0, debug=False):
        """Draw bounding boxes and labels"""
        annotated = frame.copy()
        
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            class_name = det['class']
            conf = det['confidence']
            status = det.get('status', 'NEW')
            status_color = det.get('status_color', (255, 255, 255))
            
            # Get debug info
            yolo_class = det.get('yolo_class', '')
            cnn_class = det.get('cnn_class', '')
            cnn_conf = det.get('cnn_conf', 0.0)
            
            # Draw bounding box
            color = CLASS_COLORS.get(class_name, (255, 255, 255))
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label = f"{class_name}: {conf:.2f}"
            (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(annotated, (x1, y1 - label_h - 10), (x1 + label_w, y1), color, -1)
            cv2.putText(annotated, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            
            # Draw debug info if enabled
            if debug and yolo_class and cnn_class:
                debug_label = f"Y:{yolo_class} C:{cnn_class}({cnn_conf:.2f})"
                cv2.putText(annotated, debug_label, (x1, y1 - 25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
            
            # Draw distance status
            if status != 'NEW':
                status_label = f"[{status}]"
                cv2.putText(annotated, status_label, (x1, y2 + 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 2)
        
        # Draw FPS and vehicle count
        info_y = 30
        cv2.putText(annotated, f"FPS: {fps:.1f}", (10, info_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.putText(annotated, f"Vehicles: {len(detections)}", (10, info_y + 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return annotated


def main():
    parser = argparse.ArgumentParser(description='Real-time Camera Vehicle Detection')
    parser.add_argument('--model', type=str, 
                       default='checkpoints/transfer_resnet18/best_model.pth',
                       help='Path to CNN model checkpoint')
    parser.add_argument('--camera', type=str, default='0', 
                       help='Camera device ID or video file path (default: 0)')
    parser.add_argument('--width', type=int, default=1280,
                       help='Camera width (default: 1280)')
    parser.add_argument('--height', type=int, default=720,
                       help='Camera height (default: 720)')
    parser.add_argument('--device', type=str, default='cuda', 
                       choices=['cuda', 'cpu'])
    parser.add_argument('--save', type=str, help='Save output video')
    parser.add_argument('--debug', action='store_true',
                       help='Show YOLO and CNN predictions for debugging')
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("🚗 REAL-TIME CAMERA VEHICLE DETECTION")
    print("="*60 + "\n")
    
    # Check if model exists
    if not Path(args.model).exists():
        print(f"❌ Model not found: {args.model}")
        print("\nAvailable models:")
        checkpoints_dir = Path('checkpoints')
        if checkpoints_dir.exists():
            for model_dir in checkpoints_dir.iterdir():
                if model_dir.is_dir():
                    model_file = model_dir / 'best_model.pth'
                    if model_file.exists():
                        print(f"  ✅ {model_file}")
        return
    
    # Initialize detector
    detector = CameraVehicleDetector(args.model, device=args.device)
    
    # Determine if camera is a file or device ID
    try:
        camera_id = int(args.camera)
        is_file = False
    except ValueError:
        camera_id = args.camera
        is_file = True
    
    # Open camera or video file
    if is_file:
        print(f"📹 Opening video file: {camera_id}")
    else:
        print(f"📷 Opening camera {camera_id}...")
    
    cap = cv2.VideoCapture(camera_id)
    
    if not cap.isOpened():
        if is_file:
            print(f"❌ Failed to open video file: {camera_id}")
        else:
            print(f"❌ Failed to open camera {camera_id}")
        return
    
    # Set camera resolution (only for camera devices, not video files)
    if not is_file:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    if is_file:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        print(f"✅ Video opened: {actual_width}x{actual_height}, {total_frames} frames @ {video_fps:.2f} FPS")
    else:
        print(f"✅ Camera opened: {actual_width}x{actual_height}")
    
    # Setup video writer
    writer = None
    if args.save:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(args.save, fourcc, 20, (actual_width, actual_height))
        print(f"💾 Saving to: {args.save}")
    
    print("\n🚀 Starting detection...")
    print("   Press 'q' to quit")
    print("   Press 's' to take screenshot")
    if is_file:
        print("   Press 'r' to restart video\n")
    else:
        print()
    
    frame_count = 0
    start_time = time.time()
    screenshot_count = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                if is_file:
                    print("\n🎬 End of video reached - looping...")
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                else:
                    print("❌ Failed to read frame")
                    break
            
            frame_count += 1
            
            # Detect and track
            detections = detector.detect_and_track(frame)
            
            # Calculate FPS
            elapsed = time.time() - start_time
            fps = frame_count / elapsed if elapsed > 0 else 0
            
            # Draw results
            annotated = detector.draw_detections(frame, detections, fps=fps, debug=args.debug)
            
            # Display
            cv2.imshow('Vehicle Detection - Press q to quit', annotated)
            
            # Save video
            if writer:
                writer.write(annotated)
            
            # Handle key press
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('s'):
                # Take screenshot
                screenshot_path = f"screenshot_{screenshot_count:03d}.jpg"
                cv2.imwrite(screenshot_path, annotated)
                print(f"📸 Screenshot saved: {screenshot_path}")
                screenshot_count += 1
    
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user")
    
    finally:
        # Cleanup
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()
        
        # Print statistics
        elapsed = time.time() - start_time
        avg_fps = frame_count / elapsed if elapsed > 0 else 0
        
        print("\n" + "="*60)
        print("📊 SESSION STATISTICS")
        print("="*60)
        print(f"Frames processed: {frame_count}")
        print(f"Time elapsed: {elapsed:.2f}s")
        print(f"Average FPS: {avg_fps:.2f}")
        if screenshot_count > 0:
            print(f"Screenshots taken: {screenshot_count}")
        print("="*60 + "\n")


if __name__ == "__main__":
    main()
