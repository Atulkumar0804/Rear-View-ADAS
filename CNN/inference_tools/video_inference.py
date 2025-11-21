#!/usr/bin/env python3
"""
Video Inference Script - Process video file and save output
No GUI, just processes video once and saves result

Usage:
    python video_inference.py --input video.mp4 --output result.mp4
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

# Get absolute path to YOLO model
SCRIPT_DIR = Path(__file__).parent.resolve()
CNN_DIR = SCRIPT_DIR.parent
YOLO_MODEL_PATH = str(CNN_DIR / "models" / "yolo" / "yolov8n_RearView.pt")

# Configuration
IMG_SIZE = (224, 224)
CONFIDENCE_THRESHOLD = 0.4

# Camera parameters (typical rear-view camera)
FOCAL_LENGTH = 1000  # pixels (approximate)
REAL_HEIGHT_CAR = 1.5  # meters (average car height)
REAL_HEIGHT_TRUCK = 3.0  # meters
REAL_HEIGHT_BUS = 3.2  # meters
REAL_HEIGHT_PERSON = 1.7  # meters

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
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                       std=[0.229, 0.224, 0.225])
])


class VideoDetector:
    def __init__(self, model_path, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        print(f"🔥 Device: {self.device}")
        
        # Load YOLO
        print("📦 Loading YOLO...")
        self.yolo = YOLO(YOLO_MODEL_PATH)
        print("✅ YOLO loaded")
        
        # Load CNN
        print(f"📦 Loading CNN: {model_path}")
        model_name = Path(model_path).parent.name
        print(f"   Model: {model_name}")
        
        self.cnn = create_model(model_name, num_classes=len(VEHICLE_CLASSES))
        checkpoint = torch.load(model_path, map_location=self.device)
        self.cnn.load_state_dict(checkpoint['model_state_dict'])
        self.cnn.to(self.device)
        self.cnn.eval()
        
        print("✅ CNN loaded")
        print(f"   Classes: {VEHICLE_CLASSES}\n")
        
        # Tracking history for velocity estimation
        self.prev_distances = {}  # track_id -> [distances over time]
        self.track_id_counter = 0
        self.prev_boxes = []
    
    def match_detections(self, current_boxes, prev_boxes, iou_threshold=0.3):
        """Match current detections with previous ones using IoU"""
        if not prev_boxes:
            return [-1] * len(current_boxes)
        
        matches = []
        for curr_box in current_boxes:
            best_iou = 0
            best_idx = -1
            
            for i, prev_box in enumerate(prev_boxes):
                iou = self.calculate_iou(curr_box, prev_box['bbox'])
                if iou > best_iou and iou > iou_threshold:
                    best_iou = iou
                    best_idx = prev_box.get('track_id', -1)
            
            matches.append(best_idx)
        
        return matches
    
    def calculate_iou(self, box1, box2):
        """Calculate Intersection over Union"""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        xi1 = max(x1_1, x1_2)
        yi1 = max(y1_1, y1_2)
        xi2 = min(x2_1, x2_2)
        yi2 = min(y2_1, y2_2)
        
        inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
        box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = box1_area + box2_area - inter_area
        
        return inter_area / union_area if union_area > 0 else 0
    
    def estimate_motion(self, track_id, current_distance):
        """Estimate if object is approaching, stable, or receding"""
        if track_id not in self.prev_distances:
            self.prev_distances[track_id] = []
        
        self.prev_distances[track_id].append(current_distance)
        
        # Keep only last 10 frames
        if len(self.prev_distances[track_id]) > 10:
            self.prev_distances[track_id].pop(0)
        
        # Need at least 5 frames to estimate
        if len(self.prev_distances[track_id]) < 5:
            return "stable"
        
        # Calculate trend
        distances = self.prev_distances[track_id]
        avg_change = (distances[-1] - distances[0]) / len(distances)
        
        # Threshold for motion detection (meters per frame)
        if avg_change < -0.3:  # Getting closer
            return "approaching"
        elif avg_change > 0.3:  # Getting farther
            return "receding"
        else:
            return "stable"
    
    def detect_frame(self, frame):
        """Detect vehicles in a single frame with tracking"""
        results = []
        current_boxes_raw = []
        
        # YOLO detection
        yolo_results = self.yolo(frame, verbose=False)[0]
        
        for detection in yolo_results.boxes.data:
            x1, y1, x2, y2, conf, cls_id = detection.cpu().numpy()
            cls_id = int(cls_id)
            
            if cls_id not in YOLO_CLASS_MAPPING:
                continue
            
            yolo_class = YOLO_CLASS_MAPPING[cls_id]
            
            # For persons, trust YOLO (shape-based detection is better)
            if yolo_class == 'person':
                if conf >= CONFIDENCE_THRESHOLD:
                    current_boxes_raw.append([int(x1), int(y1), int(x2), int(y2)])
                    results.append({
                        'bbox': [int(x1), int(y1), int(x2), int(y2)],
                        'class': 'person',
                        'confidence': float(conf),
                        'source': 'YOLO'
                    })
                continue
            
            # For vehicles, refine with CNN
            if conf >= CONFIDENCE_THRESHOLD:
                # Crop and classify with CNN
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                crop = frame[y1:y2, x1:x2]
                
                if crop.size > 0:
                    # Preprocess
                    crop_resized = cv2.resize(crop, IMG_SIZE)
                    crop_rgb = cv2.cvtColor(crop_resized, cv2.COLOR_BGR2RGB)
                    tensor = transform(crop_rgb).unsqueeze(0).to(self.device)
                    
                    # CNN prediction
                    with torch.no_grad():
                        output = self.cnn(tensor)
                        probs = torch.softmax(output, dim=1)[0]
                        cnn_conf, cnn_pred = torch.max(probs, 0)
                        cnn_class = VEHICLE_CLASSES[cnn_pred.item()]
                    
                    # Use CNN class if confidence is high
                    final_class = cnn_class if cnn_conf >= 0.6 else yolo_class
                    final_conf = float(cnn_conf) if cnn_conf >= 0.6 else float(conf)
                    
                    current_boxes_raw.append([x1, y1, x2, y2])
                    results.append({
                        'bbox': [x1, y1, x2, y2],
                        'class': final_class,
                        'confidence': final_conf,
                        'source': 'CNN' if cnn_conf >= 0.6 else 'YOLO'
                    })
        
        # Match with previous detections for tracking
        matches = self.match_detections(current_boxes_raw, self.prev_boxes)
        
        # Assign track IDs and estimate motion
        for i, result in enumerate(results):
            bbox_height = result['bbox'][3] - result['bbox'][1]
            distance = self.estimate_distance(bbox_height, result['class'])
            
            # Assign track ID
            if matches[i] == -1:
                # New detection
                track_id = self.track_id_counter
                self.track_id_counter += 1
            else:
                track_id = matches[i]
            
            result['track_id'] = track_id
            result['distance'] = distance
            
            # Estimate motion
            if distance:
                motion = self.estimate_motion(track_id, distance)
                result['motion'] = motion
            else:
                result['motion'] = "unknown"
        
        # Update prev_boxes for next frame
        self.prev_boxes = [{'bbox': r['bbox'], 'track_id': r['track_id']} for r in results]
        
        return results
    
    def estimate_distance(self, bbox_height, class_name):
        """Estimate distance based on bounding box height"""
        if bbox_height <= 0:
            return None
        
        # Get real-world height based on class
        real_heights = {
            'car': REAL_HEIGHT_CAR,
            'truck': REAL_HEIGHT_TRUCK,
            'bus': REAL_HEIGHT_BUS,
            'person': REAL_HEIGHT_PERSON
        }
        
        real_height = real_heights.get(class_name, REAL_HEIGHT_CAR)
        
        # Distance = (Real Height × Focal Length) / Pixel Height
        distance = (real_height * FOCAL_LENGTH) / bbox_height
        
        return distance
    
    def draw_detections(self, frame, detections, fps=None):
        """Draw bounding boxes and labels with distance and motion state"""
        annotated = frame.copy()
        
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            class_name = det['class']
            confidence = det['confidence']
            distance = det.get('distance', None)
            motion = det.get('motion', 'unknown')
            
            # Get base color for class
            color = CLASS_COLORS.get(class_name, (255, 255, 255))
            
            # Choose motion-based color for box border
            motion_colors = {
                'approaching': (0, 0, 255),    # Red - Warning!
                'receding': (0, 255, 255),     # Yellow - Moving away
                'stable': (0, 255, 0),         # Green - Safe
                'unknown': color                # Default class color
            }
            box_color = motion_colors.get(motion, color)
            
            # Draw box with motion-based color
            cv2.rectangle(annotated, (x1, y1), (x2, y2), box_color, 2)
            
            # Draw label with distance
            if distance:
                label = f"{class_name}: {confidence:.2f} | {distance:.1f}m"
            else:
                label = f"{class_name}: {confidence:.2f}"
            
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(annotated, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), box_color, -1)
            cv2.putText(annotated, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            
            # Draw distance below box (larger text)
            if distance:
                dist_text = f"{distance:.1f}m"
                dist_size, _ = cv2.getTextSize(dist_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                cv2.rectangle(annotated, (x1, y2), 
                             (x1 + dist_size[0] + 10, y2 + dist_size[1] + 10), box_color, -1)
                cv2.putText(annotated, dist_text, (x1 + 5, y2 + dist_size[1] + 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
            
            # Draw motion state on the right side of box
            if motion != 'unknown':
                motion_text = motion.upper()
                motion_size, _ = cv2.getTextSize(motion_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                motion_x = x2 - motion_size[0] - 10
                motion_y = y1 + 25
                cv2.rectangle(annotated, (motion_x - 5, motion_y - motion_size[1] - 5), 
                             (x2 - 5, motion_y + 5), box_color, -1)
                cv2.putText(annotated, motion_text, (motion_x, motion_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Draw FPS if provided
        if fps:
            fps_text = f"FPS: {fps:.1f}"
            cv2.putText(annotated, fps_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        return annotated


def process_video(input_path, output_path, model_path, device='cuda'):
    """Process video file"""
    
    # Check input
    if not Path(input_path).exists():
        print(f"❌ Input video not found: {input_path}")
        return False
    
    # Initialize detector
    detector = VideoDetector(model_path, device)
    
    # Open video
    print(f"📹 Opening: {input_path}")
    cap = cv2.VideoCapture(input_path)
    
    if not cap.isOpened():
        print(f"❌ Failed to open video")
        return False
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"✅ Video: {width}x{height} @ {fps:.2f} FPS, {total_frames} frames")
    
    # Setup writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    if not writer.isOpened():
        print(f"❌ Failed to create output video")
        cap.release()
        return False
    
    print(f"💾 Saving to: {output_path}")
    print(f"\n🚀 Processing...\n")
    
    # Process frames
    frame_count = 0
    start_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Detect
        detections = detector.detect_frame(frame)
        
        # Calculate FPS
        elapsed = time.time() - start_time
        processing_fps = frame_count / elapsed if elapsed > 0 else 0
        
        # Annotate
        annotated = detector.draw_detections(frame, detections, processing_fps)
        
        # Write
        writer.write(annotated)
        
        # Progress
        if frame_count % 10 == 0 or frame_count == total_frames:
            progress = frame_count / total_frames * 100
            print(f"   Frame {frame_count}/{total_frames} ({progress:.1f}%) - "
                  f"{processing_fps:.1f} FPS", end='\r')
    
    print()
    
    # Cleanup
    cap.release()
    writer.release()
    
    # Stats
    elapsed = time.time() - start_time
    avg_fps = frame_count / elapsed
    
    print(f"\n✅ Processing complete!")
    print(f"   Frames: {frame_count}")
    print(f"   Time: {elapsed:.1f}s")
    print(f"   Avg FPS: {avg_fps:.1f}")
    print(f"   Output: {output_path}")
    
    # Verify output
    output_file = Path(output_path)
    if output_file.exists():
        size_mb = output_file.stat().st_size / (1024 * 1024)
        print(f"   Size: {size_mb:.2f} MB")
        return True
    else:
        print(f"❌ Output file not created")
        return False


def main():
    parser = argparse.ArgumentParser(description='Video Detection Processor')
    parser.add_argument('--input', '-i', type=str, required=True,
                       help='Input video file')
    parser.add_argument('--output', '-o', type=str, required=True,
                       help='Output video file')
    parser.add_argument('--model', type=str,
                       default='checkpoints/transfer_resnet18/best_model.pth',
                       help='CNN model checkpoint')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'])
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("🎬 VIDEO DETECTION PROCESSOR")
    print("="*60 + "\n")
    
    success = process_video(args.input, args.output, args.model, args.device)
    
    if success:
        print("\n" + "="*60)
        print("🎉 Done!")
        print("="*60 + "\n")
        return 0
    else:
        print("\n" + "="*60)
        print("❌ Processing failed")
        print("="*60 + "\n")
        return 1


if __name__ == "__main__":
    exit(main())
