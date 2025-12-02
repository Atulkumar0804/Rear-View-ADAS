"""
Real-time Camera Inference for Vehicle Detection
Simplified version for easy camera-based predictions

Usage:
    # Using best model (mobilenet_inspired)
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

from ultralytics import YOLO
import os

# Get absolute path to YOLO model
SCRIPT_DIR = Path(__file__).parent.resolve()
CNN_DIR = SCRIPT_DIR.parent
# Use YOLOv11 from models folder
YOLO_MODEL_PATH = str(CNN_DIR / "models/yolo/yolo11x-seg.pt")

# Configuration
IMG_SIZE = (224, 224)
CONFIDENCE_THRESHOLD = 0.4

# Camera parameters (typical rear-view camera)
FOCAL_LENGTH = 1000  # pixels (approximate)

# Real-world heights for distance estimation (in meters)
REAL_HEIGHTS = {
    'Hatchback': 1.5,        # Average hatchback height
    'Sedan': 1.5,            # Average sedan height
    'SUV': 1.8,              # SUVs are taller
    'MUV': 1.9,              # Multi-utility vehicles
    'Bus': 3.2,              # Standard bus height
    'Truck': 3.0,            # Commercial truck height
    'Three-wheeler': 1.6,    # Auto-rickshaw height
    'Two-wheeler': 1.3,      # Motorcycle/scooter with rider
    'LCV': 2.2,              # Light commercial vehicle
    'Mini-bus': 2.5,         # Compact bus
    'Tempo-traveller': 2.4,  # Passenger van
    'Bicycle': 1.2,          # Bicycle with rider
    'Van': 2.0,              # Delivery van
    'Others': 1.5,           # Default estimate
    'Person': 1.7,           # Average human height
    'Person + Two-wheeler': 1.6,
    'Person + Bicycle': 1.6,
    'Person + Three-wheeler': 1.6
}

# Vehicle classes - 12 classes from UVH-26 Filtered (Alphabetical order)
# Excluded: Person, Bicycle, Two-wheeler (handled by YOLO)
VEHICLE_CLASSES = [
    'Bus', 'Hatchback', 'LCV', 'MUV', 'Mini-bus', 
    'Others', 'SUV', 'Sedan', 'Tempo-traveller', 'Three-wheeler', 
    'Truck', 'Van'
]

# YOLO class mapping (map YOLO COCO classes to our CNN classes)
YOLO_CLASS_MAPPING = {
    0: 'Person',         # YOLO person -> Person
    1: 'Bicycle',        # YOLO bicycle -> Bicycle
    2: 'Sedan',          # YOLO car -> Sedan (Generic Car, will be refined)
    3: 'Two-wheeler',    # YOLO motorcycle -> Two-wheeler
    5: 'Bus',            # YOLO bus -> Bus
    7: 'Truck',          # YOLO truck -> Truck
}

# Color mapping for all classes
CLASS_COLORS = {
    'Hatchback': (0, 255, 0),        # Green
    'Sedan': (0, 255, 127),          # Spring Green
    'SUV': (0, 255, 255),            # Cyan
    'MUV': (127, 255, 0),            # Chartreuse
    'Bus': (0, 165, 255),            # Orange-Blue
    'Truck': (255, 165, 0),          # Orange
    'Three-wheeler': (255, 255, 0),  # Yellow
    'Two-wheeler': (255, 0, 127),    # Deep Pink
    'LCV': (255, 127, 80),           # Coral
    'Mini-bus': (138, 43, 226),      # Blue Violet
    'Tempo-traveller': (147, 112, 219), # Medium Purple
    'Bicycle': (255, 192, 203),      # Pink
    'Van': (255, 140, 0),            # Dark Orange
    'Others': (128, 128, 128),       # Gray
    'Person': (255, 0, 255),         # Magenta
    'Person + Two-wheeler': (255, 0, 127),
    'Person + Bicycle': (255, 192, 203),
    'Person + Three-wheeler': (255, 255, 0),
}

# CNN Transform
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


class CameraVehicleDetector:
    """Real-time camera-based vehicle detector"""
    
    def __init__(self, model_path=None, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f"🔥 Device: {self.device}")
        
        # Load YOLO
        print("📦 Loading YOLO...")
        self.yolo = YOLO(YOLO_MODEL_PATH)
        print("✅ YOLO loaded")
        
        # Load Fine-tuned Classifier
        # Use CNN_DIR to locate the weights correctly
        CLASSIFIER_PATH = str(CNN_DIR / "models/classifier/weights/best.pt")
        print(f"📦 Loading Classifier: {CLASSIFIER_PATH}")
        if Path(CLASSIFIER_PATH).exists():
            self.classifier = YOLO(CLASSIFIER_PATH)
            print("✅ Classifier loaded")
        else:
            print(f"❌ Classifier not found at {CLASSIFIER_PATH}")
            print("   Falling back to YOLO-only mode")
            self.classifier = None
        
        # Tracking history for velocity estimation
        self.prev_distances = {}  # track_id -> [distances over time]
        self.track_id_counter = 0
        self.prev_boxes = []
        # Track class history for smoothing (track_id -> deque of (class, conf))
        from collections import deque, defaultdict
        self.track_classes = defaultdict(lambda: deque(maxlen=5))
        # Minimum IoU to consider same track (increase to reduce ID switches)
        self.IOU_THRESHOLD = 0.45
        # Minimum crop area to run CNN classification (avoid tiny noisy crops)
        self.MIN_CROP_AREA = 64 * 64  # pixels
        
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
        
        # Keep last 30 frames (approx 1 second) for smoother estimation
        if len(self.prev_distances[track_id]) > 30:
            self.prev_distances[track_id].pop(0)
        
        # Need at least 15 frames to estimate reliably
        if len(self.prev_distances[track_id]) < 15:
            return "stable"
        
        # Calculate trend using robust averaging
        distances = self.prev_distances[track_id]
        time_steps = len(distances) - 1
        if time_steps == 0: return "stable"
        
        # Use average of recent vs old frames to reduce noise
        # Compare average of last 5 frames vs average of first 5 frames
        window = 5
        if len(distances) < window * 2:
            # Fallback for shorter history
            avg_change = (distances[-1] - distances[0]) / time_steps
        else:
            recent_avg = sum(distances[-window:]) / window
            old_avg = sum(distances[:window]) / window
            # Effective time difference is total length minus average offset of the two windows
            effective_steps = len(distances) - window 
            avg_change = (recent_avg - old_avg) / effective_steps
        
        # Threshold for motion detection (meters per frame)
        # 0.03 m/frame * 30 fps = ~1 m/s = 3.6 km/h
        threshold = 0.03
        
        if avg_change < -threshold:  # Getting closer
            return "approaching"
        elif avg_change > threshold:  # Getting farther
            return "receding"
        else:
            return "stable"

    def estimate_distance(self, bbox_height, class_name):
        """Estimate distance based on bounding box height"""
        if bbox_height <= 0:
            return None
        
        # Get real-world height based on class
        real_height = REAL_HEIGHTS.get(class_name, 1.5)  # Default to car height
        
        # Distance = (Real Height × Focal Length) / Pixel Height
        distance = (real_height * FOCAL_LENGTH) / bbox_height
        
        return distance
    
    def detect_frame(self, frame):
        """Detect vehicles in a single frame with tracking"""
        results = []
        current_boxes_raw = []
        
        # YOLO detection
        yolo_results = self.yolo(frame, verbose=False)[0]
        
        for i, detection in enumerate(yolo_results.boxes.data):
            x1, y1, x2, y2, conf, cls_id = detection.cpu().numpy()
            cls_id = int(cls_id)
            
            # Get mask if available
            mask = None
            if yolo_results.masks is not None:
                # masks.xy is a list of arrays, one for each detection
                if i < len(yolo_results.masks.xy):
                    mask = yolo_results.masks.xy[i]
            
            if cls_id not in YOLO_CLASS_MAPPING:
                continue
            
            yolo_class = YOLO_CLASS_MAPPING[cls_id]
            
            # For Person, Bicycle, and Two-wheeler, trust YOLO (excluded from secondary classifier)
            if yolo_class in ['Person', 'Bicycle', 'Two-wheeler']:
                if conf >= CONFIDENCE_THRESHOLD:
                    current_boxes_raw.append([int(x1), int(y1), int(x2), int(y2)])
                    results.append({
                        'bbox': [int(x1), int(y1), int(x2), int(y2)],
                        'class': yolo_class,
                        'confidence': float(conf),
                        'source': 'YOLO',
                        'mask': mask
                    })
                continue
            
            # For vehicles, refine with Fine-tuned Classifier
            if conf >= CONFIDENCE_THRESHOLD:
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                # Default to YOLO class
                final_class = yolo_class
                final_conf = float(conf)
                source = 'YOLO'
                
                # Crop vehicle for classification
                crop = frame[y1:y2, x1:x2]
                
                if crop.size > 0 and self.classifier is not None:
                    crop_area = crop.shape[0] * crop.shape[1]
                    if crop_area >= self.MIN_CROP_AREA:
                        # Run classifier
                        try:
                            cls_results = self.classifier(crop, verbose=False)
                            if cls_results and len(cls_results) > 0:
                                top1 = cls_results[0].probs.top1
                                top1_conf = cls_results[0].probs.top1conf.item()
                                
                                # Get class name from classifier names
                                pred_class = cls_results[0].names[top1]
                                
                                # Trust classifier if confidence is decent
                                if top1_conf > 0.4:
                                    final_class = pred_class
                                    final_conf = top1_conf
                                    source = 'YOLO_CLS'
                        except Exception as e:
                            print(f"Classifier error: {e}")

                current_boxes_raw.append([x1, y1, x2, y2])
                results.append({
                    'bbox': [x1, y1, x2, y2],
                    'class': final_class,
                    'confidence': final_conf,
                    'source': source,
                    'mask': mask
                })
        
        # Merge overlapping riders and vehicles
        results = self.merge_rider_and_vehicle(results)
        
        # Rebuild current_boxes_raw after merging
        current_boxes_raw = [r['bbox'] for r in results]

        # Match with previous detections for tracking (use slightly higher IOU)
        matches = self.match_detections(current_boxes_raw, self.prev_boxes, iou_threshold=self.IOU_THRESHOLD)
        
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

            # --- Per-track class smoothing ---
            # Maintain history of predicted classes for the track and compute weighted vote
            cls = result['class']
            conf_val = float(result.get('confidence', 0.0))
            # append to history
            self.track_classes[track_id].append((cls, conf_val))

            # Weighted vote across history
            votes = {}
            for c, cconf in self.track_classes[track_id]:
                votes.setdefault(c, 0.0)
                votes[c] += cconf

            # Pick class with highest accumulated confidence
            if votes:
                stable_class = max(votes.items(), key=lambda x: x[1])[0]
                stable_conf = votes[stable_class] / len(self.track_classes[track_id])
                
                # If stable class differs from current but previous history strongly prefers previous, keep stable
                # EXCEPTION: If current class is Person, trust it immediately to avoid delay in correction
                if 'Person' in cls:
                    result['class'] = cls
                else:
                    result['class'] = stable_class
                    result['confidence'] = float(min(1.0, stable_conf))
        
        # Update prev_boxes for next frame
        self.prev_boxes = [{'bbox': r['bbox'], 'track_id': r['track_id']} for r in results]
        
        return results
    
    def merge_rider_and_vehicle(self, results):
        """
        Merge overlapping 'Person' and 'Two-wheeler'/'Bicycle' detections.
        Returns a filtered list of results.
        """
        final_results = []
        persons = []
        vehicles = []
        others = []
        
        ridable_classes = ['Two-wheeler', 'Bicycle', 'Three-wheeler']
        
        for r in results:
            if r['class'] == 'Person':
                persons.append(r)
            elif r['class'] in ridable_classes:
                vehicles.append(r)
            else:
                others.append(r)
        
        used_persons = set()
        
        for v in vehicles:
            v_box = v['bbox']
            # Start merged box with vehicle box
            mx1, my1, mx2, my2 = v_box
            
            # Check for overlapping persons
            merged_any = False
            for i, p in enumerate(persons):
                if i in used_persons:
                    continue
                
                p_box = p['bbox']
                
                # Check overlap
                xA = max(mx1, p_box[0])
                yA = max(my1, p_box[1])
                xB = min(mx2, p_box[2])
                yB = min(my2, p_box[3])
                
                interArea = max(0, xB - xA) * max(0, yB - yA)
                p_area = (p_box[2] - p_box[0]) * (p_box[3] - p_box[1])
                
                # Intersection over Person Area (IoP)
                # If a significant portion of the person overlaps with the vehicle
                iop = interArea / p_area if p_area > 0 else 0
                
                if iop > 0.2: # 20% overlap is enough to consider them related in 2D view
                    # Merge
                    mx1 = min(mx1, p_box[0])
                    my1 = min(my1, p_box[1])
                    mx2 = max(mx2, p_box[2])
                    my2 = max(my2, p_box[3])
                    used_persons.add(i)
                    merged_any = True
            
            # Update vehicle box
            v['bbox'] = [mx1, my1, mx2, my2]
            if merged_any:
                v['class'] = f"Person + {v['class']}"
            final_results.append(v)
            
        # Add remaining persons
        for i, p in enumerate(persons):
            if i not in used_persons:
                final_results.append(p)
                
        # Add others
        final_results.extend(others)
        
        return final_results
    
    def draw_detections(self, frame, detections, fps=None, debug=False):
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
            
            # Draw mask if available
            mask = det.get('mask')
            if mask is not None:
                # Create overlay
                overlay = annotated.copy()
                # Convert mask points to int32
                pts = mask.astype(np.int32)
                cv2.fillPoly(overlay, [pts], color)
                # Apply transparency
                alpha = 0.4
                cv2.addWeighted(overlay, alpha, annotated, 1 - alpha, 0, annotated)
            
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



def main():
    parser = argparse.ArgumentParser(description='Real-time Camera Vehicle Detection')
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
    
    # Initialize detector
    detector = CameraVehicleDetector(device=args.device)
    
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
            detections = detector.detect_frame(frame)
            
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
