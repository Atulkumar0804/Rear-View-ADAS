"""
Data Preparation Script for Rear-View Vehicle Detection & Distance Estimation

CORRECT UNDERSTANDING:
- CAM_BACK/1-9: Each folder is a different scene/view
- Within each folder: Sequential temporal frames from that scene
- Goal: Use YOLO to detect vehicles, track across frames, determine if approaching/receding

Process:
1. Load YOLO model (transfer learning - use as-is)
2. For each scene folder (1-9):
   - Process sequential frames
   - Detect vehicles using YOLO
   - Track vehicles across frames (using IoU matching)
   - Compute bounding box area changes to determine approach/recede
3. Save vehicle crops with:
   - Vehicle class (from YOLO)
   - Distance label (approaching/stationary/receding)
4. Create sequences for temporal models
5. Split into train/val/test (70/15/15)
"""

import os
import sys
import cv2
import numpy as np
import json
import shutil
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
import matplotlib.pyplot as plt

# Add parent directory to import YOLO
sys.path.append('..')
from ultralytics import YOLO

# Configuration
SOURCE_DIR = "../data/samples/CAM_BACK"
OUTPUT_DIR = "./dataset"
YOLO_MODEL = "../models/yolo/yolov8n_RearView.pt"
IMG_SIZE = (224, 224)  # Standard size for transfer learning
SEQUENCE_LENGTH = 5  # Frames to track for distance estimation
CONFIDENCE_THRESHOLD = 0.4  # YOLO detection confidence
MIN_BOX_AREA = 1500  # Minimum bbox area (pixels)
IOU_THRESHOLD = 0.3  # IoU threshold for tracking same vehicle

# Vehicle classes from YOLO (0: person, 1: bicycle, 2: car, 3: motorcycle, 5: bus, 7: truck)
YOLO_CLASS_MAPPING = {
    0: 'person',
    1: 'bicycle', 
    2: 'car',
    3: 'motorcycle',
    5: 'bus',
    7: 'truck'
}

VEHICLE_CLASSES = ['car', 'truck', 'bus', 'motorcycle', 'bicycle', 'person']
DISTANCE_LABELS = ['approaching', 'stationary', 'receding']


def compute_iou(box1, box2):
    """Compute IoU between two bounding boxes [x1, y1, x2, y2]"""
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])
    
    if x2_inter < x1_inter or y2_inter < y1_inter:
        return 0.0
    
    inter_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area
    
    return inter_area / union_area if union_area > 0 else 0.0


def compute_area(box):
    """Compute area of bounding box [x1, y1, x2, y2]"""
    return (box[2] - box[0]) * (box[3] - box[1])


class VehicleTracker:
    """Track vehicles across frames in a scene"""
    def __init__(self, scene_id):
        self.scene_id = scene_id
        self.tracks = []  # List of tracks, each track is a list of detections
        self.next_track_id = 0
    
    def update(self, detections, frame_idx):
        """
        Update tracks with new detections
        detections: list of (bbox, class_id, confidence)
        """
        if not self.tracks:
            # Initialize tracks
            for det in detections:
                self.tracks.append({
                    'track_id': self.next_track_id,
                    'detections': [(frame_idx, det)],
                    'class_id': det[1]
                })
                self.next_track_id += 1
            return
        
        # Match detections to existing tracks using IoU
        matched_tracks = set()
        matched_dets = set()
        
        for det_idx, det in enumerate(detections):
            best_iou = 0
            best_track_idx = -1
            
            for track_idx, track in enumerate(self.tracks):
                if track_idx in matched_tracks:
                    continue
                
                # Get last detection in track
                last_frame_idx, last_det = track['detections'][-1]
                
                # Only match if same class
                if last_det[1] != det[1]:
                    continue
                
                iou = compute_iou(last_det[0], det[0])
                if iou > IOU_THRESHOLD and iou > best_iou:
                    best_iou = iou
                    best_track_idx = track_idx
            
            if best_track_idx >= 0:
                self.tracks[best_track_idx]['detections'].append((frame_idx, det))
                matched_tracks.add(best_track_idx)
                matched_dets.add(det_idx)
        
        # Create new tracks for unmatched detections
        for det_idx, det in enumerate(detections):
            if det_idx not in matched_dets:
                self.tracks.append({
                    'track_id': self.next_track_id,
                    'detections': [(frame_idx, det)],
                    'class_id': det[1]
                })
                self.next_track_id += 1
    
    def get_sequences(self, min_length=5):
        """Get vehicle sequences with at least min_length detections"""
        sequences = []
        for track in self.tracks:
            if len(track['detections']) >= min_length:
                sequences.append(track)
        return sequences


class DatasetPreparer:
    def __init__(self, source_dir, output_dir, yolo_model_path):
        self.source_dir = Path(source_dir)
        self.output_dir = Path(output_dir)
        
        # Clean existing dataset
        if self.output_dir.exists():
            print(f"🗑️  Removing old dataset: {self.output_dir}")
            shutil.rmtree(self.output_dir)
        
        # Load YOLO model
        print("📦 Loading YOLO model...")
        self.yolo_model = YOLO(yolo_model_path)
        print(f"✅ YOLO model loaded successfully!")
        
        # Create output directories
        for split in ['train', 'val', 'test']:
            for cls in VEHICLE_CLASSES:
                (self.output_dir / split / cls).mkdir(parents=True, exist_ok=True)
            (self.output_dir / f'{split}_sequences').mkdir(parents=True, exist_ok=True)
        
        self.vehicle_crops = []  # Store all vehicle crop metadata
        self.sequences = []  # Store vehicle sequences
        
        self.stats = {
            'total_frames': 0,
            'total_detections': 0,
            'total_sequences': 0,
            'class_distribution': {cls: 0 for cls in VEHICLE_CLASSES},
            'distance_distribution': {label: 0 for label in DISTANCE_LABELS},
            'scenes_processed': 0
        }
    
    def process_scene_folder(self, scene_folder_path, scene_id):
        """Process all frames in a scene folder"""
        print(f"\n📁 Processing Scene {scene_id}: {scene_folder_path.name}")
        
        # Get all image files sorted by name
        image_files = sorted(scene_folder_path.glob("*.jpg"))
        if not image_files:
            print(f"⚠️  No images found in {scene_folder_path}")
            return
        
        print(f"   Found {len(image_files)} frames")
        
        # Initialize tracker for this scene
        tracker = VehicleTracker(scene_id)
        
        # Process each frame
        for frame_idx, img_path in enumerate(tqdm(image_files, desc=f"Scene {scene_id}")):
            frame = cv2.imread(str(img_path))
            if frame is None:
                continue
            
            self.stats['total_frames'] += 1
            
            # Run YOLO detection
            results = self.yolo_model(frame, conf=CONFIDENCE_THRESHOLD, verbose=False)
            
            # Extract detections
            detections = []
            for result in results:
                if result.boxes is None:
                    continue
                
                for box in result.boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    
                    # Only process vehicle classes
                    if cls_id not in YOLO_CLASS_MAPPING:
                        continue
                    
                    # Get bounding box
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    bbox = [int(x1), int(y1), int(x2), int(y2)]
                    
                    # Filter small boxes
                    if compute_area(bbox) < MIN_BOX_AREA:
                        continue
                    
                    detections.append((bbox, cls_id, conf))
                    self.stats['total_detections'] += 1
            
            # Update tracker
            tracker.update(detections, frame_idx)
            
            # Save individual crops
            for bbox, cls_id, conf in detections:
                x1, y1, x2, y2 = bbox
                crop = frame[y1:y2, x1:x2]
                
                if crop.size == 0:
                    continue
                
                # Resize to standard size
                crop_resized = cv2.resize(crop, IMG_SIZE)
                
                # Save crop info
                vehicle_class = YOLO_CLASS_MAPPING[cls_id]
                crop_info = {
                    'scene_id': scene_id,
                    'frame_idx': frame_idx,
                    'crop': crop_resized,
                    'bbox': bbox,
                    'class': vehicle_class,
                    'confidence': conf,
                    'frame_path': str(img_path)
                }
                self.vehicle_crops.append(crop_info)
                self.stats['class_distribution'][vehicle_class] += 1
        
        # Get vehicle sequences (tracks with 5+ frames)
        sequences = tracker.get_sequences(min_length=SEQUENCE_LENGTH)
        print(f"   ✅ Found {len(sequences)} vehicle sequences (5+ frames)")
        
        # Process sequences to determine distance changes
        for seq in sequences:
            track_id = seq['track_id']
            class_id = seq['class_id']
            vehicle_class = YOLO_CLASS_MAPPING[class_id]
            
            # Compute area changes
            areas = []
            frames_data = []
            
            for frame_idx, det in seq['detections']:
                bbox, cls_id, conf = det
                area = compute_area(bbox)
                areas.append(area)
                frames_data.append({
                    'frame_idx': frame_idx,
                    'bbox': bbox,
                    'area': area,
                    'confidence': conf
                })
            
            # Determine distance label based on area change
            first_area = areas[0]
            last_area = areas[-1]
            area_change = (last_area - first_area) / first_area
            
            if area_change > 0.15:  # Area increased by >15%
                distance_label = 'approaching'
            elif area_change < -0.15:  # Area decreased by >15%
                distance_label = 'receding'
            else:
                distance_label = 'stationary'
            
            # Store sequence
            sequence_info = {
                'scene_id': scene_id,
                'track_id': track_id,
                'vehicle_class': vehicle_class,
                'distance_label': distance_label,
                'num_frames': len(seq['detections']),
                'frames': frames_data,
                'area_change_pct': area_change * 100
            }
            self.sequences.append(sequence_info)
            self.stats['distance_distribution'][distance_label] += 1
            self.stats['total_sequences'] += 1
        
        self.stats['scenes_processed'] += 1
    
    def process_all_scenes(self):
        """Process all scene folders (1-9)"""
        print("\n" + "="*60)
        print("🚀 Starting Dataset Preparation with YOLO Detection")
        print("="*60)
        
        scene_folders = sorted([f for f in self.source_dir.iterdir() if f.is_dir()])
        print(f"\nFound {len(scene_folders)} scene folders: {[f.name for f in scene_folders]}")
        
        for scene_folder in scene_folders:
            scene_id = scene_folder.name
            self.process_scene_folder(scene_folder, scene_id)
    
    def split_and_save_dataset(self):
        """Split data into train/val/test and save"""
        print("\n" + "="*60)
        print("💾 Splitting and Saving Dataset")
        print("="*60)
        
        print(f"\nTotal vehicle crops: {len(self.vehicle_crops)}")
        print(f"Total sequences: {len(self.sequences)}")
        
        if len(self.vehicle_crops) == 0:
            print("❌ No vehicle crops found! Check YOLO detections.")
            return
        
        # Split vehicle crops by class for balanced split
        crops_by_class = defaultdict(list)
        for crop in self.vehicle_crops:
            crops_by_class[crop['class']].append(crop)
        
        train_crops, val_crops, test_crops = [], [], []
        
        for vehicle_class, crops in crops_by_class.items():
            n = len(crops)
            train_end = int(0.70 * n)
            val_end = int(0.85 * n)
            
            # Shuffle for randomness
            np.random.shuffle(crops)
            
            train_crops.extend(crops[:train_end])
            val_crops.extend(crops[train_end:val_end])
            test_crops.extend(crops[val_end:])
        
        print(f"\n📊 Split: Train={len(train_crops)}, Val={len(val_crops)}, Test={len(test_crops)}")
        
        # Save crops
        for split_name, crops in [('train', train_crops), ('val', val_crops), ('test', test_crops)]:
            print(f"\n💾 Saving {split_name} crops...")
            for idx, crop_info in enumerate(tqdm(crops, desc=split_name)):
                vehicle_class = crop_info['class']
                crop_img = crop_info['crop']
                
                # Save image
                output_path = self.output_dir / split_name / vehicle_class / f"{split_name}_{vehicle_class}_{idx:05d}.jpg"
                cv2.imwrite(str(output_path), crop_img)
        
        # Save sequences
        print(f"\n💾 Saving sequences...")
        seq_by_distance = defaultdict(list)
        for seq in self.sequences:
            seq_by_distance[seq['distance_label']].append(seq)
        
        for dist_label, seqs in seq_by_distance.items():
            n = len(seqs)
            train_end = int(0.70 * n)
            val_end = int(0.85 * n)
            
            np.random.shuffle(seqs)
            
            splits = {
                'train': seqs[:train_end],
                'val': seqs[train_end:val_end],
                'test': seqs[val_end:]
            }
            
            for split_name, split_seqs in splits.items():
                for idx, seq in enumerate(split_seqs):
                    seq_path = self.output_dir / f'{split_name}_sequences' / f'seq_{dist_label}_{idx:05d}.json'
                    with open(seq_path, 'w') as f:
                        json.dump(seq, f, indent=2)
        
        print("\n✅ Dataset saved successfully!")
    
    def print_statistics(self):
        """Print dataset statistics"""
        print("\n" + "="*60)
        print("📊 DATASET STATISTICS")
        print("="*60)
        
        print(f"\n🎬 Scenes processed: {self.stats['scenes_processed']}")
        print(f"🖼️  Total frames: {self.stats['total_frames']}")
        print(f"🚗 Total vehicle detections: {self.stats['total_detections']}")
        print(f"📹 Total sequences (5+ frames): {self.stats['total_sequences']}")
        
        print(f"\n📦 Vehicle Class Distribution:")
        for cls, count in self.stats['class_distribution'].items():
            print(f"   {cls:12s}: {count:4d} ({count/max(self.stats['total_detections'],1)*100:.1f}%)")
        
        print(f"\n🎯 Distance Label Distribution:")
        for label, count in self.stats['distance_distribution'].items():
            print(f"   {label:12s}: {count:4d} ({count/max(self.stats['total_sequences'],1)*100:.1f}%)")
        
        print("\n" + "="*60)
    
    def visualize_samples(self):
        """Create visualization of sample crops"""
        print("\n📸 Creating sample visualization...")
        
        # Sample 3 crops per class
        fig, axes = plt.subplots(len(VEHICLE_CLASSES), 3, figsize=(12, 2*len(VEHICLE_CLASSES)))
        fig.suptitle('Sample Vehicle Crops (YOLO Detected)', fontsize=16)
        
        for i, vehicle_class in enumerate(VEHICLE_CLASSES):
            class_crops = [c for c in self.vehicle_crops if c['class'] == vehicle_class]
            
            for j in range(3):
                ax = axes[i, j] if len(VEHICLE_CLASSES) > 1 else axes[j]
                
                if j < len(class_crops):
                    crop = class_crops[j]['crop']
                    # Convert BGR to RGB
                    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                    ax.imshow(crop_rgb)
                    conf = class_crops[j]['confidence']
                    ax.set_title(f"{vehicle_class}\n(conf: {conf:.2f})")
                else:
                    ax.text(0.5, 0.5, 'No sample', ha='center', va='center')
                
                ax.axis('off')
        
        plt.tight_layout()
        output_path = self.output_dir / 'sample_crops_yolo.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✅ Saved visualization: {output_path}")
        plt.close()


def main():
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Initialize dataset preparer
    preparer = DatasetPreparer(SOURCE_DIR, OUTPUT_DIR, YOLO_MODEL)
    
    # Process all scenes
    preparer.process_all_scenes()
    
    # Split and save
    preparer.split_and_save_dataset()
    
    # Print statistics
    preparer.print_statistics()
    
    # Visualize samples
    preparer.visualize_samples()
    
    print("\n" + "="*60)
    print("✅ DATASET PREPARATION COMPLETE!")
    print("="*60)
    print(f"\n📁 Dataset saved to: {OUTPUT_DIR}")
    print(f"\nNext steps:")
    print(f"1. Check sample crops: {OUTPUT_DIR}/sample_crops_yolo.png")
    print(f"2. Train models: python train.py")
    print(f"3. Run inference: python inference.py")
    print("\n" + "="*60)


if __name__ == "__main__":
    main()
