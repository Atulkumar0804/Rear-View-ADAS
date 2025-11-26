#!/usr/bin/env python3
"""
Fine-tune depth estimation model on rear-view camera data.
Addresses the issue of inverted/incorrect depth predictions.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
import json
import os
from pathlib import Path
from tqdm import tqdm
from transformers import DPTForDepthEstimation, DPTImageProcessor
from PIL import Image
import matplotlib.pyplot as plt


class RearViewDepthDataset(Dataset):
    """Dataset for rear-view camera depth estimation."""
    
    def __init__(self, video_paths, annotations_path, processor, split='train'):
        """
        Args:
            video_paths: List of video file paths
            annotations_path: Path to JSON with ground truth depths
            processor: DPT image processor
            split: 'train', 'val', or 'test'
        """
        self.video_paths = video_paths
        self.processor = processor
        self.split = split
        
        # Load annotations
        with open(annotations_path, 'r') as f:
            self.annotations = json.load(f)
        
        # Build frame list
        self.samples = []
        for video_path in video_paths:
            video_name = os.path.basename(video_path)
            if video_name in self.annotations:
                for frame_data in self.annotations[video_name]:
                    self.samples.append({
                        'video_path': video_path,
                        'frame_num': frame_data['frame'],
                        'detections': frame_data['detections']
                    })
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load frame from video
        cap = cv2.VideoCapture(sample['video_path'])
        cap.set(cv2.CAP_PROP_POS_FRAMES, sample['frame_num'])
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            raise ValueError(f"Cannot read frame {sample['frame_num']}")
        
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width = frame.shape[:2]
        
        # Create ground truth depth map
        depth_map = np.ones((height, width), dtype=np.float32) * 50.0  # Max depth
        
        # Fill in known depths from annotations
        for det in sample['detections']:
            x1, y1, x2, y2 = det['bbox']
            depth = det['depth']  # Ground truth depth in meters
            
            # Fill bbox region with ground truth depth
            depth_map[y1:y2, x1:x2] = depth
        
        # Process image for model
        inputs = self.processor(images=Image.fromarray(frame_rgb), return_tensors="pt")
        
        return {
            'pixel_values': inputs['pixel_values'].squeeze(0),
            'depth_map': torch.from_numpy(depth_map),
            'image': frame_rgb
        }


def create_annotation_template(video_folder, output_path):
    """
    Create annotation template JSON for manual labeling.
    Users will fill in ground truth depths.
    """
    videos = sorted(Path(video_folder).glob("cam_back_*.mp4"))
    
    annotations = {}
    
    for video_path in videos:
        video_name = video_path.name
        
        # Sample frames (every 5th frame)
        cap = cv2.VideoCapture(str(video_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        frame_samples = []
        for frame_num in range(0, total_frames, 5):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_samples.append({
                'frame': frame_num,
                'detections': [
                    # Template - users fill in actual values
                    {
                        'bbox': [100, 100, 200, 200],  # x1, y1, x2, y2
                        'depth': 10.0,  # Ground truth depth in meters
                        'class': 'car',
                        'notes': 'Manually measure or estimate'
                    }
                ]
            })
        
        cap.release()
        annotations[video_name] = frame_samples
    
    # Save template
    with open(output_path, 'w') as f:
        json.dump(annotations, f, indent=2)
    
    print(f"✅ Annotation template created: {output_path}")
    print(f"   Videos: {len(videos)}")
    print(f"   Please fill in ground truth depths manually")


def create_synthetic_annotations(video_folder, yolo_model_path, output_path):
    """
    Create synthetic annotations using YOLO + heuristics.
    Better than nothing for initial training.
    """
    from ultralytics import YOLO
    
    print("🔧 Creating synthetic depth annotations...")
    
    yolo = YOLO(yolo_model_path)
    videos = sorted(Path(video_folder).glob("cam_back_*.mp4"))
    
    annotations = {}
    
    for video_path in tqdm(videos, desc="Processing videos"):
        video_name = video_path.name
        
        cap = cv2.VideoCapture(str(video_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        
        frame_samples = []
        
        # Sample every 5th frame
        for frame_num in range(0, total_frames, 5):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = cap.read()
            if not ret:
                break
            
            # Run YOLO detection
            results = yolo(frame, verbose=False)
            
            detections = []
            for result in results:
                boxes = result.boxes
                for i in range(len(boxes)):
                    bbox = boxes.xyxy[i].cpu().numpy().astype(int)
                    conf = float(boxes.conf[i])
                    cls = int(boxes.cls[i])
                    cls_name = result.names[cls]
                    
                    if conf < 0.4:
                        continue
                    
                    x1, y1, x2, y2 = bbox
                    
                    # Heuristic depth estimation based on bbox properties
                    # Assumption: larger bbox area = closer object
                    bbox_area = (x2 - x1) * (y2 - y1)
                    bbox_height = y2 - y1
                    
                    # Bottom of bbox (y2) indicates distance
                    # Objects lower in image are closer (rear view)
                    normalized_y = y2 / height
                    
                    # Estimate depth using heuristics
                    # This is rough but better than random
                    if cls_name in ['car', 'truck', 'bus']:
                        # Vehicle depth based on size and position
                        if bbox_area > 50000:  # Very large = very close
                            depth = 3.0 + (1.0 - normalized_y) * 2.0
                        elif bbox_area > 20000:  # Large = close
                            depth = 5.0 + (1.0 - normalized_y) * 3.0
                        else:  # Small = far
                            depth = 10.0 + (1.0 - normalized_y) * 5.0
                    else:  # person
                        # People are usually closer in rear view
                        depth = 3.0 + (1.0 - normalized_y) * 4.0
                    
                    detections.append({
                        'bbox': [int(x1), int(y1), int(x2), int(y2)],
                        'depth': float(depth),
                        'class': cls_name,
                        'confidence': float(conf),
                        'heuristic': 'size_position_based'
                    })
            
            if detections:
                frame_samples.append({
                    'frame': frame_num,
                    'detections': detections
                })
        
        cap.release()
        annotations[video_name] = frame_samples
    
    # Save annotations
    with open(output_path, 'w') as f:
        json.dump(annotations, f, indent=2)
    
    print(f"\n✅ Synthetic annotations created: {output_path}")
    print(f"   Videos: {len(videos)}")
    print(f"   Total annotated frames: {sum(len(v) for v in annotations.values())}")
    
    return annotations


def train_depth_model(
    train_videos,
    val_videos,
    annotations_path,
    output_dir,
    num_epochs=10,
    batch_size=4,
    learning_rate=1e-5
):
    """Fine-tune DPT model on rear-view data."""
    
    print("🔧 Initializing depth fine-tuning...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"   Device: {device}")
    
    # Load pre-trained model
    model_name = "Intel/dpt-hybrid-midas"
    model = DPTForDepthEstimation.from_pretrained(model_name)
    processor = DPTImageProcessor.from_pretrained(model_name)
    
    model = model.to(device)
    
    # Create datasets
    print("\n📊 Loading datasets...")
    train_dataset = RearViewDepthDataset(train_videos, annotations_path, processor, 'train')
    val_dataset = RearViewDepthDataset(val_videos, annotations_path, processor, 'val')
    
    print(f"   Train samples: {len(train_dataset)}")
    print(f"   Val samples: {len(val_dataset)}")
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    # Loss and optimizer
    criterion = nn.L1Loss()  # Mean Absolute Error
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    # Training loop
    best_val_loss = float('inf')
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n🚀 Starting training for {num_epochs} epochs...\n")
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
        for batch in pbar:
            pixel_values = batch['pixel_values'].to(device)
            gt_depth = batch['depth_map'].to(device)
            
            # Forward pass
            outputs = model(pixel_values)
            predicted_depth = outputs.predicted_depth
            
            # Resize prediction to match ground truth
            predicted_depth = nn.functional.interpolate(
                predicted_depth.unsqueeze(1),
                size=gt_depth.shape[-2:],
                mode='bilinear',
                align_corners=False
            ).squeeze(1)
            
            # Loss (only on annotated regions, depth < 50)
            mask = (gt_depth < 50.0).float()
            loss = criterion(predicted_depth * mask, gt_depth * mask)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        train_loss /= len(train_loader)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]")
            for batch in pbar:
                pixel_values = batch['pixel_values'].to(device)
                gt_depth = batch['depth_map'].to(device)
                
                outputs = model(pixel_values)
                predicted_depth = outputs.predicted_depth
                
                predicted_depth = nn.functional.interpolate(
                    predicted_depth.unsqueeze(1),
                    size=gt_depth.shape[-2:],
                    mode='bilinear',
                    align_corners=False
                ).squeeze(1)
                
                mask = (gt_depth < 50.0).float()
                loss = criterion(predicted_depth * mask, gt_depth * mask)
                
                val_loss += loss.item()
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        val_loss /= len(val_loader)
        
        # Learning rate schedule
        scheduler.step()
        
        print(f"\n📊 Epoch {epoch+1}/{num_epochs}")
        print(f"   Train Loss: {train_loss:.4f}")
        print(f"   Val Loss: {val_loss:.4f}")
        print(f"   LR: {optimizer.param_groups[0]['lr']:.6f}\n")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_path = os.path.join(output_dir, 'best_depth_model')
            model.save_pretrained(save_path)
            processor.save_pretrained(save_path)
            print(f"   ✅ Best model saved: {save_path}\n")
    
    print("🎉 Training complete!")
    print(f"   Best validation loss: {best_val_loss:.4f}")
    print(f"   Model saved to: {output_dir}")


def main():
    """Main pipeline for depth model fine-tuning."""
    
    # Paths
    video_folder = "/home/atul/Desktop/atul/rear_view_adas_monocular/CNN/dataset/test_viedos"
    yolo_model = "/home/atul/Desktop/atul/rear_view_adas_monocular/CNN/models/yolo/yolov8n_RearView.pt"
    annotations_path = "/home/atul/Desktop/atul/rear_view_adas_monocular/CNN/depth_estimation/depth_annotations.json"
    output_dir = "/home/atul/Desktop/atul/rear_view_adas_monocular/CNN/depth_estimation/finetuned_model"
    
    print("="*70)
    print("🎯 DEPTH MODEL FINE-TUNING PIPELINE")
    print("="*70)
    
    # Step 1: Create synthetic annotations if not exists
    if not os.path.exists(annotations_path):
        print("\n📝 Step 1: Creating synthetic depth annotations...")
        create_synthetic_annotations(video_folder, yolo_model, annotations_path)
    else:
        print(f"\n✓ Using existing annotations: {annotations_path}")
    
    # Step 2: Split videos into train/val/test
    print("\n📂 Step 2: Splitting videos...")
    all_videos = sorted(Path(video_folder).glob("cam_back_[1-9].mp4"))
    
    # Split: 60% train, 20% val, 20% test
    n = len(all_videos)
    train_videos = [str(v) for v in all_videos[:int(0.6*n)]]
    val_videos = [str(v) for v in all_videos[int(0.6*n):int(0.8*n)]]
    test_videos = [str(v) for v in all_videos[int(0.8*n):]]
    
    print(f"   Train: {len(train_videos)} videos - {[Path(v).name for v in train_videos]}")
    print(f"   Val: {len(val_videos)} videos - {[Path(v).name for v in val_videos]}")
    print(f"   Test: {len(test_videos)} videos - {[Path(v).name for v in test_videos]}")
    
    # Step 3: Train model
    print("\n🚀 Step 3: Fine-tuning depth model...")
    train_depth_model(
        train_videos=train_videos,
        val_videos=val_videos,
        annotations_path=annotations_path,
        output_dir=output_dir,
        num_epochs=10,
        batch_size=2,  # Small batch for GPU memory
        learning_rate=1e-5
    )
    
    print("\n" + "="*70)
    print("✅ PIPELINE COMPLETE")
    print("="*70)
    print(f"\n📍 Fine-tuned model location: {output_dir}")
    print(f"📍 Annotations: {annotations_path}")
    print(f"\n💡 Next steps:")
    print(f"   1. Review annotations in {annotations_path}")
    print(f"   2. Manually correct depth values if needed")
    print(f"   3. Re-run training with corrected annotations")
    print(f"   4. Update depth_estimator.py to use finetuned model")


if __name__ == "__main__":
    main()
