#!/usr/bin/env python3
"""
Script to add Person class to UVH-26 dataset using YOLO detections

This script:
1. Loads the UVH-26 dataset
2. Runs YOLO to detect persons in the images
3. Creates crops of detected persons
4. Adds them as the 15th class (Person) to the training dataset

Usage:
    python add_person_class.py --yolo_model path/to/yolo.pt --output_dir datasets/person_crops
"""

import cv2
import json
import argparse
from pathlib import Path
from tqdm import tqdm
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("⚠️ Warning: ultralytics not installed. Install with: pip install ultralytics")


def detect_persons_in_uvh26(yolo_model_path, uvh26_root, output_dir, conf_threshold=0.5):
    """
    Detect persons in UVH-26 dataset images and create crops
    
    Args:
        yolo_model_path: Path to YOLO model (.pt file)
        uvh26_root: Root directory of UVH-26 dataset
        output_dir: Output directory for person crops
        conf_threshold: Confidence threshold for detections
    """
    
    if not YOLO_AVAILABLE:
        print("❌ Error: Cannot run person detection without ultralytics")
        print("Install with: pip install ultralytics")
        return
    
    # Load YOLO model
    print(f"📦 Loading YOLO model from {yolo_model_path}")
    model = YOLO(yolo_model_path)
    
    # Create output directories
    output_dir = Path(output_dir)
    train_dir = output_dir / 'train' / 'Person'
    val_dir = output_dir / 'val' / 'Person'
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)
    
    # Process train and val sets
    for split in ['train', 'val']:
        print(f"\n🔍 Processing {split} set...")
        
        if split == 'train':
            json_path = Path(uvh26_root) / 'UVH-26-Train' / 'UVH-26-MV-Train.json'
            img_dir = Path(uvh26_root) / 'UVH-26-Train' / 'data'
            out_dir = train_dir
        else:
            json_path = Path(uvh26_root) / 'UVH-26-Val' / 'UVH-26-MV-Val.json'
            img_dir = Path(uvh26_root) / 'UVH-26-Val' / 'data'
            out_dir = val_dir
        
        # Load annotations
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # Get unique image files
        image_files = set()
        for img_info in data['images']:
            img_name = img_info['file_name']
            # Find actual file on disk
            matching_files = list(img_dir.rglob(img_name))
            if matching_files:
                image_files.add(matching_files[0])
        
        print(f"📊 Found {len(image_files)} images to process")
        
        person_count = 0
        
        # Process each image
        for img_path in tqdm(image_files, desc=f"Detecting persons in {split}"):
            # Run YOLO detection
            results = model(str(img_path), conf=conf_threshold, verbose=False)
            
            # Process detections
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    cls = int(box.cls[0])
                    # Check if it's a person (class 0 in COCO)
                    if cls == 0:  # Person class in COCO/YOLO
                        # Get bounding box
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        conf = float(box.conf[0])
                        
                        # Load image and crop
                        img = cv2.imread(str(img_path))
                        if img is not None:
                            h, w = img.shape[:2]
                            x1, y1 = max(0, int(x1)), max(0, int(y1))
                            x2, y2 = min(w, int(x2)), min(h, int(y2))
                            
                            if x2 > x1 and y2 > y1:
                                crop = img[y1:y2, x1:x2]
                                
                                # Save crop
                                crop_name = f"{img_path.stem}_person_{person_count}_conf{conf:.2f}.jpg"
                                crop_path = out_dir / crop_name
                                cv2.imwrite(str(crop_path), crop)
                                person_count += 1
        
        print(f"✅ Extracted {person_count} person crops from {split} set")
    
    print(f"\n✅ Done! Person crops saved to {output_dir}")
    print(f"📝 You can now train with 15 classes including Person")


def main():
    parser = argparse.ArgumentParser(description='Add Person class to UVH-26 dataset')
    parser.add_argument('--yolo_model', type=str, 
                       default='yolov8n.pt',
                       help='Path to YOLO model (default: yolov8n.pt)')
    parser.add_argument('--uvh26_root', type=str,
                       default='../datasets/UVH-26',
                       help='Root directory of UVH-26 dataset')
    parser.add_argument('--output_dir', type=str,
                       default='../datasets/person_crops',
                       help='Output directory for person crops')
    parser.add_argument('--conf', type=float, default=0.5,
                       help='Confidence threshold for YOLO detections (default: 0.5)')
    
    args = parser.parse_args()
    
    print("="*60)
    print("🚀 Person Class Addition for UVH-26 Dataset")
    print("="*60)
    
    detect_persons_in_uvh26(
        yolo_model_path=args.yolo_model,
        uvh26_root=args.uvh26_root,
        output_dir=args.output_dir,
        conf_threshold=args.conf
    )


if __name__ == '__main__':
    main()
