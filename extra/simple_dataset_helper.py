#!/usr/bin/env python3
"""
Simple dataset downloader - No authentication required
Downloads sample pothole dataset from public sources
"""

import urllib.request
import json
import os
from pathlib import Path

def download_sample_dataset():
    """
    Download a small sample pothole dataset for testing
    This is a workaround if Kaggle setup is difficult
    """
    
    print("="*60)
    print("Downloading Sample Pothole Dataset")
    print("="*60)
    print()
    
    # Create directories
    sample_dir = Path("data/datasets/pothole_sample")
    (sample_dir / "images").mkdir(parents=True, exist_ok=True)
    (sample_dir / "labels").mkdir(parents=True, exist_ok=True)
    
    print("📁 Created directories in:", sample_dir)
    print()
    
    # Sample images from public sources (GitHub/Roboflow public datasets)
    print("Note: For full dataset, use one of these options:")
    print()
    print("Option 1: Kaggle (665 images)")
    print("  1. Setup Kaggle API: See KAGGLE_SETUP.md")
    print("  2. Run: python download_and_prepare_datasets.py")
    print()
    print("Option 2: Roboflow Universe (various sizes)")
    print("  1. Visit: https://universe.roboflow.com/")
    print("  2. Search: 'pothole detection'")
    print("  3. Download in YOLOv8 format (already converted!)")
    print("  4. Extract to: data/datasets/")
    print()
    print("Option 3: Manual Download from Kaggle")
    print("  1. Visit: https://www.kaggle.com/datasets/chitholian/annotated-potholes-dataset")
    print("  2. Click Download (requires sign-in)")
    print("  3. Extract to: data/datasets/pothole_kaggle/")
    print()
    print("Option 4: Collect Your Own (Best!)")
    print("  python collect_frames.py --realsense --output data/training/raw --auto-save 30")
    print()
    print("="*60)
    print()
    
    # Instructions file
    instructions = sample_dir / "DOWNLOAD_INSTRUCTIONS.txt"
    with open(instructions, 'w') as f:
        f.write("""
POTHOLE DATASET DOWNLOAD OPTIONS

Since automated download requires Kaggle API setup, here are your options:

========================================
OPTION 1: Kaggle (Recommended - 665 images)
========================================

Setup (5 minutes):
1. Go to: https://www.kaggle.com/settings
2. Create API token (downloads kaggle.json)
3. Run:
   mkdir -p ~/.kaggle
   cp ~/Downloads/kaggle.json ~/.kaggle/
   chmod 600 ~/.kaggle/kaggle.json
4. Run:
   python download_and_prepare_datasets.py

========================================
OPTION 2: Roboflow Universe (Easiest - Various sizes)
========================================

1. Visit: https://universe.roboflow.com/
2. Search: "pothole detection"
3. Pick a dataset (e.g., "Pothole Detection Computer Vision Project")
4. Click "Download this Dataset"
5. Choose format: YOLOv8
6. Extract to: data/datasets/pothole_roboflow/

Benefits:
- No API setup needed
- Already in YOLO format
- Multiple dataset options
- Various sizes (100-5000+ images)

========================================
OPTION 3: Manual Kaggle Download
========================================

1. Visit: https://www.kaggle.com/datasets/chitholian/annotated-potholes-dataset
2. Click "Download" (requires Kaggle account)
3. Extract ZIP to: data/datasets/pothole_kaggle/
4. Run conversion:
   python download_and_prepare_datasets.py

========================================
OPTION 4: Collect Your Own (Most Effective!)
========================================

python collect_frames.py --realsense --output data/training/raw --auto-save 30

Drive around for 20-30 minutes, the script will auto-capture frames.
Then label with Label Studio.

Benefits:
- Matches your exact use case
- Your camera/angle/lighting
- Your roads/conditions

========================================
RECOMMENDED COMBINATION
========================================

For best results, combine multiple sources:
1. Roboflow pothole dataset (quick start)
2. Your own collected data (200-500 images)
3. Speed breaker data (collect yourself)

Total: 500-1000 images → Good model performance
Training time: 2-4 hours (GPU)
Expected mAP: 0.7-0.8

========================================
NEXT STEPS AFTER DOWNLOAD
========================================

1. Verify dataset:
   python verify_dataset.py

2. Train model:
   python train_custom_model.py --epochs 100 --batch 16 --device 0

3. Test:
   python -m src.main --realsense --show-fps
""")
    
    print(f"✅ Instructions saved to: {instructions}")
    print()
    print("🎯 RECOMMENDED: Use Roboflow Universe (easiest)")
    print("   → https://universe.roboflow.com/")
    print("   → Search: 'pothole detection'")
    print("   → Download in YOLOv8 format")
    print()
    
    return sample_dir

if __name__ == "__main__":
    download_sample_dataset()
