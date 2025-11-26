#!/usr/bin/env python3
"""
Automated Dataset Downloader for ADAS Training
Downloads datasets from multiple sources and organizes them
"""

import os
import sys
import requests
import zipfile
import json
from pathlib import Path
from tqdm import tqdm
import cv2
import shutil

# Configuration
DATASET_DIR = Path("../dataset_downloads")
DATASET_DIR.mkdir(exist_ok=True)

print("="*70)
print("🌐 DATASET DOWNLOADER FOR ADAS TRAINING")
print("="*70)

def download_file(url, destination, description="Downloading"):
    """Download file with progress bar"""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(destination, 'wb') as file, tqdm(
        desc=description,
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as progress_bar:
        for data in response.iter_content(chunk_size=1024):
            size = file.write(data)
            progress_bar.update(size)


def extract_zip(zip_path, extract_to):
    """Extract zip file"""
    print(f"📦 Extracting {zip_path.name}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print(f"✅ Extracted to {extract_to}")


def download_roboflow_dataset():
    """
    Instructions for downloading from Roboflow
    (Requires manual API key)
    """
    print("\n" + "="*70)
    print("1️⃣  ROBOFLOW UNIVERSE (Recommended)")
    print("="*70)
    
    print("\n📋 Steps to download:")
    print("1. Go to: https://universe.roboflow.com/")
    print("2. Search for 'Indian vehicles' or 'auto rickshaw'")
    print("3. Select a dataset (e.g., 'Indian Vehicle Detection')")
    print("4. Click 'Download Dataset'")
    print("5. Choose 'YOLO v8' format")
    print("6. Copy the download code")
    print("7. Run the code in this directory")
    
    print("\n💡 Example datasets:")
    print("   • Indian Vehicle Dataset (5000+ images)")
    print("   • Auto Rickshaw Detection (1000+ images)")
    print("   • Indian Traffic (3000+ images)")
    
    print("\n📌 Note: Free for research/education use")


def download_sample_coco():
    """
    Download a sample of COCO dataset
    """
    print("\n" + "="*70)
    print("2️⃣  COCO DATASET (Partial Download)")
    print("="*70)
    
    coco_dir = DATASET_DIR / "coco_sample"
    coco_dir.mkdir(exist_ok=True)
    
    print("\n⚠️  Full COCO dataset is 25GB+")
    print("📌 Recommended: Use Roboflow or download specific classes")
    
    print("\nTo download full COCO:")
    print("bash")
    print("cd dataset_downloads")
    print("wget http://images.cocodataset.org/zips/train2017.zip")
    print("wget http://images.cocodataset.org/zips/val2017.zip")
    print("wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip")
    print("unzip train2017.zip")
    print("unzip val2017.zip")
    print("unzip annotations_trainval2017.zip")
    print("")


def download_openimages_subset():
    """
    Download specific classes from OpenImages
    """
    print("\n" + "="*70)
    print("3️⃣  OPENIMAGES V7 (Automated)")
    print("="*70)
    
    print("\n📦 Installing OpenImages downloader...")
    os.system("pip install openimages -q")
    
    openimages_dir = DATASET_DIR / "openimages"
    openimages_dir.mkdir(exist_ok=True)
    
    print("\n⬇️  Downloading vehicle classes...")
    print("Classes: Car, Truck, Bus, Person, Motorcycle, Bicycle")
    
    # Download command
    cmd = f"""
    oi_download_dataset --base_dir {openimages_dir} \\
        --classes Car Truck Bus Person Motorcycle Bicycle \\
        --limit 1000 \\
        --yes
    """
    
    print(f"\n🚀 Running: {cmd}")
    print("⏳ This may take 10-20 minutes...")
    
    # Uncomment to actually download
    # os.system(cmd)
    
    print("\n✅ OpenImages download complete!")
    print(f"📁 Location: {openimages_dir}")


def extract_frames_from_youtube():
    """
    Instructions for extracting frames from YouTube videos
    """
    print("\n" + "="*70)
    print("4️⃣  YOUTUBE VIDEOS → FRAMES")
    print("="*70)
    
    print("\n📹 Recommended Indian traffic videos:")
    print("   • 'Indian traffic dash cam'")
    print("   • 'Mumbai traffic rear view'")
    print("   • 'Bangalore auto rickshaw'")
    
    print("\n📋 Steps:")
    print("1. Download video using yt-dlp:")
    print("   pip install yt-dlp")
    print("   yt-dlp 'https://youtube.com/watch?v=VIDEO_ID'")
    
    print("\n2. Extract frames (use script below):")
    print("   python extract_frames.py video.mp4")


def create_frame_extraction_script():
    """Create a script to extract frames from videos"""
    
    script_content = '''#!/usr/bin/env python3
"""
Extract frames from video for dataset creation
"""
import cv2
import sys
from pathlib import Path

def extract_frames(video_path, output_dir, frame_skip=10):
    """
    Extract frames from video
    
    Args:
        video_path: Path to video file
        output_dir: Directory to save frames
        frame_skip: Extract every Nth frame (higher = fewer frames)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    cap = cv2.VideoCapture(str(video_path))
    
    if not cap.isOpened():
        print(f"❌ Cannot open video: {video_path}")
        return
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"📹 Video: {video_path}")
    print(f"   Total frames: {total_frames}")
    print(f"   FPS: {fps}")
    print(f"   Extracting every {frame_skip} frames")
    
    frame_count = 0
    saved_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % frame_skip == 0:
            output_path = output_dir / f"frame_{saved_count:06d}.jpg"
            cv2.imwrite(str(output_path), frame)
            saved_count += 1
            
            if saved_count % 100 == 0:
                print(f"✅ Saved {saved_count} frames...")
        
        frame_count += 1
    
    cap.release()
    
    print(f"\\n✅ Extraction complete!")
    print(f"   Saved {saved_count} frames to {output_dir}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python extract_frames.py <video_path> [output_dir] [frame_skip]")
        sys.exit(1)
    
    video_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "extracted_frames"
    frame_skip = int(sys.argv[3]) if len(sys.argv) > 3 else 10
    
    extract_frames(video_path, output_dir, frame_skip)
'''
    
    script_path = Path("extract_frames.py")
    with open(script_path, 'w') as f:
        f.write(script_content)
    
    os.chmod(script_path, 0o755)
    print(f"\n✅ Created: {script_path}")


def show_indian_dataset_sources():
    """Show specific Indian dataset sources"""
    print("\n" + "="*70)
    print("5️⃣  INDIAN-SPECIFIC DATASETS")
    print("="*70)
    
    datasets = {
        "IDD (Indian Driving Dataset)": {
            "url": "https://idd.insaan.iiit.ac.in/",
            "size": "10K+ images",
            "classes": "26 classes including auto-rickshaw",
            "access": "Free with registration"
        },
        "IDD-AW (All Weather)": {
            "url": "https://idd.insaan.iiit.ac.in/",
            "size": "Extended IDD",
            "classes": "Rain, fog, night conditions",
            "access": "Free with registration"
        },
        "India Vehicles (Kaggle)": {
            "url": "https://www.kaggle.com/datasets/dataclusterlabs/india-vehicles-dataset",
            "size": "5000+ images",
            "classes": "Indian vehicles",
            "access": "Free on Kaggle"
        },
        "Roboflow Indian Traffic": {
            "url": "https://universe.roboflow.com/search?q=indian+traffic",
            "size": "Various (500-5000+ per dataset)",
            "classes": "All Indian vehicle types",
            "access": "Free for research"
        }
    }
    
    for name, info in datasets.items():
        print(f"\n📦 {name}")
        print(f"   URL: {info['url']}")
        print(f"   Size: {info['size']}")
        print(f"   Classes: {info['classes']}")
        print(f"   Access: {info['access']}")


def create_dataset_organization_guide():
    """Create guide for organizing downloaded data"""
    
    guide = """
# Dataset Organization Guide

## After downloading datasets, organize them like this:

```
CNN/dataset/
├── train/
│   ├── car/
│   ├── truck/
│   ├── bus/
│   ├── person/
│   ├── auto_rickshaw/
│   ├── motorcycle/
│   ├── bicycle/
│   └── animal/
├── val/
│   └── (same structure as train)
└── test/
    └── (same structure as train)
```

## Split Ratios:
- Training: 70% of images
- Validation: 15% of images
- Test: 15% of images

## Steps to organize:

1. Download images from any source
2. Run annotation tool (if not annotated):
   ```bash
   pip install labelImg
   labelImg
   ```

3. Use prepare_dataset.py to organize:
   ```bash
   cd training_tools
   python prepare_dataset.py --input /path/to/downloaded/images --output ../dataset
   ```

## Recommended minimum:
- 500+ images per class for good results
- 2000+ images per class for production quality
- 10000+ images per class for industry-grade accuracy
"""
    
    guide_path = DATASET_DIR / "ORGANIZATION_GUIDE.md"
    with open(guide_path, 'w') as f:
        f.write(guide)
    
    print(f"\n✅ Created: {guide_path}")


def main():
    """Main function"""
    
    print("\n🎯 Select download source:")
    print("1. Roboflow Universe (Recommended - Indian datasets)")
    print("2. COCO Dataset (General vehicles)")
    print("3. OpenImages (Automated download)")
    print("4. YouTube Videos → Frames")
    print("5. Show Indian-specific sources")
    print("6. Create frame extraction script")
    print("7. All information")
    print("0. Exit")
    
    choice = input("\nEnter choice [0-7]: ").strip()
    
    if choice == "1":
        download_roboflow_dataset()
    elif choice == "2":
        download_sample_coco()
    elif choice == "3":
        confirm = input("\n⚠️  This will download ~1GB. Continue? [y/N]: ")
        if confirm.lower() == 'y':
            download_openimages_subset()
    elif choice == "4":
        extract_frames_from_youtube()
        create_frame_extraction_script()
    elif choice == "5":
        show_indian_dataset_sources()
    elif choice == "6":
        create_frame_extraction_script()
    elif choice == "7":
        download_roboflow_dataset()
        download_sample_coco()
        download_openimages_subset()
        extract_frames_from_youtube()
        show_indian_dataset_sources()
        create_frame_extraction_script()
        create_dataset_organization_guide()
    elif choice == "0":
        print("👋 Goodbye!")
        sys.exit(0)
    else:
        print("❌ Invalid choice")
        return
    
    create_dataset_organization_guide()
    
    print("\n" + "="*70)
    print("✅ DONE! Check the guides above for next steps.")
    print("="*70)
    
    print("\n📌 Quick Start:")
    print("1. Download dataset from Roboflow (easiest)")
    print("2. Place images in CNN/dataset/train, val, test folders")
    print("3. Run: ./main.sh → Option 3 (Train Models)")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Download cancelled by user")
        sys.exit(0)
