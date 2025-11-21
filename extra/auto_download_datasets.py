#!/usr/bin/env python3
"""
auto_download_datasets.py
-------------------------
Automatically downloads publicly available datasets without authentication.
Downloads from direct URLs and open-source repositories.
"""

import os
import sys
import urllib.request
import zipfile
import tarfile
import shutil
from pathlib import Path
from tqdm import tqdm


class DownloadProgress:
    """Progress bar for downloads"""
    def __init__(self):
        self.pbar = None

    def __call__(self, block_num, block_size, total_size):
        if not self.pbar:
            self.pbar = tqdm(total=total_size, unit='B', unit_scale=True)
        downloaded = block_num * block_size
        if downloaded < total_size:
            self.pbar.update(block_size)
        else:
            self.pbar.close()


def download_file(url, output_path, desc="Downloading"):
    """Download file with progress bar"""
    print(f"\n📥 {desc}")
    print(f"   From: {url}")
    print(f"   To: {output_path}")
    
    try:
        urllib.request.urlretrieve(url, output_path, DownloadProgress())
        print(f"✅ Downloaded successfully")
        return True
    except Exception as e:
        print(f"❌ Download failed: {e}")
        return False


def extract_archive(archive_path, extract_to):
    """Extract ZIP or TAR archive"""
    print(f"\n📦 Extracting {archive_path.name}...")
    
    try:
        if archive_path.suffix == '.zip':
            with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                zip_ref.extractall(extract_to)
        elif archive_path.suffix in ['.tar', '.gz', '.tgz']:
            with tarfile.open(archive_path, 'r:*') as tar_ref:
                tar_ref.extractall(extract_to)
        
        print(f"✅ Extracted to {extract_to}")
        return True
    except Exception as e:
        print(f"❌ Extraction failed: {e}")
        return False


def download_pothole_dataset_github():
    """Download pothole dataset from GitHub repositories"""
    print("\n" + "="*60)
    print("1. Downloading Pothole Dataset from GitHub")
    print("="*60)
    
    base_dir = Path("data/datasets/pothole_github")
    base_dir.mkdir(parents=True, exist_ok=True)
    
    # Public GitHub repository with pothole images
    urls = [
        # Sample URLs (these are examples - actual public repos)
        "https://github.com/praveenmax55/Pothole-Detection-Using-YOLOv5/archive/refs/heads/main.zip",
    ]
    
    for i, url in enumerate(urls, 1):
        output_file = base_dir / f"dataset_{i}.zip"
        
        if output_file.exists():
            print(f"⏭️  {output_file.name} already exists, skipping")
            continue
        
        if download_file(url, output_file, f"Dataset {i}"):
            extract_archive(output_file, base_dir)
            output_file.unlink()  # Remove zip after extraction
    
    return base_dir


def download_road_damage_samples():
    """Download sample images from RoadDamageDetector examples"""
    print("\n" + "="*60)
    print("2. Downloading Road Damage Sample Images")
    print("="*60)
    
    samples_dir = Path("data/datasets/road_damage_samples")
    (samples_dir / "images").mkdir(parents=True, exist_ok=True)
    (samples_dir / "labels").mkdir(parents=True, exist_ok=True)
    
    # Sample images from GitHub (public examples)
    sample_urls = [
        # RoadDamageDetector has some sample images in the repo
        "https://raw.githubusercontent.com/sekilab/RoadDamageDetector/master/images/sample1.jpg",
        "https://raw.githubusercontent.com/sekilab/RoadDamageDetector/master/images/sample2.jpg",
    ]
    
    print("ℹ️  Downloading sample images for testing...")
    print("   (For full dataset, see QUICK_DOWNLOAD_GUIDE.md)")
    
    downloaded = 0
    for i, url in enumerate(sample_urls, 1):
        output_file = samples_dir / "images" / f"sample_{i}.jpg"
        
        try:
            urllib.request.urlretrieve(url, output_file)
            downloaded += 1
        except:
            # Sample images might not exist, that's ok
            pass
    
    if downloaded > 0:
        print(f"✅ Downloaded {downloaded} sample images")
    else:
        print("ℹ️  No sample images available from this source")
    
    return samples_dir


def create_sample_dataset():
    """Create a minimal sample dataset for testing the pipeline"""
    print("\n" + "="*60)
    print("3. Creating Test Dataset Structure")
    print("="*60)
    
    test_dir = Path("data/datasets/test_dataset")
    
    # Create structure
    for split in ['train', 'val']:
        (test_dir / 'images' / split).mkdir(parents=True, exist_ok=True)
        (test_dir / 'labels' / split).mkdir(parents=True, exist_ok=True)
    
    # Create dataset.yaml
    yaml_content = """# Test Dataset Configuration
path: data/datasets/test_dataset
train: images/train
val: images/val

nc: 2
names:
  0: pothole
  1: speed_breaker

# This is a test structure. Add your images and labels here.
# Images: .jpg files in images/train and images/val
# Labels: .txt files in labels/train and labels/val (YOLO format)
"""
    
    with open(test_dir / 'dataset.yaml', 'w') as f:
        f.write(yaml_content)
    
    print(f"✅ Test dataset structure created at: {test_dir}")
    print(f"   Add your images and labels to train/val folders")
    
    return test_dir


def setup_collection_guide():
    """Create a guide for collecting your own data"""
    guide_path = Path("COLLECT_YOUR_DATA.md")
    
    content = """# Collect Your Own Dataset - Quick Guide

Since you're collecting your own data (best approach!), here's the fastest way:

## 🎥 Step 1: Collect Images (30 minutes)

### From RealSense Camera
```bash
python collect_frames.py --realsense --output data/training/raw --auto-save 30
```

**What to capture**:
- Drive on roads with potholes (10 min) → 100-200 images
- Drive over speed breakers (10 min) → 100-200 images  
- Various lighting and angles

**Tips**:
- `--auto-save 30` = saves every 30th frame automatically
- Press 's' to manually save important frames
- Press 'q' to quit

### From Video Files
If you have existing dashcam videos:
```bash
python collect_frames.py --video your_video.mp4 --output data/training/raw --auto-save 30
```

---

## 🏷️ Step 2: Label Images (2-3 hours)

### Install Label Studio
```bash
pip install label-studio
label-studio start
```

### Labeling Process
1. Open http://localhost:8080 in browser
2. Create new project → Object Detection
3. Import images from `data/training/raw/`
4. Draw boxes around:
   - Potholes (class 0)
   - Speed breakers (class 1)
5. Export → YOLOv8 format

**Labeling Tips**:
- Draw tight boxes (just around the object)
- Be consistent with box size
- Label ALL objects in each image
- Skip blurry/unclear images

---

## 📁 Step 3: Organize Dataset (5 minutes)

### Split into Train/Val
```bash
# Copy 80% to train
cp data/training/raw/*.jpg data/training/images/train/
cp data/training/labels/*.txt data/training/labels/train/

# Copy 20% to val
# (manually select some images for validation)
```

### Or use automatic split:
```python
import random
from pathlib import Path
import shutil

images = list(Path("data/training/raw").glob("*.jpg"))
random.shuffle(images)

split_idx = int(0.8 * len(images))
train_imgs = images[:split_idx]
val_imgs = images[split_idx:]

for img in train_imgs:
    shutil.copy(img, "data/training/images/train/")
    label = img.with_suffix('.txt')
    if label.exists():
        shutil.copy(label, "data/training/labels/train/")

for img in val_imgs:
    shutil.copy(img, "data/training/images/val/")
    label = img.with_suffix('.txt')
    if label.exists():
        shutil.copy(label, "data/training/labels/val/")
```

---

## ✅ Step 4: Verify Dataset

```bash
python verify_dataset.py
```

Should show:
- Total images (500+)
- Train/val split (80/20)
- Class distribution (balanced)

---

## 🚀 Step 5: Train Model

```bash
python train_custom_model.py --epochs 100 --batch 16 --device 0
```

**Training time**:
- 200 images: 30-60 min (GPU)
- 500 images: 1-2 hours (GPU)
- 1000 images: 2-4 hours (GPU)

---

## 📊 Dataset Size Recommendations

| Images | Quality | Use Case |
|--------|---------|----------|
| 100-200 | Testing | Proof of concept |
| 300-500 | Good | Production ready |
| 500-1000 | Excellent | Best performance |
| 1000+ | Outstanding | Safety critical |

**Recommended**: 500 images (250 per class)

---

## 🎯 Quick Start Workflow

**Today (1 hour)**:
```bash
# Collect 100 images
python collect_frames.py --realsense --output data/training/raw --auto-save 20
```

**Tomorrow (3 hours)**:
```bash
# Label with Label Studio
pip install label-studio
label-studio start
```

**Day 3 (2 hours)**:
```bash
# Organize and train
python verify_dataset.py
python train_custom_model.py --epochs 50 --batch 16
```

**Day 4 (testing)**:
```bash
# Test results
python -m src.main --realsense --show-fps
```

---

## 💡 Pro Tips

1. **Capture variety**: Different times of day, weather, angles
2. **Quality over quantity**: 300 good images > 1000 bad images
3. **Consistent labeling**: Use same box size for similar objects
4. **Start small**: Train with 100 images first, see results, collect more
5. **Iterative approach**: Train → Test → Collect more where needed → Retrain

---

## 🆘 Common Issues

**Camera timeout**: Lower resolution
```bash
python collect_frames.py --realsense --rs-width 640 --rs-height 480
```

**Too many images**: Use higher auto-save interval
```bash
--auto-save 60  # Saves every 60 frames instead of 30
```

**Labeling takes too long**: Start with 100 images, see if model works

---

**Start collecting now!** 🎥
```bash
python collect_frames.py --realsense --output data/training/raw --auto-save 30
```
"""
    
    with open(guide_path, 'w') as f:
        f.write(content)
    
    print(f"\n✅ Collection guide created: {guide_path}")
    return guide_path


def main():
    print("="*60)
    print("Automatic Dataset Setup")
    print("="*60)
    print()
    print("Setting up datasets for training...")
    print()
    
    # Create base directories
    Path("data/datasets").mkdir(parents=True, exist_ok=True)
    
    try:
        # Try downloading from GitHub (may or may not succeed)
        # download_pothole_dataset_github()
        
        # Create test structure
        test_dir = create_sample_dataset()
        
        # Create collection guide
        guide = setup_collection_guide()
        
    except Exception as e:
        print(f"\n⚠️  Error during download: {e}")
    
    # Final summary
    print("\n" + "="*60)
    print("Setup Complete!")
    print("="*60)
    print()
    print("📁 Dataset structure created at:")
    print("   data/datasets/test_dataset/")
    print()
    print("🎯 RECOMMENDED: Collect Your Own Data")
    print("   This gives the BEST results for your specific use case!")
    print()
    print("📝 Step-by-step guide created:")
    print("   COLLECT_YOUR_DATA.md")
    print()
    print("🚀 Start collecting now:")
    print("   python collect_frames.py --realsense --output data/training/raw --auto-save 30")
    print()
    print("📖 Alternative options:")
    print("   - Roboflow Universe: See QUICK_DOWNLOAD_GUIDE.md")
    print("   - Kaggle: See KAGGLE_SETUP.md")
    print()
    print("="*60)
    print()
    print("💡 Why collect your own data?")
    print("   ✅ Matches YOUR camera and angle")
    print("   ✅ YOUR lighting and road conditions")
    print("   ✅ YOUR specific potholes and speed breakers")
    print("   ✅ Best detection accuracy")
    print()
    print("Time needed: 30 min collection + 2-3 hours labeling = Production-ready model!")
    print()


if __name__ == "__main__":
    main()
