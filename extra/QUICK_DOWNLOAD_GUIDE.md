# 🚀 Quick Dataset Download Guide

## ✅ What's Already Downloaded

- ✅ **RoadDamageDetector Repository**: `data/datasets/road_damage/RoadDamageDetector/`
  - Tools and documentation ready
  - Images need manual download from Mendeley (optional)

---

## 📥 Download Pothole Dataset (Choose ONE Option)

### ⭐ OPTION 1: Roboflow Universe (EASIEST - RECOMMENDED)

**Best for**: Quick start, no authentication hassle

**Steps**:
1. Browser opened to: https://universe.roboflow.com/search?q=pothole%20detection
2. Pick a dataset (e.g., "Pothole Detection Computer Vision Project")
3. Click **"Download this Dataset"**
4. Choose format: **YOLOv8**
5. Extract to: `data/datasets/pothole_roboflow/`

**Benefits**:
- ✅ No API setup
- ✅ Already in YOLO format  
- ✅ Ready to train immediately
- ✅ Multiple sizes available (100-5000+ images)

---

### OPTION 2: Kaggle Dataset (665 images)

**Best for**: Standard benchmark dataset

**Setup Kaggle API** (one-time, 5 minutes):
```bash
# 1. Get token from https://www.kaggle.com/settings
# 2. Install:
mkdir -p ~/.kaggle
cp ~/Downloads/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json

# 3. Download:
python download_and_prepare_datasets.py
```

---

### OPTION 3: Manual Kaggle Download (No API)

**Steps**:
1. Visit: https://www.kaggle.com/datasets/chitholian/annotated-potholes-dataset
2. Click **"Download"** (requires Kaggle sign-in)
3. Extract ZIP to: `data/datasets/pothole_kaggle/`

---

### OPTION 4: Collect Your Own (BEST FOR YOUR USE CASE!)

**Why this is best**:
- Matches YOUR camera angle
- YOUR lighting conditions  
- YOUR roads

**Steps**:
```bash
python collect_frames.py --realsense --output data/training/raw --auto-save 30
# Drive for 20-30 minutes
# Auto-saves every 30 frames → 200-400 images
```

Then label with Label Studio:
```bash
pip install label-studio
label-studio start
# Open http://localhost:8080
# Draw boxes around potholes/speed breakers
```

---

## 🎯 Recommended Strategy

**For Best Results** - Combine multiple sources:

1. **Roboflow pothole dataset** (200-500 images) → Quick start
2. **Your own data** (200-500 images) → Custom to your roads
3. **Speed breaker data** (collect yourself) → Unique class

**Total**: 500-1000 images  
**Training time**: 2-4 hours (GPU)  
**Expected mAP**: 0.7-0.8 (excellent!)

---

## 📋 Step-by-Step Workflow

### Week 1: Get Data

**Day 1-2**: Download Roboflow pothole dataset (30 min)
```bash
# Extract to: data/datasets/pothole_roboflow/
```

**Day 3-4**: Collect your own data (1-2 hours)
```bash
python collect_frames.py --realsense --output data/training/raw --auto-save 30
# Collect 300-500 images: potholes + speed breakers
```

**Day 5**: Label your data (2-3 hours)
```bash
pip install label-studio
label-studio start
# Label speed breakers (potholes already labeled)
```

### Week 2: Train & Deploy

**Day 6**: Organize & verify
```bash
# Merge Roboflow + your data into data/training/
python verify_dataset.py
```

**Day 7**: Train model
```bash
python train_custom_model.py --epochs 100 --batch 16 --device 0
# Takes 2-4 hours (GPU) or 8-12 hours (CPU)
```

**Day 8**: Test & iterate
```bash
# Update config with trained model
python -m src.main --realsense --show-fps
```

---

## 🔍 Dataset Comparison

| Source | Images | Setup | Format | Best For |
|--------|--------|-------|--------|----------|
| **Roboflow** | 100-5000 | None | YOLOv8 ✅ | Quick start |
| **Kaggle** | 665 | API key | VOC → YOLO | Benchmark |
| **Your Own** | 200-500 | Collection | Label yourself | Best accuracy |
| **RoadDamage** | 10,000+ | Manual DL | Custom | Advanced |

---

## ✅ What to Do Right Now

1. **Download from Roboflow** (browser already opened):
   - Pick a pothole dataset
   - Download in YOLOv8 format
   - Extract to `data/datasets/pothole_roboflow/`

2. **Start collecting your own**:
   ```bash
   python collect_frames.py --realsense --output data/training/raw --auto-save 30
   ```

3. **Test current system** (already works!):
   ```bash
   python -m src.main --video data/samples/car-detection.mp4 --show-fps
   # Detects vehicles, pedestrians, animals, stop signs
   ```

---

## 📂 Where to Put Downloaded Data

```
data/
├── datasets/
│   ├── pothole_roboflow/      # ← Extract Roboflow dataset here
│   │   ├── train/
│   │   │   ├── images/
│   │   │   └── labels/
│   │   └── valid/
│   │       ├── images/
│   │       └── labels/
│   ├── pothole_kaggle/         # ← Or extract Kaggle here
│   └── road_damage/            # ✅ Already have this
│
└── training/                   # ← Your final training data
    ├── dataset.yaml
    ├── images/
    │   ├── train/
    │   └── val/
    └── labels/
        ├── train/
        └── val/
```

---

## 🆘 Need Help?

- **Roboflow download**: Pick any dataset, click download, choose YOLOv8
- **Kaggle setup**: See `KAGGLE_SETUP.md`
- **Training help**: See `TRAINING_QUICKSTART.md`
- **Full guide**: See `TRAINING.md`

---

## 💡 Pro Tip

**Start small, iterate fast**:
1. Download 100-200 images from Roboflow (5 min)
2. Train quick model (30 min)
3. Test it
4. Collect more data where needed
5. Retrain with full dataset

This approach gets you results in 1 hour instead of 1 week!

---

**Bottom Line**: Use Roboflow (easiest) + collect your own speed breaker data = Best results! 🎯

Browser is open to Roboflow - pick a dataset and download now! ✨
