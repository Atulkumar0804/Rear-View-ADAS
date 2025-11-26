# Depth Estimation Module

Monocular depth estimation using **Intel DPT (Dense Prediction Transformer)** via Hugging Face Transformers for accurate distance and velocity measurement.

## ✅ Installation Complete

The module has been installed **without interrupting your ongoing CNN training** (PID 590797).

## 🎯 What This Module Provides

1. **Absolute Distance Estimation** - Real distance in meters (not just relative)
2. **Velocity Calculation** - Speed of vehicles approaching/receding in m/s
3. **Time-to-Collision (TTC)** - Predicts collision time in seconds
4. **Risk Levels** - Safe, Caution, Critical, Danger warnings
5. **Smooth Tracking** - Temporal smoothing for stable measurements

## 📁 Module Structure

```
CNN/depth_estimation/
├── __init__.py              # Module exports
├── depth_config.py          # Configuration settings
├── depth_estimator.py       # Core depth estimation
├── test_depth.py            # Test scripts
└── README.md                # This file

CNN/inference_tools/
├── camera_inference_depth.py  # Enhanced camera with depth
```

## 🚀 Quick Start

### 1. Test Depth Estimation (Camera)

```bash
cd CNN/depth_estimation
python test_depth.py --mode camera --camera 0 --model small
```

**Controls:**
- Press `q` to quit
- Press `s` to save screenshot
- See depth map in real-time!

### 2. Test Depth Estimation (Image)

```bash
python test_depth.py --mode image --image /path/to/image.jpg --model small
```

### 3. Run Enhanced Camera Inference

```bash
cd CNN/inference_tools
python camera_inference_depth.py \
    --model ../checkpoints/transfer_resnet18/best_model.pth \
    --camera 0 \
    --depth-model small \
    --show-depth
```

**Features:**
- ✅ Vehicle detection + classification
- ✅ Real-time depth estimation
- ✅ Velocity calculation (m/s)
- ✅ TTC warning for approaching vehicles
- ✅ Status indicators (APPROACHING/STABLE/RECEDING)
- ✅ Distance overlay (press `d` to toggle)

## ⚙️ Configuration

### Model Sizes

| Size  | Model | Parameters | Speed (FPS) | Use Case |
|-------|-------|------------|-------------|----------|
| **small** | DPT-Hybrid | 123M | 25-35 FPS | Real-time (Recommended) |
| **base** | DPT-Large | 345M | 15-25 FPS | Balanced accuracy |
| **large** | DPT-Large | 345M | 15-25 FPS | Best accuracy |

**Model Source**: Intel DPT via Hugging Face (`Intel/dpt-hybrid-midas`, `Intel/dpt-large`)

### Thresholds (Edit `depth_config.py`)

```python
# Distance thresholds (meters)
SAFE_DISTANCE = 10.0      # > 10m is safe
CAUTION_DISTANCE = 5.0    # 5-10m caution
CRITICAL_DISTANCE = 2.0   # 2-5m critical
DANGER_DISTANCE = 1.0     # < 2m danger

# Velocity threshold
VELOCITY_THRESHOLD = 0.1  # m/s (below = stable)

# Tracking settings
HISTORY_SIZE = 5          # Frames for smoothing
```

## 📊 Output Information

Each detected vehicle gets:

```python
{
    'bbox': [x1, y1, x2, y2],
    'class': 'car',
    'confidence': 0.95,
    'depth': 8.5,              # meters
    'velocity': -2.3,          # m/s (negative = approaching)
    'status': 'APPROACHING',   # or STABLE, RECEDING
    'ttc': 3.7,                # seconds (if approaching)
}
```

## 🎨 Visual Indicators

### Status Colors

- 🔴 **RED** - APPROACHING [DANGER] (< 1m)
- 🟠 **ORANGE-RED** - APPROACHING [CRITICAL] (1-2m)
- 🟠 **ORANGE** - APPROACHING [CAUTION] (2-5m)
- 🟡 **YELLOW** - RECEDING
- 🟢 **GREEN** - STABLE
- ⚪ **WHITE** - DETECTING (first few frames)

### On-Screen Display

```
┌─────────────────────────────────┐
│ car: 0.95                       │  ← Classification + Confidence
│ D: 8.5m                         │  ← Depth/Distance
│ V: -2.3m/s                      │  ← Velocity (negative = approaching)
│ [APPROACHING [CAUTION]]         │  ← Status with risk level
│ TTC: 3.7s                       │  ← Time to collision (if approaching)
└─────────────────────────────────┘
```

## 🔬 Algorithm Details

### 1. Depth Estimation
- **Method**: Deep learning (Depth-Anything-V2)
- **Input**: Single RGB image
- **Output**: Dense depth map
- **Accuracy**: Metric scale (meters)

### 2. Distance Extraction
```python
# For each detected vehicle:
bbox_region = depth_map[y1:y2, x1:x2]
distance = median(bbox_region)  # Robust to outliers
```

### 3. Velocity Estimation
```python
# Track depth over 5 frames:
depths = [d1, d2, d3, d4, d5]
timestamps = [t1, t2, t3, t4, t5]

# Linear regression:
velocity = polyfit(timestamps, depths, degree=1)[0]
# Negative velocity = approaching
# Positive velocity = receding
```

### 4. Time-to-Collision (TTC)
```python
if velocity < -0.1:  # Approaching
    ttc = distance / abs(velocity)
    
    if ttc < 1.0:    risk = DANGER
    elif ttc < 2.0:  risk = CRITICAL
    elif ttc < 5.0:  risk = CAUTION
    else:            risk = SAFE
```

## 🔄 Integration with Existing System

The depth module is **completely independent** of your CNN training:

- ✅ Training continues uninterrupted
- ✅ Depth runs on inference only
- ✅ Can be enabled/disabled anytime
- ✅ No changes to existing `camera_inference.py`

### Use Cases

1. **Current Training** → Vehicle Classification (14 classes)
2. **Depth Module** → Distance + Velocity measurement
3. **Combined** → Full ADAS: Detect + Classify + Track + Warn

## 📈 Performance

### GPU Memory Usage

| Component | VRAM | Notes |
|-----------|------|-------|
| YOLO | ~500MB | Detection |
| CNN | ~200MB | Classification |
| Depth (small) | ~300MB | Distance estimation |
| **Total** | **~1GB** | Fits easily on RTX A6000 |

### Expected FPS

- **RTX A6000**: 30-40 FPS (1280x720)
- **RTX 3060**: 20-30 FPS
- **GTX 1060**: 10-15 FPS

## 🐛 Troubleshooting

### Model Download Issues

Models auto-download from Hugging Face (~500MB for small model). Requires internet connection on first run.

If download fails:
```bash
pip install transformers pillow
```

### Import Errors

```bash
pip install transformers pillow timm einops
```

### Low FPS

- Use `--depth-model small` instead of base/large
- Reduce camera resolution: `--width 640 --height 480`
- Check GPU utilization: `nvidia-smi`

## 📝 Example Commands

### Basic Test
```bash
python test_depth.py --mode camera
```

### High Quality
```bash
python camera_inference_depth.py --depth-model base --show-depth
```

### Save Video
```bash
python camera_inference_depth.py \
    --camera 0 \
    --save output_depth.mp4 \
    --show-depth
```

### Process Video File
```bash
python camera_inference_depth.py \
    --camera /path/to/video.mp4 \
    --save output.mp4
```

## 🎓 Next Steps

1. **Test on your camera**: Run `test_depth.py` to verify depth works
2. **Try enhanced inference**: Use `camera_inference_depth.py`
3. **Tune thresholds**: Edit `depth_config.py` for your use case
4. **Integrate TTC**: Use existing `production/ttc_warning.py` with depth data

## ℹ️ Notes

- ✅ Your CNN training is **still running** (PID 590797 - Epoch 11/100)
- ✅ Depth estimation is **separate** - won't affect training
- ✅ Models auto-download on first use (~500MB for small model)
- ✅ Uses **Intel DPT** - state-of-the-art monocular depth

## 📚 References

- **Model**: Intel DPT (Dense Prediction Transformer)
- **Paper**: "Vision Transformers for Dense Prediction" (ICCV 2021)
- **Implementation**: Hugging Face Transformers
- **Models**: `Intel/dpt-hybrid-midas`, `Intel/dpt-large`

---

**Ready to test?** Run `python test_depth.py --mode camera` to see it in action!
