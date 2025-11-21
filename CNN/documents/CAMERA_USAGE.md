# 📷 Real-Time Camera Vehicle Detection

Use your webcam or external camera to detect and classify vehicles in real-time using trained CNN models.

## 🚀 Quick Start (Easiest Method)

```bash
cd CNN
./run_camera.sh
```

This will automatically:
- Find and use the best trained model
- Open your default camera (camera 0)
- Start real-time detection

## 📋 Manual Usage

### Basic Usage (Default Camera)

```bash
cd CNN
python camera_inference.py
```

### Using Specific Model

```bash
# Use MobileNet (fastest)
python camera_inference.py --model checkpoints/mobilenet_inspired/best_model.pth

# Use ResNet (best accuracy)
python camera_inference.py --model checkpoints/transfer_resnet18/best_model.pth
```

### Using External Camera

```bash
# Camera ID 1 (usually external USB camera)
python camera_inference.py --camera 1

# Camera ID 2
python camera_inference.py --camera 2
```

### Custom Resolution

```bash
# Lower resolution for faster FPS
python camera_inference.py --width 640 --height 480

# HD resolution
python camera_inference.py --width 1920 --height 1080
```

### Save Video Output

```bash
# Record detection session
python camera_inference.py --save output_recording.mp4
```

### Combined Options

```bash
python camera_inference.py \
    --model checkpoints/transfer_resnet18/best_model.pth \
    --camera 0 \
    --width 1280 \
    --height 720 \
    --save session_recording.mp4
```

## ⌨️ Keyboard Controls

While detection is running:

- **`q`** - Quit the application
- **`s`** - Take screenshot (saves as `screenshot_XXX.jpg`)

## 🎯 What You'll See

The system will display:

1. **Bounding Boxes** - Colored boxes around detected vehicles:
   - 🟢 Green = Car
   - 🟠 Orange = Truck
   - 🔵 Blue = Bus
   - 🟣 Magenta = Person

2. **Labels** - Class name and confidence score above each box

3. **Distance Status** - Below each box:
   - 🔴 **APPROACHING** - Vehicle getting closer (warning!)
   - 🟡 **RECEDING** - Vehicle moving away
   - 🟢 **STABLE** - Vehicle maintaining distance
   - ⚪ **NEW** - Just detected

4. **Stats** - Top-left corner:
   - FPS (frames per second)
   - Vehicle count

## 📊 Performance Tips

### For Better FPS:
- Use lower resolution: `--width 640 --height 480`
- Use lighter model: `--model checkpoints/mobilenet_inspired/best_model.pth`
- Use CPU if GPU is busy: `--device cpu`

### For Better Accuracy:
- Use best model: `--model checkpoints/transfer_resnet18/best_model.pth`
- Use higher resolution: `--width 1280 --height 720`
- Ensure good lighting
- Keep camera stable

## 🔧 Troubleshooting

### Camera Not Opening

```bash
# List available cameras
ls -lh /dev/video*

# Try different camera IDs
python camera_inference.py --camera 0
python camera_inference.py --camera 1
python camera_inference.py --camera 2
```

### Low FPS

```bash
# Use lighter model
python camera_inference.py --model checkpoints/mobilenet_inspired/best_model.pth

# Reduce resolution
python camera_inference.py --width 640 --height 480

# Use CPU (if GPU has issues)
python camera_inference.py --device cpu
```

### Model Not Found

```bash
# Check available models
ls -lh checkpoints/*/best_model.pth

# Train models if needed
python train_v2.py
```

### YOLO Model Not Found

```bash
# Check YOLO model exists
ls -lh ../models/yolo/yolov8n_RearView.pt

# If missing, the YOLO model path is hardcoded in camera_inference.py
# Make sure it's at: ../models/yolo/yolov8n_RearView.pt
```

## 📝 Example Output

```
============================================================
🚗 REAL-TIME CAMERA VEHICLE DETECTION
============================================================

🔥 Device: cuda
📦 Loading YOLO...
✅ YOLO loaded
📦 Loading CNN: checkpoints/transfer_resnet18/best_model.pth
   Model: transfer_resnet18
✅ CNN loaded
   Classes: ['car', 'truck', 'bus', 'person']

📷 Opening camera 0...
✅ Camera opened: 1280x720
💾 Saving to: output.mp4

🚀 Starting detection...
   Press 'q' to quit
   Press 's' to take screenshot

📸 Screenshot saved: screenshot_000.jpg
📸 Screenshot saved: screenshot_001.jpg

⚠️  Interrupted by user

============================================================
📊 SESSION STATISTICS
============================================================
Frames processed: 1247
Time elapsed: 52.34s
Average FPS: 23.82
Screenshots taken: 2
============================================================
```

## 🎥 Example Scenarios

### Parking Lot Monitoring
```bash
python camera_inference.py --save parking_monitoring.mp4
```

### Rear-View Camera Simulation
```bash
# Mount camera on back of vehicle
python camera_inference.py --width 1920 --height 1080
```

### Traffic Analysis
```bash
# Use external camera pointed at road
python camera_inference.py --camera 1 --save traffic_analysis.mp4
```

## 🔬 Technical Details

### Detection Pipeline:
1. **Frame Capture** - Read frame from camera
2. **YOLO Detection** - Fast object detection
3. **CNN Classification** - Refine vehicle type
4. **Tracking** - Match vehicles across frames using IoU
5. **Distance Estimation** - Compute area changes
6. **Visualization** - Draw boxes and labels

### Models Used:
- **YOLO** - Initial detection (fast, ~30 FPS)
- **CNN** - Classification refinement (accurate)
- **Tracking** - IoU-based matching

### Performance:
- **Best Model**: transfer_resnet18 (98.31% val accuracy)
- **Fastest Model**: mobilenet_inspired (~30 FPS)
- **Smallest Model**: squeezenet_inspired (4.6 MB)

## 📚 Related Files

- `camera_inference.py` - Main camera detection script
- `run_camera.sh` - Quick start shell script
- `inference_v2.py` - Full inference script (video/camera)
- `models/architectures.py` - CNN model definitions
- `checkpoints/` - Trained model weights

## 🆘 Need Help?

Check the main documentation:
```bash
cat README_V2.md
cat NEXT_STEPS.md
```

Or run with default settings:
```bash
./run_camera.sh
```

---

**Happy Detecting! 🚗📷**
