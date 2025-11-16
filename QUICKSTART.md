# Quick Start Guide - Rear-View ADAS

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### 1. Live Camera Detection (Laptop Webcam)
```bash
python3 live_camera.py
```
- Uses your laptop's built-in camera
- Shows real-time vehicle detection with warnings
- Press 'q' to quit

### 2. Process Video File
```bash
python3 src/main.py --source data/samples/car-detection.mp4
```

### 3. Custom Video with Output
```bash
python3 src/main.py --source path/to/video.mp4 --output result.mp4
```

## Controls (Live Camera)
- **'q'** - Quit
- **ESC** - Exit

## Color Codes
- 🟢 **Green** - Safe (no collision risk)
- 🟡 **Orange** - Warning (potential collision)
- 🔴 **Red** - Critical (imminent collision)

## Display Information
- **ID** - Vehicle tracking ID
- **Z** - Distance from camera (meters)
- **v** - Relative velocity (m/s)
- **TTC** - Time to collision (seconds)
- **Level** - Warning level (NONE/WARN/CRITICAL)

## Troubleshooting

### Camera not working
```bash
# Check available cameras
ls /dev/video*

# Try different camera index
# Edit live_camera.py and change: cap = cv2.VideoCapture(1)
```

### Low FPS
- Lower resolution in camera_config.yaml
- Reduce confidence threshold in model_config.yaml
- Use smaller YOLO model

### No detections
- Check lighting conditions
- Adjust conf_thres in config/model_config.yaml
- Verify YOLO model is downloaded in models/yolo/
