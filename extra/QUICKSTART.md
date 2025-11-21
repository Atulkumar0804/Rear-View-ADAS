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

### 4. Intel RealSense Camera
```bash
python3 -m src.main --realsense --rs-width 1280 --rs-height 720 --rs-fps 30
# or the feature-rich live view with snapshots
python3 live_camera.py --realsense
```
- Requires `pyrealsense2` (already listed in `requirements.txt`).
- Optional flags: `--rs-serial`, `--rs-use-depth` for depth-aligned streaming.
- If frames time out, either lower resolution/FPS or increase `--rs-timeout 8000`.
- Pass `--rs-color-only` when you want to disable the depth stream entirely.

### 5. Use Your Mobile Phone Camera
You have two simple choices:

**Option A – IP/RTSP stream (works on any OS)**
1. Install an IP camera app (e.g., *IP Webcam* / *DroidCam OBS* on Android, *EpocCam* / *iCam* on iOS).
2. Start the stream and note the RTSP/HTTP URL it displays, e.g. `rtsp://192.168.1.20:8554/live`.
3. Run the pipeline pointing to that URL:
	```bash
	python3 -m src.main --video rtsp://192.168.1.20:8554/live --show-fps
	```
4. Make sure the phone and the ADAS laptop are on the same Wi‑Fi network. Lower the phone’s resolution/FPS inside the app if the stream lags.

**Option B – Virtual webcam (USB/Wi‑Fi)**
1. Install a virtual webcam bridge (e.g., *DroidCam* / *Iriun* / *Camo*).
2. Connect the phone via USB or Wi‑Fi; the app exposes it as a new webcam index (often `/dev/video2`).
3. Launch the live viewer with that index:
	```bash
	python3 live_camera.py --camera 2
	```

For best depth estimates, update `config/camera_config.yaml` with your phone camera’s focal length / mounting height if it differs significantly from the default action camera setup.

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

### Too many noisy detections
- Increase `min_bbox_area` / `motion_min_area` in `config/model_config.yaml`
- Tighten `min_aspect_ratio` and `max_aspect_ratio`
- Prefer the YOLO detector (install PyTorch + ultralytics) for best precision
