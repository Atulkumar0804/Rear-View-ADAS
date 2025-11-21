# Rear-View Monocular ADAS (Prototype)

A prototype rear-facing monocular ADAS pipeline for two-wheelers.  
This repo contains a working prototype and a modular project skeleton to detect & classify vehicles behind a scooter, estimate relative distance & relative speed, predict short-term trajectory (t+5s), and raise collision/proximity warnings.

> This project is intended for research / prototyping. Do **not** use it as a safety-critical product without extensive validation and hardware-in-the-loop testing.

---

## Features
- **Multi-class detection**: Vehicles (car, truck, bus, motorcycle, bicycle, train), pedestrians, animals (dog, cat, horse, cow, etc.), traffic infrastructure (traffic lights, stop signs, fire hydrants, parking meters)
- **Custom class training**: Built-in workflow to train detection for potholes, speed breakers, road signs, and other hazards (see `TRAINING.md`)
- YOLO-based detection (if you provide a YOLO model and install `ultralytics`) or fallback motion-based detector for quick testing
- Simple centroid tracker for per-object identity (placeholder for ByteTrack/DeepSORT)
- Two monocular depth estimators:
  - ground-projection from camera intrinsics & mounting height
  - bbox-height (class priors)
- Per-track Kalman smoothing (uses `filterpy` if available, otherwise a simple builtin)
- Constant-velocity trajectory prediction up to 5 seconds
- TTC and multi-level warning logic (NONE / WARN / CRITICAL)
- **Intel RealSense D400 series support** with configurable resolution/FPS
- Modular layout intended for easy replacement of detection/tracking modules
- Optional segmentation-style overlay that fills vehicles, pedestrians,
  traffic lights, and detected lane lines with class-specific colors for
  quick scene understanding

---

## Detection Classes

### Currently Supported (Pretrained COCO)
- **Vehicles**: car, truck, bus, motorcycle, bicycle, train
- **Pedestrians**: person
- **Animals**: dog, cat, horse, cow, sheep, bird, elephant, bear, zebra, giraffe
- **Infrastructure**: traffic_light, stop_sign, fire_hydrant, parking_meter

### Train Your Own (Custom Classes)
To detect **potholes, speed breakers, road signs, pedestrian crossings**, and other custom hazards:
1. **Quick Start**: See `TRAINING_QUICKSTART.md`
2. **Full Guide**: See `TRAINING.md` for detailed instructions
3. **Tools**: Use `collect_frames.py` to gather training data from your camera
4. **Training**: Run `train_custom_model.py` with your labeled dataset

---

## Quickstart

### 1. Create & activate virtual environment (recommended)
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate
