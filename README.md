# Rear-View Monocular ADAS (Prototype)

A prototype rear-facing monocular ADAS pipeline for two-wheelers.  
This repo contains a working prototype and a modular project skeleton to detect & classify vehicles behind a scooter, estimate relative distance & relative speed, predict short-term trajectory (t+5s), and raise collision/proximity warnings.

> This project is intended for research / prototyping. Do **not** use it as a safety-critical product without extensive validation and hardware-in-the-loop testing.

---

## Features
- YOLO-based detection (if you provide a YOLO model and install `ultralytics`) or fallback motion-based detector for quick testing.
- Simple centroid tracker for per-object identity (placeholder for ByteTrack/DeepSORT).
- Two monocular depth estimators:
  - ground-projection from camera intrinsics & mounting height
  - bbox-height (class priors)
- Per-track Kalman smoothing (uses `filterpy` if available, otherwise a simple builtin).
- Constant-velocity trajectory prediction up to 5 seconds.
- TTC and multi-level warning logic (NONE / WARN / CRITICAL).
- Modular layout intended for easy replacement of detection/tracking modules.

---

## Quickstart

### 1. Create & activate virtual environment (recommended)
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate
