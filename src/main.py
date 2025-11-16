"""
src/main.py
Entry point for the Rear-View Monocular ADAS prototype.

Usage:
    python src/main.py                # uses webcam (index 0)
    python src/main.py --video file.mp4
    python src/main.py --camera 1
    python src/main.py --model models/yolo/yolov8n_RearView.pt

This script is intentionally small — heavy logic lives in pipeline.py
"""

import argparse
import sys
from pipeline import RearADASPipeline

def parse_args():
    p = argparse.ArgumentParser(description="Rear-view monocular ADAS demo")
    p.add_argument("--video", "-v", type=str, default=None,
                   help="Path to video file. If omitted, webcam used.")
    p.add_argument("--camera", "-c", type=int, default=0,
                   help="Camera index for webcam (default 0).")
    p.add_argument("--model", "-m", type=str, default=None,
                   help="Optional path to YOLO model (overrides config).")
    p.add_argument("--imgsz", type=int, default=None,
                   help="Optional YOLO image size override.")
    p.add_argument("--show-fps", action="store_true", help="Show FPS overlay")
    return p.parse_args()

def main():
    args = parse_args()

    try:
        pipeline = RearADASPipeline(model_path=args.model, img_size=args.imgsz, show_fps=args.show_fps)
    except Exception as e:
        print("Failed to initialize pipeline:", e)
        raise

    if args.video:
        src = args.video
    else:
        src = args.camera

    print("Starting pipeline. Press 'q' to quit.")
    pipeline.run(video_source=src)

if __name__ == "__main__":
    main()
