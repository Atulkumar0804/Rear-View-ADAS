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
from src.pipeline import RearADASPipeline

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
    p.add_argument("--realsense", action="store_true",
                   help="Use Intel RealSense camera instead of OpenCV capture.")
    p.add_argument("--rs-width", type=int, default=1280, help="RealSense color width")
    p.add_argument("--rs-height", type=int, default=720, help="RealSense color height")
    p.add_argument("--rs-fps", type=int, default=30, help="RealSense stream FPS")
    p.add_argument("--rs-serial", type=str, default=None, help="Optional RealSense device serial")
    p.add_argument("--rs-use-depth", action="store_true", help="Enable depth stream (alignment only)")
    p.add_argument("--rs-timeout", type=int, default=10000, help="RealSense frame wait timeout (ms)")
    p.add_argument("--rs-color-only", action="store_true", help="Disable depth stream even if rs-use-depth is set")
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

    rs_cfg = None
    if args.realsense:
        rs_cfg = {
            "width": args.rs_width,
            "height": args.rs_height,
            "fps": args.rs_fps,
            "serial": args.rs_serial,
            "use_depth": args.rs_use_depth,
            "timeout_ms": args.rs_timeout,
            "color_only": args.rs_color_only,
        }

    print("Starting pipeline. Press 'q' to quit.")
    pipeline.run(video_source=src, use_realsense=args.realsense, realsense_cfg=rs_cfg)

if __name__ == "__main__":
    main()
