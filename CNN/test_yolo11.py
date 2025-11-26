#!/usr/bin/env python3
"""
Quick test script to verify YOLOv11 integration
Tests detection on a sample image or video frame
"""

import cv2
import sys
from pathlib import Path
from ultralytics import YOLO

# Get YOLOv11 model path
PROJECT_ROOT = Path(__file__).parent.parent
YOLO_MODEL_PATH = str(PROJECT_ROOT / "yolo11s.pt")

def test_yolo11_image(image_path):
    """Test YOLOv11 on a single image"""
    print(f"\n{'='*60}")
    print("🧪 Testing YOLOv11 on Image")
    print(f"{'='*60}\n")
    
    # Check if image exists
    if not Path(image_path).exists():
        print(f"❌ Image not found: {image_path}")
        return False
    
    # Load YOLO
    print(f"📦 Loading YOLOv11 from: {YOLO_MODEL_PATH}")
    model = YOLO(YOLO_MODEL_PATH)
    print("✅ YOLOv11 loaded successfully!\n")
    
    # Run inference
    print(f"🔍 Running detection on: {image_path}")
    results = model(image_path, verbose=True)
    
    # Process results
    for r in results:
        print(f"\n📊 Detection Results:")
        print(f"   Detections: {len(r.boxes)}")
        
        if len(r.boxes) > 0:
            print(f"\n   Details:")
            for i, box in enumerate(r.boxes, 1):
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                cls_name = model.names[cls_id]
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                print(f"      {i}. {cls_name} ({conf:.2f}) at [{x1},{y1},{x2},{y2}]")
        
        # Save annotated image
        annotated = r.plot()
        output_path = "yolo11_test_output.jpg"
        cv2.imwrite(output_path, annotated)
        print(f"\n💾 Saved annotated image to: {output_path}")
    
    print(f"\n{'='*60}")
    print("✅ YOLOv11 Test Complete!")
    print(f"{'='*60}\n")
    
    return True


def test_yolo11_video_frame():
    """Test YOLOv11 on first frame of test video"""
    print(f"\n{'='*60}")
    print("🧪 Testing YOLOv11 on Video Frame")
    print(f"{'='*60}\n")
    
    # Look for test video
    video_paths = [
        "dataset/test_viedos/test1_clean.mp4",
        "dataset/test_viedos/test1.mp4",
    ]
    
    video_path = None
    for path in video_paths:
        if Path(path).exists():
            video_path = path
            break
    
    if not video_path:
        print("❌ No test video found. Please provide an image path.")
        return False
    
    # Extract first frame
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print(f"❌ Failed to read frame from: {video_path}")
        return False
    
    # Save temp frame
    temp_frame = "temp_test_frame.jpg"
    cv2.imwrite(temp_frame, frame)
    print(f"📹 Extracted frame from: {video_path}")
    
    # Test on frame
    result = test_yolo11_image(temp_frame)
    
    # Cleanup
    Path(temp_frame).unlink(missing_ok=True)
    
    return result


def main():
    if len(sys.argv) > 1:
        # Test on provided image
        image_path = sys.argv[1]
        test_yolo11_image(image_path)
    else:
        # Test on video frame
        test_yolo11_video_frame()


if __name__ == "__main__":
    main()
