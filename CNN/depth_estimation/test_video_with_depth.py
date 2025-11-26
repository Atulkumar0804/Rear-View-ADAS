"""
Test Depth Estimation on Video with Full Warnings
Shows: Distance (m), Velocity (m/s), TTC (s), Risk Levels
"""

import sys
from pathlib import Path
import cv2
import numpy as np
import time

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from depth_estimation import DepthEstimator, VehicleDepthTracker, DepthConfig

def test_video_with_depth(video_path: str, model_size: str = 'small', save_output: bool = True):
    """Test depth estimation on video with full warning system"""
    
    print("\n" + "="*70)
    print("🎥 DEPTH ESTIMATION TEST - FULL WARNING SYSTEM")
    print("="*70)
    
    # Check video exists
    if not Path(video_path).exists():
        print(f"❌ Video not found: {video_path}")
        return
    
    print(f"📹 Video: {video_path}")
    
    # Initialize depth estimator
    print("\n🔧 Initializing Depth Estimator...")
    estimator = DepthEstimator(model_size=model_size, device='cuda')
    
    # Initialize tracker
    tracker = VehicleDepthTracker(estimator)
    
    print(f"✅ System ready!\n")
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Failed to open video")
        return
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"📊 Video Info:")
    print(f"   Resolution: {width}x{height}")
    print(f"   FPS: {fps:.2f}")
    print(f"   Total frames: {total_frames}")
    print(f"   Duration: {total_frames/fps:.1f}s")
    
    # Setup output video
    writer = None
    if save_output:
        output_path = Path(video_path).stem + "_depth_warnings.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        print(f"💾 Output: {output_path}\n")
    
    print("🚀 Processing video...")
    print("   Press 'q' to quit early\n")
    
    frame_count = 0
    start_time = time.time()
    
    # Simulated detections (we'll just use full frame for demo)
    # In real use, you'd get these from YOLO
    detections = [{
        'bbox': [width//4, height//4, 3*width//4, 3*height//4],
        'track_id': 1,
        'class': 'car',
        'confidence': 0.95
    }]
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("\n🎬 End of video")
                break
            
            frame_count += 1
            
            # Estimate depth
            depth_map = estimator.estimate_depth(frame)
            
            # Update tracking with depth
            timestamp = frame_count / fps
            enhanced_detections = tracker.update(depth_map, detections, timestamp)
            
            # Draw results
            annotated = draw_warnings(frame, depth_map, enhanced_detections, frame_count, fps)
            
            # Save
            if writer:
                writer.write(annotated)
            
            # Display
            cv2.imshow('Depth Estimation with Warnings - Press q to quit', annotated)
            
            # Progress
            if frame_count % 30 == 0:
                progress = (frame_count / total_frames) * 100
                elapsed = time.time() - start_time
                current_fps = frame_count / elapsed if elapsed > 0 else 0
                print(f"   Progress: {progress:.1f}% ({frame_count}/{total_frames}) - FPS: {current_fps:.1f}")
            
            # Check for quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\n⚠️  Stopped by user")
                break
    
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user")
    
    finally:
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()
        
        # Final stats
        elapsed = time.time() - start_time
        avg_fps = frame_count / elapsed if elapsed > 0 else 0
        
        print("\n" + "="*70)
        print("📊 PROCESSING COMPLETE")
        print("="*70)
        print(f"Frames processed: {frame_count}/{total_frames}")
        print(f"Time elapsed: {elapsed:.2f}s")
        print(f"Average FPS: {avg_fps:.2f}")
        if save_output:
            print(f"Output saved: {output_path}")
        print("="*70 + "\n")


def draw_warnings(frame, depth_map, detections, frame_num, fps):
    """Draw comprehensive warnings on frame"""
    
    annotated = frame.copy()
    h, w = frame.shape[:2]
    
    # Create depth visualization (small overlay)
    depth_colored = cv2.applyColorMap(
        cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8),
        cv2.COLORMAP_INFERNO
    )
    depth_small = cv2.resize(depth_colored, (320, 240))
    
    # Overlay depth map in corner
    x_offset = w - 330
    y_offset = 10
    annotated[y_offset:y_offset+240, x_offset:x_offset+320] = depth_small
    cv2.rectangle(annotated, (x_offset-2, y_offset-2), 
                 (x_offset+322, y_offset+242), (0, 255, 0), 2)
    cv2.putText(annotated, "Depth Map", (x_offset, y_offset-10),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Draw detection info
    for det in detections:
        x1, y1, x2, y2 = det['bbox']
        
        # Get depth info
        depth = det.get('depth', 0)
        velocity = det.get('velocity')
        status = det.get('status', 'DETECTING')
        status_color = det.get('status_color', (255, 255, 255))
        ttc = det.get('ttc')
        
        # Draw bounding box
        color = (0, 255, 0)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)
        
        # Title box
        title = f"{det['class'].upper()}: {det['confidence']:.2f}"
        (title_w, title_h), _ = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(annotated, (x1, y1 - title_h - 15), (x1 + title_w + 10, y1), color, -1)
        cv2.putText(annotated, title, (x1 + 5, y1 - 8),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        
        # Info panel (below bbox)
        info_y = y2 + 25
        line_height = 25
        
        # Distance
        if depth > 0:
            dist_text = f"Distance: {depth:.1f}m"
            cv2.putText(annotated, dist_text, (x1, info_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            info_y += line_height
        
        # Velocity
        if velocity is not None:
            vel_text = f"Velocity: {velocity:.2f} m/s"
            vel_color = (0, 0, 255) if velocity < 0 else (0, 255, 255)
            cv2.putText(annotated, vel_text, (x1, info_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, vel_color, 2)
            info_y += line_height
        
        # Status with color
        cv2.putText(annotated, f"[{status}]", (x1, info_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
        info_y += line_height
        
        # TTC warning
        if ttc is not None and ttc < 10:
            ttc_text = f"TTC: {ttc:.1f}s"
            ttc_color = (0, 0, 255) if ttc < 2 else (0, 165, 255)
            cv2.putText(annotated, ttc_text, (x1, info_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, ttc_color, 2)
            
            # Big warning if critical
            if ttc < 2:
                warn_text = "⚠️ COLLISION WARNING ⚠️"
                (warn_w, warn_h), _ = cv2.getTextSize(warn_text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 3)
                warn_x = (w - warn_w) // 2
                warn_y = 50
                cv2.rectangle(annotated, (warn_x - 10, warn_y - warn_h - 10),
                             (warn_x + warn_w + 10, warn_y + 10), (0, 0, 255), -1)
                cv2.putText(annotated, warn_text, (warn_x, warn_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)
    
    # Frame info (top-left)
    cv2.putText(annotated, f"Frame: {frame_num}", (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(annotated, f"FPS: {fps:.1f}", (10, 60),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Feature indicators (top-left, below FPS)
    features = [
        "✅ Absolute Distance",
        "✅ Velocity (m/s)",
        "✅ TTC Prediction",
        "✅ Risk Warnings"
    ]
    
    for i, feature in enumerate(features):
        cv2.putText(annotated, feature, (10, 100 + i*25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    return annotated


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Test Depth Estimation on Video')
    parser.add_argument('--video', type=str, required=True,
                       help='Path to video file')
    parser.add_argument('--model', type=str, default='small',
                       choices=['small', 'base', 'large'],
                       help='Depth model size')
    parser.add_argument('--no-save', action='store_true',
                       help='Do not save output video')
    
    args = parser.parse_args()
    
    # Print config
    DepthConfig.print_config()
    
    # Run test
    test_video_with_depth(args.video, model_size=args.model, save_output=not args.no_save)
