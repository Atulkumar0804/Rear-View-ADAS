#!/usr/bin/env python3
"""
Live camera demo for Rear-View ADAS
Works with laptop webcam or external camera
Press 'q' to quit, 's' to save snapshot
"""

import cv2
import time
import sys
from src.pipeline import RearADASPipeline

def run_live_camera(camera_index=0):
    print("=" * 70)
    print("REAR-VIEW ADAS - LIVE CAMERA MODE")
    print("=" * 70)
    
    # Initialize pipeline
    print("\nInitializing ADAS pipeline...")
    pipeline = RearADASPipeline(show_fps=True)
    print("✅ Pipeline ready!")
    
    # Open camera
    print(f"\nOpening camera {camera_index}...")
    cap = cv2.VideoCapture(camera_index)
    
    if not cap.isOpened():
        print(f"❌ Cannot open camera {camera_index}")
        print("\nTrying alternative camera indices...")
        for idx in [1, 2, -1]:
            cap = cv2.VideoCapture(idx)
            if cap.isOpened():
                print(f"✅ Camera {idx} opened successfully!")
                camera_index = idx
                break
        else:
            print("❌ No camera found. Exiting.")
            return
    
    # Set camera properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"\n📹 Camera Info:")
    print(f"   Index: {camera_index}")
    print(f"   Resolution: {actual_width}x{actual_height}")
    print(f"   FPS: {actual_fps:.1f}")
    
    print("\n" + "=" * 70)
    print("CONTROLS:")
    print("  Press 'q' or ESC to quit")
    print("  Press 's' to save snapshot")
    print("  Press 'p' to pause/resume")
    print("=" * 70)
    print("\n🎥 Starting live detection...\n")
    
    frame_count = 0
    snapshot_count = 0
    paused = False
    start_time = time.time()
    fps_time = time.time()
    fps_counter = 0
    current_fps = 0
    
    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("❌ Failed to grab frame")
                break
            
            frame_count += 1
            fps_counter += 1
            
            # Calculate FPS every second
            if time.time() - fps_time >= 1.0:
                current_fps = fps_counter / (time.time() - fps_time)
                fps_counter = 0
                fps_time = time.time()
            
            # Detect vehicles
            detections = pipeline.detector.detect(frame)
            
            # Track objects
            tracks = pipeline.tracker.update(detections)
            
            # Process each track
            for tr in tracks:
                tid = tr.track_id
                bbox = tr.bbox
                cls = tr.cls
                
                # Ground projection and depth
                X, Z_proj = pipeline.projector.bottom_to_ground(bbox)
                Z_bbox = pipeline.depther.bbox_depth(bbox, cls)
                Z = 0.6 * Z_proj + 0.4 * Z_bbox
                
                # Kalman smoothing
                x, z, vx, vz = pipeline.kf.predict_and_update(tid, X, Z)
                
                # Trajectory prediction
                preds = pipeline.predictor.predict(x, z, vx, vz)
                
                # Warning level
                level, ttc = pipeline.warnsys.decide(x, z, vx, vz, preds)
                
                # Draw annotations
                x1, y1, x2, y2 = bbox
                
                # Color based on warning level
                if level == "CRITICAL":
                    color = (0, 0, 255)  # Red
                    thickness = 3
                elif level == "WARN":
                    color = (0, 165, 255)  # Orange
                    thickness = 2
                else:
                    color = (0, 255, 0)  # Green
                    thickness = 2
                
                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
                
                # Draw label
                label = f"ID:{tid} {cls}"
                cv2.putText(frame, label, (x1, max(y1-10, 15)),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
                # Draw depth and velocity
                ttc_str = f"{ttc:.1f}s" if ttc < 999 else "inf"
                info1 = f"Z:{z:.1f}m  v:{vz:+.1f}m/s"
                info2 = f"{level} TTC:{ttc_str}"
                
                cv2.putText(frame, info1, (x1, min(y2+22, actual_height-25)),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                cv2.putText(frame, info2, (x1, min(y2+42, actual_height-5)),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Display info overlay
            overlay_y = 25
            cv2.putText(frame, f"FPS: {current_fps:.1f}", (10, overlay_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            overlay_y += 30
            cv2.putText(frame, f"Tracks: {len(tracks)}", (10, overlay_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            overlay_y += 30
            cv2.putText(frame, f"Frame: {frame_count}", (10, overlay_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
            # Show warning legend
            legend_y = actual_height - 60
            cv2.putText(frame, "Legend:", (10, legend_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            cv2.rectangle(frame, (70, legend_y-10), (90, legend_y+2), (0, 255, 0), -1)
            cv2.putText(frame, "Safe", (95, legend_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            
            cv2.rectangle(frame, (70, legend_y+8), (90, legend_y+20), (0, 165, 255), -1)
            cv2.putText(frame, "Warn", (95, legend_y+18),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 165, 255), 1)
            
            cv2.rectangle(frame, (70, legend_y+26), (90, legend_y+38), (0, 0, 255), -1)
            cv2.putText(frame, "Critical", (95, legend_y+36),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
            
            # Title
            cv2.putText(frame, "Rear-View ADAS - Live Camera", 
                       (actual_width - 350, 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Show frame
        cv2.imshow('Rear-View ADAS - Live Camera', frame)
        
        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q') or key == 27:  # q or ESC
            print("\n🛑 Stopping...")
            break
        elif key == ord('s'):  # Save snapshot
            filename = f"snapshot_{snapshot_count:03d}.jpg"
            cv2.imwrite(filename, frame)
            snapshot_count += 1
            print(f"📸 Saved: {filename}")
        elif key == ord('p'):  # Pause/resume
            paused = not paused
            status = "⏸️  PAUSED" if paused else "▶️  RESUMED"
            print(f"{status}")
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    
    elapsed = time.time() - start_time
    avg_fps = frame_count / elapsed if elapsed > 0 else 0
    
    print("\n" + "=" * 70)
    print("SESSION SUMMARY")
    print("=" * 70)
    print(f"Frames processed: {frame_count}")
    print(f"Time elapsed: {elapsed:.1f}s")
    print(f"Average FPS: {avg_fps:.1f}")
    print(f"Snapshots saved: {snapshot_count}")
    print("=" * 70)

if __name__ == "__main__":
    camera_idx = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    
    try:
        run_live_camera(camera_idx)
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
