#!/usr/bin/env python3
"""
Create videos from frame folders and test with depth estimation.
Converts frames from CAM_BACK folders 1-9 into videos and runs ADAS testing.
"""

import cv2
import os
import sys
import glob
import numpy as np
from pathlib import Path
from typing import List, Tuple

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


def get_frame_size(image_path: str) -> Tuple[int, int]:
    """Get dimensions of the first frame."""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Cannot read image: {image_path}")
    height, width = img.shape[:2]
    return width, height


def create_video_from_frames(
    frame_folder: str,
    output_path: str,
    fps: int = 30,
    codec: str = 'mp4v'
) -> bool:
    """
    Create video from frames in a folder.
    
    Args:
        frame_folder: Path to folder containing frames
        output_path: Output video path
        fps: Frames per second for output video
        codec: Video codec (mp4v, avc1, etc.)
    
    Returns:
        True if successful, False otherwise
    """
    print(f"\n📁 Processing folder: {frame_folder}")
    
    # Get all image files sorted by name
    frame_files = sorted(glob.glob(os.path.join(frame_folder, "*.jpg")))
    if not frame_files:
        frame_files = sorted(glob.glob(os.path.join(frame_folder, "*.png")))
    
    if not frame_files:
        print(f"   ❌ No frames found in {frame_folder}")
        return False
    
    num_frames = len(frame_files)
    print(f"   📊 Found {num_frames} frames")
    
    # Get frame dimensions from first frame
    try:
        width, height = get_frame_size(frame_files[0])
        print(f"   📐 Frame size: {width}x{height}")
    except Exception as e:
        print(f"   ❌ Error reading frame: {e}")
        return False
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*codec)
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    if not out.isOpened():
        print(f"   ❌ Failed to create video writer")
        return False
    
    # Write frames to video
    print(f"   🎬 Creating video: {os.path.basename(output_path)}")
    for i, frame_path in enumerate(frame_files):
        frame = cv2.imread(frame_path)
        if frame is None:
            print(f"   ⚠️  Warning: Could not read frame {i+1}/{num_frames}")
            continue
        
        # Ensure frame has correct dimensions
        if frame.shape[1] != width or frame.shape[0] != height:
            frame = cv2.resize(frame, (width, height))
        
        out.write(frame)
        
        # Progress indicator
        if (i + 1) % 10 == 0 or (i + 1) == num_frames:
            print(f"   Progress: {i+1}/{num_frames} frames", end='\r')
    
    print(f"\n   ✅ Video created: {output_path}")
    print(f"   Duration: {num_frames/fps:.2f}s @ {fps} FPS")
    
    out.release()
    return True


def create_all_videos(
    cam_back_folder: str,
    output_folder: str,
    fps: int = 30
) -> List[str]:
    """
    Create videos from all numbered folders (1-9).
    
    Args:
        cam_back_folder: Path to CAM_BACK folder containing 1-9 subfolders
        output_folder: Output folder for videos
        fps: Frames per second
    
    Returns:
        List of created video paths
    """
    print("=" * 60)
    print("🎥 VIDEO CREATION FROM FRAMES")
    print("=" * 60)
    
    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)
    
    created_videos = []
    
    # Process folders 1-9
    for folder_num in range(1, 10):
        frame_folder = os.path.join(cam_back_folder, str(folder_num))
        
        if not os.path.exists(frame_folder):
            print(f"\n⚠️  Folder {folder_num} not found, skipping...")
            continue
        
        # Output video name
        output_video = os.path.join(output_folder, f"cam_back_{folder_num}.mp4")
        
        # Create video
        success = create_video_from_frames(frame_folder, output_video, fps=fps)
        
        if success:
            created_videos.append(output_video)
            # Get file size
            size_mb = os.path.getsize(output_video) / (1024 * 1024)
            print(f"   💾 File size: {size_mb:.2f} MB\n")
    
    print("=" * 60)
    print(f"✅ CREATED {len(created_videos)} VIDEOS")
    print("=" * 60)
    
    return created_videos


def test_video_with_adas(
    video_path: str,
    yolo_model: str,
    depth_model: str = 'small',
    use_finetuned: bool = True
) -> str:
    """
    Test a video with the full ADAS system.
    
    Args:
        video_path: Path to input video
        yolo_model: Path to YOLO model
        depth_model: Depth model size (small/base/large)
        use_finetuned: If True, use fine-tuned depth model
    
    Returns:
        Path to output video
    """
    from test_full_adas import VehicleDetectorWithDepth, draw_full_annotations
    import time
    
    print(f"\n{'='*60}")
    print(f"🚗 TESTING: {os.path.basename(video_path)}")
    print(f"{'='*60}")
    
    # Initialize detector
    print("\n🔧 Initializing Detection System...")
    detector = VehicleDetectorWithDepth(yolo_model, depth_model, use_finetuned=use_finetuned)
    print("✅ System ready!\n")
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Cannot open video: {video_path}")
        return None
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0
    
    print(f"📊 Video Info:")
    print(f"   Resolution: {width}x{height}")
    print(f"   FPS: {fps:.2f}")
    print(f"   Total frames: {total_frames}")
    print(f"   Duration: {duration:.1f}s\n")
    
    # Output video path
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    output_path = os.path.join(os.path.dirname(video_path), f"{video_name}_adas.mp4")
    print(f"💾 Output: {output_path}\n")
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    if not out.isOpened():
        print("❌ Failed to create video writer")
        cap.release()
        return None
    
    print("🚀 Processing video...")
    
    frame_count = 0
    start_time = time.time()
    last_depth_map = None
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        timestamp = frame_count / fps if fps > 0 else frame_count * 0.033
        
        # Process frame with ADAS
        detections, depth_map = detector.process_frame(frame, timestamp)
        
        # Draw annotations
        annotated_frame = draw_full_annotations(
            frame.copy(),
            depth_map,
            detections,
            frame_count,
            fps
        )
        
        # Write frame
        out.write(annotated_frame)
        
        # Progress indicator
        if frame_count % 10 == 0 or frame_count == total_frames:
            elapsed = time.time() - start_time
            current_fps = frame_count / elapsed if elapsed > 0 else 0
            progress = (frame_count / total_frames) * 100
            num_vehicles = len(detections)
            print(f"   Progress: {progress:.1f}% ({frame_count}/{total_frames}) - "
                  f"Vehicles: {num_vehicles} - FPS: {current_fps:.1f}", end='\r')
    
    print("\n\n🎬 End of video")
    
    # Cleanup
    cap.release()
    out.release()
    
    elapsed = time.time() - start_time
    avg_fps = frame_count / elapsed if elapsed > 0 else 0
    
    print(f"\n📊 PROCESSING COMPLETE")
    print(f"   Frames processed: {frame_count}/{total_frames}")
    print(f"   Time elapsed: {elapsed:.2f}s")
    print(f"   Average FPS: {avg_fps:.2f}")
    
    # Get output file size
    if os.path.exists(output_path):
        size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(f"   Output saved: {output_path}")
        print(f"   File size: {size_mb:.2f} MB")
        return output_path
    else:
        print(f"   ❌ Failed to save output video")
        return None


def main():
    """Main function to create videos and test with ADAS."""
    
    # Paths
    cam_back_folder = "/home/atul/Desktop/atul/rear_view_adas_monocular/extra/data/samples/CAM_BACK"
    output_folder = "/home/atul/Desktop/atul/rear_view_adas_monocular/CNN/dataset/test_viedos"
    yolo_model = "/home/atul/Desktop/atul/rear_view_adas_monocular/CNN/models/yolo/yolov8n_RearView.pt"
    
    # Check if folders exist
    if not os.path.exists(cam_back_folder):
        print(f"❌ CAM_BACK folder not found: {cam_back_folder}")
        return
    
    if not os.path.exists(yolo_model):
        print(f"❌ YOLO model not found: {yolo_model}")
        return
    
    # Step 1: Create videos from frames
    print("\n" + "="*60)
    print("STEP 1: CREATE VIDEOS FROM FRAMES")
    print("="*60)
    
    created_videos = create_all_videos(
        cam_back_folder=cam_back_folder,
        output_folder=output_folder,
        fps=30
    )
    
    if not created_videos:
        print("\n❌ No videos were created")
        return
    
    print(f"\n✅ Created videos:")
    for video in created_videos:
        print(f"   - {video}")
    
    # Step 2: Test all videos with ADAS
    print("\n\n" + "="*60)
    print("STEP 2: TEST VIDEOS WITH ADAS")
    print("="*60)
    
    tested_videos = []
    
    for video_path in created_videos:
        output_path = test_video_with_adas(
            video_path=video_path,
            yolo_model=yolo_model,
            depth_model='small'
        )
        
        if output_path:
            tested_videos.append(output_path)
    
    # Final summary
    print("\n\n" + "="*60)
    print("🎉 ALL PROCESSING COMPLETE")
    print("="*60)
    
    print(f"\n📹 Created Videos: {len(created_videos)}")
    for video in created_videos:
        size_mb = os.path.getsize(video) / (1024 * 1024)
        print(f"   ✓ {os.path.basename(video)} ({size_mb:.2f} MB)")
    
    print(f"\n🚗 ADAS Tested Videos: {len(tested_videos)}")
    for video in tested_videos:
        size_mb = os.path.getsize(video) / (1024 * 1024)
        print(f"   ✓ {os.path.basename(video)} ({size_mb:.2f} MB)")
    
    print(f"\n💾 All videos saved to: {output_folder}")
    print("="*60)


if __name__ == "__main__":
    main()
