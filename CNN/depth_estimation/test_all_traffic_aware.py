#!/usr/bin/env python3
"""
Test all videos with traffic-aware collision detection.
This version properly handles traffic scenarios where both vehicles are stopped/moving slowly.
"""

import os
import subprocess
import sys
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
VENV_PYTHON = PROJECT_ROOT / ".venv" / "bin" / "python"
TEST_SCRIPT = PROJECT_ROOT / "CNN" / "depth_estimation" / "test_full_adas.py"
VIDEO_DIR = PROJECT_ROOT / "CNN" / "dataset" / "test_viedos"
YOLO_MODEL = PROJECT_ROOT / "CNN" / "models" / "yolo" / "yolov8n_RearView.pt"
OUTPUT_DIR = PROJECT_ROOT

def test_video(video_path, video_name):
    """Test a single video with traffic-aware ADAS."""
    print(f"\n{'='*70}")
    print(f"🎥 Processing: {video_name}")
    print(f"{'='*70}")
    
    cmd = [
        str(VENV_PYTHON),
        str(TEST_SCRIPT),
        "--video", str(video_path),
        "--yolo", str(YOLO_MODEL),
        "--model", "small"
    ]
    
    try:
        result = subprocess.run(cmd, cwd=OUTPUT_DIR, capture_output=False, text=True)
        
        if result.returncode == 0:
            print(f"✅ {video_name} processed successfully")
            return True
        else:
            print(f"❌ {video_name} failed with code {result.returncode}")
            return False
    except Exception as e:
        print(f"❌ Error processing {video_name}: {e}")
        return False

def main():
    """Process all test videos."""
    print("\n" + "="*70)
    print("🚗 TRAFFIC-AWARE ADAS TESTING")
    print("="*70)
    print("\nKey Improvements:")
    print("  ✓ Considers relative velocity (not just proximity)")
    print("  ✓ No false warnings in traffic jams")
    print("  ✓ Only warns when gap is actively closing")
    print("  ✓ Differentiates 'Traffic' vs 'Approaching' scenarios")
    print("="*70)
    
    # Find all test videos
    video_files = sorted(VIDEO_DIR.glob("cam_back_*.mp4"))
    video_files = [v for v in video_files if not v.name.endswith("_adas.mp4") and not v.name.endswith("_full_adas.mp4")]
    
    print(f"\n📹 Found {len(video_files)} videos to process\n")
    
    success_count = 0
    failed_count = 0
    
    for video_path in video_files:
        video_name = video_path.name
        
        if test_video(video_path, video_name):
            success_count += 1
        else:
            failed_count += 1
    
    # Summary
    print("\n" + "="*70)
    print("📊 PROCESSING SUMMARY")
    print("="*70)
    print(f"✅ Successful: {success_count}/{len(video_files)}")
    print(f"❌ Failed: {failed_count}/{len(video_files)}")
    
    # List output files
    output_files = sorted(OUTPUT_DIR.glob("cam_back_*_full_adas.mp4"))
    print(f"\n📁 Output files: {len(output_files)}")
    for f in output_files:
        size_mb = f.stat().st_size / (1024 * 1024)
        print(f"   {f.name} ({size_mb:.1f} MB)")
    
    print("\n" + "="*70)
    print("✨ All videos processed with TRAFFIC-AWARE collision detection!")
    print("="*70)

if __name__ == "__main__":
    main()
