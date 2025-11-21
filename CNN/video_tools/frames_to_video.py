#!/usr/bin/env python3
"""
Create video from image frames

Usage:
    # From a folder of images
    python frames_to_video.py --input frames_folder/ --output output.mp4
    
    # With specific FPS
    python frames_to_video.py --input frames_folder/ --output output.mp4 --fps 30
    
    # From numbered sequence
    python frames_to_video.py --input frames/ --output video.mp4 --pattern "frame_%04d.jpg"
    
    # With specific resolution
    python frames_to_video.py --input frames/ --output video.mp4 --width 1920 --height 1080
"""

import cv2
import numpy as np
import argparse
from pathlib import Path
import re
import sys

def natural_sort_key(s):
    """Sort strings with numbers naturally (1, 2, 10 instead of 1, 10, 2)"""
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split('([0-9]+)', str(s))]

def create_video_from_frames(input_dir, output_path, fps=30, width=None, height=None, 
                             pattern=None, codec='mp4v', quality=95):
    """
    Create video from image frames
    
    Args:
        input_dir: Directory containing image frames
        output_path: Output video file path
        fps: Frames per second (default: 30)
        width: Output width (if None, use first image width)
        height: Output height (if None, use first image height)
        pattern: File pattern (e.g., "frame_%04d.jpg")
        codec: Video codec (default: 'mp4v')
        quality: JPEG quality for encoding (0-100)
    """
    
    input_path = Path(input_dir)
    
    if not input_path.exists():
        print(f"❌ Error: Input directory not found: {input_dir}")
        return False
    
    # Get list of image files
    if pattern:
        # Use pattern matching
        images = sorted(input_path.glob(pattern))
    else:
        # Get all common image formats
        extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
        images = []
        for ext in extensions:
            images.extend(input_path.glob(ext))
        
        # Sort naturally
        images = sorted(images, key=natural_sort_key)
    
    if not images:
        print(f"❌ Error: No images found in {input_dir}")
        if pattern:
            print(f"   Pattern used: {pattern}")
        return False
    
    print(f"📁 Found {len(images)} images")
    print(f"📷 First image: {images[0].name}")
    print(f"📷 Last image: {images[-1].name}")
    print()
    
    # Read first image to get dimensions
    first_frame = cv2.imread(str(images[0]))
    if first_frame is None:
        print(f"❌ Error: Could not read first image: {images[0]}")
        return False
    
    h, w = first_frame.shape[:2]
    
    # Use specified dimensions or original
    out_width = width if width else w
    out_height = height if height else h
    
    print(f"📊 Video properties:")
    print(f"   Resolution: {out_width}x{out_height}")
    print(f"   FPS: {fps}")
    print(f"   Codec: {codec}")
    print(f"   Duration: {len(images)/fps:.2f} seconds")
    print()
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*codec)
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (out_width, out_height))
    
    if not out.isOpened():
        print(f"❌ Error: Could not create video writer")
        return False
    
    print("🎬 Creating video...")
    
    # Process each frame
    for i, img_path in enumerate(images):
        # Read image
        frame = cv2.imread(str(img_path))
        
        if frame is None:
            print(f"⚠️  Warning: Could not read {img_path.name}, skipping...")
            continue
        
        # Resize if needed
        if frame.shape[1] != out_width or frame.shape[0] != out_height:
            frame = cv2.resize(frame, (out_width, out_height))
        
        # Write frame
        out.write(frame)
        
        # Progress indicator
        if (i + 1) % 10 == 0 or (i + 1) == len(images):
            progress = (i + 1) / len(images) * 100
            print(f"   Progress: {i+1}/{len(images)} ({progress:.1f}%)", end='\r')
    
    print()
    
    # Release video writer
    out.release()
    
    # Verify output
    output_file = Path(output_path)
    if output_file.exists():
        size_mb = output_file.stat().st_size / (1024 * 1024)
        print()
        print(f"✅ Video created successfully!")
        print(f"📹 Output: {output_path}")
        print(f"💾 Size: {size_mb:.2f} MB")
        print(f"⏱️  Duration: {len(images)/fps:.2f} seconds")
        return True
    else:
        print(f"❌ Error: Video file was not created")
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Create video from image frames',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Basic usage
  python frames_to_video.py --input frames/ --output video.mp4
  
  # Custom FPS
  python frames_to_video.py --input frames/ --output video.mp4 --fps 60
  
  # With pattern
  python frames_to_video.py --input frames/ --output video.mp4 --pattern "frame_*.jpg"
  
  # Resize video
  python frames_to_video.py --input frames/ --output video.mp4 --width 1920 --height 1080
  
  # High quality
  python frames_to_video.py --input frames/ --output video.mp4 --codec XVID --quality 100
        '''
    )
    
    parser.add_argument('--input', '-i', type=str, required=True,
                       help='Input directory containing image frames')
    parser.add_argument('--output', '-o', type=str, required=True,
                       help='Output video file path (e.g., output.mp4)')
    parser.add_argument('--fps', type=int, default=30,
                       help='Frames per second (default: 30)')
    parser.add_argument('--width', type=int, default=None,
                       help='Output width (default: use image width)')
    parser.add_argument('--height', type=int, default=None,
                       help='Output height (default: use image height)')
    parser.add_argument('--pattern', type=str, default=None,
                       help='File pattern (e.g., "frame_*.jpg")')
    parser.add_argument('--codec', type=str, default='mp4v',
                       choices=['mp4v', 'XVID', 'H264', 'MJPG'],
                       help='Video codec (default: mp4v)')
    parser.add_argument('--quality', type=int, default=95,
                       help='Video quality 0-100 (default: 95)')
    
    args = parser.parse_args()
    
    print("="*60)
    print("🎬 FRAMES TO VIDEO CONVERTER")
    print("="*60)
    print()
    
    success = create_video_from_frames(
        input_dir=args.input,
        output_path=args.output,
        fps=args.fps,
        width=args.width,
        height=args.height,
        pattern=args.pattern,
        codec=args.codec,
        quality=args.quality
    )
    
    if success:
        print()
        print("="*60)
        print("🎉 Done!")
        print("="*60)
        sys.exit(0)
    else:
        print()
        print("="*60)
        print("❌ Failed to create video")
        print("="*60)
        sys.exit(1)


if __name__ == "__main__":
    main()
