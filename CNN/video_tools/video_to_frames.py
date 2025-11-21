#!/usr/bin/env python3
"""
Extract frames from video

Usage:
    # Extract all frames
    python video_to_frames.py --input video.mp4 --output frames/
    
    # Extract every Nth frame
    python video_to_frames.py --input video.mp4 --output frames/ --skip 10
    
    # Extract specific time range
    python video_to_frames.py --input video.mp4 --output frames/ --start 5 --end 30
    
    # Extract with custom naming
    python video_to_frames.py --input video.mp4 --output frames/ --prefix "frame_"
"""

import cv2
import argparse
from pathlib import Path
import sys

def extract_frames(video_path, output_dir, skip=1, start_time=None, end_time=None, 
                   prefix="frame", format="jpg", quality=95):
    """
    Extract frames from video
    
    Args:
        video_path: Path to input video
        output_dir: Directory to save frames
        skip: Extract every Nth frame (default: 1 = all frames)
        start_time: Start time in seconds (default: None = from beginning)
        end_time: End time in seconds (default: None = to end)
        prefix: Filename prefix (default: "frame")
        format: Image format (jpg, png, bmp)
        quality: JPEG quality 0-100
    """
    
    # Open video
    cap = cv2.VideoCapture(str(video_path))
    
    if not cap.isOpened():
        print(f"❌ Error: Could not open video: {video_path}")
        return False
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = total_frames / fps if fps > 0 else 0
    
    print(f"📹 Video properties:")
    print(f"   Resolution: {width}x{height}")
    print(f"   FPS: {fps:.2f}")
    print(f"   Total frames: {total_frames}")
    print(f"   Duration: {duration:.2f} seconds")
    print()
    
    # Calculate frame range
    start_frame = int(start_time * fps) if start_time else 0
    end_frame = int(end_time * fps) if end_time else total_frames
    
    # Ensure valid range
    start_frame = max(0, start_frame)
    end_frame = min(total_frames, end_frame)
    
    print(f"⚙️  Extraction settings:")
    print(f"   Frame range: {start_frame} to {end_frame}")
    print(f"   Skip: every {skip} frame(s)")
    print(f"   Expected frames: {(end_frame - start_frame) // skip}")
    print()
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"📁 Output directory: {output_dir}")
    print()
    print("🎬 Extracting frames...")
    
    # Set to start frame
    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    frame_count = 0
    saved_count = 0
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            break
        
        current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
        
        # Check if we're past end frame
        if current_frame >= end_frame:
            break
        
        # Skip frames
        if frame_count % skip == 0:
            # Save frame
            output_filename = f"{prefix}_{saved_count:06d}.{format}"
            output_file = output_path / output_filename
            
            # Set quality for JPEG
            if format.lower() in ['jpg', 'jpeg']:
                cv2.imwrite(str(output_file), frame, 
                           [cv2.IMWRITE_JPEG_QUALITY, quality])
            elif format.lower() == 'png':
                cv2.imwrite(str(output_file), frame,
                           [cv2.IMWRITE_PNG_COMPRESSION, 9])
            else:
                cv2.imwrite(str(output_file), frame)
            
            saved_count += 1
            
            # Progress
            if saved_count % 10 == 0:
                progress = (current_frame - start_frame) / (end_frame - start_frame) * 100
                print(f"   Progress: {saved_count} frames saved ({progress:.1f}%)", end='\r')
        
        frame_count += 1
    
    print()
    
    # Release video
    cap.release()
    
    print()
    print(f"✅ Extraction complete!")
    print(f"📸 Frames saved: {saved_count}")
    print(f"📁 Location: {output_dir}")
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description='Extract frames from video',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Extract all frames
  python video_to_frames.py --input video.mp4 --output frames/
  
  # Extract every 10th frame
  python video_to_frames.py --input video.mp4 --output frames/ --skip 10
  
  # Extract frames from 5s to 30s
  python video_to_frames.py --input video.mp4 --output frames/ --start 5 --end 30
  
  # Save as PNG with custom prefix
  python video_to_frames.py --input video.mp4 --output frames/ --format png --prefix "img"
  
  # High quality JPEG
  python video_to_frames.py --input video.mp4 --output frames/ --quality 100
        '''
    )
    
    parser.add_argument('--input', '-i', type=str, required=True,
                       help='Input video file path')
    parser.add_argument('--output', '-o', type=str, required=True,
                       help='Output directory for frames')
    parser.add_argument('--skip', type=int, default=1,
                       help='Extract every Nth frame (default: 1 = all frames)')
    parser.add_argument('--start', type=float, default=None,
                       help='Start time in seconds (default: from beginning)')
    parser.add_argument('--end', type=float, default=None,
                       help='End time in seconds (default: to end)')
    parser.add_argument('--prefix', type=str, default='frame',
                       help='Filename prefix (default: "frame")')
    parser.add_argument('--format', type=str, default='jpg',
                       choices=['jpg', 'jpeg', 'png', 'bmp'],
                       help='Image format (default: jpg)')
    parser.add_argument('--quality', type=int, default=95,
                       help='JPEG quality 0-100 (default: 95)')
    
    args = parser.parse_args()
    
    # Check if input exists
    if not Path(args.input).exists():
        print(f"❌ Error: Video file not found: {args.input}")
        sys.exit(1)
    
    print("="*60)
    print("🎬 VIDEO TO FRAMES EXTRACTOR")
    print("="*60)
    print()
    
    success = extract_frames(
        video_path=args.input,
        output_dir=args.output,
        skip=args.skip,
        start_time=args.start,
        end_time=args.end,
        prefix=args.prefix,
        format=args.format,
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
        print("❌ Failed to extract frames")
        print("="*60)
        sys.exit(1)


if __name__ == "__main__":
    main()
