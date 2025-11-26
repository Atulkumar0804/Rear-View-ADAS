#!/usr/bin/env python3
"""
Test all videos with fine-tuned depth model and analyze results.
"""

import os
import sys
import json
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from test_full_adas import VehicleDetectorWithDepth, draw_full_annotations
import time


def analyze_video_detections(video_path, detector):
    """
    Analyze detections in a video and collect statistics.
    
    Returns:
        dict: Statistics including depth ranges, velocities, vehicle counts
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    stats = {
        'video_name': Path(video_path).name,
        'total_frames': total_frames,
        'fps': fps,
        'frames_with_vehicles': 0,
        'total_detections': 0,
        'depth_values': [],
        'velocity_values': [],
        'vehicle_counts_per_frame': [],
        'class_distribution': {},
        'approaching_count': 0,
        'receding_count': 0,
        'stable_count': 0,
        'collision_warnings': 0,
    }
    
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        timestamp = frame_count / fps if fps > 0 else frame_count * 0.033
        
        # Process frame
        detections, depth_map = detector.process_frame(frame, timestamp)
        
        # Collect statistics
        if detections:
            stats['frames_with_vehicles'] += 1
            stats['total_detections'] += len(detections)
            stats['vehicle_counts_per_frame'].append(len(detections))
            
            for det in detections:
                # Depth
                if 'depth' in det:
                    stats['depth_values'].append(det['depth'])
                
                # Velocity
                if 'velocity' in det:
                    stats['velocity_values'].append(det['velocity'])
                
                # Class
                cls = det.get('class', 'unknown')
                stats['class_distribution'][cls] = stats['class_distribution'].get(cls, 0) + 1
                
                # Status
                status = det.get('status', '')
                if 'APPROACHING' in status:
                    stats['approaching_count'] += 1
                elif 'RECEDING' in status:
                    stats['receding_count'] += 1
                elif 'STABLE' in status:
                    stats['stable_count'] += 1
                
                # TTC warnings
                if 'ttc' in det and det['ttc'] is not None and det['ttc'] < 2.0:
                    stats['collision_warnings'] += 1
    
    cap.release()
    
    # Calculate summary statistics
    if stats['depth_values']:
        stats['depth_min'] = float(np.min(stats['depth_values']))
        stats['depth_max'] = float(np.max(stats['depth_values']))
        stats['depth_mean'] = float(np.mean(stats['depth_values']))
        stats['depth_std'] = float(np.std(stats['depth_values']))
    
    if stats['velocity_values']:
        stats['velocity_min'] = float(np.min(stats['velocity_values']))
        stats['velocity_max'] = float(np.max(stats['velocity_values']))
        stats['velocity_mean'] = float(np.mean(stats['velocity_values']))
        stats['velocity_std'] = float(np.std(stats['velocity_values']))
    
    if stats['vehicle_counts_per_frame']:
        stats['avg_vehicles_per_frame'] = float(np.mean(stats['vehicle_counts_per_frame']))
        stats['max_vehicles_in_frame'] = int(np.max(stats['vehicle_counts_per_frame']))
    
    return stats


def test_all_videos(video_folder, yolo_model, output_folder):
    """Test all videos and save outputs with analysis."""
    
    print("="*80)
    print("🎯 TESTING ALL VIDEOS WITH FINE-TUNED DEPTH MODEL")
    print("="*80)
    
    # Get all videos
    videos = sorted(Path(video_folder).glob("cam_back_[1-9].mp4"))
    
    if not videos:
        print(f"❌ No videos found in {video_folder}")
        return
    
    print(f"\n📹 Found {len(videos)} videos to test\n")
    
    # Initialize detector (with fine-tuned model)
    print("🔧 Initializing Detection System (Fine-Tuned Model)...")
    detector = VehicleDetectorWithDepth(yolo_model, 'small', use_finetuned=True)
    print("✅ System ready!\n")
    
    os.makedirs(output_folder, exist_ok=True)
    
    all_stats = []
    
    for video_path in videos:
        video_name = video_path.stem
        print("\n" + "="*80)
        print(f"🚗 TESTING: {video_path.name}")
        print("="*80)
        
        # Open video
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"❌ Failed to open video")
            continue
        
        # Get properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"📊 Video Info:")
        print(f"   Resolution: {width}x{height}")
        print(f"   FPS: {fps:.2f}")
        print(f"   Frames: {total_frames}")
        print(f"   Duration: {total_frames/fps:.1f}s")
        
        # Output path
        output_path = os.path.join(output_folder, f"{video_name}_finetuned.mp4")
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        print(f"\n🚀 Processing...")
        
        frame_count = 0
        start_time = time.time()
        
        # Statistics for this video
        video_stats = {
            'video_name': video_path.name,
            'total_frames': total_frames,
            'fps': fps,
            'frames_with_vehicles': 0,
            'total_detections': 0,
            'depth_values': [],
            'velocity_values': [],
            'vehicle_counts': [],
            'class_counts': {},
            'approaching': 0,
            'receding': 0,
            'stable': 0,
            'warnings': 0,
        }
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            timestamp = frame_count / fps if fps > 0 else frame_count * 0.033
            
            # Process frame
            detections, depth_map = detector.process_frame(frame, timestamp)
            
            # Collect stats
            if detections:
                video_stats['frames_with_vehicles'] += 1
                video_stats['total_detections'] += len(detections)
                video_stats['vehicle_counts'].append(len(detections))
                
                for det in detections:
                    if 'depth' in det and det['depth'] is not None:
                        video_stats['depth_values'].append(det['depth'])
                    if 'velocity' in det and det['velocity'] is not None:
                        video_stats['velocity_values'].append(det['velocity'])
                    
                    cls = det.get('class', 'unknown')
                    video_stats['class_counts'][cls] = video_stats['class_counts'].get(cls, 0) + 1
                    
                    status = det.get('status', '')
                    if 'APPROACHING' in status:
                        video_stats['approaching'] += 1
                    elif 'RECEDING' in status:
                        video_stats['receding'] += 1
                    elif 'STABLE' in status:
                        video_stats['stable'] += 1
                    
                    if 'ttc' in det and det['ttc'] and det['ttc'] < 2.0:
                        video_stats['warnings'] += 1
            
            # Draw annotations
            annotated = draw_full_annotations(frame, depth_map, detections, frame_count, fps)
            out.write(annotated)
            
            # Progress
            if frame_count % 10 == 0 or frame_count == total_frames:
                elapsed = time.time() - start_time
                current_fps = frame_count / elapsed if elapsed > 0 else 0
                progress = (frame_count / total_frames) * 100
                print(f"   Progress: {progress:.1f}% ({frame_count}/{total_frames}) - "
                      f"Vehicles: {len(detections)} - FPS: {current_fps:.1f}", end='\r')
        
        cap.release()
        out.release()
        
        elapsed = time.time() - start_time
        avg_fps = frame_count / elapsed if elapsed > 0 else 0
        
        print(f"\n\n✅ Complete!")
        print(f"   Time: {elapsed:.2f}s")
        print(f"   Avg FPS: {avg_fps:.2f}")
        print(f"   Output: {output_path}")
        
        # Calculate final statistics
        if video_stats['depth_values']:
            video_stats['depth_min'] = float(np.min(video_stats['depth_values']))
            video_stats['depth_max'] = float(np.max(video_stats['depth_values']))
            video_stats['depth_mean'] = float(np.mean(video_stats['depth_values']))
            video_stats['depth_std'] = float(np.std(video_stats['depth_values']))
        
        if video_stats['velocity_values']:
            video_stats['velocity_min'] = float(np.min(video_stats['velocity_values']))
            video_stats['velocity_max'] = float(np.max(video_stats['velocity_values']))
            video_stats['velocity_mean'] = float(np.mean(video_stats['velocity_values']))
        
        if video_stats['vehicle_counts']:
            video_stats['avg_vehicles'] = float(np.mean(video_stats['vehicle_counts']))
            video_stats['max_vehicles'] = int(np.max(video_stats['vehicle_counts']))
        
        # Print summary
        print(f"\n📊 Video Statistics:")
        print(f"   Frames with vehicles: {video_stats['frames_with_vehicles']}/{total_frames}")
        print(f"   Total detections: {video_stats['total_detections']}")
        if 'avg_vehicles' in video_stats:
            print(f"   Avg vehicles/frame: {video_stats['avg_vehicles']:.1f}")
            print(f"   Max vehicles/frame: {video_stats['max_vehicles']}")
        if 'depth_mean' in video_stats:
            print(f"   Depth range: {video_stats['depth_min']:.1f}m - {video_stats['depth_max']:.1f}m")
            print(f"   Depth mean: {video_stats['depth_mean']:.1f}m ± {video_stats['depth_std']:.1f}m")
        if 'velocity_mean' in video_stats:
            print(f"   Velocity range: {video_stats['velocity_min']:.2f} - {video_stats['velocity_max']:.2f} m/s")
            print(f"   Velocity mean: {video_stats['velocity_mean']:.2f} m/s")
        print(f"   Vehicle classes: {video_stats['class_counts']}")
        print(f"   Motion: Approaching={video_stats['approaching']}, "
              f"Stable={video_stats['stable']}, Receding={video_stats['receding']}")
        print(f"   Collision warnings: {video_stats['warnings']}")
        
        all_stats.append(video_stats)
    
    # Save statistics to JSON
    stats_path = os.path.join(output_folder, 'analysis_results.json')
    with open(stats_path, 'w') as f:
        json.dump(all_stats, f, indent=2)
    
    print("\n\n" + "="*80)
    print("📊 OVERALL ANALYSIS")
    print("="*80)
    
    # Aggregate statistics
    total_detections = sum(s['total_detections'] for s in all_stats)
    total_frames = sum(s['total_frames'] for s in all_stats)
    all_depths = [d for s in all_stats for d in s.get('depth_values', [])]
    all_velocities = [v for s in all_stats for v in s.get('velocity_values', [])]
    
    print(f"\n📹 Videos processed: {len(all_stats)}")
    print(f"🎬 Total frames: {total_frames}")
    print(f"🚗 Total detections: {total_detections}")
    
    if all_depths:
        print(f"\n📏 Depth Statistics (all videos):")
        print(f"   Min: {np.min(all_depths):.1f}m")
        print(f"   Max: {np.max(all_depths):.1f}m")
        print(f"   Mean: {np.mean(all_depths):.1f}m ± {np.std(all_depths):.1f}m")
        print(f"   Median: {np.median(all_depths):.1f}m")
    
    if all_velocities:
        print(f"\n🏃 Velocity Statistics (all videos):")
        print(f"   Min: {np.min(all_velocities):.2f} m/s")
        print(f"   Max: {np.max(all_velocities):.2f} m/s")
        print(f"   Mean: {np.mean(all_velocities):.2f} m/s ± {np.std(all_velocities):.2f} m/s")
    
    # Class distribution
    all_classes = {}
    for s in all_stats:
        for cls, count in s.get('class_counts', {}).items():
            all_classes[cls] = all_classes.get(cls, 0) + count
    
    print(f"\n🚙 Vehicle Class Distribution:")
    for cls, count in sorted(all_classes.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / total_detections * 100) if total_detections > 0 else 0
        print(f"   {cls}: {count} ({percentage:.1f}%)")
    
    # Motion analysis
    total_approaching = sum(s['approaching'] for s in all_stats)
    total_stable = sum(s['stable'] for s in all_stats)
    total_receding = sum(s['receding'] for s in all_stats)
    total_warnings = sum(s['warnings'] for s in all_stats)
    
    print(f"\n🎯 Motion Analysis:")
    print(f"   Approaching: {total_approaching} ({total_approaching/total_detections*100:.1f}%)")
    print(f"   Stable: {total_stable} ({total_stable/total_detections*100:.1f}%)")
    print(f"   Receding: {total_receding} ({total_receding/total_detections*100:.1f}%)")
    
    print(f"\n⚠️  Collision Warnings: {total_warnings}")
    
    print(f"\n💾 Results saved to:")
    print(f"   Videos: {output_folder}/")
    print(f"   Analysis: {stats_path}")
    
    print("\n" + "="*80)
    print("✅ ALL TESTING COMPLETE")
    print("="*80)
    
    return all_stats


def main():
    """Main function."""
    
    video_folder = "/home/atul/Desktop/atul/rear_view_adas_monocular/CNN/dataset/test_viedos"
    yolo_model = "/home/atul/Desktop/atul/rear_view_adas_monocular/CNN/models/yolo/yolov8n_RearView.pt"
    output_folder = "/home/atul/Desktop/atul/rear_view_adas_monocular/CNN/dataset/test_viedos"
    
    test_all_videos(video_folder, yolo_model, output_folder)


if __name__ == "__main__":
    main()
