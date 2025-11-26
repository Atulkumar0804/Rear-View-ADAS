"""
Test Depth Estimation on Sample Images
Verify Depth-Anything-V2 is working correctly
"""

import cv2
import numpy as np
import sys
from pathlib import Path
import time

# Add parent directory to path to import depth_estimation
sys.path.insert(0, str(Path(__file__).parent.parent))
from depth_estimation import DepthEstimator, DepthConfig

def test_depth_on_image(image_path: str, model_size: str = 'small'):
    """Test depth estimation on a single image"""
    
    print("\n" + "="*60)
    print("DEPTH ESTIMATION TEST")
    print("="*60)
    
    # Check if image exists
    if not Path(image_path).exists():
        print(f"❌ Image not found: {image_path}")
        return
    
    # Load image
    print(f"📷 Loading image: {image_path}")
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"❌ Failed to load image")
        return
    
    print(f"   Image shape: {image.shape}")
    
    # Initialize depth estimator
    estimator = DepthEstimator(model_size=model_size, device='cuda')
    
    # Estimate depth
    print("\n🔍 Estimating depth...")
    start = time.time()
    depth_map = estimator.estimate_depth(image)
    elapsed = time.time() - start
    
    print(f"✅ Depth estimated in {elapsed*1000:.1f}ms")
    print(f"   Depth shape: {depth_map.shape}")
    print(f"   Depth range: {depth_map.min():.3f} - {depth_map.max():.3f}")
    
    # Visualize depth
    depth_colored = estimator.visualize_depth(depth_map, colormap=cv2.COLORMAP_INFERNO)
    
    # Create side-by-side comparison
    h, w = image.shape[:2]
    comparison = np.zeros((h, w*2, 3), dtype=np.uint8)
    comparison[:, :w] = image
    comparison[:, w:] = depth_colored
    
    # Add labels
    cv2.putText(comparison, "Original Image", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(comparison, "Depth Map", (w + 10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Add depth info
    cv2.putText(comparison, f"Min: {depth_map.min():.2f}m", (w + 10, h - 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(comparison, f"Max: {depth_map.max():.2f}m", (w + 10, h - 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Save result
    output_path = Path(image_path).stem + f"_depth_{model_size}.jpg"
    cv2.imwrite(output_path, comparison)
    print(f"\n💾 Saved result to: {output_path}")
    
    # Display
    print("\n📺 Displaying result (press any key to close)...")
    cv2.imshow('Depth Estimation Test', comparison)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    print("\n✅ Test completed successfully!")
    print("="*60 + "\n")


def test_depth_on_camera(camera_id = 0, model_size: str = 'small'):
    """Test depth estimation on live camera feed"""
    
    print("\n" + "="*60)
    print("LIVE CAMERA DEPTH ESTIMATION TEST")
    print("="*60)
    
    # Initialize depth estimator
    estimator = DepthEstimator(model_size=model_size, device='cuda')
    
    # Open camera
    print(f"\n📷 Opening camera {camera_id}...")
    cap = cv2.VideoCapture(camera_id)
    
    if not cap.isOpened():
        print(f"❌ Failed to open camera {camera_id}")
        return
    
    print("✅ Camera opened")
    print("\n🚀 Starting live depth estimation...")
    print("   Press 'q' to quit")
    print("   Press 's' to save screenshot\n")
    
    frame_count = 0
    start_time = time.time()
    screenshot_count = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Failed to read frame")
                break
            
            frame_count += 1
            
            # Estimate depth
            depth_map = estimator.estimate_depth(frame)
            
            # Visualize
            depth_colored = estimator.visualize_depth(depth_map)
            
            # Calculate FPS
            elapsed = time.time() - start_time
            fps = frame_count / elapsed if elapsed > 0 else 0
            
            # Create side-by-side
            h, w = frame.shape[:2]
            comparison = np.zeros((h, w*2, 3), dtype=np.uint8)
            comparison[:, :w] = frame
            comparison[:, w:] = depth_colored
            
            # Add info
            cv2.putText(comparison, "Original", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(comparison, "Depth", (w + 10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(comparison, f"FPS: {fps:.1f}", (10, h - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Display
            cv2.imshow('Live Depth Estimation - Press q to quit', comparison)
            
            # Handle keys
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                screenshot_path = f"depth_screenshot_{screenshot_count:03d}.jpg"
                cv2.imwrite(screenshot_path, comparison)
                print(f"📸 Screenshot saved: {screenshot_path}")
                screenshot_count += 1
    
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        
        # Print stats
        elapsed = time.time() - start_time
        avg_fps = frame_count / elapsed if elapsed > 0 else 0
        
        print("\n" + "="*60)
        print("📊 SESSION STATISTICS")
        print("="*60)
        print(f"Frames processed: {frame_count}")
        print(f"Time elapsed: {elapsed:.2f}s")
        print(f"Average FPS: {avg_fps:.2f}")
        if screenshot_count > 0:
            print(f"Screenshots: {screenshot_count}")
        print("="*60 + "\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Test Depth Estimation')
    parser.add_argument('--mode', type=str, default='camera', 
                       choices=['image', 'camera'],
                       help='Test mode: image or camera')
    parser.add_argument('--image', type=str, 
                       help='Path to test image (for image mode)')
    parser.add_argument('--camera', type=str, default='0',
                       help='Camera ID or video file path (for camera mode)')
    parser.add_argument('--model', type=str, default='small',
                       choices=['small', 'base', 'large'],
                       help='Model size')
    
    args = parser.parse_args()
    
    # Print config
    DepthConfig.print_config()
    
    if args.mode == 'image':
        if args.image is None:
            print("❌ Please provide --image path for image mode")
        else:
            test_depth_on_image(args.image, model_size=args.model)
    else:
        test_depth_on_camera(camera_id=args.camera, model_size=args.model)
