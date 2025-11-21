"""
Test camera availability
"""
import cv2

print("🔍 Testing available cameras...\n")

for i in range(5):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        print(f"✅ Camera {i}: {width}x{height} @ {fps} FPS")
        cap.release()
    else:
        print(f"❌ Camera {i}: Not available")

print("\n💡 To use camera, run:")
print("   python camera_inference.py --camera 0")
print("   OR")
print("   ./run_camera.sh")
