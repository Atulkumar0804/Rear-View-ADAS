#!/usr/bin/env python3
"""
Quick test to verify camera detection system is ready
"""

import sys
import torch
from pathlib import Path

print("="*60)
print("🔍 CAMERA DETECTION SYSTEM - VERIFICATION TEST")
print("="*60)
print()

# Test 1: Check Python version
print("1. Python Version:")
print(f"   ✅ Python {sys.version.split()[0]}")
print()

# Test 2: Check PyTorch
print("2. PyTorch:")
print(f"   ✅ Version: {torch.__version__}")
print(f"   {'✅' if torch.cuda.is_available() else '⚠️ '} CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"   ✅ GPU: {torch.cuda.get_device_name(0)}")
print()

# Test 3: Check YOLO model
print("3. YOLO Model:")
yolo_path = Path("../models/yolo/yolov8n_RearView.pt")
if yolo_path.exists():
    print(f"   ✅ Found: {yolo_path}")
else:
    print(f"   ❌ Missing: {yolo_path}")
print()

# Test 4: Check trained CNN models
print("4. Trained CNN Models:")
checkpoints_dir = Path("checkpoints")
if checkpoints_dir.exists():
    models_found = 0
    for model_dir in sorted(checkpoints_dir.iterdir()):
        if model_dir.is_dir():
            model_file = model_dir / "best_model.pth"
            if model_file.exists():
                size_mb = model_file.stat().st_size / (1024*1024)
                print(f"   ✅ {model_dir.name}: {size_mb:.1f} MB")
                models_found += 1
    
    if models_found == 0:
        print("   ❌ No models found - run: python train_v2.py")
else:
    print("   ❌ No checkpoints directory")
print()

# Test 5: Check camera availability
print("5. Camera Availability:")
try:
    import cv2
    camera_found = False
    for i in range(5):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            print(f"   ✅ Camera {i}: {width}x{height} @ {fps} FPS")
            cap.release()
            camera_found = True
    
    if not camera_found:
        print("   ⚠️  No cameras detected")
except Exception as e:
    print(f"   ❌ Error checking cameras: {e}")
print()

# Test 6: Test model loading
print("6. Model Loading Test:")
try:
    sys.path.append('.')
    from models.architectures import create_model
    
    # Test creating transfer_resnet18 model
    model = create_model('transfer_resnet18', num_classes=4, pretrained=False)
    print("   ✅ Model creation successful")
    print(f"   ✅ Model type: transfer_resnet18")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   ✅ Total parameters: {total_params:,}")
    print(f"   ✅ Trainable parameters: {trainable_params:,}")
    
except Exception as e:
    print(f"   ❌ Error loading model: {e}")
print()

# Test 7: Check scripts
print("7. Execution Scripts:")
scripts = [
    'camera_inference.py',
    'run_camera.sh',
    'menu.sh',
    'test_camera.py'
]
for script in scripts:
    script_path = Path(script)
    if script_path.exists():
        print(f"   ✅ {script}")
    else:
        print(f"   ❌ {script}")
print()

# Final verdict
print("="*60)
print("📊 SYSTEM STATUS")
print("="*60)

checkpoints_exist = checkpoints_dir.exists() and any(
    (model_dir / "best_model.pth").exists() 
    for model_dir in checkpoints_dir.iterdir() 
    if model_dir.is_dir()
)

if checkpoints_exist and yolo_path.exists():
    print("✅ System is READY for camera detection!")
    print()
    print("🚀 To start detection, run:")
    print("   ./run_camera.sh")
    print("   OR")
    print("   python camera_inference.py --camera 2")
elif not checkpoints_exist:
    print("⚠️  System needs training!")
    print()
    print("📋 Next steps:")
    print("   1. Train models: python train_v2.py")
    print("   2. Then run: ./run_camera.sh")
else:
    print("⚠️  System has issues - check errors above")

print("="*60)
print()
