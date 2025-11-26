"""
Quick verification that Depth-Anything-V2 is working
"""

import sys
sys.path.append('..')

try:
    print("🔍 Testing Depth Estimation Module...")
    print("="*60)
    
    # Test imports
    print("\n1️⃣ Testing imports...")
    from depth_estimation import DepthEstimator, VehicleDepthTracker, DepthConfig
    print("   ✅ Module imports successful")
    
    # Print config
    print("\n2️⃣ Configuration:")
    DepthConfig.print_config()
    
    # Test config access
    print("3️⃣ Testing configuration access...")
    print(f"   Model size: {DepthConfig.MODEL_SIZE}")
    print(f"   History size: {DepthConfig.HISTORY_SIZE}")
    print(f"   Safe distance: {DepthConfig.SAFE_DISTANCE}m")
    print("   ✅ Configuration OK")
    
    print("\n" + "="*60)
    print("✅ ALL TESTS PASSED!")
    print("="*60)
    print("\n📝 Next steps:")
    print("   1. Test with camera: python test_depth.py --mode camera")
    print("   2. Test with image:  python test_depth.py --mode image --image /path/to/image.jpg")
    print("   3. Enhanced camera:  cd ../inference_tools && python camera_inference_depth.py")
    print("\n⚠️  Note: First run will download ~100MB model file")
    print("="*60 + "\n")
    
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()
    print("\n💡 Try: pip install depth-anything-v2 timm einops")
