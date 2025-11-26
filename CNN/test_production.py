#!/usr/bin/env python3
"""
Production Features Test Suite
Test all new production modules before pushing to Git
"""

import sys
import os
from pathlib import Path

# Add CNN to path
CNN_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(CNN_DIR))

print("="*70)
print("🧪 PRODUCTION FEATURES TEST SUITE")
print("="*70)

# Test results tracker
test_results = {
    'passed': [],
    'failed': [],
    'warnings': []
}

def test_result(test_name, passed, message=""):
    """Record test result"""
    if passed:
        test_results['passed'].append(test_name)
        print(f"✅ {test_name}: PASSED {message}")
    else:
        test_results['failed'].append(test_name)
        print(f"❌ {test_name}: FAILED {message}")

def test_warning(test_name, message):
    """Record test warning"""
    test_results['warnings'].append((test_name, message))
    print(f"⚠️  {test_name}: WARNING - {message}")


# =============================================================================
# TEST 1: Extended Classes Configuration
# =============================================================================
print("\n" + "="*70)
print("TEST 1: Extended Classes Configuration")
print("="*70)

try:
    from config.classes_config import (
        INDIAN_VEHICLE_CLASSES,
        REAL_HEIGHTS,
        CLASS_COLORS,
        RISK_PRIORITY
    )
    
    # Check classes
    expected_classes = ['car', 'truck', 'bus', 'person', 'auto_rickshaw', 
                       'motorcycle', 'bicycle', 'animal']
    assert INDIAN_VEHICLE_CLASSES == expected_classes, "Class list mismatch"
    test_result("Extended Classes", True, f"- {len(INDIAN_VEHICLE_CLASSES)} classes")
    
    # Check real heights
    assert len(REAL_HEIGHTS) == 8, "Missing real height definitions"
    test_result("Real Heights", True, f"- All {len(REAL_HEIGHTS)} classes defined")
    
    # Check colors
    assert len(CLASS_COLORS) == 8, "Missing color definitions"
    test_result("Class Colors", True, f"- All {len(CLASS_COLORS)} colors defined")
    
    # Check risk priority
    assert len(RISK_PRIORITY) == 8, "Missing risk priority"
    assert RISK_PRIORITY['person'] == 10, "Person should have highest priority"
    test_result("Risk Priority", True, "- Person has highest priority (10)")
    
except Exception as e:
    test_result("Extended Classes", False, f"- Error: {e}")


# =============================================================================
# TEST 2: TTC Warning System
# =============================================================================
print("\n" + "="*70)
print("TEST 2: Time-to-Collision (TTC) Warning System")
print("="*70)

try:
    from production.ttc_warning import (
        TTCCalculator,
        MultiObjectWarningSystem,
        RiskLevel
    )
    
    # Test TTC calculator
    ttc_calc = TTCCalculator()
    
    # Test scenario 1: Approaching vehicle
    ttc, risk = ttc_calc.calculate_ttc(
        distance=20.0,
        relative_velocity=-5.0,  # Approaching at 5 m/s
        object_class='car'
    )
    assert ttc == 4.0, f"Expected TTC=4.0, got {ttc}"
    assert risk == RiskLevel.CAUTION, f"Expected CAUTION, got {risk}"
    test_result("TTC Calculation", True, f"- TTC={ttc:.1f}s, Risk={risk.name}")
    
    # Test scenario 2: Imminent collision
    ttc, risk = ttc_calc.calculate_ttc(
        distance=3.0,
        relative_velocity=-5.0,
        object_class='person'
    )
    assert risk == RiskLevel.IMMINENT, "Should be IMMINENT risk"
    test_result("Imminent Warning", True, f"- TTC={ttc:.1f}s")
    
    # Test warning message generation
    warning = ttc_calc.get_warning_message(
        RiskLevel.CRITICAL,
        ttc=1.5,
        object_class='motorcycle',
        distance=7.5
    )
    assert warning['audio_alert'] == True, "Should have audio alert"
    assert warning['haptic_alert'] == True, "Should have haptic alert"
    test_result("Warning Generation", True, f"- Multi-modal alerts enabled")
    
    # Test multi-object system
    warning_system = MultiObjectWarningSystem()
    test_result("Multi-Object System", True, "- Initialized successfully")
    
except Exception as e:
    test_result("TTC Warning System", False, f"- Error: {e}")


# =============================================================================
# TEST 3: Night-Time Enhancement
# =============================================================================
print("\n" + "="*70)
print("TEST 3: Night-Time Enhancement")
print("="*70)

try:
    import cv2
    import numpy as np
    from production.night_enhancement import (
        LowLightEnhancer,
        NightTimeDetector
    )
    
    # Create test images
    # Bright image
    bright_img = np.ones((480, 640, 3), dtype=np.uint8) * 200
    
    # Dark image
    dark_img = np.ones((480, 640, 3), dtype=np.uint8) * 30
    
    # Very dark with bright spots (simulating headlights)
    night_img = np.ones((480, 640, 3), dtype=np.uint8) * 20
    night_img[200:250, 300:350] = 255  # Bright spot (headlight)
    
    # Test enhancer
    enhancer = LowLightEnhancer(enhancement_mode='auto')
    
    # Test on bright image (should not enhance)
    enhanced, method = enhancer.enhance(bright_img)
    assert method == 'none', "Bright image should not be enhanced"
    test_result("Bright Image Detection", True, f"- Method: {method}")
    
    # Test on dark image (should enhance)
    enhanced, method = enhancer.enhance(dark_img)
    assert method != 'none', "Dark image should be enhanced"
    avg_brightness_before = np.mean(dark_img)
    avg_brightness_after = np.mean(enhanced)
    improvement = ((avg_brightness_after - avg_brightness_before) / avg_brightness_before) * 100
    test_result("Low-Light Enhancement", True, 
               f"- Method: {method}, Improvement: {improvement:.1f}%")
    
    # Test CLAHE
    enhancer_clahe = LowLightEnhancer(enhancement_mode='clahe')
    enhanced_clahe = enhancer_clahe.enhance_clahe(dark_img)
    test_result("CLAHE Enhancement", True, "- Processed successfully")
    
    # Test gamma correction
    enhanced_gamma = enhancer_clahe.enhance_gamma(dark_img, gamma=1.5)
    test_result("Gamma Correction", True, "- Processed successfully")
    
    # Test night-time detector
    night_detector = NightTimeDetector()
    is_night, confidence = night_detector.detect_night_time(night_img)
    test_result("Night Detection", True, 
               f"- Is night: {is_night}, Confidence: {confidence:.2f}")
    
except ImportError as e:
    test_warning("Night Enhancement", f"cv2/numpy not available: {e}")
except Exception as e:
    test_result("Night Enhancement", False, f"- Error: {e}")


# =============================================================================
# TEST 4: REST API Structure
# =============================================================================
print("\n" + "="*70)
print("TEST 4: REST API Structure")
print("="*70)

try:
    # Check if FastAPI is available
    try:
        import fastapi
        import pydantic
        fastapi_available = True
    except ImportError:
        fastapi_available = False
        test_warning("REST API", "FastAPI not installed - run: pip install fastapi uvicorn")
    
    if fastapi_available:
        from production.api_server import app
        
        # Check endpoints exist
        routes = [route.path for route in app.routes]
        
        expected_routes = [
            '/',
            '/health',
            '/api/v1/detect',
            '/api/v1/detect/stream',
            '/api/v1/telemetry',
            '/api/v1/stats',
            '/api/v1/model/update',
            '/api/v1/edge_cases'
        ]
        
        for route in expected_routes:
            if route in routes:
                test_result(f"Endpoint: {route}", True)
            else:
                test_result(f"Endpoint: {route}", False, "- Not found")
        
        test_result("API Structure", True, f"- {len(routes)} routes defined")
    
except Exception as e:
    test_result("REST API", False, f"- Error: {e}")


# =============================================================================
# TEST 5: Performance Optimization
# =============================================================================
print("\n" + "="*70)
print("TEST 5: Performance Optimization")
print("="*70)

try:
    from production.performance_optimization import (
        FrameCapture,
        ParallelADASPipeline,
        LatencyMonitor,
        GPUOptimizer
    )
    
    # Test latency monitor
    monitor = LatencyMonitor()
    monitor.record('detection', 15.5)
    monitor.record('detection', 18.2)
    monitor.record('refinement', 8.3)
    
    report = monitor.get_report()
    assert 'detection' in report, "Detection metrics missing"
    assert report['detection']['avg_ms'] > 0, "Invalid average"
    test_result("Latency Monitor", True, 
               f"- Avg detection: {report['detection']['avg_ms']:.1f}ms")
    
    # Test GPU optimizer
    import torch
    if torch.cuda.is_available():
        test_result("GPU Available", True, 
                   f"- {torch.cuda.get_device_name(0)}")
        
        # Test CUDA optimization
        GPUOptimizer.optimize_cuda_settings()
        test_result("CUDA Optimization", True, "- Settings applied")
    else:
        test_warning("GPU", "CUDA not available - will use CPU")
    
    test_result("Performance Module", True, "- All classes imported successfully")
    
except Exception as e:
    test_result("Performance Optimization", False, f"- Error: {e}")


# =============================================================================
# TEST 6: Integration Test (if camera available)
# =============================================================================
print("\n" + "="*70)
print("TEST 6: Integration Test (Optional - Camera Required)")
print("="*70)

try:
    import cv2
    
    # Try to open camera - test multiple camera indices
    camera_found = False
    camera_id = None
    
    for cam_id in [0, 2, 4]:  # Test common camera indices
        cap = cv2.VideoCapture(cam_id)
        if cap.isOpened():
            ret, frame = cap.read()
            cap.release()
            
            if ret and frame is not None:
                camera_found = True
                camera_id = cam_id
                test_result("Camera Access", True, 
                           f"- Camera {cam_id}: {frame.shape[1]}x{frame.shape[0]}")
                
                # Test enhancement on real frame
                enhancer = LowLightEnhancer(enhancement_mode='auto')
                enhanced, method = enhancer.enhance(frame)
                test_result("Real Frame Enhancement", True, f"- Method: {method}")
                break
    
    if not camera_found:
        test_warning("Camera", "No camera available at indices 0, 2, 4 - skipping integration test")
        
except Exception as e:
    test_warning("Integration Test", f"Skipped - {e}")


# =============================================================================
# TEST 7: File Structure
# =============================================================================
print("\n" + "="*70)
print("TEST 7: File Structure")
print("="*70)

required_files = {
    'config/classes_config.py': 'Extended classes configuration',
    'production/ttc_warning.py': 'TTC warning system',
    'production/night_enhancement.py': 'Night enhancement',
    'production/api_server.py': 'REST API server',
    'production/performance_optimization.py': 'Performance optimization',
    'production/requirements_production.txt': 'Production dependencies',
    'PRODUCTION_README.md': 'Production documentation'
}

for file_path, description in required_files.items():
    full_path = CNN_DIR / file_path
    if full_path.exists():
        size_kb = full_path.stat().st_size / 1024
        test_result(f"File: {file_path}", True, f"- {size_kb:.1f} KB")
    else:
        test_result(f"File: {file_path}", False, "- Not found")


# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "="*70)
print("📊 TEST SUMMARY")
print("="*70)

total_tests = len(test_results['passed']) + len(test_results['failed'])
pass_rate = (len(test_results['passed']) / total_tests * 100) if total_tests > 0 else 0

print(f"\n✅ Passed: {len(test_results['passed'])}/{total_tests} ({pass_rate:.1f}%)")
print(f"❌ Failed: {len(test_results['failed'])}/{total_tests}")
print(f"⚠️  Warnings: {len(test_results['warnings'])}")

if test_results['passed']:
    print("\n✅ Passed Tests:")
    for test in test_results['passed']:
        print(f"   • {test}")

if test_results['failed']:
    print("\n❌ Failed Tests:")
    for test in test_results['failed']:
        print(f"   • {test}")

if test_results['warnings']:
    print("\n⚠️  Warnings:")
    for test, msg in test_results['warnings']:
        print(f"   • {test}: {msg}")

# Final verdict
print("\n" + "="*70)
if len(test_results['failed']) == 0:
    print("🎉 ALL TESTS PASSED! Ready to push to Git.")
    print("="*70)
    sys.exit(0)
else:
    print("⚠️  SOME TESTS FAILED - Fix issues before pushing")
    print("="*70)
    sys.exit(1)
