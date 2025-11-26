#!/usr/bin/env python3
"""
Test the full detection pipeline: YOLOv11 + MobileNet-inspired CNN
Quick verification that both models work together
"""

import torch
from torchvision import transforms
import cv2
import numpy as np
from pathlib import Path
import sys

sys.path.append('.')
from models.architectures import create_model
from ultralytics import YOLO

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
CNN_DIR = Path(__file__).parent
YOLO_MODEL_PATH = str(PROJECT_ROOT / "yolo11s.pt")
CNN_MODEL_PATH = "checkpoints/mobilenet_inspired/best_model.pth"

# Config
IMG_SIZE = (224, 224)
VEHICLE_CLASSES = ['car', 'truck', 'bus', 'person']
CONFIDENCE_THRESHOLD = 0.4

YOLO_CLASS_MAPPING = {
    0: 'person',
    2: 'car',
    5: 'bus',
    7: 'truck'
}

CLASS_COLORS = {
    'car': (0, 255, 0),
    'truck': (255, 165, 0),
    'bus': (0, 165, 255),
    'person': (255, 0, 255),
}

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                       std=[0.229, 0.224, 0.225])
])


def test_pipeline(image_path):
    """Test full YOLOv11 + CNN pipeline"""
    print(f"\n{'='*60}")
    print("🧪 Testing Full Detection Pipeline")
    print(f"{'='*60}\n")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔥 Device: {device}\n")
    
    # Load YOLO
    print(f"📦 Loading YOLOv11...")
    yolo = YOLO(YOLO_MODEL_PATH)
    print("✅ YOLOv11 loaded\n")
    
    # Load CNN
    print(f"📦 Loading MobileNet-inspired CNN...")
    cnn = create_model('mobilenet_inspired', num_classes=len(VEHICLE_CLASSES))
    checkpoint = torch.load(CNN_MODEL_PATH, map_location=device)
    cnn.load_state_dict(checkpoint['model_state_dict'])
    cnn.to(device)
    cnn.eval()
    print("✅ CNN loaded\n")
    
    # Load image
    if not Path(image_path).exists():
        print(f"❌ Image not found: {image_path}")
        return False
    
    frame = cv2.imread(image_path)
    print(f"📸 Image: {frame.shape[1]}x{frame.shape[0]}\n")
    
    # YOLO detection
    print("🔍 Running YOLOv11 detection...")
    yolo_results = yolo(frame, verbose=False)[0]
    
    detections = []
    for detection in yolo_results.boxes.data:
        x1, y1, x2, y2, conf, cls_id = detection.cpu().numpy()
        cls_id = int(cls_id)
        
        if cls_id not in YOLO_CLASS_MAPPING:
            continue
        
        yolo_class = YOLO_CLASS_MAPPING[cls_id]
        
        # For persons, trust YOLO
        if yolo_class == 'person':
            if conf >= CONFIDENCE_THRESHOLD:
                detections.append({
                    'bbox': [int(x1), int(y1), int(x2), int(y2)],
                    'class': 'person',
                    'confidence': float(conf),
                    'source': 'YOLO'
                })
            continue
        
        # For vehicles, refine with CNN
        if conf >= CONFIDENCE_THRESHOLD:
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            crop = frame[y1:y2, x1:x2]
            
            if crop.size > 0:
                # CNN classification
                crop_resized = cv2.resize(crop, IMG_SIZE)
                crop_rgb = cv2.cvtColor(crop_resized, cv2.COLOR_BGR2RGB)
                tensor = transform(crop_rgb).unsqueeze(0).to(device)
                
                with torch.no_grad():
                    output = cnn(tensor)
                    probs = torch.softmax(output, dim=1)[0]
                    cnn_conf, cnn_pred = torch.max(probs, 0)
                    cnn_class = VEHICLE_CLASSES[cnn_pred.item()]
                
                # Fusion
                final_class = cnn_class if cnn_conf >= 0.6 else yolo_class
                final_conf = float(cnn_conf) if cnn_conf >= 0.6 else float(conf)
                
                detections.append({
                    'bbox': [x1, y1, x2, y2],
                    'class': final_class,
                    'confidence': final_conf,
                    'source': 'CNN' if cnn_conf >= 0.6 else 'YOLO',
                    'yolo_class': yolo_class,
                    'cnn_class': cnn_class,
                    'cnn_conf': float(cnn_conf)
                })
    
    print(f"✅ Found {len(detections)} detections\n")
    
    # Print results
    if detections:
        print("📊 Detection Results:")
        for i, det in enumerate(detections, 1):
            info = f"   {i}. {det['class']} ({det['confidence']:.2f}) - {det['source']}"
            if 'cnn_class' in det:
                info += f" [YOLO: {det['yolo_class']}, CNN: {det['cnn_class']}({det['cnn_conf']:.2f})]"
            print(info)
        print()
    
    # Draw on image
    annotated = frame.copy()
    for det in detections:
        x1, y1, x2, y2 = det['bbox']
        class_name = det['class']
        confidence = det['confidence']
        source = det['source']
        
        color = CLASS_COLORS.get(class_name, (255, 255, 255))
        
        # Draw box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        
        # Label
        label = f"{class_name}: {confidence:.2f} [{source}]"
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(annotated, (x1, y1 - label_size[1] - 10), 
                     (x1 + label_size[0], y1), color, -1)
        cv2.putText(annotated, label, (x1, y1 - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    
    # Save
    output_path = "full_pipeline_test_output.jpg"
    cv2.imwrite(output_path, annotated)
    print(f"💾 Saved annotated image to: {output_path}")
    
    print(f"\n{'='*60}")
    print("✅ Full Pipeline Test Complete!")
    print(f"{'='*60}\n")
    
    return True


def main():
    # Try to find a test image
    test_paths = [
        "yolo11_test_output.jpg",
        "dataset/test_viedos/test1_clean.mp4"
    ]
    
    image_path = None
    
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        # Extract frame from video if needed
        for path in test_paths:
            if Path(path).exists():
                if path.endswith('.mp4'):
                    cap = cv2.VideoCapture(path)
                    ret, frame = cap.read()
                    cap.release()
                    if ret:
                        temp_path = "temp_pipeline_test.jpg"
                        cv2.imwrite(temp_path, frame)
                        image_path = temp_path
                        break
                else:
                    image_path = path
                    break
    
    if not image_path:
        print("❌ No test image found. Usage: python test_full_pipeline.py <image_path>")
        return
    
    test_pipeline(image_path)
    
    # Cleanup
    if image_path == "temp_pipeline_test.jpg":
        Path(image_path).unlink(missing_ok=True)


if __name__ == "__main__":
    main()
