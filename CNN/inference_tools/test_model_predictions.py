#!/usr/bin/env python3
"""
Test CNN model predictions on sample images to debug person vs car confusion
"""

import torch
from torchvision import transforms
import cv2
import numpy as np
from pathlib import Path
import sys

sys.path.append('.')
from models.architectures import create_model

# Vehicle classes
VEHICLE_CLASSES = ['car', 'truck', 'bus', 'person']

# Transform
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def test_model_on_dataset(model_path, num_samples=10):
    """Test model predictions on actual dataset samples"""
    
    print("="*60)
    print("🔍 CNN MODEL PREDICTION TEST")
    print("="*60)
    print()
    
    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    checkpoint = torch.load(model_path, map_location=device)
    model_name = Path(model_path).parent.name
    
    model = create_model(model_name, num_classes=4, pretrained=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"Model: {model_name}")
    print()
    
    # Test on samples from each class
    dataset_dir = Path('dataset/test')
    
    for class_name in VEHICLE_CLASSES:
        class_dir = dataset_dir / class_name
        
        if not class_dir.exists():
            print(f"⚠️  {class_name}: directory not found")
            continue
        
        images = list(class_dir.glob('*.jpg'))[:num_samples]
        
        print(f"\n{class_name.upper()} class ({len(images)} samples):")
        print("-" * 60)
        
        correct = 0
        predictions = {cls: 0 for cls in VEHICLE_CLASSES}
        
        for img_path in images:
            # Load and preprocess
            img = cv2.imread(str(img_path))
            img_resized = cv2.resize(img, (224, 224))
            img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
            img_tensor = transform(img_rgb).unsqueeze(0).to(device)
            
            # Predict
            with torch.no_grad():
                outputs = model(img_tensor)
                probs = torch.softmax(outputs, dim=1)
                conf, pred = torch.max(probs, 1)
            
            pred_class = VEHICLE_CLASSES[pred.item()]
            confidence = conf.item()
            
            predictions[pred_class] += 1
            
            if pred_class == class_name:
                correct += 1
                status = "✅"
            else:
                status = "❌"
            
            # Show all class probabilities
            all_probs = probs[0].cpu().numpy()
            prob_str = " | ".join([f"{cls}:{p:.2f}" for cls, p in zip(VEHICLE_CLASSES, all_probs)])
            
            print(f"  {status} {img_path.name}: {pred_class} ({confidence:.3f}) [{prob_str}]")
        
        accuracy = (correct / len(images) * 100) if images else 0
        print(f"\n  Accuracy: {correct}/{len(images)} ({accuracy:.1f}%)")
        print(f"  Predictions: {predictions}")
    
    print("\n" + "="*60)
    print("✅ Test complete!")
    print("="*60)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Test CNN model predictions')
    parser.add_argument('--model', type=str,
                       default='checkpoints/transfer_resnet18/best_model.pth',
                       help='Path to model checkpoint')
    parser.add_argument('--samples', type=int, default=10,
                       help='Number of samples per class')
    
    args = parser.parse_args()
    
    if not Path(args.model).exists():
        print(f"❌ Model not found: {args.model}")
        print("\nAvailable models:")
        for model in Path('checkpoints').glob('*/best_model.pth'):
            print(f"  {model}")
        sys.exit(1)
    
    test_model_on_dataset(args.model, args.samples)
