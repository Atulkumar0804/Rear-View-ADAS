# Adding Person Class to UVH-26 Training

## Overview
The training pipeline now supports 15 classes:
- **14 Vehicle Classes** from UVH-26: Hatchback, Sedan, SUV, MUV, Bus, Truck, Three-wheeler, Two-wheeler, LCV, Mini-bus, Tempo-traveller, Bicycle, Van, Other
- **1 Person Class**: For pedestrian detection

## Current Status
✅ Model architectures updated to support 15 classes
✅ Dataset loader supports Person class
✅ Training script ready

## Option 1: Train Without Person Class (Current)
The model will train on 14 vehicle classes from UVH-26 dataset. Person detection can be handled separately by YOLO during inference.

```bash
cd /home/atul/Desktop/atul/rear_view_adas_monocular/CNN/training_tools
python train.py
```

## Option 2: Add Person Class Before Training

### Step 1: Generate Person Crops Using YOLO

```bash
cd /home/atul/Desktop/atul/rear_view_adas_monocular/CNN/scripts

# Option A: Use default YOLOv8n (will auto-download)
python add_person_class.py \
    --yolo_model yolov8n.pt \
    --uvh26_root ../datasets/UVH-26 \
    --output_dir ../datasets/person_crops \
    --conf 0.5

# Option B: Use your custom YOLO model
python add_person_class.py \
    --yolo_model ../models/yolo/yolov8n_RearView.pt \
    --uvh26_root ../datasets/UVH-26 \
    --output_dir ../datasets/person_crops \
    --conf 0.3
```

This will:
- Scan all UVH-26 images (train + val)
- Detect persons using YOLO
- Extract person crops with bounding boxes
- Save to `datasets/person_crops/train/Person/` and `datasets/person_crops/val/Person/`

### Step 2: Train with 15 Classes

Once person crops are generated, the training script will automatically include them:

```bash
cd /home/atul/Desktop/atul/rear_view_adas_monocular/CNN/training_tools
python train.py
```

The dataset loader will detect person crops and add them as the 15th class.

## Expected Results

### Training Output (with Person class):
```
✅ Loaded 180000 valid train samples (skipped 5000 missing files)
✅ Added 15000 Person samples from datasets/person_crops/train/Person
📊 Class distribution:
   Hatchback         :  24196 samples
   Sedan             :  12799 samples
   ...
   Person            :  15000 samples
```

### Training Output (without Person class):
```
✅ Loaded 180000 valid train samples (skipped 5000 missing files)
📊 Class distribution:
   Hatchback         :  24196 samples
   Sedan             :  12799 samples
   ...
   Person            :      0 samples (run add_person_class.py)
```

## Model Architecture Support
All three CNN architectures support 15 classes:
- `mobilenet_inspired` - Default 15 classes
- `squeezenet_inspired` - Default 15 classes  
- `resnet_inspired` - Default 15 classes

## Inference with Person Class

After training with 15 classes, use the model for inference:

```python
from models.architectures import create_model
import torch

# Load trained model
model = create_model('mobilenet_inspired', num_classes=15)
checkpoint = torch.load('checkpoints/mobilenet_inspired/best_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])

# Classes
classes = [
    'Hatchback', 'Sedan', 'SUV', 'MUV', 'Bus', 'Truck', 
    'Three-wheeler', 'Two-wheeler', 'LCV', 'Mini-bus', 
    'Tempo-traveller', 'Bicycle', 'Van', 'Other', 'Person'
]

# Run inference
output = model(image)
pred_class = classes[output.argmax()]
```

## Notes
- Person class extraction from UVH-26 using YOLO is optional
- Model trains fine with 14 classes if Person samples are not available
- You can add person samples from any other dataset (COCO, custom, etc.)
- Person crops should be placed in `datasets/person_crops/{train,val}/Person/`
