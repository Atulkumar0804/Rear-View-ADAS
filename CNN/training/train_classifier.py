from ultralytics import YOLO
import os
from pathlib import Path

# Get absolute path to CNN root
SCRIPT_DIR = Path(__file__).parent.resolve()
CNN_DIR = SCRIPT_DIR.parent

# Load a model
model = YOLO('yolo11m-cls.pt')  # load a pretrained model (recommended for training)

# Train the model
results = model.train(
    data=str(CNN_DIR / 'dataset/uvh26_cls'), 
    epochs=10, 
    imgsz=224,
    project=str(CNN_DIR / 'models/classifier'),
    name='uvh26_finetune',
    exist_ok=True
)

# Validate
metrics = model.val()
print(metrics.top1)   # top1 accuracy
print(metrics.top5)   # top5 accuracy
