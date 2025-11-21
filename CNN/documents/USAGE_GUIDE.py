"""
Complete Usage Guide & Demo

Run this to see all available commands and examples.
"""

import sys

USAGE_GUIDE = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                    CNN VEHICLE DETECTION - USAGE GUIDE                       ║
╚══════════════════════════════════════════════════════════════════════════════╝

🎯 OVERVIEW
-----------
This system uses deep learning to detect and classify vehicles in rear-view
camera images, and estimate if they're approaching, stationary, or receding.

📁 PROJECT STRUCTURE
--------------------
CNN/
├── prepare_dataset.py          # Step 1: Prepare data
├── train.py                    # Step 2: Train models
├── hyperparameter_tuning.py    # Step 3: Optimize (optional)
├── inference.py                # Step 4: Run inference
├── models/                     # Model architectures
├── dataset/                    # Processed data (auto-generated)
├── checkpoints/                # Trained weights (auto-generated)
└── plots/                      # Visualizations (auto-generated)

🚀 QUICK START
--------------
1. Install dependencies:
   $ pip install -r requirements.txt

2. One-command setup (recommended):
   $ bash quickstart.sh

   This will:
   - Prepare dataset from CAM_BACK images
   - Train multiple model architectures
   - Generate comparison plots
   - Test inference on sample video

🔧 STEP-BY-STEP WORKFLOW
------------------------

STEP 1: Prepare Dataset
------------------------
$ python prepare_dataset.py

What it does:
- Loads images from ../data/samples/CAM_BACK/1-9/
- Detects vehicles using computer vision
- Extracts vehicle crops
- Creates image sequences for distance estimation
- Splits into train (70%), val (15%), test (15%)
- Generates sample visualizations

Output:
- dataset/train/: Training images
- dataset/val/: Validation images
- dataset/test/: Test images
- dataset/*_sequences/: Sequential frames for LSTM
- plots/sample_crops.png: Visualization

Expected: ~280 train, ~60 val, ~60 test images


STEP 2: Train Models
--------------------
$ python train.py

What it does:
- Trains 5 different architectures in parallel:
  1. MobileNet-Inspired (lightweight)
  2. SqueezeNet-Inspired (efficient)
  3. ResNet-Inspired (deep)
  4. Transfer-MobileNetV2 (pre-trained, recommended)
  5. Transfer-ResNet18 (pre-trained)

- Features:
  * Data augmentation (flip, rotate, color jitter)
  * Learning rate scheduling
  * Early stopping (patience=10)
  * Automatic checkpoint saving

- Training time:
  * GPU: 30-60 minutes per model
  * CPU: 2-4 hours per model

Output:
- checkpoints/*/best_model.pth: Best model weights
- checkpoints/*/checkpoint_epoch_*.pth: Periodic checkpoints
- plots/*_history.png: Training curves
- plots/*_confusion_matrix.png: Per-class performance
- plots/results_summary.json: Model comparison

Expected accuracy: 85-93% (transfer learning performs best)


STEP 3: Hyperparameter Tuning (Optional)
-----------------------------------------
$ python hyperparameter_tuning.py --method random --trials 20

What it does:
- Tests different combinations of:
  * Learning rates: [0.0001, 0.001, 0.01]
  * Batch sizes: [16, 32, 64]
  * Optimizers: [Adam, SGD, RMSprop]
  * Weight decay: [0, 1e-4, 1e-3]

Methods:
- Random search (recommended): Tests N random configurations
  $ python hyperparameter_tuning.py --method random --trials 30

- Grid search (exhaustive): Tests all combinations
  $ python hyperparameter_tuning.py --method grid

Output:
- plots/random_search_results.json: All tested configurations
- plots/random_search_visualization.png: Analysis plots

Expected time: 
- Random (20 trials): 3-5 hours on GPU
- Grid search (81 configs): 12-24 hours on GPU


STEP 4: Inference
-----------------
A) Video File:
$ python inference.py \\
    --model checkpoints/Transfer-MobileNetV2/best_model.pth \\
    --video ../data/samples/test_video.mp4 \\
    --output output_video.mp4

B) Live Camera:
$ python inference.py \\
    --model checkpoints/Transfer-MobileNetV2/best_model.pth \\
    --camera 0

C) Specify device:
$ python inference.py \\
    --model checkpoints/Transfer-MobileNetV2/best_model.pth \\
    --video test.mp4 \\
    --device cpu

What it shows:
- Bounding boxes around detected vehicles
- Vehicle class labels (car, truck, bus, etc.)
- Confidence scores
- Real-time FPS counter
- Vehicle count

Controls:
- Press 'q' to quit
- ESC to exit


📊 ANALYZING RESULTS
--------------------
All plots are saved to ./plots/:

1. Training History (*_history.png):
   - Loss curves (train & validation)
   - Accuracy curves
   - Learning rate schedule
   - Overfitting gap (train_acc - val_acc)

2. Confusion Matrix (*_confusion_matrix.png):
   - Per-class accuracy
   - Common misclassifications
   - Class balance

3. Hyperparameter Analysis:
   - Learning rate vs accuracy scatter plot
   - Batch size vs accuracy
   - Optimizer comparison bar chart
   - Top N best configurations

4. Sample Data (sample_crops.png):
   - Example vehicle crops from each class


🎯 MODEL SELECTION GUIDE
-------------------------
Choose model based on your requirements:

| Model              | Accuracy | Speed    | Size  | Use Case                    |
|--------------------|----------|----------|-------|-----------------------------|
| MobileNet-Inspired | ⭐⭐⭐    | ⭐⭐⭐⭐⭐ | Small | Edge devices, real-time     |
| SqueezeNet-Inspired| ⭐⭐     | ⭐⭐⭐⭐⭐ | Tiny  | Memory-constrained          |
| ResNet-Inspired    | ⭐⭐⭐⭐   | ⭐⭐⭐    | Large | High accuracy needed        |
| Transfer-MobileNetV2| ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐   | Small | Best overall (RECOMMENDED) |
| Transfer-ResNet18  | ⭐⭐⭐⭐⭐ | ⭐⭐⭐    | Medium| Highest accuracy            |

Recommendation: Start with Transfer-MobileNetV2 for best accuracy/speed trade-off


🔬 ADVANCED USAGE
-----------------

1. Fine-tune a pre-trained model:
   ```python
   from models.architectures import create_model
   
   # Load with frozen backbone
   model = create_model('transfer_mobilenet', num_classes=6, freeze_backbone=True)
   
   # Train classifier only (fast, 10 epochs)
   train(model, epochs=10, lr=0.001)
   
   # Unfreeze last 3 layers
   model.unfreeze_backbone(num_layers=3)
   
   # Fine-tune with low learning rate (20 epochs)
   train(model, epochs=20, lr=0.0001)
   ```

2. Custom data augmentation:
   Edit train.py, modify train_transform:
   ```python
   train_transform = transforms.Compose([
       transforms.ToPILImage(),
       transforms.RandomHorizontalFlip(p=0.5),
       transforms.RandomRotation(15),
       transforms.ColorJitter(brightness=0.3, contrast=0.3),
       transforms.RandomAffine(degrees=0, translate=(0.15, 0.15)),
       transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
       transforms.ToTensor(),
       transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
   ])
   ```

3. Train LSTM for distance estimation:
   ```bash
   # Modify train.py to use SequenceDataset
   python train_lstm.py  # (create this script based on train.py)
   ```

4. Export to ONNX for deployment:
   ```python
   import torch
   
   model.eval()
   dummy_input = torch.randn(1, 3, 224, 224)
   torch.onnx.export(model, dummy_input, "model.onnx")
   ```


🐛 TROUBLESHOOTING
------------------

Problem: Out of GPU Memory
Solution:
- Reduce batch size: Edit BATCH_SIZE in train.py
- Use smaller model: MobileNet or SqueezeNet
- Enable gradient checkpointing

Problem: Low Accuracy
Solutions:
- More training epochs: Increase NUM_EPOCHS
- Better data augmentation: Modify train_transform
- Use transfer learning (pre-trained models)
- Collect more training data
- Run hyperparameter tuning

Problem: Slow Training
Solutions:
- Check GPU is being used: torch.cuda.is_available()
- Increase batch size (if memory allows)
- Reduce image size (not recommended, affects accuracy)
- Use smaller model
- Increase num_workers in DataLoader

Problem: Model Overfitting
Symptoms: Train acc >> Val acc
Solutions:
- More data augmentation
- Increase dropout rate
- Add weight decay (L2 regularization)
- Early stopping (already enabled)
- Reduce model capacity

Problem: Inference Too Slow
Solutions:
- Use MobileNet or SqueezeNet
- Export to ONNX and use optimized runtime
- Reduce input image size
- Use GPU for inference


📚 UNDERSTANDING THE OUTPUT
---------------------------

Training Plots:
- Loss decreasing → Learning is progressing
- Val loss increasing → Overfitting (stop early)
- Train acc > Val acc → Normal, small gap is OK
- Large gap (>10%) → Overfitting, need regularization

Confusion Matrix:
- Diagonal (high numbers) → Correct predictions
- Off-diagonal → Misclassifications
- Check which classes are confused

Test Accuracy:
- 80-85%: Good for limited data
- 85-90%: Very good
- 90-95%: Excellent (transfer learning)
- <80%: Need more data or tuning


📖 REFERENCES
-------------
- MobileNets: https://arxiv.org/abs/1704.04861
- SqueezeNet: https://arxiv.org/abs/1602.07360
- ResNet: https://arxiv.org/abs/1512.03385
- Transfer Learning: https://arxiv.org/abs/1411.1792


💡 TIPS & BEST PRACTICES
-------------------------
1. Always start with transfer learning (fastest path to good results)
2. Use random search for hyperparameter tuning (faster than grid)
3. Monitor validation loss, not train loss (avoid overfitting)
4. Save checkpoints frequently (training can be interrupted)
5. Test on held-out test set only ONCE at the very end
6. Use data augmentation for small datasets
7. Visualize training curves to debug issues
8. Fine-tune with low learning rate (10x smaller)


🤝 SUPPORT
----------
- Check README.md for detailed documentation
- Review plots/ folder for training visualizations
- Examine checkpoints/ for saved models
- Read model architectures in models/architectures.py


✅ NEXT STEPS
-------------
After training completes:
1. Check plots/*_history.png for training curves
2. Review plots/*_confusion_matrix.png for per-class performance
3. Compare models in plots/results_summary.json
4. Test inference on your own videos/camera
5. Deploy best model to production

Good luck! 🚀

╚══════════════════════════════════════════════════════════════════════════════╝
"""

def main():
    print(USAGE_GUIDE)

if __name__ == "__main__":
    main()
