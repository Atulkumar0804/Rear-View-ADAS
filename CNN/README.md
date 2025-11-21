# CNN Vehicle Detection System 🚗# CNN-Based Vehicle Detection & Distance Estimation



Real-time vehicle detection, classification, and distance estimation using YOLO + CNN fusion.Complete deep learning pipeline for rear-view vehicle detection, classification, and distance estimation using sequential frame analysis.



---## 📁 Project Structure



## 📁 **NEW: Organized Structure**```

CNN/

The folder has been reorganized for easy navigation. **See [FOLDER_STRUCTURE.md](FOLDER_STRUCTURE.md) for complete guide.**├── prepare_dataset.py          # Dataset preparation & splitting

├── train.py                    # Main training script

```├── hyperparameter_tuning.py    # Hyperparameter optimization

CNN/├── inference.py                # Inference & real-time detection

├── inference_tools/       # 📹 Real-time detection (camera + video)├── models/

├── training_tools/        # 🏋️ Model training & dataset prep│   └── architectures.py        # Model architectures (MobileNet, SqueezeNet, ResNet, LSTM)

├── video_tools/           # 🎥 Video/frames conversion├── utils/

├── scripts/               # ⚡ Quick-start scripts│   └── visualization.py        # Plotting utilities

├── documents/             # 📖 All documentation├── dataset/                    # Generated dataset folder

├── models/                # 🧠 Model architectures│   ├── train/

├── checkpoints/           # 💾 Trained models│   ├── val/

├── dataset/               # 📊 Training data│   ├── test/

└── archived/              # 🗄️ Old versions (not used)│   ├── train_sequences/

```│   ├── val_sequences/

│   └── test_sequences/

---├── checkpoints/                # Saved model weights

└── plots/                      # Training curves & visualizations

## 🚀 Quick Start```



### 1. Camera Detection (Real-time)## 🚀 Quick Start

```bash

cd scripts/### 1. Prepare Dataset

./run_camera.sh

```bash

# OR manuallycd CNN

cd inference_tools/python prepare_dataset.py

python camera_inference.py --camera 2```

```

This will:

### 2. Process Video File- Load images from `data/samples/CAM_BACK/1-9/`

```bash- Detect vehicles using computer vision techniques

cd inference_tools/- Create 70/15/15 train/val/test splits

python video_inference.py --input video.mp4 --output detected.mp4- Generate image sequences for distance estimation

```- Save processed data to `./dataset/`



### 3. Train New Model**Output**: ~400 images split into 6 classes:

```bash- car

cd training_tools/- truck

python train_transfer.py --model resnet18 --epochs 50- bus

```- motorcycle

- bicycle

### 4. Video Tools- no_vehicle

```bash

cd video_tools/### 2. Train Models

# Frames → Video

python frames_to_video.py --input frames/ --output video.mp4```bash

python train.py

# Video → Frames```

python video_to_frames.py --input video.mp4 --output frames/

```Trains multiple architectures:

- **MobileNet-Inspired**: Lightweight depthwise separable convolutions

---- **SqueezeNet-Inspired**: Fire modules for efficient inference

- **ResNet-Inspired**: Deep residual networks

## 🎯 Key Features- **Transfer Learning**: Pre-trained MobileNetV2 & ResNet18



- ✅ **YOLO + CNN Fusion** - Combines object detection + classification**Features**:

- ✅ **Distance Estimation** - Monocular depth using bounding box height- Data augmentation (rotation, flip, color jitter)

- ✅ **Person Detection** - Fixed fusion logic (100% accuracy)- Learning rate scheduling

- ✅ **Real-time Tracking** - ByteTrack integration- Early stopping

- ✅ **Transfer Learning** - ResNet18, MobileNetV2, ShuffleNetV2- Automatic checkpoint saving

- ✅ **Video Processing** - Batch video analysis with annotations- Comprehensive plotting

- ✅ **Interactive Tools** - Menu-driven scripts

### 3. Hyperparameter Tuning

---

```bash

## 📊 Model Performance# Random search (recommended)

python hyperparameter_tuning.py --method random --trials 30

| Model | Val Acc | Test Acc | Speed |

|-------|---------|----------|-------|# Grid search (exhaustive)

| **ResNet18** ⭐ | 98.31% | 94.38% | 30 FPS |python hyperparameter_tuning.py --method grid

| MobileNetV2 | 96.88% | 91.25% | 35 FPS |```

| ShuffleNetV2 | 95.63% | 90.00% | 40 FPS |

Optimizes:

**Classes**: car, truck, bus, person- Learning rate: [0.0001, 0.001, 0.01]

- Batch size: [16, 32, 64]

---- Optimizer: [Adam, SGD, RMSprop]

- Weight decay: [0, 1e-4, 1e-3]

## 📖 Documentation

### 4. Inference

All documentation is in `documents/` folder:

```bash

- **[FOLDER_STRUCTURE.md](FOLDER_STRUCTURE.md)** - 📁 Complete navigation guidepython inference.py --model checkpoints/Transfer-MobileNetV2/best_model.pth \

- **COMPLETE_DOCUMENTATION.md** - 📚 Full system documentation                    --video ../data/samples/test_video.mp4

- **CAMERA_USAGE.md** - 📹 Camera detection guide```

- **VIDEO_FRAMES_TOOLS.md** - 🎥 Video tools reference

- **PERSON_CAR_FIX.md** - 🔧 Person detection fix details## 📊 Model Architectures



---### 1. MobileNet-Inspired

- **Parameters**: ~2.2M

## 🛠️ Requirements- **Architecture**: Depthwise separable convolutions

- **Speed**: Very fast (~30 FPS)

```bash- **Use case**: Edge devices, real-time applications

pip install -r requirements.txt

```### 2. SqueezeNet-Inspired

- **Parameters**: ~1.2M

**Key dependencies:**- **Architecture**: Fire modules (squeeze + expand)

- PyTorch >= 1.9.0- **Speed**: Very fast (~35 FPS)

- torchvision- **Use case**: Memory-constrained environments

- ultralytics (YOLO)

- opencv-python### 3. ResNet-Inspired

- numpy- **Parameters**: ~11M

- pillow- **Architecture**: Residual blocks with skip connections

- **Speed**: Fast (~25 FPS)

---- **Use case**: High accuracy requirements



## 💡 Common Tasks### 4. Transfer Learning (Recommended)

- **Base**: Pre-trained on ImageNet

### Find Camera ID- **Fine-tuning**: Custom classifier head

```bash- **Speed**: 20-30 FPS

cd scripts/- **Use case**: Best accuracy with less data

python test_camera.py

```### 5. LSTM Distance Estimator

- **Input**: 5 consecutive frames

### Verify System- **Output**: [approaching, stationary, receding]

```bash- **Architecture**: CNN features + LSTM temporal modeling

cd scripts/

python verify_system.py## 📈 Expected Performance

```

| Model | Parameters | Val Acc | Test Acc | Speed (FPS) |

### Interactive Menu|-------|-----------|---------|----------|-------------|

```bash| MobileNet-Inspired | 2.2M | 85-88% | 84-87% | 30 |

cd scripts/| SqueezeNet-Inspired | 1.2M | 83-86% | 82-85% | 35 |

./menu.sh| ResNet-Inspired | 11M | 88-91% | 87-90% | 25 |

```| Transfer-MobileNetV2 | 3.5M | 90-93% | 89-92% | 28 |

| Transfer-ResNet18 | 11.7M | 91-94% | 90-93% | 22 |

### Test Model Accuracy

```bash## 🎯 Training Features

cd inference_tools/

python test_model_predictions.py### Data Augmentation

``````python

- RandomHorizontalFlip()

---- RandomRotation(10°)

- ColorJitter(brightness, contrast, saturation)

## 📁 Important Paths- RandomAffine(translate=0.1)

```

| What | Where |

|------|-------|### Regularization

| 🎥 Camera Detection | `inference_tools/camera_inference.py` |- Dropout (0.2-0.5)

| 🎬 Video Processing | `inference_tools/video_inference.py` |- Weight decay (L2 regularization)

| 🏋️ Training | `training_tools/train_transfer.py` |- Batch normalization

| 🤖 Best Model | `checkpoints/transfer_resnet18/best_model.pth` |- Early stopping (patience=10)

| 📖 Full Docs | `documents/` |

### Optimization

---- Adam/SGD/RMSprop optimizers

- ReduceLROnPlateau scheduler

## 🎓 Getting Started- Gradient clipping

- Mixed precision training (optional)

**For Beginners:**

1. Read [FOLDER_STRUCTURE.md](FOLDER_STRUCTURE.md)## 📉 Generated Plots

2. Run `scripts/verify_system.py`

3. Try `scripts/menu.sh` (interactive)All plots saved to `./plots/`:

4. Test camera: `scripts/run_camera.sh`

1. **Training History**

**For Developers:**   - Loss curves (train & val)

1. Check `documents/COMPLETE_DOCUMENTATION.md`   - Accuracy curves (train & val)

2. Train custom model: `training_tools/train_transfer.py`   - Learning rate schedule

3. Modify fusion logic: `inference_tools/camera_inference.py`   - Overfitting gap

4. Create video datasets: `video_tools/`

2. **Confusion Matrix**

---   - Per-class performance

   - Misclassification patterns

## 🔥 Recent Updates

3. **Hyperparameter Analysis**

- ✅ Organized folder structure (v2.0)   - Learning rate vs accuracy

- ✅ Added video inference with distance labels   - Batch size vs accuracy

- ✅ Fixed person detection (YOLO-first fusion)   - Optimizer comparison

- ✅ Added video/frames conversion tools   - Top N configurations

- ✅ Created interactive scripts and menus

- ✅ Comprehensive documentation4. **Sample Data**

   - Vehicle crops visualization

---   - Sequence frames

   - Detection examples

## 📞 Need Help?

## 🔧 Advanced Usage

1. **Navigation**: See [FOLDER_STRUCTURE.md](FOLDER_STRUCTURE.md)

2. **Documentation**: Check `documents/` folder### Fine-Tuning Pre-trained Models

3. **Interactive**: Run `scripts/menu.sh`

4. **System Check**: Run `scripts/verify_system.py````python

from models.architectures import create_model

---

# Load with frozen backbone

**Last Updated**: November 21, 2025  model = create_model('transfer_mobilenet', num_classes=6, freeze_backbone=True)

**Version**: 2.0 - Organized Structure  

**Status**: ✅ Production Ready# Train classifier only (fast)

train(model, epochs=10)

# Unfreeze and fine-tune
model.unfreeze_backbone(num_layers=3)  # Unfreeze last 3 layers
train(model, epochs=20, lr=0.0001)  # Lower learning rate
```

### Custom Model Configuration

```python
# Custom ResNet depth
model = ResNetInspired(num_classes=6, num_blocks=[3, 4, 6, 3])

# Custom MobileNet width
model = MobileNetInspired(num_classes=6, width_multiplier=1.5)
```

### Distance Estimation Training

```python
from train import SequenceDataset, LSTM DistanceEstimator

# Load sequence data
train_seq = SequenceDataset('./dataset', 'train')
val_seq = SequenceDataset('./dataset', 'val')

# Train LSTM model
model = LSTMDistanceEstimator(num_classes=3)
train(model, train_seq, val_seq)
```

## 🐛 Troubleshooting

### Out of Memory (OOM)
- Reduce batch size: `--batch-size 16`
- Use mixed precision training
- Smaller model: Use MobileNet or SqueezeNet

### Low Accuracy
- Increase training epochs: `--epochs 100`
- More data augmentation
- Try transfer learning
- Hyperparameter tuning

### Slow Training
- Enable GPU: Check `torch.cuda.is_available()`
- Increase `num_workers` in DataLoader
- Use smaller model
- Mixed precision training

## 📝 Requirements

```bash
torch>=2.0.0
torchvision>=0.15.0
opencv-python>=4.8.0
numpy>=1.24.0
matplotlib>=3.7.0
seaborn>=0.12.0
scikit-learn>=1.3.0
tqdm>=4.65.0
```

## 🎓 Algorithm Details

### Vehicle Detection
1. **Background Subtraction**: Gaussian blur + adaptive thresholding
2. **Contour Detection**: OpenCV findContours
3. **Size Filtering**: Area > 5000 pixels, aspect ratio checks
4. **Classification**: Based on size & aspect ratio heuristics

### Distance Estimation
1. **Bounding Box Tracking**: IoU-based matching across frames
2. **Area Change Analysis**: Growing area → approaching vehicle
3. **Temporal Smoothing**: LSTM for sequential frame analysis
4. **Labels**: 
   - `approaching`: Area increase > 10%
   - `receding`: Area decrease > 10%
   - `stationary`: Area change < 10%

## 🏆 Best Practices

1. **Start with transfer learning** - Faster convergence, better accuracy
2. **Use random search** for hyperparameter tuning (faster than grid)
3. **Monitor overfitting** - Check train/val gap
4. **Save checkpoints frequently** - Training can be interrupted
5. **Fine-tune progressively** - Freeze → unfreeze → low LR
6. **Validate on test set** only after all tuning is complete

## 📚 References

- MobileNets: [Howard et al., 2017]
- SqueezeNet: [Iandola et al., 2016]
- ResNet: [He et al., 2015]
- Transfer Learning: [Yosinski et al., 2014]

## 🤝 Contributing

This is a research project. Feel free to experiment with:
- New architectures
- Different augmentation strategies
- Advanced distance estimation algorithms
- Ensemble methods

## 📄 License

MIT License - See LICENSE file for details
