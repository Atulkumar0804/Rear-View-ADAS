# Optimized Hyperparameters for UVH-26 Training

## Training Configuration (252K samples on RTX A6000)

### Core Hyperparameters

| Parameter | Value | Justification |
|-----------|-------|---------------|
| **Batch Size** | 128 | Large batch for stable gradients with 48GB VRAM, better gradient estimates |
| **Epochs** | 100 | More epochs for large dataset convergence |
| **Learning Rate** | 0.001 | Standard Adam/AdamW starting LR, adjusted by scheduler |
| **Weight Decay** | 1e-4 | L2 regularization to prevent overfitting |
| **Label Smoothing** | 0.1 | Improves generalization on large datasets |
| **Gradient Clipping** | 1.0 | Prevents exploding gradients, ensures stable training |

### Optimizer & Scheduler

**Optimizer:** `AdamW`
- Better weight decay implementation than Adam
- Decouples weight decay from gradient updates
- More effective regularization

**Learning Rate Scheduler:** `CosineAnnealingWarmRestarts`
- **T_0 = 10** epochs (initial restart period)
- **T_mult = 2** (restart period doubles each time: 10→20→40)
- **eta_min = 1e-6** (minimum LR to avoid complete stagnation)
- **Benefits:**
  - Periodic learning rate restarts help escape local minima
  - Cosine annealing provides smooth LR decay
  - Better than ReduceLROnPlateau for large datasets
  - Steps per batch for fine-grained control

### Early Stopping

| Parameter | Value | Reasoning |
|-----------|-------|-----------|
| **Patience** | 15 epochs | Longer patience for large dataset fluctuations |
| **Min Delta** | 0.0005 | Small threshold for validation accuracy improvement |
| **Mode** | 'max' | Monitor validation accuracy (higher is better) |

### Regularization

| Technique | Value/Config | Purpose |
|-----------|--------------|---------|
| **Dropout (MobileNet)** | 0.4 | Prevent co-adaptation in lightweight model |
| **Dropout (SqueezeNet)** | 0.5 | Standard dropout for fire modules |
| **Dropout (ResNet)** | 0.5 | Regularize deep model before final layer |
| **Label Smoothing** | 0.1 | Soft targets reduce overconfidence |
| **Weight Decay** | 1e-4 | L2 penalty on weights |
| **Gradient Clipping** | 1.0 | Prevent exploding gradients |

### Data Augmentation (Enhanced)

```python
train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(p=0.5),          # Mirror vehicles
    transforms.RandomRotation(20),                   # +5° more aggressive
    transforms.ColorJitter(                          # Stronger color variations
        brightness=0.4,  # ±40% brightness
        contrast=0.4,    # ±40% contrast
        saturation=0.3,  # ±30% saturation
        hue=0.15         # ±15% hue shift
    ),
    transforms.RandomAffine(                         # More geometric variation
        degrees=0,
        translate=(0.15, 0.15),  # ±15% translation
        scale=(0.85, 1.15)       # 85-115% scaling
    ),
    transforms.RandomPerspective(                    # NEW: perspective transform
        distortion_scale=0.2,
        p=0.3                    # 30% chance
    ),
    transforms.RandomGrayscale(p=0.1),              # NEW: 10% grayscale
    transforms.ToTensor(),
    transforms.Normalize(                            # ImageNet normalization
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
    transforms.RandomErasing(                        # NEW: occlusion robustness
        p=0.3,
        scale=(0.02, 0.15)       # Erase 2-15% of image
    )
])
```

### Loss Function

**CrossEntropyLoss with Label Smoothing**
```python
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
```
- Prevents overconfident predictions
- Improves calibration on large datasets
- Better generalization to unseen data

## Training Schedule Overview

### Per-Epoch Timeline (Estimate)
- **Batch Size:** 128
- **Train Batches:** 252,723 / 128 ≈ 1,975 batches/epoch
- **Val Batches:** 63,400 / 128 ≈ 495 batches/epoch
- **Time per Epoch:** ~2-3 minutes (RTX A6000)
- **Total Time per Model:** 100 epochs × 2.5 min ≈ 4-5 hours

### Learning Rate Schedule (CosineAnnealingWarmRestarts)

| Epochs | LR Range | Notes |
|--------|----------|-------|
| 1-10 | 1e-3 → 1e-6 | First cosine cycle (T_0=10) |
| 11-30 | 1e-3 → 1e-6 | Second cycle (T_0×2=20 epochs) |
| 31-70 | 1e-3 → 1e-6 | Third cycle (T_0×4=40 epochs) |
| 71-100 | Continues | May trigger early stopping before completion |

Each restart gives model chance to escape local minima with fresh momentum.

## Model-Specific Configurations

### MobileNetInspired
- **Parameters:** ~3.2M
- **Dropout:** 0.4 (after avgpool, before classifier)
- **Best For:** Speed, mobile deployment
- **Training Time:** ~4 hours

### SqueezeNetInspired
- **Parameters:** ~1.2M
- **Dropout:** 0.5 (in classifier conv)
- **Best For:** Smallest model size, edge devices
- **Training Time:** ~3 hours

### ResNetInspired
- **Parameters:** ~11M
- **Dropout:** 0.5 (before final FC layer)
- **Best For:** Highest accuracy, desktop/server
- **Training Time:** ~5 hours

## Expected Results

### Performance Targets (on UVH-26)

| Model | Expected Val Acc | Expected Test Acc | Notes |
|-------|------------------|-------------------|-------|
| **MobileNetInspired** | 92-95% | 91-94% | Best speed/accuracy tradeoff |
| **SqueezeNetInspired** | 89-92% | 88-91% | Smallest, fastest |
| **ResNetInspired** | 94-97% | 93-96% | Highest accuracy, slowest |

### Key Metrics to Monitor

1. **Validation Accuracy** - Primary metric for early stopping
2. **Training Loss** - Should decrease smoothly
3. **Learning Rate** - Should show cosine pattern with restarts
4. **Gradient Norm** - Should stay below clipping threshold (1.0)
5. **Per-Class Accuracy** - Check for class imbalance issues

### Warning Signs

⚠️ **Overfitting Indicators:**
- Train acc >> Val acc (>5% gap)
- Val loss increasing while train loss decreases
- Solution: Increase dropout, weight decay, or augmentation

⚠️ **Underfitting Indicators:**
- Both train and val acc plateau below 85%
- Solution: Increase model capacity or reduce regularization

⚠️ **Instability:**
- Loss spikes or NaN values
- Solution: Reduce LR, increase gradient clipping, check data quality

## Monitoring Commands

```bash
# Watch live training progress
tail -f /home/atul/Desktop/atul/rear_view_adas_monocular/CNN/training_tools/training.log

# Check GPU usage
watch -n 1 nvidia-smi

# Monitor process
ps aux | grep train.py

# Check learning rate and metrics
grep -E "Learning Rate|Val Acc" training.log | tail -20
```

## Post-Training Analysis

After training completes:

1. **Compare Models:** Review final results table
2. **Check Confusion Matrices:** Identify misclassification patterns
3. **Analyze Per-Class Performance:** Focus on smallest classes (Mini-bus, Other)
4. **Learning Curves:** Look for overfitting or underfitting patterns
5. **Select Best Model:** Based on test accuracy and deployment constraints

## Production Deployment

**Recommended Model Selection:**
- **Edge Device (Jetson):** SqueezeNetInspired (smallest, fastest)
- **Embedded PC:** MobileNetInspired (best tradeoff)
- **Server/Cloud:** ResNetInspired (highest accuracy)

## References

- **AdamW Paper:** Decoupled Weight Decay Regularization (Loshchilov & Hutter, 2019)
- **Cosine Annealing:** SGDR: Stochastic Gradient Descent with Warm Restarts (Loshchilov & Hutter, 2017)
- **Label Smoothing:** Rethinking the Inception Architecture (Szegedy et al., 2016)
- **Data Augmentation:** Practical recommendations for gradient-based training (Bengio, 2012)

---

**Last Updated:** 2025-11-22
**Configuration File:** `/CNN/training_tools/train.py`
**Model Definitions:** `/CNN/models/architectures.py`
