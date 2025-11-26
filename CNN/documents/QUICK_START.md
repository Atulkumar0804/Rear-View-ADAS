# Quick Reference: Optimized Training Configuration

## Training Command
```bash
cd /home/atul/Desktop/atul/rear_view_adas_monocular/CNN/training_tools
PYTHONPATH=/home/atul/Desktop/atul/rear_view_adas_monocular/CNN \
    nohup /home/atul/Desktop/atul/rear_view_adas_monocular/.venv/bin/python \
    -u train.py > training_optimized.log 2>&1 &

# Get PID
echo $!
```

## Monitor Commands
```bash
# Live progress
tail -f training_optimized.log

# GPU usage
watch -n 1 nvidia-smi

# Learning rate pattern
grep "Learning Rate" training_optimized.log | tail -20

# Validation accuracy trend
grep "Val Acc" training_optimized.log | tail -20

# Check if training is still running
ps aux | grep train.py
```

## Configuration at a Glance

```
Batch Size:       128
Epochs:           100
Learning Rate:    0.001 → 1e-6 (CosineAnnealingWarmRestarts)
Optimizer:        AdamW (weight_decay=1e-4)
Scheduler:        T_0=10, T_mult=2, eta_min=1e-6
Early Stopping:   15 epochs patience, monitor val_acc
Label Smoothing:  0.1
Gradient Clip:    1.0

Dropout:
- MobileNet:      0.4
- SqueezeNet:     0.5
- ResNet:         0.5

Data Augmentation:
✅ HorizontalFlip (p=0.5)
✅ Rotation (20°)
✅ ColorJitter (0.4/0.4/0.3/0.15)
✅ Affine (translate=0.15, scale=0.85-1.15)
✅ Perspective (distortion=0.2, p=0.3)
✅ Grayscale (p=0.1)
✅ RandomErasing (p=0.3, 2-15%)
```

## Expected Timeline
- **Time per epoch:** ~2-3 minutes
- **Total epochs:** 100 (may stop early)
- **Per model:** ~4-5 hours
- **All 3 models:** ~12-15 hours

## Success Metrics
- ✅ Val Acc > 92% (MobileNet)
- ✅ Val Acc > 89% (SqueezeNet)
- ✅ Val Acc > 94% (ResNet)
- ✅ No NaN/Inf losses
- ✅ Smooth learning curves
- ✅ GPU utilization >80%

## Files to Review After Training
```
CNN/
├── checkpoints/
│   ├── mobilenet_inspired/best_model.pth
│   ├── squeezenet_inspired/best_model.pth
│   └── resnet_inspired/best_model.pth
├── plots/
│   ├── mobilenet_inspired_history.png
│   ├── mobilenet_inspired_confusion_matrix.png
│   ├── squeezenet_inspired_history.png
│   ├── squeezenet_inspired_confusion_matrix.png
│   ├── resnet_inspired_history.png
│   └── resnet_inspired_confusion_matrix.png
└── training_tools/
    └── training_optimized.log
```

## Troubleshooting

**If loss is NaN:**
- Reduce learning rate to 0.0005
- Check for corrupt images in dataset
- Verify batch size fits in GPU memory

**If accuracy plateaus early (<85%):**
- Check data quality and labels
- Verify augmentation isn't too aggressive
- Review class imbalance

**If training is too slow (>5 min/epoch):**
- Increase NUM_WORKERS (currently 8)
- Check CPU bottleneck with `htop`
- Verify data is on fast SSD, not HDD

**If GPU OOM (Out of Memory):**
- Reduce batch size: 128 → 96 → 64
- Clear GPU cache: `torch.cuda.empty_cache()`
- Check no other processes using GPU

## Key Papers Referenced
1. AdamW: Loshchilov & Hutter (ICLR 2019)
2. Cosine Annealing: Loshchilov & Hutter (ICLR 2017)
3. Label Smoothing: Szegedy et al. (CVPR 2016)

---
**Status:** ✅ Ready to train
**Last Updated:** 2025-11-22
