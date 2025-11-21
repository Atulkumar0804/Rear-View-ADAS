#!/bin/bash
#
# Quick Start Script for Camera Inference
# Automatically detects best model and runs camera detection
#

echo "==========================================="
echo "🚗 CAMERA VEHICLE DETECTION - QUICK START"
echo "==========================================="
echo ""

# Navigate to CNN directory
cd "$(dirname "$0")"

# Activate virtual environment
if [ -f "../.venv/bin/activate" ]; then
    echo "🔧 Activating virtual environment..."
    source ../.venv/bin/activate
else
    echo "⚠️  Virtual environment not found at ../.venv"
    echo "   Attempting to use system Python..."
fi

# Check if models exist
if [ ! -d "checkpoints" ]; then
    echo "❌ No models found! Please train models first:"
    echo "   python train_v2.py"
    exit 1
fi

# Find best model (prefer transfer_resnet18)
MODEL=""
if [ -f "checkpoints/transfer_resnet18/best_model.pth" ]; then
    MODEL="checkpoints/transfer_resnet18/best_model.pth"
    echo "✅ Using best model: transfer_resnet18 (98.31% val acc)"
elif [ -f "checkpoints/resnet_inspired/best_model.pth" ]; then
    MODEL="checkpoints/resnet_inspired/best_model.pth"
    echo "✅ Using model: resnet_inspired (97.19% val acc)"
elif [ -f "checkpoints/transfer_mobilenet/best_model.pth" ]; then
    MODEL="checkpoints/transfer_mobilenet/best_model.pth"
    echo "✅ Using model: transfer_mobilenet (94.94% val acc)"
else
    # Find any model
    MODEL=$(find checkpoints -name "best_model.pth" | head -1)
    if [ -z "$MODEL" ]; then
        echo "❌ No trained models found!"
        exit 1
    fi
    echo "✅ Using model: $MODEL"
fi

echo ""
echo "📋 Instructions:"
echo "   - Point camera at vehicles"
echo "   - Press 'q' to quit"
echo "   - Press 's' to take screenshot"
echo ""
echo "🚀 Starting camera detection..."
echo ""

# Run camera inference
python camera_inference.py --model "$MODEL"

echo ""
echo "✅ Detection session ended"
echo ""
