#!/bin/bash
#
# Simple Camera Detection Runner
# Usage: ./run_camera_simple.sh [camera_id]
#

PYTHON="/home/atul/Desktop/atul/rear_view_adas_monocular/.venv/bin/python"
CNN_DIR="/home/atul/Desktop/atul/rear_view_adas_monocular/CNN"
CAMERA_ID="${1:-2}"
MODEL="$CNN_DIR/checkpoints/transfer_resnet18/best_model.pth"

echo "========================================"
echo "🚗 CAMERA DETECTION"
echo "========================================"
echo ""
echo "📹 Camera ID: $CAMERA_ID"
echo "🤖 Model: transfer_resnet18"
echo ""

if [ ! -f "$MODEL" ]; then
    echo "❌ Model not found: $MODEL"
    echo ""
    echo "Available models:"
    find "$CNN_DIR/checkpoints" -name "best_model.pth" 2>/dev/null
    exit 1
fi

cd "$CNN_DIR/inference_tools"
$PYTHON camera_inference.py --camera "$CAMERA_ID" --model "$MODEL"
