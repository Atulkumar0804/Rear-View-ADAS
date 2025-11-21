#!/bin/bash
#
# Quick Start Guide - All CNN Operations
#

echo "========================================"
echo "🚗 CNN VEHICLE DETECTION - QUICK START"
echo "========================================"
echo ""

# Get Python from virtual environment
PYTHON="/home/atul/Desktop/atul/rear_view_adas_monocular/.venv/bin/python"

cd "$(dirname "$0")"

# Function to check if model exists
check_models() {
    if [ -d "../checkpoints" ] && [ -n "$(find ../checkpoints -name 'best_model.pth' 2>/dev/null)" ]; then
        return 0
    else
        return 1
    fi
}

# Main menu
echo "Select an option:"
echo ""
echo "1. 📷 Run Camera Detection (Real-time)"
echo "2. 🎥 Run Video Detection"
echo "3. 🏋️  Train Models"
echo "4. 🔍 Test Camera Availability"
echo "5. 📊 View Training Results"
echo "6. 🧹 Clean Deprecated Files"
echo "7. ❌ Exit"
echo ""
read -p "Enter choice [1-7]: " choice

case $choice in
    1)
        # Camera detection
        if ! check_models; then
            echo "❌ No models found! Please train first (option 3)"
            exit 1
        fi
        
        echo ""
        echo "📷 Starting camera detection..."
        echo ""
        
        # Find camera
        read -p "Enter camera ID (default: 2): " camera_id
        camera_id=${camera_id:-2}
        
        # Find best model
        if [ -f "../checkpoints/transfer_resnet18/best_model.pth" ]; then
            MODEL="../checkpoints/transfer_resnet18/best_model.pth"
        else
            MODEL=$(find ../checkpoints -name "best_model.pth" | head -1)
        fi
        
        cd ../inference_tools
        $PYTHON camera_inference.py --camera "$camera_id" --model "$MODEL"
        ;;
        
    2)
        # Video detection
        if ! check_models; then
            echo "❌ No models found! Please train first (option 3)"
            exit 1
        fi
        
        echo ""
        read -p "Enter video path: " video_path
        
        if [ ! -f "$video_path" ]; then
            echo "❌ Video file not found: $video_path"
            exit 1
        fi
        
        if [ -f "../checkpoints/transfer_resnet18/best_model.pth" ]; then
            MODEL="../checkpoints/transfer_resnet18/best_model.pth"
        else
            MODEL=$(find ../checkpoints -name "best_model.pth" | head -1)
        fi
        
        cd ../inference_tools
        $PYTHON video_inference.py --model "$MODEL" --input "$video_path" --output detected_output.mp4
        ;;
        
    3)
        # Training
        echo ""
        echo "🏋️  Training models..."
        echo ""
        
        if [ ! -d "../dataset/train" ]; then
            echo "⚠️  Dataset not found. Preparing dataset first..."
            cd ../training_tools
            $PYTHON prepare_dataset.py
        fi
        
        cd ../training_tools
        $PYTHON train.py
        ;;
        
    4)
        # Test camera
        echo ""
        cd ../scripts
        $PYTHON test_camera.py
        ;;
        
    5)
        # View results
        echo ""
        echo "📊 Training Results:"
        echo ""
        
        if [ -d "../checkpoints" ]; then
            for model_dir in ../checkpoints/*/; do
                model_name=$(basename "$model_dir")
                if [ -f "${model_dir}best_model.pth" ]; then
                    echo "✅ $model_name"
                    
                    # Try to extract accuracy from checkpoint
                    if [ -f "${model_dir}training_log.json" ]; then
                        echo "   Training log found"
                    fi
                    echo ""
                fi
            done
        else
            echo "❌ No checkpoints found. Train models first."
        fi
        
        echo "View plots:"
        if [ -d "../plots" ]; then
            ls -lh ../plots/
        fi
        ;;
        
    6)
        # Clean deprecated files
        echo ""
        echo "🧹 Cleaning deprecated files..."
        echo ""
        echo "The archived/ folder contains old versions:"
        if [ -d "../archived" ]; then
            ls -lh ../archived/
        fi
        echo ""
        read -p "Remove archived folder? [y/N]: " confirm
        
        if [ "$confirm" = "y" ] || [ "$confirm" = "Y" ]; then
            rm -rf ../archived
            echo "✅ Archived files removed"
        else
            echo "❌ Cancelled"
        fi
        ;;
        
    7)
        echo "👋 Goodbye!"
        exit 0
        ;;
        
    *)
        echo "❌ Invalid choice"
        exit 1
        ;;
esac

echo ""
echo "✅ Done!"
echo ""
