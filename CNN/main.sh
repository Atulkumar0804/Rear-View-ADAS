#!/bin/bash
#
# Main CNN Launcher - Easy access to all features
#

PYTHON="/home/atul/Desktop/atul/rear_view_adas_monocular/.venv/bin/python"
CNN_DIR="/home/atul/Desktop/atul/rear_view_adas_monocular/CNN"
# Use mobilenet_inspired model (15 classes, 89.11% accuracy)
MODEL="$CNN_DIR/checkpoints/mobilenet_inspired/best_model.pth"
export PYTHONPATH="$CNN_DIR:$PYTHONPATH"

clear
echo "================================================================"
echo "🚗 CNN VEHICLE DETECTION - MAIN LAUNCHER"
echo "================================================================"
echo ""
echo "Select what you want to run:"
echo ""
echo "  1. 📹 Camera Detection (Real-time)"
echo "  2. 🎬 Video Processing"  
echo "  3. 🏋️  Train Models"
echo "  4. 🔍 Test Camera"
echo "  5. ❌ Exit"
echo ""
read -p "Enter choice [1-5]: " choice

case $choice in
    1)
        echo ""
        echo "📹 Starting Camera Detection..."
        echo ""
        cd "$CNN_DIR/inference_tools"
        $PYTHON camera_inference.py --camera 4 --model "$MODEL"
        ;;
    
    2)
        echo ""
        read -p "Enter video path: " video_path
        if [ ! -f "$video_path" ]; then
            echo "❌ Video not found: $video_path"
            exit 1
        fi
        
        output="${video_path%.*}_detected.mp4"
        echo ""
        echo "🎬 Processing video..."
        echo "   Input: $video_path"
        echo "   Output: $output"
        echo ""
        
        cd "$CNN_DIR/inference_tools"
        $PYTHON video_inference.py --input "$video_path" --output "$output" --model "$MODEL"
        ;;
    
    3)
        echo ""
        echo "🏋️  Starting Training..."
        echo ""
        cd "$CNN_DIR"
        
        if [ ! -d "$CNN_DIR/dataset/train" ]; then
            echo "⚠️  Dataset not found. Preparing dataset first..."
            cd "$CNN_DIR/training_tools"
            $PYTHON prepare_dataset.py
            cd "$CNN_DIR"
        fi
        
        $PYTHON training_tools/train.py
        ;;
    
    4)
        echo ""
        echo "🔍 Testing cameras..."
        echo ""
        cd "$CNN_DIR/scripts"
        $PYTHON test_camera.py
        ;;
    
    5)
        echo "👋 Goodbye!"
        exit 0
        ;;
    
    *)
        echo "❌ Invalid choice"
        exit 1
        ;;
esac
