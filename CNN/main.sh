#!/bin/bash
#
# Main CNN Launcher - Easy access to all features
#

PYTHON="/home/atul/Desktop/atul/rear_view_adas_monocular/.venv/bin/python"
CNN_DIR="/home/atul/Desktop/atul/rear_view_adas_monocular/CNN"
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
    echo "  4. 🌐 Launch Web Interface"
    echo "  5. ❌ Exit"
    echo ""
    read -p "Enter choice [1-5]: " choice

    case $choice in
        1)
            echo ""
            echo "📹 Starting Camera Detection..."
            echo ""
            cd "$CNN_DIR/inference"
            $PYTHON camera_inference.py --camera 4
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
            
            cd "$CNN_DIR/inference"
            $PYTHON video_inference.py --input "$video_path" --output "$output"
            ;;
        
        3)
            echo ""
            echo "🏋️  Starting Training..."
            echo ""
            cd "$CNN_DIR/training"
            $PYTHON train_classifier.py
            ;;

        4)
            echo ""
            echo "🌐 Starting Web Interface..."
            echo ""
            streamlit run "$CNN_DIR/interface/app.py"
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