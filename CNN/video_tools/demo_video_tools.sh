#!/bin/bash
#
# Demo: Video/Frames Conversion Tools
#

echo "=========================================="
echo "🎬 VIDEO/FRAMES TOOLS - DEMO"
echo "=========================================="
echo ""

cd "$(dirname "$0")"

# Activate virtual environment
if [ -f "../.venv/bin/activate" ]; then
    source ../.venv/bin/activate
fi

# Check if test data exists
if [ ! -d "../data/samples/CAM_BACK/1" ]; then
    echo "❌ Test data not found at ../data/samples/CAM_BACK/1"
    echo "   Using alternative test method..."
    echo ""
fi

echo "Select demo:"
echo ""
echo "1. 📹 Create video from image frames"
echo "2. 📸 Extract frames from video"
echo "3. 🔄 Convert CAM_BACK scene to video"
echo "4. 📊 Show help for frames_to_video"
echo "5. 📊 Show help for video_to_frames"
echo "6. ❌ Exit"
echo ""
read -p "Enter choice [1-6]: " choice

case $choice in
    1)
        echo ""
        echo "📹 Creating video from image frames..."
        echo ""
        
        read -p "Enter frames folder path: " frames_folder
        read -p "Enter output video name (default: output.mp4): " output_name
        output_name=${output_name:-output.mp4}
        read -p "Enter FPS (default: 30): " fps
        fps=${fps:-30}
        
        python frames_to_video.py --input "$frames_folder" --output "$output_name" --fps "$fps"
        ;;
        
    2)
        echo ""
        echo "📸 Extracting frames from video..."
        echo ""
        
        read -p "Enter video file path: " video_file
        read -p "Enter output folder (default: frames/): " output_folder
        output_folder=${output_folder:-frames/}
        read -p "Extract every Nth frame (default: 1 = all): " skip
        skip=${skip:-1}
        
        python video_to_frames.py --input "$video_file" --output "$output_folder" --skip "$skip"
        ;;
        
    3)
        echo ""
        echo "🔄 Converting CAM_BACK scene to video..."
        echo ""
        
        # Use scene 1 as example
        if [ -d "../data/samples/CAM_BACK/1" ]; then
            echo "Using CAM_BACK/1 (40 frames)..."
            python frames_to_video.py \
                --input ../data/samples/CAM_BACK/1 \
                --output cam_back_scene1.mp4 \
                --fps 10 \
                --quality 95
            
            echo ""
            echo "✅ Video created: cam_back_scene1.mp4"
            echo "   You can play it with: vlc cam_back_scene1.mp4"
        else
            echo "❌ CAM_BACK/1 not found"
            echo "   Please provide your own frames folder"
        fi
        ;;
        
    4)
        echo ""
        python frames_to_video.py --help
        ;;
        
    5)
        echo ""
        python video_to_frames.py --help
        ;;
        
    6)
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
