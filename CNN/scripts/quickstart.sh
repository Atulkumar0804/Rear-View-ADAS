#!/bin/bash

# Quickstart Script for CNN Vehicle Detection Training
# Updated version for YOLO-detected dataset

set -e  # Exit on error

echo "================================================================"
echo "🚀 CNN Vehicle Detection - Quick Start"
echo "================================================================"
echo ""

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Get Python from virtual environment
PYTHON="/home/atul/Desktop/atul/rear_view_adas_monocular/.venv/bin/python"

# Step 1: Install dependencies
echo -e "${BLUE}📦 Step 1: Installing dependencies...${NC}"
$PYTHON -m pip install -q torch torchvision opencv-python numpy matplotlib seaborn scikit-learn tqdm ultralytics Pillow
echo -e "${GREEN}✅ Dependencies installed${NC}"
echo ""

# Step 2: Prepare dataset with YOLO
echo -e "${BLUE}📊 Step 2: Preparing dataset (YOLO detection)...${NC}"
cd ../training_tools
$PYTHON prepare_dataset.py
echo -e "${GREEN}✅ Dataset prepared${NC}"
echo ""

# Check if dataset was created
if [ ! -d "../dataset/train" ]; then
    echo -e "${YELLOW}⚠️  Dataset preparation failed. Exiting.${NC}"
    exit 1
fi

# Count samples
TRAIN_COUNT=$(find ../dataset/train -name "*.jpg" | wc -l)
VAL_COUNT=$(find ../dataset/val -name "*.jpg" | wc -l)
TEST_COUNT=$(find ../dataset/test -name "*.jpg" | wc -l)

echo -e "${GREEN}   Train: $TRAIN_COUNT | Val: $VAL_COUNT | Test: $TEST_COUNT${NC}"
echo ""

# Step 3: Train models
echo -e "${BLUE}🔥 Step 3: Training CNN models...${NC}"
echo -e "${YELLOW}   This will take 2-4 hours depending on your hardware${NC}"
$PYTHON train.py
echo -e "${GREEN}✅ Training complete${NC}"
echo ""

# Step 4: Optional - Hyperparameter tuning
read -p "Run hyperparameter tuning? (y/N) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo -e "${BLUE}🔧 Step 4: Hyperparameter tuning...${NC}"
    $PYTHON hyperparameter_tuning.py --method random --trials 20
    echo -e "${GREEN}✅ Hyperparameter tuning complete${NC}"
else
    echo -e "${YELLOW}⏭️  Skipping hyperparameter tuning${NC}"
fi
echo ""

# Step 5: Test inference
echo -e "${BLUE}🎬 Step 5: Testing inference...${NC}"

# Find best model
BEST_MODEL=$(find ../checkpoints -name "best_model.pth" | head -1)

if [ -z "$BEST_MODEL" ]; then
    echo -e "${YELLOW}⚠️  No trained model found. Please train first.${NC}"
else
    echo -e "${GREEN}   Using model: $BEST_MODEL${NC}"
    
    # Check if test video exists
    if [ -f "../../data/datasets/test_videos/cam_back_1.mp4" ]; then
        cd ../inference_tools
        $PYTHON video_inference.py --model "$BEST_MODEL" --input ../../data/datasets/test_videos/cam_back_1.mp4 --output output_demo.mp4
        echo -e "${GREEN}✅ Inference complete! Output saved to output_demo.mp4${NC}"
    else
        echo -e "${YELLOW}⚠️  No test video found. Skipping inference demo.${NC}"
        echo -e "${YELLOW}   Run: cd inference_tools && python video_inference.py --model $BEST_MODEL --input VIDEO.mp4${NC}"
    fi
fi
echo ""

cd ../scripts

# Summary
echo "================================================================"
echo -e "${GREEN}✅ SETUP COMPLETE!${NC}"
echo "================================================================"
echo ""
echo "📁 Generated files:"
echo "   - ../dataset/                : Prepared training data"
echo "   - ../checkpoints/            : Trained model weights"
echo "   - ../plots/                  : Training visualizations"
if [ -f "../inference_tools/output_demo.mp4" ]; then
    echo "   - output_demo.mp4            : Demo inference video"
fi
echo ""
echo "🚀 Next steps:"
echo "   1. Check visualizations: ls ../plots/"
echo "   2. Run camera: cd ../inference_tools && python camera_inference.py --camera 2"
echo "   3. Process video: cd ../inference_tools && python video_inference.py --input VIDEO.mp4 --output result.mp4"
echo ""
echo "================================================================"
