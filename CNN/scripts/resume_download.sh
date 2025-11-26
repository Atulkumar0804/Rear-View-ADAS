#!/bin/bash

echo "==================================================================="
echo "📥 UVH-26 Dataset Download - Resume Instructions"
echo "==================================================================="
echo ""
echo "Your download was rate-limited after downloading 1.7GB (1,566 files)"
echo ""
echo "To resume, you need a FREE Hugging Face account:"
echo ""
echo "STEP 1: Create account (if you don't have one)"
echo "   Visit: https://huggingface.co/join"
echo ""
echo "STEP 2: Get your access token"
echo "   1. Go to: https://huggingface.co/settings/tokens"
echo "   2. Click 'New token'"
echo "   3. Name it: 'dataset-download'"
echo "   4. Type: 'Read'"
echo "   5. Copy the token"
echo ""
echo "STEP 3: Login with your token"
echo "   Run: huggingface-cli login"
echo "   Paste your token when prompted"
echo ""
echo "STEP 4: Resume download"
echo "   cd CNN/scripts"
echo "   python3 download_uvh26.py"
echo ""
echo "==================================================================="
echo ""
read -p "Do you want to login now? (y/n): " answer

if [ "$answer" = "y" ]; then
    echo ""
    echo "Running: huggingface-cli login"
    echo ""
    huggingface-cli login
    
    if [ $? -eq 0 ]; then
        echo ""
        echo "✅ Login successful! Resuming download..."
        echo ""
        cd /home/atul/Desktop/atul/rear_view_adas_monocular/CNN/scripts
        python3 download_uvh26.py
    fi
else
    echo ""
    echo "No problem! Run this when you're ready:"
    echo "   bash resume_download.sh"
fi
