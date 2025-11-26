#!/bin/bash

echo "=================================================================="
echo "🔐 Quick Resume - UVH-26 Dataset Download"
echo "=================================================================="
echo ""
echo "Please paste your Hugging Face token (from browser):"
echo "(It starts with 'hf_...')"
echo ""
read -s HF_TOKEN
echo ""

if [ -z "$HF_TOKEN" ]; then
    echo "❌ No token provided!"
    exit 1
fi

echo "✅ Token received!"
echo ""
echo "Authenticating with Hugging Face..."

# Login using environment variable
export HF_TOKEN="$HF_TOKEN"
echo "$HF_TOKEN" | huggingface-cli login --token "$HF_TOKEN" 2>/dev/null

if [ $? -eq 0 ]; then
    echo "✅ Authentication successful!"
else
    echo "⚠️  Authentication may have issues, but trying download anyway..."
fi

echo ""
echo "📥 Resuming download..."
echo "   Current: 1.7GB (1,566 files)"
echo "   Target: ~27GB (26,652 files)"
echo ""

cd /home/atul/Desktop/atul/rear_view_adas_monocular/CNN/scripts
HF_TOKEN="$HF_TOKEN" python3 download_uvh26.py

echo ""
echo "=================================================================="
