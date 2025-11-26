#!/bin/bash

# Authenticate with Hugging Face and Resume UVH-26 Download
# ========================================================

echo "🔐 Hugging Face Authentication Setup"
echo "===================================="
echo ""
echo "You've been rate-limited by Hugging Face. Let's fix this!"
echo ""
echo "📝 Steps to get your HF Token:"
echo "   1. Visit: https://huggingface.co/settings/tokens"
echo "   2. Login or create account (free)"
echo "   3. Click 'New token' button"
echo "   4. Name: 'dataset-download'"
echo "   5. Type: Select 'Read'"
echo "   6. Click 'Generate token'"
echo "   7. Copy the token (starts with 'hf_...')"
echo ""
echo "────────────────────────────────────────────────────────"
echo ""
read -p "Paste your HF token here (or press Ctrl+C to cancel): " HF_TOKEN
echo ""

if [ -z "$HF_TOKEN" ]; then
    echo "❌ No token provided. Exiting..."
    exit 1
fi

# Validate token format
if [[ ! $HF_TOKEN =~ ^hf_ ]]; then
    echo "⚠️  Warning: Token doesn't start with 'hf_'"
    read -p "   Continue anyway? (y/n): " confirm
    if [ "$confirm" != "y" ]; then
        echo "❌ Cancelled"
        exit 1
    fi
fi

echo "✅ Token received!"
echo ""
echo "📊 Current Download Status:"
cd /home/atul/Desktop/atul/rear_view_adas_monocular/CNN/datasets/UVH-26/
CURRENT_SIZE=$(du -sh . 2>/dev/null | cut -f1)
CURRENT_FILES=$(find . -type f 2>/dev/null | wc -l)
echo "   Size: $CURRENT_SIZE"
echo "   Files: $CURRENT_FILES / 26,652"
echo "   Progress: $(echo "scale=1; $CURRENT_FILES * 100 / 26652" | bc)%"
echo ""
echo "🚀 Starting authenticated download with 8 parallel threads..."
echo "────────────────────────────────────────────────────────"
echo ""

# Run download with authentication
cd /home/atul/Desktop/atul/rear_view_adas_monocular/CNN/scripts
HF_TOKEN="$HF_TOKEN" /usr/bin/python3 download_uvh26.py

echo ""
echo "✅ Download complete or interrupted"
echo ""
echo "📊 Final Status:"
cd /home/atul/Desktop/atul/rear_view_adas_monocular/CNN/datasets/UVH-26/
FINAL_SIZE=$(du -sh . 2>/dev/null | cut -f1)
FINAL_FILES=$(find . -type f 2>/dev/null | wc -l)
echo "   Size: $FINAL_SIZE"
echo "   Files: $FINAL_FILES / 26,652"
echo "   Progress: $(echo "scale=1; $FINAL_FILES * 100 / 26652" | bc)%"
