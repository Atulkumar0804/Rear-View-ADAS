#!/usr/bin/env python3
"""
Download UVH-26 Dataset from Hugging Face
Dataset: https://huggingface.co/datasets/iisc-aim/UVH-26
Urban Vehicles and Hazards dataset with 26 classes
"""

import os
from pathlib import Path
import sys

print("="*70)
print("📥 UVH-26 Dataset Downloader")
print("="*70)
print("\nDataset: iisc-aim/UVH-26")
print("Source: Hugging Face")
print("Classes: 26 urban vehicle and hazard classes")
print()

# Check if huggingface_hub is installed
try:
    from huggingface_hub import snapshot_download
    print("✅ huggingface_hub is installed")
except ImportError:
    print("❌ huggingface_hub not installed")
    print("\nInstalling huggingface_hub...")
    os.system("pip install huggingface_hub")
    print("\nPlease run the script again after installation.")
    sys.exit(1)

# Set download directory
SCRIPT_DIR = Path(__file__).parent.resolve()
CNN_DIR = SCRIPT_DIR.parent
DATASET_DIR = CNN_DIR / "datasets" / "UVH-26"

print(f"📁 Download directory: {DATASET_DIR}")
print()

# Create directory
DATASET_DIR.mkdir(parents=True, exist_ok=True)

# Check for authentication token
hf_token = os.environ.get('HF_TOKEN', None)

if hf_token:
    print("✅ Using HF_TOKEN from environment for authentication")
else:
    print("⚠️  No HF_TOKEN found - attempting anonymous download")
    print("   (May be rate-limited. Set HF_TOKEN environment variable to avoid this)")

print()
print("🚀 Starting download...")
print("⏳ This may take a while depending on your internet speed...")
print()

try:
    # Download the dataset with parallel downloads
    kwargs = {
        "repo_id": "iisc-aim/UVH-26",
        "repo_type": "dataset",
        "local_dir": str(DATASET_DIR),
        "local_dir_use_symlinks": False,
        "resume_download": True,
        "max_workers": 8  # Parallel downloads (8 threads)
    }
    
    # Add token if available
    if hf_token:
        kwargs["token"] = hf_token
    
    print("⚡ Using 8 parallel download threads for faster speed!")
    print()
    
    downloaded_path = snapshot_download(**kwargs)
    
    print("="*70)
    print("✅ Download Complete!")
    print("="*70)
    print(f"\n📂 Dataset saved to: {downloaded_path}")
    
    # List downloaded files
    print("\n📋 Downloaded files:")
    for item in Path(downloaded_path).rglob("*"):
        if item.is_file():
            size_mb = item.stat().st_size / (1024 * 1024)
            print(f"   • {item.name} ({size_mb:.2f} MB)")
    
    # Check for common dataset structures
    print("\n🔍 Analyzing dataset structure...")
    
    common_dirs = ['train', 'val', 'test', 'images', 'labels', 'annotations']
    found_dirs = []
    
    for dir_name in common_dirs:
        dir_path = Path(downloaded_path) / dir_name
        if dir_path.exists():
            found_dirs.append(dir_name)
            print(f"   ✅ Found: {dir_name}/")
    
    if not found_dirs:
        print("   ℹ️  No standard directories found - check dataset structure manually")
    
    print("\n" + "="*70)
    print("📖 Next Steps:")
    print("="*70)
    print(f"1. Explore the dataset: cd {DATASET_DIR}")
    print("2. Check README or documentation for dataset format")
    print("3. Convert to your required format if needed")
    print("4. Update training script to use this dataset")
    print()
    
except Exception as e:
    print("="*70)
    print("❌ Download Failed!")
    print("="*70)
    print(f"\nError: {e}")
    print("\n💡 Troubleshooting:")
    print("1. Check your internet connection")
    print("2. Verify you have enough disk space")
    print("3. Try running: huggingface-cli login")
    print("4. Visit: https://huggingface.co/datasets/iisc-aim/UVH-26")
    sys.exit(1)
