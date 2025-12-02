import json
import os
import cv2
import random
from pathlib import Path
from tqdm import tqdm
import shutil

# Paths
DATASET_ROOT = Path('/home/atul/Desktop/atul/rear_view_adas_monocular/CNN/datasets/UVH-26')
TRAIN_JSON = DATASET_ROOT / 'UVH-26-Train/UVH-26-MV-Train.json'
VAL_JSON = DATASET_ROOT / 'UVH-26-Val/UVH-26-MV-Val.json'
TRAIN_IMG_DIR = DATASET_ROOT / 'UVH-26-Train/data'
VAL_IMG_DIR = DATASET_ROOT / 'UVH-26-Val/data'

OUTPUT_DIR = Path('/home/atul/Desktop/atul/rear_view_adas_monocular/CNN/datasets/uvh26_cls')

def setup_directories():
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    
    for split in ['train', 'val', 'test']:
        (OUTPUT_DIR / split).mkdir(parents=True, exist_ok=True)

def index_images(root_dir):
    print(f"Indexing images in {root_dir}...")
    image_map = {}
    # Search recursively for images
    for ext in ['*.jpg', '*.png', '*.jpeg']:
        for img_path in root_dir.rglob(ext):
            image_map[img_path.name] = str(img_path)
    return image_map

def process_split(json_path, img_map, split_name, target_split_names=None):
    print(f"Processing {json_path}...")
    with open(json_path, 'r') as f:
        data = json.load(f)
        
    categories = {cat['id']: cat['name'] for cat in data['categories']}
    images = {img['id']: img for img in data['images']}
    
    # If splitting val into val/test
    if target_split_names:
        image_ids = list(images.keys())
        random.shuffle(image_ids)
        mid = len(image_ids) // 2
        split_map = {
            img_id: target_split_names[0] if i < mid else target_split_names[1]
            for i, img_id in enumerate(image_ids)
        }
    else:
        split_map = {img_id: split_name for img_id in images.keys()}

    # Process annotations
    count = 0
    for ann in tqdm(data['annotations']):
        img_id = ann['image_id']
        if img_id not in images: continue
        
        img_info = images[img_id]
        filename = img_info['file_name']
        
        # Find image path
        if filename not in img_map:
            # Try without extension matching
            found = False
            base = os.path.splitext(filename)[0]
            for k, v in img_map.items():
                if os.path.splitext(k)[0] == base:
                    src_path = v
                    found = True
                    break
            if not found:
                continue
        else:
            src_path = img_map[filename]
            
        # Load image
        img = cv2.imread(src_path)
        if img is None: continue
        
        # Crop
        x, y, w, h = map(int, ann['bbox'])
        # Clamp
        h_img, w_img = img.shape[:2]
        x = max(0, x)
        y = max(0, y)
        w = min(w, w_img - x)
        h = min(h, h_img - y)
        
        if w < 10 or h < 10: continue # Skip tiny crops
        
        crop = img[y:y+h, x:x+w]
        
        # Save
        cat_name = categories[ann['category_id']]
        # Clean category name
        cat_name = cat_name.replace(' ', '_')
        
        target_split = split_map[img_id]
        save_dir = OUTPUT_DIR / target_split / cat_name
        save_dir.mkdir(parents=True, exist_ok=True)
        
        save_path = save_dir / f"{img_id}_{ann['id']}.jpg"
        cv2.imwrite(str(save_path), crop)
        count += 1
        
    print(f"Saved {count} crops for {split_name if not target_split_names else target_split_names}")

def main():
    setup_directories()
    
    # Index images
    train_map = index_images(TRAIN_IMG_DIR)
    val_map = index_images(VAL_IMG_DIR)
    
    # Process Train
    process_split(TRAIN_JSON, train_map, 'train')
    
    # Process Val (Split into val and test)
    process_split(VAL_JSON, val_map, 'val', target_split_names=['val', 'test'])
    
    print("Dataset preparation complete!")
    print(f"Data saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
