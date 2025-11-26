"""
Training Script for Vehicle Classification & Distance Estimation
Updated to work with YOLO-detected dataset

Features:
- Transfer learning from pre-trained models (MobileNetV2, ResNet18)
- Vehicle classification (car, truck, bus, person)
- Distance estimation from sequences (approaching/stationary/receding)
- Learning rate scheduling
- Early stopping
- Comprehensive evaluation and plotting
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import numpy as np
import cv2
import json
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import sys
import time

sys.path.append('.')
from models.architectures import create_model

# Configuration - Optimized for UVH-26 (252K samples) on RTX A6000
BATCH_SIZE = 128  # Larger batch for better gradient estimates with RTX A6000 (48GB VRAM)
NUM_EPOCHS = 100  # More epochs for large dataset convergence
LEARNING_RATE = 0.001  # Standard Adam LR, will be adjusted by scheduler
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_WORKERS = 8  # Increased for faster data loading
EARLY_STOPPING_PATIENCE = 15  # Increased patience for large dataset
WEIGHT_DECAY = 1e-4  # L2 regularization
LABEL_SMOOTHING = 0.1  # Label smoothing for better generalization

print(f"🔥 Using device: {DEVICE}")
if torch.cuda.is_available():
    print(f"   GPU: {torch.cuda.get_device_name(0)}")

# Data augmentation - Enhanced for large dataset
train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(20),  # Increased rotation
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3, hue=0.15),  # Stronger color jitter
    transforms.RandomAffine(degrees=0, translate=(0.15, 0.15), scale=(0.85, 1.15)),  # More aggressive affine
    transforms.RandomPerspective(distortion_scale=0.2, p=0.3),  # Add perspective transform
    transforms.RandomGrayscale(p=0.1),  # Random grayscale for robustness
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.3, scale=(0.02, 0.15))  # Random erasing for occlusion robustness
])

val_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


class VehicleDataset(Dataset):
    """Dataset for vehicle classification from YOLO detections"""
    def __init__(self, root_dir, split='train', transform=None):
        self.root_dir = Path(root_dir) / split
        self.transform = transform
        
        # Only classes that exist in dataset
        self.classes = ['car', 'truck', 'bus', 'person']
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        self.samples = []
        for cls in self.classes:
            cls_dir = self.root_dir / cls
            if cls_dir.exists():
                for img_path in cls_dir.glob("*.jpg"):
                    self.samples.append((str(img_path), self.class_to_idx[cls]))
        
        print(f"✅ Loaded {len(self.samples)} {split} samples across {len(self.classes)} classes")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: Could not read {img_path}")
            # Return a black image
            img = np.zeros((224, 224, 3), dtype=np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        if self.transform:
            img = self.transform(img)
        
        return img, label


class UVH26Dataset(Dataset):
    """Dataset for UVH-26 vehicle classification from COCO annotations
    
    Supports 15 classes (14 vehicle classes + Person):
    1-Hatchback, 2-Sedan, 3-SUV, 4-MUV, 5-Bus, 6-Truck, 7-Three-wheeler,
    8-Two-wheeler, 9-LCV, 10-Mini-bus, 11-Tempo-traveller, 12-Bicycle, 13-Van, 14-Other, 15-Person
    """
    def __init__(self, root_dir, split='train', transform=None, use_staple=False, person_crops_dir=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.split = split
        
        # 15 classes: 14 UVH-26 vehicle classes + Person (1-indexed in JSON, 0-indexed for model)
        self.classes = [
            'Hatchback', 'Sedan', 'SUV', 'MUV', 'Bus', 'Truck', 'Three-wheeler',
            'Two-wheeler', 'LCV', 'Mini-bus', 'Tempo-traveller', 'Bicycle', 'Van', 'Other', 'Person'
        ]
        
        # Load COCO format annotations
        if split == 'train':
            json_file = 'UVH-26-ST-Train.json' if use_staple else 'UVH-26-MV-Train.json'
            ann_path = self.root_dir / 'UVH-26-Train' / json_file
            self.img_dir = self.root_dir / 'UVH-26-Train' / 'data'
        else:  # val or test
            json_file = 'UVH-26-ST-Val.json' if use_staple else 'UVH-26-MV-Val.json'
            ann_path = self.root_dir / 'UVH-26-Val' / json_file
            self.img_dir = self.root_dir / 'UVH-26-Val' / 'data'
        
        print(f"📂 Loading annotations from: {ann_path}")
        with open(ann_path, 'r') as f:
            self.coco_data = json.load(f)
        
        # Build a fast lookup of available image files (filename -> full path)
        print(f"📂 Scanning available image files...")
        self.available_files = {}
        for img_file in self.img_dir.rglob('*.png'):
            filename = img_file.name  # Just the filename without path
            self.available_files[filename] = img_file
        print(f"✅ Found {len(self.available_files)} image files on disk")
        
        # Build image_id to filename mapping
        self.img_id_to_file = {img['id']: img['file_name'] for img in self.coco_data['images']}
        
        # Build samples: [(image_path, bbox, category_id), ...]
        # Only include samples where the image file actually exists
        self.samples = []
        skipped = 0
        for ann in self.coco_data['annotations']:
            img_id = ann['image_id']
            if img_id in self.img_id_to_file:
                img_file = self.img_id_to_file[img_id]
                
                # Check if file actually exists on disk
                if img_file not in self.available_files:
                    skipped += 1
                    continue
                    
                bbox = ann['bbox']  # [x, y, width, height]
                category_id = ann['category_id'] - 1  # Convert to 0-indexed (0-13)
                
                # Map category ID: 0-12 stay same, 13 (Others) maps to 13, add Person as class 14 later
                # Skip if category out of range
                if 0 <= category_id < 14:  # Accept original 14 classes
                    full_path = self.available_files[img_file]
                    self.samples.append((full_path, bbox, category_id))
        
        print(f"✅ Loaded {len(self.samples)} valid {split} samples (skipped {skipped} missing files)")
        
        # Add Person class samples if person_crops_dir is provided
        if person_crops_dir is not None:
            person_dir = Path(person_crops_dir) / split / 'Person'
            if person_dir.exists():
                person_count = 0
                for img_path in person_dir.glob('*.jpg'):
                    # No bbox needed for person crops (already cropped)
                    # Use full image bbox [0, 0, width, height] - will be handled in __getitem__
                    self.samples.append((img_path, None, 14))  # 14 is Person class index
                    person_count += 1
                print(f"✅ Added {person_count} Person samples from {person_dir}")
            else:
                print(f"⚠️ Person crops directory not found: {person_dir}")
                print(f"   Run: python scripts/add_person_class.py to generate Person samples")
        
        # Print class distribution
        class_counts = {i: 0 for i in range(len(self.classes))}
        for sample in self.samples:
            cat_id = sample[2]  # category_id is the 3rd element
            if cat_id < len(self.classes):
                class_counts[cat_id] += 1
        print(f"📊 Class distribution:")
        for i, cls in enumerate(self.classes):
            if class_counts[i] > 0:
                print(f"   {cls:<18}: {class_counts[i]:>6} samples")
            elif i == 14:  # Person class
                print(f"   {cls:<18}: {class_counts[i]:>6} samples (run add_person_class.py)")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, bbox, label = self.samples[idx]
        
        # Load image (img_path is already the full Path object)
        img = cv2.imread(str(img_path))
        
        if img is None:
            # Should not happen since we filtered missing files, but just in case
            print(f"⚠️ Warning: Could not read {img_path}")
            img = np.zeros((224, 224, 3), dtype=np.uint8)
        else:
            # If bbox is None, image is already cropped (e.g., Person crops)
            if bbox is not None:
                # Crop to bounding box [x, y, width, height]
                x, y, w, h = [int(v) for v in bbox]
                h_img, w_img = img.shape[:2]
                
                # Clamp bbox to image boundaries
                x1 = max(0, x)
                y1 = max(0, y)
                x2 = min(w_img, x + w)
                y2 = min(h_img, y + h)
                
                if x2 > x1 and y2 > y1:
                    img = img[y1:y2, x1:x2]
                else:
                    # Invalid bbox, use full image
                    pass
            
            # Resize to 224x224
            img = cv2.resize(img, (224, 224))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        if self.transform:
            img = self.transform(img)
        
        return img, label


class SequenceDataset(Dataset):
    """Dataset for distance estimation from tracked sequences"""
    def __init__(self, root_dir, split='train', transform=None, base_img_dir='../data/samples/CAM_BACK'):
        self.seq_dir = Path(root_dir) / f'{split}_sequences'
        self.transform = transform
        self.base_img_dir = Path(base_img_dir)
        
        self.distance_labels = ['approaching', 'stationary', 'receding']
        self.label_to_idx = {label: idx for idx, label in enumerate(self.distance_labels)}
        
        self.sequences = []
        for seq_file in self.seq_dir.glob("seq_*.json"):
            with open(seq_file, 'r') as f:
                seq_data = json.load(f)
                if seq_data['num_frames'] >= 5:  # Ensure minimum length
                    self.sequences.append(seq_data)
        
        print(f"✅ Loaded {len(self.sequences)} {split} sequences")
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        seq = self.sequences[idx]
        
        # Extract first and last 3 frames for distance comparison
        frames_to_load = [0, len(seq['frames'])//4, len(seq['frames'])//2, 
                          3*len(seq['frames'])//4, len(seq['frames'])-1]
        
        images = []
        scene_id = seq['scene_id']
        
        for frame_data in [seq['frames'][i] for i in frames_to_load if i < len(seq['frames'])]:
            # Find the image file
            scene_folder = self.base_img_dir / str(scene_id)
            
            # Get image files sorted
            img_files = sorted(scene_folder.glob("*.jpg"))
            if frame_data['frame_idx'] < len(img_files):
                img_path = img_files[frame_data['frame_idx']]
                img = cv2.imread(str(img_path))
                if img is not None:
                    # Crop to bbox
                    x1, y1, x2, y2 = frame_data['bbox']
                    crop = img[y1:y2, x1:x2]
                    if crop.size > 0:
                        crop = cv2.resize(crop, (224, 224))
                        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                        if self.transform:
                            crop = self.transform(crop)
                        images.append(crop)
        
        # Pad if needed
        while len(images) < 5:
            if len(images) > 0:
                images.append(images[-1])
            else:
                # Create blank image
                blank = torch.zeros(3, 224, 224)
                images.append(blank)
        
        # Stack frames
        images = torch.stack(images[:5])
        
        # Get distance label
        label = self.label_to_idx[seq['distance_label']]
        
        return images, label


class EarlyStopping:
    """Early stopping to prevent overfitting"""
    def __init__(self, patience=15, min_delta=0.0005, mode='min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, metric):
        score = -metric if self.mode == 'min' else metric
        
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0


class Trainer:
    """Trainer class for model training and evaluation"""
    def __init__(self, model, train_loader, val_loader, criterion, optimizer, 
                 scheduler, device, checkpoint_dir, model_name):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.model_name = model_name
        
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': [],
            'lr': []
        }
        
        self.best_val_acc = 0.0
        self.early_stopping = EarlyStopping(patience=EARLY_STOPPING_PATIENCE, min_delta=0.0005, mode='max')
        self.grad_clip = 1.0  # Gradient clipping threshold
    
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc='Training')
        for batch_idx, (inputs, labels) in enumerate(pbar):
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            
            # Handle sequences (LSTM input)
            if len(inputs.shape) == 5:  # [batch, seq_len, C, H, W]
                batch_size, seq_len = inputs.shape[0], inputs.shape[1]
                inputs = inputs.view(batch_size * seq_len, *inputs.shape[2:])
            
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            
            # Handle LSTM output
            if len(inputs.shape) == 5:
                outputs = outputs.view(batch_size, seq_len, -1)[:, -1, :]
            
            loss = self.criterion(outputs, labels)
            loss.backward()
            
            # Gradient clipping for stable training
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            
            self.optimizer.step()
            
            # Step scheduler per batch for CosineAnnealingWarmRestarts
            if isinstance(self.scheduler, optim.lr_scheduler.CosineAnnealingWarmRestarts):
                self.scheduler.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 
                            'acc': f'{100.*correct/total:.2f}%'})
        
        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    def validate(self):
        """Validate the model"""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc='Validation')
            for inputs, labels in pbar:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                # Handle sequences
                if len(inputs.shape) == 5:
                    batch_size, seq_len = inputs.shape[0], inputs.shape[1]
                    inputs = inputs.view(batch_size * seq_len, *inputs.shape[2:])
                
                outputs = self.model(inputs)
                
                if len(inputs.shape) == 5:
                    outputs = outputs.view(batch_size, seq_len, -1)[:, -1, :]
                
                loss = self.criterion(outputs, labels)
                
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
                pbar.set_postfix({'loss': f'{loss.item():.4f}', 
                                'acc': f'{100.*correct/total:.2f}%'})
        
        epoch_loss = running_loss / len(self.val_loader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc, all_preds, all_labels
    
    def train(self, num_epochs):
        """Train the model for multiple epochs"""
        print(f"\n{'='*60}")
        print(f"Training {self.model_name}")
        print(f"{'='*60}\n")
        
        start_time = time.time()
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            print("-" * 40)
            
            # Train
            train_loss, train_acc = self.train_epoch()
            
            # Validate
            val_loss, val_acc, val_preds, val_labels = self.validate()
            
            # Save history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['lr'].append(self.optimizer.param_groups[0]['lr'])
            
            # Learning rate scheduling
            if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(val_loss)
            elif isinstance(self.scheduler, optim.lr_scheduler.CosineAnnealingWarmRestarts):
                # For CosineAnnealingWarmRestarts, step is called per batch in train_epoch
                pass
            elif self.scheduler:
                self.scheduler.step()
            
            current_lr = self.optimizer.param_groups[0]['lr']
            
            print(f"\nTrain Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
            print(f"Learning Rate: {current_lr:.6f}")
            
            # Save best model based on validation accuracy
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                checkpoint_path = self.checkpoint_dir / 'best_model.pth'
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'val_acc': val_acc,
                    'val_loss': val_loss,
                    'history': self.history
                }, checkpoint_path)
                print(f"💾 Saved best model (Val Acc: {val_acc:.2f}%)")
            
            # Early stopping based on validation accuracy
            self.early_stopping(val_acc)
            if self.early_stopping.early_stop:
                print(f"\n⚠️  Early stopping triggered at epoch {epoch+1}")
                break
        
        training_time = time.time() - start_time
        print(f"\n{'='*60}")
        print(f"Training Complete!")
        print(f"Time: {training_time/60:.2f} minutes")
        print(f"Best Val Acc: {self.best_val_acc:.2f}%")
        print(f"{'='*60}\n")
        
        return self.history


def plot_training_history(history, save_path, model_name):
    """Plot training history"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'Training History - {model_name}', fontsize=16)
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Loss
    axes[0, 0].plot(epochs, history['train_loss'], 'b-', label='Train Loss')
    axes[0, 0].plot(epochs, history['val_loss'], 'r-', label='Val Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Loss Curves')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Accuracy
    axes[0, 1].plot(epochs, history['train_acc'], 'b-', label='Train Acc')
    axes[0, 1].plot(epochs, history['val_acc'], 'r-', label='Val Acc')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy (%)')
    axes[0, 1].set_title('Accuracy Curves')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Learning Rate
    axes[1, 0].plot(epochs, history['lr'], 'g-')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Learning Rate')
    axes[1, 0].set_title('Learning Rate Schedule')
    axes[1, 0].set_yscale('log')
    axes[1, 0].grid(True)
    
    # Overfitting Gap
    gap = [train - val for train, val in zip(history['train_acc'], history['val_acc'])]
    axes[1, 1].plot(epochs, gap, 'purple')
    axes[1, 1].axhline(y=0, color='k', linestyle='--', alpha=0.3)
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Train Acc - Val Acc (%)')
    axes[1, 1].set_title('Overfitting Gap')
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved training history plot: {save_path}")


def plot_confusion_matrix(y_true, y_pred, classes, save_path, model_name):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes,
                cbar_kws={'label': 'Count'})
    plt.title(f'Confusion Matrix - {model_name}', fontsize=14)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    # Add accuracy per class
    acc_per_class = cm.diagonal() / cm.sum(axis=1)
    for i, acc in enumerate(acc_per_class):
        plt.text(len(classes) + 0.5, i + 0.5, f'{acc*100:.1f}%', 
                va='center', ha='left', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved confusion matrix: {save_path}")


def evaluate_model(model, test_loader, classes, device, model_name, save_dir):
    """Evaluate model on test set"""
    print(f"\n{'='*60}")
    print(f"Evaluating {model_name} on Test Set")
    print(f"{'='*60}\n")
    
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc='Testing'):
            inputs = inputs.to(device)
            
            # Handle sequences
            if len(inputs.shape) == 5:
                batch_size, seq_len = inputs.shape[0], inputs.shape[1]
                inputs = inputs.view(batch_size * seq_len, *inputs.shape[2:])
            
            outputs = model(inputs)
            
            if len(inputs.shape) == 5:
                outputs = outputs.view(batch_size, seq_len, -1)[:, -1, :]
            
            _, predicted = outputs.max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    # Calculate metrics
    test_acc = accuracy_score(all_labels, all_preds) * 100
    
    print(f"\n📊 Test Accuracy: {test_acc:.2f}%\n")
    # Print classification report safely: if some classes have zero samples,
    # sklearn.classification_report raises an error when target_names length
    # doesn't match the number of labels present. Filter to present labels.
    print("Classification Report:")
    if len(all_labels) == 0:
        print("No test samples available to create classification report.")
    else:
        present_labels = sorted(list(set(all_labels)))
        # Build target names for present labels only
        present_target_names = [classes[i] for i in present_labels]
        try:
            print(classification_report(all_labels, all_preds, labels=present_labels,
                                        target_names=present_target_names, digits=4))
        except Exception as e:
            # Fallback: print a simpler report and continue
            print(f"⚠️  Could not produce full classification_report: {e}")
            from collections import Counter
            counts = Counter(all_labels)
            print("Label counts in test set:")
            for lbl in present_labels:
                print(f"  {lbl}: {counts[lbl]} samples - {classes[lbl]}")
    
    # Plot confusion matrix
    cm_path = save_dir / f'{model_name}_confusion_matrix.png'
    plot_confusion_matrix(all_labels, all_preds, classes, cm_path, model_name)
    
    return test_acc


def main():
    """Main training pipeline"""
    print("\n" + "="*60)
    print("🚀 Vehicle Detection Training Pipeline")
    print("="*60)
    
    # Get script directory and go up one level to CNN directory
    script_dir = Path(__file__).parent.resolve()
    cnn_dir = script_dir.parent
    
    # Use UVH-26 dataset
    dataset_dir = cnn_dir / 'datasets' / 'UVH-26'
    person_crops_dir = cnn_dir / 'datasets' / 'person_crops'  # Optional Person class directory
    plots_dir = cnn_dir / 'plots'
    checkpoints_dir = cnn_dir / 'checkpoints'
    
    plots_dir.mkdir(exist_ok=True)
    checkpoints_dir.mkdir(exist_ok=True)
    
    # ================================
    # Part 1: Vehicle Classification (UVH-26) + Person
    # ================================
    print("\n" + "="*60)
    print("📦 PART 1: Vehicle Classification on UVH-26 Dataset + Person")
    print("="*60)
    
    # Load UVH-26 datasets (15 classes: 14 vehicles + Person)
    print("\n🔄 Loading UVH-26 datasets...")
    train_dataset = UVH26Dataset(dataset_dir, split='train', transform=train_transform, 
                                  use_staple=False, person_crops_dir=person_crops_dir)
    
    # Load full validation set, then split into val and test (50/50)
    print("🔄 Loading validation dataset for splitting...")
    full_val_dataset = UVH26Dataset(dataset_dir, split='val', transform=val_transform, 
                                     use_staple=False, person_crops_dir=person_crops_dir)
    
    # Split validation set: 50% for validation, 50% for testing
    val_size = len(full_val_dataset) // 2
    test_size = len(full_val_dataset) - val_size
    
    # Use random_split with a fixed seed for reproducibility
    torch.manual_seed(42)
    val_dataset, test_dataset = random_split(full_val_dataset, [val_size, test_size])
    
    print(f"✅ Split validation set: {val_size:,} val samples, {test_size:,} test samples")
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, 
                             num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                           num_workers=NUM_WORKERS, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=True)
    
    num_classes = len(train_dataset.classes)
    print(f"\n📊 Classes: {train_dataset.classes}")
    print(f"📊 Number of classes: {num_classes}")
    
    # Train custom CNN architectures
    models_to_train = [
        'mobilenet_inspired',
        'squeezenet_inspired', 
        'resnet_inspired'
    ]
    
    results = {}
    
    for model_name in models_to_train:
        print(f"\n{'='*60}")
        print(f"🔧 Training: {model_name}")
        print(f"{'='*60}")
        # Create checkpoint directory
        model_checkpoint_dir = checkpoints_dir / model_name
        model_checkpoint_dir.mkdir(exist_ok=True)

        # If a best checkpoint already exists, skip training and evaluate
        best_ckpt = model_checkpoint_dir / 'best_model.pth'
        if best_ckpt.exists():
            print(f"⏭️  Checkpoint found for {model_name}, checking compatibility...")
            model = create_model(model_name, num_classes=num_classes)
            model = model.to(DEVICE)
            checkpoint = torch.load(best_ckpt, map_location=DEVICE)
            
            # Check if checkpoint is compatible by trying to load it
            try:
                model.load_state_dict(checkpoint['model_state_dict'], strict=True)
                print(f"✅ Checkpoint compatible, skipping training and evaluating existing model.")
            except (RuntimeError, KeyError) as e:
                print(f"⚠️  Checkpoint for {model_name} is incompatible: {e}")
                print(f"   Removing old checkpoint and training from scratch...")
                # Don't use the incompatible checkpoint - will train from scratch below
            else:
                # Checkpoint loaded successfully, evaluate and skip training
                test_acc = evaluate_model(model, test_loader, train_dataset.classes,
                                         DEVICE, model_name, plots_dir)
                results[model_name] = {
                    'best_val_acc': checkpoint.get('val_acc', 0.0),
                    'test_acc': test_acc
                }
                continue

        # Create model
        model = create_model(model_name, num_classes=num_classes)
        model = model.to(DEVICE)
        
        # Loss with label smoothing for better generalization
        criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
        
        # Optimizer with weight decay
        optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        
        # Cosine Annealing with Warm Restarts for better convergence
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=10, T_mult=2, eta_min=1e-6
        )
        
        # Train
        trainer = Trainer(model, train_loader, val_loader, criterion, optimizer,
                  scheduler, DEVICE, model_checkpoint_dir, model_name)
        history = trainer.train(NUM_EPOCHS)
        
        # Plot history
        history_plot_path = plots_dir / f'{model_name}_history.png'
        plot_training_history(history, history_plot_path, model_name)
        
        # Load best model and evaluate
        checkpoint = torch.load(model_checkpoint_dir / 'best_model.pth')
        model.load_state_dict(checkpoint['model_state_dict'])
        
        test_acc = evaluate_model(model, test_loader, train_dataset.classes, 
                                 DEVICE, model_name, plots_dir)
        
        results[model_name] = {
            'best_val_acc': trainer.best_val_acc,
            'test_acc': test_acc
        }
    
    # ================================
    # Print Final Results
    # ================================
    print("\n" + "="*60)
    print("📊 FINAL RESULTS")
    print("="*60)
    print(f"\n{'Model':<25} {'Val Acc':<12} {'Test Acc':<12}")
    print("-" * 60)
    for model_name, metrics in results.items():
        print(f"{model_name:<25} {metrics['best_val_acc']:>10.2f}%  {metrics['test_acc']:>10.2f}%")
    
    # Find best model
    best_model = max(results.items(), key=lambda x: x[1]['test_acc'])
    print(f"\n🏆 Best Model: {best_model[0]} (Test Acc: {best_model[1]['test_acc']:.2f}%)")
    
    print("\n" + "="*60)
    print("✅ Training Complete!")
    print("="*60)
    print(f"\n📁 Checkpoints saved to: {checkpoints_dir}")
    print(f"📊 Plots saved to: {plots_dir}")
    print(f"\nNext: Run inference with: python inference.py --model checkpoints/{best_model[0]}/best_model.pth")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
