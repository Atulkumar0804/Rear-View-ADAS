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
from torch.utils.data import Dataset, DataLoader
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

# Configuration
BATCH_SIZE = 32
NUM_EPOCHS = 50
LEARNING_RATE = 0.001
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_WORKERS = 4
EARLY_STOPPING_PATIENCE = 10

print(f"🔥 Using device: {DEVICE}")
if torch.cuda.is_available():
    print(f"   GPU: {torch.cuda.get_device_name(0)}")

# Data augmentation
train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
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
    def __init__(self, patience=10, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        
    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
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
        self.early_stopping = EarlyStopping(patience=EARLY_STOPPING_PATIENCE)
    
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc='Training')
        for inputs, labels in pbar:
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
            self.optimizer.step()
            
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
            
            # Update learning rate
            if self.scheduler:
                self.scheduler.step(val_loss)
            
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Save history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['lr'].append(current_lr)
            
            print(f"\nTrain Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
            print(f"Learning Rate: {current_lr:.6f}")
            
            # Save best model
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                checkpoint_path = self.checkpoint_dir / 'best_model.pth'
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_acc': val_acc,
                    'history': self.history
                }, checkpoint_path)
                print(f"✅ Saved best model (Val Acc: {val_acc:.2f}%)")
            
            # Early stopping
            self.early_stopping(val_loss)
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
    print("Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=classes, digits=4))
    
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
    
    dataset_dir = cnn_dir / 'dataset'
    plots_dir = cnn_dir / 'plots'
    checkpoints_dir = cnn_dir / 'checkpoints'
    
    plots_dir.mkdir(exist_ok=True)
    checkpoints_dir.mkdir(exist_ok=True)
    
    # ================================
    # Part 1: Vehicle Classification
    # ================================
    print("\n" + "="*60)
    print("📦 PART 1: Vehicle Classification")
    print("="*60)
    
    # Load datasets
    train_dataset = VehicleDataset(dataset_dir, split='train', transform=train_transform)
    val_dataset = VehicleDataset(dataset_dir, split='val', transform=val_transform)
    test_dataset = VehicleDataset(dataset_dir, split='test', transform=val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, 
                             num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                           num_workers=NUM_WORKERS, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=True)
    
    num_classes = len(train_dataset.classes)
    print(f"\n📊 Classes: {train_dataset.classes}")
    print(f"📊 Number of classes: {num_classes}")
    
    # Models to train
    models_to_train = [
        'mobilenet_inspired',
        'squeezenet_inspired', 
        'resnet_inspired',
        'transfer_mobilenet',
        'transfer_resnet18'
    ]
    
    results = {}
    
    for model_name in models_to_train:
        print(f"\n{'='*60}")
        print(f"🔧 Training: {model_name}")
        print(f"{'='*60}")
        
        # Create model
        model = create_model(model_name, num_classes=num_classes)
        model = model.to(DEVICE)
        
        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
                                                         factor=0.5, patience=5)
        
        # Create checkpoint directory
        model_checkpoint_dir = checkpoints_dir / model_name
        model_checkpoint_dir.mkdir(exist_ok=True)
        
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
