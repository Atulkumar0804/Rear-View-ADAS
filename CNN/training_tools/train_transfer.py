"""
Continue training remaining models (transfer learning models only)
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
from train_v2 import VehicleDataset, Trainer, plot_training_history, plot_confusion_matrix, evaluate_model

# Configuration
BATCH_SIZE = 32
NUM_EPOCHS = 50
LEARNING_RATE = 0.001
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_WORKERS = 4

print(f"🔥 Using device: {DEVICE}")

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

def main():
    # Get script directory and go up one level to CNN directory
    script_dir = Path(__file__).parent.resolve()
    cnn_dir = script_dir.parent
    
    dataset_dir = cnn_dir / 'dataset'
    plots_dir = cnn_dir / 'plots'
    checkpoints_dir = cnn_dir / 'checkpoints'
    
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
    
    # Only train remaining transfer learning models
    models_to_train = [
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
        model_checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
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
    
    # Print results
    print("\n" + "="*60)
    print("📊 TRANSFER LEARNING RESULTS")
    print("="*60)
    print(f"\n{'Model':<25} {'Val Acc':<12} {'Test Acc':<12}")
    print("-" * 60)
    for model_name, metrics in results.items():
        print(f"{model_name:<25} {metrics['best_val_acc']:>10.2f}%  {metrics['test_acc']:>10.2f}%")
    
    print("\n" + "="*60)
    print("✅ Transfer Learning Models Complete!")
    print("="*60)

if __name__ == "__main__":
    main()
