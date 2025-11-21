"""
Hyperparameter Tuning Script

Performs grid search over:
- Learning rates
- Batch sizes
- Optimizers
- Dropout rates
- Number of layers (for custom models)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import itertools
import json
from pathlib import Path
import sys

sys.path.append('.')
from models.architectures import create_model
from train import VehicleDataset, train_transform, val_transform, Trainer, plot_training_history

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameter search space
HP_SPACE = {
    'learning_rate': [0.0001, 0.001, 0.01],
    'batch_size': [16, 32, 64],
    'optimizer': ['adam', 'sgd', 'rmsprop'],
    'weight_decay': [0, 1e-4, 1e-3],
}

def train_with_config(config, train_dataset, val_dataset, num_epochs=20):
    """Train model with specific hyperparameter configuration"""
    
    print(f"\n{'='*60}")
    print("Testing configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print(f"{'='*60}")
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], 
                             shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], 
                           shuffle=False, num_workers=4)
    
    # Create model
    model = create_model('transfer_mobilenet', num_classes=6, freeze_backbone=True)
    
    # Loss
    criterion = nn.CrossEntropyLoss()
    
    # Optimizer
    if config['optimizer'] == 'adam':
        optimizer = optim.Adam(model.parameters(), 
                              lr=config['learning_rate'],
                              weight_decay=config['weight_decay'])
    elif config['optimizer'] == 'sgd':
        optimizer = optim.SGD(model.parameters(), 
                             lr=config['learning_rate'],
                             momentum=0.9,
                             weight_decay=config['weight_decay'])
    else:  # rmsprop
        optimizer = optim.RMSprop(model.parameters(), 
                                 lr=config['learning_rate'],
                                 weight_decay=config['weight_decay'])
    
    # Scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
                                                     factor=0.5, patience=3)
    
    # Train
    save_dir = './checkpoints/hp_tuning/temp'
    trainer = Trainer(model, train_loader, val_loader, criterion, 
                     optimizer, scheduler, save_dir=save_dir)
    
    # Reduce patience for faster tuning
    trainer.patience = 5
    
    history = trainer.train(num_epochs)
    
    return {
        'best_val_acc': trainer.best_val_acc,
        'final_train_loss': history['train_loss'][-1],
        'final_val_loss': history['val_loss'][-1],
        'history': history
    }

def grid_search():
    """Perform grid search over hyperparameters"""
    
    print("="*60)
    print("🔍 HYPERPARAMETER TUNING - GRID SEARCH")
    print("="*60)
    
    # Load datasets
    print("\n📂 Loading datasets...")
    train_dataset = VehicleDataset('./dataset', 'train', transform=train_transform)
    val_dataset = VehicleDataset('./dataset', 'val', transform=val_transform)
    
    # Generate all combinations
    keys = list(HP_SPACE.keys())
    values = list(HP_SPACE.values())
    combinations = list(itertools.product(*values))
    
    print(f"\n🔬 Total configurations to test: {len(combinations)}")
    
    results = []
    
    for idx, combo in enumerate(combinations):
        config = dict(zip(keys, combo))
        config_id = f"config_{idx+1:03d}"
        
        print(f"\n[{idx+1}/{len(combinations)}] {config_id}")
        
        try:
            result = train_with_config(config, train_dataset, val_dataset, num_epochs=15)
            result['config'] = config
            result['config_id'] = config_id
            results.append(result)
            
            print(f"✅ Val Acc: {result['best_val_acc']:.2f}%")
            
        except Exception as e:
            print(f"❌ Failed: {e}")
            continue
    
    # Sort by validation accuracy
    results.sort(key=lambda x: x['best_val_acc'], reverse=True)
    
    # Print top 10 configurations
    print(f"\n{'='*60}")
    print("🏆 TOP 10 CONFIGURATIONS")
    print(f"{'='*60}\n")
    
    for i, result in enumerate(results[:10]):
        print(f"{i+1}. {result['config_id']} - Val Acc: {result['best_val_acc']:.2f}%")
        print(f"   Config: {result['config']}")
        print()
    
    # Save results
    output_file = './plots/hyperparameter_tuning_results.json'
    with open(output_file, 'w') as f:
        # Convert history to serializable format
        for result in results:
            if 'history' in result:
                del result['history']
        json.dump(results, f, indent=2)
    
    print(f"✅ Results saved to: {output_file}")
    
    return results

def random_search(num_trials=20):
    """Perform random search over hyperparameters"""
    
    print("="*60)
    print("🎲 HYPERPARAMETER TUNING - RANDOM SEARCH")
    print("="*60)
    
    import random
    
    # Load datasets
    print("\n📂 Loading datasets...")
    train_dataset = VehicleDataset('./dataset', 'train', transform=train_transform)
    val_dataset = VehicleDataset('./dataset', 'val', transform=val_transform)
    
    results = []
    
    for trial in range(num_trials):
        # Random configuration
        config = {
            'learning_rate': random.choice([0.0001, 0.0005, 0.001, 0.005, 0.01]),
            'batch_size': random.choice([8, 16, 32, 64]),
            'optimizer': random.choice(['adam', 'sgd', 'rmsprop']),
            'weight_decay': random.choice([0, 1e-5, 1e-4, 1e-3]),
        }
        
        config_id = f"random_{trial+1:03d}"
        print(f"\n[{trial+1}/{num_trials}] {config_id}")
        
        try:
            result = train_with_config(config, train_dataset, val_dataset, num_epochs=15)
            result['config'] = config
            result['config_id'] = config_id
            results.append(result)
            
            print(f"✅ Val Acc: {result['best_val_acc']:.2f}%")
            
        except Exception as e:
            print(f"❌ Failed: {e}")
            continue
    
    # Sort by validation accuracy
    results.sort(key=lambda x: x['best_val_acc'], reverse=True)
    
    # Print top 5
    print(f"\n{'='*60}")
    print("🏆 TOP 5 CONFIGURATIONS")
    print(f"{'='*60}\n")
    
    for i, result in enumerate(results[:5]):
        print(f"{i+1}. {result['config_id']} - Val Acc: {result['best_val_acc']:.2f}%")
        print(f"   Config: {result['config']}")
        print()
    
    # Save results
    output_file = './plots/random_search_results.json'
    with open(output_file, 'w') as f:
        for result in results:
            if 'history' in result:
                del result['history']
        json.dump(results, f, indent=2)
    
    print(f"✅ Results saved to: {output_file}")
    
    return results

def visualize_hp_results(results, save_path):
    """Visualize hyperparameter tuning results"""
    import matplotlib.pyplot as plt
    import numpy as np
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Extract data
    lrs = [r['config']['learning_rate'] for r in results]
    batch_sizes = [r['config']['batch_size'] for r in results]
    optimizers = [r['config']['optimizer'] for r in results]
    val_accs = [r['best_val_acc'] for r in results]
    
    # Learning rate vs accuracy
    axes[0, 0].scatter(lrs, val_accs, alpha=0.6)
    axes[0, 0].set_xlabel('Learning Rate')
    axes[0, 0].set_ylabel('Validation Accuracy (%)')
    axes[0, 0].set_xscale('log')
    axes[0, 0].set_title('Learning Rate vs Accuracy')
    axes[0, 0].grid(True)
    
    # Batch size vs accuracy
    axes[0, 1].scatter(batch_sizes, val_accs, alpha=0.6)
    axes[0, 1].set_xlabel('Batch Size')
    axes[0, 1].set_ylabel('Validation Accuracy (%)')
    axes[0, 1].set_title('Batch Size vs Accuracy')
    axes[0, 1].grid(True)
    
    # Optimizer comparison
    opt_dict = {}
    for opt, acc in zip(optimizers, val_accs):
        if opt not in opt_dict:
            opt_dict[opt] = []
        opt_dict[opt].append(acc)
    
    opts = list(opt_dict.keys())
    means = [np.mean(opt_dict[opt]) for opt in opts]
    stds = [np.std(opt_dict[opt]) for opt in opts]
    
    axes[1, 0].bar(opts, means, yerr=stds, capsize=5)
    axes[1, 0].set_xlabel('Optimizer')
    axes[1, 0].set_ylabel('Validation Accuracy (%)')
    axes[1, 0].set_title('Optimizer Comparison')
    axes[1, 0].grid(True, axis='y')
    
    # Top configs
    top_n = 10
    top_results = sorted(results, key=lambda x: x['best_val_acc'], reverse=True)[:top_n]
    config_names = [r['config_id'] for r in top_results]
    accs = [r['best_val_acc'] for r in top_results]
    
    axes[1, 1].barh(range(len(config_names)), accs)
    axes[1, 1].set_yticks(range(len(config_names)))
    axes[1, 1].set_yticklabels(config_names, fontsize=8)
    axes[1, 1].set_xlabel('Validation Accuracy (%)')
    axes[1, 1].set_title(f'Top {top_n} Configurations')
    axes[1, 1].grid(True, axis='x')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved visualization: {save_path}")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Hyperparameter Tuning')
    parser.add_argument('--method', type=str, default='random', 
                       choices=['grid', 'random'], help='Search method')
    parser.add_argument('--trials', type=int, default=20, 
                       help='Number of trials for random search')
    args = parser.parse_args()
    
    # Create output directory
    Path('./plots').mkdir(exist_ok=True)
    Path('./checkpoints/hp_tuning').mkdir(parents=True, exist_ok=True)
    
    if args.method == 'grid':
        results = grid_search()
    else:
        results = random_search(args.trials)
    
    # Visualize
    viz_path = f'./plots/{args.method}_search_visualization.png'
    visualize_hp_results(results, viz_path)
    
    print(f"\n✅ Hyperparameter tuning complete!")

if __name__ == "__main__":
    main()
