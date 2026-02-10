"""
Main Training Script for ROP Detection
Loads configuration from best_config.json, trains the model,
saves checkpoints and training history, and plots training curves.
"""

import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam, AdamW, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import cv2
from PIL import Image
from torchvision import transforms
from sklearn.metrics import roc_auc_score

# Project imports
import sys
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import ROPDataset
from src.models.resnet import ResNet18ROP
from src.training.loss import get_loss


# ============== CONFIGURATION ==============
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CONFIG_PATH = PROJECT_ROOT / "best_config.json"
TRAIN_CSV = PROJECT_ROOT / "data" / "splits" / "train.csv"
VAL_CSV = PROJECT_ROOT / "data" / "splits" / "val.csv"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
CHECKPOINT_DIR.mkdir(exist_ok=True)


# ============== CUSTOM TRANSFORMS ==============
class CLAHETransform:
    """Apply CLAHE on specified channel of a retinal image."""
    
    def __init__(self, clip_limit=2.0, tile_grid_size=(8, 8), channel='green'):
        self.clahe = cv2.createCLAHE(
            clipLimit=clip_limit,
            tileGridSize=tile_grid_size
        )
        self.channel = channel

    def __call__(self, img):
        img = np.array(img)
        
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        
        if self.channel == 'green':
            # Use green channel
            channel_img = img[:, :, 1]
        elif self.channel == 'luminance':
            # Convert to LAB and use L channel
            lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
            channel_img = lab[:, :, 0]
        else:
            # Grayscale
            channel_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        enhanced = self.clahe.apply(channel_img)
        img = np.stack([enhanced, enhanced, enhanced], axis=2)
        
        return img


def get_transforms_from_config(config, train=True):
    """Build transforms based on config file."""
    transform_list = []
    
    # Resize
    image_size = config.get('image_size', 224)
    transform_list.append(transforms.Resize((image_size, image_size)))
    
    # Data augmentation (training only)
    if train:
        if config.get('horizontal_flip', False):
            transform_list.append(transforms.RandomHorizontalFlip(p=0.5))
        if config.get('vertical_flip', False):
            transform_list.append(transforms.RandomVerticalFlip(p=0.5))
        rotation = config.get('rotation_degrees', 0)
        if rotation > 0:
            transform_list.append(transforms.RandomRotation(rotation))
    
    # CLAHE
    if config.get('use_clahe', True):
        clip_limit = config.get('clahe_clip_limit', 2.0)
        tile_grid = tuple(config.get('clahe_tile_grid', [8, 8]))
        channel = config.get('clahe_channel', 'green')
        transform_list.append(CLAHETransform(clip_limit, tile_grid, channel))
    
    # To tensor and normalize
    transform_list.append(transforms.ToTensor())
    transform_list.append(transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ))
    
    return transforms.Compose(transform_list)


def compute_pos_weight(csv_path):
    """Compute positive class weight for imbalanced dataset."""
    df = pd.read_csv(csv_path)
    n_pos = (df["ROP Label"] == 1).sum()
    n_neg = (df["ROP Label"] == 0).sum()
    if n_pos == 0:
        return torch.tensor([1.0], dtype=torch.float32)
    return torch.tensor([n_neg / n_pos], dtype=torch.float32)


def train_one_epoch(model, loader, optimizer, criterion, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    all_probs, all_labels = [], []
    
    pbar = tqdm(loader, desc="Training", leave=False)
    for imgs, labels in pbar:
        imgs = imgs.to(device)
        labels = labels.to(device).unsqueeze(1)
        
        optimizer.zero_grad()
        logits = model(imgs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        probs = torch.sigmoid(logits).detach()
        all_probs.extend(probs.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    avg_loss = total_loss / len(loader)
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except:
        auc = 0.5
    
    return avg_loss, auc


@torch.no_grad()
def validate(model, loader, criterion, device):
    """Validate the model."""
    model.eval()
    total_loss = 0.0
    all_probs, all_labels = [], []
    
    for imgs, labels in tqdm(loader, desc="Validating", leave=False):
        imgs = imgs.to(device)
        labels = labels.to(device).unsqueeze(1)
        
        logits = model(imgs)
        loss = criterion(logits, labels)
        probs = torch.sigmoid(logits)
        
        total_loss += loss.item()
        all_probs.extend(probs.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(loader)
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except:
        auc = 0.5
    
    return avg_loss, auc


def plot_training_curves(history, save_path):
    """Plot and save training curves."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Loss curve
    ax1 = axes[0]
    ax1.plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    ax1.plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training & Validation Loss', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([1, len(epochs)])
    
    # Mark best epoch
    best_epoch = np.argmin(history['val_loss']) + 1
    ax1.axvline(x=best_epoch, color='green', linestyle='--', alpha=0.7, label=f'Best (Epoch {best_epoch})')
    
    # AUC curve
    ax2 = axes[1]
    ax2.plot(epochs, history['train_auc'], 'b-', label='Train AUC', linewidth=2)
    ax2.plot(epochs, history['val_auc'], 'r-', label='Val AUC', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('AUC-ROC', fontsize=12)
    ax2.set_title('Training & Validation AUC-ROC', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([1, len(epochs)])
    ax2.set_ylim([0, 1.05])
    
    # Mark best AUC epoch
    best_auc_epoch = np.argmax(history['val_auc']) + 1
    best_auc_val = max(history['val_auc'])
    ax2.axvline(x=best_auc_epoch, color='green', linestyle='--', alpha=0.7)
    ax2.annotate(f'Best: {best_auc_val:.4f}', 
                 xy=(best_auc_epoch, best_auc_val),
                 xytext=(best_auc_epoch + 2, best_auc_val - 0.1),
                 fontsize=10, color='green',
                 arrowprops=dict(arrowstyle='->', color='green', alpha=0.7))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"📊 Training curves saved to: {save_path}")


def save_checkpoint(model, optimizer, epoch, history, config, path):
    """Save model checkpoint with full state."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'history': history,
        'config': config
    }
    torch.save(checkpoint, path)


def main():
    print("=" * 60)
    print("🔬 ROP Detection - Model Training")
    print("=" * 60)
    
    # Load configuration
    print(f"\n📋 Loading config from: {CONFIG_PATH}")
    with open(CONFIG_PATH, 'r') as f:
        config = json.load(f)
    
    print("\n📌 Configuration:")
    for key, value in config.items():
        print(f"   {key}: {value}")
    
    # Extract hyperparameters
    batch_size = config.get('batch_size', 32)
    epochs = config.get('epochs', 50)
    lr = config.get('learning_rate', 1e-4)
    weight_decay = config.get('weight_decay', 0.0)
    optimizer_name = config.get('optimizer', 'adamw').lower()
    
    print(f"\n🖥️  Device: {DEVICE}")
    print(f"📊 Batch Size: {batch_size}")
    print(f"🔄 Epochs: {epochs}")
    print(f"📈 Learning Rate: {lr}")
    print(f"⚖️  Weight Decay: {weight_decay}")
    
    # Create datasets
    print("\n📂 Loading datasets...")
    train_transform = get_transforms_from_config(config, train=True)
    val_transform = get_transforms_from_config(config, train=False)
    
    train_ds = ROPDataset(TRAIN_CSV, transforms=train_transform)
    val_ds = ROPDataset(VAL_CSV, transforms=val_transform)
    
    print(f"   Train samples: {len(train_ds)}")
    print(f"   Val samples: {len(val_ds)}")
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    
    # Create model
    print("\n🧠 Creating model...")
    model = ResNet18ROP(pretrained=True).to(DEVICE)
    
    # Compute class weights
    pos_weight = compute_pos_weight(TRAIN_CSV).to(DEVICE)
    print(f"   Positive class weight: {pos_weight.item():.4f}")
    
    criterion = get_loss(pos_weight)
    
    # Create optimizer
    if optimizer_name == 'adamw':
        optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == 'adam':
        optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == 'sgd':
        optimizer = SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9)
    else:
        optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # Learning rate scheduler
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, verbose=True)
    
    # Training history
    history = {
        'train_loss': [],
        'train_auc': [],
        'val_loss': [],
        'val_auc': [],
        'lr': []
    }
    
    best_auc = 0.0
    best_epoch = 0
    
    print("\n" + "=" * 60)
    print("🚀 Starting Training...")
    print("=" * 60 + "\n")
    
    for epoch in range(1, epochs + 1):
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch:02d}/{epochs} (LR: {current_lr:.2e})")
        print("-" * 40)
        
        # Train
        train_loss, train_auc = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE)
        
        # Validate
        val_loss, val_auc = validate(model, val_loader, criterion, DEVICE)
        
        # Update scheduler
        scheduler.step(val_auc)
        
        # Record history
        history['train_loss'].append(train_loss)
        history['train_auc'].append(train_auc)
        history['val_loss'].append(val_loss)
        history['val_auc'].append(val_auc)
        history['lr'].append(current_lr)
        
        print(f"   Train Loss: {train_loss:.4f} | Train AUC: {train_auc:.4f}")
        print(f"   Val Loss:   {val_loss:.4f} | Val AUC:   {val_auc:.4f}")
        
        # Save best model
        if val_auc > best_auc:
            best_auc = val_auc
            best_epoch = epoch
            
            # Save model weights only (for inference)
            torch.save(model.state_dict(), CHECKPOINT_DIR / "best_model.pth")
            
            # Save full checkpoint (for resuming training)
            save_checkpoint(
                model, optimizer, epoch, history, config,
                CHECKPOINT_DIR / "best_checkpoint.pth"
            )
            print(f"   ✅ New best model saved! (AUC: {best_auc:.4f})")
        
        print()
    
    # Save final model
    torch.save(model.state_dict(), CHECKPOINT_DIR / "final_model.pth")
    save_checkpoint(
        model, optimizer, epochs, history, config,
        CHECKPOINT_DIR / "final_checkpoint.pth"
    )
    
    # Save training history
    history_df = pd.DataFrame(history)
    history_df.to_csv(CHECKPOINT_DIR / "training_history.csv", index=False)
    
    # Save config used
    with open(CHECKPOINT_DIR / "training_config.json", 'w') as f:
        json.dump(config, f, indent=2)
    
    print("=" * 60)
    print("🎉 Training Complete!")
    print("=" * 60)
    print(f"\n📊 Best Validation AUC: {best_auc:.4f} (Epoch {best_epoch})")
    print(f"\n💾 Saved files in {CHECKPOINT_DIR}:")
    print("   • best_model.pth       - Best model weights (for inference)")
    print("   • best_checkpoint.pth  - Full checkpoint (for resume/evaluation)")
    print("   • final_model.pth      - Final epoch model weights")
    print("   • final_checkpoint.pth - Final epoch full checkpoint")
    print("   • training_history.csv - Training metrics per epoch")
    print("   • training_config.json - Config used for training")
    
    # Plot training curves
    print("\n📈 Plotting training curves...")
    plot_training_curves(history, CHECKPOINT_DIR / "training_curves.png")
    
    return history, best_auc


if __name__ == "__main__":
    history, best_auc = main()
