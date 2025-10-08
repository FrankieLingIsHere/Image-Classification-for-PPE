#!/usr/bin/env python3
"""
Enhanced PPE Training Script with Advanced Regularization
Features:
- Data Augmentation
- Increased Weight Decay
- Proper Early Stopping
- Dropout (if available in model)
- Better Learning Rate Scheduling
"""

import os
import sys
import argparse
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from pathlib import Path
import torchvision.transforms as transforms

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.models.ssd import SSD300
from src.dataset.ppe_dataset import PPEDataset
from src.models.loss import PPELoss


def get_augmented_transforms():
    """Get data augmentation transforms for training"""
    import torchvision.transforms.v2 as transforms_v2
    return {
        'train': transforms_v2.Compose([
            transforms_v2.ToPILImage(),
            transforms_v2.Resize((300, 300)),
            transforms_v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms_v2.RandomHorizontalFlip(p=0.5),
            transforms_v2.ToTensor(),
            transforms_v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    }


def get_validation_transforms():
    """Get transforms for validation (no augmentation)"""
    import torchvision.transforms.v2 as transforms_v2
    return {
        'val': transforms_v2.Compose([
            transforms_v2.ToPILImage(),
            transforms_v2.Resize((300, 300)),
            transforms_v2.ToTensor(),
            transforms_v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    }


def custom_collate_fn(batch):
    """Custom collate function for variable-sized annotations"""
    images = []
    all_annotations = []
    filenames = []
    
    for item in batch:
        if len(item) == 3:
            image, annotations, filename = item
        else:
            image, annotations = item
            filename = "unknown"
        
        images.append(image)
        all_annotations.append(annotations)
        filenames.append(filename)
    
    # Stack images
    images = torch.stack(images, dim=0)
    
    return images, all_annotations, filenames


def validate_model(model, val_loader, criterion, device):
    """Validate the model and return average loss"""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch_idx, (images, annotations, filenames) in enumerate(val_loader):
            try:
                images = images.to(device)
                
                # Extract boxes and labels from annotations
                boxes = []
                labels = []
                for target in annotations:
                    if len(target['bboxes']) > 0:
                        boxes.append(target['bboxes'])
                        labels.append(target['labels'])
                    else:
                        # Handle empty annotations
                        boxes.append(torch.zeros(0, 4))
                        labels.append(torch.zeros(0, dtype=torch.long))
                
                # Forward pass
                loc_preds, class_preds = model(images)
                
                # Calculate loss
                loss = criterion(loc_preds, class_preds, boxes, labels)
                
                if torch.isfinite(loss):
                    total_loss += loss.item()
                    num_batches += 1
                else:
                    print(f"  Warning: Invalid validation loss detected: {loss.item()}")
                    
            except Exception as e:
                print(f"  Validation error on batch {batch_idx}: {e}")
                continue
    
    return total_loss / num_batches if num_batches > 0 else float('inf')

class EarlyStopping:
    """Early stopping to stop training when validation loss stops improving"""
    def __init__(self, patience=7, min_delta=0.001, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = float('inf')
        self.counter = 0
        self.best_weights = None
        
    def __call__(self, val_loss, model):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            if self.restore_best_weights:
                self.best_weights = model.state_dict().copy()
            return False
        else:
            self.counter += 1
            if self.counter >= self.patience:
                if self.restore_best_weights and self.best_weights:
                    model.load_state_dict(self.best_weights)
                return True
        return False


def add_weight_decay(model, weight_decay=1e-5, skip_list=()):
    """Separate parameters for weight decay"""
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if len(param.shape) == 1 or name in skip_list:
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {'params': no_decay, 'weight_decay': 0.},
        {'params': decay, 'weight_decay': weight_decay}]


def train_with_validation(args):
    """Main training function with validation monitoring"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    print("Loading datasets...")
    
    # Training dataset with augmentation
    train_dataset = PPEDataset(
        data_dir='data',
        split='train',
        transforms=get_augmented_transforms()
    )
    
    # Validation dataset without augmentation
    val_dataset = PPEDataset(
        data_dir='data',
        split='val',
        transforms=get_validation_transforms()
    )
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # Data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        collate_fn=custom_collate_fn,
        num_workers=0,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        collate_fn=custom_collate_fn,
        num_workers=0,
        pin_memory=True
    )
    
    # Initialize model
    print("Initializing model...")
    model = SSD300(n_classes=train_dataset.get_num_classes())
    model = model.to(device)
    
    # Loss function (priors are created inside the model)
    criterion = PPELoss(priors_cxcy=model.priors_cxcy)
    
    # Optimizer with separated weight decay
    param_groups = add_weight_decay(model, weight_decay=args.weight_decay)
    optimizer = optim.AdamW(param_groups, lr=args.lr)
    
    # Learning rate scheduler - more aggressive
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.5, 
        patience=5, 
        min_lr=1e-6
    )
    
    # Early stopping
    early_stopping = EarlyStopping(patience=args.patience, min_delta=0.001)
    
    # Training tracking
    train_losses = []
    val_losses = []
    learning_rates = []
    best_val_loss = float('inf')
    
    print(f"\nStarting training for {args.epochs} epochs...")
    print(f"Output directory: {output_dir}")
    print(f"Weight decay: {args.weight_decay}")
    print(f"Early stopping patience: {args.patience}")
    print(f"Data augmentation: Enabled")
    
    for epoch in range(args.epochs):
        print(f"\n\ud83d\udcc8 Epoch {epoch}")
        print("-" * 40)
        
        # Training phase
        model.train()
        total_train_loss = 0.0
        train_batches = 0
        
        for batch_idx, (images, annotations, filenames) in enumerate(train_loader):
            try:
                images = images.to(device)
                
                # Extract boxes and labels from annotations
                boxes = []
                labels = []
                for target in annotations:
                    if len(target['bboxes']) > 0:
                        boxes.append(target['bboxes'])
                        labels.append(target['labels'])
                    else:
                        # Handle empty annotations
                        boxes.append(torch.zeros(0, 4))
                        labels.append(torch.zeros(0, dtype=torch.long))
                
                # Zero gradients
                optimizer.zero_grad()
                
                # Forward pass
                loc_preds, class_preds = model(images)
                
                # Calculate loss
                loss = criterion(loc_preds, class_preds, boxes, labels)
                
                # Check for invalid loss
                if not torch.isfinite(loss):
                    print(f"  Warning: Invalid loss detected: {loss.item()}")
                    continue
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                # Update weights
                optimizer.step()
                
                total_train_loss += loss.item()
                train_batches += 1
                
                if batch_idx == 0:
                    print(f"  Batch {batch_idx}/{len(train_loader)-1}, Loss: {loss.item():.4f}")
                    
            except Exception as e:
                print(f"  Training error on batch {batch_idx}: {e}")
                continue
        
        # Calculate average training loss
        avg_train_loss = total_train_loss / train_batches if train_batches > 0 else float('inf')
        
        # Validation phase
        print("  Validating...")
        avg_val_loss = validate_model(model, val_loader, criterion, device)
        
        # Update learning rate scheduler
        scheduler.step(avg_val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Store metrics
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        learning_rates.append(current_lr)
        
        # Print epoch results
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Val Loss: {avg_val_loss:.4f}")
        print(f"  Learning Rate: {current_lr:.6f}")
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_path = output_dir / 'best_model_regularized.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss,
                'train_losses': train_losses,
                'val_losses': val_losses
            }, best_model_path)
            print(f"  \u2705 New best validation loss!")
            print(f"  \ud83d\udcbe Best model saved to: {best_model_path}")
        
        # Check early stopping
        if early_stopping(avg_val_loss, model):
            print(f"\n\ud83d\uded1 Early stopping triggered after {epoch + 1} epochs")
            print(f"Best validation loss: {early_stopping.best_loss:.4f}")
            break
        
        # Save training curves periodically
        if (epoch + 1) % 5 == 0:
            plt.figure(figsize=(12, 4))
            
            plt.subplot(1, 2, 1)
            plt.plot(train_losses, label='Training Loss', color='blue', linewidth=2)
            plt.plot(val_losses, label='Validation Loss', color='red', linewidth=2)
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Training vs Validation Loss')
            plt.legend()
            plt.grid(True)
            
            plt.subplot(1, 2, 2)
            plt.plot(learning_rates, label='Learning Rate', color='green', linewidth=2)
            plt.xlabel('Epoch')
            plt.ylabel('Learning Rate')
            plt.title('Learning Rate Schedule')
            plt.legend()
            plt.grid(True)
            plt.yscale('log')
            
            plt.tight_layout()
            curves_path = output_dir / f'training_curves_regularized_epoch_{epoch}.png'
            plt.savefig(curves_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Training curves saved to: {curves_path}")
    
    # Final training curves
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(train_losses, label='Training Loss', color='blue', linewidth=2)
    plt.plot(val_losses, label='Validation Loss', color='red', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training vs Validation Loss\n(With Regularization)')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 3, 2)
    plt.plot(learning_rates, label='Learning Rate', color='green', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Schedule')
    plt.legend()
    plt.grid(True)
    plt.yscale('log')
    
    plt.subplot(1, 3, 3)
    gap = [abs(train - val) for train, val in zip(train_losses, val_losses)]
    plt.plot(gap, label='Train-Val Gap', color='orange', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss Difference')
    plt.title('Overfitting Monitor\n(Lower is Better)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    final_curves_path = output_dir / 'final_training_curves_regularized.png'
    plt.savefig(final_curves_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n\ud83d\udcca Final training curves saved to: {final_curves_path}")
    
    return best_val_loss

def main():
    parser = argparse.ArgumentParser(description='Enhanced PPE Training with Regularization')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for training')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-3,
                        help='Weight decay (L2 regularization)')
    parser.add_argument('--patience', type=int, default=10,
                        help='Early stopping patience')
    parser.add_argument('--output_dir', type=str, default='models',
                        help='Output directory for models and plots')
    
    args = parser.parse_args()
    
    print("\ud83e\uddba Enhanced PPE Model Training with Advanced Regularization")
    print("=" * 65)
    
    best_loss = train_with_validation(args)
    
    print(f"\n\ud83c\udfaf Training completed!")
    print(f"Best validation loss: {best_loss:.4f}")

if __name__ == '__main__':
    main()