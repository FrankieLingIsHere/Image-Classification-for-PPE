#!/usr/bin/env python3
"""
Fixed training script for PPE detection with proper error handling
"""

import os
import sys
import argparse
import time
from datetime import datetime
import yaml

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.dataset.ppe_dataset import PPEDataset
from src.models.ssd import SSD300
from src.models.loss import MultiBoxLoss

def create_model(num_classes=13):
    """Create SSD300 model"""
    model = SSD300(n_classes=num_classes)
    return model

def get_prior_boxes(model):
    """Get prior boxes from model"""
    return model.create_prior_boxes()

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train SSD PPE Detection Model')
    
    # Data arguments
    parser.add_argument('--data_dir', type=str, default='data',
                       help='Path to dataset directory')
    parser.add_argument('--batch_size', type=int, default=4,
                       help='Batch size for training')
    parser.add_argument('--num_workers', type=int, default=0,
                       help='Number of data loading workers (0 for Windows)')
    
    # Model arguments
    parser.add_argument('--img_size', type=int, default=300,
                       help='Input image size')
    parser.add_argument('--num_classes', type=int, default=13,
                       help='Number of classes including background')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-3,
                       help='Initial learning rate')
    parser.add_argument('--momentum', type=float, default=0.9,
                       help='Momentum for SGD optimizer')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                       help='Weight decay')
    
    # Loss arguments
    parser.add_argument('--alpha', type=float, default=1.0,
                       help='Weighting factor for localization loss')
    parser.add_argument('--neg_pos_ratio', type=int, default=3,
                       help='Ratio of negative to positive samples')
    
    # Checkpoint arguments
    parser.add_argument('--resume', type=str, default='',
                       help='Path to checkpoint to resume from')
    parser.add_argument('--save_dir', type=str, default='models',
                       help='Directory to save model checkpoints')
    parser.add_argument('--save_freq', type=int, default=10,
                       help='Frequency of saving checkpoints')
    
    # Logging arguments
    parser.add_argument('--log_dir', type=str, default='logs',
                       help='Directory for tensorboard logs')
    parser.add_argument('--print_freq', type=int, default=100,
                       help='Frequency of printing training progress')
    
    # Other arguments
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (auto, cpu, cuda)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    return parser.parse_args()

def setup_device(device_arg):
    """Setup device for training"""
    if device_arg == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device_arg)
    
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    return device

def save_checkpoint(model, optimizer, epoch, loss, save_dir, filename='checkpoint_latest.pth'):
    """Save model checkpoint"""
    os.makedirs(save_dir, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'timestamp': datetime.now().isoformat()
    }
    
    filepath = os.path.join(save_dir, filename)
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved: {filepath}")

def load_checkpoint(filepath, model, optimizer=None):
    """Load model checkpoint"""
    checkpoint = torch.load(filepath, map_location='cpu')
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    epoch = checkpoint.get('epoch', 0)
    loss = checkpoint.get('loss', float('inf'))
    
    print(f"Checkpoint loaded from epoch {epoch}")
    return epoch, loss

def train_epoch(model, train_loader, criterion, optimizer, device, epoch, args, writer=None):
    """Train for one epoch"""
    model.train()
    
    total_loss = 0
    total_batches = len(train_loader)
    
    print(f"\nEpoch {epoch + 1}/{args.epochs}")
    print("-" * 50)
    
    for batch_idx, (images, boxes, labels, filenames) in enumerate(train_loader):
        # Move to device
        images = images.to(device)
        boxes = [box.to(device) for box in boxes]
        labels = [label.to(device) for label in labels]
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        try:
            predicted_locs, predicted_scores = model(images)
            
            # Debug prints for the first batch
            if batch_idx == 0 and epoch == 0:
                print(f"Debug info:")
                print(f"  Images shape: {images.shape}")
                print(f"  Predicted locs shape: {predicted_locs.shape}")
                print(f"  Predicted scores shape: {predicted_scores.shape}")
                print(f"  Number of boxes in batch: {[len(box) for box in boxes]}")
                print(f"  Number of labels in batch: {[len(label) for label in labels]}")
            
            # Calculate loss
            loss = criterion(predicted_locs, predicted_scores, boxes, labels)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Update statistics
            total_loss += loss.item()
            
            # Print progress
            if batch_idx % args.print_freq == 0:
                progress = 100.0 * batch_idx / total_batches
                print(f"  Batch {batch_idx}/{total_batches} ({progress:.1f}%) | Loss: {loss.item():.4f}")
            
            # Log to tensorboard
            if writer:
                step = epoch * total_batches + batch_idx
                writer.add_scalar('Loss/Train_Batch', loss.item(), step)
        
        except Exception as e:
            print(f"Error in batch {batch_idx}: {e}")
            print(f"  Images shape: {images.shape}")
            if len(predicted_locs.shape) > 0:
                print(f"  Predicted locs shape: {predicted_locs.shape}")
            if len(predicted_scores.shape) > 0:
                print(f"  Predicted scores shape: {predicted_scores.shape}")
            raise e
    
    avg_loss = total_loss / total_batches
    print(f"  Average Loss: {avg_loss:.4f}")
    
    return avg_loss

def validate_epoch(model, val_loader, criterion, device, epoch, args, writer=None):
    """Validate for one epoch"""
    model.eval()
    
    total_loss = 0
    total_batches = len(val_loader)
    
    with torch.no_grad():
        for batch_idx, (images, boxes, labels, filenames) in enumerate(val_loader):
            # Move to device
            images = images.to(device)
            boxes = [box.to(device) for box in boxes]
            labels = [label.to(device) for label in labels]
            
            # Forward pass
            predicted_locs, predicted_scores = model(images)
            
            # Calculate loss
            loss = criterion(predicted_locs, predicted_scores, boxes, labels)
            
            total_loss += loss.item()
    
    avg_loss = total_loss / total_batches
    print(f"  Validation Loss: {avg_loss:.4f}")
    
    if writer:
        writer.add_scalar('Loss/Validation', avg_loss, epoch)
    
    return avg_loss

def main():
    """Main training function"""
    args = parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    
    # Setup device
    device = setup_device(args.device)
    
    # Create directories
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    print("🚀 Starting PPE Detection Training")
    print("=" * 50)
    print(f"Dataset: {args.data_dir}")
    print(f"Classes: {args.num_classes}")
    print(f"Batch size: {args.batch_size}")
    print(f"Epochs: {args.epochs}")
    print(f"Device: {device}")
    
    # Create datasets
    print("\n📁 Loading datasets...")
    
    train_dataset = PPEDataset(
        data_dir=args.data_dir,
        split='train',
        img_size=args.img_size,
        augment=True
    )
    
    val_dataset = PPEDataset(
        data_dir=args.data_dir,
        split='val',
        img_size=args.img_size,
        augment=False
    )
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=False,  # Set to False for CPU
        collate_fn=train_dataset.collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False,  # Set to False for CPU
        collate_fn=val_dataset.collate_fn
    )
    
    # Create model
    print("\n🧠 Creating model...")
    model = create_model(args.num_classes)
    model.to(device)
    
    # Get prior boxes
    priors_cxcy = get_prior_boxes(model)
    priors_cxcy = priors_cxcy.to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Prior boxes shape: {priors_cxcy.shape}")
    
    # Create criterion
    criterion = MultiBoxLoss(
        priors_cxcy=priors_cxcy,
        threshold=0.5,
        neg_pos_ratio=args.neg_pos_ratio,
        alpha=args.alpha
    )
    
    # Create optimizer
    optimizer = optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )
    
    # Setup tensorboard
    writer = SummaryWriter(args.log_dir)
    
    # Resume training if specified
    start_epoch = 0
    if args.resume and os.path.exists(args.resume):
        start_epoch, _ = load_checkpoint(args.resume, model, optimizer)
    
    # Training loop
    print(f"\n🏃 Starting training from epoch {start_epoch + 1}...")
    best_val_loss = float('inf')
    
    try:
        for epoch in range(start_epoch, args.epochs):
            
            # Train
            train_loss = train_epoch(model, train_loader, criterion, optimizer, device, epoch, args, writer)
            
            # Validate
            val_loss = validate_epoch(model, val_loader, criterion, device, epoch, args, writer)
            
            # Log epoch results
            writer.add_scalar('Loss/Train_Epoch', train_loss, epoch)
            writer.add_scalar('Loss/Validation_Epoch', val_loss, epoch)
            
            # Save checkpoint
            if (epoch + 1) % args.save_freq == 0:
                save_checkpoint(model, optimizer, epoch, train_loss, args.save_dir, 
                              f'checkpoint_epoch_{epoch + 1}.pth')
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(model, optimizer, epoch, val_loss, args.save_dir, 
                              'best_model.pth')
                print(f"  🎯 New best model! Validation loss: {val_loss:.4f}")
            
            # Always save latest
            save_checkpoint(model, optimizer, epoch, train_loss, args.save_dir)
            
    except KeyboardInterrupt:
        print("\n⏹️ Training interrupted by user")
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        raise e
    
    finally:
        writer.close()
        print(f"\n✅ Training completed!")
        print(f"Best validation loss: {best_val_loss:.4f}")
        print(f"Models saved in: {args.save_dir}")
        print(f"Logs saved in: {args.log_dir}")

if __name__ == '__main__':
    main()