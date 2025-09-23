#!/usr/bin/env python3
"""
Training script for SSD PPE Detection Model

This script trains an SSD model to detect Personal Protective Equipment (PPE)
in construction environments for OSHA compliance monitoring.
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

from src.dataset.ppe_dataset import PPEDataset, create_sample_data_structure
from src.models.ssd import build_ssd_model
from src.models.loss import PPELoss, create_prior_boxes
from src.utils.utils import save_checkpoint, load_checkpoint, AverageMeter, adjust_learning_rate


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train SSD PPE Detection Model')
    
    # Data arguments
    parser.add_argument('--data_dir', type=str, default='data',
                       help='Path to dataset directory')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size for training')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers')
    
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
    parser.add_argument('--lr_decay_epochs', type=int, nargs='+', default=[80, 100],
                       help='Epochs at which to decay learning rate')
    parser.add_argument('--lr_decay_factor', type=float, default=0.1,
                       help='Learning rate decay factor')
    
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
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"Memory: {torch.cuda.get_device_properties(device).total_memory / 1e9:.1f} GB")
    
    return device


def create_data_loaders(args):
    """Create training and validation data loaders"""
    
    # Check if data directory exists and has the required structure
    if not os.path.exists(args.data_dir):
        print(f"Data directory {args.data_dir} not found. Creating sample structure...")
        create_sample_data_structure(args.data_dir)
        print(f"Please add your data to {args.data_dir} following the structure in README.md")
        return None, None
    
    # Create datasets
    try:
        train_dataset = PPEDataset(
            data_dir=args.data_dir,
            split='train',
            img_size=args.img_size
        )
        
        val_dataset = PPEDataset(
            data_dir=args.data_dir,
            split='val',
            img_size=args.img_size
        )
        
        print(f"Train dataset: {len(train_dataset)} images")
        print(f"Validation dataset: {len(val_dataset)} images")
        
        if len(train_dataset) == 0:
            print("No training data found. Please add images and annotations to the data directory.")
            return None, None
        
    except Exception as e:
        print(f"Error creating datasets: {e}")
        return None, None
    
    # Create data loaders with Windows-compatible settings
    # Use num_workers=0 to avoid multiprocessing issues on Windows
    num_workers = 0 if os.name == 'nt' else args.num_workers
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn
    ) if len(val_dataset) > 0 else None
    
    return train_loader, val_loader


def collate_fn(batch):
    """Custom collate function for batching"""
    images = []
    targets = []
    filenames = []
    
    for image, target, filename in batch:
        images.append(image)
        targets.append(target)
        filenames.append(filename)
    
    images = torch.stack(images, 0)
    
    # Extract boxes and labels for loss computation
    boxes = [target['bboxes'] for target in targets]
    labels = [target['labels'] for target in targets]
    
    return images, boxes, labels, filenames


def train_epoch(model, train_loader, criterion, optimizer, device, epoch, args, writer=None):
    """Train for one epoch"""
    model.train()
    
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    
    end = time.time()
    
    for i, (images, boxes, labels, filenames) in enumerate(train_loader):
        # Measure data loading time
        data_time.update(time.time() - end)
        
        # Move to device
        images = images.to(device)
        boxes = [box.to(device) for box in boxes]
        labels = [label.to(device) for label in labels]
        
        # Forward pass
        predicted_locs, predicted_scores = model(images)
        
        # Compute loss
        loss = criterion(predicted_locs, predicted_scores, boxes, labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Update metrics
        losses.update(loss.item(), images.size(0))
        batch_time.update(time.time() - end)
        end = time.time()
        
        # Print progress
        if i % args.print_freq == 0:
            print(f'Epoch: [{epoch}][{i}/{len(train_loader)}] '
                  f'Time {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                  f'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                  f'Loss {losses.val:.4f} ({losses.avg:.4f})')
        
        # Log to tensorboard
        if writer:
            global_step = epoch * len(train_loader) + i
            writer.add_scalar('Train/Loss', loss.item(), global_step)
    
    return losses.avg


def validate(model, val_loader, criterion, device, epoch, args, writer=None):
    """Validate the model"""
    model.eval()
    
    batch_time = AverageMeter()
    losses = AverageMeter()
    
    with torch.no_grad():
        end = time.time()
        
        for i, (images, boxes, labels, filenames) in enumerate(val_loader):
            # Move to device
            images = images.to(device)
            boxes = [box.to(device) for box in boxes]
            labels = [label.to(device) for label in labels]
            
            # Forward pass
            predicted_locs, predicted_scores = model(images)
            
            # Compute loss
            loss = criterion(predicted_locs, predicted_scores, boxes, labels)
            
            # Update metrics
            losses.update(loss.item(), images.size(0))
            batch_time.update(time.time() - end)
            end = time.time()
        
        print(f'Validation: [{epoch}] '
              f'Time {batch_time.avg:.3f} '
              f'Loss {losses.avg:.4f}')
        
        # Log to tensorboard
        if writer:
            writer.add_scalar('Val/Loss', losses.avg, epoch)
    
    return losses.avg


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
    
    # Setup tensorboard
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = os.path.join(args.log_dir, f'ppe_ssd_{timestamp}')
    writer = SummaryWriter(log_dir)
    
    # Save configuration
    config_path = os.path.join(args.save_dir, 'config.yaml')
    with open(config_path, 'w') as f:
        yaml.dump(vars(args), f, default_flow_style=False)
    
    # Create data loaders
    train_loader, val_loader = create_data_loaders(args)
    if train_loader is None:
        print("Failed to create data loaders. Exiting.")
        return
    
    # Create model
    model = build_ssd_model(num_classes=args.num_classes)
    model = model.to(device)
    
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Create loss function
    priors_cxcy = create_prior_boxes()
    criterion = PPELoss(
        priors_cxcy=priors_cxcy.to(device),
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
    
    # Resume from checkpoint if specified
    start_epoch = 0
    best_loss = float('inf')
    if args.resume:
        start_epoch, best_loss = load_checkpoint(args.resume, model, optimizer)
    
    # Training loop
    print(f"Starting training from epoch {start_epoch}")
    train_losses = []
    val_losses = []
    
    for epoch in range(start_epoch, args.epochs):
        # Adjust learning rate
        lr = adjust_learning_rate(optimizer, epoch, args.lr, args.lr_decay_epochs, args.lr_decay_factor)
        print(f'Epoch {epoch}: Learning rate = {lr}')
        
        # Train for one epoch
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, epoch, args, writer)
        train_losses.append(train_loss)
        
        # Validate
        val_loss = None
        if val_loader is not None:
            val_loss = validate(model, val_loader, criterion, device, epoch, args, writer)
            val_losses.append(val_loss)
        
        # Save checkpoint
        is_best = val_loss is not None and val_loss < best_loss
        if val_loss is not None:
            best_loss = min(val_loss, best_loss)
        
        if epoch % args.save_freq == 0 or is_best:
            checkpoint_path = os.path.join(args.save_dir, f'checkpoint_epoch_{epoch}.pth')
            save_checkpoint({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_loss': best_loss,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'config': vars(args)
            }, checkpoint_path)
            
            if is_best:
                best_path = os.path.join(args.save_dir, 'best_model.pth')
                save_checkpoint({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_loss': best_loss,
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'config': vars(args)
                }, best_path)
    
    # Save final model
    final_path = os.path.join(args.save_dir, 'final_model.pth')
    save_checkpoint({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_loss': best_loss,
        'config': vars(args)
    }, final_path)
    
    writer.close()
    print("Training completed!")
    
    # Print summary
    print(f"\nTraining Summary:")
    print(f"Total epochs: {args.epochs}")
    print(f"Final training loss: {train_losses[-1]:.4f}")
    if val_losses:
        print(f"Best validation loss: {best_loss:.4f}")
    print(f"Model saved to: {args.save_dir}")
    print(f"Logs saved to: {log_dir}")


if __name__ == '__main__':
    main()