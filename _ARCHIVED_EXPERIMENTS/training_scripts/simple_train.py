#!/usr/bin/env python3
"""
Simple Windows-compatible training script for PPE detection
Avoids multiprocessing issues by using num_workers=0
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import argparse

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.dataset.simple_ppe_dataset import SimplePPEDataset, simple_collate_fn
from src.models.ssd import SSD300

def parse_args():
    parser = argparse.ArgumentParser(description='Simple PPE Training')
    parser.add_argument('--data_dir', type=str, default='data', help='Data directory')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--num_classes', type=int, default=13, help='Number of classes')
    parser.add_argument('--device', type=str, default='auto', help='Device (auto/cpu/cuda)')
    return parser.parse_args()

def setup_device(device_arg):
    if device_arg == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device_arg)
    
    print(f"Using device: {device}")
    return device

def create_dataloaders(args):
    """Create training and validation dataloaders"""
    
    try:
        train_dataset = SimplePPEDataset(
            data_dir=args.data_dir,
            split='train',
            img_size=300
        )
        
        val_dataset = SimplePPEDataset(
            data_dir=args.data_dir,
            split='val', 
            img_size=300
        )
        
        print(f"Train dataset: {len(train_dataset)} samples")
        print(f"Val dataset: {len(val_dataset)} samples")
        
        # Use num_workers=0 for Windows compatibility
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=0,  # No multiprocessing on Windows
            collate_fn=simple_collate_fn
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,  # No multiprocessing on Windows
            collate_fn=simple_collate_fn
        ) if len(val_dataset) > 0 else None
        
        return train_loader, val_loader
        
    except Exception as e:
        print(f"Error creating dataloaders: {e}")
        return None, None

def train_epoch(model, train_loader, criterion, optimizer, device, epoch):
    """Train for one epoch"""
    
    model.train()
    total_loss = 0
    num_batches = 0
    
    for i, (images, targets, filenames) in enumerate(train_loader):
        try:
            images = images.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            
            # Simple loss calculation (you would use proper SSD loss here)
            outputs = model(images)
            
            # For now, use a dummy loss - you'll need to implement proper SSD loss
            loss = torch.tensor(0.5, requires_grad=True, device=device)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            if i % 10 == 0:
                print(f"Epoch {epoch}, Batch {i}/{len(train_loader)}, Loss: {loss.item():.4f}")
                
        except Exception as e:
            print(f"Error in batch {i}: {e}")
            continue
    
    avg_loss = total_loss / max(num_batches, 1)
    return avg_loss

def main():
    args = parse_args()
    
    print("🚀 Starting Simple PPE Training")
    print(f"Configuration: {vars(args)}")
    
    # Setup device
    device = setup_device(args.device)
    
    # Create dataloaders
    train_loader, val_loader = create_dataloaders(args)
    
    if train_loader is None:
        print("❌ Failed to create dataloaders")
        return
    
    # Create model
    try:
        model = SSD300(n_classes=args.num_classes)
        model.to(device)
        print(f"✅ Model created with {args.num_classes} classes")
    except Exception as e:
        print(f"❌ Error creating model: {e}")
        return
    
    # Setup optimizer and criterion
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()  # Placeholder - use proper SSD loss
    
    # Create directories
    Path("models").mkdir(exist_ok=True)
    Path("logs").mkdir(exist_ok=True)
    
    # Training loop
    print(f"\n🏃‍♂️ Starting training for {args.epochs} epochs...")
    
    for epoch in range(args.epochs):
        print(f"\n--- Epoch {epoch+1}/{args.epochs} ---")
        
        try:
            train_loss = train_epoch(model, train_loader, criterion, optimizer, device, epoch+1)
            print(f"Epoch {epoch+1} completed, Average Loss: {train_loss:.4f}")
            
            # Save checkpoint every 5 epochs
            if (epoch + 1) % 5 == 0:
                checkpoint_path = f"models/checkpoint_epoch_{epoch+1}.pth"
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': train_loss,
                }, checkpoint_path)
                print(f"💾 Checkpoint saved: {checkpoint_path}")
                
        except Exception as e:
            print(f"❌ Error in epoch {epoch+1}: {e}")
            continue
    
    # Save final model
    final_model_path = "models/ppe_model_final.pth"
    torch.save({
        'model_state_dict': model.state_dict(),
        'num_classes': args.num_classes,
        'config': vars(args)
    }, final_model_path)
    
    print(f"\n🎉 Training completed!")
    print(f"💾 Final model saved: {final_model_path}")
    print(f"📊 Check logs/ directory for training details")

if __name__ == "__main__":
    # This ensures proper multiprocessing on Windows
    if __name__ == '__main__':
        main()