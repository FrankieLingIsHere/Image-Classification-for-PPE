#!/usr/bin/env python3
"""
Simple PPE Detection Training Script
Fixed for Windows and current codebase
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.dataset.ppe_dataset import PPEDataset
from src.models.ssd import SSD300  
from src.models.loss import MultiBoxLoss

def collate_fn(batch):
    """
    Custom collate function for PPE dataset
    """
    images = []
    boxes = []
    labels = []
    filenames = []
    
    for item in batch:
        if len(item) >= 4:
            images.append(item[0])
            boxes.append(item[1])
            labels.append(item[2])
            filenames.append(item[3])
        else:
            print(f"Warning: Batch item has {len(item)} elements, expected 4")
            continue
    
    # Stack images
    images = torch.stack(images, 0)
    
    return images, boxes, labels, filenames

def main():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='data')
    parser.add_argument('--num_classes', type=int, default=13) 
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--device', default='auto')
    args = parser.parse_args()
    
    # Setup device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"🚀 PPE Detection Training")
    print(f"Device: {device}")
    print(f"Classes: {args.num_classes}")
    print(f"Batch size: {args.batch_size}")
    
    # Create datasets
    print("\n📁 Loading datasets...")
    
    try:
        train_dataset = PPEDataset(
            data_dir=args.data_dir,
            split='train',
            img_size=300
        )
        
        val_dataset = PPEDataset(
            data_dir=args.data_dir,
            split='val', 
            img_size=300
        )
        
        print(f"Training samples: {len(train_dataset)}")
        print(f"Validation samples: {len(val_dataset)}")
        
    except Exception as e:
        print(f"❌ Error loading datasets: {e}")
        return
    
    # Create data loaders with Windows-safe settings
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,  # Windows safe
        pin_memory=False,
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,  # Windows safe
        pin_memory=False,
        collate_fn=collate_fn
    )
    
    # Create model
    print("\n🧠 Creating model...")
    
    try:
        model = SSD300(n_classes=args.num_classes)
        model.to(device)
        
        # Get prior boxes
        priors_cxcy = model.create_prior_boxes()
        priors_cxcy = priors_cxcy.to(device)
        
        print(f"Model created successfully")
        print(f"Prior boxes shape: {priors_cxcy.shape}")
        print(f"Expected: (8732, 4)")
        
    except Exception as e:
        print(f"❌ Error creating model: {e}")
        return
    
    # Create loss function
    criterion = MultiBoxLoss(
        priors_cxcy=priors_cxcy,
        threshold=0.5,
        neg_pos_ratio=3,
        alpha=1.0
    )
    
    # Create optimizer
    optimizer = optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=0.9,
        weight_decay=0.0005
    )
    
    # Create output directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    
    print(f"\n🏃 Starting training...")
    
    # Training loop
    model.train()
    best_loss = float('inf')
    
    try:
        for epoch in range(args.epochs):
            print(f"\nEpoch {epoch + 1}/{args.epochs}")
            print("-" * 40)
            
            total_loss = 0
            num_batches = len(train_loader)
            
            for batch_idx, (images, boxes, labels, filenames) in enumerate(train_loader):
                
                # Move to device
                images = images.to(device)
                boxes = [box.to(device) for box in boxes]
                labels = [label.to(device) for label in labels]
                
                # Debug info for first batch
                if batch_idx == 0 and epoch == 0:
                    print(f"  Debug - Images: {images.shape}")
                    print(f"  Debug - Batch size: {len(boxes)}")
                    if len(boxes) > 0:
                        print(f"  Debug - First box tensor: {boxes[0].shape if hasattr(boxes[0], 'shape') else 'Not tensor'}")
                
                # Zero gradients
                optimizer.zero_grad()
                
                # Forward pass
                try:
                    predicted_locs, predicted_scores = model(images)
                    
                    # Debug shapes for first batch
                    if batch_idx == 0 and epoch == 0:
                        print(f"  Debug - Pred locs: {predicted_locs.shape}")
                        print(f"  Debug - Pred scores: {predicted_scores.shape}")
                    
                    # Calculate loss
                    loss = criterion(predicted_locs, predicted_scores, boxes, labels)
                    
                    # Backward pass
                    loss.backward()
                    optimizer.step()
                    
                    total_loss += loss.item()
                    
                    # Print progress
                    if batch_idx % 10 == 0:
                        progress = 100.0 * batch_idx / num_batches
                        print(f"  Batch {batch_idx}/{num_batches} ({progress:.1f}%) | Loss: {loss.item():.4f}")
                        
                except Exception as e:
                    print(f"❌ Error in batch {batch_idx}: {e}")
                    if 'shape' in str(e).lower():
                        print(f"  Images: {images.shape}")
                        print(f"  Predicted locs: {predicted_locs.shape if 'predicted_locs' in locals() else 'Not created'}")
                        print(f"  Predicted scores: {predicted_scores.shape if 'predicted_scores' in locals() else 'Not created'}")
                        print(f"  Prior boxes: {priors_cxcy.shape}")
                    raise e
            
            avg_loss = total_loss / num_batches
            print(f"  Average Loss: {avg_loss:.4f}")
            
            # Save best model
            if avg_loss < best_loss:
                best_loss = avg_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': avg_loss,
                }, 'models/best_model.pth')
                print(f"  🎯 New best model saved! Loss: {avg_loss:.4f}")
            
            # Save checkpoint every 5 epochs
            if (epoch + 1) % 5 == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': avg_loss,
                }, f'models/checkpoint_epoch_{epoch + 1}.pth')
                print(f"  💾 Checkpoint saved")
    
    except KeyboardInterrupt:
        print("\n⏹️ Training interrupted by user")
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        raise e
    
    print(f"\n✅ Training completed!")
    print(f"Best loss: {best_loss:.4f}")
    print(f"Models saved in: models/")
    
    # Test inference
    print(f"\n🧪 Testing inference...")
    
    try:
        model.eval()
        with torch.no_grad():
            # Get a sample from validation set
            if len(val_loader) > 0:
                images, boxes, labels, filenames = next(iter(val_loader))
                images = images.to(device)
                
                predicted_locs, predicted_scores = model(images)
                print(f"  ✅ Inference test successful!")
                print(f"  Input: {images.shape}")
                print(f"  Output locs: {predicted_locs.shape}")
                print(f"  Output scores: {predicted_scores.shape}")
            else:
                print("  ⚠️ No validation data available for testing")
    
    except Exception as e:
        print(f"  ❌ Inference test failed: {e}")

if __name__ == '__main__':
    main()