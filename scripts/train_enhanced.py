#!/usr/bin/env python3
"""
Enhanced PPE Training Script with Validation Monitoring
"""

import os
import sys
import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.models.ssd import SSD300
from src.dataset.ppe_dataset import PPEDataset
from src.models.loss import PPELoss

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
        for batch_idx, (images, annotations_list, filenames) in enumerate(val_loader):
            images = images.to(device)
            
            # Forward pass
            loc_preds, class_preds = model(images)
            
            # Extract boxes and labels from annotations_list
            boxes = [ann['bboxes'] for ann in annotations_list]
            labels = [ann['labels'] for ann in annotations_list]
            
            # Calculate loss
            loss = criterion(loc_preds, class_preds, boxes, labels)
            
            total_loss += loss.item()
            num_batches += 1
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    return avg_loss

def plot_training_curves(train_losses, val_losses, save_path):
    """Plot and save training curves"""
    plt.figure(figsize=(12, 5))
    
    # Plot losses
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss', color='blue')
    plt.plot(val_losses, label='Validation Loss', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    
    # Plot loss difference
    plt.subplot(1, 2, 2)
    if len(train_losses) > 1:
        train_improvement = [(train_losses[0] - loss) / train_losses[0] * 100 for loss in train_losses]
        val_improvement = [(val_losses[0] - loss) / val_losses[0] * 100 for loss in val_losses]
        
        plt.plot(train_improvement, label='Training Improvement %', color='blue')
        plt.plot(val_improvement, label='Validation Improvement %', color='red')
        plt.xlabel('Epoch')
        plt.ylabel('Improvement %')
        plt.title('Training Progress')
        plt.legend()
        plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Training curves saved to: {save_path}")

def train_with_validation(args):
    """Enhanced training with validation monitoring"""
    
    print("🦺 Enhanced PPE Model Training with Validation")
    print("=" * 60)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create datasets
    print("Loading datasets...")
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
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,  # Windows compatibility
        collate_fn=custom_collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=custom_collate_fn
    )
    
    # Initialize model
    print("Initializing model...")
    num_classes = len(train_dataset.ppe_classes)
    model = SSD300(n_classes=num_classes)
    model.to(device)
    
    # Get prior boxes from model
    priors_cxcy = model.priors_cxcy
    
    # Load checkpoint if resuming
    start_epoch = 0
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint.get('epoch', 0) + 1
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        train_losses = checkpoint.get('train_losses', [])
        val_losses = checkpoint.get('val_losses', [])
        print(f"Resuming from epoch {start_epoch}")
    
    # Setup loss and optimizer
    criterion = PPELoss(priors_cxcy=priors_cxcy)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step, gamma=args.lr_gamma)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    print(f"\nStarting training for {args.epochs} epochs...")
    print(f"Output directory: {output_dir}")
    
    # Training loop
    for epoch in range(start_epoch, args.epochs):
        print(f"\n📈 Epoch {epoch}")
        print("-" * 40)
        
        # Training phase
        model.train()
        train_loss = 0.0
        num_batches = 0
        
        for batch_idx, (images, annotations_list, filenames) in enumerate(train_loader):
            images = images.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            loc_preds, class_preds = model(images)
            
            # Extract boxes and labels from annotations_list
            boxes = [ann['bboxes'] for ann in annotations_list]
            labels = [ann['labels'] for ann in annotations_list]
            
            # Calculate loss
            loss = criterion(loc_preds, class_preds, boxes, labels)
            
            # Check for invalid loss values
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"  Warning: Invalid loss detected: {loss.item()}")
                continue
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            num_batches += 1
            
            # Print progress
            if batch_idx % 10 == 0:
                print(f"  Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}")
        
        avg_train_loss = train_loss / num_batches
        train_losses.append(avg_train_loss)
        
        # Validation phase
        print("  Validating...")
        avg_val_loss = validate_model(model, val_loader, criterion, device)
        val_losses.append(avg_val_loss)
        
        # Update learning rate
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        # Print epoch summary
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Val Loss: {avg_val_loss:.4f}")
        print(f"  Learning Rate: {current_lr:.6f}")
        
        # Check for improvement
        is_best = avg_val_loss < best_val_loss
        if is_best:
            best_val_loss = avg_val_loss
            print(f"  ✅ New best validation loss!")
        
        # Save checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
            'best_val_loss': best_val_loss,
            'train_losses': train_losses,
            'val_losses': val_losses
        }
        
        # Save regular checkpoint
        checkpoint_path = output_dir / f"checkpoint_epoch_{epoch}.pth"
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = output_dir / "best_model.pth"
            torch.save(checkpoint, best_path)
            print(f"  💾 Best model saved to: {best_path}")
        
        # Plot training curves every 5 epochs
        if (epoch + 1) % 5 == 0:
            plot_path = output_dir / f"training_curves_epoch_{epoch}.png"
            plot_training_curves(train_losses, val_losses, plot_path)
        
        # Early stopping check
        if args.early_stopping > 0:
            if len(val_losses) >= args.early_stopping:
                recent_losses = val_losses[-args.early_stopping:]
                if all(loss >= best_val_loss for loss in recent_losses[1:]):
                    print(f"\n🛑 Early stopping triggered after {args.early_stopping} epochs without improvement")
                    break
    
    # Final training curves
    final_plot_path = output_dir / "final_training_curves.png"
    plot_training_curves(train_losses, val_losses, final_plot_path)
    
    # Training summary
    print(f"\n🎉 Training completed!")
    print(f"📊 Final Results:")
    print(f"  Best validation loss: {best_val_loss:.4f}")
    print(f"  Final train loss: {train_losses[-1]:.4f}")
    print(f"  Final val loss: {val_losses[-1]:.4f}")
    print(f"  Total improvement: {((train_losses[0] - train_losses[-1]) / train_losses[0] * 100):.1f}%")
    
    return best_val_loss

def main():
    parser = argparse.ArgumentParser(description='Enhanced PPE Training with Validation')
    
    # Data arguments
    parser.add_argument('--data_dir', type=str, default='data',
                       help='Path to dataset directory')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size for training')
    parser.add_argument('--img_size', type=int, default=300,
                       help='Input image size')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=20,
                       help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Initial learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                       help='Weight decay for optimizer')
    parser.add_argument('--lr_step', type=int, default=10,
                       help='Learning rate decay step')
    parser.add_argument('--lr_gamma', type=float, default=0.5,
                       help='Learning rate decay factor')
    
    # Model arguments
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--output_dir', type=str, default='models',
                       help='Output directory for checkpoints')
    parser.add_argument('--early_stopping', type=int, default=5,
                       help='Early stopping patience (0 to disable)')
    
    args = parser.parse_args()
    
    # Start training
    best_loss = train_with_validation(args)
    print(f"\n🏆 Training finished with best validation loss: {best_loss:.4f}")

if __name__ == "__main__":
    main()