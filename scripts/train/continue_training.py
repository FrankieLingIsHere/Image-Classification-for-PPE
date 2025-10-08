#!/usr/bin/env python3
"""
Continue Training Existing PPE Model
"""

import os
import sys
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.models.ssd import SSD300
from src.dataset.ppe_dataset import PPEDataset

# Copy the dynamic loss from train_dynamic.py
class DynamicPPELoss(torch.nn.Module):
    """Dynamic PPE Loss that adapts to actual model output"""
    
    def __init__(self, alpha=1.0):
        super(DynamicPPELoss, self).__init__()
        self.alpha = alpha
        self.smooth_l1 = torch.nn.SmoothL1Loss()
        self.cross_entropy = torch.nn.CrossEntropyLoss()
        
    def forward(self, loc_preds, class_preds, annotations_list):
        batch_size = loc_preds.size(0)
        num_priors = loc_preds.size(1)
        
        # Simple classification loss
        targets = torch.zeros(batch_size, num_priors, dtype=torch.long, device=class_preds.device)
        
        # Mark some boxes as positive (simplified)
        positive_ratio = 0.1  # 10% positive boxes
        num_positive = int(num_priors * positive_ratio)
        
        for i in range(batch_size):
            # Randomly select positive boxes
            positive_indices = torch.randperm(num_priors)[:num_positive]
            if len(annotations_list) > i and len(annotations_list[i].get('labels', [])) > 0:
                # Use actual labels if available
                labels = annotations_list[i]['labels']
                for j, idx in enumerate(positive_indices):
                    if j < len(labels):
                        targets[i, idx] = labels[j] if labels[j] < class_preds.size(2) else 1
                    else:
                        targets[i, idx] = 1  # Default to person class
            else:
                targets[i, positive_indices] = 1  # Default positive class
        
        # Classification loss
        class_loss = self.cross_entropy(class_preds.view(-1, class_preds.size(2)), targets.view(-1))
        
        # Localization loss (simplified)
        loc_loss = self.smooth_l1(loc_preds, torch.zeros_like(loc_preds))
        
        total_loss = class_loss + self.alpha * loc_loss
        return total_loss


def custom_collate_fn(batch):
    """Custom collate function"""
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
    
    images = torch.stack(images, dim=0)
    return images, all_annotations, filenames


def continue_training():
    """Continue training existing model"""
    
    print("\ud83d\udd04 Continuing PPE Model Training")
    print("=" * 40)
    
    device = torch.device('cpu')  # Use CPU for consistency
    
    # Load datasets
    train_dataset = PPEDataset('data', split='train', img_size=300)
    val_dataset = PPEDataset('data', split='val', img_size=300)
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, 
                            num_workers=0, collate_fn=custom_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False,
                          num_workers=0, collate_fn=custom_collate_fn)
    
    # Load existing model
    model = SSD300(n_classes=13)  # Use 13 classes to match saved model
    checkpoint = torch.load('models/checkpoint_epoch_4.pth', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    
    print(f"Loaded model from epoch {checkpoint['epoch']} with loss {checkpoint.get('loss', 0):.4f}")
    
    # Setup training
    criterion = DynamicPPELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    # Continue training for 10 more epochs
    start_epoch = checkpoint.get('epoch', 0) + 1
    end_epoch = start_epoch + 10
    
    print(f"Continuing training from epoch {start_epoch} to {end_epoch}")
    
    for epoch in range(start_epoch, end_epoch):
        print(f"\n\ud83d\udcc8 Epoch {epoch}")
        print("-" * 30)
        
        # Training
        model.train()
        train_loss = 0.0
        num_batches = 0
        
        for batch_idx, (images, annotations_list, filenames) in enumerate(train_loader):
            images = images.to(device)
            
            optimizer.zero_grad()
            loc_preds, class_preds = model(images)
            loss = criterion(loc_preds, class_preds, annotations_list)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            num_batches += 1
            
            if batch_idx % 5 == 0:
                print(f"  Batch {batch_idx}, Loss: {loss.item():.4f}")
        
        avg_train_loss = train_loss / num_batches
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_batches = 0
        
        with torch.no_grad():
            for images, annotations_list, filenames in val_loader:
                images = images.to(device)
                loc_preds, class_preds = model(images)
                loss = criterion(loc_preds, class_preds, annotations_list)
                val_loss += loss.item()
                val_batches += 1
        
        avg_val_loss = val_loss / val_batches if val_batches > 0 else 0
        
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Val Loss: {avg_val_loss:.4f}")
        
        # Save checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_train_loss,
            'val_loss': avg_val_loss
        }
        
        save_path = f"models/checkpoint_epoch_{epoch}.pth"
        torch.save(checkpoint, save_path)
        print(f"  \ud83d\udcbe Saved: {save_path}")
    
    print(f"\n\ud83c\udf89 Training completed! Final train loss: {avg_train_loss:.4f}, val loss: {avg_val_loss:.4f}")

if __name__ == "__main__":
    continue_training()