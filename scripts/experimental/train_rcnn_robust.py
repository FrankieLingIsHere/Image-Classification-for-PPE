#!/usr/bin/env python3
"""
Robust Faster R-CNN training with error handling and progress tracking
"""
import os
import sys
import json
import argparse
import time
from pathlib import Path
import torch
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import torchvision.transforms as T

from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.dataset.ppe_dataset import load_ppe_images_and_annotations

PPE_CLASSES = [
    'background', 'person', 'hard_hat', 'safety_vest', 'safety_gloves',
    'safety_boots', 'eye_protection', 'no_hard_hat', 'no_safety_vest',
    'no_safety_gloves', 'no_safety_boots', 'no_eye_protection'
]

class RobustPPEDataset(Dataset):
    """More robust PPE dataset with better error handling"""
    def __init__(self, data_dir, split='train', transforms=None):
        self.data_dir = Path(data_dir)
        self.split = split
        self.class2idx = {c: i for i, c in enumerate(PPE_CLASSES)}
        self.images_info = load_ppe_images_and_annotations(data_dir, self.class2idx, split)
        self.transforms = transforms or T.Compose([
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        print(f"Loaded {len(self.images_info)} images for {split} split")

    def __len__(self):
        return len(self.images_info)

    def __getitem__(self, idx):
        info = self.images_info[idx]
        img_path = self.data_dir / 'images' / info['file_name']
        boxes = info.get('boxes', [])
        labels = info.get('labels', [])

        # Load image with robust error handling
        try:
            img = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Warning: Could not load {img_path}: {e}")
            # Return a dummy image
            img = Image.new('RGB', (300, 300), color=(128, 128, 128))
            boxes, labels = [], []

        # Convert to tensors
        try:
            if boxes:
                boxes = torch.tensor(boxes, dtype=torch.float32)
                labels = torch.tensor(labels, dtype=torch.int64)
                # Validate boxes
                if len(boxes) > 0:
                    boxes = torch.clamp(boxes, min=0)
                    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
                    valid = (x2 > x1) & (y2 > y1)
                    if valid.sum().item() != len(boxes):
                        boxes = boxes[valid]
                        labels = labels[valid]
            else:
                boxes = torch.zeros((0, 4), dtype=torch.float32)
                labels = torch.zeros((0,), dtype=torch.int64)
        except Exception as e:
            print(f"Warning: Error processing annotations for {img_path}: {e}")
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)

        # Apply transforms
        try:
            img_tensor = self.transforms(img)
        except Exception as e:
            print(f"Warning: Error applying transforms to {img_path}: {e}")
            img_tensor = T.ToTensor()(img)

        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': torch.tensor([idx], dtype=torch.int64)
        }

        return img_tensor, target, info.get('img_id', str(idx))


def collate_fn(batch):
    """Custom collate function"""
    images = []
    targets = []
    ids = []
    for img, target, img_id in batch:
        if img is not None and target is not None:
            images.append(img)
            targets.append(target)
            ids.append(img_id)
    return images, targets, ids


def train_one_epoch(model, optimizer, data_loader, device, epoch, max_epochs):
    """Train for one epoch with progress tracking"""
    model.train()
    total_loss = 0.0
    loss_components = {}
    num_batches = 0
    
    start_time = time.time()
    
    for batch_idx, (imgs, targets, _) in enumerate(data_loader):
        try:
            # Skip empty batches
            if len(imgs) == 0:
                print(f"  Batch {batch_idx}: Empty batch, skipping")
                continue
            
            # Move to device
            imgs = list(img.to(device) for img in imgs)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            # Forward pass
            try:
                loss_dict = model(imgs, targets)
            except Exception as e:
                print(f"  Batch {batch_idx}: Error in model forward pass: {e}")
                continue
            
            losses = sum(loss for loss in loss_dict.values())
            
            # Check for NaN
            if torch.isnan(losses):
                print(f"  Batch {batch_idx}: NaN loss detected, skipping")
                continue
            
            # Backward pass
            optimizer.zero_grad()
            losses.backward()
            
            # Gradient clipping to prevent explosion
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            total_loss += losses.item()
            num_batches += 1
            
            for k, v in loss_dict.items():
                loss_components[k] = loss_components.get(k, 0.0) + float(v.item())
            
            # Print progress
            if (batch_idx + 1) % 5 == 0:
                elapsed = time.time() - start_time
                avg_loss = total_loss / num_batches if num_batches > 0 else 0
                print(f"  Epoch {epoch}/{max_epochs} | Batch {batch_idx+1} | Avg Loss: {avg_loss:.4f} | Time: {elapsed:.1f}s")
        
        except Exception as e:
            print(f"  Batch {batch_idx}: Unexpected error: {e}")
            continue
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    avg_components = {k: v / num_batches for k, v in loss_components.items()} if num_batches > 0 else {}
    
    return avg_loss, avg_components


def main():
    """Main training function"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--augment', action='store_true')
    parser.add_argument('--optimizer', type=str, default='adam')
    parser.add_argument('--step_lr', action='store_true')
    args = parser.parse_args()

    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create datasets
    print("Creating datasets...")
    train_dataset = RobustPPEDataset('data', 'train')
    val_dataset = RobustPPEDataset('data', 'test')
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0
    )

    # Create model
    print("Creating model...")
    model = fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, len(PPE_CLASSES))
    model.to(device)

    # Create optimizer
    if args.optimizer == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    
    # Create scheduler
    if args.step_lr:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    else:
        scheduler = None

    # Training loop
    print(f"\nStarting training for {args.epochs} epochs...\n")
    
    best_loss = float('inf')
    checkpoint_dir = Path('models')
    checkpoint_dir.mkdir(exist_ok=True)
    
    for epoch in range(1, args.epochs + 1):
        print(f"Epoch {epoch}/{args.epochs}")
        
        # Train
        avg_loss, loss_components = train_one_epoch(
            model, optimizer, train_loader, device, epoch, args.epochs
        )
        
        print(f"  Average Loss: {avg_loss:.4f}")
        for k, v in loss_components.items():
            print(f"    {k}: {v:.4f}")
        
        # Save checkpoint if improved
        if avg_loss < best_loss:
            best_loss = avg_loss
            checkpoint_path = checkpoint_dir / 'rcnn_baseline.pth'
            torch.save(model.state_dict(), checkpoint_path)
            print(f"  Saved checkpoint to {checkpoint_path}")
        
        # Step scheduler
        if scheduler:
            scheduler.step()
            print(f"  Learning rate: {optimizer.param_groups[0]['lr']:.6f}")
    
    print(f"\nTraining complete! Best loss: {best_loss:.4f}")
    print(f"Model saved to: {checkpoint_dir / 'rcnn_baseline.pth'}")


if __name__ == '__main__':
    main()
