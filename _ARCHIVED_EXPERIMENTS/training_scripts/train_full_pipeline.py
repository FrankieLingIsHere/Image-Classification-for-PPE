#!/usr/bin/env python3
"""
FULL TRAINING PIPELINE: SSL Pretraining + Enhanced Detection
Implements Option D with all stages:
1. Self-supervised pretraining
2. Multi-task learning (detection + segmentation)
3. Spatial constraints + Context awareness
4. Full end-to-end retraining
"""

import torch
import torch.optim as optim
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
from tqdm import tqdm
from PIL import Image
import json
from datetime import datetime
import sys
import os

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.train.ssl_pretraining import pretrain_ssl
from src.models.enhanced_ppe_detector import load_enhanced_detector


class PPEDatasetWithSegmentation(Dataset):
    """
    PPE dataset with semantic segmentation masks for multi-task learning.
    Includes comprehensive data augmentation for better training.
    """
    PPE_CLASSES = [
        'background', 'person', 'hard_hat', 'safety_vest', 'safety_gloves',
        'safety_boots', 'eye_protection', 'no_hard_hat', 'no_safety_vest',
        'no_safety_gloves', 'no_safety_boots', 'no_eye_protection'
    ]
    
    def __init__(self, data_dir='data', split='train', transforms=None, image_size=640, augment=True):
        self.data_dir = Path(data_dir)
        self.split = split
        self.image_size = image_size  # Resize all images to this size
        self.augment = augment and split == 'train'  # Only augment training set
        self.class2idx = {c: i for i, c in enumerate(self.PPE_CLASSES)}
        
        # Load split file
        split_file = self.data_dir / 'splits' / f'{split}.txt'
        with open(split_file) as f:
            self.image_names = [line.strip() for line in f if line.strip()]
        
        # Create augmentation pipeline
        if self.augment:
            # Training augmentation: aggressive augmentation for better generalization
            self.augmentation = T.Compose([
                T.RandomHorizontalFlip(p=0.5),
                T.RandomVerticalFlip(p=0.1),
                T.RandomRotation(degrees=15),
                T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
                T.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
                T.RandomPerspective(distortion_scale=0.2, p=0.3),
                T.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
            ])
        else:
            self.augmentation = None
        
        # Default transforms with resizing
        if transforms is None:
            transforms = T.Compose([
                T.Resize((image_size, image_size), Image.BILINEAR),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
            ])
        
        self.transforms = transforms
        
        aug_str = " (with augmentation)" if self.augment else ""
        print(f"[Dataset] Loaded {len(self.image_names)} images for split '{split}' (resized to {image_size}x{image_size}){aug_str}")
    
    def __len__(self):
        return len(self.image_names)
    
    def __getitem__(self, idx):
        img_name = self.image_names[idx]
        
        # Load image
        img_path = self.data_dir / 'images' / img_name
        img = Image.open(img_path).convert('RGB')
        
        # Get original image dimensions for bbox scaling
        orig_w, orig_h = img.size
        
        # Load annotations
        anno_path = self.data_dir / 'annotations' / img_name.replace('.jpg', '.xml').replace('.png', '.xml')
        boxes = []
        labels = []
        
        if anno_path.exists():
            tree = ET.parse(anno_path)
            root = tree.getroot()
            
            for obj in root.findall('object'):
                name_elem = obj.find('name')
                bndbox = obj.find('bndbox')
                
                if name_elem is not None and bndbox is not None:
                    class_name = name_elem.text
                    if class_name in self.class2idx:
                        xmin = float(bndbox.find('xmin').text)
                        ymin = float(bndbox.find('ymin').text)
                        xmax = float(bndbox.find('xmax').text)
                        ymax = float(bndbox.find('ymax').text)
                        
                        boxes.append([xmin, ymin, xmax, ymax])
                        labels.append(self.class2idx[class_name])
        
        # Apply augmentation before resizing (if enabled)
        if self.augment:
            img = self.augmentation(img)
        
        # Resize image (PIL handles resizing)
        img = img.resize((self.image_size, self.image_size), Image.BILINEAR)
        
        # Scale bounding boxes to new image size and filter invalid/zero-area boxes
        scale_x = self.image_size / orig_w if orig_w > 0 else 1.0
        scale_y = self.image_size / orig_h if orig_h > 0 else 1.0

        scaled_boxes = []
        scaled_labels = []
        for box, label in zip(boxes, labels):
            x1, y1, x2, y2 = box
            x1_scaled = max(0, min(self.image_size, x1 * scale_x))
            y1_scaled = max(0, min(self.image_size, y1 * scale_y))
            x2_scaled = max(0, min(self.image_size, x2 * scale_x))
            y2_scaled = max(0, min(self.image_size, y2 * scale_y))

            # Only keep boxes with positive area (and at least 1px in each dimension)
            if (x2_scaled > x1_scaled + 0.0) and (y2_scaled > y1_scaled + 0.0):
                if (x2_scaled - x1_scaled) >= 1.0 and (y2_scaled - y1_scaled) >= 1.0:
                    scaled_boxes.append([x1_scaled, y1_scaled, x2_scaled, y2_scaled])
                    scaled_labels.append(label)
                else:
                    # tiny box -> skip
                    continue
            else:
                # invalid box (zero or negative area) -> skip
                continue

        # Convert to tensors
        img_tensor = T.ToTensor()(img)
        img_tensor = T.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])(img_tensor)

        if len(scaled_boxes) > 0:
            boxes_tensor = torch.as_tensor(scaled_boxes, dtype=torch.float32)
            labels_tensor = torch.as_tensor(scaled_labels, dtype=torch.int64)
        else:
            # No valid boxes for this image
            boxes_tensor = torch.zeros((0, 4), dtype=torch.float32)
            labels_tensor = torch.zeros((0,), dtype=torch.int64)
        
        # Create segmentation masks (auxiliary task)
        h, w = self.image_size, self.image_size
        seg_masks = torch.zeros(3, h, w, dtype=torch.uint8)  # 0=bg, 1=person, 2=ppe
        
        for box, label in zip(scaled_boxes, labels):
            x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            
            if x2 > x1 and y2 > y1:
                if label == 1:  # Person
                    seg_masks[1, y1:y2, x1:x2] = 1
                elif label > 1:  # PPE
                    seg_masks[2, y1:y2, x1:x2] = 1
        
        return {
            'image': img_tensor,
            'boxes': boxes_tensor,
            'labels': labels_tensor,
            'seg_masks': seg_masks,
            'image_name': img_name
        }


def collate_fn(batch):
    """Custom collate function for DataLoader."""
    images = torch.stack([item['image'] for item in batch])
    
    targets = []
    for item in batch:
        targets.append({
            'boxes': item['boxes'],
            'labels': item['labels'],
            'seg_masks': item['seg_masks']
        })
    
    return images, targets


def train_epoch_enhanced(model, dataloader, optimizer, device, epoch, num_epochs):
    """Train one epoch with enhanced model."""
    model.train()
    total_loss = 0
    loss_dict_total = {}
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
    
    for batch_idx, (images, targets) in enumerate(pbar):
        # Images is already a stacked tensor (batch_size, 3, H, W)
        images = images.to(device)
        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v 
                   for k, v in t.items()} for t in targets]
        
        # Forward pass - convert images to list format for Faster R-CNN
        images_list = [images[i] for i in range(images.size(0))]
        loss_dict = model(images_list, targets, extract_seg=True)
        
        # Compute total loss
        loss = sum(v for v in loss_dict.values() if isinstance(v, torch.Tensor))
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
        
        # Update loss dict
        for k, v in loss_dict.items():
            if isinstance(v, torch.Tensor):
                if k not in loss_dict_total:
                    loss_dict_total[k] = 0
                loss_dict_total[k] += v.item()
        
        # Update progress bar
        pbar.set_postfix({'loss': loss.item()})
    
    avg_loss = total_loss / len(dataloader)
    
    # Print detailed losses
    print(f"\nEpoch {epoch+1} Loss Summary:")
    for k, v in loss_dict_total.items():
        avg_v = v / len(dataloader)
        print(f"  {k}: {avg_v:.6f}")
    print(f"  Total: {avg_loss:.6f}\n")
    
    return avg_loss


def train_full_pipeline(
    ssl_epochs=20,
    detection_epochs=50,
    batch_size=4,
    learning_rate=5e-5,
    data_dir='data',
    output_dir='models',
    device='cuda'
):
    """
    Full training pipeline:
    1. SSL pretraining (20 epochs)
    2. Enhanced detection training (50 epochs)
    """
    
    print("\n" + "="*80)
    print("FULL TRAINING PIPELINE - OPTION D")
    print("="*80)
    print(f"SSL Epochs: {ssl_epochs}")
    print(f"Detection Epochs: {detection_epochs}")
    print(f"Batch Size: {batch_size}")
    print(f"Learning Rate: {learning_rate}")
    print("="*80 + "\n")
    
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # STAGE 1: Self-Supervised Pretraining
    ssl_backbone_path = output_dir / 'ssl_backbone_best.pth'
    
    # Check if SSL backbone already exists
    if ssl_backbone_path.exists():
        print("\n" + "="*80)
        print("= STAGE 1: SELF-SUPERVISED PRETRAINING (SKIPPED - CHECKPOINT EXISTS)")
        print("="*80)
        print(f"Found existing SSL backbone: {ssl_backbone_path}")
        print("Skipping SSL pretraining and using pretrained backbone.\n")
        backbone = None  # Will load from checkpoint
    else:
        print("\n" + "="*80)
        print("= STAGE 1: SELF-SUPERVISED PRETRAINING")
        print("="*80 + "\n")
        
        backbone = pretrain_ssl(
            num_epochs=ssl_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            data_dir=data_dir,
            output_dir=str(output_dir),
            device=str(device)
        )
    
    # STAGE 2-4: Enhanced Detection with Multi-Task Learning
    print("\n" + "="*80)
    print("= STAGE 2-4: ENHANCED DETECTION WITH MULTI-TASK LEARNING")
    print("="*80 + "\n")
    
    print("[1/5] Loading enhanced detector with SSL backbone...")
    model = load_enhanced_detector(
        num_classes=12,
        pretrained_backbone_path=str(ssl_backbone_path),
        device=str(device)
    )
    model = model.to(device)
    
    print("[2/5] Creating training dataset with segmentation masks...")
    train_dataset = PPEDatasetWithSegmentation(
        data_dir=data_dir,
        split='train',
        image_size=640
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=True
    )
    
    print("[3/5] Setting up optimizer and scheduler...")
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=detection_epochs)
    
    print("[4/5] Starting enhanced detection training...\n")
    
    best_loss = float('inf')
    losses = []
    
    for epoch in range(detection_epochs):
        avg_loss = train_epoch_enhanced(
            model, train_loader, optimizer, device, epoch, detection_epochs
        )
        losses.append(avg_loss)
        scheduler.step()
        
        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            save_path = output_dir / 'ppe_enhanced_best.pth'
            torch.save({
                'model_state_dict': model.state_dict(),
                'epoch': epoch,
                'loss': avg_loss,
                'losses': losses
            }, save_path)
            print(f"✓ Saved best model to {save_path}\n")
    
    # Save final model
    final_path = output_dir / 'ppe_enhanced_final.pth'
    torch.save({
        'model_state_dict': model.state_dict(),
        'epoch': detection_epochs,
        'loss': avg_loss,
        'losses': losses
    }, final_path)
    
    print("\n" + "="*80)
    print("✓ FULL TRAINING COMPLETE!")
    print("="*80)
    print(f"Final model: {final_path}")
    print(f"Best model:  {output_dir / 'ppe_enhanced_best.pth'}")
    print(f"Final loss: {avg_loss:.6f}")
    print(f"Best loss:  {best_loss:.6f}")
    print("\nNext steps:")
    print("1. Run evaluation: python scripts/eval/evaluate_detection_performance.py")
    print("2. Use best model for inference")
    print("="*80 + "\n")
    
    return model


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Full Training Pipeline with SSL + Multi-Task Learning')
    parser.add_argument('--ssl_epochs', type=int, default=20, help='SSL pretraining epochs')
    parser.add_argument('--detection_epochs', type=int, default=50, help='Detection training epochs')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--lr', type=float, default=5e-5, help='Learning rate')
    parser.add_argument('--data_dir', type=str, default='data', help='Data directory')
    parser.add_argument('--output_dir', type=str, default='models', help='Output directory')
    parser.add_argument('--device', type=str, default='cuda', help='cuda or cpu')
    
    args = parser.parse_args()
    
    model = train_full_pipeline(
        ssl_epochs=args.ssl_epochs,
        detection_epochs=args.detection_epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        device=args.device
    )
