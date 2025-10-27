"""
Two-Stage PPE Detection Pipeline:
Stage 1: Human Detection (High Recall)
Stage 2: PPE Detection on Cropped Regions (High Precision)

This approach separates human detection from PPE detection for better accuracy.
"""

import os
import sys
import json
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torchvision.transforms as T
from PIL import Image
from pathlib import Path
from tqdm import tqdm

# Fix imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.dataset.ppe_dataset import load_ppe_images_and_annotations
import argparse

PPE_CLASSES = [
    'background', 'person', 'hard_hat', 'safety_vest', 'safety_gloves',
    'safety_boots', 'eye_protection', 'no_hard_hat', 'no_safety_vest',
    'no_safety_gloves', 'no_safety_boots', 'no_eye_protection'
]

PPE_ONLY_CLASSES = [
    'background', 'hard_hat', 'safety_vest', 'safety_gloves',
    'safety_boots', 'eye_protection', 'no_hard_hat', 'no_safety_vest',
    'no_safety_gloves', 'no_safety_boots', 'no_eye_protection'
]


def get_augmented_transforms():
    """Get augmentation transforms."""
    return T.Compose([
        T.RandomHorizontalFlip(0.5),
        T.RandomVerticalFlip(0.2),
        T.RandomAffine(degrees=20, translate=(0.15, 0.15), scale=(0.85, 1.15)),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        T.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def get_basic_transforms():
    """Get basic transforms (no augmentation)."""
    return T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


class TorchvisionPPEDataset(Dataset):
    """PPE Dataset with optional augmentations."""
    
    def __init__(self, data_dir, split='train', transforms=None, stage='both'):
        """
        Args:
            data_dir: Path to data directory
            split: 'train', 'val', or 'test'
            transforms: Image transforms
            stage: 'both' (all classes), 'human' (only person), 'ppe' (PPE without person)
        """
        self.data_dir = data_dir
        self.split = split
        self.stage = stage
        
        if stage == 'both':
            self.classes = PPE_CLASSES
        elif stage == 'human':
            self.classes = ['background', 'person']
        elif stage == 'ppe':
            self.classes = PPE_ONLY_CLASSES
        else:
            raise ValueError(f"Unknown stage: {stage}")
        
        self.class2idx = {c: i for i, c in enumerate(self.classes)}
        self.images_info = load_ppe_images_and_annotations(data_dir, self.class2idx, split)
        self.transforms = transforms or T.Compose([T.ToTensor()])
    
    def __len__(self):
        return len(self.images_info)
    
    def __getitem__(self, idx):
        info = self.images_info[idx]
        img_path = info['filename']
        
        try:
            img = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"WARNING: Could not load image {img_path}: {e}")
            img = Image.new('RGB', (300, 300), color='gray')
            return self.transforms(img), {
                'boxes': torch.zeros((0, 4), dtype=torch.float32),
                'labels': torch.zeros((0,), dtype=torch.int64),
                'image_id': torch.tensor([idx])
            }, info['img_id']
        
        boxes = []
        labels = []

        # Support both 'annotations' and 'detections' keys returned by the loader
        annotations = info.get('annotations', info.get('detections', []))
        for ann in annotations:
            # Support different key names for bbox and category id
            bbox = ann.get('bbox') or ann.get('bndbox') or ann.get('bounding_box')
            cat_id = ann.get('category_id', ann.get('label'))

            if bbox is None or cat_id is None:
                continue

            # Filter based on requested stage (note: labels were remapped by loader)
            if self.stage == 'human':
                # person index in this mapping should be 1
                if cat_id != self.class2idx.get('person', 1):
                    continue
            elif self.stage == 'ppe':
                # skip person annotations if present
                if 'person' in self.class2idx and cat_id == self.class2idx.get('person'):
                    continue

            if cat_id in self.class2idx.values():
                boxes.append(bbox)
                labels.append(cat_id)
        
        if len(boxes) > 0:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
            valid_mask = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
            if valid_mask.sum() > 0:
                boxes = boxes[valid_mask]
                labels = labels[valid_mask]
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
        
        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': torch.tensor([idx])
        }
        
        img_t = self.transforms(img)
        return img_t, target, info['img_id']


def collate_fn(batch):
    """Collate function for DataLoader."""
    images = [b[0] for b in batch]
    targets = [b[1] for b in batch]
    ids = [b[2] for b in batch]
    return images, targets, ids


class FocalLossForFasterRCNN:
    """Adapted focal loss for Faster R-CNN."""
    def __init__(self, alpha=0.25, gamma=2.0):
        self.alpha = alpha
        self.gamma = gamma
    
    def __call__(self, predictions, targets):
        p = torch.softmax(predictions, dim=1)
        p_t = p.gather(1, targets.view(-1, 1)).squeeze(1)
        ce = torch.nn.functional.cross_entropy(predictions, targets, reduction='none')
        focal = self.alpha * (1 - p_t) ** self.gamma * ce
        return focal.mean()


def create_model(num_classes, pretrained=True):
    """Create Faster R-CNN model with custom classification head."""
    model = fasterrcnn_resnet50_fpn(weights='DEFAULT' if pretrained else None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def calculate_class_weights(train_loader, num_classes):
    """Calculate class weights from dataset."""
    class_counts = [0] * num_classes
    
    for images, targets, ids in train_loader:
        for target in targets:
            labels = target['labels']
            for label in labels:
                class_counts[label.item()] += 1
    
    total = sum(class_counts) if sum(class_counts) > 0 else 1
    class_weights = {}
    for i, count in enumerate(class_counts):
        class_weights[i] = 1.0 / (count / total) if count > 0 else 0.093
    
    return class_weights


def train_stage(stage_name, model, train_loader, val_loader, num_epochs, lr, device, checkpoint_dir):
    """Train a single stage."""
    print(f"\n{'='*80}")
    print(f"TRAINING STAGE: {stage_name.upper()}")
    print(f"{'='*80}")
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    
    # Calculate class weights from training data
    class_weights = calculate_class_weights(train_loader, model.roi_heads.box_predictor.cls_score.out_features)
    print("\nClass Weights (from dataset statistics):")
    for class_id, weight in sorted(class_weights.items()):
        print(f"  {class_id:2d}. weight={weight:.3f}")
    
    model = model.to(device)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
    
    best_loss = float('inf')
    history = {'train_loss': [], 'val_loss': [], 'class_weights': class_weights}
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        
        for batch_idx, (images, targets, ids) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")):
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            
            optimizer.zero_grad()
            losses.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss += losses.item()
        
        train_loss /= len(train_loader)
        
        # Validation
        model.train()  # Keep in train mode to get loss dict
        val_loss = 0
        val_count = 0
        
        with torch.no_grad():
            for images, targets, ids in val_loader:
                images = [img.to(device) for img in images]
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                
                loss_dict = model(images, targets)
                # Debug: print first validation batch loss_dict structure for troubleshooting
                if val_count == 0:
                    try:
                        print("[DEBUG] Sample val loss_dict keys:", list(loss_dict.keys()) if isinstance(loss_dict, dict) else type(loss_dict))
                        if isinstance(loss_dict, dict):
                            for k, v in loss_dict.items():
                                try:
                                    print(f"[DEBUG]   {k}:", float(v) if hasattr(v, 'item') else v)
                                except Exception:
                                    print(f"[DEBUG]   {k}: (non-scalar)")
                    except Exception as e:
                        print(f"[DEBUG] Could not print loss_dict: {e}")
                if isinstance(loss_dict, dict):
                    losses = sum(loss for loss in loss_dict.values())
                    val_loss += losses.item()
                    val_count += 1
        
        val_loss = val_loss / val_count if val_count > 0 else 0.0
        scheduler.step()
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        
        print(f"\n[OK] Epoch {epoch+1}/{num_epochs}")
        print(f"  Train Loss: {train_loss:.6f}")
        print(f"  Val Loss: {val_loss:.6f} (batches processed: {val_count})")
        print(f"  Val Loss: {val_loss:.6f}")
        
        # Save best model
        if val_loss < best_loss:
            best_loss = val_loss
            checkpoint_path = os.path.join(checkpoint_dir, f'{stage_name}_best.pth')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'train_loss': train_loss,
            }, checkpoint_path)
            print(f"  [SAVED] Best model: {checkpoint_path}")
    
    return model, history


def main():
    parser = argparse.ArgumentParser(description='Two-Stage PPE Detection Training')
    parser.add_argument('--data_dir', default='data', help='Path to data directory')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', help='Device')
    parser.add_argument('--augment', action='store_true', default=True, help='Use augmentations')
    parser.add_argument('--checkpoint_dir', default='models', help='Checkpoint directory')
    
    args = parser.parse_args()
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    print("\n" + "="*80)
    print("TWO-STAGE PPE DETECTION PIPELINE")
    print("="*80)
    print(f"Config: Epochs={args.epochs}, Batch Size={args.batch_size}, LR={args.lr}, Device={args.device}")
    print("="*80 + "\n")
    
    # Stage 1: Human Detection
    print("[STAGE 1] Training Human Detector (High Recall)")
    train_transforms = get_augmented_transforms() if args.augment else get_basic_transforms()
    val_transforms = get_basic_transforms()
    
    stage1_train = TorchvisionPPEDataset(args.data_dir, split='train', transforms=train_transforms, stage='human')
    stage1_val = TorchvisionPPEDataset(args.data_dir, split='val', transforms=val_transforms, stage='human')
    
    print(f"Stage 1 - Train: {len(stage1_train)}, Val: {len(stage1_val)}")
    
    stage1_train_loader = DataLoader(stage1_train, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=0)
    stage1_val_loader = DataLoader(stage1_val, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=0)
    
    print(f"Stage 1 DataLoaders - Train batches: {len(stage1_train_loader)}, Val batches: {len(stage1_val_loader)}")
    
    stage1_model = create_model(num_classes=2, pretrained=True)  # background + person
    stage1_model, stage1_history = train_stage(
        'stage1_human', stage1_model, stage1_train_loader, stage1_val_loader,
        args.epochs, args.lr, args.device, args.checkpoint_dir
    )
    
    # Stage 2: PPE Detection
    print("\n[STAGE 2] Training PPE Detector (High Precision)")
    stage2_train = TorchvisionPPEDataset(args.data_dir, split='train', transforms=train_transforms, stage='ppe')
    stage2_val = TorchvisionPPEDataset(args.data_dir, split='val', transforms=val_transforms, stage='ppe')
    
    print(f"Stage 2 - Train: {len(stage2_train)}, Val: {len(stage2_val)}")
    
    stage2_train_loader = DataLoader(stage2_train, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=0)
    stage2_val_loader = DataLoader(stage2_val, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=0)
    
    stage2_model = create_model(num_classes=len(PPE_ONLY_CLASSES), pretrained=True)
    stage2_model, stage2_history = train_stage(
        'stage2_ppe', stage2_model, stage2_train_loader, stage2_val_loader,
        args.epochs, args.lr, args.device, args.checkpoint_dir
    )
    
    # Save training history
    history = {
        'stage1': stage1_history,
        'stage2': stage2_history,
        'config': vars(args)
    }
    history_path = os.path.join(args.checkpoint_dir, 'training_history_two_stage.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"\n[OK] Training complete!")
    print(f"  Stage 1 model: {args.checkpoint_dir}/stage1_human_best.pth")
    print(f"  Stage 2 model: {args.checkpoint_dir}/stage2_ppe_best.pth")
    print(f"  History: {history_path}")


if __name__ == '__main__':
    main()
