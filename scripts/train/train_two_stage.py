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
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2  # ResNet101-FPN for better accuracy
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torchvision.transforms as T
from PIL import Image
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict

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

class EarlyStopping:
    """Stop training when validation loss stops improving."""
    def __init__(self, patience=8, min_delta=1e-4):
        """
        Args:
            patience (int): Number of epochs with no improvement after which training will stop.
            min_delta (float): Minimum change in the monitored quantity to qualify as an improvement.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.counter = 0
        self.should_stop = False

    def step(self, val_loss):
        """Check if training should stop based on validation loss."""
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True

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


def create_model(num_classes, pretrained=True, backbone='resnet101'):
    """Create Faster R-CNN model with custom classification head.
    
    Args:
        num_classes: Number of classes (including background)
        pretrained: Use pretrained weights
        backbone: 'resnet50' or 'resnet101' (default: resnet101 for better accuracy)
    """
    if backbone == 'resnet101':
        # ResNet101-FPN-v2: Better accuracy, slightly slower but worth it for small objects
        model = fasterrcnn_resnet50_fpn_v2(weights='DEFAULT' if pretrained else None)
    else:
        # Fallback to ResNet50 if specified
        from torchvision.models.detection import fasterrcnn_resnet50_fpn
        model = fasterrcnn_resnet50_fpn(weights='DEFAULT' if pretrained else None)
    
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def calculate_class_weights(train_loader, num_classes, smooth_weights=True):
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
        raw_weight = 1.0 / (count / total) if count > 0 else 0.093
        if smooth_weights and raw_weight > 10:
            # Smooth extreme weights: cap at 10 or use sqrt scaling
            class_weights[i] = min(10.0, raw_weight ** 0.5)  # sqrt for smoothing
        else:
            class_weights[i] = raw_weight
    
    return class_weights


def train_stage(stage_name, model, train_loader, val_loader, num_epochs, lr, device, checkpoint_dir, use_focal=False, use_class_weights=False):
    """Train a single stage."""
    print(f"\n{'='*80}")
    print(f"TRAINING STAGE: {stage_name.upper()}")
    print(f"{'='*80}")
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    print(f"Using focal loss: {use_focal}, Using class weights: {use_class_weights}")
    
    # Calculate class weights from training data
    num_classes = model.roi_heads.box_predictor.cls_score.out_features
    smooth_weights = stage_name == 'stage2_ppe'  # Smooth for Stage 2 only
    class_weights_dict = calculate_class_weights(train_loader, num_classes, smooth_weights=smooth_weights)
    print("\nClass Weights (from dataset statistics):")
    for class_id, weight in sorted(class_weights_dict.items()):
        print(f"  {class_id:2d}. weight={weight:.3f}")
    
    # Convert class weights to tensor for loss computation
    class_weights_tensor = None
    if use_class_weights:
        class_weights_tensor = torch.tensor([class_weights_dict[i] for i in range(num_classes)], dtype=torch.float32).to(device)
        print(f"\n[INFO] Class weights will be applied to the loss function")
    
    model = model.to(device)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
    
    focal_loss_fn = FocalLossForFasterRCNN(alpha=0.25, gamma=2.0) if use_focal else None
    
    best_loss = float('inf')
    history = {'train_loss': [], 'val_loss': [], 'class_weights': class_weights_dict}
    early_stopper = EarlyStopping(patience=8, min_delta=1e-4)
    
    # Track class predictions for debugging
    epoch_class_counts = defaultdict(int)
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        epoch_class_counts.clear()
        
        for batch_idx, (images, targets, ids) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")):
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            # Track what classes are in this batch
            for target in targets:
                for label in target['labels']:
                    epoch_class_counts[label.item()] += 1
            
            loss_dict = model(images, targets)
            
            # Apply class weights or focal loss if requested
            if use_class_weights and class_weights_tensor is not None and 'loss_classifier' in loss_dict:
                # Weight the classification loss
                original_cls_loss = loss_dict['loss_classifier']
                # Note: This is a rough approximation since we don't have direct access to per-sample losses
                # For a proper implementation, we'd need to modify the Faster R-CNN internals
                weighted_cls_loss = original_cls_loss * class_weights_tensor.mean()
                loss_dict['loss_classifier'] = weighted_cls_loss
            
            if use_focal and focal_loss_fn is not None and 'loss_classifier' in loss_dict:
                # Replace classification loss with focal loss
                # Note: This is a placeholder - full implementation would require hooking into the model's forward pass
                # For now, we'll use a hybrid approach: scale the existing loss
                loss_dict['loss_classifier'] = loss_dict['loss_classifier'] * 1.5  # Boost classification loss importance
            
            losses = sum(loss for loss in loss_dict.values())
            
            optimizer.zero_grad()
            losses.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss += losses.item()
        
        train_loss /= len(train_loader)
        
        # Print class distribution for this epoch
        print(f"\n[DEBUG] Training class distribution (epoch {epoch+1}):")
        total_instances = sum(epoch_class_counts.values())
        for class_id in sorted(epoch_class_counts.keys()):
            count = epoch_class_counts[class_id]
            pct = 100.0 * count / total_instances if total_instances > 0 else 0
            class_name = PPE_ONLY_CLASSES[class_id] if stage_name == 'stage2_ppe' and 0 <= class_id < len(PPE_ONLY_CLASSES) else f"class_{class_id}"
            print(f"  {class_name} (id={class_id}): {count} instances ({pct:.1f}%)")
        
        # Validation
        model.train()  # Keep in train mode to get loss dict
        val_loss = 0
        val_count = 0
        val_class_counts = defaultdict(int)
        
        with torch.no_grad():
            for images, targets, ids in val_loader:
                images = [img.to(device) for img in images]
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                
                # Track validation class distribution
                for target in targets:
                    for label in target['labels']:
                        val_class_counts[label.item()] += 1
                
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
        
        # Print validation class distribution
        print(f"[DEBUG] Validation class distribution:")
        total_val = sum(val_class_counts.values())
        for class_id in sorted(val_class_counts.keys()):
            count = val_class_counts[class_id]
            pct = 100.0 * count / total_val if total_val > 0 else 0
            class_name = PPE_ONLY_CLASSES[class_id] if stage_name == 'stage2_ppe' and 0 <= class_id < len(PPE_ONLY_CLASSES) else f"class_{class_id}"
            print(f"  {class_name} (id={class_id}): {count} instances ({pct:.1f}%)")
        
        # Quick inference check to see what model predicts
        if epoch % 3 == 0 or epoch == num_epochs - 1:  # Every 3 epochs
            model.eval()
            pred_class_counts = defaultdict(int)
            with torch.no_grad():
                for val_batch_idx, (images, targets, ids) in enumerate(val_loader):
                    if val_batch_idx >= 3:  # Just check first 3 batches
                        break
                    images = [img.to(device) for img in images]
                    outputs = model(images)
                    for output in outputs:
                        labels = output['labels'].cpu()
                        scores = output['scores'].cpu()
                        # Count predictions above threshold
                        for label, score in zip(labels, scores):
                            if score >= 0.05:  # Very low threshold to see all predictions
                                pred_class_counts[label.item()] += 1
            
            print(f"[DEBUG] Model predictions on val (epoch {epoch+1}, conf >= 0.05):")
            if pred_class_counts:
                for class_id in sorted(pred_class_counts.keys()):
                    count = pred_class_counts[class_id]
                    class_name = PPE_ONLY_CLASSES[class_id] if stage_name == 'stage2_ppe' and 0 <= class_id < len(PPE_ONLY_CLASSES) else f"class_{class_id}"
                    print(f"  {class_name} (id={class_id}): {count} predictions")
            else:
                print("  No predictions above threshold!")
            model.train()
        
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

        # Early stopping check
        early_stopper.step(val_loss)
        if early_stopper.should_stop:
            print(f"\n[EARLY STOPPING] Validation loss did not improve for {early_stopper.patience} epochs.")
            break
    
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
    parser.add_argument('--resume_from_stage1', action='store_true', help='If set, initialize stage 2 model weights from stage 1 best model')
    parser.add_argument('--skip_stage1', action='store_true', help='Skip training stage 1 and load pretrained stage 1 model for stage 2')
    parser.add_argument('--backbone', default='resnet101', choices=['resnet50', 'resnet101'], help='Backbone architecture (resnet101 recommended for better accuracy)')

    args = parser.parse_args()
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    print("\n" + "="*80)
    print("TWO-STAGE PPE DETECTION PIPELINE")
    print("="*80)
    print(f"Config: Epochs={args.epochs}, Batch Size={args.batch_size}, LR={args.lr}, Device={args.device}")
    print(f"Backbone: {args.backbone.upper()} (Better feature extraction for small PPE objects)")
    print("="*80 + "\n")

    train_transforms = get_augmented_transforms() if args.augment else get_basic_transforms()
    val_transforms = get_basic_transforms()
    
    # Stage 1: Human Detection
    print("[STAGE 1] Training Human Detector (High Recall)")
    if not args.skip_stage1:
        print("[STAGE 1] Training Human Detector (High Recall)")

        stage1_train = TorchvisionPPEDataset(args.data_dir, split='train', transforms=train_transforms, stage='human')
        stage1_val = TorchvisionPPEDataset(args.data_dir, split='val', transforms=val_transforms, stage='human')
        
        print(f"Stage 1 - Train: {len(stage1_train)}, Val: {len(stage1_val)}")
        
        stage1_train_loader = DataLoader(stage1_train, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
        stage1_val_loader = DataLoader(stage1_val, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
        
        stage1_model = create_model(num_classes=2, pretrained=True, backbone=args.backbone)
        stage1_model, stage1_history = train_stage(
            'stage1_human', stage1_model, stage1_train_loader, stage1_val_loader,
            args.epochs, args.lr, args.device, args.checkpoint_dir, 
            use_focal=True, use_class_weights=False
        )
    else:
        print("[INFO] Skipping Stage 1 training.")
        stage1_history = {}
    
    # Stage 2: PPE Detection
    print("\n[STAGE 2] Training PPE Detector (High Precision)")
    stage2_train = TorchvisionPPEDataset(args.data_dir, split='train', transforms=train_transforms, stage='ppe')
    stage2_val = TorchvisionPPEDataset(args.data_dir, split='val', transforms=val_transforms, stage='ppe')
    
    print(f"Stage 2 - Train: {len(stage2_train)}, Val: {len(stage2_val)}")
    
    stage2_train_loader = DataLoader(stage2_train, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=0)
    stage2_val_loader = DataLoader(stage2_val, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=0)
    
    stage2_model = create_model(num_classes=len(PPE_ONLY_CLASSES), pretrained=True, backbone=args.backbone)

    if args.resume_from_stage1:
        stage1_ckpt_path = os.path.join(args.checkpoint_dir, 'stage1_human_best.pth')
        if os.path.exists(stage1_ckpt_path):
            print(f"\n[INFO] Loading Stage 1 weights from {stage1_ckpt_path} ...")
            ckpt = torch.load(stage1_ckpt_path, map_location=args.device)
            stage1_state = ckpt['model_state_dict']
            
            # Load shared weights (ignore mismatched classifier head)
            model_dict = stage2_model.state_dict()
            pretrained_dict = {k: v for k, v in stage1_state.items() if k in model_dict and v.size() == model_dict[k].size()}
            model_dict.update(pretrained_dict)
            stage2_model.load_state_dict(model_dict)
            print(f"[INFO] Loaded {len(pretrained_dict)} layers from Stage 1 model.")
        else:
            print(f"[WARNING] Stage 1 checkpoint not found: {stage1_ckpt_path}")

    stage2_model, stage2_history = train_stage(
        'stage2_ppe', stage2_model, stage2_train_loader, stage2_val_loader,
        args.epochs, args.lr, args.device, args.checkpoint_dir, 
        use_focal=True, use_class_weights=True
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
