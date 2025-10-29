"""
Check for false positives in model predictions.
Compares predictions against ground truth to identify:
1. False positive workers (detected workers that don't exist)
2. False positive PPE (PPE detected where there is none)
3. Precision/Recall statistics
"""

import torch
import torchvision
from torchvision import transforms as T
from torchvision.ops import box_iou
from PIL import Image
import sys
import os
from collections import defaultdict, Counter

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.dataset.ppe_dataset import load_ppe_images_and_annotations

PPE_CLASSES = [
    'background', 'person', 'hard_hat', 'safety_vest', 'safety_gloves',
    'safety_boots', 'eye_protection', 'no_hard_hat', 'no_safety_vest',
    'no_safety_gloves', 'no_safety_boots', 'no_eye_protection'
]


def load_model(model_path, backbone='resnet101', device='cpu'):
    """Load Faster R-CNN model"""
    num_classes = len(PPE_CLASSES)
    
    if backbone == 'resnet101':
        try:
            from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
            from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
            
            model = fasterrcnn_resnet50_fpn_v2(weights=None)
            in_features = model.roi_heads.box_predictor.cls_score.in_features
            model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        except:
            model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None, num_classes=num_classes)
    else:
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None, num_classes=num_classes)
    
    checkpoint = torch.load(model_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    return model


def check_false_positives(model, data_dir, split='test', device='cpu', conf_threshold=0.3, iou_threshold=0.5, num_images=10):
    """
    Check for false positives in predictions.
    
    Args:
        model: Trained model
        data_dir: Data directory
        split: Dataset split
        device: Device
        conf_threshold: Confidence threshold for predictions
        iou_threshold: IoU threshold for matching predictions to GT
        num_images: Number of images to check
    """
    class2idx = {c: i for i, c in enumerate(PPE_CLASSES)}
    images_info = load_ppe_images_and_annotations(data_dir, class2idx, split)[:num_images]
    
    transforms = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Statistics
    stats = {
        'per_class': defaultdict(lambda: {
            'gt_count': 0,
            'pred_count': 0,
            'true_positives': 0,
            'false_positives': 0,
            'false_negatives': 0
        }),
        'per_image': []
    }
    
    print(f"\n{'='*80}")
    print(f"FALSE POSITIVE ANALYSIS ({num_images} {split.upper()} images)")
    print(f"{'='*80}")
    print(f"Confidence threshold: {conf_threshold}")
    print(f"IoU threshold: {iou_threshold}\n")
    
    with torch.no_grad():
        for idx, info in enumerate(images_info):
            img = Image.open(info['filename']).convert('RGB')
            img_tensor = transforms(img).unsqueeze(0).to(device)
            
            # Run inference
            outputs = model(img_tensor)[0]
            
            # Filter by confidence
            keep = outputs['scores'] >= conf_threshold
            pred_boxes = outputs['boxes'][keep].cpu()
            pred_labels = outputs['labels'][keep].cpu()
            pred_scores = outputs['scores'][keep].cpu()
            
            # Ground truth - extract from detections
            detections = info.get('detections', [])
            if len(detections) == 0:
                print(f"  ⚠️  No annotations found, skipping...")
                continue
                
            gt_boxes = torch.tensor([det['bbox'] for det in detections])
            gt_labels = torch.tensor([det['label'] for det in detections])
            
            # Image stats
            img_stats = {
                'filename': os.path.basename(info['filename']),
                'gt_workers': (gt_labels == 1).sum().item(),
                'pred_workers': (pred_labels == 1).sum().item(),
                'false_positive_workers': 0,
                'false_positive_ppe': 0
            }
            
            # Match predictions to GT for each class
            for class_id in range(1, len(PPE_CLASSES)):  # Skip background
                class_name = PPE_CLASSES[class_id]
                
                # GT boxes for this class
                gt_mask = gt_labels == class_id
                class_gt_boxes = gt_boxes[gt_mask]
                
                # Predicted boxes for this class
                pred_mask = pred_labels == class_id
                class_pred_boxes = pred_boxes[pred_mask]
                
                stats['per_class'][class_name]['gt_count'] += len(class_gt_boxes)
                stats['per_class'][class_name]['pred_count'] += len(class_pred_boxes)
                
                if len(class_gt_boxes) == 0 and len(class_pred_boxes) == 0:
                    continue
                
                if len(class_gt_boxes) == 0:
                    # All predictions are false positives
                    stats['per_class'][class_name]['false_positives'] += len(class_pred_boxes)
                    if class_name == 'person':
                        img_stats['false_positive_workers'] += len(class_pred_boxes)
                    else:
                        img_stats['false_positive_ppe'] += len(class_pred_boxes)
                    continue
                
                if len(class_pred_boxes) == 0:
                    # All GT are false negatives
                    stats['per_class'][class_name]['false_negatives'] += len(class_gt_boxes)
                    continue
                
                # Calculate IoU between predictions and GT
                ious = box_iou(class_pred_boxes, class_gt_boxes)
                
                # For each prediction, check if it matches any GT
                matched_gt = set()
                for pred_idx in range(len(class_pred_boxes)):
                    max_iou, gt_idx = ious[pred_idx].max(dim=0)
                    
                    if max_iou >= iou_threshold:
                        # True positive
                        stats['per_class'][class_name]['true_positives'] += 1
                        matched_gt.add(gt_idx.item())
                    else:
                        # False positive
                        stats['per_class'][class_name]['false_positives'] += 1
                        if class_name == 'person':
                            img_stats['false_positive_workers'] += 1
                        else:
                            img_stats['false_positive_ppe'] += 1
                
                # Unmatched GT boxes are false negatives
                stats['per_class'][class_name]['false_negatives'] += (len(class_gt_boxes) - len(matched_gt))
            
            stats['per_image'].append(img_stats)
            
            # Print image summary
            if img_stats['false_positive_workers'] > 0 or img_stats['false_positive_ppe'] > 0:
                print(f"📷 {img_stats['filename']}")
                print(f"   GT workers: {img_stats['gt_workers']}, Predicted: {img_stats['pred_workers']}")
                if img_stats['false_positive_workers'] > 0:
                    print(f"   ⚠️  FALSE POSITIVE WORKERS: {img_stats['false_positive_workers']}")
                if img_stats['false_positive_ppe'] > 0:
                    print(f"   ⚠️  FALSE POSITIVE PPE: {img_stats['false_positive_ppe']}")
    
    # Calculate overall statistics
    print(f"\n{'='*80}")
    print(f"OVERALL STATISTICS")
    print(f"{'='*80}\n")
    
    total_fp_workers = sum(img['false_positive_workers'] for img in stats['per_image'])
    total_fp_ppe = sum(img['false_positive_ppe'] for img in stats['per_image'])
    
    print(f"Total false positive workers: {total_fp_workers}")
    print(f"Total false positive PPE: {total_fp_ppe}")
    
    print(f"\n{'='*80}")
    print(f"PER-CLASS PRECISION & RECALL")
    print(f"{'='*80}\n")
    
    for class_name in ['person', 'hard_hat', 'safety_vest', 'safety_gloves', 'safety_boots', 
                       'eye_protection', 'no_hard_hat', 'no_safety_vest', 'no_safety_gloves', 
                       'no_safety_boots', 'no_eye_protection']:
        s = stats['per_class'][class_name]
        
        if s['gt_count'] == 0 and s['pred_count'] == 0:
            continue
        
        precision = s['true_positives'] / s['pred_count'] if s['pred_count'] > 0 else 0.0
        recall = s['true_positives'] / s['gt_count'] if s['gt_count'] > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        print(f"{class_name:25s}:")
        print(f"  GT: {s['gt_count']:4d} | Pred: {s['pred_count']:4d} | TP: {s['true_positives']:4d} | FP: {s['false_positives']:4d} | FN: {s['false_negatives']:4d}")
        print(f"  Precision: {precision:.3f} | Recall: {recall:.3f} | F1: {f1:.3f}")
        
        if s['false_positives'] > s['true_positives'] and s['pred_count'] > 5:
            print(f"  ⚠️  HIGH FALSE POSITIVE RATE!")
    
    print(f"\n{'='*80}")
    print(f"✅ Analysis complete!")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Check for false positives')
    parser.add_argument('--model', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--data_dir', type=str, default='data', help='Data directory')
    parser.add_argument('--split', type=str, default='test', help='Dataset split')
    parser.add_argument('--num_images', type=int, default=10, help='Number of images to check')
    parser.add_argument('--conf_threshold', type=float, default=0.3, help='Confidence threshold')
    parser.add_argument('--iou_threshold', type=float, default=0.5, help='IoU threshold for matching')
    parser.add_argument('--backbone', type=str, default='resnet101', help='Backbone architecture')
    
    args = parser.parse_args()
    
    print(f"\n🔍 Loading model from: {args.model}")
    model = load_model(args.model, backbone=args.backbone)
    
    check_false_positives(
        model, 
        args.data_dir, 
        split=args.split, 
        conf_threshold=args.conf_threshold,
        iou_threshold=args.iou_threshold,
        num_images=args.num_images
    )
