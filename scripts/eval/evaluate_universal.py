"""
Universal PPE Detection Evaluator
Works with ANY trained Faster R-CNN model (single-stage or two-stage output).

Features:
- Proper mAP calculation (COCO-style)
- Worker counting (person detection)
- PPE-to-worker association (IoU-based)
- Violation detection (missing PPE per worker)
"""

import os
import sys
import json
import torch
import numpy as np
from pathlib import Path
from PIL import Image
import torchvision.transforms as T
from collections import defaultdict
from typing import Dict, List, Tuple

# Add project root
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from torchvision.models.detection import fasterrcnn_resnet50_fpn, fasterrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from src.dataset.ppe_dataset import load_ppe_images_and_annotations

PPE_CLASSES = [
    'background', 'person', 'hard_hat', 'safety_vest', 'safety_gloves',
    'safety_boots', 'eye_protection', 'no_hard_hat', 'no_safety_vest',
    'no_safety_gloves', 'no_safety_boots', 'no_eye_protection'
]


def calculate_iou(box1, box2):
    """Calculate IoU between two boxes [x1, y1, x2, y2]."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0


def associate_ppe_with_workers(detections, person_iou_threshold=0.3):
    """
    Associate PPE items with workers based on IoU overlap.
    
    Args:
        detections: Dict with 'boxes', 'labels', 'scores'
        person_iou_threshold: Minimum IoU for PPE to be assigned to worker
    
    Returns:
        workers: List of dicts, each with 'person_box', 'ppe_items', 'violations'
    """
    boxes = detections['boxes']
    labels = detections['labels']
    scores = detections['scores']
    
    # Separate person and PPE detections
    person_indices = [i for i, label in enumerate(labels) if label == 1]  # person=1
    ppe_indices = [i for i, label in enumerate(labels) if label != 1 and label != 0]  # not person, not background
    
    workers = []
    for person_idx in person_indices:
        person_box = boxes[person_idx]
        person_score = scores[person_idx]
        
        worker = {
            'person_box': person_box.tolist(),
            'person_score': float(person_score),
            'ppe_items': [],
            'ppe_classes_found': set()
        }
        
        # Find PPE items overlapping with this worker
        for ppe_idx in ppe_indices:
            ppe_box = boxes[ppe_idx]
            iou = calculate_iou(person_box, ppe_box)
            
            if iou >= person_iou_threshold:
                worker['ppe_items'].append({
                    'class': int(labels[ppe_idx]),
                    'class_name': PPE_CLASSES[labels[ppe_idx]],
                    'score': float(scores[ppe_idx]),
                    'box': ppe_box.tolist()
                })
                worker['ppe_classes_found'].add(int(labels[ppe_idx]))
        
        # Check for violations (missing required PPE)
        required_ppe = {2, 3, 4, 5, 6}  # hard_hat, vest, gloves, boots, eye_protection
        found_ppe = {item['class'] for item in worker['ppe_items'] if item['class'] in required_ppe}
        
        worker['violations'] = []
        for required_class in required_ppe:
            if required_class not in found_ppe:
                worker['violations'].append(PPE_CLASSES[required_class])
        
        workers.append(worker)
    
    return workers


def calculate_ap(pred_boxes, pred_scores, pred_labels, gt_boxes, gt_labels, class_id, iou_threshold=0.5):
    """Calculate AP for a single class."""
    # Filter predictions and ground truth for this class
    pred_mask = pred_labels == class_id
    gt_mask = gt_labels == class_id
    
    class_pred_boxes = pred_boxes[pred_mask]
    class_pred_scores = pred_scores[pred_mask]
    class_gt_boxes = gt_boxes[gt_mask]
    
    if len(class_gt_boxes) == 0:
        return 0.0 if len(class_pred_boxes) > 0 else None  # No GT for this class
    
    if len(class_pred_boxes) == 0:
        return 0.0  # No predictions but GT exists
    
    # Sort predictions by score (descending)
    sorted_indices = torch.argsort(class_pred_scores, descending=True)
    class_pred_boxes = class_pred_boxes[sorted_indices]
    class_pred_scores = class_pred_scores[sorted_indices]
    
    # Track which GT boxes have been matched
    gt_matched = [False] * len(class_gt_boxes)
    
    tp = []
    fp = []
    
    for pred_box in class_pred_boxes:
        max_iou = 0
        max_gt_idx = -1
        
        for gt_idx, gt_box in enumerate(class_gt_boxes):
            iou = calculate_iou(pred_box, gt_box)
            if iou > max_iou:
                max_iou = iou
                max_gt_idx = gt_idx
        
        if max_iou >= iou_threshold and not gt_matched[max_gt_idx]:
            tp.append(1)
            fp.append(0)
            gt_matched[max_gt_idx] = True
        else:
            tp.append(0)
            fp.append(1)
    
    # Calculate precision and recall
    tp_cumsum = np.cumsum(tp)
    fp_cumsum = np.cumsum(fp)
    
    recalls = tp_cumsum / len(class_gt_boxes)
    precisions = tp_cumsum / (tp_cumsum + fp_cumsum)
    
    # Calculate AP (11-point interpolation)
    ap = 0
    for t in np.arange(0, 1.1, 0.1):
        if np.sum(recalls >= t) == 0:
            p = 0
        else:
            p = np.max(precisions[recalls >= t])
        ap += p / 11
    
    return ap


def load_model(model_path, num_classes=12, backbone='resnet101', device='cpu'):
    """Load a trained Faster R-CNN model."""
    if backbone == 'resnet101':
        model = fasterrcnn_resnet50_fpn_v2(weights=None)
    else:
        model = fasterrcnn_resnet50_fpn(weights=None)
    
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    return model


def evaluate_model(model, data_dir, split='val', device='cpu', conf_threshold=0.05, nms_threshold=0.5):
    """
    Comprehensive evaluation of PPE detection model.
    
    Returns:
        results: Dict with mAP, per-class AP, worker stats, violations
    """
    # Load dataset
    class2idx = {c: i for i, c in enumerate(PPE_CLASSES)}
    images_info = load_ppe_images_and_annotations(data_dir, class2idx, split)
    
    transforms = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    all_detections = []
    all_ground_truths = []
    worker_stats = {'total_workers': 0, 'total_violations': 0, 'violation_types': defaultdict(int)}
    
    print(f"\n{'='*80}")
    print(f"EVALUATING ON {split.upper()} SET ({len(images_info)} images)")
    print(f"{'='*80}\n")
    
    with torch.no_grad():
        for idx, info in enumerate(images_info):
            img = Image.open(info['filename']).convert('RGB')
            img_tensor = transforms(img).unsqueeze(0).to(device)
            
            # Run inference
            outputs = model(img_tensor)[0]
            
            # Filter by confidence
            keep = outputs['scores'] >= conf_threshold
            boxes = outputs['boxes'][keep].cpu()
            labels = outputs['labels'][keep].cpu()
            scores = outputs['scores'][keep].cpu()
            
            # Apply NMS (per class)
            from torchvision.ops import nms
            final_boxes = []
            final_labels = []
            final_scores = []
            
            for class_id in torch.unique(labels):
                class_mask = labels == class_id
                class_boxes = boxes[class_mask]
                class_scores = scores[class_mask]
                
                keep_nms = nms(class_boxes, class_scores, nms_threshold)
                
                final_boxes.append(class_boxes[keep_nms])
                final_labels.append(labels[class_mask][keep_nms])
                final_scores.append(class_scores[keep_nms])
            
            if len(final_boxes) > 0:
                boxes = torch.cat(final_boxes)
                labels = torch.cat(final_labels)
                scores = torch.cat(final_scores)
            else:
                boxes = torch.zeros((0, 4))
                labels = torch.zeros((0,), dtype=torch.long)
                scores = torch.zeros((0,))
            
            detections = {'boxes': boxes, 'labels': labels, 'scores': scores}
            
            # Associate PPE with workers
            workers = associate_ppe_with_workers(detections)
            worker_stats['total_workers'] += len(workers)
            
            for worker in workers:
                if worker['violations']:
                    worker_stats['total_violations'] += 1
                    for violation in worker['violations']:
                        worker_stats['violation_types'][violation] += 1
            
            # Prepare ground truth
            gt_boxes = []
            gt_labels = []
            for det in info.get('detections', []):
                bbox = det.get('bbox')
                label = det.get('label')
                if bbox and label is not None:
                    gt_boxes.append(bbox)
                    gt_labels.append(label)
            
            if len(gt_boxes) > 0:
                gt_boxes = torch.tensor(gt_boxes, dtype=torch.float32)
                gt_labels = torch.tensor(gt_labels, dtype=torch.long)
            else:
                gt_boxes = torch.zeros((0, 4))
                gt_labels = torch.zeros((0,), dtype=torch.long)
            
            all_detections.append({'boxes': boxes, 'labels': labels, 'scores': scores})
            all_ground_truths.append({'boxes': gt_boxes, 'labels': gt_labels})
            
            if (idx + 1) % 10 == 0:
                print(f"Processed {idx + 1}/{len(images_info)} images...")
    
    # Calculate mAP
    print("\n" + "="*80)
    print("CALCULATING mAP (IoU=0.5)")
    print("="*80 + "\n")
    
    class_aps = {}
    for class_id in range(1, len(PPE_CLASSES)):  # Skip background
        class_name = PPE_CLASSES[class_id]
        
        # Concatenate all predictions and GT for this class
        all_pred_boxes = []
        all_pred_scores = []
        all_pred_labels = []
        all_gt_boxes = []
        all_gt_labels = []
        
        for det, gt in zip(all_detections, all_ground_truths):
            all_pred_boxes.append(det['boxes'])
            all_pred_scores.append(det['scores'])
            all_pred_labels.append(det['labels'])
            all_gt_boxes.append(gt['boxes'])
            all_gt_labels.append(gt['labels'])
        
        if len(all_pred_boxes) > 0:
            pred_boxes = torch.cat(all_pred_boxes)
            pred_scores = torch.cat(all_pred_scores)
            pred_labels = torch.cat(all_pred_labels)
        else:
            pred_boxes = torch.zeros((0, 4))
            pred_scores = torch.zeros((0,))
            pred_labels = torch.zeros((0,), dtype=torch.long)
        
        if len(all_gt_boxes) > 0:
            gt_boxes = torch.cat(all_gt_boxes)
            gt_labels = torch.cat(all_gt_labels)
        else:
            gt_boxes = torch.zeros((0, 4))
            gt_labels = torch.zeros((0,), dtype=torch.long)
        
        ap = calculate_ap(pred_boxes, pred_scores, pred_labels, gt_boxes, gt_labels, class_id)
        
        if ap is not None:
            class_aps[class_name] = ap
            print(f"  {class_name:25s}: AP = {ap:.4f}")
    
    # Calculate mean AP
    if class_aps:
        mAP = np.mean(list(class_aps.values()))
    else:
        mAP = 0.0
    
    print("\n" + "="*80)
    print(f"MEAN AVERAGE PRECISION (mAP@0.5): {mAP:.4f}")
    print("="*80)
    
    # Worker statistics
    print("\n" + "="*80)
    print("WORKER & VIOLATION STATISTICS")
    print("="*80)
    print(f"  Total Workers Detected: {worker_stats['total_workers']}")
    print(f"  Workers with Violations: {worker_stats['total_violations']}")
    if worker_stats['total_workers'] > 0:
        violation_rate = 100.0 * worker_stats['total_violations'] / worker_stats['total_workers']
        print(f"  Violation Rate: {violation_rate:.1f}%")
    
    if worker_stats['violation_types']:
        print("\n  Most Common Violations:")
        sorted_violations = sorted(worker_stats['violation_types'].items(), key=lambda x: x[1], reverse=True)
        for violation_type, count in sorted_violations:
            print(f"    {violation_type:25s}: {count} workers")
    
    results = {
        'mAP': mAP,
        'class_aps': class_aps,
        'worker_stats': {
            'total_workers': worker_stats['total_workers'],
            'total_violations': worker_stats['total_violations'],
            'violation_types': dict(worker_stats['violation_types'])
        }
    }
    
    return results


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Universal PPE Detection Evaluator')
    parser.add_argument('--model', required=True, help='Path to trained model (.pth file)')
    parser.add_argument('--data_dir', default='data', help='Data directory')
    parser.add_argument('--split', default='val', choices=['train', 'val', 'test'], help='Dataset split to evaluate')
    parser.add_argument('--num_classes', type=int, default=12, help='Number of classes (including background)')
    parser.add_argument('--backbone', default='resnet101', choices=['resnet50', 'resnet101'], help='Model backbone')
    parser.add_argument('--device', default='cpu', help='Device')
    parser.add_argument('--conf_threshold', type=float, default=0.05, help='Confidence threshold')
    parser.add_argument('--nms_threshold', type=float, default=0.5, help='NMS IoU threshold')
    parser.add_argument('--output', default=None, help='Save results to JSON file')
    
    args = parser.parse_args()
    
    # Load model
    print(f"\nLoading model from: {args.model}")
    model = load_model(args.model, args.num_classes, args.backbone, args.device)
    
    # Evaluate
    results = evaluate_model(
        model, 
        args.data_dir, 
        args.split, 
        args.device, 
        args.conf_threshold, 
        args.nms_threshold
    )
    
    # Save results if requested
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n✅ Results saved to: {args.output}")
    
    print("\n✅ Evaluation complete!\n")
