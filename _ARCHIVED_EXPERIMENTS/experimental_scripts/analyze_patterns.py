#!/usr/bin/env python3
"""
Detailed analysis of false positives and missed PPE patterns.
This will help identify architectural improvements needed.
"""

import torch
import torchvision.transforms as T
from pathlib import Path
from PIL import Image, ImageDraw
import xml.etree.ElementTree as ET
from collections import defaultdict, Counter
import json
import numpy as np
from torchvision.models.detection import fasterrcnn_resnet50_fpn

PPE_CLASSES = [
    'background', 'person', 'hard_hat', 'safety_vest', 'safety_gloves',
    'safety_boots', 'eye_protection', 'no_hard_hat', 'no_safety_vest',
    'no_safety_gloves', 'no_safety_boots', 'no_eye_protection'
]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
LOW_THRESHOLD = 0.08  # Use lower threshold for realistic predictions

# Load model
print("[1] Loading model...")
model = fasterrcnn_resnet50_fpn(pretrained=False, num_classes=len(PPE_CLASSES))
checkpoint = torch.load('models/rcnn_baseline.pth', map_location=device)
model.load_state_dict(checkpoint['model_state_dict'], strict=False)
model = model.to(device)
model.eval()

# Get test images
test_split = Path('data/splits/test.txt')
with open(test_split) as f:
    test_images = [line.strip() for line in f if line.strip()]

print(f"[2] Processing {len(test_images)} test images...")

# Statistics
false_positive_analysis = defaultdict(list)  # false_pos_class -> [details]
missed_ppe_analysis = defaultdict(list)      # missed_class -> [details]
size_analysis = defaultdict(lambda: {'fp': [], 'missed': [], 'correct': []})

transform = T.Compose([T.ToTensor()])

for idx, img_name in enumerate(test_images, 1):
    img_path = Path('data/images') / img_name
    anno_path = Path('data/annotations') / img_name.replace('.jpg', '.xml').replace('.png', '.xml')
    
    if not img_path.exists() or not anno_path.exists():
        continue
    
    print(f"  [{idx:2d}/{len(test_images)}] {img_name}", end='', flush=True)
    
    # Load image
    img = Image.open(img_path).convert('RGB')
    img_w, img_h = img.size
    img_tensor = transform(img).unsqueeze(0).to(device)
    
    # Get predictions
    with torch.no_grad():
        outputs = model([img_tensor.squeeze(0)])
    
    pred = outputs[0]
    pred_boxes = pred['boxes'].cpu().numpy()
    pred_scores = pred['scores'].cpu().numpy()
    pred_labels = pred['labels'].cpu().numpy()
    
    # Filter by threshold (exclude background)
    mask = (pred_scores >= LOW_THRESHOLD) & (pred_labels > 0)
    pred_boxes = pred_boxes[mask]
    pred_scores = pred_scores[mask]
    pred_labels = pred_labels[mask]
    
    # Load ground truth
    gt_boxes = []
    gt_labels = []
    tree = ET.parse(anno_path)
    root = tree.getroot()
    for obj in root.findall('object'):
        name_elem = obj.find('name')
        bndbox_elem = obj.find('bndbox')
        if name_elem is not None and bndbox_elem is not None:
            cls_name = name_elem.text
            try:
                cls_idx = PPE_CLASSES.index(cls_name)
                xmin = float(bndbox_elem.find('xmin').text)
                ymin = float(bndbox_elem.find('ymin').text)
                xmax = float(bndbox_elem.find('xmax').text)
                ymax = float(bndbox_elem.find('ymax').text)
                gt_boxes.append([xmin, ymin, xmax, ymax])
                gt_labels.append(cls_idx)
            except:
                pass
    
    gt_boxes = np.array(gt_boxes)
    gt_labels = np.array(gt_labels)
    
    # Calculate IoU
    def iou(box1, box2):
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2
        inter_x1 = max(x1_min, x2_min)
        inter_y1 = max(y1_min, y2_min)
        inter_x2 = min(x1_max, x2_max)
        inter_y2 = min(y1_max, y2_max)
        if inter_x2 < inter_x1 or inter_y2 < inter_y1:
            return 0.0
        inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
        union_area = (x1_max - x1_min) * (y1_max - y1_min) + (x2_max - x2_min) * (y2_max - y2_min) - inter_area
        return inter_area / union_area if union_area > 0 else 0.0
    
    # Match predictions to ground truth
    matched_gt = set()
    for pred_idx, (pred_box, pred_score, pred_label) in enumerate(zip(pred_boxes, pred_scores, pred_labels)):
        best_iou = 0
        best_gt_idx = -1
        pred_class = PPE_CLASSES[pred_label]
        pred_size = (pred_box[2] - pred_box[0]) * (pred_box[3] - pred_box[1])
        
        if len(gt_boxes) > 0:
            ious = [iou(pred_box, gt_box) for gt_box in gt_boxes]
            best_iou = max(ious)
            best_gt_idx = np.argmax(ious)
        
        # Check if this is a correct match
        if best_iou > 0.5 and best_gt_idx >= 0:
            gt_class = PPE_CLASSES[gt_labels[best_gt_idx]]
            if pred_class == gt_class:
                matched_gt.add(best_gt_idx)
                size_analysis[pred_class]['correct'].append(np.sqrt(pred_size))
                continue
        
        # False positive
        false_positive_analysis[pred_class].append({
            'img': img_name,
            'score': float(pred_score),
            'size': np.sqrt(pred_size),
            'img_size': (img_w, img_h),
            'matched_gt': gt_labels[best_gt_idx] if best_gt_idx >= 0 else -1,
            'iou': float(best_iou)
        })
        size_analysis[pred_class]['fp'].append(np.sqrt(pred_size))
    
    # Find missed ground truth
    for gt_idx, (gt_box, gt_label) in enumerate(zip(gt_boxes, gt_labels)):
        if gt_idx not in matched_gt:
            gt_class = PPE_CLASSES[gt_label]
            gt_size = (gt_box[2] - gt_box[0]) * (gt_box[3] - gt_box[1])
            missed_ppe_analysis[gt_class].append({
                'img': img_name,
                'size': np.sqrt(gt_size),
                'img_size': (img_w, img_h),
            })
            size_analysis[gt_class]['missed'].append(np.sqrt(gt_size))
    
    print(" OK")


# Generate report
print("\n" + "="*80)
print("FALSE POSITIVE ANALYSIS")
print("="*80)

total_fp = sum(len(v) for v in false_positive_analysis.values())
print(f"\nTotal False Positives: {total_fp}")
print(f"\nBreakdown by class:")
for cls, fps in sorted(false_positive_analysis.items(), key=lambda x: -len(x[1])):
    count = len(fps)
    avg_score = np.mean([f['score'] for f in fps])
    avg_size = np.mean([f['size'] for f in fps])
    print(f"\n  {cls:20s}: {count:3d} FPs")
    print(f"    - Avg confidence: {avg_score:.4f}")
    print(f"    - Avg size (sqrt_pixels): {avg_size:.1f} (small={avg_size<10}, medium={10<=avg_size<50}, large={avg_size>=50})")
    print(f"    - Top 3 examples:")
    for fp in sorted(fps, key=lambda x: -x['score'])[:3]:
        print(f"      * {fp['img']:20s} score={fp['score']:.4f} size={fp['size']:.1f} iou={fp['iou']:.2f}")

print("\n" + "="*80)
print("MISSED PPE ANALYSIS")
print("="*80)

total_missed = sum(len(v) for v in missed_ppe_analysis.values())
print(f"\nTotal Missed PPE: {total_missed}")
print(f"\nBreakdown by class:")
for cls, missed in sorted(missed_ppe_analysis.items(), key=lambda x: -len(x[1])):
    count = len(missed)
    avg_size = np.mean([m['size'] for m in missed])
    print(f"\n  {cls:20s}: {count:3d} missed")
    print(f"    - Avg size (sqrt_pixels): {avg_size:.1f} (small={avg_size<10}, medium={10<=avg_size<50}, large={avg_size>=50})")
    print(f"    - Examples:")
    for m in missed[:3]:
        print(f"      * {m['img']:20s} size={m['size']:.1f}")

print("\n" + "="*80)
print("SIZE ANALYSIS - PREDICTED vs GROUND TRUTH")
print("="*80)
print("\nClass-wise size distribution (sqrt_pixels):\n")
for cls in sorted(size_analysis.keys()):
    stats = size_analysis[cls]
    print(f"  {cls:20s}:")
    if stats['correct']:
        print(f"    Correct: avg={np.mean(stats['correct']):.1f}, min={np.min(stats['correct']):.1f}, max={np.max(stats['correct']):.1f}")
    if stats['missed']:
        print(f"    Missed:  avg={np.mean(stats['missed']):.1f}, min={np.min(stats['missed']):.1f}, max={np.max(stats['missed']):.1f}")
    if stats['fp']:
        print(f"    FP:      avg={np.mean(stats['fp']):.1f}, min={np.min(stats['fp']):.1f}, max={np.max(stats['fp']):.1f}")

# Architecture recommendations
print("\n" + "="*80)
print("ARCHITECTURAL INSIGHTS & RECOMMENDATIONS")
print("="*80)

small_missed = sum(1 for cls_missed in missed_ppe_analysis.values() 
                   for m in cls_missed if m['size'] < 15)
large_fp = sum(1 for cls_fps in false_positive_analysis.values() 
               for fp in cls_fps if fp['size'] > 30)
high_conf_fp = sum(1 for cls_fps in false_positive_analysis.values() 
                   for fp in cls_fps if fp['score'] > 0.15)

print(f"\nKey Findings:")
print(f"  1. Small missed PPE (size < 15): {small_missed} cases")
print(f"  2. Large false positives (size > 30): {large_fp} cases")
print(f"  3. High confidence false positives (score > 0.15): {high_conf_fp} cases")

print(f"\nRecommended Improvements:")
if small_missed > 0:
    print(f"  - Add FPN with higher resolution features (already have this)")
    print(f"  - Use multi-scale training")
    print(f"  - Add small object detection branch")
    
if large_fp > 0:
    print(f"  - Add context modeling (graph-based or attention)")
    print(f"  - Use hard negative mining in training")
    print(f"  - Add context-aware NMS (remove detections inconsistent with image semantics)")

if high_conf_fp > 0:
    print(f"  - Add self-supervised pretraining on worker/PPE images")
    print(f"  - Use confidence calibration loss")
    print(f"  - Implement auxiliary semantic segmentation task")

print(f"\nProposed Architecture Enhancements:")
print(f"  1. SELF-SUPERVISED PRETRAINING")
print(f"     - Use contrastive learning on PPE/worker image patches")
print(f"     - Improves feature quality for downstream detection")
print(f"     - Especially helps with confidence calibration")
print(f"\n  2. MULTI-TASK LEARNING")
print(f"     - Add semantic segmentation (PPE vs non-PPE vs background)")
print(f"     - Add instance segmentation for precise localization")
print(f"     - Helps model understand spatial context")
print(f"\n  3. CONTEXT-AWARE MODULES")
print(f"     - Graph Attention Network (GAT) on detected boxes")
print(f"     - Model relationships: person <- PPE items")
print(f"     - Filter impossible detections (e.g., boots floating in sky)")
print(f"\n  4. HARD NEGATIVE MINING")
print(f"     - Identify difficult false positive cases in training")
print(f"     - Increase sampling of hard negatives")
print(f"     - Improves precision")
print(f"\n  5. MULTI-SCALE PYRAMID")
print(f"     - Already using FPN, but can add explicit multi-scale branch")
print(f"     - Separate detection head for small/medium/large objects")

print("\n" + "="*80)
