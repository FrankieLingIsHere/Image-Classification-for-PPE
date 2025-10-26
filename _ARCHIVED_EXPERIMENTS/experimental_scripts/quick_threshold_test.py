#!/usr/bin/env python3
"""
Quick evaluation with adjusted threshold to diagnose the low score issue.
"""
import torch
import torchvision.transforms as T
from pathlib import Path
from PIL import Image
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import xml.etree.ElementTree as ET
from collections import defaultdict
import sys

PPE_CLASSES = [
    'background', 'person', 'hard_hat', 'safety_vest', 'safety_gloves',
    'safety_boots', 'eye_protection', 'no_hard_hat', 'no_safety_vest',
    'no_safety_gloves', 'no_safety_boots', 'no_eye_protection'
]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load model
print("[Loading model...]")
model = fasterrcnn_resnet50_fpn(pretrained=False, num_classes=len(PPE_CLASSES))
checkpoint = torch.load('models/rcnn_baseline.pth', map_location=device)
model.load_state_dict(checkpoint['model_state_dict'], strict=False)
model = model.to(device)
model.eval()

# Get test images
test_split = Path('data/splits/test.txt')
with open(test_split) as f:
    test_images = [line.strip() for line in f if line.strip()]

print(f"[Processing {len(test_images)} test images...]")

# Test different thresholds
thresholds_to_test = [0.08, 0.06, 0.04, 0.02]

for threshold in thresholds_to_test:
    print(f"\n{'='*70}")
    print(f"THRESHOLD: {threshold}")
    print(f"{'='*70}")
    
    total_gt = 0
    total_pred = 0
    correct_matches = 0
    
    for img_name in test_images[:5]:  # Just first 5 for speed
        img_path = Path('data/images') / img_name
        if not img_path.exists():
            continue
        
        # Load image
        img = Image.open(img_path).convert('RGB')
        transform = T.Compose([T.ToTensor()])
        img_tensor = transform(img).unsqueeze(0).to(device)
        
        # Get predictions
        with torch.no_grad():
            outputs = model([img_tensor.squeeze(0)])
        
        pred = outputs[0]
        boxes = pred['boxes'].cpu().numpy()
        scores = pred['scores'].cpu().numpy()
        labels = pred['labels'].cpu().numpy()
        
        # Filter by threshold (exclude person/background)
        mask = (scores >= threshold) & (labels >= 2)  # labels >= 2 = PPE items
        filtered_labels = labels[mask]
        filtered_scores = scores[mask]
        
        # Load ground truth
        anno_path = Path('data/annotations') / img_name.replace('.jpg', '.xml').replace('.png', '.xml')
        gt_classes = []
        if anno_path.exists():
            tree = ET.parse(anno_path)
            root = tree.getroot()
            for obj in root.findall('object'):
                name_elem = obj.find('name')
                if name_elem is not None:
                    cls_name = name_elem.text
                    if cls_name != 'person' and cls_name != 'background':
                        gt_classes.append(cls_name)
        
        total_gt += len(gt_classes)
        total_pred += len(filtered_labels)
        
        # Count matches
        for pred_label in filtered_labels:
            class_name = PPE_CLASSES[pred_label]
            if class_name in gt_classes:
                correct_matches += 1
        
        print(f"{img_name:20s}: GT={len(gt_classes):2d}, Pred={len(filtered_labels):2d}")
    
    recall = correct_matches / total_gt if total_gt > 0 else 0
    precision = correct_matches / total_pred if total_pred > 0 else 0
    print(f"\nAggregate: Recall={recall:.2%}, Precision={precision:.2%}, Matches={correct_matches}/{total_gt}")
