#!/usr/bin/env python3
"""
Debug script to inspect model predictions on actual test images.
"""

import torch
import torchvision.transforms as T
from pathlib import Path
from PIL import Image
import json
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

PPE_CLASSES = [
    'background', 'person', 'hard_hat', 'safety_vest', 'safety_gloves',
    'safety_boots', 'eye_protection', 'no_hard_hat', 'no_safety_vest',
    'no_safety_gloves', 'no_safety_boots', 'no_eye_protection'
]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load model
print("Loading model from models/rcnn_baseline.pth...")
model = fasterrcnn_resnet50_fpn(pretrained=False, num_classes=len(PPE_CLASSES))
checkpoint = torch.load('models/rcnn_baseline.pth', map_location=device)
model.load_state_dict(checkpoint['model_state_dict'], strict=False)
model = model.to(device)
model.eval()
print(f"[OK] Model loaded with {len(PPE_CLASSES)} classes")

# Load test image
test_img_path = Path('data/images/image95.jpg')
if not test_img_path.exists():
    print(f"ERROR: {test_img_path} not found")
    exit(1)

print(f"\nProcessing: {test_img_path}")
img = Image.open(test_img_path).convert('RGB')
print(f"  Image size: {img.size}")

# Prepare input
transform = T.Compose([T.ToTensor()])
img_tensor = transform(img).unsqueeze(0).to(device)

# Get predictions
with torch.no_grad():
    outputs = model([img_tensor.squeeze(0)])

print(f"\n[PREDICTIONS]")
pred = outputs[0]
boxes = pred['boxes'].cpu().numpy()
scores = pred['scores'].cpu().numpy()
labels = pred['labels'].cpu().numpy()

print(f"  Total detections: {len(boxes)}")
print(f"  Score range: [{scores.min():.4f}, {scores.max():.4f}]")
print(f"  Unique labels: {sorted(set(labels))}")

# Print first 20 detections
print(f"\nFirst 20 detections (all):")
for i in range(min(20, len(boxes))):
    label_idx = int(labels[i])
    class_name = PPE_CLASSES[label_idx] if label_idx < len(PPE_CLASSES) else f"UNKNOWN_{label_idx}"
    bbox = boxes[i]
    score = scores[i]
    print(f"  {i:2d}: {class_name:20s} score={score:.4f} box=[{bbox[0]:.1f}, {bbox[1]:.1f}, {bbox[2]:.1f}, {bbox[3]:.1f}]")

# Count by class
from collections import Counter
label_counts = Counter(labels)
print(f"\nDetections by class:")
for label_idx in sorted(label_counts.keys()):
    class_name = PPE_CLASSES[label_idx] if label_idx < len(PPE_CLASSES) else f"UNKNOWN_{label_idx}"
    count = label_counts[label_idx]
    avg_score = scores[labels == label_idx].mean()
    print(f"  {class_name:20s}: {count:3d} detections (avg score={avg_score:.4f})")

# Check ground truth for this image
print(f"\n[GROUND TRUTH]")
anno_path = Path('data/annotations/image95.xml')
if anno_path.exists():
    import xml.etree.ElementTree as ET
    tree = ET.parse(anno_path)
    root = tree.getroot()
    objects = root.findall('object')
    print(f"  Total objects: {len(objects)}")
    gt_classes = Counter()
    for obj in objects:
        name_elem = obj.find('name')
        if name_elem is not None:
            cls = name_elem.text
            gt_classes[cls] += 1
    print(f"  By class:")
    for cls in sorted(gt_classes.keys()):
        print(f"    {cls:20s}: {gt_classes[cls]}")
