"""
Quick diagnostic script to inspect what the model is actually predicting
"""

import torch
import torchvision
from torchvision import transforms as T
from PIL import Image
import sys
import os
from collections import Counter

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

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
            print("⚠️  Using ResNet50 as fallback")
            model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None, num_classes=num_classes)
    else:
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None, num_classes=num_classes)
    
    # Load weights
    checkpoint = torch.load(model_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    return model


def inspect_predictions(model, data_dir, split='test', device='cpu', num_images=5):
    """Inspect predictions on a few test images"""
    
    # Load a few test images
    split_file = os.path.join(data_dir, 'splits', f'{split}.txt')
    with open(split_file, 'r') as f:
        image_files = [line.strip() for line in f.readlines()][:num_images]
    
    transforms = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    print(f"\n{'='*80}")
    print(f"INSPECTING PREDICTIONS ON {num_images} {split.upper()} IMAGES")
    print(f"{'='*80}\n")
    
    all_class_counts = Counter()
    all_scores = []
    
    with torch.no_grad():
        for img_name in image_files:
            img_path = os.path.join(data_dir, 'images', img_name)
            img = Image.open(img_path).convert('RGB')
            img_tensor = transforms(img).unsqueeze(0).to(device)
            
            # Run inference
            outputs = model(img_tensor)[0]
            
            boxes = outputs['boxes'].cpu()
            labels = outputs['labels'].cpu()
            scores = outputs['scores'].cpu()
            
            print(f"\n📷 Image: {img_name}")
            print(f"   Total predictions: {len(labels)}")
            
            # Show predictions by class (all confidence levels)
            class_counts = Counter()
            for label, score in zip(labels, scores):
                class_name = PPE_CLASSES[label.item()]
                class_counts[class_name] += 1
                all_class_counts[class_name] += 1
                all_scores.append(score.item())
            
            print(f"   Predictions by class (all confidences):")
            for class_name, count in class_counts.most_common():
                print(f"     {class_name:25s}: {count:3d}")
            
            # Show high-confidence predictions
            high_conf_mask = scores >= 0.3
            if high_conf_mask.sum() > 0:
                high_conf_labels = labels[high_conf_mask]
                high_conf_scores = scores[high_conf_mask]
                print(f"\n   High-confidence predictions (>=0.3):")
                for label, score in zip(high_conf_labels, high_conf_scores):
                    class_name = PPE_CLASSES[label.item()]
                    print(f"     {class_name:25s}: {score:.3f}")
            else:
                print(f"\n   ⚠️  NO predictions with confidence >= 0.3")
    
    print(f"\n\n{'='*80}")
    print(f"AGGREGATE STATISTICS (across {num_images} images)")
    print(f"{'='*80}\n")
    print(f"Total predictions: {sum(all_class_counts.values())}")
    print(f"\nPredictions by class:")
    for class_name, count in all_class_counts.most_common():
        print(f"  {class_name:25s}: {count:4d}")
    
    if all_scores:
        import numpy as np
        scores_array = np.array(all_scores)
        print(f"\nConfidence score distribution:")
        print(f"  Min:    {scores_array.min():.4f}")
        print(f"  Max:    {scores_array.max():.4f}")
        print(f"  Mean:   {scores_array.mean():.4f}")
        print(f"  Median: {np.median(scores_array):.4f}")
        print(f"  P90:    {np.percentile(scores_array, 90):.4f}")
        print(f"  P95:    {np.percentile(scores_array, 95):.4f}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Inspect model predictions')
    parser.add_argument('--model', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--data_dir', type=str, default='data', help='Data directory')
    parser.add_argument('--split', type=str, default='test', help='Dataset split')
    parser.add_argument('--num_images', type=int, default=5, help='Number of images to inspect')
    parser.add_argument('--backbone', type=str, default='resnet101', help='Backbone architecture')
    
    args = parser.parse_args()
    
    print(f"\n🔍 Loading model from: {args.model}")
    model = load_model(args.model, backbone=args.backbone)
    
    inspect_predictions(model, args.data_dir, split=args.split, num_images=args.num_images)
    print("\n✅ Inspection complete!\n")
