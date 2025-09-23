#!/usr/bin/env python3
"""
Simple PPE Model Testing with Debug Output
"""

import os
import sys
import torch
import torch.nn.functional as F
import cv2
import numpy as np
from PIL import Image

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.models.ssd import SSD300

def test_model_debug():
    """Test model with detailed debugging"""
    
    print("🔧 PPE Model Debug Test")
    print("=" * 40)
    
    # Load model
    device = torch.device('cpu')
    model_path = 'models/checkpoint_epoch_4.pth'
    
    print("Loading model...")
    model = SSD300(n_classes=13)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print("✅ Model loaded successfully")
    
    # Load and preprocess image
    image_path = 'data/images/image1.png'
    print(f"Loading image: {image_path}")
    
    image = cv2.imread(image_path)
    if image is None:
        print("❌ Could not load image")
        return
    
    print(f"Original image shape: {image.shape}")
    
    # Convert BGR to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Resize to 300x300
    image_resized = cv2.resize(image_rgb, (300, 300))
    
    # Normalize to 0-1
    image_norm = image_resized.astype(np.float32) / 255.0
    
    # Apply ImageNet normalization
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    image_final = (image_norm - mean) / std
    
    # Convert to tensor
    input_tensor = torch.from_numpy(image_final).permute(2, 0, 1).unsqueeze(0).float()
    print(f"Input tensor shape: {input_tensor.shape}")
    print(f"Input tensor dtype: {input_tensor.dtype}")
    print(f"Input tensor range: [{input_tensor.min():.3f}, {input_tensor.max():.3f}]")
    
    # Run inference
    print("\nRunning inference...")
    with torch.no_grad():
        loc_preds, class_preds = model(input_tensor)
    
    print(f"Location predictions shape: {loc_preds.shape}")
    print(f"Class predictions shape: {class_preds.shape}")
    
    # Apply softmax to class predictions
    class_probs = F.softmax(class_preds, dim=2)
    print(f"Class probabilities shape: {class_probs.shape}")
    
    # Check max scores for each class
    print("\nMax confidence per class:")
    for class_id in range(1, 13):  # Skip background
        max_conf = class_probs[0, :, class_id].max().item()
        if max_conf > 0.01:  # Only show if > 1%
            class_name = {
                1: 'person', 2: 'hard_hat', 3: 'safety_vest', 4: 'safety_gloves',
                5: 'safety_boots', 6: 'eye_protection', 7: 'no_hard_hat',
                8: 'no_safety_vest', 9: 'no_safety_gloves', 10: 'no_safety_boots',
                11: 'no_eye_protection', 12: 'unknown'
            }.get(class_id, f'class_{class_id}')
            print(f"  {class_name}: {max_conf:.4f}")
    
    # Test with very low confidence threshold
    confidence_thresholds = [0.01, 0.05, 0.1, 0.2]
    
    for threshold in confidence_thresholds:
        print(f"\nDetections with confidence > {threshold}:")
        detections_found = 0
        
        for class_id in range(1, 13):
            class_scores = class_probs[0, :, class_id]
            above_threshold = class_scores > threshold
            count = above_threshold.sum().item()
            
            if count > 0:
                detections_found += count
                class_name = {
                    1: 'person', 2: 'hard_hat', 3: 'safety_vest', 4: 'safety_gloves',
                    5: 'safety_boots', 6: 'eye_protection', 7: 'no_hard_hat',
                    8: 'no_safety_vest', 9: 'no_safety_gloves', 10: 'no_safety_boots',
                    11: 'no_eye_protection', 12: 'unknown'
                }.get(class_id, f'class_{class_id}')
                max_score = class_scores[above_threshold].max().item()
                print(f"  {class_name}: {count} detections (max: {max_score:.4f})")
        
        if detections_found == 0:
            print("  No detections found")
    
    print("\n✅ Debug test completed!")

if __name__ == "__main__":
    test_model_debug()