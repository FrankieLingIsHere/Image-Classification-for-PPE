#!/usr/bin/env python3
"""
Simple test to check dataset format and model inference
"""

import os
import sys
import torch

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.models.ssd import SSD300
from src.dataset.ppe_dataset import PPEDataset

def simple_test():
    print("🔍 Simple Dataset and Model Test")
    print("=" * 40)
    
    # Load a single test image
    dataset = PPEDataset(
        data_dir="data",
        split='test',
        transforms=None,
        img_size=300
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    if len(dataset) > 0:
        # Get first item
        item = dataset[0]
        print(f"Item type: {type(item)}")
        print(f"Item length: {len(item)}")
        
        if len(item) >= 2:
            image, annotations = item[0], item[1]
            print(f"Image shape: {image.shape}")
            print(f"Annotations type: {type(annotations)}")
            print(f"Annotations: {annotations}")
            
            # Test model
            model = SSD300(n_classes=12)
            model.eval()
            
            with torch.no_grad():
                # Add batch dimension
                image_batch = image.unsqueeze(0)
                print(f"Input batch shape: {image_batch.shape}")
                
                # Forward pass
                predictions = model(image_batch)
                print(f"Predictions type: {type(predictions)}")
                
                if isinstance(predictions, tuple):
                    print(f"Predictions length: {len(predictions)}")
                    for i, pred in enumerate(predictions):
                        print(f"  Prediction {i} shape: {pred.shape}")
                else:
                    print(f"Predictions shape: {predictions.shape}")

if __name__ == "__main__":
    simple_test()
