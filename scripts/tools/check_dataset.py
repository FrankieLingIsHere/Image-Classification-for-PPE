#!/usr/bin/env python3
"""
Quick test to see what the dataset returns
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.dataset.ppe_dataset import PPEDataset

def check_dataset_format():
    """Check what the dataset actually returns"""
    
    dataset = PPEDataset(
        data_dir='data',
        split='train',
        img_size=300
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    if len(dataset) > 0:
        sample = dataset[0]
        print(f"Sample type: {type(sample)}")
        
        if isinstance(sample, tuple):
            print(f"Tuple length: {len(sample)}")
            for i, item in enumerate(sample):
                print(f"  Item {i}: {type(item)} - {item.shape if hasattr(item, 'shape') else item}")
        
        elif isinstance(sample, dict):
            print(f"Dictionary keys: {sample.keys()}")
            for key, value in sample.items():
                print(f"  {key}: {type(value)} - {value.shape if hasattr(value, 'shape') else value}")

if __name__ == "__main__":
    check_dataset_format()