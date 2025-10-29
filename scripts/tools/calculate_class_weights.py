"""
Pre-calculate class weights from training annotations.
Saves weights to a JSON file to avoid recalculation overhead during training.
"""

import os
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from collections import defaultdict
import argparse
import numpy as np


def parse_xml_annotation(xml_path):
    """Parse VOC XML annotation to extract class labels."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    labels = []
    for obj in root.findall('object'):
        name = obj.find('name').text
        labels.append(name)
    
    return labels


def calculate_class_weights(data_dir, split='train', strategy='inverse_freq'):
    """
    Calculate class weights from training data.
    
    Args:
        data_dir: Root data directory
        split: Which split to use ('train', 'val', 'test')
        strategy: 'inverse_freq' or 'effective_num'
    
    Returns:
        Dictionary mapping class names to weights
    """
    # Read split file
    split_file = Path(data_dir) / 'splits' / f'{split}.txt'
    if not split_file.exists():
        raise FileNotFoundError(f"Split file not found: {split_file}")
    
    with open(split_file, 'r') as f:
        image_stems = [line.strip() for line in f if line.strip()]
    
    print(f"Calculating class weights for {split} split ({len(image_stems)} images)...")
    
    # Count instances per class
    class_counts = defaultdict(int)
    annotations_dir = Path(data_dir) / 'annotations'
    
    for stem in image_stems:
        xml_path = annotations_dir / f"{stem}.xml"
        if not xml_path.exists():
            print(f"Warning: Missing annotation for {stem}")
            continue
        
        labels = parse_xml_annotation(xml_path)
        for label in labels:
            class_counts[label] += 1
    
    if not class_counts:
        raise ValueError("No annotations found!")
    
    print(f"\nClass distribution:")
    total_instances = sum(class_counts.values())
    for cls, count in sorted(class_counts.items()):
        pct = 100 * count / total_instances
        print(f"  {cls:<20} {count:>6} instances ({pct:>5.2f}%)")
    
    print(f"\nTotal instances: {total_instances}")
    
    # Calculate weights based on strategy
    num_classes = len(class_counts)
    
    if strategy == 'inverse_freq':
        # Inverse frequency (MATCHING train_with_confidence.py):
        # weight = total_instances / (count * num_classes)
        # NO normalization - keep raw weights for better class balancing
        weights = {}
        for cls, count in class_counts.items():
            weights[cls] = total_instances / (count * num_classes)
        
    elif strategy == 'effective_num':
        # Effective number of samples (from Class-Balanced Loss paper)
        # weight = (1 - beta) / (1 - beta^n)
        beta = 0.9999
        weights = {}
        for cls, count in class_counts.items():
            effective_num = (1.0 - np.power(beta, count)) / (1.0 - beta)
            weights[cls] = 1.0 / effective_num
    
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    
    # Use raw weights without normalization
    # This preserves the relative importance better for loss weighting
    normalized_weights = weights
    
    print(f"\nClass weights ({strategy}):")
    for cls in sorted(normalized_weights.keys()):
        print(f"  {cls:<20} {normalized_weights[cls]:.4f}")
    
    return {
        'weights': normalized_weights,
        'class_counts': dict(class_counts),
        'total_instances': total_instances,
        'num_images': len(image_stems),
        'strategy': strategy,
        'split': split
    }


def main():
    parser = argparse.ArgumentParser(description='Calculate class weights')
    parser.add_argument('--data_dir', type=str, default='data',
                        help='Root data directory')
    parser.add_argument('--split', type=str, default='train',
                        choices=['train', 'val', 'test'],
                        help='Which split to calculate weights from')
    parser.add_argument('--strategy', type=str, default='inverse_freq',
                        choices=['inverse_freq', 'effective_num'],
                        help='Weighting strategy')
    parser.add_argument('--output', type=str, default='configs/class_weights.json',
                        help='Output JSON file')
    
    args = parser.parse_args()
    
    # Calculate weights
    result = calculate_class_weights(args.data_dir, args.split, args.strategy)
    
    # Save to file
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"\n✅ Class weights saved to: {output_path}")
    print(f"   Total images: {result['num_images']}")
    print(f"   Total instances: {result['total_instances']}")
    print(f"   Strategy: {result['strategy']}")


if __name__ == '__main__':
    main()
