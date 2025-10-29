"""
Create train/val/test splits from images and annotations.
Ensures stratification by PPE class distribution.
"""

import os
import json
import random
import xml.etree.ElementTree as ET
from pathlib import Path
from collections import defaultdict
import argparse


def parse_xml_annotation(xml_path):
    """Parse VOC XML annotation to extract class labels."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    classes = set()
    for obj in root.findall('object'):
        name = obj.find('name').text
        classes.add(name)
    
    return classes


def get_image_with_annotations(data_dir):
    """Get list of images that have corresponding annotations."""
    images_dir = Path(data_dir) / 'images'
    annotations_dir = Path(data_dir) / 'annotations'
    
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png']:
        image_files.extend(images_dir.glob(ext))
    
    # Keep only images that have annotations
    valid_images = []
    for img_path in sorted(image_files):
        xml_path = annotations_dir / f"{img_path.stem}.xml"
        if xml_path.exists():
            valid_images.append(img_path.stem)
    
    return valid_images


def stratified_split(image_stems, annotations_dir, train_ratio=0.8, val_ratio=0.1, seed=42):
    """
    Create stratified splits ensuring class distribution is maintained.
    
    Args:
        image_stems: List of image filenames (without extension)
        annotations_dir: Path to annotations directory
        train_ratio: Fraction for training set
        val_ratio: Fraction for validation set
        seed: Random seed for reproducibility
    """
    random.seed(seed)
    
    # Parse all annotations to get class distributions
    image_classes = {}
    class_counts = defaultdict(int)
    
    print("Parsing annotations...")
    for stem in image_stems:
        xml_path = Path(annotations_dir) / f"{stem}.xml"
        classes = parse_xml_annotation(xml_path)
        image_classes[stem] = classes
        
        for cls in classes:
            class_counts[cls] += 1
    
    print(f"\nTotal images: {len(image_stems)}")
    print(f"Class distribution:")
    for cls, count in sorted(class_counts.items()):
        print(f"  {cls}: {count} images")
    
    # Group images by their class combinations for stratification
    class_signature_groups = defaultdict(list)
    for stem, classes in image_classes.items():
        # Create a signature from sorted class names
        signature = tuple(sorted(classes))
        class_signature_groups[signature].append(stem)
    
    print(f"\nFound {len(class_signature_groups)} unique class combinations")
    
    # Split each group proportionally
    train_set = []
    val_set = []
    test_set = []
    
    for signature, stems in class_signature_groups.items():
        random.shuffle(stems)
        n = len(stems)
        
        # For small groups, ensure at least 1 sample in val/test if possible
        if n >= 3:
            n_train = max(1, int(n * train_ratio))
            n_val = max(1, int(n * val_ratio))
        elif n == 2:
            n_train = 1
            n_val = 1
        else:  # n == 1
            n_train = 1
            n_val = 0
        
        train_set.extend(stems[:n_train])
        val_set.extend(stems[n_train:n_train + n_val])
        test_set.extend(stems[n_train + n_val:])
    
    # Shuffle final sets
    random.shuffle(train_set)
    random.shuffle(val_set)
    random.shuffle(test_set)
    
    return train_set, val_set, test_set


def verify_splits(train_set, val_set, test_set, annotations_dir):
    """Verify class distribution across splits."""
    def count_classes(image_stems):
        class_counts = defaultdict(int)
        for stem in image_stems:
            xml_path = Path(annotations_dir) / f"{stem}.xml"
            classes = parse_xml_annotation(xml_path)
            for cls in classes:
                class_counts[cls] += 1
        return class_counts
    
    print("\n" + "="*60)
    print("SPLIT VERIFICATION")
    print("="*60)
    
    train_classes = count_classes(train_set)
    val_classes = count_classes(val_set)
    test_classes = count_classes(test_set)
    
    all_classes = set(train_classes.keys()) | set(val_classes.keys()) | set(test_classes.keys())
    
    print(f"\nTrain: {len(train_set)} images")
    print(f"Val:   {len(val_set)} images")
    print(f"Test:  {len(test_set)} images")
    print(f"Total: {len(train_set) + len(val_set) + len(test_set)} images")
    
    print("\nClass distribution across splits:")
    print(f"{'Class':<20} {'Train':<10} {'Val':<10} {'Test':<10} {'Total':<10}")
    print("-" * 60)
    
    for cls in sorted(all_classes):
        train_count = train_classes.get(cls, 0)
        val_count = val_classes.get(cls, 0)
        test_count = test_classes.get(cls, 0)
        total = train_count + val_count + test_count
        
        print(f"{cls:<20} {train_count:<10} {val_count:<10} {test_count:<10} {total:<10}")


def save_splits(train_set, val_set, test_set, output_dir):
    """Save split files."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save train split
    with open(output_dir / 'train.txt', 'w') as f:
        for stem in sorted(train_set):
            f.write(f"{stem}\n")
    
    # Save val split
    with open(output_dir / 'val.txt', 'w') as f:
        for stem in sorted(val_set):
            f.write(f"{stem}\n")
    
    # Save test split
    with open(output_dir / 'test.txt', 'w') as f:
        for stem in sorted(test_set):
            f.write(f"{stem}\n")
    
    print(f"\n✅ Splits saved to {output_dir}/")
    print(f"   - train.txt: {len(train_set)} images")
    print(f"   - val.txt:   {len(val_set)} images")
    print(f"   - test.txt:  {len(test_set)} images")


def main():
    parser = argparse.ArgumentParser(description='Create train/val/test splits')
    parser.add_argument('--data_dir', type=str, default='data',
                        help='Root data directory containing images/ and annotations/')
    parser.add_argument('--output_dir', type=str, default='data/splits',
                        help='Output directory for split files')
    parser.add_argument('--train_ratio', type=float, default=0.8,
                        help='Training set ratio (default: 0.8)')
    parser.add_argument('--val_ratio', type=float, default=0.1,
                        help='Validation set ratio (default: 0.1)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility (default: 42)')
    
    args = parser.parse_args()
    
    # Get all valid images
    print(f"Scanning {args.data_dir}...")
    image_stems = get_image_with_annotations(args.data_dir)
    
    if not image_stems:
        print(f"❌ No images with annotations found in {args.data_dir}")
        return
    
    # Create stratified splits
    train_set, val_set, test_set = stratified_split(
        image_stems,
        Path(args.data_dir) / 'annotations',
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed
    )
    
    # Verify splits
    verify_splits(train_set, val_set, test_set, Path(args.data_dir) / 'annotations')
    
    # Save splits
    save_splits(train_set, val_set, test_set, args.output_dir)
    
    print("\n✅ Done!")


if __name__ == '__main__':
    main()
