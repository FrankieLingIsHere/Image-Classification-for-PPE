"""
Create ground truth CSV table from annotations.
Shows counts of person and each PPE class per image.
"""

import os
import csv
import xml.etree.ElementTree as ET
from pathlib import Path
from collections import defaultdict
import argparse


# PPE class names (excluding background)
PPE_CLASSES = [
    'person', 'hard_hat', 'safety_vest', 'safety_gloves',
    'safety_boots', 'eye_protection', 'no_hard_hat', 'no_safety_vest',
    'no_safety_gloves', 'no_safety_boots', 'no_eye_protection'
]


def parse_xml_annotation(xml_path):
    """Parse VOC XML annotation and count instances per class."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    class_counts = defaultdict(int)
    
    for obj in root.findall('object'):
        name = obj.find('name').text
        if name in PPE_CLASSES:
            class_counts[name] += 1
    
    return class_counts


def create_ground_truth_table(data_dir, split='all', output_file='ground_truth_table.csv'):
    """
    Create ground truth CSV table with PPE counts per image.
    
    Args:
        data_dir: Root data directory
        split: 'train', 'val', 'test', or 'all'
        output_file: Output CSV file path
    """
    annotations_dir = Path(data_dir) / 'annotations'
    
    # Get list of images to process
    if split == 'all':
        # Get all annotation files
        xml_files = sorted(annotations_dir.glob('*.xml'))
        image_stems = [f.stem for f in xml_files]
    else:
        # Read from split file
        split_file = Path(data_dir) / 'splits' / f'{split}.txt'
        if not split_file.exists():
            print(f"Error: Split file not found: {split_file}")
            return
        
        with open(split_file, 'r') as f:
            image_stems = [line.strip() for line in f if line.strip()]
    
    print(f"Processing {len(image_stems)} images from '{split}' split...")
    
    # Collect data for each image
    table_data = []
    overall_totals = defaultdict(int)
    
    for stem in image_stems:
        xml_path = annotations_dir / f"{stem}.xml"
        
        if not xml_path.exists():
            print(f"Warning: Annotation not found for {stem}")
            continue
        
        # Parse annotation
        class_counts = parse_xml_annotation(xml_path)
        
        # Create row for this image
        row = {'image': stem}
        total_items = 0
        
        for cls in PPE_CLASSES:
            count = class_counts.get(cls, 0)
            row[cls] = count
            overall_totals[cls] += count
            total_items += count
        
        row['total'] = total_items
        overall_totals['total'] += total_items
        
        table_data.append(row)
    
    # Sort by image name (natural sort for image1, image2, ..., image10, etc.)
    def natural_sort_key(item):
        import re
        return [int(text) if text.isdigit() else text.lower()
                for text in re.split('([0-9]+)', item['image'])]
    
    table_data.sort(key=natural_sort_key)
    
    # Write CSV file
    fieldnames = ['image'] + PPE_CLASSES + ['total']
    
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        # Write header
        writer.writeheader()
        
        # Write data rows
        for row in table_data:
            writer.writerow(row)
        
        # Write total row
        total_row = {'image': 'TOTAL'}
        for cls in PPE_CLASSES:
            total_row[cls] = overall_totals[cls]
        total_row['total'] = overall_totals['total']
        
        writer.writerow(total_row)
    
    # Print summary
    print(f"\n{'='*80}")
    print("GROUND TRUTH SUMMARY")
    print(f"{'='*80}")
    print(f"Total images: {len(table_data)}")
    print(f"\nClass counts:")
    for cls in PPE_CLASSES:
        count = overall_totals[cls]
        pct = 100 * count / overall_totals['total'] if overall_totals['total'] > 0 else 0
        print(f"  {cls:20s}: {count:>6} ({pct:>5.1f}%)")
    print(f"\n  {'Total items':20s}: {overall_totals['total']:>6}")
    
    print(f"\n✅ Ground truth table saved to: {output_file}")
    print(f"   Rows: {len(table_data) + 1} (including TOTAL row)")
    print(f"   Columns: {len(fieldnames)}")


def main():
    parser = argparse.ArgumentParser(description='Create ground truth CSV table')
    parser.add_argument('--data_dir', type=str, default='data',
                        help='Root data directory')
    parser.add_argument('--split', type=str, default='all',
                        choices=['all', 'train', 'val', 'test'],
                        help='Which split to process (default: all)')
    parser.add_argument('--output', type=str, default='ground_truth_table.csv',
                        help='Output CSV file')
    
    args = parser.parse_args()
    
    create_ground_truth_table(args.data_dir, args.split, args.output)


if __name__ == '__main__':
    main()
