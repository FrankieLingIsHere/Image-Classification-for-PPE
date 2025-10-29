"""
Generate ground truth CSV table from annotations.
Lists PPE counts for each image with totals.
"""

import xml.etree.ElementTree as ET
from pathlib import Path
import csv
from collections import defaultdict
import argparse


def parse_xml_annotation(xml_path):
    """Parse VOC XML annotation to count objects by class."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    class_counts = defaultdict(int)
    for obj in root.findall('object'):
        name = obj.find('name').text
        class_counts[name] += 1
    
    return class_counts


def generate_ground_truth_table(data_dir, split, output_csv):
    """
    Generate ground truth CSV table.
    
    Args:
        data_dir: Root data directory
        split: Which split to use ('train', 'val', 'test', or 'all')
        output_csv: Output CSV file path
    """
    data_dir = Path(data_dir)
    annotations_dir = data_dir / 'annotations'
    splits_dir = data_dir / 'splits'
    
    # Define PPE classes
    ppe_classes = [
        'person', 'hard_hat', 'safety_vest', 'safety_gloves', 'safety_boots',
        'eye_protection', 'no_hard_hat', 'no_safety_vest', 'no_safety_gloves',
        'no_safety_boots', 'no_eye_protection'
    ]
    
    # Get image stems based on split
    if split == 'all':
        # Get all images with annotations
        image_stems = []
        for xml_file in sorted(annotations_dir.glob('*.xml')):
            image_stems.append(xml_file.stem)
    else:
        # Read from split file
        split_file = splits_dir / f'{split}.txt'
        if not split_file.exists():
            raise FileNotFoundError(f"Split file not found: {split_file}")
        
        with open(split_file, 'r') as f:
            image_stems = [line.strip() for line in f if line.strip()]
    
    print(f"Processing {len(image_stems)} images from '{split}' split...")
    
    # Collect data for all images
    table_data = []
    total_counts = defaultdict(int)
    
    for stem in sorted(image_stems):
        xml_path = annotations_dir / f"{stem}.xml"
        
        if not xml_path.exists():
            print(f"Warning: Missing annotation for {stem}")
            continue
        
        # Parse annotation
        class_counts = parse_xml_annotation(xml_path)
        
        # Create row data
        row = {'image': stem}
        row_total = 0
        
        for cls in ppe_classes:
            count = class_counts.get(cls, 0)
            row[cls] = count
            row_total += count
            total_counts[cls] += count
        
        row['total_objects'] = row_total
        total_counts['total_objects'] += row_total
        
        table_data.append(row)
    
    # Write to CSV
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', newline='') as f:
        # Define columns
        columns = ['image'] + ppe_classes + ['total_objects']
        
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        
        # Write data rows
        for row in table_data:
            writer.writerow(row)
        
        # Write total row
        total_row = {'image': 'TOTAL'}
        for cls in ppe_classes:
            total_row[cls] = total_counts[cls]
        total_row['total_objects'] = total_counts['total_objects']
        writer.writerow(total_row)
    
    print(f"\n✅ Ground truth table saved to: {output_path}")
    print(f"   Total images: {len(table_data)}")
    print(f"   Total objects: {total_counts['total_objects']}")
    
    # Print summary
    print(f"\nClass distribution:")
    for cls in ppe_classes:
        count = total_counts[cls]
        pct = 100 * count / total_counts['total_objects'] if total_counts['total_objects'] > 0 else 0
        print(f"  {cls:20s}: {count:4d} ({pct:5.2f}%)")


def main():
    parser = argparse.ArgumentParser(description='Generate ground truth CSV table')
    parser.add_argument('--data_dir', type=str, default='data',
                        help='Root data directory')
    parser.add_argument('--split', type=str, default='all',
                        choices=['train', 'val', 'test', 'all'],
                        help='Which split to process (default: all)')
    parser.add_argument('--output', type=str, default='data/ground_truth_table.csv',
                        help='Output CSV file path')
    
    args = parser.parse_args()
    
    generate_ground_truth_table(args.data_dir, args.split, args.output)


if __name__ == '__main__':
    main()
