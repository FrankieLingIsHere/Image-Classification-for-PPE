#!/usr/bin/env python3
import json
from pathlib import Path
from collections import Counter

anno_dir = Path('data/annotations')
all_classes = Counter()
total_annotations = 0

print('Checking ground truth annotations...')
print('=' * 60)

# Check test set
test_file = Path('data/splits/test.txt')
if test_file.exists():
    with open(test_file) as f:
        test_images = [line.strip() for line in f if line.strip()]
    
    print(f'\nTest images: {len(test_images)}')
    
    for i, img_name in enumerate(test_images):
        anno_file = anno_dir / img_name.replace('.jpg', '.xml')
        if anno_file.exists():
            import xml.etree.ElementTree as ET
            tree = ET.parse(anno_file)
            root = tree.getroot()
            objects = root.findall('object')
            if i < 5:
                print(f'  {img_name}: {len(objects)} objects')
            for obj in objects:
                name_elem = obj.find('name')
                if name_elem is not None:
                    cls = name_elem.text
                    all_classes[cls] += 1
                    total_annotations += 1

print(f'\nTotal annotations in all {len(test_images)} test images: {total_annotations}')
print(f'\nClass distribution (ALL test images):')
for cls, count in sorted(all_classes.items(), key=lambda x: -x[1]):
    pct = 100 * count / total_annotations if total_annotations > 0 else 0
    print(f'  {cls:20s}: {count:4d} ({pct:5.1f}%)')

print(f'\nAverage objects per image: {total_annotations / len(test_images):.2f}')
