from src.dataset.ppe_dataset import load_ppe_images_and_annotations
from collections import Counter

PPE_CLASSES = [
    'background', 'person', 'hard_hat', 'safety_vest', 'safety_gloves',
    'safety_boots', 'eye_protection', 'no_hard_hat', 'no_safety_vest',
    'no_safety_gloves', 'no_safety_boots', 'no_eye_protection'
]

class2idx = {c: i for i, c in enumerate(PPE_CLASSES)}

print("\n" + "="*70)
print("LABEL DISTRIBUTION ANALYSIS")
print("="*70)

for split in ['train', 'val', 'test']:
    images = load_ppe_images_and_annotations('data', class2idx, split)
    
    label_counts = Counter()
    for img_info in images:
        # Check both 'annotations' and 'detections' keys
        annotations = img_info.get('annotations', img_info.get('detections', []))
        for ann in annotations:
            # Check for both 'category_id' and 'label' keys
            cat_id = ann.get('category_id', ann.get('label'))
            if cat_id is not None:
                label_counts[cat_id] += 1
    
    total_annotations = sum(label_counts.values())
    
    print(f"\n{split.upper()} SPLIT:")
    print(f"  Total images: {len(images)}")
    print(f"  Total annotations: {total_annotations}")
    print(f"\n  Class Distribution:")
    
    if total_annotations > 0:
        for class_id in sorted(label_counts.keys()):
            class_name = PPE_CLASSES[class_id] if class_id < len(PPE_CLASSES) else f'unknown_{class_id}'
            count = label_counts[class_id]
            percentage = (count / total_annotations * 100) if total_annotations > 0 else 0
            bar_length = int(count / max(label_counts.values()) * 30) if label_counts else 0
            bar = '[' + '█' * bar_length + '-' * (30 - bar_length) + ']'
            print(f"    {class_id:2d}. {class_name:20s} : {count:4d} ({percentage:5.1f}%) {bar}")
    else:
        print("    No annotations found")

# Summary statistics
print("\n" + "="*70)
print("SUMMARY")
print("="*70)

all_images = load_ppe_images_and_annotations('data', class2idx, 'train')
all_counts = Counter()
for img_info in all_images:
    annotations = img_info.get('annotations', img_info.get('detections', []))
    for ann in annotations:
        cat_id = ann.get('category_id', ann.get('label'))
        if cat_id is not None:
            all_counts[cat_id] += 1

if len(all_counts) > 0:
    print("\nClass Imbalance Ratio (max/min):")
    max_count = max(all_counts.values())
    min_count = min(all_counts.values())
    print(f"  Max: {max_count} (person)")
    print(f"  Min: {min_count} (no_safety_boots)")
    print(f"  Ratio: {max_count/min_count:.1f}x imbalance")

    print("\nRare Classes (< 5% of max):")
    threshold = max_count * 0.05
    for class_id, count in sorted(all_counts.items()):
        if count < threshold:
            class_name = PPE_CLASSES[class_id]
            print(f"  - {class_name}: {count} instances ({count/max_count*100:.1f}%)")
else:
    print("\nNo annotations found in dataset")
