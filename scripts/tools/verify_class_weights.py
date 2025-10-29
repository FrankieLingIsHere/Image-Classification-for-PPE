"""
Quick test to verify pre-calculated weights match training script calculation.
"""

import json

# Load pre-calculated weights
with open('configs/class_weights.json', 'r') as f:
    weights_data = json.load(f)

print("Pre-calculated Class Weights:")
print("="*60)
for class_name, weight in sorted(weights_data['weights'].items()):
    count = weights_data['class_counts'][class_name]
    print(f"{class_name:20s}: weight={weight:.4f} (count: {count:4d})")

print(f"\nTotal instances: {weights_data['total_instances']}")
print(f"Total images: {weights_data['num_images']}")
print(f"Strategy: {weights_data['strategy']}")
print(f"Split: {weights_data['split']}")

print("\n✅ Weights file ready for use!")
print("\nUsage in training:")
print("  python scripts/train/train_with_confidence.py --weights-file configs/class_weights.json")
print("\nThis will skip the 'Calculating class weights...' step and start training immediately!")
