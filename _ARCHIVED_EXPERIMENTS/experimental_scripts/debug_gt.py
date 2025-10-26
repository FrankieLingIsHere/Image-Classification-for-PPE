#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from scripts.eval.evaluate_detection_performance import PPEDetectionEvaluator

evaluator = PPEDetectionEvaluator(
    model_path='models/rcnn_baseline.pth',
    data_dir='data',
    config_path='configs/ppe_config.yaml',
    output_dir='outputs/evaluation_results'
)

print("Loading ground truth...")
gt = evaluator._load_ground_truth(split='test')

print(f"Loaded {len(gt)} ground truth images")

for i, (img_name, annos) in enumerate(list(gt.items())[:5]):
    print(f"\n{img_name}:")
    print(f"  Annotations: {len(annos)}")
    for anno in annos:
        print(f"    - {anno['class']}: {anno['bbox']}")
