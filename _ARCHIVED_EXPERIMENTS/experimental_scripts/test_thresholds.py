#!/usr/bin/env python3
"""
Test evaluation with different confidence thresholds to find the optimal one.
"""
import subprocess
import sys
from pathlib import Path

# Test different confidence thresholds
thresholds = [0.5, 0.3, 0.2, 0.15, 0.1, 0.05]

print("=" * 70)
print("TESTING EVALUATION WITH DIFFERENT CONFIDENCE THRESHOLDS")
print("=" * 70)

for threshold in thresholds:
    print(f"\n\n{'='*70}")
    print(f"Testing threshold: {threshold}")
    print(f"{'='*70}")
    
    cmd = [
        sys.executable, 'scripts/eval/evaluate_detection_performance.py',
        '--model_path', 'models/rcnn_baseline.pth',
        '--split', 'test',
        '--conf_threshold', str(threshold)
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    # Extract the summary lines
    for line in result.stdout.split('\n'):
        if 'Overall mAP' in line or 'RESULT' in line or 'Missed' in line or 'False positive' in line or 'Missed violation' in line:
            print(line)
