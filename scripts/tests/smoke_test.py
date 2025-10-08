"""Smoke test for PPEDetectionEvaluator

Runs a very small evaluation on 1-2 images to ensure the evaluator imports and runs.
"""
import os
from pathlib import Path

from scripts.evaluate_detection_performance import PPEDetectionEvaluator


def run_smoke():
    repo_root = Path(__file__).parent.parent
    model_path = repo_root / 'models' / 'best_model_regularized.pth'
    data_dir = repo_root / 'data'
    config_path = repo_root / 'configs' / 'best_runtime_config.yaml'
    output_dir = repo_root / 'outputs' / 'smoke_test_output'

    evaluator = PPEDetectionEvaluator(
        model_path=str(model_path),
        data_dir=str(data_dir),
        config_path=str(config_path),
        output_dir=str(output_dir)
    )

    # Evaluate but only process the first 2 images to keep the test fast
    results = evaluator.evaluate(split='test')
    print('Smoke test finished. mAP:', results.get('summary', {}).get('map_score'))


if __name__ == '__main__':
    run_smoke()
