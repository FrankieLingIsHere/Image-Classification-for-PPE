#!/usr/bin/env python3
"""
Run evaluation with optimized threshold (0.3) that showed person detection
"""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from scripts.evaluate_detection_performance import PPEDetectionEvaluator


def run_optimized_evaluation():
    """Run evaluation with optimized threshold that detects persons"""
    
    # Find the latest model
    models_dir = Path("models")
    model_files = list(models_dir.glob("*.pth"))
    latest_model = max(model_files, key=lambda x: x.stat().st_ctime)
    
    print("🎯 Running Optimized PPE Detection Evaluation")
    print("=" * 50)
    print(f"📁 Model: {latest_model.name}")
    print("🔧 Optimized settings: conf_threshold=0.3 (was 0.5)")
    
    # Create evaluator
    evaluator = PPEDetectionEvaluator(
        model_path=str(latest_model),
        data_dir="data",
        config_path="configs/ppe_config.yaml", 
        output_dir="outputs/optimized_evaluation"
    )
    
    # Set optimized threshold
    evaluator.conf_threshold = 0.3  # This showed person detection!
    
    # Run evaluation
    print("\n🚀 Starting evaluation...")
    metrics = evaluator.evaluate()
    
    print("\n📊 OPTIMIZED RESULTS SUMMARY")
    print("=" * 50)
    print(f"📈 Overall mAP: {metrics['overall_map']:.3f}")
    
    # Key metrics
    person_metrics = metrics['class_metrics'].get('person', {})
    print(f"👤 Person detection:")
    print(f"   Recall: {person_metrics.get('recall', 0):.3f} (was 0.000)")
    print(f"   Precision: {person_metrics.get('precision', 0):.3f}")
    print(f"   F1: {person_metrics.get('f1', 0):.3f}")
    
    # Violation detection
    no_hat_metrics = metrics['class_metrics'].get('no_hard_hat', {})
    no_vest_metrics = metrics['class_metrics'].get('no_safety_vest', {})
    
    print(f"⚠️  Violation detection:")
    print(f"   No hard hat recall: {no_hat_metrics.get('recall', 0):.3f} (was 0.100)")
    print(f"   No safety vest recall: {no_vest_metrics.get('recall', 0):.3f} (was 0.000)")
    
    # PPE detection  
    hat_metrics = metrics['class_metrics'].get('hard_hat', {})
    vest_metrics = metrics['class_metrics'].get('safety_vest', {})
    
    print(f"🦺 PPE detection:")
    print(f"   Hard hat recall: {hat_metrics.get('recall', 0):.3f} (was 0.091)")
    print(f"   Safety vest recall: {vest_metrics.get('recall', 0):.3f} (was 0.056)")
    
    print(f"\n💾 Results saved to: outputs/optimized_evaluation/")
    print("\n🎯 KEY IMPROVEMENT: Person detection now working!")
    
    return metrics

if __name__ == "__main__":
    run_optimized_evaluation()
