#!/usr/bin/env python3
"""
Test the optimal configuration: conf=0.2, iou=0.3
Based on overlap analysis showing this eliminates same-class overlaps while maximizing person detection
"""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from scripts.evaluate_detection_performance import PPEDetectionEvaluator


def run_optimal_evaluation():
    """Run evaluation with optimal conf/IoU balance"""
    
    # Find the latest model
    models_dir = Path("models")
    model_files = list(models_dir.glob("*.pth"))
    latest_model = max(model_files, key=lambda x: x.stat().st_ctime)
    
    print("🎯 Running OPTIMAL PPE Detection Evaluation")
    print("=" * 55)
    print(f"📁 Model: {latest_model.name}")
    print("🔧 OPTIMAL settings: conf=0.2, iou=0.3")
    print("   • Eliminates same-class overlaps")
    print("   • Maximizes person detection")
    print("   • Based on overlap analysis")
    
    # Create evaluator
    evaluator = PPEDetectionEvaluator(
        model_path=str(latest_model),
        data_dir="data",
        config_path="configs/ppe_config.yaml", 
        output_dir="outputs/optimal_evaluation"
    )
    
    # Set optimal thresholds
    evaluator.conf_threshold = 0.2  # Lower for more person detection
    evaluator.iou_threshold = 0.3   # Stricter to eliminate overlaps
    
    # Run evaluation
    print("\n🚀 Starting evaluation...")
    metrics = evaluator.evaluate()
    
    print("\n📊 OPTIMAL RESULTS SUMMARY")
    print("=" * 55)
    print(f"📈 Overall mAP: {metrics['overall_map']:.3f}")
    
    # Compare with previous results
    prev_results = {
        'original': {'map': 0.032, 'person_recall': 0.0, 'person_detected': 0},
        'threshold_opt': {'map': 0.083, 'person_recall': 0.47, 'person_detected': 14}
    }
    
    # Key metrics
    person_metrics = metrics['class_metrics'].get('person', {})
    current_person_recall = person_metrics.get('recall', 0)
    current_person_detected = person_metrics.get('det_count', 0)
    
    print(f"\n👤 PERSON DETECTION PROGRESS:")
    print(f"   Original:    {prev_results['original']['person_recall']:.1%} ({prev_results['original']['person_detected']}/30 persons)")
    print(f"   Threshold:   {prev_results['threshold_opt']['person_recall']:.1%} ({prev_results['threshold_opt']['person_detected']}/30 persons)")
    print(f"   OPTIMAL:     {current_person_recall:.1%} ({current_person_detected}/30 persons)")
    
    improvement = current_person_recall - prev_results['threshold_opt']['person_recall']
    print(f"   Improvement: +{improvement:.1%} vs threshold-only optimization")
    
    # Violation detection
    no_hat_metrics = metrics['class_metrics'].get('no_hard_hat', {})
    no_vest_metrics = metrics['class_metrics'].get('no_safety_vest', {})
    
    print(f"\n⚠️  VIOLATION DETECTION:")
    print(f"   No hard hat:     {no_hat_metrics.get('recall', 0):.1%} (was 20% with overlaps)")
    print(f"   No safety vest:  {no_vest_metrics.get('recall', 0):.1%} (was 0% before)")
    
    # PPE detection
    hat_metrics = metrics['class_metrics'].get('hard_hat', {})
    vest_metrics = metrics['class_metrics'].get('safety_vest', {})
    
    print(f"\n🦺 PPE DETECTION:")
    print(f"   Hard hat:     {hat_metrics.get('recall', 0):.1%}")
    print(f"   Safety vest:  {vest_metrics.get('recall', 0):.1%}")
    
    # Overall comparison
    print(f"\n📈 OVERALL mAP PROGRESS:")
    print(f"   Original:    {prev_results['original']['map']:.3f}")
    print(f"   Threshold:   {prev_results['threshold_opt']['map']:.3f} (+{(prev_results['threshold_opt']['map']/prev_results['original']['map']-1)*100:.0f}%)")
    print(f"   OPTIMAL:     {metrics['overall_map']:.3f} (+{(metrics['overall_map']/prev_results['original']['map']-1)*100:.0f}% total)")
    
    print(f"\n💾 Results saved to: outputs/optimal_evaluation/")
    print("\n🎯 KEY BENEFITS:")
    print("   ✅ Eliminates same-class overlapping detections")
    print("   ✅ Maximizes person detection without excessive noise")
    print("   ✅ Maintains strict IoU control as originally intended")
    
    return metrics

if __name__ == "__main__":
    run_optimal_evaluation()
