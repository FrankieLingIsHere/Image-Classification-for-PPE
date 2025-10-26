#!/usr/bin/env python3
"""
Detailed analysis of model performance:
- Missed items by class
- False positives by class with details
- Confidence distribution
- Most severe classes
- Comparison with baseline
"""

import json
import glob
from pathlib import Path
from collections import defaultdict
import numpy as np

def load_latest_results(pattern):
    """Load the latest evaluation results matching pattern."""
    files = glob.glob(pattern)
    if not files:
        return None
    latest = max(files, key=lambda x: Path(x).stat().st_mtime)
    print(f"  Loaded: {Path(latest).name}")
    with open(latest) as f:
        return json.load(f)

def analyze_model_performance(results, model_name):
    """Analyze model performance from evaluation results."""
    
    print(f"\n{'='*100}")
    print(f"{model_name.upper()} - DETAILED ANALYSIS")
    print(f"{'='*100}")
    
    # Overall metrics
    summary = results.get('summary', {})
    print(f"\nOVERALL METRICS:")
    print(f"  mAP: {summary.get('mAP', 'N/A'):.3f}" if isinstance(summary.get('mAP'), (int, float)) else f"  mAP: {summary.get('mAP', 'N/A')}")
    print(f"  Total Images: {summary.get('total_images', 'N/A')}")
    
    # Per-class metrics
    print(f"\n{'CLASS-WISE PERFORMANCE':^100}")
    print(f"{'-'*100}")
    print(f"{'Class':<25} {'GT':<8} {'Detected':<10} {'Recall':<10} {'Missed':<10} {'FP':<10} {'AP':<10}")
    print(f"{'-'*100}")
    
    per_class = results.get('per_class_metrics', {})
    
    missed_by_class = {}
    fp_by_class = {}
    
    for class_name in sorted(per_class.keys()):
        metrics = per_class[class_name]
        
        gt_count = metrics.get('gt_count', 0)
        detected = metrics.get('detected', 0)
        missed = metrics.get('missed', 0)
        fp = metrics.get('false_positives', 0)
        recall = metrics.get('recall', 0)
        ap = metrics.get('ap', 0)
        
        missed_by_class[class_name] = missed
        fp_by_class[class_name] = fp
        
        print(f"{class_name:<25} {gt_count:<8} {detected:<10} {recall:<10.1%} {missed:<10} {fp:<10} {ap:<10.3f}")
    
    print(f"{'-'*100}")
    
    # Problem cases
    print(f"\n{'PROBLEM ANALYSIS':^100}")
    print(f"{'-'*100}")
    
    problem_cases = results.get('problem_cases', {})
    
    missed_workers = problem_cases.get('Missed workers', [])
    false_positives = problem_cases.get('False positives', [])
    missed_violations = problem_cases.get('Missed violations', [])
    
    print(f"\nMISSED WORKERS: {len(missed_workers)}")
    for case in missed_workers[:5]:
        print(f"  {case}")
    
    print(f"\nFALSE POSITIVES: {len(false_positives)}")
    fp_summary = defaultdict(int)
    fp_details_by_class = defaultdict(list)
    for case in false_positives:
        class_name = case.get('class', 'unknown')
        fp_summary[class_name] += 1
        fp_details_by_class[class_name].append(case)
    
    for class_name in sorted(fp_summary.keys(), key=lambda x: fp_summary[x], reverse=True):
        count = fp_summary[class_name]
        print(f"  {class_name}: {count} instances")
        # Show first 3 examples
        for detail in fp_details_by_class[class_name][:3]:
            img = detail.get('image', 'unknown')
            gt_count = detail.get('gt_count', 0)
            det_count = detail.get('det_count', 0)
            print(f"    - {img}: {gt_count} GT, {det_count} detections")
    
    print(f"\nMISSED VIOLATIONS: {len(missed_violations)}")
    missed_viol_summary = defaultdict(int)
    missed_viol_details = defaultdict(list)
    for case in missed_violations:
        class_name = case.get('class', 'unknown')
        missed_viol_summary[class_name] += 1
        missed_viol_details[class_name].append(case)
    
    for class_name in sorted(missed_viol_summary.keys(), key=lambda x: missed_viol_summary[x], reverse=True):
        count = missed_viol_summary[class_name]
        print(f"  {class_name}: {count} instances")
        # Show first 3 examples
        for detail in missed_viol_details[class_name][:3]:
            img = detail.get('image', 'unknown')
            missed_count = detail.get('missed_count', 0)
            print(f"    - {img}: {missed_count} missed")
    
    # Most severe classes
    print(f"\n{'MOST SEVERE CLASSES':^100}")
    print(f"{'-'*100}")
    
    # Sort by missed count
    top_missed = sorted(missed_by_class.items(), key=lambda x: x[1], reverse=True)[:5]
    print("\nTop classes with MOST MISSED items:")
    for class_name, missed_count in top_missed:
        if missed_count > 0:
            gt = per_class[class_name].get('gt_count', 0)
            print(f"  {class_name:<25} Missed: {missed_count:>3} / {gt:>3} ({missed_count/max(gt,1):.0%} miss rate)")
    
    # Sort by FP count
    top_fp = sorted(fp_by_class.items(), key=lambda x: x[1], reverse=True)[:5]
    print("\nTop classes with MOST FALSE POSITIVES:")
    for class_name, fp_count in top_fp:
        if fp_count > 0:
            detected = per_class[class_name].get('detected', 0)
            total_dets = detected + fp_count
            print(f"  {class_name:<25} FP: {fp_count:>3} / {total_dets:>3} ({fp_count/max(total_dets,1):.0%} FP rate)")
    
    print(f"{'-'*100}")
    
    return {
        'summary': summary,
        'per_class': per_class,
        'problem_cases': problem_cases,
        'top_missed': top_missed,
        'top_fp': top_fp
    }

def compare_models(baseline_data, enhanced_data):
    """Compare baseline and enhanced models."""
    
    print(f"\n\n{'='*100}")
    print(f"{'BASELINE VS ENHANCED COMPARISON':^100}")
    print(f"{'='*100}")
    
    baseline_summary = baseline_data['summary']
    enhanced_summary = enhanced_data['summary']
    
    baseline_map = baseline_summary.get('mAP', 0)
    enhanced_map = enhanced_summary.get('mAP', 0)
    map_diff = enhanced_map - baseline_map
    map_pct = (map_diff / max(baseline_map, 0.001)) * 100 if baseline_map > 0 else 0
    
    print(f"\n{'OVERALL mAP CHANGE':^100}")
    print(f"{'-'*100}")
    print(f"  Baseline mAP: {baseline_map:.4f}")
    print(f"  Enhanced mAP: {enhanced_map:.4f}")
    print(f"  Difference:   {map_diff:+.4f} ({map_pct:+.1f}%)")
    print(f"{'-'*100}")
    
    print(f"\n{'CLASS-WISE COMPARISON':^100}")
    print(f"{'-'*100}")
    print(f"{'Class':<25} {'Baseline AP':<15} {'Enhanced AP':<15} {'Change':<15}")
    print(f"{'-'*100}")
    
    baseline_per_class = baseline_data['per_class']
    enhanced_per_class = enhanced_data['per_class']
    
    all_classes = set(baseline_per_class.keys()) | set(enhanced_per_class.keys())
    
    for class_name in sorted(all_classes):
        baseline_ap = baseline_per_class.get(class_name, {}).get('ap', 0)
        enhanced_ap = enhanced_per_class.get(class_name, {}).get('ap', 0)
        ap_diff = enhanced_ap - baseline_ap
        ap_pct = (ap_diff / max(baseline_ap, 0.001)) * 100 if baseline_ap > 0 else (100 if enhanced_ap > 0 else 0)
        
        symbol = "↑" if ap_diff > 0 else ("↓" if ap_diff < 0 else "→")
        print(f"{class_name:<25} {baseline_ap:<15.4f} {enhanced_ap:<15.4f} {symbol} {ap_diff:+.4f} ({ap_pct:+.0f}%)")
    
    print(f"{'-'*100}")
    
    print(f"\n{'IMPROVEMENTS':^100}")
    print(f"{'-'*100}")
    
    # Count improvements
    improved = 0
    degraded = 0
    for class_name in all_classes:
        baseline_ap = baseline_per_class.get(class_name, {}).get('ap', 0)
        enhanced_ap = enhanced_per_class.get(class_name, {}).get('ap', 0)
        if enhanced_ap > baseline_ap:
            improved += 1
        elif enhanced_ap < baseline_ap:
            degraded += 1
    
    print(f"  Classes improved: {improved}")
    print(f"  Classes degraded: {degraded}")
    print(f"  Classes unchanged: {len(all_classes) - improved - degraded}")
    
    print(f"{'-'*100}")

def main():
    print("Loading evaluation results...\n")
    
    baseline_results = load_latest_results('outputs/eval_rcnn_baseline_no_rescorer/evaluation_results_*.json')
    enhanced_results = load_latest_results('outputs/evaluation_results/evaluation_results_*.json')
    
    if not baseline_results:
        print("ERROR: Could not find baseline results")
        return
    
    if not enhanced_results:
        print("ERROR: Could not find enhanced results")
        return
    
    baseline_data = analyze_model_performance(baseline_results, "Baseline Faster R-CNN")
    enhanced_data = analyze_model_performance(enhanced_results, "Enhanced PPE Detector")
    
    compare_models(baseline_data, enhanced_data)
    
    print(f"\n{'='*100}\n")

if __name__ == '__main__':
    main()
