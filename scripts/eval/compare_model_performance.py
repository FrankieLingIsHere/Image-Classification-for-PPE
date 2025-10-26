#!/usr/bin/env python3
"""
Compare performance between baseline Faster R-CNN and Enhanced PPE Detector.

This script analyzes:
1. Missed items by class
2. False positives by class and details
3. Confidence distribution
4. Severe class identification
5. Side-by-side comparison
"""

import json
import glob
from pathlib import Path
from collections import defaultdict
import numpy as np

def load_evaluation_results(pattern):
    """Load the latest evaluation results matching pattern."""
    files = glob.glob(pattern)
    if not files:
        return None
    # Sort by modification time and get latest
    latest = max(files, key=lambda x: Path(x).stat().st_mtime)
    with open(latest) as f:
        return json.load(f)

def analyze_detections(results):
    """Analyze detection details from evaluation results."""
    analysis = {
        'by_class': defaultdict(lambda: {
            'gt_total': 0,
            'detected': 0,
            'missed': 0,
            'missed_details': [],
            'false_positives': 0,
            'fp_details': [],
            'confidence_scores': []
        }),
        'total_gt': 0,
        'total_detected': 0,
        'total_missed': 0,
        'total_fps': 0,
    }
    
    # Process image results
    if 'images' in results:
        for img_data in results['images']:
            # Ground truth boxes
            if 'ground_truth' in img_data:
                for gt in img_data['ground_truth']:
                    class_name = gt.get('class', 'unknown')
                    analysis['by_class'][class_name]['gt_total'] += 1
                    analysis['total_gt'] += 1
            
            # Detections
            if 'detections' in img_data:
                for det in img_data['detections']:
                    class_name = det.get('class', 'unknown')
                    conf = det.get('confidence', 0)
                    is_tp = det.get('tp', False)
                    is_fp = det.get('fp', False)
                    
                    analysis['by_class'][class_name]['confidence_scores'].append(conf)
                    analysis['total_detected'] += 1
                    
                    if is_tp:
                        analysis['by_class'][class_name]['detected'] += 1
                    if is_fp:
                        analysis['by_class'][class_name]['false_positives'] += 1
                        analysis['total_fps'] += 1
                        analysis['by_class'][class_name]['fp_details'].append({
                            'image': img_data.get('image', 'unknown'),
                            'confidence': conf,
                            'bbox': det.get('bbox')
                        })
            
            # Missed detections
            if 'unmatched_gt' in img_data:
                for missed in img_data['unmatched_gt']:
                    class_name = missed.get('class', 'unknown')
                    analysis['by_class'][class_name]['missed'] += 1
                    analysis['total_missed'] += 1
                    analysis['by_class'][class_name]['missed_details'].append({
                        'image': img_data.get('image', 'unknown'),
                        'bbox': missed.get('bbox')
                    })
    
    return analysis

def print_comparison_table(baseline_results, enhanced_results):
    """Print side-by-side comparison."""
    
    baseline_analysis = analyze_detections(baseline_results)
    enhanced_analysis = analyze_detections(enhanced_results)
    
    print("\n" + "="*100)
    print("DETAILED PERFORMANCE COMPARISON: Baseline Faster R-CNN vs Enhanced PPE Detector")
    print("="*100)
    
    # Get all classes
    all_classes = sorted(set(
        list(baseline_analysis['by_class'].keys()) + 
        list(enhanced_analysis['by_class'].keys())
    ))
    
    print("\n" + "CLASS BREAKDOWN".center(100))
    print("-" * 100)
    print(f"{'Class':<20} {'Metric':<15} {'Baseline':<20} {'Enhanced':<20} {'Diff':<15}")
    print("-" * 100)
    
    for cls in all_classes:
        baseline_cls = baseline_analysis['by_class'].get(cls, {})
        enhanced_cls = enhanced_analysis['by_class'].get(cls, {})
        
        baseline_gt = baseline_cls.get('gt_total', 0)
        enhanced_gt = enhanced_cls.get('gt_total', 0)
        
        baseline_detected = baseline_cls.get('detected', 0)
        enhanced_detected = enhanced_cls.get('detected', 0)
        
        baseline_missed = baseline_cls.get('missed', 0)
        enhanced_missed = enhanced_cls.get('missed', 0)
        
        baseline_fp = baseline_cls.get('false_positives', 0)
        enhanced_fp = enhanced_cls.get('false_positives', 0)
        
        # Calculate recall
        baseline_recall = baseline_detected / max(baseline_gt, 1)
        enhanced_recall = enhanced_detected / max(enhanced_gt, 1)
        
        print(f"{cls:<20} {'GT Count':<15} {baseline_gt:<20} {enhanced_gt:<20} {enhanced_gt-baseline_gt:+d}")
        print(f"{'':<20} {'Detected':<15} {baseline_detected:<20} {enhanced_detected:<20} {enhanced_detected-baseline_detected:+d}")
        print(f"{'':<20} {'Recall':<15} {baseline_recall:<20.1%} {enhanced_recall:<20.1%} {enhanced_recall-baseline_recall:+.1%}")
        print(f"{'':<20} {'Missed':<15} {baseline_missed:<20} {enhanced_missed:<20} {enhanced_missed-baseline_missed:+d}")
        print(f"{'':<20} {'False Pos':<15} {baseline_fp:<20} {enhanced_fp:<20} {enhanced_fp-baseline_fp:+d}")
        print("-" * 100)
    
    # Overall metrics
    print("\nOVERALL METRICS".center(100))
    print("-" * 100)
    print(f"{'Metric':<30} {'Baseline':<20} {'Enhanced':<20} {'Change':<15}")
    print("-" * 100)
    
    baseline_total_gt = baseline_analysis['total_gt']
    enhanced_total_gt = enhanced_analysis['total_gt']
    baseline_total_detected = baseline_analysis['total_detected']
    enhanced_total_detected = enhanced_analysis['total_detected']
    baseline_total_missed = baseline_analysis['total_missed']
    enhanced_total_missed = enhanced_analysis['total_missed']
    baseline_total_fps = baseline_analysis['total_fps']
    enhanced_total_fps = enhanced_analysis['total_fps']
    
    print(f"{'Total Ground Truth Boxes':<30} {baseline_total_gt:<20} {enhanced_total_gt:<20} {enhanced_total_gt-baseline_total_gt:+d}")
    print(f"{'Total Detections':<30} {baseline_total_detected:<20} {enhanced_total_detected:<20} {enhanced_total_detected-baseline_total_detected:+d}")
    print(f"{'Total Missed Items':<30} {baseline_total_missed:<20} {enhanced_total_missed:<20} {enhanced_total_missed-baseline_total_missed:+d}")
    print(f"{'Total False Positives':<30} {baseline_total_fps:<20} {enhanced_total_fps:<20} {enhanced_total_fps-baseline_total_fps:+d}")
    
    baseline_recall_overall = baseline_total_detected / max(baseline_total_gt, 1)
    enhanced_recall_overall = enhanced_total_detected / max(enhanced_total_gt, 1)
    baseline_fp_rate = baseline_total_fps / max(baseline_total_detected, 1)
    enhanced_fp_rate = enhanced_total_fps / max(enhanced_total_detected, 1)
    
    print(f"{'Overall Recall':<30} {baseline_recall_overall:<20.1%} {enhanced_recall_overall:<20.1%} {enhanced_recall_overall-baseline_recall_overall:+.1%}")
    print(f"{'False Positive Rate':<30} {baseline_fp_rate:<20.1%} {enhanced_fp_rate:<20.1%} {enhanced_fp_rate-baseline_fp_rate:+.1%}")
    print("-" * 100)
    
    # Identify most severe classes
    print("\nMOST PROBLEMATIC CLASSES".center(100))
    print("-" * 100)
    
    # Sort by missed count (enhanced model)
    missed_by_class = sorted(
        [(cls, enhanced_analysis['by_class'][cls]['missed']) 
         for cls in all_classes],
        key=lambda x: x[1],
        reverse=True
    )
    
    print("\nTop classes with MOST MISSED items (Enhanced Model):")
    for cls, missed_count in missed_by_class[:5]:
        if missed_count > 0:
            gt_count = enhanced_analysis['by_class'][cls]['gt_total']
            print(f"  {cls:<25} Missed: {missed_count:>3} / {gt_count:>3} ({missed_count/max(gt_count,1):.0%})")
    
    # Sort by FP count (enhanced model)
    fp_by_class = sorted(
        [(cls, enhanced_analysis['by_class'][cls]['false_positives']) 
         for cls in all_classes],
        key=lambda x: x[1],
        reverse=True
    )
    
    print("\nTop classes with MOST FALSE POSITIVES (Enhanced Model):")
    for cls, fp_count in fp_by_class[:5]:
        if fp_count > 0:
            det_count = enhanced_analysis['by_class'][cls]['detected'] + fp_count
            print(f"  {cls:<25} FP: {fp_count:>3} / {det_count:>3} ({fp_count/max(det_count,1):.0%})")
    
    # Detailed FP analysis for top 3 FP classes
    print("\nDETAILED FALSE POSITIVE ANALYSIS (Top 3 Classes):")
    print("-" * 100)
    for cls, fp_count in fp_by_class[:3]:
        if fp_count > 0:
            fp_details = enhanced_analysis['by_class'][cls]['fp_details']
            print(f"\n{cls} - {fp_count} false positives:")
            for fp in fp_details[:10]:  # Show first 10
                print(f"  {fp['image']:<30} Confidence: {fp['confidence']:.3f}")
            if len(fp_details) > 10:
                print(f"  ... and {len(fp_details) - 10} more")
    
    # Detailed missed analysis for top 3 missed classes
    print("\nDETAILED MISSED ITEM ANALYSIS (Top 3 Classes):")
    print("-" * 100)
    for cls, missed_count in missed_by_class[:3]:
        if missed_count > 0:
            missed_details = enhanced_analysis['by_class'][cls]['missed_details']
            print(f"\n{cls} - {missed_count} missed items:")
            for missed in missed_details[:10]:  # Show first 10
                print(f"  {missed['image']:<30} BBox: {missed['bbox']}")
            if len(missed_details) > 10:
                print(f"  ... and {len(missed_details) - 10} more")
    
    print("\n" + "="*100)

def main():
    print("Loading evaluation results...")
    
    # Load latest results
    baseline_results = load_evaluation_results(
        'outputs/eval_rcnn_baseline_no_rescorer/evaluation_results_*.json'
    )
    enhanced_results = load_evaluation_results(
        'outputs/evaluation_results/evaluation_results_*.json'
    )
    
    if not baseline_results:
        print("ERROR: Could not find baseline evaluation results")
        print("Available: outputs/eval_rcnn_baseline_no_rescorer/")
        return
    
    if not enhanced_results:
        print("ERROR: Could not find enhanced model evaluation results")
        print("Available: outputs/evaluation_results/")
        return
    
    print(f"Baseline mAP: {baseline_results.get('overall_metrics', {}).get('mAP', 'N/A')}")
    print(f"Enhanced mAP: {enhanced_results.get('overall_metrics', {}).get('mAP', 'N/A')}")
    
    print_comparison_table(baseline_results, enhanced_results)

if __name__ == '__main__':
    main()
