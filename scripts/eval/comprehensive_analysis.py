#!/usr/bin/env python3
"""
Comprehensive performance analysis showing:
1. Missed items by class with details
2. False positives by class with details  
3. Confidence analysis
4. Most severe classes
5. Comparison with baseline
"""

import json
import glob
from pathlib import Path
from collections import defaultdict

def load_latest_results(pattern):
    """Load the latest evaluation results matching pattern."""
    files = glob.glob(pattern)
    if not files:
        return None
    latest = max(files, key=lambda x: Path(x).stat().st_mtime)
    print(f"  Loaded: {Path(latest).name}")
    with open(latest) as f:
        return json.load(f)

def analyze_and_print(results, model_name):
    """Analyze and print performance details."""
    
    print(f"\n{'='*120}")
    print(f"{model_name:^120}")
    print(f"{'='*120}")
    
    summary = results.get('summary', {})
    per_class = results.get('per_class_metrics', {})
    problem_cases = results.get('problem_cases', {})
    detection_results = results.get('detection_results', {})
    
    map_score = summary.get('map_score', 0)
    
    print(f"\nOVERALL METRICS:")
    print(f"  mAP: {map_score:.4f}")
    print(f"  Confidence Threshold: {summary.get('conf_threshold', 0.5)}")
    print(f"  IoU Threshold: {summary.get('iou_threshold', 0.45)}")
    
    # Class-wise metrics
    print(f"\n{'CLASS-WISE METRICS':^120}")
    print(f"{'-'*120}")
    print(f"{'Class':<25} {'GT':<8} {'TP':<8} {'FP':<8} {'FN':<8} {'Recall':<10} {'Precision':<12} {'AP':<10}")
    print(f"{'-'*120}")
    
    missed_by_class = {}
    fp_by_class = {}
    
    for class_name in sorted(per_class.keys()):
        metrics = per_class[class_name]
        gt_count = metrics.get('gt_count', 0)
        tp = metrics.get('tp', 0)
        fp = metrics.get('fp', 0)
        fn = metrics.get('fn', 0)
        recall = metrics.get('recall', 0)
        precision = metrics.get('precision', 0)
        ap = metrics.get('ap', 0)
        
        missed_by_class[class_name] = fn
        fp_by_class[class_name] = fp
        
        print(f"{class_name:<25} {gt_count:<8} {tp:<8} {fp:<8} {fn:<8} {recall:<10.1%} {precision:<12.1%} {ap:<10.4f}")
    
    print(f"{'-'*120}")
    
    # Most problematic
    print(f"\n{'MOST PROBLEMATIC CLASSES (Enhanced Model)':^120}")
    print(f"{'-'*120}")
    
    # Top missed
    top_missed = sorted(missed_by_class.items(), key=lambda x: x[1], reverse=True)[:5]
    print("\n[MISSED ITEMS] TOP CLASSES WITH MOST MISSED ITEMS:")
    for class_name, missed_count in top_missed:
        if missed_count > 0:
            gt_count = per_class[class_name].get('gt_count', 0)
            detected = per_class[class_name].get('tp', 0)
            miss_rate = missed_count / max(gt_count, 1)
            print(f"  {class_name:<25} MISSED: {missed_count:>3}/{gt_count:>3} ({miss_rate:>5.0%} miss rate)")
    
    # Top FP
    top_fp = sorted(fp_by_class.items(), key=lambda x: x[1], reverse=True)[:5]
    print("\n[FALSE POS] TOP CLASSES WITH MOST FALSE POSITIVES:")
    for class_name, fp_count in top_fp:
        if fp_count > 0:
            tp = per_class[class_name].get('tp', 0)
            total_dets = tp + fp_count
            fp_rate = fp_count / max(total_dets, 1)
            print(f"  {class_name:<25} FP: {fp_count:>3}/{total_dets:>3} ({fp_rate:>5.0%} FP rate)")
    
    # Detailed problem analysis
    print(f"\n{'DETAILED PROBLEM ANALYSIS':^120}")
    print(f"{'-'*120}")
    
    # False positives
    false_positives = problem_cases.get('False positives', [])
    print(f"\nFALSE POSITIVES: {len(false_positives)} instances")
    fp_by_img = defaultdict(list)
    for fp in false_positives:
        img = fp.get('image', 'unknown')
        fp_by_img[img].append(fp)
    
    # Show worst FP cases
    worst_fp_imgs = sorted([(img, len(fps)) for img, fps in fp_by_img.items()], key=lambda x: x[1], reverse=True)[:5]
    for img, fp_count in worst_fp_imgs:
        print(f"  IMAGE: {img:<35} {fp_count} false positives")
        for fp in fp_by_img[img]:
            class_name = fp.get('class', 'unknown')
            gt_count = fp.get('gt_count', 0)
            det_count = fp.get('det_count', 0)
            print(f"     - {class_name}: GT={gt_count}, Detections={det_count}")
    
    # Missed items
    missed_violations = problem_cases.get('Missed violations', [])
    print(f"\nMISSED VIOLATIONS/ITEMS: {len(missed_violations)} instances")
    missed_by_img = defaultdict(list)
    for missed in missed_violations:
        img = missed.get('image', 'unknown')
        missed_by_img[img].append(missed)
    
    # Show worst missed cases
    worst_missed_imgs = sorted([(img, len(miss)) for img, miss in missed_by_img.items()], key=lambda x: x[1], reverse=True)[:5]
    for img, missed_count in worst_missed_imgs:
        print(f"  IMAGE: {img:<35} {missed_count} missed items")
        for missed in missed_by_img[img]:
            class_name = missed.get('class', 'unknown')
            missed_cnt = missed.get('missed_count', 0)
            print(f"     - {class_name}: {missed_cnt} missed")
    
    # Confidence distribution
    print(f"\n{'CONFIDENCE DISTRIBUTION':^120}")
    print(f"{'-'*120}")
    conf_by_class = defaultdict(list)
    
    # detection_results can be dict or list
    if isinstance(detection_results, dict):
        det_items = detection_results.items()
    elif isinstance(detection_results, list):
        det_items = enumerate(detection_results)
    else:
        det_items = []
    
    for img_id, det_info in det_items:
        for det in det_info.get('detections', []):
            class_name = det.get('class', 'unknown')
            conf = det.get('confidence', 0)
            conf_by_class[class_name].append(conf)
    
    for class_name in sorted(conf_by_class.keys()):
        confs = conf_by_class[class_name]
        if confs:
            avg_conf = sum(confs) / len(confs)
            min_conf = min(confs)
            max_conf = max(confs)
            print(f"  {class_name:<25} Avg: {avg_conf:.3f}, Min: {min_conf:.3f}, Max: {max_conf:.3f}, Count: {len(confs)}")
    
    print(f"{'-'*120}\n")

def compare_models(baseline_results, enhanced_results):
    """Compare baseline vs enhanced."""
    
    print(f"\n{'='*120}")
    print(f"{'BASELINE vs ENHANCED COMPARISON':^120}")
    print(f"{'='*120}")
    
    baseline_map = baseline_results.get('summary', {}).get('map_score', 0)
    enhanced_map = enhanced_results.get('summary', {}).get('map_score', 0)
    
    map_change = enhanced_map - baseline_map
    map_pct = (map_change / max(baseline_map, 0.001)) * 100 if baseline_map > 0 else 0
    
    print(f"\nOVERALL mAP:")
    print(f"  Baseline: {baseline_map:.4f}")
    print(f"  Enhanced: {enhanced_map:.4f}")
    print(f"  Change: {map_change:+.4f} ({map_pct:+.1f}%)")
    
    baseline_per_class = baseline_results.get('per_class_metrics', {})
    enhanced_per_class = enhanced_results.get('per_class_metrics', {})
    
    print(f"\n{'CLASS-WISE COMPARISON':^120}")
    print(f"{'-'*120}")
    print(f"{'Class':<25} {'Baseline AP':<15} {'Enhanced AP':<15} {'Change':<20} {'Result':<15}")
    print(f"{'-'*120}")
    
    improved = 0
    degraded = 0
    
    for class_name in sorted(set(baseline_per_class.keys()) | set(enhanced_per_class.keys())):
        baseline_ap = baseline_per_class.get(class_name, {}).get('ap', 0)
        enhanced_ap = enhanced_per_class.get(class_name, {}).get('ap', 0)
        
        ap_change = enhanced_ap - baseline_ap
        ap_pct = (ap_change / max(baseline_ap, 0.001)) * 100 if baseline_ap > 0 else (100 if enhanced_ap > 0 else 0)
        
        if enhanced_ap > baseline_ap:
            symbol = "UP IMPROVED"
            improved += 1
        elif enhanced_ap < baseline_ap:
            symbol = "DOWN DEGRADED"
            degraded += 1
        else:
            symbol = "SAME"
        
        change_str = f"{ap_change:+.4f} ({ap_pct:+.0f}%)"
        print(f"{class_name:<25} {baseline_ap:<15.4f} {enhanced_ap:<15.4f} {change_str:<20} {symbol:<15}")
    
    print(f"{'-'*120}")
    print(f"\nSummary: {improved} improved, {degraded} degraded, {len(baseline_per_class) - improved - degraded} unchanged")
    print(f"{'-'*120}\n")

def main():
    print("Loading evaluation results...\n")
    
    baseline_results = load_latest_results('outputs/eval_rcnn_baseline_no_rescorer/evaluation_results_*.json')
    enhanced_results = load_latest_results('outputs/evaluation_results/evaluation_results_*.json')
    
    if not baseline_results or not enhanced_results:
        print("ERROR: Could not find both evaluation results")
        return
    
    analyze_and_print(baseline_results, "BASELINE FASTER R-CNN (ResNet50+FPN)")
    analyze_and_print(enhanced_results, "ENHANCED PPE DETECTOR (4-Stage Training)")
    
    compare_models(baseline_results, enhanced_results)

if __name__ == '__main__':
    main()
