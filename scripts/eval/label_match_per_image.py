"""Compute per-image label-only match percentage.

Usage:
  python label_match_per_image.py --results <evaluation_results.json> --out <csv_output_optional>

Output CSV columns:
 image, gt_count, detected_count, matched_count, recall_pct, gt_labels, detected_labels

This ignores bbox IoU and uses class-specific confidence thresholds present in the evaluation JSON under
results['effective_config']['class_conf_thresholds'] (falls back to results['summary']['conf_threshold']).
"""
import json
import argparse
from pathlib import Path
import csv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', required=True)
    parser.add_argument('--out', required=False)
    args = parser.parse_args()

    res_path = Path(args.results)
    if not res_path.exists():
        raise FileNotFoundError(f"Results file not found: {res_path}")

    with open(res_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    eff = data.get('effective_config', {}) or data.get('results', {}).get('effective_config', {})
    class_thresholds = eff.get('class_conf_thresholds', {}) if eff else {}
    default_conf = eff.get('conf_threshold', data.get('summary', {}).get('conf_threshold', 0.3))

    out_path = Path(args.out) if args.out else res_path.parent / f"label_match_per_image_{res_path.stem}.csv"

    rows = []
    totals = {'images': 0, 'gt_labels': 0, 'matched': 0}

    for item in data.get('detection_results', []):
        image = item.get('image')
        gts = item.get('ground_truth', [])
        dets = item.get('detections', [])

        gt_labels = []
        for g in gts:
            cls = g.get('class')
            if cls:
                gt_labels.append(cls)
        gt_set = set(gt_labels)

        detected_set = set()
        for d in dets:
            cls = d.get('class')
            conf = d.get('confidence', 0.0)
            thr = class_thresholds.get(cls, default_conf)
            try:
                thr = float(thr)
            except Exception:
                thr = default_conf
            if conf >= thr:
                detected_set.add(cls)

        matched = gt_set.intersection(detected_set)
        gt_count = len(gt_set)
        matched_count = len(matched)
        detected_count = len(detected_set)
        recall_pct = (matched_count / gt_count * 100.0) if gt_count > 0 else None

        rows.append({
            'image': image,
            'gt_count': gt_count,
            'detected_count': detected_count,
            'matched_count': matched_count,
            'recall_pct': f"{recall_pct:.2f}" if recall_pct is not None else '',
            'gt_labels': ';'.join(sorted(gt_set)),
            'detected_labels': ';'.join(sorted(detected_set))
        })

        totals['images'] += 1
        totals['gt_labels'] += gt_count
        totals['matched'] += matched_count

    # write CSV
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w', newline='', encoding='utf-8') as cf:
        fieldnames = ['image','gt_count','detected_count','matched_count','recall_pct','gt_labels','detected_labels']
        writer = csv.DictWriter(cf, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    overall_recall = (totals['matched'] / totals['gt_labels'] * 100.0) if totals['gt_labels'] > 0 else 0.0
    print(f"Wrote per-image label match CSV: {out_path}")
    print(f"Images: {totals['images']}, GT labels total: {totals['gt_labels']}, Matched total: {totals['matched']}, Overall recall: {overall_recall:.2f}%")

if __name__ == '__main__':
    main()
