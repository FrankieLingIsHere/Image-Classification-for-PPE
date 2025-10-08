#!/usr/bin/env python3
"""
Automatically adjust per-class confidence thresholds based on det/gt ratios.

Algorithm (simple heuristic):
- Compute GT count per class from VOC XMLs in data/annotations
- Compute detection count per class from processed JSONs (use outputs/rcnn_combined_eval/processed_jsons if present, else outputs/rcnn_baseline_adamw)
- For classes where det_count / gt_count > 2.0, raise threshold by 0.1
- For classes where det_count / gt_count between 1.2 and 2.0, raise threshold by 0.05
- For classes where det_count / gt_count < 0.5, lower threshold by 0.05 (to recover recall)

The script writes a recommended thresholds CSV and runs the evaluator with the new thresholds.
"""
import xml.etree.ElementTree as ET
from pathlib import Path
import json
import collections
import shutil
import sys

ROOT = Path(__file__).parent.parent
ANN_DIR = ROOT / 'data' / 'annotations'
PROC_DETS = ROOT / 'outputs' / 'rcnn_combined_eval' / 'processed_jsons'
FALLBACK_DETS = ROOT / 'outputs' / 'rcnn_baseline_adamw'
OUT_DIR = ROOT / 'outputs' / 'auto_thresholds'
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_gt_counts():
    counts = collections.Counter()
    for p in ANN_DIR.glob('*.xml'):
        try:
            tree = ET.parse(p)
            root = tree.getroot()
            for obj in root.findall('object'):
                name = obj.find('name').text
                counts[name] += 1
        except Exception:
            continue
    return counts


def load_det_counts(det_dir):
    counts = collections.Counter()
    if not det_dir.exists():
        return counts
    for j in det_dir.glob('*_results.json'):
        try:
            with open(j, 'r', encoding='utf-8') as f:
                data = json.load(f)
            for d in data.get('detections', []):
                counts[d.get('class_name')] += 1
        except Exception:
            continue
    return counts


def propose_thresholds(current_map, gt_counts, det_counts):
    # current_map: existing thresholds dict
    res = dict(current_map)
    for cls, gt in gt_counts.items():
        det = det_counts.get(cls, 0)
        ratio = (det / gt) if gt > 0 else float('inf')
        base = res.get(cls, 0.12)
        if ratio > 2.0:
            base = min(0.95, base + 0.10)
        elif ratio > 1.2:
            base = min(0.95, base + 0.05)
        elif ratio < 0.5:
            base = max(0.01, base - 0.05)
        res[cls] = round(base, 3)
    return res


def main():
    gt = load_gt_counts()
    dets = load_det_counts(PROC_DETS)
    if not dets:
        dets = load_det_counts(FALLBACK_DETS)

    # load current thresholds from eval script if present
    cur = {
        'hard_hat': 0.25, 'safety_vest': 0.25, 'no_hard_hat': 0.35,
        'safety_gloves': 0.45, 'safety_boots': 0.40, 'eye_protection': 0.25,
        'no_eye_protection': 0.20, 'no_safety_vest': 0.30, 'no_safety_gloves': 0.35,
        'person': 0.20
    }

    proposed = propose_thresholds(cur, gt, dets)

    out_csv = OUT_DIR / 'proposed_thresholds.csv'
    with open(out_csv, 'w', encoding='utf-8') as f:
        f.write('class,current,gt_count,det_count,proposed\n')
        for cls in sorted(set(list(gt.keys()) + list(dets.keys()) + list(cur.keys()))):
            f.write(f"{cls},{cur.get(cls)},{gt.get(cls,0)},{dets.get(cls,0)},{proposed.get(cls)}\n")

    print('Wrote proposed thresholds to', out_csv)

    # apply proposed thresholds and evaluate using the existing functions
    # ensure repo root is on sys.path for local imports
    sys.path.insert(0, str(ROOT))
    from scripts.eval_rcnn_combined_config import apply_and_write_all, evaluate

    DETS_DIR = ROOT / 'outputs' / 'rcnn_baseline_adamw'
    EVAL_OUT = ROOT / 'outputs' / 'auto_thresholds_eval'
    if EVAL_OUT.exists():
        shutil.rmtree(EVAL_OUT)
    EVAL_OUT.mkdir(parents=True, exist_ok=True)

    proc_dir = EVAL_OUT / 'processed_jsons'
    print('Applying proposed thresholds and writing processed JSONs to', proc_dir)
    apply_and_write_all(DETS_DIR, proc_dir, proposed, {'person_conf_min': 0.20, 'person_final_max': 4, 'person_merge_iou': 0.45, 'person_overlap_threshold': 0.0, 'final_conf_min': 0.5})

    eval_out = EVAL_OUT / 'evaluation'
    if eval_out.exists():
        shutil.rmtree(eval_out)
    eval_out.mkdir(parents=True, exist_ok=True)
    print('Running evaluation...')
    evaluate(proc_dir, eval_out)

if __name__ == '__main__':
    main()
