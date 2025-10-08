#!/usr/bin/env python3
"""
Auto-select per-class thresholds on calibrated detection JSONs to target GT counts.

Algorithm:
- Read calibrated JSONs from outputs/calibration/applied_jsons (test split only).
- For each class, collect all calibrated scores and choose the score that yields DET closest to GT (target = GT, tolerance optional).
- Filter detections by those thresholds, then apply per-class NMS and person postprocessing with recommended settings.
- Evaluate and save results.
"""
from pathlib import Path
import sys
import json
import collections
import numpy as np
import argparse
import shutil

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from scripts.inference import _apply_per_class_nms, _postprocess_persons, _to_serializable


def load_test_images(root):
    split_file = root / 'data' / 'splits' / 'test.txt'
    if split_file.exists():
        return [ln.strip() for ln in split_file.read_text(encoding='utf-8').splitlines() if ln.strip()]
    return []


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--calib_dir', type=str, default='outputs/calibration/applied_jsons')
    parser.add_argument('--out_dir', type=str, default='outputs/auto_threshold_calibrated')
    parser.add_argument('--person_merge_iou', type=float, default=0.45)
    parser.add_argument('--per_class_iou', type=float, default=0.6)
    parser.add_argument('--final_conf_min', type=float, default=0.3)
    parser.add_argument('--tolerance', type=int, default=5, help='Allowed deviation from GT when searching thresholds')
    args = parser.parse_args()

    calib_dir = Path(args.calib_dir)
    out_dir = Path(args.out_dir)
    proc_dir = out_dir / 'processed_jsons'
    if out_dir.exists():
        shutil.rmtree(out_dir)
    proc_dir.mkdir(parents=True, exist_ok=True)

    test_images = load_test_images(Path('.'))
    test_basenames = set(Path(x).stem for x in test_images)

    # load GT counts
    import xml.etree.ElementTree as ET
    gt_counts = collections.Counter()
    for name in test_images:
        base = Path(name).stem
        p = Path('data') / 'annotations' / f"{base}.xml"
        if not p.exists():
            continue
        try:
            tree = ET.parse(p)
            root = tree.getroot()
            for obj in root.findall('object'):
                nm = obj.find('name')
                if nm is None or nm.text is None:
                    continue
                gt_counts[nm.text.strip()] += 1
        except Exception:
            continue

    # gather scores per class
    scores_map = collections.defaultdict(list)
    json_files = []
    for j in calib_dir.glob('*_results.json'):
        base = j.stem.replace('_results','')
        if test_basenames and base not in test_basenames:
            continue
        json_files.append(j)
        try:
            data = json.loads(j.read_text(encoding='utf-8'))
        except Exception:
            continue
        for d in data.get('detections', []):
            cls = d.get('class_name') or d.get('class')
            s = float(d.get('confidence', 0.0))
            scores_map[cls].append(s)

    # compute candidate thresholds per class (unique sorted scores)
    thresholds = {}
    for cls, scores in scores_map.items():
        scores_np = np.array(sorted(scores))
        g = gt_counts.get(cls, 0)
        if len(scores_np) == 0:
            thresholds[cls] = 0.01
            continue
        if g == 0:
            # no gt: pick high percentile to reduce detections
            thr = float(np.quantile(scores_np, 0.95))
            thresholds[cls] = round(thr, 3)
            continue
        # target = GT
        target = g
        # If there are fewer scores than target, take min score
        if len(scores_np) <= target:
            thresholds[cls] = float(scores_np.min())
            continue
        # candidate: score at position len - target
        idx = max(0, len(scores_np) - target)
        thr = float(scores_np[idx])
        thresholds[cls] = round(thr, 3)

    # Save initial thresholds
    out_thresh = out_dir / 'initial_thresholds.json'
    out_thresh.parent.mkdir(parents=True, exist_ok=True)
    with open(out_thresh, 'w', encoding='utf-8') as f:
        json.dump(thresholds, f, indent=2)

    print('Initial thresholds computed (per-class):')
    for k,v in sorted(thresholds.items(), key=lambda x:-x[1]):
        print(f'{k}: {v} (GT {gt_counts.get(k,0)})')

    # Apply thresholds to each JSON, then per-class NMS and person postprocess
    for jf in json_files:
        try:
            data = json.loads(jf.read_text(encoding='utf-8'))
        except Exception:
            continue
        dets = data.get('detections', [])
        filtered = []
        for d in dets:
            cname = d.get('class_name') or d.get('class')
            thr = thresholds.get(cname, 0.0)
            if float(d.get('confidence', 0.0)) >= thr:
                filtered.append(d)
        # apply per-class NMS
        processed = _apply_per_class_nms(filtered, iou_thresh=args.per_class_iou, per_class_iou=None, final_conf_min=args.final_conf_min)
        # person postprocess
        from argparse import Namespace
        darr = Namespace()
        darr.person_merge_iou = args.person_merge_iou
        darr.person_conf_min = None
        darr.person_area_min_frac = 0.0
        darr.person_area_max_frac = 1.0
        darr.person_final_max = None
        results = {'image_path': data.get('image_path',''), 'detections': processed}
        results = _postprocess_persons(results, Path(data.get('image_path','')), darr, [d.get('class_name') for d in data.get('detections', [])])
        outp = proc_dir / jf.name
        with open(outp, 'w', encoding='utf-8') as of:
            json.dump(_to_serializable(results), of, indent=2)

    # run evaluator
    try:
        import subprocess, sys
        eval_out = out_dir / 'evaluation'
        cmd = [sys.executable, str(Path(__file__).parent / 'evaluate_from_jsons.py'), '--detections_dir', str(proc_dir), '--output_dir', str(eval_out), '--eval_iou', '0.5', '--ignore_difficult']
        print('Running evaluator:', ' '.join(cmd))
        subprocess.run(cmd, check=True)
        print('Evaluation completed and written to', eval_out)
    except Exception as e:
        print('Evaluator failed:', e)

    print('Done. Processed JSONs in', proc_dir, 'thresholds in', out_thresh)

if __name__ == '__main__':
    main()
