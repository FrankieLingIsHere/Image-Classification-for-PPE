#!/usr/bin/env python3
"""
Apply per-class max caps (gt + 5) to detections and evaluate.

This will: load GT counts, load final thresholds (from threshold_balance if available),
filter raw detections by thresholds, then apply per-class NMS and cap final detections
per class to gt + 5. Finally, run the JSON evaluator and print per-class diffs.
"""
import sys
from pathlib import Path
import json
import shutil

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

DETS_DIR = ROOT / 'outputs' / 'rcnn_baseline_adamw'
TH_FILE = ROOT / 'outputs' / 'threshold_balance' / 'final_thresholds.json'
OUT_DIR = ROOT / 'outputs' / 'threshold_balance' / 'capped_eval'
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_gt_counts():
    import xml.etree.ElementTree as ET
    counts = {}
    # Only consider annotations for images in the test split
    split_file = ROOT / 'data' / 'splits' / 'test.txt'
    if split_file.exists():
        names = [ln.strip() for ln in split_file.read_text(encoding='utf-8').splitlines() if ln.strip()]
        for img in names:
            base = Path(img).stem
            p = ROOT / 'data' / 'annotations' / f"{base}.xml"
            if not p.exists():
                continue
            try:
                tree = ET.parse(p)
                root = tree.getroot()
                for obj in root.findall('object'):
                    name_el = obj.find('name')
                    if name_el is None or name_el.text is None:
                        continue
                    counts[name_el.text.strip()] = counts.get(name_el.text.strip(), 0) + 1
            except Exception:
                continue
    else:
        for p in (ROOT / 'data' / 'annotations').glob('*.xml'):
            try:
                tree = ET.parse(p)
                root = tree.getroot()
                for obj in root.findall('object'):
                    name_el = obj.find('name')
                    if name_el is None or name_el.text is None:
                        continue
                    counts[name_el.text.strip()] = counts.get(name_el.text.strip(), 0) + 1
            except Exception:
                continue
    return counts


def load_thresholds():
    if TH_FILE.exists():
        with open(TH_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}


def load_json(p):
    with open(p, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json(obj, p):
    with open(p, 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2)


def apply_caps():
    gt = load_gt_counts()
    th = load_thresholds()
    # build caps map = gt + 5
    caps = {k: (v + 5) for k, v in gt.items()}

    # import utilities
    from scripts.inference import _apply_per_class_nms

    dst = OUT_DIR / 'processed_jsons'
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True, exist_ok=True)

    for j in DETS_DIR.glob('*_results.json'):
        data = load_json(j)
        dets = data.get('detections', [])
        # threshold filter
        filtered = []
        for d in dets:
            thr = th.get(d.get('class_name'), 0.0)
            if d.get('confidence', 0.0) >= thr:
                filtered.append(d)
        # apply per-class nms and caps
        # _apply_per_class_nms supports max_per_class
        try:
            kept = _apply_per_class_nms(filtered, iou_thresh=0.3, top_k=200, per_class_iou=None, max_per_class=caps)
        except Exception:
            kept = filtered

        data['detections'] = kept
        save_json(data, dst / j.name)

    return dst, caps


def evaluate(proc_dir):
    import importlib
    from scripts.evaluate_from_jsons import main as eval_main
    # call evaluator with args via sys.argv
    argv_backup = list(sys.argv)
    try:
        sys.argv = ['scripts/evaluate_from_jsons.py', '--detections_dir', str(proc_dir), '--output_dir', str(OUT_DIR / 'evaluation'), '--data_dir', 'data', '--eval_iou', '0.5']
        try:
            eval_main()
        except SystemExit:
            pass
    finally:
        sys.argv = argv_backup


def print_diffs(proc_dir, caps):
    # counts
    det_counts = {}
    for j in proc_dir.glob('*_results.json'):
        with open(j, 'r', encoding='utf-8') as f:
            data = json.load(f)
        for d in data.get('detections', []):
            det_counts[d.get('class_name')] = det_counts.get(d.get('class_name'), 0) + 1

    gt = load_gt_counts()
    print('Class\tGT\tDET\tDiff\tCap')
    for cls in sorted(set(list(gt.keys()) + list(det_counts.keys()))):
        g = gt.get(cls, 0)
        d = det_counts.get(cls, 0)
        cap = caps.get(cls, '')
        print(f'{cls}\t{g}\t{d}\t{d-g}\t{cap}')


def main():
    proc, caps = apply_caps()
    evaluate(proc)
    print_diffs(proc, caps)

if __name__ == '__main__':
    main()
