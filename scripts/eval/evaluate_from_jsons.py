#!/usr/bin/env python3
"""Evaluate precomputed detection JSONs (e.g. from R-CNN) against ground truth.

This is a copy moved to `scripts/eval/` and adjusted so imports work from the new location.
"""
import os
import json
import argparse
from pathlib import Path
from collections import defaultdict
from PIL import Image
import xml.etree.ElementTree as ET
import numpy as np

# Ensure repo root importability
import sys
sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.utils.utils import calculate_iou


PPE_CLASSES = [
    'background', 'person', 'hard_hat', 'safety_vest', 'safety_gloves',
    'safety_boots', 'eye_protection', 'no_hard_hat', 'no_safety_vest',
    'no_safety_gloves', 'no_safety_boots', 'no_eye_protection'
]


def parse_xml(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    size = root.find('size')
    width = int(size.find('width').text)
    height = int(size.find('height').text)
    anns = []
    for obj in root.findall('object'):
        cls = obj.find('name').text
        difficult = int(obj.find('difficult').text) if obj.find('difficult') is not None else 0
        bbox = obj.find('bndbox')
        x1 = float(int(bbox.find('xmin').text) - 1)
        y1 = float(int(bbox.find('ymin').text) - 1)
        x2 = float(int(bbox.find('xmax').text) - 1)
        y2 = float(int(bbox.find('ymax').text) - 1)
        anns.append({'class': cls, 'bbox': [x1, y1, x2, y2], 'difficult': difficult})
    return width, height, anns


def load_detections(json_path, width, height):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    dets = []
    for d in data.get('detections', []):
        bx = d.get('bbox', [])
        if len(bx) == 4:
            if 0.0 <= bx[0] <= 1.0 and 0.0 <= bx[2] <= 1.0:
                x1 = bx[0] * width
                y1 = bx[1] * height
                x2 = bx[2] * width
                y2 = bx[3] * height
            else:
                x1, y1, x2, y2 = bx
            dets.append({'class': d.get('class_name', d.get('class', 'unknown')), 'bbox': [x1, y1, x2, y2], 'confidence': d.get('confidence', 0.0)})
    return dets


def compute_ap(gt_boxes, det_boxes, eval_iou=0.5):
    if len(gt_boxes) == 0:
        return 0.0, 0, len(det_boxes), 0
    if len(det_boxes) == 0:
        return 0.0, 0, 0, len(gt_boxes)

    det_sorted = sorted(det_boxes, key=lambda x: x['confidence'], reverse=True)
    tp = np.zeros(len(det_sorted))
    fp = np.zeros(len(det_sorted))
    matched = set()
    for i, det in enumerate(det_sorted):
        best_iou = 0.0
        best_j = -1
        for j, gt in enumerate(gt_boxes):
            if j in matched:
                continue
            iou = calculate_iou(det['bbox'], gt['bbox'])
            if iou > best_iou:
                best_iou = iou
                best_j = j
        if best_iou >= eval_iou:
            tp[i] = 1
            matched.add(best_j)
        else:
            fp[i] = 1

    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)
    precisions = tp_cum / (tp_cum + fp_cum + 1e-8)
    recalls = tp_cum / max(1, len(gt_boxes))

    ap = 0.0
    for r in np.arange(0, 1.1, 0.1):
        p_vals = precisions[recalls >= r]
        p = np.max(p_vals) if len(p_vals) > 0 else 0.0
        ap += p / 11.0

    tp_total = int(tp.sum())
    fp_total = int(fp.sum())
    fn_total = len(gt_boxes) - tp_total
    return ap, tp_total, fp_total, fn_total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--detections_dir', type=str, required=True)
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--output_dir', type=str, default='outputs/eval_rcnn_from_jsons')
    parser.add_argument('--eval_iou', type=float, default=0.5)
    parser.add_argument('--ignore_difficult', action='store_true',
                        help='If set, skip GT objects with difficult=1')
    args = parser.parse_args()

    det_dir = Path(args.detections_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    split_file = Path(args.data_dir) / 'splits' / 'test.txt'
    with open(split_file, 'r') as f:
        images = [ln.strip() for ln in f.readlines() if ln.strip()]

    per_class_metrics = {c: {'ap_list': [], 'tp': 0, 'fp': 0, 'fn': 0, 'gt': 0, 'det': 0} for c in PPE_CLASSES[1:]}

    for img_name in images:
        base = os.path.splitext(img_name)[0]
        ann_path = Path(args.data_dir) / 'annotations' / f'{base}.xml'
        img_path = Path(args.data_dir) / 'images' / img_name
        if not ann_path.exists():
            continue
        width, height, gt_anns = parse_xml(ann_path)
        if args.ignore_difficult:
            gt_anns = [g for g in gt_anns if g.get('difficult', 0) == 0]

        det_json = det_dir / f'{base}_results.json'
        if not det_json.exists():
            dets = []
        else:
            dets = load_detections(det_json, width, height)

        for cls in PPE_CLASSES[1:]:
            gt_boxes = [g for g in gt_anns if g['class'] == cls]
            det_boxes = [d for d in dets if d['class'] == cls]
            ap, tp, fp, fn = compute_ap(gt_boxes, det_boxes, eval_iou=args.eval_iou)
            per_class_metrics[cls]['ap_list'].append(ap)
            per_class_metrics[cls]['tp'] += tp
            per_class_metrics[cls]['fp'] += fp
            per_class_metrics[cls]['fn'] += fn
            per_class_metrics[cls]['gt'] += len(gt_boxes)
            per_class_metrics[cls]['det'] += len(det_boxes)

    results = {}
    ap_list = []
    for cls, data in per_class_metrics.items():
        ap = float(np.mean(data['ap_list'])) if len(data['ap_list']) > 0 else 0.0
        tp = data['tp']
        fp = data['fp']
        fn = data['fn']
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        results[cls] = {'ap': ap, 'precision': precision, 'recall': recall, 'f1': f1, 'gt_count': data['gt'], 'det_count': data['det']}
        ap_list.append(ap)

    map_score = float(np.mean(ap_list)) if len(ap_list) > 0 else 0.0

    summary = {'map_score': map_score, 'per_class_metrics': results}
    with open(out_dir / 'evaluation_summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)

    print(f"R-CNN evaluation (from JSONs)  mAP: {map_score:.3f}")
    for cls, m in results.items():
        print(f"  {cls}: AP={m['ap']:.3f} prec={m['precision']:.3f} rec={m['recall']:.3f} gt={m['gt_count']} det={m['det_count']}")


if __name__ == '__main__':
    main()
