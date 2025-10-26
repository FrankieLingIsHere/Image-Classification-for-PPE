#!/usr/bin/env python3
"""Generate side-by-side visual comparisons (raw vs processed) for top-N images by detection count.

Saves images to outputs/visual_comparisons/{image}_compare.jpg

By default compares raw detections in outputs/rcnn_baseline_adamw against
processed detections in outputs/threshold_balance/capped_eval/processed_jsons if available,
otherwise falls back to outputs/percentile_thresholds/processed_jsons.
"""
from pathlib import Path
import json
import os
import cv2
import numpy as np

ROOT = Path(__file__).parent.parent
RAW_DIR = ROOT / 'outputs' / 'rcnn_baseline_adamw'
CAND_PROCESSED = [ROOT / 'outputs' / 'threshold_balance' / 'capped_eval' / 'processed_jsons',
                   ROOT / 'outputs' / 'percentile_thresholds' / 'processed_jsons',
                   ROOT / 'outputs' / 'rcnn_combined_eval' / 'processed_jsons']
OUT_DIR = ROOT / 'outputs' / 'visual_comparisons'
OUT_DIR.mkdir(parents=True, exist_ok=True)
IMG_DIR = ROOT / 'data' / 'images'
CLASS_FILE = ROOT / 'configs' / 'dataset_class_names.txt'

if CLASS_FILE.exists():
    with open(CLASS_FILE, 'r', encoding='utf-8') as f:
        CLASS_NAMES = [ln.strip() for ln in f.readlines() if ln.strip()]
else:
    CLASS_NAMES = []

# helper to draw detections on image (opencv)
def draw_dets_on_image(image, dets, class_names=None, box_color=(0,255,0)):
    # image: numpy BGR uint8
    h, w = image.shape[:2]
    out = image.copy()
    # color palette
    colors = [(0,255,0),(0,128,255),(255,0,0),(0,255,255),(255,0,255),(128,0,128),(0,128,0),(128,128,0),(0,0,255),(128,0,0)]
    for d in dets:
        bx = d.get('bbox', [])
        if not bx or len(bx) < 4:
            continue
        # detect fractional
        try:
            if max(bx) <= 1.0:
                x1 = int(bx[0] * w)
                y1 = int(bx[1] * h)
                x2 = int(bx[2] * w)
                y2 = int(bx[3] * h)
            else:
                x1, y1, x2, y2 = int(bx[0]), int(bx[1]), int(bx[2]), int(bx[3])
        except Exception:
            continue
        cls = d.get('class_name') or d.get('class') or 'unknown'
        score = d.get('confidence', 0.0)
        col = colors[hash(cls) % len(colors)]
        cv2.rectangle(out, (x1,y1), (x2,y2), col, 2)
        label = f"{cls} {score:.2f}"
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)[0]
        cv2.rectangle(out, (x1, y1 - t_size[1] - 6), (x1 + t_size[0] + 6, y1), col, -1)
        cv2.putText(out, label, (x1+3, y1-4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1, cv2.LINE_AA)
    return out

# find processed dir that exists
PROCESSED_DIR = None
for cand in CAND_PROCESSED:
    if cand.exists():
        PROCESSED_DIR = cand
        break

# collect raw files and counts
files = []
for j in RAW_DIR.glob('*_results.json'):
    try:
        data = json.load(open(j, 'r', encoding='utf-8'))
        n = len(data.get('detections', []))
        files.append((n, j))
    except Exception:
        continue

files_sorted = sorted(files, key=lambda x: -x[0])
TOP_N = 12
selected = files_sorted[:TOP_N]

report = []
for n, raw_json in selected:
    base = raw_json.stem.replace('_results','')
    # load image
    img_path = None
    for ext in ['.jpg','.png','.jpeg']:
        p = IMG_DIR / f"{base}{ext}"
        if p.exists():
            img_path = p
            break
    if img_path is None:
        # try raw json may contain image_path
        try:
            jdata = json.load(open(raw_json, 'r', encoding='utf-8'))
            ip = jdata.get('image_path')
            if ip and Path(ip).exists():
                img_path = Path(ip)
        except Exception:
            pass
    if img_path is None:
        print(f"Image not found for {base}, skipping")
        continue
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"Failed to load image {img_path}")
        continue

    # load raw detections
    try:
        raw_data = json.load(open(raw_json, 'r', encoding='utf-8'))
        raw_dets = raw_data.get('detections', [])
    except Exception:
        raw_dets = []

    # load processed detections
    proc_file = None
    if PROCESSED_DIR:
        pf = PROCESSED_DIR / f"{base}_results.json"
        if pf.exists():
            proc_file = pf
    proc_dets = []
    if proc_file:
        try:
            pd = json.load(open(proc_file, 'r', encoding='utf-8'))
            proc_dets = pd.get('detections', [])
        except Exception:
            proc_dets = []

    # draw
    raw_vis = draw_dets_on_image(img, raw_dets, CLASS_NAMES)
    proc_vis = draw_dets_on_image(img, proc_dets, CLASS_NAMES)

    # resize to same height if needed
    h = 800
    raw_resized = cv2.resize(raw_vis, (int(raw_vis.shape[1] * h / raw_vis.shape[0]), h))
    proc_resized = cv2.resize(proc_vis, (int(proc_vis.shape[1] * h / proc_vis.shape[0]), h))

    # concatenate horizontally
    concat = np.concatenate([raw_resized, proc_resized], axis=1)

    out_name = OUT_DIR / f"{base}_compare.jpg"
    cv2.imwrite(str(out_name), concat)
    report.append((base, len(raw_dets), len(proc_dets), str(out_name)))

# print summary
print('Generated visual comparisons:')
for r in report:
    print(f'{r[0]} - raw:{r[1]} proc:{r[2]} -> {r[3]}')

print('\nAll comparison images saved to', OUT_DIR)
