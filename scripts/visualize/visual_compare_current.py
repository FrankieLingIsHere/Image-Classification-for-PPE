#!/usr/bin/env python3
"""
Create side-by-side visual comparisons (raw calibrated vs processed) for top N images.

Uses:
 - raw directory: outputs/calibration/applied_jsons
 - processed directory: outputs/safe_maxap_cap_calibrated/processed_jsons

Saves comparisons to outputs/visual_comparisons_current/
"""
from pathlib import Path
import json
import numpy as np
import shutil
import os
import argparse

ROOT = Path(__file__).parent.parent
import sys
sys.path.insert(0, str(ROOT))
from src.utils.utils import visualize_detections


def load_jsons(dir_path):
    d = Path(dir_path)
    files = list(d.glob('*_results.json'))
    data = {}
    for f in files:
        try:
            data[f.stem.replace('_results','')] = json.loads(f.read_text(encoding='utf-8'))
        except Exception:
            continue
    return data


def draw_side_by_side(image_path, raw, proc, class_names, out_path):
    # read image via matplotlib / PIL
    from PIL import Image
    im = Image.open(image_path).convert('RGB')
    im_np = np.array(im) / 255.0

    # prepare raw arrays
    raw_boxes = [d['bbox'] for d in raw.get('detections', [])]
    raw_scores = [d['confidence'] for d in raw.get('detections', [])]
    raw_labels = []
    for d in raw.get('detections', []):
        # try to map class_name to index in class_names
        try:
            raw_labels.append(class_names.index(d.get('class_name')))
        except Exception:
            raw_labels.append(0)

    proc_boxes = [d['bbox'] for d in proc.get('detections', [])]
    proc_scores = [d['confidence'] for d in proc.get('detections', [])]
    proc_labels = []
    for d in proc.get('detections', []):
        try:
            proc_labels.append(class_names.index(d.get('class_name')))
        except Exception:
            proc_labels.append(0)

    # create side-by-side by plotting raw on left, processed on right and saving figure
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1,2, figsize=(16,8))
    axes[0].imshow(im_np); axes[0].axis('off'); axes[0].set_title('Raw calibrated')
    axes[1].imshow(im_np); axes[1].axis('off'); axes[1].set_title('Processed (safe maxap cap)')

    # helper to draw on axis
    import matplotlib.patches as patches
    colors = ['red','blue','green','yellow','orange','purple','cyan','magenta','brown','grey']
    h,w = im_np.shape[:2]

    for i, b in enumerate(raw_boxes):
        bx = list(map(float, b))
        if max(bx) <= 1.0:
            x1 = bx[0]*w; y1 = bx[1]*h; x2 = bx[2]*w; y2 = bx[3]*h
        else:
            x1,y1,x2,y2 = bx
        rect = patches.Rectangle((x1,y1), x2-x1, y2-y1, linewidth=2, edgecolor=colors[raw_labels[i]%len(colors)], facecolor='none')
        axes[0].add_patch(rect)
        axes[0].text(x1, max(0,y1-5), f"{class_names[raw_labels[i]]}:{raw_scores[i]:.2f}", color='white', backgroundcolor=colors[raw_labels[i]%len(colors)])

    for i, b in enumerate(proc_boxes):
        bx = list(map(float, b))
        if max(bx) <= 1.0:
            x1 = bx[0]*w; y1 = bx[1]*h; x2 = bx[2]*w; y2 = bx[3]*h
        else:
            x1,y1,x2,y2 = bx
        rect = patches.Rectangle((x1,y1), x2-x1, y2-y1, linewidth=2, edgecolor=colors[proc_labels[i]%len(colors)], facecolor='none')
        axes[1].add_patch(rect)
        axes[1].text(x1, max(0,y1-5), f"{class_names[proc_labels[i]]}:{proc_scores[i]:.2f}", color='white', backgroundcolor=colors[proc_labels[i]%len(colors)])

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw_dir', default='outputs/calibration/applied_jsons')
    parser.add_argument('--proc_dir', default='outputs/safe_maxap_cap_calibrated/processed_jsons')
    parser.add_argument('--out_dir', default='outputs/visual_comparisons_current')
    parser.add_argument('--top_n', type=int, default=12)
    args = parser.parse_args()

    raw = load_jsons(args.raw_dir)
    proc = load_jsons(args.proc_dir)

    # gather class names if available
    class_file = Path('configs') / 'dataset_class_names.txt'
    if class_file.exists():
        class_names = [ln.strip() for ln in class_file.read_text(encoding='utf-8').splitlines() if ln.strip()]
    else:
        class_names = ['background','person','hard_hat','safety_vest','safety_gloves','safety_boots','eye_protection','no_hard_hat','no_safety_vest']

    # compute per-image delta counts
    deltas = []
    for img, r in raw.items():
        pr = proc.get(img, {'detections':[]})
        d = abs(len(r.get('detections',[])) - len(pr.get('detections',[])))
        deltas.append((img, d))

    deltas = sorted(deltas, key=lambda x: x[1], reverse=True)
    out_dir = Path(args.out_dir)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    for img, diff in deltas[:args.top_n]:
        raw_j = raw.get(img)
        proc_j = proc.get(img, {'detections':[]})
        # find image file path
        possible = list(Path('data/images').glob(f"{img}.*"))
        if not possible:
            print('Image file not found for', img)
            continue
        img_path = possible[0]
        out_path = out_dir / f"{img}_compare.jpg"
        draw_side_by_side(img_path, raw_j, proc_j, class_names, out_path)
        print('Saved comparison for', img, 'diff', diff)
        count += 1
    print('Wrote', count, 'comparisons to', out_dir)

if __name__ == '__main__':
    main()
