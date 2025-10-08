#!/usr/bin/env python3
"""
Apply postprocessing (per-class NMS + person cleanup) to existing detection JSONs.

This avoids needing a model checkpoint and lets us test the recommended settings quickly.

Usage example (PowerShell):
    .venv/Scripts/python.exe scripts/run_inference_postprocess.py --detections_dir outputs/rcnn_baseline_adamw --out_dir outputs/postproc_rcnn_recommended --person_merge_iou 0.45 --per_class_iou 0.6 --final_conf_min 0.3 --save_images

"""
from pathlib import Path
import sys
import os
import json
import argparse
import shutil

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from scripts.inference import _apply_per_class_nms, _postprocess_persons, _to_serializable


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--detections_dir', required=True)
    p.add_argument('--out_dir', required=True)
    p.add_argument('--person_merge_iou', type=float, default=0.45)
    p.add_argument('--per_class_iou', type=float, default=0.6)
    p.add_argument('--final_conf_min', type=float, default=0.3)
    p.add_argument('--save_images', action='store_true')
    p.add_argument('--save_json', action='store_true')
    return p.parse_args()


def main():
    args = parse_args()
    det_dir = Path(args.detections_dir)
    out_dir = Path(args.out_dir)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    proc_dir = out_dir / 'processed_jsons'
    proc_dir.mkdir()

    # class_post_nms_iou as dict defaulting to per-class iou
    class_post = {}

    files = list(det_dir.glob('*_results.json'))
    print(f"Found {len(files)} detection files in {det_dir}")
    for f in files:
        try:
            j = json.loads(f.read_text(encoding='utf-8'))
        except Exception as e:
            print('skip', f, e)
            continue
        dets = j.get('detections', [])
        # convert bbox coordinate styles if needed (keep as-is)
        # apply per-class NMS
        processed = _apply_per_class_nms(dets, iou_thresh=args.per_class_iou, per_class_iou=None, final_conf_min=args.final_conf_min)
        # create dummy args namespace for person postprocess
        from argparse import Namespace
        darr = Namespace()
        darr.person_merge_iou = args.person_merge_iou
        darr.person_conf_min = None
        darr.person_area_min_frac = 0.0
        darr.person_area_max_frac = 1.0
        darr.person_final_max = None
        # person postprocess expects image path; we'll pass a dummy Path
        proto = {'image_path': j.get('image_path','')}
        results = {'image_path': j.get('image_path',''), 'detections': processed}
        results = _postprocess_persons(results, Path(j.get('image_path','')), darr, [d.get('class_name') for d in j.get('detections', [])])
        # write processed json
        outp = proc_dir / f.name
        with open(outp, 'w', encoding='utf-8') as of:
            json.dump(_to_serializable(results), of, indent=2)
    print('Processed JSONs written to', proc_dir)

    # run evaluator using the repository Python executable
    try:
        import subprocess, sys
        eval_out = out_dir / 'evaluation'
        cmd = [sys.executable, str(Path(__file__).parent / 'evaluate_from_jsons.py'), '--detections_dir', str(proc_dir), '--output_dir', str(eval_out), '--eval_iou', '0.5', '--ignore_difficult']
        print('Running evaluator:', ' '.join(cmd))
        subprocess.run(cmd, check=True)
        print('Evaluation completed and written to', eval_out)
    except Exception as e:
        print('Evaluator failed:', e)

if __name__ == '__main__':
    main()
