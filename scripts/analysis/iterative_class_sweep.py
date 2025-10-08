#!/usr/bin/env python3
"""Iterative greedy per-class sweep: for each noisy class, test a small list of candidate thresholds
and keep the threshold that improves mAP. Repeat for a few rounds.
"""
import sys
from pathlib import Path
import json
import shutil

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from scripts.eval_rcnn_combined_config import apply_and_write_all, evaluate

DETS_DIR = ROOT / 'outputs' / 'rcnn_baseline_adamw'
OUT_BASE = ROOT / 'outputs' / 'iterative_sweep'
OUT_BASE.mkdir(parents=True, exist_ok=True)

# starting base thresholds (from last auto-adjust run)
base = {
    'person': 0.20,
    'hard_hat': 0.25,
    'safety_vest': 0.25,
    'safety_gloves': 0.45,
    'safety_boots': 0.40,
    'eye_protection': 0.25,
    'no_hard_hat': 0.35,
    'no_eye_protection': 0.20,
    'no_safety_vest': 0.30,
    'no_safety_gloves': 0.35,
}

# classes to sweep and candidate thresholds
sweep_plan = {
    'eye_protection': [0.08, 0.12, 0.18, 0.25],
    'no_eye_protection': [0.08, 0.12, 0.18, 0.25],
    'safety_gloves': [0.25, 0.35, 0.45],
    'safety_boots': [0.2, 0.3, 0.4],
    'no_hard_hat': [0.2, 0.3, 0.35],
    'no_safety_gloves': [0.25, 0.35, 0.45],
}

person_cfg = {'person_conf_min': 0.20, 'person_final_max': 4, 'person_merge_iou': 0.45, 'person_overlap_threshold': 0.0, 'final_conf_min': 0.5}

def eval_with_map(th_map, run_id):
    proc = OUT_BASE / f'run_{run_id}' / 'processed_jsons'
    if proc.exists():
        shutil.rmtree(proc)
    apply_and_write_all(DETS_DIR, proc, th_map, person_cfg)
    eval_out = OUT_BASE / f'run_{run_id}' / 'evaluation'
    if eval_out.exists():
        shutil.rmtree(eval_out)
    eval_out.mkdir(parents=True, exist_ok=True)
    evaluate(proc, eval_out)
    summary = eval_out / 'evaluation_summary.json'
    if summary.exists():
        with open(summary, 'r', encoding='utf-8') as f:
            s = json.load(f)
        # map_key may be 'map_score' or 'mAP'
        return float(s.get('map_score') or s.get('mAP') or s.get('map') or 0.0)
    return 0.0

def main():
    best_map = eval_with_map(base, 'base')
    print('Base mAP:', best_map)

    improved = True
    run_idx = 1
    rounds = 0
    while improved and rounds < 3:
        improved = False
        rounds += 1
        for cls, candidates in sweep_plan.items():
            current = base.get(cls, 0.12)
            best_local = best_map
            best_thr = current
            for thr in candidates:
                if thr == current:
                    continue
                trial = dict(base)
                trial[cls] = thr
                run_id = f'r{rounds}_{cls}_{int(thr*100)}'
                print('Testing', cls, thr)
                m = eval_with_map(trial, run_id)
                print('mAP:', m)
                if m > best_local + 1e-6:
                    best_local = m
                    best_thr = thr
            if best_thr != current:
                print(f'Updating {cls}: {current} -> {best_thr} (mAP {best_local} > {best_map})')
                base[cls] = best_thr
                best_map = best_local
                improved = True
            run_idx += 1

    # write final thresholds
    out_json = OUT_BASE / 'final_thresholds.json'
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(base, f, indent=2)
    print('Finished. best mAP:', best_map)
    print('Final thresholds written to', out_json)

if __name__ == '__main__':
    main()
