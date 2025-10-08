#!/usr/bin/env python3
"""
Fine-grained sweep around the best person_conf and PPE thresholds.
- person_conf: 0.08,0.09,0.10,0.11,0.12
- PPE threshold sets: baseline (hard_hat=0.25,safety_vest=0.25), lower (0.22), higher (0.28)
Saves per-run outputs under outputs/sweep_fine and writes summary CSV.
"""
import os
import sys
from pathlib import Path
import csv
import pandas as pd

sys.path.append(str(Path(__file__).parent.parent))
from scripts.evaluate_detection_performance import PPEDetectionEvaluator


def parse_class_metrics(csv_path):
    df = pd.read_csv(csv_path)
    overall_map = df['ap'].mean() if 'ap' in df.columns else 0.0
    person_row = df[df['class'] == 'person'] if 'class' in df.columns else pd.DataFrame()
    person_recall = float(person_row['recall'].iloc[0]) if not person_row.empty else 0.0
    no_hat_row = df[df['class'] == 'no_hard_hat'] if 'class' in df.columns else pd.DataFrame()
    no_vest_row = df[df['class'] == 'no_safety_vest'] if 'class' in df.columns else pd.DataFrame()
    no_hat_recall = float(no_hat_row['recall'].iloc[0]) if not no_hat_row.empty else 0.0
    no_vest_recall = float(no_vest_row['recall'].iloc[0]) if not no_vest_row.empty else 0.0
    return overall_map, person_recall, (no_hat_recall + no_vest_recall) / 2.0


def parse_problem_analysis(txt_path):
    with open(txt_path, 'r', encoding='utf-8') as f:
        text = f.read()
    fp_count = 0
    if 'FALSE_POSITIVES' in text:
        parts = text.split('FALSE_POSITIVES:')
        if len(parts) > 1:
            tail = parts[1]
            stop_tokens = ['MISSED_VIOLATIONS', 'MISSED_WORKERS']
            end_idx = len(tail)
            for t in stop_tokens:
                idx = tail.find(t)
                if idx != -1 and idx < end_idx:
                    end_idx = idx
            fp_section = tail[:end_idx]
            for line in fp_section.splitlines():
                if line.strip().startswith('{'):
                    fp_count += 1
    return fp_count


def run_fine_sweep():
    models_dir = Path('models')
    model_files = list(models_dir.glob('*.pth'))
    if not model_files:
        print('No model found in models/')
        return
    latest_model = max(model_files, key=lambda x: x.stat().st_ctime)

    person_confs = [0.08, 0.09, 0.10, 0.11, 0.12]
    ppe_sets = {
        'baseline': {'hard_hat':0.25, 'safety_vest':0.25},
        'lower': {'hard_hat':0.22, 'safety_vest':0.22},
        'higher': {'hard_hat':0.28, 'safety_vest':0.28}
    }

    out_base = Path('outputs/sweep_fine')
    out_base.mkdir(parents=True, exist_ok=True)
    results = []

    for pc in person_confs:
        for set_name, ppe_thr in ppe_sets.items():
            name = f'pc_{pc:.2f}_{set_name}'
            out_dir = out_base / name
            print(f'Running: person_conf={pc:.2f}, set={set_name}')

            evaluator = PPEDetectionEvaluator(
                model_path=str(latest_model),
                data_dir='data',
                config_path='configs/ppe_config.yaml',
                output_dir=str(out_dir)
            )
            evaluator.conf_threshold = pc
            evaluator.iou_threshold = 0.3
            evaluator.person_overlap_threshold = 0.10
            # set per-class thresholds
            evaluator.class_conf_thresholds['person'] = pc
            evaluator.class_conf_thresholds['hard_hat'] = ppe_thr['hard_hat']
            evaluator.class_conf_thresholds['safety_vest'] = ppe_thr['safety_vest']

            try:
                evaluator.evaluate()
            except Exception as e:
                print('  Eval failed:', e)
                continue

            # find outputs
            class_csv = None
            problem_txt = None
            for p in out_dir.glob('class_metrics_*.csv'):
                class_csv = p
                break
            for p in out_dir.glob('problem_analysis_*.txt'):
                problem_txt = p
                break

            if class_csv is None:
                print('  Missing class metrics for', name)
                continue

            overall_map, person_recall, violation_recall = parse_class_metrics(class_csv)
            fp_count = parse_problem_analysis(problem_txt) if problem_txt is not None else 0

            results.append({
                'person_conf': pc,
                'ppe_set': set_name,
                'overall_map': overall_map,
                'person_recall': person_recall,
                'violation_recall': violation_recall,
                'false_positive_count': fp_count,
                'out_dir': str(out_dir)
            })

    # save summary
    csv_path = out_base / 'sweep_fine_results.csv'
    pd.DataFrame(results).to_csv(csv_path, index=False)
    print('Fine sweep done ->', csv_path)

if __name__ == '__main__':
    run_fine_sweep()
