#!/usr/bin/env python3
"""
Sweep person confidence vs person-overlap thresholds.
For each (person_conf, overlap) config the script will:
 - instantiate PPEDetectionEvaluator with iou_threshold=0.3
 - set evaluator.conf_threshold = person_conf
 - set evaluator.person_overlap_threshold = overlap
 - run evaluator.evaluate() (results saved under a unique output dir)
 - parse saved class_metrics_*.csv and problem_analysis_*.txt to extract:
     - overall mAP (mean AP across classes)
     - person recall
     - avg violation recall (no_hard_hat & no_safety_vest)
     - false positive count (from problem_analysis False Positives entries)
 - save aggregated results to outputs/sweep/sweep_results.csv

This is intended as a quick configuration sweep to find a good person_conf / overlap setting.
"""

import os
import sys
from pathlib import Path
import csv
import glob
import pandas as pd

sys.path.append(str(Path(__file__).parent.parent))
from scripts.evaluate_detection_performance import PPEDetectionEvaluator


def parse_class_metrics(csv_path):
    df = pd.read_csv(csv_path)
    # compute overall mAP as mean of 'ap' column
    overall_map = df['ap'].mean() if 'ap' in df.columns else 0.0
    # find person recall
    if 'class' in df.columns:
        person_row = df[df['class'] == 'person']
        person_recall = float(person_row['recall'].iloc[0]) if not person_row.empty else 0.0
        no_hat_row = df[df['class'] == 'no_hard_hat']
        no_vest_row = df[df['class'] == 'no_safety_vest']
        no_hat_recall = float(no_hat_row['recall'].iloc[0]) if not no_hat_row.empty else 0.0
        no_vest_recall = float(no_vest_row['recall'].iloc[0]) if not no_vest_row.empty else 0.0
    else:
        person_recall = 0.0
        no_hat_recall = 0.0
        no_vest_recall = 0.0

    return overall_map, person_recall, (no_hat_recall + no_vest_recall) / 2.0


def parse_problem_analysis(txt_path):
    # Count False Positives entries
    with open(txt_path, 'r', encoding='utf-8') as f:
        text = f.read()
    # crude parsing: find 'FALSE_POSITIVES' section and count lines between it and next section
    fp_count = 0
    if 'FALSE_POSITIVES' in text:
        parts = text.split('FALSE_POSITIVES:')
        if len(parts) > 1:
            tail = parts[1]
            # stop at next section 'MISSED_VIOLATIONS' or end
            stop_tokens = ['MISSED_VIOLATIONS', 'MISSED_WORKERS']
            end_idx = len(tail)
            for t in stop_tokens:
                idx = tail.find(t)
                if idx != -1 and idx < end_idx:
                    end_idx = idx
            fp_section = tail[:end_idx]
            # count lines that look like dict entries starting with two spaces and '{'
            for line in fp_section.splitlines():
                line = line.strip()
                if line.startswith('{'):
                    fp_count += 1
    return fp_count


def run_sweep():
    models_dir = Path('models')
    model_files = list(models_dir.glob('*.pth'))
    if not model_files:
        print('No model found in models/')
        return
    latest_model = max(model_files, key=lambda x: x.stat().st_ctime)

    person_confs = [0.1, 0.15, 0.2, 0.25, 0.3]
    overlaps = [0.1, 0.2, 0.3, 0.4]

    results = []
    out_base = Path('outputs/sweep')
    out_base.mkdir(parents=True, exist_ok=True)

    for pc in person_confs:
        for ov in overlaps:
            name = f'personconf_{pc:.2f}_overlap_{ov:.2f}'
            out_dir = out_base / name
            if out_dir.exists():
                # remove existing folder to avoid mixing results
                # (safer to skip, but we'll proceed to overwrite)
                pass

            print(f'Running sweep config: person_conf={pc:.2f}, overlap={ov:.2f}')
            evaluator = PPEDetectionEvaluator(
                model_path=str(latest_model),
                data_dir='data',
                config_path='configs/ppe_config.yaml',
                output_dir=str(out_dir)
            )
            # set thresholds
            evaluator.conf_threshold = pc
            evaluator.iou_threshold = 0.3
            evaluator.person_overlap_threshold = ov

            # run evaluation (writes files into out_dir)
            try:
                evaluator.evaluate()
            except Exception as e:
                print(f'  Evaluation failed for {name}: {e}')
                continue

            # find generated files
            class_csv = None
            problem_txt = None
            for p in out_dir.glob('class_metrics_*.csv'):
                class_csv = p
                break
            for p in out_dir.glob('problem_analysis_*.txt'):
                problem_txt = p
                break

            if class_csv is None:
                print(f'  Missing class metrics for {name}')
                continue
            if problem_txt is None:
                print(f'  Missing problem analysis for {name}')

            overall_map, person_recall, violation_recall = parse_class_metrics(class_csv)
            fp_count = parse_problem_analysis(problem_txt) if problem_txt is not None else 0

            results.append({
                'person_conf': pc,
                'person_overlap': ov,
                'overall_map': overall_map,
                'person_recall': person_recall,
                'violation_recall': violation_recall,
                'false_positive_count': fp_count,
                'out_dir': str(out_dir)
            })

    # write summary CSV
    csv_path = out_base / 'sweep_results.csv'
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['person_conf','person_overlap','overall_map','person_recall','violation_recall','false_positive_count','out_dir'])
        writer.writeheader()
        for r in results:
            writer.writerow(r)

    print('\nSweep complete. Summary saved to:', csv_path)

if __name__ == '__main__':
    run_sweep()
