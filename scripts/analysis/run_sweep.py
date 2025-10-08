#!/usr/bin/env python3
"""
Run a small sweep of person postprocessing params over a sample of images.

This script creates temp YAML configs in outputs/sweep_configs/, runs inference per-image per-config
(using scripts/inference.py), runs check_redundant_detections.py at IoU 0.3 and 0.5 and collects metrics
into outputs/sweep/sweep_results.csv.
"""
import os
import subprocess
import sys
import csv
from pathlib import Path
import yaml

ROOT = Path(__file__).parent.parent
OUT = ROOT / 'outputs' / 'sweep'
CFG_DIR = OUT / 'sweep_configs'
OUT.mkdir(parents=True, exist_ok=True)
CFG_DIR.mkdir(parents=True, exist_ok=True)

# sample of 10 images from data/splits/test.txt
images = [
    'image61.jpg', 'image26.png', 'image118.jpg', 'image124.png', 'image59.png',
    'image18.jpg', 'image111.png', 'image77.png', 'image95.jpg', 'image78.jpg'
]

# base runtime config to copy from
base_cfg_path = ROOT / 'configs' / 'best_runtime_config.yaml'
with open(base_cfg_path, 'r', encoding='utf-8') as f:
    base_cfg = yaml.safe_load(f)

# sweep combos (smaller grid)
person_merge_ious = [0.35, 0.45, 0.5]
person_conf_and_final = [ (0.12, 6), (0.15, 4) ]

combos = []
for pm in person_merge_ious:
    for conf_min, final_max in person_conf_and_final:
        name = f"pm{int(pm*100)}_pc{int(conf_min*100)}_max{final_max}"
        cfg = dict(base_cfg)
        cfg['post_nms_iou'] = 0.45
        cfg['class_post_nms_iou'] = {'person': 0.45}
        cfg['max_detections_per_class'] = {'person': 6}
        # person-specific
        cfg['person_conf_min'] = conf_min
        cfg['person_area_min_frac'] = 0.001
        cfg['person_area_max_frac'] = 0.6
        cfg['person_merge_iou'] = pm
        cfg['person_final_max'] = final_max
        combos.append((name, cfg))

# CSV header
csv_path = OUT / 'sweep_results.csv'
with open(csv_path, 'w', newline='', encoding='utf-8') as csvf:
    writer = csv.writer(csvf)
    writer.writerow([
        'config_name', 'image', 'total_detections', 'person_detections',
        'duplicate_pairs_iou_0.3', 'gt_multi_match_iou_0.3',
        'duplicate_pairs_iou_0.5', 'gt_multi_match_iou_0.5'
    ])

# run sweep
for cfg_name, cfg in combos:
    cfg_file = CFG_DIR / f"{cfg_name}.yaml"
    with open(cfg_file, 'w', encoding='utf-8') as cf:
        yaml.safe_dump(cfg, cf)

    print(f"Running config: {cfg_name}")
    config_out_dir = OUT / cfg_name
    config_out_dir.mkdir(parents=True, exist_ok=True)

    for img in images:
        img_path = ROOT / 'data' / 'images' / img
        if not img_path.exists():
            print(f"Image missing: {img_path}, skipping")
            continue

        # run inference
        cmd = [
            sys.executable, 'scripts/inference.py',
            '--model_path', 'models/best_model_regularized.pth',
            '--input', str(img_path),
            '--output_dir', str(config_out_dir),
            '--save_images', '--save_json',
            '--config_path', str(cfg_file)
        ]
        print('  running inference for', img)
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            # log and skip this image
            err_log = OUT / 'errors.log'
            with open(err_log, 'a', encoding='utf-8') as ef:
                ef.write(f"Config={cfg_name} Image={img} inference failed:\n")
                ef.write(proc.stdout + '\n')
                ef.write(proc.stderr + '\n')
            print(f"  inference failed for {img}, see {err_log}")
            continue

        stem = Path(img).stem
        result_json = config_out_dir / f"{stem}_results.json"
        if not result_json.exists():
            print(f"  result JSON missing for {img} (inference likely failed), skipping checks")
            # write a row indicating failure
            with open(csv_path, 'a', newline='', encoding='utf-8') as csvf:
                writer = csv.writer(csvf)
                writer.writerow([cfg_name, img, 0, 0, 'NA', 'NA', 'NA', 'NA'])
            continue
        ann_xml = ROOT / 'data' / 'annotations' / f"{stem}.xml"

        # helper to run checker and parse
        def run_checker(iou):
            cmd2 = [
                sys.executable, 'scripts/check_redundant_detections.py',
                '--json', str(result_json),
                '--annotation', str(ann_xml),
                '--iou_dup', str(iou),
                '--iou_match', str(iou)
            ]
            proc = subprocess.run(cmd2, check=True, capture_output=True, text=True)
            out = proc.stdout
            # parse total detections
            total = None
            dup = 0
            gt_multi = 0
            for line in out.splitlines():
                if line.startswith('Total detections:'):
                    total = int(line.split(':')[-1].strip())
                if line.strip().startswith('Duplicate same-class pairs:'):
                    dup = int(line.split(':')[-1].strip())
                if line.strip().startswith('GT boxes with >1 matching detections:'):
                    part = line.split(':')[-1].strip()
                    gt_multi = int(part.split('/')[0].strip())
            return total, dup, gt_multi

        # run for iou 0.3 and 0.5
        tot_03, dup03, gtm03 = run_checker(0.3)
        tot_05, dup05, gtm05 = run_checker(0.5)

        # person count in result_json
        import json
        with open(result_json, 'r', encoding='utf-8') as jf:
            data = json.load(jf)
        persons = sum(1 for d in data.get('detections', []) if d.get('class_name') == 'person')

        # write CSV
        with open(csv_path, 'a', newline='', encoding='utf-8') as csvf:
            writer = csv.writer(csvf)
            writer.writerow([
                cfg_name, img, tot_03 or tot_05 or 0, persons, dup03, gtm03, dup05, gtm05
            ])

print('Sweep completed. Results at', csv_path)
