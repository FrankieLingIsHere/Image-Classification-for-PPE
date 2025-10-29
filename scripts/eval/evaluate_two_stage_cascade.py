#!/usr/bin/env python3
"""Evaluate two-stage cascade (Stage1 person -> Stage2 PPE on crops).

This script runs Stage1 on full images to get person boxes, crops each person
region, runs Stage2 on those crops to get PPE detections, maps detections back
to original image coordinates, and writes per-image JSON results compatible
with `evaluate_from_jsons.py`.

It is intentionally self-contained and avoids importing the larger evaluator to
remain robust to interpreter/version differences.
"""
import os
import sys
import json
import argparse
from pathlib import Path
from PIL import Image
import torch
from torchvision.transforms import functional as F
import torchvision.transforms as T
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.ops import nms

# Ensure repo root on path for utility functions
sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.utils.utils import calculate_iou  # used by downstream evaluator

PPE_ONLY_CLASSES = [
    'background', 'hard_hat', 'safety_vest', 'safety_gloves',
    'safety_boots', 'eye_protection', 'no_hard_hat', 'no_safety_vest',
    'no_safety_gloves', 'no_safety_boots', 'no_eye_protection'
]


def create_fasterrcnn(num_classes, pretrained=False):
    """Create a Faster R-CNN and replace the head to expected number of classes."""
    model = fasterrcnn_resnet50_fpn(weights='DEFAULT' if pretrained else None)
    # Replace head
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def load_checkpoint_to_model(ckpt_path, model, device):
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(ckpt_path)
    raw = torch.load(ckpt_path, map_location='cpu')
    if isinstance(raw, dict) and ('model_state_dict' in raw or 'state_dict' in raw):
        sd = raw.get('model_state_dict', raw.get('state_dict'))
    else:
        sd = raw

    try:
        model.load_state_dict(sd, strict=False)
        print(f"[OK] Loaded checkpoint {ckpt_path} (partial/relaxed load)")
    except Exception as e:
        print(f"[WARN] load_state_dict warning: {e}")
        # still attempt non-strict load above; if it raised we continue

    model.to(device)
    model.eval()
    return model


def transform_for_model():
    # Use same normalization used during training (ToTensor + Normalize)
    return T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def run_cascade_on_image(img_path, stage1, stage2, device, cfg):
    """Return list of detections for the original image as dicts.
    Each detection: {'class': name, 'bbox': [x1,y1,x2,y2], 'confidence': float}
    """
    pil = Image.open(img_path).convert('RGB')
    width, height = pil.size

    t = transform_for_model()
    img_t = t(pil).to(device)

    # Stage 1 detect persons
    with torch.no_grad():
        outputs = stage1([img_t])

    if not outputs or len(outputs) == 0:
        return []

    out = outputs[0]
    boxes = out.get('boxes', torch.zeros((0, 4))).cpu()
    labels = out.get('labels', torch.zeros((0,), dtype=torch.int64)).cpu()
    scores = out.get('scores', torch.zeros((0,))).cpu()

    # Debug: print all Stage1 detections for first few images
    global debug_counter
    img_name = Path(img_path).name
    if debug_counter < 3:
        print(f"[DEBUG] {img_name} Stage1 raw detections: {len(scores)} boxes")
        for i in range(min(10, len(scores))):  # print top 10
            print(f"  Box {i}: label={labels[i].item()}, score={scores[i].item():.3f}, bbox={boxes[i].tolist()}")
        debug_counter += 1

    person_indices = [i for i, lab in enumerate(labels.tolist()) if int(lab) == 1 and scores[i] >= cfg['stage1_conf']]

    detections = []

    for i in person_indices:
        bx = boxes[i].tolist()  # [x1,y1,x2,y2]
        x1, y1, x2, y2 = [max(0, float(v)) for v in bx]
        score = float(scores[i].item())
        detections.append({'class': 'person', 'bbox': [x1, y1, x2, y2], 'confidence': score})  # Add person detections

    for i in person_indices:
        bx = boxes[i].tolist()  # [x1,y1,x2,y2]
        x1, y1, x2, y2 = [max(0, float(v)) for v in bx]
        # optional padding
        pad = cfg.get('person_crop_pad', 0.03)
        w = max(1.0, x2 - x1)
        h = max(1.0, y2 - y1)
        x1p = max(0, x1 - pad * w)
        y1p = max(0, y1 - pad * h)
        x2p = min(width, x2 + pad * w)
        y2p = min(height, y2 + pad * h)

        crop = pil.crop((x1p, y1p, x2p, y2p))
        crop_t = t(crop).to(device)

        with torch.no_grad():
            out2s = stage2([crop_t])

        if not out2s or len(out2s) == 0:
            continue
        out2 = out2s[0]
        b2 = out2.get('boxes', torch.zeros((0, 4))).cpu()
        l2 = out2.get('labels', torch.zeros((0,), dtype=torch.int64)).cpu()
        s2 = out2.get('scores', torch.zeros((0,))).cpu()

        for j in range(b2.shape[0]):
            score = float(s2[j].item())
            if score < cfg['stage2_conf']:
                continue
            lab = int(l2[j].item())
            # Stage2 class indices assume PPE_ONLY_CLASSES mapping: background=0
            if lab < 0 or lab >= len(PPE_ONLY_CLASSES):
                cname = 'unknown'
            else:
                cname = PPE_ONLY_CLASSES[lab]

            bx_crop = b2[j].tolist()
            # map crop bbox back to original image coords
            cx1, cy1, cx2, cy2 = bx_crop
            # bbox coordinates in crop are in same pixel space as crop
            ox1 = float(x1p + cx1)
            oy1 = float(y1p + cy1)
            ox2 = float(x1p + cx2)
            oy2 = float(y1p + cy2)

            detections.append({'class': cname, 'bbox': [ox1, oy1, ox2, oy2], 'confidence': score})

    # Optionally do simple class-wise NMS to reduce duplicates
    if cfg.get('nms', True) and len(detections) > 0:
        # group by class
        grouped = {}
        for d in detections:
            grouped.setdefault(d['class'], []).append(d)

        final = []
        for cls, items in grouped.items():
            boxes_t = torch.tensor([it['bbox'] for it in items], dtype=torch.float32)
            scores_t = torch.tensor([it['confidence'] for it in items], dtype=torch.float32)
            if boxes_t.numel() == 0:
                continue
            keep = nms(boxes_t, scores_t, cfg.get('nms_iou', 0.45))
            for k in keep.tolist():
                final.append(items[k])
        detections = final

    # Convert bbox to absolute coordinates (already absolute) and return
    return detections


def write_results_json(base_out_dir, img_name, detections):
    base = os.path.splitext(img_name)[0]
    out_path = Path(base_out_dir) / f"{base}_results.json"
    payload = {'detections': []}
    for d in detections:
        payload['detections'].append({'class': d['class'], 'bbox': d['bbox'], 'confidence': float(d['confidence'])})
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2)


def main():
    global debug_counter
    debug_counter = 0
    parser = argparse.ArgumentParser(description='Cascade evaluate Stage1->Stage2')
    parser.add_argument('--stage1_ckpt', type=str, default='models/stage1_human_best.pth',
                        help='Stage1 (person) checkpoint; default: models/stage1_human_best.pth')
    parser.add_argument('--stage2_ckpt', type=str, default='models/stage2_ppe_best.pth',
                        help='Stage2 (PPE) checkpoint; default: models/stage2_ppe_best.pth')
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--split', type=str, default='val', choices=['train', 'val', 'test'])
    parser.add_argument('--output_dir', type=str, default='outputs/eval_two_stage_from_jsons')
    parser.add_argument('--max_images', type=int, default=None)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--stage1_conf', type=float, default=0.3)
    parser.add_argument('--stage2_conf', type=float, default=0.1)
    parser.add_argument('--person_crop_pad', type=float, default=0.03,
                        help='Padding fraction around person bbox when cropping')
    parser.add_argument('--use_gt_persons', action='store_true',
                        help='If set, use ground-truth person boxes from annotations as crops for Stage2 (isolates Stage2 performance)')
    parser.add_argument('--use_history', action='store_true',
                        help='If set and models/training_history_two_stage.json exists, read its config to set crop pad and other defaults')
    args = parser.parse_args()

    device = torch.device(args.device)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load image list
    split_file = Path(args.data_dir) / 'splits' / f"{args.split}.txt"
    with open(split_file, 'r') as f:
        images = [ln.strip() for ln in f.readlines() if ln.strip()]
    if args.max_images:
        images = images[:args.max_images]

    # Optionally read saved history for defaults (crop pad etc.)
    history_path = Path('models') / 'training_history_two_stage.json'
    if args.use_history and history_path.exists():
        try:
            with open(history_path, 'r', encoding='utf-8') as hf:
                hist = json.load(hf)
            cfg_from_hist = hist.get('config', {})
            # if the training config contains useful defaults, apply them
            if 'person_crop_pad' in cfg_from_hist:
                args.person_crop_pad = float(cfg_from_hist['person_crop_pad'])
            print(f"[INFO] Loaded config from {history_path}")
        except Exception as e:
            print(f"[WARN] Could not read history: {e}")

    # Create models
    print("[INFO] Creating Stage 1 model (person detector)")
    stage1 = create_fasterrcnn(num_classes=2, pretrained=False)
    try:
        stage1 = load_checkpoint_to_model(args.stage1_ckpt, stage1, device)
    except FileNotFoundError:
        print(f"[WARN] Stage1 checkpoint not found: {args.stage1_ckpt}; Stage1 will be unavailable unless --use_gt_persons is set")
        stage1 = None

    print("[INFO] Creating Stage 2 model (PPE detector on crops)")
    stage2 = create_fasterrcnn(num_classes=len(PPE_ONLY_CLASSES), pretrained=False)
    stage2 = load_checkpoint_to_model(args.stage2_ckpt, stage2, device)

    cfg = {'stage1_conf': args.stage1_conf, 'stage2_conf': args.stage2_conf, 'person_crop_pad': args.person_crop_pad, 'nms': True, 'nms_iou': 0.45}

    # Process images and write per-image JSONs
    print(f"[INFO] Running cascade on {len(images)} images (split={args.split})")
    debug_counter = 0
    for img_name in images:
        img_path = Path(args.data_dir) / 'images' / img_name
        if not img_path.exists():
            print(f"[WARN] Image not found: {img_path}")
            continue

        # If requested, use ground-truth person boxes as crops for Stage2 (isolated Stage2 eval)
        if args.use_gt_persons:
            ann_json = Path(args.data_dir) / 'annotations' / f"{Path(img_name).stem}.json"
            ann_xml = Path(args.data_dir) / 'annotations' / f"{Path(img_name).stem}.xml"
            person_boxes = []
            if ann_json.exists():
                try:
                    with open(ann_json, 'r', encoding='utf-8') as af:
                        data = json.load(af)
                    for ann in data.get('annotations', []) + data.get('detections', []):
                        if ann.get('class') == 'person' or ann.get('category') == 'person' or ann.get('label') == 'person':
                            person_boxes.append(ann.get('bbox'))
                except Exception:
                    pass
            elif ann_xml.exists():
                # simple parsing to extract person bboxes (xmin,ymin,xmax,ymax)
                try:
                    import xml.etree.ElementTree as ET
                    tree = ET.parse(str(ann_xml))
                    root = tree.getroot()
                    for obj in root.findall('object'):
                        name = obj.find('name').text
                        if name != 'person':
                            continue
                        bbox = obj.find('bndbox')
                        x1 = float(bbox.find('xmin').text)
                        y1 = float(bbox.find('ymin').text)
                        x2 = float(bbox.find('xmax').text)
                        y2 = float(bbox.find('ymax').text)
                        person_boxes.append([x1, y1, x2, y2])
                except Exception:
                    pass

            # For each GT person box, crop and run stage2
            dets = []
            pil = Image.open(img_path).convert('RGB')
            width, height = pil.size
            t = transform_for_model()
            for bx in person_boxes:
                x1, y1, x2, y2 = [float(v) for v in bx]
                pad = args.person_crop_pad
                w = max(1.0, x2 - x1); h = max(1.0, y2 - y1)
                x1p = max(0, x1 - pad * w); y1p = max(0, y1 - pad * h)
                x2p = min(width, x2 + pad * w); y2p = min(height, y2 + pad * h)
                crop = pil.crop((x1p, y1p, x2p, y2p))
                crop_t = t(crop).to(device)
                with torch.no_grad():
                    out2s = stage2([crop_t])
                if not out2s:
                    continue
                out2 = out2s[0]
                b2 = out2.get('boxes', torch.zeros((0, 4))).cpu()
                l2 = out2.get('labels', torch.zeros((0,), dtype=torch.int64)).cpu()
                s2 = out2.get('scores', torch.zeros((0,))).cpu()
                for j in range(b2.shape[0]):
                    score = float(s2[j].item())
                    if score < args.stage2_conf:
                        continue
                    lab = int(l2[j].item())
                    cname = PPE_ONLY_CLASSES[lab] if 0 <= lab < len(PPE_ONLY_CLASSES) else 'unknown'
                    cx1, cy1, cx2, cy2 = b2[j].tolist()
                    ox1 = float(x1p + cx1); oy1 = float(y1p + cy1); ox2 = float(x1p + cx2); oy2 = float(y1p + cy2)
                    dets.append({'class': cname, 'bbox': [ox1, oy1, ox2, oy2], 'confidence': score})

        else:
            dets = run_cascade_on_image(img_path, stage1, stage2, device, cfg)

        write_results_json(out_dir, img_name, dets)

    print(f"[OK] Wrote per-image JSONs to {out_dir}")

    # After writing JSONs, instruct the user to run evaluate_from_jsons.py to compute metrics
    print("\nNow run the existing evaluator on the generated JSONs to compute mAP:")
    print(f"python scripts/eval/evaluate_from_jsons.py --detections_dir {out_dir} --data_dir {args.data_dir}")


if __name__ == '__main__':
    main()
