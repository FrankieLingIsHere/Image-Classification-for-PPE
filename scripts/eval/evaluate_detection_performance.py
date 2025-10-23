#!/usr/bin/env python3
"""
PPE Detection Performance Evaluation Script (moved to scripts/eval)

This file is a copy of the top-level evaluator adapted for the new location.
"""

import os
import json
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

import torch
import torchvision.transforms as transforms
from PIL import Image
import xml.etree.ElementTree as ET
import yaml

# Ensure repo root is importable
import sys
sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.models.ssd import build_ssd_model
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from src.dataset.ppe_dataset import PPEDataset
from src.utils.utils import calculate_iou


def _normalize_label(name: str) -> str:
    if not name:
        return name
    s = name.strip().lower().replace(' ', '_')

    mapping = {
        'no_safety_glove': 'no_safety_gloves',
        'no_safety_gloves': 'no_safety_gloves',
        'no_safety_boot': 'no_safety_boots',
        'no_safety_boots': 'no_safety_boots',
        'no_eye_protection': 'no_eye_protection',
        'no_hardhat': 'no_hard_hat',
        'nohardhat': 'no_hard_hat',
        'no_hard_hat': 'no_hard_hat',
        'hardhat': 'hard_hat',
        'safetyvest': 'safety_vest',
        'no_safety_vest': 'no_safety_vest'
    }

    return mapping.get(s, s)


class PPEDetectionEvaluator:
    """Comprehensive evaluation of PPE detection performance"""
    
    def __init__(self, model_path, data_dir, config_path, output_dir, rescorer_alpha: float = 1.0):
        self.model_path = model_path
        self.data_dir = Path(data_dir)
        self.config_path = config_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        default_class_names = [
            'background', 'person', 'hard_hat', 'safety_vest',
            'safety_gloves', 'safety_boots', 'eye_protection',
            'no_hard_hat', 'no_safety_vest'
        ]
        class_file = Path(__file__).resolve().parents[2] / 'configs' / 'dataset_class_names.txt'
        if class_file.exists():
            with open(class_file, 'r', encoding='utf-8') as cf:
                cls = [ln.strip() for ln in cf.readlines() if ln.strip()]
            self.class_names = cls
        else:
            self.class_names = default_class_names
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self._load_model()

        # Lazy rescorer (load once on first use)
        self.rescorer = None
        self.rescorer_path = Path(__file__).resolve().parents[2] / 'models' / 'rescorer.pth'

        # blending weight for rescorer (alpha in [0,1])
        # alpha=1.0 => fully use multiplicative rescore (original behavior)
        # alpha=0.0 => ignore rescorer
        self.rescorer_alpha = float(rescorer_alpha)

        self.conf_threshold = 0.5
        self.iou_threshold = 0.45
        self.eval_iou_threshold = 0.5
        self.person_overlap_threshold = 0.3
        self.class_conf_thresholds = {
            'person': 0.10,
            'hard_hat': 0.25,
            'safety_vest': 0.25,
            'safety_gloves': 0.20,
            'safety_boots': 0.20,
            'eye_protection': 0.20,
            'no_hard_hat': 0.12,
            'no_safety_vest': 0.12
        }

        self.effective_config = {
            'conf_threshold': self.conf_threshold,
            'iou_threshold': self.iou_threshold,
            'eval_iou_threshold': self.eval_iou_threshold,
            'person_overlap_threshold': self.person_overlap_threshold,
            'class_conf_thresholds': self.class_conf_thresholds
        }

        if self.config_path and os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    cfg = yaml.safe_load(f)
                for k in ['conf_threshold', 'iou_threshold', 'eval_iou_threshold', 'person_overlap_threshold']:
                    if k in cfg:
                        setattr(self, k, cfg[k])
                        self.effective_config[k] = cfg[k]

                if 'class_conf_thresholds' in cfg and isinstance(cfg['class_conf_thresholds'], dict):
                    self.class_conf_thresholds.update(cfg['class_conf_thresholds'])
                    self.effective_config['class_conf_thresholds'] = self.class_conf_thresholds
            except Exception as e:
                print(f"[WARN]  Failed to load config {self.config_path}: {e}")
        
        self.results = {
            'summary': {},
            'per_class_metrics': {},
            'detection_results': [],
            'problem_cases': {
                'missed_workers': [],
                'false_positives': [],
                'missed_violations': []
            }
        }

    def _load_model(self):
        print(f"Loading model from: {self.model_path}")
        if os.path.exists(self.model_path):
            checkpoint = torch.load(self.model_path, map_location='cpu')
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint
            if 'pred_convs.cl_conv4_3.weight' in state_dict:
                cl_weight_shape = state_dict['pred_convs.cl_conv4_3.weight'].shape[0]
                n_classes_from_checkpoint = cl_weight_shape // 4
                print(f"Detected {n_classes_from_checkpoint} classes from checkpoint")
                if n_classes_from_checkpoint != len(self.class_names):
                    print(f"Adjusting class count from {len(self.class_names)} to {n_classes_from_checkpoint}")
                    if n_classes_from_checkpoint > len(self.class_names):
                        while len(self.class_names) < n_classes_from_checkpoint:
                            self.class_names.append(f'class_{len(self.class_names)}')
                    else:
                        self.class_names = self.class_names[:n_classes_from_checkpoint]
            else:
                n_classes_from_checkpoint = len(self.class_names)
        else:
            n_classes_from_checkpoint = len(self.class_names)

        # First attempt: SSD model (backwards-compatible)
        try:
            model = build_ssd_model(num_classes=n_classes_from_checkpoint)
            if os.path.exists(self.model_path):
                checkpoint = torch.load(self.model_path, map_location=self.device)
                sd = checkpoint.get('model_state_dict', checkpoint)
                try:
                    model.load_state_dict(sd)
                    print("[OK] SSD model loaded successfully")
                    self.model_type = 'ssd'
                except Exception as e:
                    # If SSD load failed, fall through to try RCNN
                    print(f"[WARN] SSD load failed ({e}), attempting Faster R-CNN load...")
                    raise
            else:
                print("[WARN]  Model file not found, using random SSD weights")
                self.model_type = 'ssd'

        except Exception:
            # Try loading as a Faster R-CNN checkpoint (common for rcnn_baseline.pth)
            try:
                print("Attempting to construct Faster R-CNN model (resnet50_fpn)")
                model = fasterrcnn_resnet50_fpn(pretrained=False, num_classes=n_classes_from_checkpoint)
                if os.path.exists(self.model_path):
                    checkpoint = torch.load(self.model_path, map_location=self.device)
                    sd = checkpoint.get('model_state_dict', checkpoint)
                    try:
                        # Attempt a non-strict load to tolerate small key differences
                        model.load_state_dict(sd, strict=False)
                        print("[OK] Faster R-CNN loaded (partial/relaxed match)")
                    except Exception as ee:
                        print(f"[WARN] Faster R-CNN load warning: {ee}")
                else:
                    print("[WARN]  Model file not found, using random R-CNN weights")

                self.model_type = 'rcnn'
            except Exception as final_e:
                # If both fail, re-raise the original error
                raise RuntimeError(f"Failed to load model as SSD or Faster R-CNN: {final_e}")

        model.to(self.device)
        model.eval()
        return model

    def _load_rescorer_once(self):
        """Load the RelationalRescorer from disk once and cache it on the evaluator instance.
        Loading is tolerant to mismatched checkpoint shapes by filtering matching-shaped params.
        Returns the loaded model or None if unavailable/failed.
        """
        if getattr(self, 'rescorer', None) is not None:
            return self.rescorer

        if not self.rescorer_path.exists():
            return None

        try:
            from src.models.relational_rescorer import RelationalRescorer
        except Exception as e:
            if os.environ.get('PPE_RESCORER_DEBUG') == '1':
                print(f"[PPE_RESCORER] import failed: {e}")
            return None

        try:
            # Inspect checkpoint to infer architecture (hidden size) if possible
            raw = torch.load(str(self.rescorer_path), map_location=self.device)
            state_for_arch = None
            if isinstance(raw, dict) and ('model_state_dict' in raw or 'state_dict' in raw):
                state_for_arch = raw.get('model_state_dict', raw.get('state_dict'))
            else:
                state_for_arch = raw

            inferred_hidden = None
            if isinstance(state_for_arch, dict):
                # common key from the MLP: 'net.0.weight' -> shape (hidden, node_feat_dim)
                for candidate in ['net.0.weight', 'net.0.weight_orig', 'net.0._packed_params']:
                    if candidate in state_for_arch:
                        try:
                            inferred_hidden = int(state_for_arch[candidate].size(0))
                            break
                        except Exception:
                            pass

            if inferred_hidden is not None:
                r = RelationalRescorer(node_feat_dim=6, hidden=inferred_hidden)
            else:
                r = RelationalRescorer(node_feat_dim=6)
            # raw remains loaded for state extraction below
            if isinstance(raw, dict) and ('model_state_dict' in raw or 'state_dict' in raw):
                state = raw.get('model_state_dict', raw.get('state_dict'))
            else:
                state = raw

            model_sd = r.state_dict()
            if isinstance(state, dict) and any(isinstance(v, torch.Tensor) for v in state.values()):
                filtered = {}
                loaded_keys = []
                for k, v in state.items():
                    if k in model_sd and isinstance(v, torch.Tensor) and v.size() == model_sd[k].size():
                        filtered[k] = v
                        loaded_keys.append(k)
                if len(filtered) > 0:
                    r.load_state_dict(filtered, strict=False)
                    if os.environ.get('PPE_RESCORER_DEBUG') == '1':
                        print(f"[PPE_RESCORER] loaded {len(loaded_keys)} matching params from checkpoint")
                else:
                    try:
                        r.load_state_dict(state, strict=False)
                    except Exception as e:
                        if os.environ.get('PPE_RESCORER_DEBUG') == '1':
                            print(f"[PPE_RESCORER] failed to load any params from checkpoint: {e}")
            else:
                try:
                    r.load_state_dict(state)
                except Exception:
                    try:
                        r.load_state_dict(state, strict=False)
                    except Exception as e:
                        if os.environ.get('PPE_RESCORER_DEBUG') == '1':
                            print(f"[PPE_RESCORER] failed to load checkpoint: {e}")

            try:
                r.to(self.device)
            except Exception:
                pass

            r.eval()
            self.rescorer = r
            return self.rescorer
        except Exception as e:
            if os.environ.get('PPE_RESCORER_DEBUG') == '1':
                print(f"[PPE_RESCORER] _load_rescorer_once failed: {e}")
            return None

    def _load_ground_truth(self, split='test'):
        split_file = self.data_dir / 'splits' / f'{split}.txt'
        annotations_dir = self.data_dir / 'annotations'
        
        if not split_file.exists():
            raise FileNotFoundError(f"Split file not found: {split_file}")
        
        with open(split_file, 'r') as f:
            test_images = [line.strip() for line in f.readlines()]
        
        ground_truth = {}
        
        for img_name in test_images:
            xml_path = annotations_dir / f"{img_name.split('.')[0]}.xml"
            json_path = annotations_dir / f"{img_name.split('.')[0]}.json"
            
            if xml_path.exists():
                ground_truth[img_name] = self._parse_xml_annotation(xml_path)
            elif json_path.exists():
                ground_truth[img_name] = self._parse_json_annotation(json_path)
            else:
                print(f"[WARN]  No annotation found for {img_name}")
        
        print(f"[OK] Loaded ground truth for {len(ground_truth)} images")
        return ground_truth

    def _parse_xml_annotation(self, xml_path):
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        annotations = []
        for obj in root.findall('object'):
            class_name = obj.find('name').text
            bbox = obj.find('bndbox')
            
            x1 = float(bbox.find('xmin').text)
            y1 = float(bbox.find('ymin').text)
            x2 = float(bbox.find('xmax').text)
            y2 = float(bbox.find('ymax').text)
            
            annotations.append({
                'class': class_name,
                'bbox': [x1, y1, x2, y2],
                'difficult': int(obj.find('difficult').text) if obj.find('difficult') is not None else 0
            })
        
        return annotations

    def _parse_json_annotation(self, json_path):
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        annotations = []
        for ann in data.get('annotations', []):
            bbox = ann['bbox']
            if len(bbox) == 4 and 'width' in str(ann):
                x1, y1, w, h = bbox
                bbox = [x1, y1, x1 + w, y1 + h]
            
            annotations.append({
                'class': ann['class'],
                'bbox': bbox,
                'difficult': ann.get('difficult', 0)
            })
        
        return annotations

    def _detect_image(self, image_path):
        image = Image.open(image_path).convert('RGB')
        original_size = image.size

        if getattr(self, 'model_type', 'ssd') == 'ssd':
            transform = transforms.Compose([
                transforms.Resize((300, 300)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ])
            input_tensor = transform(image).unsqueeze(0).to(self.device)
            with torch.no_grad():
                predictions = self.model(input_tensor)
                predicted_locs, predicted_scores = predictions
                det_boxes_batch, det_labels_batch, det_scores_batch = self.model.detect_objects(
                    predicted_locs, predicted_scores, 
                    min_score=self.conf_threshold, 
                    max_overlap=self.iou_threshold, 
                    top_k=200
                )
            det_boxes = det_boxes_batch[0]
            det_labels = det_labels_batch[0]
            det_scores = det_scores_batch[0]

            detections = []
            for i in range(len(det_boxes)):
                x1 = float(det_boxes[i][0] * original_size[0])
                y1 = float(det_boxes[i][1] * original_size[1])
                x2 = float(det_boxes[i][2] * original_size[0])
                y2 = float(det_boxes[i][3] * original_size[1])
                class_idx = int(det_labels[i])
                class_name = self.class_names[class_idx] if class_idx < len(self.class_names) else 'unknown'
                detections.append({
                    'class': class_name,
                    'bbox': [x1, y1, x2, y2],
                    'confidence': float(det_scores[i])
                })
            return detections

        # Faster R-CNN branch: torchvision detection API
        # Use a simpler transform: ToTensor (pixels in [0,1])
        transform = transforms.Compose([
            transforms.ToTensor()
        ])
        input_tensor = transform(image).to(self.device)
        with torch.no_grad():
            outputs = self.model([input_tensor])
        # outputs is a list of dicts with keys: boxes (N,4), labels (N), scores (N)
        out = outputs[0]
        boxes = out.get('boxes', [])
        labels = out.get('labels', [])
        scores = out.get('scores', [])

        detections = []
        for i in range(len(boxes)):
            bx = boxes[i].cpu().numpy().tolist()
            x1, y1, x2, y2 = [float(v) for v in bx]
            label_idx = int(labels[i].cpu().item())
            class_name = self.class_names[label_idx] if label_idx < len(self.class_names) else 'unknown'
            conf = float(scores[i].cpu().item())
            detections.append({'class': class_name, 'bbox': [x1, y1, x2, y2], 'confidence': conf})

        # Apply trained rescorer if available (cached loader)
        try:
            if len(detections) > 0:
                r = self._load_rescorer_once()
                if r is not None:
                    node_feats = []
                    scores_list = []
                    for d in detections:
                        x1, y1, x2, y2 = d['bbox']
                        cx = (x1 + x2) / 2.0 / original_size[0]
                        cy = (y1 + y2) / 2.0 / original_size[1]
                        w = max(0.0, (x2 - x1) / original_size[0])
                        h = max(0.0, (y2 - y1) / original_size[1])
                        area = w * h
                        s = float(d.get('confidence', 0.0))
                        node_feats.append([s, cx, cy, w, h, area])
                        scores_list.append(s)

                    node_feats_t = torch.tensor(node_feats, dtype=torch.float32, device=self.device)
                    scores_t = torch.tensor(scores_list, dtype=torch.float32, device=self.device)

                    with torch.no_grad():
                        try:
                            rescores = r(node_feats_t, None, None, scores_t)
                            # rescores are logits or probabilities depending on model type;
                            # try to coerce to probabilities if values outside [0,1]
                            rr_t = rescores
                            rr_np = rr_t.detach().cpu().numpy()
                            # if values appear to be logits (outside 0..1), apply sigmoid
                            if rr_np.min() < 0.0 or rr_np.max() > 1.0:
                                rr_t = torch.sigmoid(rr_t)
                                rr_np = rr_t.detach().cpu().numpy()

                            # multiplicative rescore
                            mult = (scores_t * rr_t)
                            # convex blend with alpha: new_conf = (1-alpha)*orig + alpha*mult
                            alpha = float(getattr(self, 'rescorer_alpha', 1.0))
                            blended_t = (1.0 - alpha) * scores_t + alpha * mult
                            blended = blended_t.cpu().numpy().tolist()
                            rr = rr_np
                            for det, new_s in zip(detections, blended):
                                det['confidence'] = float(new_s)
                            if os.environ.get('PPE_RESCORER_DEBUG') == '1':
                                import numpy as _np
                                print(f"[PPE_RESCORER] image={Path(image_path).name} rescore_min={_np.min(rr):.4f} mean={_np.mean(rr):.4f} max={_np.max(rr):.4f} shape={rr.shape}")
                                try:
                                    before = np.array(scores_list)
                                    after = np.array(blended)
                                    diffs = after - before
                                    print(f"[PPE_RESCORER] image={Path(image_path).name} conf_before_mean={before.mean():.4f} conf_after_mean={after.mean():.4f} diff_mean={diffs.mean():.4f}")
                                except Exception:
                                    pass
                        except Exception as e:
                            if os.environ.get('PPE_RESCORER_DEBUG') == '1':
                                print(f"[PPE_RESCORER] forward failed for {Path(image_path).name}: {e}")
        except Exception:
            pass

        return detections

    def _filter_ppe_by_person(self, detections, min_overlap=None):
        if min_overlap is None:
            min_overlap = self.person_overlap_threshold

        person_boxes = [d['bbox'] for d in detections if d['class'] == 'person']

        if not person_boxes:
            return [d for d in detections if d['class'] == 'person']

        filtered = []
        for d in detections:
            if d['class'] == 'person':
                filtered.append(d)
                continue

            thr = self.class_conf_thresholds.get(d['class'], self.conf_threshold)
            if d.get('confidence', 0.0) < thr:
                continue

            bx = d['bbox']
            keep = False
            for pb in person_boxes:
                x1 = max(bx[0], pb[0])
                y1 = max(bx[1], pb[1])
                x2 = min(bx[2], pb[2])
                y2 = min(bx[3], pb[3])
                if x2 <= x1 or y2 <= y1:
                    continue
                inter = (x2 - x1) * (y2 - y1)
                area1 = (bx[2] - bx[0]) * (bx[3] - bx[1])
                area2 = (pb[2] - pb[0]) * (pb[3] - pb[1])
                union = area1 + area2 - inter
                iou = inter / union if union > 0 else 0
                if iou >= min_overlap:
                    keep = True
                    break

            if keep:
                filtered.append(d)

        return filtered

    def _calculate_metrics(self, ground_truth, detections):
        class_metrics = {}
        
        for class_name in self.class_names[1:]:
            gt_boxes = [ann for ann in ground_truth if ann['class'] == class_name]
            det_boxes = [det for det in detections if det['class'] == class_name]
            ap = self._calculate_ap(gt_boxes, det_boxes)
            tp, fp, fn = self._calculate_tp_fp_fn(gt_boxes, det_boxes)
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            class_metrics[class_name] = {
                'ap': ap,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'tp': tp,
                'fp': fp,
                'fn': fn,
                'gt_count': len(gt_boxes),
                'det_count': len(det_boxes)
            }
        
        map_score = np.mean([metrics['ap'] for metrics in class_metrics.values()])
        
        return class_metrics, map_score

    def _calculate_ap(self, gt_boxes, det_boxes):
        if len(gt_boxes) == 0:
            return 0.0 if len(det_boxes) == 0 else 0.0
        
        if len(det_boxes) == 0:
            return 0.0
        
        det_boxes = sorted(det_boxes, key=lambda x: x['confidence'], reverse=True)
        tp = np.zeros(len(det_boxes))
        fp = np.zeros(len(det_boxes))
        matched_gt = set()
        
        for i, det in enumerate(det_boxes):
            best_iou = 0
            best_gt_idx = -1
            
            for j, gt in enumerate(gt_boxes):
                if j in matched_gt:
                    continue
                
                iou = calculate_iou(det['bbox'], gt['bbox'])
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = j
            
            if best_iou >= self.eval_iou_threshold:
                tp[i] = 1
                matched_gt.add(best_gt_idx)
            else:
                fp[i] = 1
        
        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)
        
        precision = tp_cumsum / (tp_cumsum + fp_cumsum)
        recall = tp_cumsum / len(gt_boxes)
        
        ap = 0.0
        for r in np.arange(0, 1.1, 0.1):
            p_values = precision[recall >= r]
            ap += (np.max(p_values) if len(p_values) > 0 else 0.0) / 11
        
        return ap

    def _calculate_tp_fp_fn(self, gt_boxes, det_boxes):
        if len(gt_boxes) == 0:
            return 0, len(det_boxes), 0
        if len(det_boxes) == 0:
            return 0, 0, len(gt_boxes)
        
        matched_gt = set()
        tp = 0
        
        for det in det_boxes:
            best_iou = 0
            best_gt_idx = -1
            
            for i, gt in enumerate(gt_boxes):
                if i in matched_gt:
                    continue
                
                iou = calculate_iou(det['bbox'], gt['bbox'])
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = i
            
            if best_iou >= self.eval_iou_threshold:
                tp += 1
                matched_gt.add(best_gt_idx)
        
        fp = len(det_boxes) - tp
        fn = len(gt_boxes) - tp
        
        return tp, fp, fn

    def _analyze_problem_cases(self, image_name, ground_truth, detections):
        gt_persons = [ann for ann in ground_truth if ann['class'] == 'person']
        det_persons = [det for det in detections if det['class'] == 'person']
        
        if len(gt_persons) > len(det_persons):
            self.results['problem_cases']['missed_workers'].append({
                'image': image_name,
                'gt_persons': len(gt_persons),
                'detected_persons': len(det_persons),
                'missed_count': len(gt_persons) - len(det_persons)
            })
        
        env_classes = ['hard_hat', 'safety_vest', 'safety_gloves']
        for class_name in env_classes:
            gt_items = [ann for ann in ground_truth if ann['class'] == class_name]
            det_items = [det for det in detections if det['class'] == class_name]
            
            if len(det_items) > len(gt_items) * 1.5:
                self.results['problem_cases']['false_positives'].append({
                    'image': image_name,
                    'class': class_name,
                    'gt_count': len(gt_items),
                    'det_count': len(det_items)
                })
        
        violation_classes = ['no_hard_hat', 'no_safety_vest']
        for class_name in violation_classes:
            gt_violations = [ann for ann in ground_truth if ann['class'] == class_name]
            det_violations = [det for det in detections if det['class'] == class_name]
            
            if len(gt_violations) > len(det_violations):
                self.results['problem_cases']['missed_violations'].append({
                    'image': image_name,
                    'class': class_name,
                    'missed_count': len(gt_violations) - len(det_violations)
                })

    def evaluate(self, split='test', max_images=None):
        print(f"[INFO] Starting PPE Detection Evaluation on {split} split")
        print(f"[INFO] Output directory: {self.output_dir}")

        ground_truth_data = self._load_ground_truth(split)

        items = list(ground_truth_data.items())
        if max_images is not None and isinstance(max_images, int) and max_images > 0:
            items = items[:max_images]

        all_gt = []
        all_detections = []

        for i, (img_name, gt_annotations) in enumerate(items):
            print(f"Processing {i+1}/{len(items)}: {img_name}", end='\r')
            
            img_path = self.data_dir / 'images' / img_name
            if not img_path.exists():
                print(f"[WARN]  Image not found: {img_path}")
                continue
            
            detections = self._detect_image(img_path)

            filtered_detections = self._filter_ppe_by_person(detections)

            self.results['detection_results'].append({
                'image': img_name,
                'ground_truth': gt_annotations,
                'raw_detections': detections,
                'detections': filtered_detections
            })

            self._analyze_problem_cases(img_name, gt_annotations, filtered_detections)

            all_gt.extend(gt_annotations)
            all_detections.extend(filtered_detections)

        print(f"\n[OK] Processed {len(items)} images")

        print("[INFO] Calculating metrics...")
        class_metrics, map_score = self._calculate_metrics(all_gt, all_detections)

        self.results['summary'] = {
            'total_images': len(items),
            'map_score': map_score,
            'evaluation_date': datetime.now().isoformat(),
            'model_path': str(self.model_path),
            'conf_threshold': self.conf_threshold,
            'iou_threshold': self.iou_threshold
        }

        self.results['per_class_metrics'] = class_metrics

        # Calculate label-only metrics (ignore bbox IoU; match by class presence)
        self.results['label_only_metrics'] = self._calculate_label_only_metrics()

        self._save_results()

        self._generate_visualizations()

        print(f"[DONE] Evaluation complete! Results saved to: {self.output_dir}")
        print(f"[RESULT] Overall mAP: {map_score:.3f}")

        return self.results

    def _save_results(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        try:
            self.results['effective_config'] = getattr(self, 'effective_config', {})
        except Exception:
            self.results['effective_config'] = {}

        results_file = self.output_dir / f'evaluation_results_{timestamp}.json'
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)

        try:
            config_file = self.output_dir / f'effective_config_{timestamp}.yaml'
            with open(config_file, 'w', encoding='utf-8') as cf:
                yaml.safe_dump(self.results.get('effective_config', {}), cf)
        except Exception:
            pass
        
        summary_data = []
        for class_name, metrics in self.results['per_class_metrics'].items():
            summary_data.append({
                'class': class_name,
                'ap': metrics['ap'],
                'precision': metrics['precision'],
                'recall': metrics['recall'],
                'f1': metrics['f1'],
                'gt_count': metrics['gt_count'],
                'det_count': metrics['det_count']
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_file = self.output_dir / f'class_metrics_{timestamp}.csv'
        summary_df.to_csv(summary_file, index=False)

        # Save label-only metrics CSV if available
        try:
            lom = self.results.get('label_only_metrics', {})
            if lom:
                lom_rows = []
                for cls, vals in lom.items():
                    lom_rows.append({'class': cls, 'gt_count': vals.get('gt_count', 0), 'detected_count': vals.get('detected_count', 0), 'recall': vals.get('recall', 0.0)})
                lom_df = pd.DataFrame(lom_rows)
                lom_file = self.output_dir / f'label_only_metrics_{timestamp}.csv'
                lom_df.to_csv(lom_file, index=False)
        except Exception:
            pass
        
        problems_file = self.output_dir / f'problem_analysis_{timestamp}.txt'
        with open(problems_file, 'w') as f:
            f.write("PPE Detection Problem Analysis\n")
            f.write("=" * 40 + "\n\n")
            
            f.write(f"Missed Workers: {len(self.results['problem_cases']['missed_workers'])} cases\n")
            f.write(f"False Positives: {len(self.results['problem_cases']['false_positives'])} cases\n")
            f.write(f"Missed Violations: {len(self.results['problem_cases']['missed_violations'])} cases\n\n")
            
            for category, cases in self.results['problem_cases'].items():
                f.write(f"\n{category.upper()}:\n")
                f.write("-" * 20 + "\n")
                for case in cases[:10]:
                    f.write(f"  {case}\n")
        
        print(f"[INFO] Results saved:")
        print(f"  - Complete results: {results_file}")
        print(f"  - Class metrics: {summary_file}")
        print(f"  - Problem analysis: {problems_file}")

        # Append a short worklog entry to the repo WORKLOG.md
        try:
            self._append_worklog_entry(results_file, summary_file, problems_file)
        except Exception:
            pass

    def _generate_visualizations(self):
        plt.style.use('seaborn-v0_8')
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        classes = list(self.results['per_class_metrics'].keys())
        aps = [self.results['per_class_metrics'][c]['ap'] for c in classes]
        f1s = [self.results['per_class_metrics'][c]['f1'] for c in classes]
        
        ax1.bar(classes, aps, color='skyblue', alpha=0.7)
        ax1.set_title('Average Precision by Class')
        ax1.set_ylabel('AP Score')
        ax1.tick_params(axis='x', rotation=45)
        
        ax2.bar(classes, f1s, color='lightcoral', alpha=0.7)
        ax2.set_title('F1 Score by Class')
        ax2.set_ylabel('F1 Score')
        ax2.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'class_performance.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        problem_counts = [
            len(self.results['problem_cases']['missed_workers']),
            len(self.results['problem_cases']['false_positives']),
            len(self.results['problem_cases']['missed_violations'])
        ]
        problem_labels = ['Missed Workers', 'False Positives', 'Missed Violations']
        
        ax.bar(problem_labels, problem_counts, color=['orange', 'red', 'purple'], alpha=0.7)
        ax.set_title('Problem Cases Summary')
        ax.set_ylabel('Number of Cases')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'problem_summary.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"[INFO] Visualizations saved to: {self.output_dir}")

    def _append_worklog_entry(self, results_file, summary_file, problems_file):
        """Append a short evaluation summary entry to the repository WORKLOG.md file.

        The entry includes timestamp, model, thresholds, mAP, counts and the three main output file paths.
        This is intentionally lightweight and tolerant of missing WORKLOG file.
        """
        try:
            repo_root = Path(__file__).resolve().parents[2]
            worklog_path = repo_root / 'WORKLOG.md'

            map_score = self.results.get('summary', {}).get('map_score')
            total_images = self.results.get('summary', {}).get('total_images')
            model_path = self.results.get('summary', {}).get('model_path', str(self.model_path))

            now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

            entry_lines = []
            entry_lines.append('\n')
            entry_lines.append(f"### Evaluation — {now}\n")
            entry_lines.append('\n')
            entry_lines.append(f"- Model: `{model_path}`\n")
            entry_lines.append(f"- Images evaluated: {total_images}\n")
            entry_lines.append(f"- Overall mAP: {map_score:.3f}" + "\n")
            entry_lines.append(f"- Conf threshold: {self.conf_threshold}\n")
            entry_lines.append(f"- Eval IoU threshold: {self.eval_iou_threshold}\n")
            entry_lines.append('\n')
            entry_lines.append("Saved files:\n")
            entry_lines.append(f"- Results JSON: `{results_file}`\n")
            entry_lines.append(f"- Class metrics CSV: `{summary_file}`\n")
            entry_lines.append(f"- Problem analysis: `{problems_file}`\n")
            entry_lines.append('\n')

            # Ensure WORKLOG exists, create if missing
            if not worklog_path.exists():
                with open(worklog_path, 'w', encoding='utf-8') as wf:
                    wf.write('# Project Worklog — Image-Classification-for-PPE\n\n')

            with open(worklog_path, 'a', encoding='utf-8') as wf:
                wf.writelines(entry_lines)

            print(f"[INFO] WORKLOG updated: {worklog_path}")
        except Exception as e:
            print(f"[WARN] Failed to append WORKLOG entry: {e}")

    def _calculate_label_only_metrics(self):
        """Compute simple label-only recall per class: for each ground-truth instance, check if any detection of the same class
        exists in the same image (using class-specific confidence thresholds). Ignore bbox IoU entirely.
        Returns a dict {class: {gt_count, detected_count, recall}}."""

        lom = {}
        # Initialize counts
        for cls in self.class_names[1:]:
            lom[cls] = {'gt_count': 0, 'detected_count': 0}

        for item in self.results.get('detection_results', []):
            image = item.get('image')
            gts = item.get('ground_truth', [])
            dets = item.get('detections', [])

            # build set of detected classes in this image that pass class-specific confidence thresholds
            detected_classes = set()
            for d in dets:
                cls = d.get('class')
                conf = d.get('confidence', 0.0)
                thr = self.class_conf_thresholds.get(cls, self.conf_threshold)
                if conf >= thr:
                    detected_classes.add(cls)

            # for each GT instance, increment gt_count and mark detected_count if class is present in detected_classes
            for g in gts:
                cls = g.get('class')
                if cls not in lom:
                    # unseen class; create entry
                    lom[cls] = {'gt_count': 0, 'detected_count': 0}
                lom[cls]['gt_count'] += 1
                if cls in detected_classes:
                    lom[cls]['detected_count'] += 1

        # compute recall
        for cls, vals in lom.items():
            gt = vals.get('gt_count', 0)
            detd = vals.get('detected_count', 0)
            vals['recall'] = (detd / gt) if gt > 0 else None

        return lom


def main():
    parser = argparse.ArgumentParser(description='Evaluate PPE Detection Performance')
    parser.add_argument('--model_path', type=str, required=True, 
                      help='Path to trained model checkpoint')
    parser.add_argument('--data_dir', type=str, default='data',
                      help='Path to dataset directory')
    parser.add_argument('--config_path', type=str, default='configs/ppe_config.yaml',
                      help='Path to config file')
    parser.add_argument('--output_dir', type=str, default='outputs/evaluation_results',
                      help='Output directory for results')
    parser.add_argument('--split', type=str, default='test', choices=['train', 'val', 'test'],
                      help='Dataset split to evaluate on')
    parser.add_argument('--max_images', type=int, default=None,
                      help='Limit number of images to process (for quick runs)')
    parser.add_argument('--rescorer_alpha', type=float, default=1.0,
                      help='Blending weight for rescorer (0.0..1.0). 1.0 uses multiplicative rescore fully.')
    
    args = parser.parse_args()
    
    evaluator = PPEDetectionEvaluator(
        model_path=args.model_path,
        data_dir=args.data_dir,
        config_path=args.config_path,
        output_dir=args.output_dir,
        rescorer_alpha=args.rescorer_alpha
    )
    
    results = evaluator.evaluate(split=args.split, max_images=args.max_images)
    
    print("\n" + "="*50)
    print("EVALUATION SUMMARY")
    print("="*50)
    print(f"Overall mAP: {results['summary']['map_score']:.3f}")
    print(f"Total images: {results['summary']['total_images']}")
    print(f"[WARN]  Problem cases:")
    print(f"   - Missed workers: {len(results['problem_cases']['missed_workers'])}")
    print(f"   - False positives: {len(results['problem_cases']['false_positives'])}")
    print(f"   - Missed violations: {len(results['problem_cases']['missed_violations'])}")


if __name__ == "__main__":
    main()
