#!/usr/bin/env python3
"""
Inference script for SSD PPE Detection Model (moved to scripts/eval)

This is a copy of the top-level `scripts/inference.py` adapted so imports
work when run from `scripts/eval/`.
"""

import os
import sys
import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torchvision.io import read_image
from torchvision.transforms import v2 as transforms
import cv2
import numpy as np
from PIL import Image

# Ensure repo root is on sys.path (two levels up from scripts/eval)
sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.models.ssd import build_ssd_model
from src.models.hybrid_ppe_model import HybridPPEModel
from src.utils.utils import (
    visualize_detections, check_ppe_compliance, 
    generate_compliance_report, load_checkpoint
)

import torch
from torchvision.ops import nms as torch_nms


def _to_serializable(obj):
    """Recursively convert numpy types to python builtins for JSON serialization."""
    import numpy as _np

    if isinstance(obj, dict):
        return {k: _to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_serializable(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_to_serializable(v) for v in obj)
    if isinstance(obj, _np.ndarray):
        return obj.tolist()
    if isinstance(obj, (_np.integer,)):
        return int(obj)
    if isinstance(obj, (_np.floating,)):
        return float(obj)
    if obj is None:
        return None
    return obj


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='PPE Detection Inference')
    
    # Model arguments
    parser.add_argument('--model_type', type=str, default='ssd', choices=['ssd', 'hybrid'],
                       help='Type of model to use (ssd or hybrid)')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--num_classes', type=int, default=9,
                       help='Number of classes including background')
    parser.add_argument('--img_size', type=int, default=300,
                       help='Input image size')
    
    # Input arguments
    parser.add_argument('--input', type=str, required=True,
                       help='Path to input image or directory')
    parser.add_argument('--output_dir', type=str, default='results',
                       help='Directory to save results')
    
    # Detection arguments
    parser.add_argument('--conf_threshold', type=float, default=0.5,
                       help='Confidence threshold for detections')
    parser.add_argument('--nms_threshold', type=float, default=0.45,
                       help='NMS threshold for overlapping boxes')
    parser.add_argument('--top_k', type=int, default=200,
                       help='Maximum number of detections per image')
    parser.add_argument('--final_conf_min', type=float, default=0.4,
                       help='If set, any detection with confidence >= this will be kept regardless of person overlap')
    
    # Output arguments
    parser.add_argument('--save_images', action='store_true',
                       help='Save images with detection visualizations')
    parser.add_argument('--save_reports', action='store_true',
                       help='Save compliance reports')
    parser.add_argument('--save_json', action='store_true',
                       help='Save detection results as JSON')
    
    # Other arguments
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (auto, cpu, cuda)')

    parser.add_argument('--config_path', type=str, default='configs/best_runtime_config.yaml',
                       help='Optional YAML config to override thresholds')
    
    return parser.parse_args()


def setup_device(device_arg):
    """Setup device for inference"""
    if device_arg == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device_arg)
    
    print(f"Using device: {device}")
    return device


def load_model(model_path, num_classes, device):
    """Load trained model"""
    # Try to infer number of classes from checkpoint if available
    inferred_num_classes = num_classes
    if os.path.exists(model_path):
        ckpt = torch.load(model_path, map_location='cpu')
        if 'model_state_dict' in ckpt:
            state_dict = ckpt['model_state_dict']
        else:
            state_dict = ckpt

        # Check for classification conv weight to infer classes
        key = 'pred_convs.cl_conv4_3.weight'
        if key in state_dict:
            cl_weight_shape = state_dict[key].shape[0]
            # conv4_3 typically has 4 anchors
            inferred_num_classes = cl_weight_shape // 4
            print(f"Detected {inferred_num_classes} classes from checkpoint")

    model = build_ssd_model(num_classes=inferred_num_classes)

    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model = model.to(device)
    model.eval()

    print(f"Model loaded from {model_path}")
    return model, inferred_num_classes


def preprocess_image(image_path, img_size=300):
    """Preprocess image for inference"""
    # Read image
    image = read_image(image_path)

    # read_image may return images with 1 (grayscale), 3 (RGB) or 4 (RGBA) channels.
    # Normalize to 3-channel RGB by dropping alpha or repeating grayscale channel.
    if image.ndim == 3 and image.shape[0] == 4:
        # Drop alpha channel
        image = image[:3, :, :]
    elif image.ndim == 3 and image.shape[0] == 1:
        # Grayscale -> repeat to 3 channels
        image = image.repeat(3, 1, 1)
    
    # Define transforms
    transform = transforms.Compose([
        transforms.Resize(size=(img_size, img_size)),
        transforms.ToPureTensor(),
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Apply transforms
    image_tensor = transform(image)
    
    return image_tensor, image


def postprocess_detections(predicted_locs, predicted_scores, model, conf_threshold, nms_threshold, top_k):
    """Post-process model predictions to get final detections"""
    # Apply detection logic from the model
    det_boxes, det_labels, det_scores = model.detect_objects(
        predicted_locs, predicted_scores,
        min_score=conf_threshold,
        max_overlap=nms_threshold,
        top_k=top_k
    )
    
    return det_boxes[0], det_labels[0], det_scores[0]


def detect_ppe(model, image_tensor, device, conf_threshold=0.5, nms_threshold=0.45, top_k=200):
    """Detect PPE in a single image"""
    # Add batch dimension
    image_batch = image_tensor.unsqueeze(0).to(device)
    
    with torch.no_grad():
        # Forward pass
        predicted_locs, predicted_scores = model(image_batch)
        
        # Post-process detections
        boxes, labels, scores = postprocess_detections(
            predicted_locs, predicted_scores, model,
            conf_threshold, nms_threshold, top_k
        )
    
    return boxes, labels, scores


def process_single_image_hybrid(image_path, model, device, args, class_names):
    """Process a single image using hybrid model"""
    print(f"Processing with hybrid model: {image_path}")
    
    # Load image with OpenCV for hybrid model
    import cv2
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Error: Could not load image {image_path}")
        return {}
    
    # Analyze image using hybrid model
    results = model.analyze_image(image)
    
    # Extract PPE detections
    ppe_analysis = results.get('ppe_analysis', {})
    detections = ppe_analysis.get('detections', [])
    
    # Create results dictionary in expected format
    results_formatted = {
        'image_path': str(image_path),
        'detections': detections,
        'num_detections': len(detections),
        'description': results.get('visual_description', ''),
        'safety_analysis': results.get('safety_analysis', ''),
        'compliance': ppe_analysis.get('compliance_status', {})
    }
    
    # Generate output file paths
    base_name = image_path.stem
    output_base = os.path.join(args.output_dir, base_name)
    
    # Save visualization if requested
    if args.save_images and len(detections) > 0:
        # Create visualization using OpenCV
        vis_image = image.copy()
        for det in detections:
            bbox = det['bbox']
            x1, y1, x2, y2 = bbox
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{det['class']} ({det['confidence']:.2f})"
            cv2.putText(vis_image, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        vis_path = f"{output_base}_detection.jpg"
        cv2.imwrite(vis_path, vis_image)
        print(f"Saved visualization: {vis_path}")
    
    # Save JSON if requested
    if args.save_json:
        json_path = f"{output_base}_results.json"
        with open(json_path, 'w') as f:
            import json
            json.dump(results_formatted, f, indent=2)
        print(f"Saved JSON: {json_path}")
    
    # Save report if requested
    if args.save_reports:
        report_path = f"{output_base}_report.txt"
        with open(report_path, 'w') as f:
            f.write(f"PPE Detection Report\n")
            f.write(f"Image: {image_path}\n")
            f.write(f"Description: {results_formatted['description']}\n\n")
            f.write(f"Safety Analysis: {results_formatted['safety_analysis']}\n\n")
            f.write(f"Detections ({len(detections)}):\n")
            for i, det in enumerate(detections, 1):
                f.write(f"{i}. {det['class']} (confidence: {det['confidence']:.3f})\n")
        print(f"Saved report: {report_path}")
    
    return results_formatted


def process_single_image(image_path, model, device, args, class_names):
    """Process a single image"""
    print(f"Processing: {image_path}")
    
    # Preprocess image
    image_tensor, original_image = preprocess_image(image_path, args.img_size)
    
    # Detect PPE
    boxes, labels, scores = detect_ppe(
        model, image_tensor, device,
        args.conf_threshold, args.nms_threshold, args.top_k
    )
    
    # Convert tensors to numpy for further processing
    if len(boxes) > 0:
        boxes_np = boxes.cpu().numpy()
        labels_np = labels.cpu().numpy()
        scores_np = scores.cpu().numpy()
    else:
        boxes_np = np.array([])
        labels_np = np.array([])
        scores_np = np.array([])
    
    # Create results dictionary
    results = {
        'image_path': str(image_path),
        'detections': [],
        'num_detections': len(boxes_np)
    }

    # Store detection results
    for i in range(len(boxes_np)):
        detection = {
            'class_id': int(labels_np[i]),
            'class_name': class_names[labels_np[i]],
            'confidence': float(scores_np[i]),
            'bbox': boxes_np[i].tolist()
        }
        results['detections'].append(detection)

    # Apply optional person-first & per-class confidence filtering if provided in args
    if hasattr(args, 'class_conf_thresholds') and isinstance(args.class_conf_thresholds, dict):
        person_overlap = getattr(args, 'person_overlap_threshold', 0.0)

        # Extract person boxes
        person_boxes = [d['bbox'] for d in results['detections'] if d['class_name'] == 'person']

        if person_boxes:
            filtered = []
            # Decide which classes should require overlap with a person
            person_dependent = set(['hard_hat', 'safety_vest', 'eye_protection', 'no_hard_hat', 'no_safety_vest'])
            person_dependent = set([c for c in person_dependent if c in class_names])

            for d in results['detections']:
                    if d['class_name'] == 'person':
                        thr = args.class_conf_thresholds.get('person', args.conf_threshold)
                        if d['confidence'] >= thr:
                            filtered.append(d)
                        continue

                    thr = args.class_conf_thresholds.get(d['class_name'], args.conf_threshold)
                    if d['confidence'] < thr:
                        continue

                    if person_overlap > 0.0 and d['class_name'] in person_dependent:
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
                            if iou >= person_overlap:
                                keep = True
                                break

                        if keep:
                            filtered.append(d)
                    else:
                        filtered.append(d)

            results['detections'] = filtered
            results['num_detections'] = len(filtered)

            if len(filtered) > 0:
                boxes_np = np.array([f['bbox'] for f in filtered])
                labels_np = np.array([f['class_id'] for f in filtered])
                scores_np = np.array([f['confidence'] for f in filtered])
            else:
                boxes_np = np.array([])
                labels_np = np.array([])
                scores_np = np.array([])
    
    # Check PPE compliance
    compliance = check_ppe_compliance(boxes_np, labels_np, scores_np, class_names, args.conf_threshold)
    results['compliance'] = compliance
    
    # Generate output file paths
    image_name = Path(image_path).stem
    output_image_path = os.path.join(args.output_dir, f"{image_name}_detection.jpg")
    output_report_path = os.path.join(args.output_dir, f"{image_name}_report.txt")
    output_json_path = os.path.join(args.output_dir, f"{image_name}_results.json")
    raw_image_path = os.path.join(args.output_dir, f"{image_name}_raw_detection.jpg")
    raw_json_path = os.path.join(args.output_dir, f"{image_name}_raw_results.json")
    filtered_image_path = os.path.join(args.output_dir, f"{image_name}_filtered_detection.jpg")
    filtered_json_path = os.path.join(args.output_dir, f"{image_name}_filtered_results.json")
    
    # Save visualization
    if args.save_images and len(boxes_np) > 0 and getattr(args, 'post_nms_iou', 0.0) == 0.0:
        if original_image.shape[0] == 3:  # CHW format
            vis_image = original_image.permute(1, 2, 0).numpy()
        else:
            vis_image = original_image.numpy()
        
        if vis_image.max() > 1:
            vis_image = vis_image / 255.0

        visualize_detections(
            vis_image, boxes_np, labels_np, scores_np, class_names,
            threshold=args.conf_threshold, save_path=output_image_path,
            title=f"PPE Detection - {image_name}"
        )
    else:
        if args.save_images and len(boxes_np) > 0 and getattr(args, 'post_nms_iou', 0.0) > 0.0:
            if original_image.shape[0] == 3:
                vis_image = original_image.permute(1, 2, 0).numpy()
            else:
                vis_image = original_image.numpy()
            if vis_image.max() > 1:
                vis_image = vis_image / 255.0
            visualize_detections(
                vis_image, boxes_np, labels_np, scores_np, class_names,
                threshold=args.conf_threshold, save_path=raw_image_path,
                title=f"PPE Detection - {image_name} (raw)"
            )
            print(f"Raw visualization saved to {raw_image_path}")
    
    # Save compliance report
    if args.save_reports:
        report_text = generate_compliance_report(compliance, output_report_path)
        print(f"Compliance Report for {image_name}:")
        print(report_text)
    
    # Save JSON results
    if args.save_json:
        if getattr(args, 'post_nms_iou', 0.0) > 0.0:
            with open(raw_json_path, 'w', encoding='utf-8') as f:
                json.dump(_to_serializable(results), f, indent=2)
            print(f"Raw JSON saved to {raw_json_path}")
        else:
            with open(output_json_path, 'w', encoding='utf-8') as f:
                json.dump(_to_serializable(results), f, indent=2)
    
    return results


def _apply_per_class_nms(detections, iou_thresh=0.3, top_k=200, per_class_iou=None, max_per_class=None, final_conf_min=None):
    """Apply per-class NMS on a list of detection dicts."""
    if not detections:
        return []

    kept = []
    from collections import defaultdict
    cls_map = defaultdict(list)
    for i, d in enumerate(detections):
        cls_map[d['class_name']].append(i)

    for cls, idxs in cls_map.items():
        if len(idxs) == 0:
            continue
        boxes = []
        scores = []
        for i in idxs:
            bx = detections[i]['bbox']
            boxes.append([float(bx[0]), float(bx[1]), float(bx[2]), float(bx[3])])
            scores.append(float(detections[i]['confidence']))

        boxes_t = torch.tensor(boxes, dtype=torch.float32)
        scores_t = torch.tensor(scores, dtype=torch.float32)

        if boxes_t.numel() == 0:
            continue

        cls_iou = iou_thresh
        if per_class_iou and cls in per_class_iou:
            try:
                cls_iou = float(per_class_iou[cls])
            except Exception:
                pass

        keep_idxs = torch_nms(boxes_t, scores_t, cls_iou)
        if keep_idxs.numel() == 0:
            continue

        keep_idxs = keep_idxs.tolist()
        keep_idxs = sorted(keep_idxs, key=lambda k: scores[k], reverse=True)

        high_conf_local = set()
        try:
            if final_conf_min is not None:
                for local_i, s in enumerate(scores):
                    if float(s) >= float(final_conf_min):
                        high_conf_local.add(local_i)
        except Exception:
            high_conf_local = set()

        cls_max = top_k
        if max_per_class and cls in max_per_class:
            try:
                cls_max = int(max_per_class[cls])
            except Exception:
                cls_max = top_k

        final_keep = []
        for local_i in sorted(list(high_conf_local), key=lambda i: scores[i], reverse=True):
            final_keep.append(idxs[local_i])

        for local_k in keep_idxs:
            global_idx = idxs[local_k]
            if global_idx in final_keep:
                continue
            if len(final_keep) >= cls_max:
                break
            final_keep.append(global_idx)

        for gidx in final_keep:
            kept.append(detections[gidx])

    return kept


def _postprocess_persons(results, image_path, args, class_names):
    """Person-specific postprocessing"""
    dets = results.get('detections', [])
    person_idxs = [i for i, d in enumerate(dets) if d['class_name'] == 'person']
    if not person_idxs:
        return results

    try:
        from PIL import Image
        im = Image.open(str(image_path))
        W, H = im.size
    except Exception:
        W = H = None

    filtered_idxs = []
    for i in person_idxs:
        d = dets[i]
        conf = d['confidence']
        bx = d['bbox']
        if W and H:
            bxw = (bx[2] - bx[0])
            bxh = (bx[3] - bx[1])
            if 0.0 <= bx[0] <= 1.0 and 0.0 <= bx[1] <= 1.0 and 0.0 <= bx[2] <= 1.0 and 0.0 <= bx[3] <= 1.0:
                area_frac = float(max(0.0, bxw * bxh))
            else:
                area_frac = float(max(0.0, (bxw * bxh) / float(W * H)))
        else:
            area_frac = 0.0

        if getattr(args, 'person_conf_min', None) is not None and conf < args.person_conf_min:
            continue
        if area_frac < getattr(args, 'person_area_min_frac', 0.0):
            continue
        if getattr(args, 'person_area_max_frac', 1.0) and area_frac > args.person_area_max_frac:
            continue
        filtered_idxs.append(i)

    persons = [dets[i] for i in filtered_idxs]
    if not persons:
        results['detections'] = [d for d in dets if d['class_name'] != 'person']
        return results

    per_iou = getattr(args, 'person_merge_iou', 0.3)
    boxes = [[p['bbox'][0], p['bbox'][1], p['bbox'][2], p['bbox'][3]] for p in persons]
    scores = [p['confidence'] for p in persons]
    if len(boxes) > 0:
        boxes_t = torch.tensor(boxes, dtype=torch.float32)
        scores_t = torch.tensor(scores, dtype=torch.float32)
        keep = torch_nms(boxes_t, scores_t, per_iou).tolist()
    else:
        keep = []

    kept_persons = [persons[k] for k in keep]

    final_max = getattr(args, 'person_final_max', None)
    if final_max is not None:
        kept_persons = sorted(kept_persons, key=lambda x: x['confidence'], reverse=True)[:int(final_max)]

    others = [d for d in dets if d['class_name'] != 'person']
    new_dets = others + kept_persons
    results['detections'] = new_dets
    return results


def process_directory(input_dir, model, device, args, class_names):
    """Process all images in a directory"""
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    
    image_files = []
    for ext in image_extensions:
        image_files.extend(Path(input_dir).glob(f"*{ext}"))
        image_files.extend(Path(input_dir).glob(f"*{ext.upper()}"))
    
    if not image_files:
        print(f"No images found in {input_dir}")
        return []
    
    print(f"Found {len(image_files)} images to process")
    
    all_results = []
    for image_file in image_files:
        try:
            results = process_single_image(image_file, model, device, args, class_names)
            all_results.append(results)
        except Exception as e:
            print(f"Error processing {image_file}: {e}")
            continue
    
    summary_path = os.path.join(args.output_dir, "summary_results.json")
    with open(summary_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    compliant_count = sum(1 for r in all_results if r['compliance']['compliant'])
    total_count = len(all_results)
    
    summary_report = f"""
PPE DETECTION SUMMARY REPORT
============================

Total Images Processed: {total_count}
Compliant Images: {compliant_count}
Non-Compliant Images: {total_count - compliant_count}
Compliance Rate: {compliant_count / total_count * 100:.1f}%

Detailed results saved to: {summary_path}
"""
    
    print(summary_report)
    
    summary_report_path = os.path.join(args.output_dir, "summary_report.txt")
    with open(summary_report_path, 'w') as f:
        f.write(summary_report)
    
    return all_results


def main():
    """Main inference function"""
    args = parse_args()
    
    device = setup_device(args.device)
    
    os.makedirs(args.output_dir, exist_ok=True)

    if args.config_path and os.path.exists(args.config_path):
        try:
            import yaml
            with open(args.config_path, 'r', encoding='utf-8') as cf:
                cfg = yaml.safe_load(cf)
            if 'class_conf_thresholds' in cfg:
                args.class_conf_thresholds = cfg['class_conf_thresholds']
            else:
                args.class_conf_thresholds = {}

            args.person_overlap_threshold = cfg.get('person_overlap_threshold', 0.0)
            if 'conf_threshold' in cfg:
                args.conf_threshold = cfg['conf_threshold']
            if 'nms_threshold' in cfg:
                args.nms_threshold = cfg['nms_threshold']
            if 'iou_threshold' in cfg:
                args.nms_threshold = cfg['iou_threshold']
            args.post_nms_iou = cfg.get('post_nms_iou', 0.0)
            args.class_post_nms_iou = cfg.get('class_post_nms_iou', {})
            args.max_detections_per_class = cfg.get('max_detections_per_class', {})
            args.person_conf_min = cfg.get('person_conf_min', None)
            args.person_area_min_frac = cfg.get('person_area_min_frac', 0.001)
            args.person_area_max_frac = cfg.get('person_area_max_frac', 0.6)
            args.person_merge_iou = cfg.get('person_merge_iou', 0.3)
            args.person_final_max = cfg.get('person_final_max', None)
        except Exception as e:
            print(f"\u26a0\ufe0f Failed to load config {args.config_path}: {e}")
            args.class_conf_thresholds = {}
            args.person_overlap_threshold = 0.0
    
    if args.model_type == 'hybrid':
        model = HybridPPEModel(
            ppe_model_path=args.model_path,
            device=str(device)
        )
        process_func = process_single_image_hybrid
        inferred_classes = args.num_classes
    else:
        model, inferred_classes = load_model(args.model_path, args.num_classes, device)
        process_func = process_single_image
    
    # Load canonical class names from configs at repo root
    class_file = Path(__file__).resolve().parents[2] / 'configs' / 'dataset_class_names.txt'
    if class_file.exists():
        with open(class_file, 'r', encoding='utf-8') as cf:
            cls = [ln.strip() for ln in cf.readlines() if ln.strip()]
        class_names = cls
    else:
        class_names = [
            'background', 'person', 'hard_hat', 'safety_vest',
            'safety_gloves', 'safety_boots', 'eye_protection',
            'no_hard_hat', 'no_safety_vest'
        ]

    if inferred_classes > len(class_names):
        class_names = class_names + [f'class_{i}' for i in range(len(class_names), inferred_classes)]
    else:
        class_names = class_names[:inferred_classes]
    
    input_path = Path(args.input)
    
    if input_path.is_file():
        results = process_func(input_path, model, device, args, class_names)
        if getattr(args, 'post_nms_iou', 0.0) and results.get('detections'):
            before_cnt = len(results['detections'])
            results['detections'] = _apply_per_class_nms(
                results['detections'],
                iou_thresh=args.post_nms_iou,
                per_class_iou=getattr(args, 'class_post_nms_iou', None),
                max_per_class=getattr(args, 'max_detections_per_class', None),
                top_k=getattr(args, 'top_k', 200)
            )
            results['num_detections'] = len(results['detections'])
            after_cnt = results['num_detections']
            try:
                results = _postprocess_persons(results, input_path, args, class_names)
                results['num_detections'] = len(results.get('detections', []))
            except Exception as e:
                print(f"Warning: person postprocessing failed: {e}")
            print(f"Per-class NMS applied: detections before={before_cnt}, after={after_cnt}")

            base_name = input_path.stem
            output_json_path = os.path.join(args.output_dir, f"{base_name}_results.json")
            with open(output_json_path, 'w', encoding='utf-8') as f:
                json.dump(_to_serializable(results), f, indent=2)

            if args.save_images and results.get('detections'):
                boxes_np = np.array([d['bbox'] for d in results['detections']])
                labels_np = np.array([d['class_id'] for d in results['detections']])
                scores_np = np.array([d['confidence'] for d in results['detections']])

                from torchvision.io import read_image
                img_tensor = read_image(str(input_path))
                if img_tensor.max() > 1:
                    vis_img = img_tensor.permute(1,2,0).numpy() / 255.0
                else:
                    vis_img = img_tensor.permute(1,2,0).numpy()

                output_image_path = os.path.join(args.output_dir, f"{base_name}_detection.jpg")
                visualize_detections(
                    vis_img, boxes_np, labels_np, scores_np, class_names,
                    threshold=args.conf_threshold, save_path=output_image_path,
                    title=f"PPE Detection - {base_name}"
                )
        print(f"Detection completed for {input_path}")
        
    elif input_path.is_dir():
        if args.model_type == 'hybrid':
            print("Directory processing not implemented for hybrid model yet")
            return
        results = process_directory(input_path, model, device, args, class_names)
        print(f"Detection completed for {len(results)} images in {input_path}")
        
    else:
        print(f"Error: {input_path} is not a valid file or directory")
        return
    
    print(f"Results saved to: {args.output_dir}")


if __name__ == '__main__':
    main()
