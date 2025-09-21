#!/usr/bin/env python3
"""
Inference script for SSD PPE Detection Model

This script performs inference on images to detect Personal Protective Equipment (PPE)
and assess OSHA compliance in construction environments.
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

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from models.ssd import build_ssd_model
from utils.utils import (
    visualize_detections, check_ppe_compliance, 
    generate_compliance_report, load_checkpoint
)


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='PPE Detection Inference')
    
    # Model arguments
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
    model = build_ssd_model(num_classes=num_classes)
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    
    print(f"Model loaded from {model_path}")
    return model


def preprocess_image(image_path, img_size=300):
    """Preprocess image for inference"""
    # Read image
    image = read_image(image_path)
    
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
    
    # Check PPE compliance
    compliance = check_ppe_compliance(boxes_np, labels_np, scores_np, class_names, args.conf_threshold)
    results['compliance'] = compliance
    
    # Generate output file paths
    image_name = Path(image_path).stem
    output_image_path = os.path.join(args.output_dir, f"{image_name}_detection.jpg")
    output_report_path = os.path.join(args.output_dir, f"{image_name}_report.txt")
    output_json_path = os.path.join(args.output_dir, f"{image_name}_results.json")
    
    # Save visualization
    if args.save_images and len(boxes_np) > 0:
        # Convert original image for visualization
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
    
    # Save compliance report
    if args.save_reports:
        report_text = generate_compliance_report(compliance, output_report_path)
        print(f"Compliance Report for {image_name}:")
        print(report_text)
    
    # Save JSON results
    if args.save_json:
        with open(output_json_path, 'w') as f:
            json.dump(results, f, indent=2)
    
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
    
    # Save summary results
    summary_path = os.path.join(args.output_dir, "summary_results.json")
    with open(summary_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Generate summary report
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
    
    # Setup device
    device = setup_device(args.device)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model
    model = load_model(args.model_path, args.num_classes, device)
    
    # Define class names (should match training)
    class_names = [
        'background',     # 0
        'person',         # 1
        'hard_hat',       # 2
        'safety_vest',    # 3
        'safety_gloves',  # 4
        'safety_boots',   # 5
        'eye_protection', # 6
        'no_hard_hat',    # 7
        'no_safety_vest'  # 8
    ]
    
    # Process input
    input_path = Path(args.input)
    
    if input_path.is_file():
        # Process single image
        results = process_single_image(input_path, model, device, args, class_names)
        print(f"Detection completed for {input_path}")
        
    elif input_path.is_dir():
        # Process directory
        results = process_directory(input_path, model, device, args, class_names)
        print(f"Detection completed for {len(results)} images in {input_path}")
        
    else:
        print(f"Error: {input_path} is not a valid file or directory")
        return
    
    print(f"Results saved to: {args.output_dir}")


if __name__ == '__main__':
    main()