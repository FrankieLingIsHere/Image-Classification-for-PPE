#!/usr/bin/env python3
"""
PPE Detection Model Testing Script
"""

import os
import sys
import argparse
import torch
import torch.nn.functional as F
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.models.ssd import SSD300

class PPEInferenceEngine:
    """PPE Detection Inference Engine"""
    
    def __init__(self, model_path, device='auto'):
        """Initialize inference engine"""
        
        self.device = torch.device('cuda' if torch.cuda.is_available() and device == 'auto' else 'cpu')
        print("Using device: {}".format(self.device))
        
        # PPE class names (matching your training config)
        self.class_names = {
            0: 'background',
            1: 'person',
            2: 'hard_hat',
            3: 'safety_vest',
            4: 'safety_gloves',
            5: 'safety_boots',
            6: 'eye_protection',
            7: 'no_hard_hat',
            8: 'no_safety_vest',
            9: 'no_safety_gloves',
            10: 'no_safety_boots',
            11: 'no_eye_protection',
            12: 'unknown'  # Additional class
        }
        
        # Load model
        self.model = self._load_model(model_path)
        
    def _load_model(self, model_path):
        """Load the trained model"""
        print("Loading model from: {}".format(model_path))
        
        # Initialize model
        model = SSD300(n_classes=len(self.class_names))
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        model.to(self.device)
        model.eval()
        
        print("Model loaded successfully!")
        print("Trained for {} epochs with final loss: {:.4f}".format(
            checkpoint.get('epoch', 'unknown'), 
            checkpoint.get('loss', 0)
        ))
        
        return model
    
    def preprocess_image(self, image_path, target_size=300):
        """Preprocess image for inference"""
        
        # Load image
        if isinstance(image_path, str):
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image = image_path
        
        original_image = image.copy()
        
        # Resize image
        image = cv2.resize(image, (target_size, target_size))
        
        # Convert to tensor and normalize
        image = image.astype(np.float32) / 255.0
        
        # ImageNet normalization
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        image = (image - mean) / std
        
        # Convert to tensor
        image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float()
        
        return image.to(self.device), original_image
    
    def postprocess_detections(self, predictions, confidence_threshold=0.3, nms_threshold=0.5):
        """Post-process model predictions"""
        
        # predictions is a tuple of (loc_preds, class_preds)
        loc_preds, class_preds = predictions
        
        # Apply softmax to class predictions
        class_probs = F.softmax(class_preds, dim=2)
        
        batch_size = class_probs.size(0)
        detections = []
        
        for i in range(batch_size):
            # Get predictions for this image
            boxes = loc_preds[i]  # [num_priors, 4]
            scores = class_probs[i]  # [num_priors, num_classes]
            
            # Process each class (skip background class 0)
            for class_id in range(1, scores.size(1)):
                class_scores = scores[:, class_id]
                
                # Filter by confidence threshold
                mask = class_scores > confidence_threshold
                if not mask.any():
                    continue
                
                # Get filtered boxes and scores
                filtered_boxes = boxes[mask]
                filtered_scores = class_scores[mask]
                
                # Apply NMS
                keep_indices = self._apply_nms(filtered_boxes, filtered_scores, nms_threshold)
                
                # Add detections
                for idx in keep_indices:
                    detection = {
                        'bbox': filtered_boxes[idx].cpu().numpy(),
                        'confidence': filtered_scores[idx].item(),
                        'class_id': class_id,
                        'class_name': self.class_names[class_id]
                    }
                    detections.append(detection)
        
        # Sort by confidence
        detections.sort(key=lambda x: x['confidence'], reverse=True)
        
        return detections
    
    def _apply_nms(self, boxes, scores, threshold):
        """Apply Non-Maximum Suppression"""
        if len(boxes) == 0:
            return []
        
        # Convert to corner format if needed
        if boxes.size(1) == 4:
            # Assume format is [cx, cy, w, h], convert to [x1, y1, x2, y2]
            x1 = boxes[:, 0] - boxes[:, 2] / 2
            y1 = boxes[:, 1] - boxes[:, 3] / 2
            x2 = boxes[:, 0] + boxes[:, 2] / 2
            y2 = boxes[:, 1] + boxes[:, 3] / 2
            boxes = torch.stack([x1, y1, x2, y2], dim=1)
        
        # Simple NMS implementation
        keep = []
        indices = torch.argsort(scores, descending=True)
        
        while len(indices) > 0:
            current = indices[0]
            keep.append(current.item())
            
            if len(indices) == 1:
                break
            
            # Calculate IoU with remaining boxes
            current_box = boxes[current].unsqueeze(0)
            remaining_boxes = boxes[indices[1:]]
            
            ious = self._calculate_iou(current_box, remaining_boxes)
            
            # Keep boxes with IoU less than threshold
            indices = indices[1:][ious < threshold]
        
        return keep
    
    def _calculate_iou(self, box1, boxes2):
        """Calculate IoU between one box and multiple boxes"""
        # box1: [1, 4], boxes2: [N, 4]
        # Format: [x1, y1, x2, y2]
        
        # Calculate intersection
        x1 = torch.max(box1[:, 0].unsqueeze(1), boxes2[:, 0].unsqueeze(0))
        y1 = torch.max(box1[:, 1].unsqueeze(1), boxes2[:, 1].unsqueeze(0))
        x2 = torch.min(box1[:, 2].unsqueeze(1), boxes2[:, 2].unsqueeze(0))
        y2 = torch.min(box1[:, 3].unsqueeze(1), boxes2[:, 3].unsqueeze(0))
        
        intersection = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)
        
        # Calculate areas
        area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
        area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
        
        # Calculate union
        union = area1.unsqueeze(1) + area2.unsqueeze(0) - intersection
        
        # Calculate IoU
        iou = intersection / (union + 1e-6)
        
        return iou.squeeze(0)
    
    def predict(self, image_path, confidence_threshold=0.3):
        """Run inference on an image"""
        
        # Preprocess image
        input_tensor, original_image = self.preprocess_image(image_path)
        
        # Run inference
        with torch.no_grad():
            predictions = self.model(input_tensor)
        
        # Post-process predictions
        detections = self.postprocess_detections(predictions, confidence_threshold)
        
        return detections, original_image
    
    def visualize_predictions(self, image, detections, output_path=None):
        """Visualize predictions on image"""
        
        fig, ax = plt.subplots(1, figsize=(12, 8))
        ax.imshow(image)
        
        # Define colors for different classes
        colors = {
            'person': 'blue',
            'hard_hat': 'green',
            'safety_vest': 'orange',
            'safety_gloves': 'purple',
            'safety_boots': 'brown',
            'eye_protection': 'pink',
            'no_hard_hat': 'red',
            'no_safety_vest': 'red',
            'no_safety_gloves': 'red',
            'no_safety_boots': 'red',
            'no_eye_protection': 'red'
        }
        
        height, width = image.shape[:2]
        
        for detection in detections:
            bbox = detection['bbox']
            class_name = detection['class_name']
            confidence = detection['confidence']
            
            # Convert normalized coordinates to pixel coordinates
            # Assuming bbox format is [cx, cy, w, h] normalized
            cx, cy, w, h = bbox
            x1 = (cx - w/2) * width
            y1 = (cy - h/2) * height
            box_width = w * width
            box_height = h * height
            
            # Create rectangle
            color = colors.get(class_name, 'yellow')
            rect = patches.Rectangle(
                (x1, y1), box_width, box_height,
                linewidth=2, edgecolor=color, facecolor='none'
            )
            ax.add_patch(rect)
            
            # Add label
            label = "{}: {:.2f}".format(class_name, confidence)
            ax.text(x1, y1-5, label, fontsize=10, color=color, 
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.7))
        
        ax.set_title("PPE Detection Results")
        ax.axis('off')
        
        if output_path:
            plt.savefig(output_path, bbox_inches='tight', dpi=150)
            print("Results saved to: {}".format(output_path))
        else:
            plt.show()
        
        plt.close()
    
    def analyze_safety_compliance(self, detections):
        """Analyze safety compliance from detections"""
        
        violations = []
        compliant_items = []
        
        for detection in detections:
            class_name = detection['class_name']
            
            if class_name.startswith('no_'):
                violations.append({
                    'type': class_name.replace('no_', ''),
                    'confidence': detection['confidence']
                })
            elif class_name != 'person' and class_name != 'background':
                compliant_items.append({
                    'type': class_name,
                    'confidence': detection['confidence']
                })
        
        # Calculate compliance score
        total_ppe_items = len(violations) + len(compliant_items)
        compliance_score = len(compliant_items) / total_ppe_items * 100 if total_ppe_items > 0 else 100
        
        return {
            'violations': violations,
            'compliant_items': compliant_items,
            'compliance_score': compliance_score,
            'is_safe': len(violations) == 0
        }

def main():
    parser = argparse.ArgumentParser(description='Test PPE Detection Model')
    parser.add_argument('--model', type=str, default='models/checkpoint_epoch_4.pth',
                       help='Path to trained model checkpoint')
    parser.add_argument('--image', type=str, required=True,
                       help='Path to test image or directory')
    parser.add_argument('--confidence', type=float, default=0.3,
                       help='Confidence threshold for detection')
    parser.add_argument('--output', type=str, default=None,
                       help='Output path for visualization')
    
    args = parser.parse_args()
    
    # Check if model exists
    if not os.path.exists(args.model):
        print("Model not found: {}".format(args.model))
        return
    
    # Find image path
    image_path = args.image
    if os.path.isdir(args.image):
        # Find first image in directory
        image_files = [f for f in os.listdir(args.image) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        if image_files:
            image_path = os.path.join(args.image, image_files[0])
            print("Using image: {}".format(image_path))
        else:
            print("No images found in directory: {}".format(args.image))
            return
    
    # Initialize inference engine
    print("\n🦺 PPE Detection Testing")
    print("=" * 40)
    
    inference_engine = PPEInferenceEngine(args.model)
    
    # Run inference
    print("\nRunning inference on: {}".format(os.path.basename(image_path)))
    detections, original_image = inference_engine.predict(image_path, args.confidence)
    
    # Print results
    print("\n📊 Detection Results:")
    print("Found {} objects".format(len(detections)))
    
    for i, detection in enumerate(detections, 1):
        print("{}. {}: {:.2f}%".format(
            i, detection['class_name'], detection['confidence'] * 100
        ))
    
    # Analyze safety compliance
    safety_analysis = inference_engine.analyze_safety_compliance(detections)
    
    print("\n🛡️ Safety Analysis:")
    print("Compliance Score: {:.1f}%".format(safety_analysis['compliance_score']))
    print("Status: {}".format("✅ SAFE" if safety_analysis['is_safe'] else "⚠️ VIOLATIONS DETECTED"))
    
    if safety_analysis['violations']:
        print("\nViolations:")
        for violation in safety_analysis['violations']:
            print("- Missing {}: {:.1f}% confidence".format(
                violation['type'], violation['confidence'] * 100
            ))
    
    if safety_analysis['compliant_items']:
        print("\nCompliant PPE:")
        for item in safety_analysis['compliant_items']:
            print("- {}: {:.1f}% confidence".format(
                item['type'], item['confidence'] * 100
            ))
    
    # Visualize results
    output_path = args.output
    if output_path is None:
        output_path = "test_result.jpg"
    
    inference_engine.visualize_predictions(original_image, detections, output_path)
    
    print("\n✅ Testing completed!")

if __name__ == "__main__":
    main()