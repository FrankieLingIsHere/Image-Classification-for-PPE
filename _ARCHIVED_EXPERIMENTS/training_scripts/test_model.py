#!/usr/bin/env python3
"""
Test PPE Detection Model on Single Images
"""

import os
import sys
import argparse
import torch
import torchvision.transforms as transforms
from PIL import Image, ImageDraw, ImageFont
import numpy as np

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.models.ssd import SSD300

class PPEInferenceEngine:
    """PPE Detection Inference Engine"""
    
    def __init__(self, model_path, device='auto'):
        """Initialize inference engine"""
        
        self.device = torch.device('cuda' if torch.cuda.is_available() and device == 'auto' else 'cpu')
        print(f"Using device: {self.device}")
        
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
            11: 'no_eye_protection'
        }
        
        # Load model
        self.model = self._load_model(model_path)
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((300, 300)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    def _load_model(self, model_path):
        """Load trained model"""
        
        print(f"Loading model from {model_path}")
        
        # Initialize model
        model = SSD300(n_classes=13)  # Updated to match your config
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            epoch = checkpoint.get('epoch', 'unknown')
            loss = checkpoint.get('loss', 'unknown')
            print(f"  Loaded checkpoint from epoch {epoch}, loss: {loss}")
        else:
            model.load_state_dict(checkpoint)
            print(f"  Loaded model weights")
        
        model.to(self.device)
        model.eval()
        
        return model
    
    def predict(self, image_path, confidence_threshold=0.5, nms_threshold=0.5):
        """Run inference on single image"""
        
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        original_size = image.size
        
        # Transform for model
        input_tensor = self.transform(image)
        if isinstance(input_tensor, Image.Image):
            # If transform didn't work properly, convert manually
            input_tensor = transforms.ToTensor()(image)
        input_tensor = input_tensor.unsqueeze(0).to(self.device)
        
        # Run inference
        with torch.no_grad():
            predictions = self.model(input_tensor)
            
            if isinstance(predictions, tuple):
                predicted_locs, predicted_scores = predictions
            else:
                predicted_locs = predictions['boxes']
                predicted_scores = predictions['scores']
        
        # Post-process predictions
        detections = self._post_process(predicted_locs, predicted_scores, 
                                      confidence_threshold, nms_threshold,
                                      original_size)
        
        return detections, image
    
    def _post_process(self, predicted_locs, predicted_scores, conf_threshold, nms_threshold, original_size):
        """Post-process model predictions"""
        
        # Convert to numpy for easier processing
        if len(predicted_scores.shape) == 3:
            # Shape: [batch, n_priors, n_classes]
            scores = predicted_scores[0]  # Remove batch dimension
            boxes = predicted_locs[0]     # Remove batch dimension
        else:
            scores = predicted_scores
            boxes = predicted_locs
        
        # Apply softmax to get probabilities
        scores = torch.softmax(scores, dim=1)
        
        detections = []
        
        # For each class (skip background class 0)
        for class_id in range(1, scores.shape[1]):
            class_scores = scores[:, class_id]
            
            # Filter by confidence threshold
            mask = class_scores > conf_threshold
            if not mask.any():
                continue
            
            class_boxes = boxes[mask]
            class_scores = class_scores[mask]
            
            # Convert boxes to original image coordinates
            # Note: This is simplified - real SSD needs proper decode
            class_boxes = self._decode_boxes(class_boxes, original_size)
            
            # Simple NMS (you might want to use torchvision.ops.nms for better results)
            keep_indices = self._simple_nms(class_boxes, class_scores, nms_threshold)
            
            for idx in keep_indices:
                detections.append({
                    'class_id': class_id,
                    'class_name': self.class_names[class_id],
                    'confidence': class_scores[idx].item(),
                    'bbox': class_boxes[idx].tolist()
                })
        
        # Sort by confidence
        detections.sort(key=lambda x: x['confidence'], reverse=True)
        
        return detections
    
    def _decode_boxes(self, boxes, original_size):
        """Convert normalized boxes to pixel coordinates"""
        
        # Simple conversion (this is simplified - real SSD uses prior box decoding)
        width, height = original_size
        
        # Assuming boxes are in format [cx, cy, w, h] normalized
        decoded_boxes = boxes.clone()
        
        # Convert center-size to corner format
        decoded_boxes[:, 0] = (boxes[:, 0] - boxes[:, 2]/2) * width   # x_min
        decoded_boxes[:, 1] = (boxes[:, 1] - boxes[:, 3]/2) * height  # y_min  
        decoded_boxes[:, 2] = (boxes[:, 0] + boxes[:, 2]/2) * width   # x_max
        decoded_boxes[:, 3] = (boxes[:, 1] + boxes[:, 3]/2) * height  # y_max
        
        # Clamp to image bounds
        decoded_boxes[:, 0] = torch.clamp(decoded_boxes[:, 0], 0, width)
        decoded_boxes[:, 1] = torch.clamp(decoded_boxes[:, 1], 0, height)
        decoded_boxes[:, 2] = torch.clamp(decoded_boxes[:, 2], 0, width)
        decoded_boxes[:, 3] = torch.clamp(decoded_boxes[:, 3], 0, height)
        
        return decoded_boxes
    
    def _simple_nms(self, boxes, scores, threshold):
        """Simple Non-Maximum Suppression"""
        
        if len(boxes) == 0:
            return []
        
        # Sort by score
        sorted_indices = torch.argsort(scores, descending=True)
        keep = []
        
        while len(sorted_indices) > 0:
            # Keep the highest scoring box
            current_idx = sorted_indices[0]
            keep.append(current_idx.item())
            
            if len(sorted_indices) == 1:
                break
            
            # Calculate IoU with remaining boxes
            current_box = boxes[current_idx]
            remaining_boxes = boxes[sorted_indices[1:]]
            
            ious = self._calculate_iou(current_box.unsqueeze(0), remaining_boxes)
            
            # Keep boxes with IoU below threshold
            mask = ious < threshold
            sorted_indices = sorted_indices[1:][mask]
        
        return keep
    
    def _calculate_iou(self, box1, boxes2):
        """Calculate IoU between one box and multiple boxes"""
        
        # Get intersection coordinates
        x1 = torch.max(box1[:, 0], boxes2[:, 0])
        y1 = torch.max(box1[:, 1], boxes2[:, 1])
        x2 = torch.min(box1[:, 2], boxes2[:, 2])
        y2 = torch.min(box1[:, 3], boxes2[:, 3])
        
        # Calculate intersection area
        intersection = torch.clamp(x2 - x1, 0) * torch.clamp(y2 - y1, 0)
        
        # Calculate areas
        area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
        area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
        
        # Calculate IoU
        union = area1 + area2 - intersection
        iou = intersection / (union + 1e-8)
        
        return iou
    
    def visualize_predictions(self, image, detections, save_path=None):
        """Visualize predictions on image"""
        
        # Create a copy for drawing
        vis_image = image.copy()
        draw = ImageDraw.Draw(vis_image)
        
        # Try to load a font
        try:
            font = ImageFont.truetype("arial.ttf", 20)
        except:
            font = ImageFont.load_default()
        
        # Color map for different classes
        colors = {
            'person': '#FF0000',
            'hard_hat': '#00FF00',
            'safety_vest': '#0000FF', 
            'safety_gloves': '#FFFF00',
            'safety_boots': '#FF00FF',
            'eye_protection': '#00FFFF',
            'no_hard_hat': '#FF4444',
            'no_safety_vest': '#FF8888',
            'no_safety_gloves': '#FFAAAA',
            'no_safety_boots': '#FFCCCC',
            'no_eye_protection': '#FFDDDD'
        }
        
        for detection in detections:
            bbox = detection['bbox']
            class_name = detection['class_name']
            confidence = detection['confidence']
            
            # Get color
            color = colors.get(class_name, '#FFFFFF')
            
            # Draw bounding box
            draw.rectangle(bbox, outline=color, width=3)
            
            # Draw label
            label = f"{class_name}: {confidence:.2f}"
            
            # Get text size
            text_bbox = draw.textbbox((0, 0), label, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
            
            # Draw label background
            label_bbox = [bbox[0], bbox[1] - text_height - 5, 
                         bbox[0] + text_width + 5, bbox[1]]
            draw.rectangle(label_bbox, fill=color)
            
            # Draw text
            draw.text((bbox[0] + 2, bbox[1] - text_height - 3), 
                     label, fill='white', font=font)
        
        if save_path:
            vis_image.save(save_path)
            print(f"Visualization saved to {save_path}")
        
        return vis_image


def main():
    parser = argparse.ArgumentParser(description='Test PPE Detection Model')
    parser.add_argument('--model', type=str, default='models/checkpoint_epoch_4.pth',
                       help='Path to trained model checkpoint')
    parser.add_argument('--image', type=str, required=True,
                       help='Path to test image')
    parser.add_argument('--output', type=str, default='test_result.jpg',
                       help='Path to save visualization')
    parser.add_argument('--confidence', type=float, default=0.3,
                       help='Confidence threshold')
    parser.add_argument('--nms', type=float, default=0.5,
                       help='NMS threshold')
    
    args = parser.parse_args()
    
    # Check if model exists
    if not os.path.exists(args.model):
        print(f"❌ Model not found: {args.model}")
        print("Available models:")
        models_dir = 'models'
        if os.path.exists(models_dir):
            for f in os.listdir(models_dir):
                if f.endswith('.pth'):
                    print(f"  {os.path.join(models_dir, f)}")
        return
    
    # Check if image exists
    if not os.path.exists(args.image):
        print(f"❌ Image not found: {args.image}")
        return
    
    print("🔍 PPE Detection Inference")
    print("=" * 40)
    
    # Initialize inference engine
    inference_engine = PPEInferenceEngine(args.model)
    
    # Run inference
    print(f"Testing on: {args.image}")
    detections, original_image = inference_engine.predict(
        args.image, 
        confidence_threshold=args.confidence,
        nms_threshold=args.nms
    )
    
    # Print results
    print(f"\n📋 Detection Results ({len(detections)} objects found):")
    print("-" * 60)
    
    if detections:
        for i, det in enumerate(detections):
            print(f"{i+1}. {det['class_name']}: {det['confidence']:.3f} "
                  f"at [{det['bbox'][0]:.0f}, {det['bbox'][1]:.0f}, "
                  f"{det['bbox'][2]:.0f}, {det['bbox'][3]:.0f}]")
        
        # Check for safety violations
        violations = [d for d in detections if d['class_name'].startswith('no_')]
        
        print(f"\n🦺 Safety Analysis:")
        if violations:
            print(f"  ⚠️ {len(violations)} SAFETY VIOLATIONS detected:")
            for violation in violations:
                print(f"    • {violation['class_name'].replace('no_', 'Missing ')} "
                      f"(confidence: {violation['confidence']:.3f})")
        else:
            print(f"  ✅ No safety violations detected")
        
        # Generate visualization
        vis_image = inference_engine.visualize_predictions(
            original_image, detections, args.output
        )
        
    else:
        print("  No objects detected above confidence threshold")
    
    print(f"\n✅ Inference completed!")


if __name__ == "__main__":
    main()