#!/usr/bin/env python3
"""
Visual Test - Show model predictions on test images
"""

import os
import sys
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import numpy as np

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.models.ssd import SSD300
from src.dataset.ppe_dataset import PPEDataset

class PPEVisualTester:
    def __init__(self, model_path):
        self.device = torch.device('cpu')
        self.num_classes = 12
        
        # PPE Class mapping
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
        
        # Colors for each class
        self.colors = [
            'red', 'blue', 'green', 'orange', 'purple', 'brown',
            'pink', 'gray', 'olive', 'cyan', 'magenta', 'yellow'
        ]
        
        # Load model
        self.model = self.load_model(model_path)
    
    def load_model(self, model_path):
        """Load trained model"""
        model = SSD300(n_classes=self.num_classes)
        
        checkpoint = torch.load(model_path, map_location=self.device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
            
        model.to(self.device)
        model.eval()
        print(f"✅ Model loaded successfully")
        return model
    
    def visualize_predictions(self, num_images=6, confidence_threshold=0.1):
        """Visualize model predictions on test images"""
        
        # Load test dataset
        dataset = PPEDataset(
            data_dir="data",
            split='test',
            transforms=None,
            img_size=300
        )
        
        print(f"📊 Loaded {len(dataset)} test images")
        
        # Create figure
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        with torch.no_grad():
            for i in range(min(num_images, len(dataset))):
                image, annotations, filename = dataset[i]
                
                # Prepare image for model
                image_batch = image.unsqueeze(0).to(self.device)
                
                # Get predictions
                predictions = self.model(image_batch)
                detections = self.process_predictions(predictions, confidence_threshold)
                
                # Convert image for display
                img_np = image.permute(1, 2, 0).numpy()
                img_np = (img_np * 255).astype(np.uint8)
                
                # Plot
                ax = axes[i]
                ax.imshow(img_np)
                ax.set_title(f"{filename}\nDetections: {len(detections)}")
                ax.axis('off')
                
                # Draw ground truth (green boxes)
                if isinstance(annotations, dict) and 'bboxes' in annotations:
                    gt_boxes = annotations['bboxes']
                    gt_labels = annotations['labels']
                    
                    for box, label in zip(gt_boxes, gt_labels):
                        x1, y1, x2, y2 = box * 300  # Scale to image size
                        width, height = x2 - x1, y2 - y1
                        
                        rect = patches.Rectangle(
                            (x1, y1), width, height,
                            linewidth=2, edgecolor='green', facecolor='none',
                            linestyle='--'
                        )
                        ax.add_patch(rect)
                        
                        # Add ground truth label
                        ax.text(x1, y1-5, f"GT: {self.class_names[label.item()]}", 
                               color='green', fontsize=8, weight='bold')
                
                # Draw predictions (red boxes)
                for det in detections:
                    bbox = det['bbox']
                    # Convert from center-size to corner format if needed
                    x1, y1, x2, y2 = bbox
                    
                    # Scale to image size (assuming bbox is normalized)
                    if max(bbox) <= 1.0:
                        x1, y1, x2, y2 = [coord * 300 for coord in bbox]
                    
                    width, height = x2 - x1, y2 - y1
                    
                    color = self.colors[det['class_id'] % len(self.colors)]
                    rect = patches.Rectangle(
                        (x1, y1), width, height,
                        linewidth=2, edgecolor=color, facecolor='none'
                    )
                    ax.add_patch(rect)
                    
                    # Add prediction label
                    ax.text(x1, y2+10, 
                           f"{det['class_name']}: {det['confidence']:.2f}", 
                           color=color, fontsize=8, weight='bold')
                
                print(f"  Image {i+1}: {filename} - {len(detections)} detections")
                if detections:
                    for det in detections[:3]:  # Show first 3
                        print(f"    - {det['class_name']}: {det['confidence']:.3f}")
        
        plt.tight_layout()
        plt.savefig('models/test_predictions_visual.png', dpi=150, bbox_inches='tight')
        plt.show()
        print(f"💾 Visualization saved to: models/test_predictions_visual.png")
    
    def process_predictions(self, predictions, confidence_threshold=0.1):
        """Process model predictions to extract detections"""
        if isinstance(predictions, tuple) and len(predictions) == 2:
            locs, confs = predictions
        else:
            return []
        
        # Remove batch dimension
        if locs.dim() == 3:
            locs = locs.squeeze(0)  # [8096, 4]
        if confs.dim() == 3:
            confs = confs.squeeze(0)  # [8096, 12]
        
        # Apply softmax to confidence scores
        confs = F.softmax(confs, dim=1)
        
        detections = []
        
        # For each class (skip background class 0)
        for class_id in range(1, self.num_classes):
            class_scores = confs[:, class_id]
            
            # Filter by confidence threshold
            valid_indices = class_scores > confidence_threshold
            
            if valid_indices.sum() > 0:
                valid_scores = class_scores[valid_indices]
                valid_locs = locs[valid_indices]
                
                # Sort by confidence
                sorted_indices = torch.argsort(valid_scores, descending=True)
                
                # Take top detections to avoid clutter
                for idx in sorted_indices[:10]:  # Top 10 per class
                    score = valid_scores[idx]
                    loc = valid_locs[idx]
                    
                    detections.append({
                        'class_id': class_id,
                        'class_name': self.class_names[class_id],
                        'confidence': score.item(),
                        'bbox': loc.tolist()
                    })
        
        # Sort all detections by confidence
        detections.sort(key=lambda x: x['confidence'], reverse=True)
        
        return detections[:20]  # Return top 20 detections

def main():
    print("🔍 PPE Visual Testing")
    print("=" * 40)
    
    model_path = "models/best_model_regularized.pth"
    
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        return
    
    tester = PPEVisualTester(model_path)
    
    # Test with very low confidence threshold to see everything
    print("\n🖼️ Generating visual predictions with low confidence threshold...")
    tester.visualize_predictions(num_images=6, confidence_threshold=0.05)

if __name__ == "__main__":
    main()
