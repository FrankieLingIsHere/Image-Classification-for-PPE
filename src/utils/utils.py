import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import List, Tuple, Dict
import os


def save_checkpoint(state, filename='checkpoint.pth'):
    """Save model checkpoint"""
    torch.save(state, filename)
    print(f"Checkpoint saved to {filename}")


def load_checkpoint(checkpoint_path, model, optimizer=None):
    """Load model checkpoint"""
    if os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        
        start_epoch = checkpoint.get('epoch', 0)
        best_loss = checkpoint.get('best_loss', float('inf'))
        
        if optimizer and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        print(f"Checkpoint loaded from {checkpoint_path} (epoch {start_epoch})")
        return start_epoch, best_loss
    else:
        print(f"No checkpoint found at {checkpoint_path}")
        return 0, float('inf')


def calculate_map(det_boxes, det_labels, det_scores, true_boxes, true_labels, n_classes):
    """
    Calculate mean Average Precision (mAP) for object detection
    
    Args:
        det_boxes: list of detected boxes for each image
        det_labels: list of detected labels for each image  
        det_scores: list of detected scores for each image
        true_boxes: list of ground truth boxes for each image
        true_labels: list of ground truth labels for each image
        n_classes: number of classes
    
    Returns:
        mAP score
    """
    # Store all detections and ground truths
    all_detections = []
    all_annotations = []
    
    for i in range(len(det_boxes)):
        # Detections for this image
        if len(det_boxes[i]) > 0:
            for j in range(len(det_boxes[i])):
                all_detections.append({
                    'image_id': i,
                    'bbox': det_boxes[i][j].cpu().numpy(),
                    'score': det_scores[i][j].cpu().item(),
                    'class': det_labels[i][j].cpu().item()
                })
        
        # Ground truth for this image
        if len(true_boxes[i]) > 0:
            for j in range(len(true_boxes[i])):
                all_annotations.append({
                    'image_id': i,
                    'bbox': true_boxes[i][j].cpu().numpy(),
                    'class': true_labels[i][j].cpu().item(),
                    'difficult': False
                })
    
    # Calculate AP for each class
    aps = []
    for c in range(1, n_classes):  # Skip background class
        # Get detections and annotations for this class
        class_detections = [d for d in all_detections if d['class'] == c]
        class_annotations = [a for a in all_annotations if a['class'] == c]
        
        if len(class_annotations) == 0:
            continue
            
        # Sort detections by confidence
        class_detections.sort(key=lambda x: x['score'], reverse=True)
        
        # Track which annotations have been matched
        matched = [False] * len(class_annotations)
        
        tp = []
        fp = []
        
        for detection in class_detections:
            # Find best matching annotation
            best_iou = 0
            best_idx = -1
            
            for i, annotation in enumerate(class_annotations):
                if annotation['image_id'] != detection['image_id']:
                    continue
                
                # Calculate IoU
                iou = calculate_iou(detection['bbox'], annotation['bbox'])
                if iou > best_iou:
                    best_iou = iou
                    best_idx = i
            
            # Check if match is good enough and not already matched
            if best_iou >= 0.5 and not matched[best_idx]:
                tp.append(1)
                fp.append(0)
                matched[best_idx] = True
            else:
                tp.append(0)
                fp.append(1)
        
        # Calculate precision and recall
        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)
        
        recalls = tp_cumsum / len(class_annotations)
        precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-8)
        
        # Calculate AP using 11-point interpolation
        ap = 0
        for t in np.arange(0, 1.1, 0.1):
            if np.sum(recalls >= t) == 0:
                p = 0
            else:
                p = np.max(precisions[recalls >= t])
            ap += p / 11
        
        aps.append(ap)
    
    if len(aps) == 0:
        return 0.0
    
    return np.mean(aps)


def calculate_iou(box1, box2):
    """Calculate Intersection over Union (IoU) of two bounding boxes"""
    # Determine the coordinates of the intersection rectangle
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    if x2 <= x1 or y2 <= y1:
        return 0.0
    
    # Calculate area of intersection
    intersection = (x2 - x1) * (y2 - y1)
    
    # Calculate area of both bounding boxes
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    # Calculate IoU
    union = area1 + area2 - intersection
    return intersection / union if union > 0 else 0.0


def visualize_detections(image, boxes, labels, scores, class_names, 
                        threshold=0.5, save_path=None, title="PPE Detection"):
    """
    Visualize detection results on an image
    
    Args:
        image: input image (numpy array or torch tensor)
        boxes: detected bounding boxes
        labels: detected class labels
        scores: detection scores
        class_names: list of class names
        threshold: minimum score threshold for visualization
        save_path: path to save the visualization
        title: title for the plot
    """
    # Convert tensor to numpy if needed
    if torch.is_tensor(image):
        if image.shape[0] == 3:  # CHW format
            image = image.permute(1, 2, 0)
        image = image.cpu().numpy()
    
    # Ensure image is in [0, 1] range
    if image.max() > 1:
        image = image / 255.0
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.imshow(image)
    ax.set_title(title, fontsize=16)
    
    # Define colors for different classes
    colors = ['red', 'blue', 'green', 'yellow', 'orange', 'purple', 'cyan', 'magenta', 'brown']
    
    # Draw bounding boxes
    for i in range(len(boxes)):
        if scores[i] < threshold:
            continue
            
        box = boxes[i]
        label = labels[i]
        score = scores[i]
        
        # Convert relative coordinates to absolute if needed
        if box.max() <= 1:
            h, w = image.shape[:2]
            box = [box[0] * w, box[1] * h, box[2] * w, box[3] * h]
        
        # Create rectangle
        x, y, x2, y2 = box
        width = x2 - x
        height = y2 - y
        
        color = colors[label % len(colors)]
        rect = patches.Rectangle((x, y), width, height, 
                               linewidth=2, edgecolor=color, facecolor='none')
        ax.add_patch(rect)
        
        # Add label
        class_name = class_names[label] if label < len(class_names) else f"Class {label}"
        ax.text(x, y - 5, f"{class_name}: {score:.2f}", 
               bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.7),
               fontsize=10, color='white', weight='bold')
    
    ax.axis('off')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to {save_path}")
    
    plt.show()


def check_ppe_compliance(boxes, labels, scores, class_names, threshold=0.5):
    """
    Check PPE compliance based on detections
    
    Args:
        boxes: detected bounding boxes
        labels: detected class labels  
        scores: detection scores
        class_names: list of class names
        threshold: minimum score threshold
    
    Returns:
        dict with compliance information
    """
    compliance_report = {
        'compliant': True,
        'violations': [],
        'detected_ppe': [],
        'workers_count': 0,
        'severity': 'low'
    }
    
    # Filter detections by threshold
    valid_detections = [(box, label, score) for box, label, score in zip(boxes, labels, scores) 
                       if score >= threshold]
    
    # Count workers and PPE
    workers = 0
    hard_hats = 0
    safety_vests = 0
    violations = []
    
    for box, label, score in valid_detections:
        class_name = class_names[label] if label < len(class_names) else f"Class {label}"
        
        if 'person' in class_name.lower():
            workers += 1
        elif 'hard_hat' in class_name.lower():
            hard_hats += 1
        elif 'safety_vest' in class_name.lower():
            safety_vests += 1
        elif 'no_hard_hat' in class_name.lower():
            violations.append(f"Worker without hard hat (Score: {score:.2f})")
            compliance_report['severity'] = 'high'
        elif 'no_safety_vest' in class_name.lower():
            violations.append(f"Worker without safety vest (Score: {score:.2f})")
            compliance_report['severity'] = 'high'
        
        compliance_report['detected_ppe'].append({
            'class': class_name,
            'score': score,
            'bbox': box
        })
    
    compliance_report['workers_count'] = workers
    compliance_report['violations'] = violations
    
    # Determine overall compliance
    if violations:
        compliance_report['compliant'] = False
    elif workers > 0 and (hard_hats == 0 or safety_vests == 0):
        compliance_report['compliant'] = False
        compliance_report['violations'].append("Insufficient PPE detection - workers may be missing required equipment")
        compliance_report['severity'] = 'medium'
    
    return compliance_report


def generate_compliance_report(compliance_data, save_path=None):
    """
    Generate a detailed compliance report
    
    Args:
        compliance_data: compliance information from check_ppe_compliance
        save_path: path to save the report
    
    Returns:
        formatted report string
    """
    report = []
    report.append("="*60)
    report.append("          PPE COMPLIANCE REPORT")
    report.append("="*60)
    report.append("")
    
    # Overall status
    status = "✅ COMPLIANT" if compliance_data['compliant'] else "❌ NON-COMPLIANT"
    report.append(f"Status: {status}")
    report.append(f"Severity: {compliance_data['severity'].upper()}")
    report.append(f"Workers Detected: {compliance_data['workers_count']}")
    report.append("")
    
    # Violations
    if compliance_data['violations']:
        report.append("VIOLATIONS DETECTED:")
        report.append("-" * 30)
        for i, violation in enumerate(compliance_data['violations'], 1):
            report.append(f"{i}. {violation}")
        report.append("")
    
    # Detected PPE
    if compliance_data['detected_ppe']:
        report.append("DETECTED PPE EQUIPMENT:")
        report.append("-" * 30)
        ppe_counts = {}
        for ppe in compliance_data['detected_ppe']:
            class_name = ppe['class']
            ppe_counts[class_name] = ppe_counts.get(class_name, 0) + 1
        
        for ppe_type, count in ppe_counts.items():
            report.append(f"{ppe_type}: {count}")
        report.append("")
    
    # Recommendations
    report.append("RECOMMENDATIONS:")
    report.append("-" * 30)
    if compliance_data['compliant']:
        report.append("• Continue maintaining current safety standards")
    else:
        if compliance_data['severity'] == 'high':
            report.append("• IMMEDIATE ACTION REQUIRED")
            report.append("• Stop work until all workers have proper PPE")
        report.append("• Ensure all workers wear hard hats")
        report.append("• Ensure all workers wear high-visibility safety vests")
        report.append("• Conduct safety briefing")
        report.append("• Review and reinforce PPE policies")
    
    report.append("")
    report.append("="*60)
    
    report_text = "\n".join(report)
    
    if save_path:
        with open(save_path, 'w') as f:
            f.write(report_text)
        print(f"Compliance report saved to {save_path}")
    
    return report_text


def create_training_visualization(train_losses, val_losses, save_path=None):
    """Create training progress visualization"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Training loss
    ax1.plot(train_losses, label='Training Loss', color='blue')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss Progress')
    ax1.legend()
    ax1.grid(True)
    
    # Validation loss
    if val_losses:
        ax2.plot(val_losses, label='Validation Loss', color='red')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.set_title('Validation Loss Progress')
        ax2.legend()
        ax2.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Training visualization saved to {save_path}")
    
    plt.show()


class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch, initial_lr, decay_epochs, decay_factor=0.1):
    """Adjust learning rate based on epoch"""
    lr = initial_lr
    for decay_epoch in decay_epochs:
        if epoch >= decay_epoch:
            lr *= decay_factor
    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    
    return lr


if __name__ == "__main__":
    # Test utility functions
    print("Testing utility functions...")
    
    # Test compliance checking
    dummy_boxes = [[0.1, 0.1, 0.3, 0.3], [0.5, 0.5, 0.7, 0.7]]
    dummy_labels = [1, 7]  # person, no_hard_hat
    dummy_scores = [0.9, 0.8]
    class_names = ['background', 'person', 'hard_hat', 'safety_vest', 'safety_gloves', 
                  'safety_boots', 'eye_protection', 'no_hard_hat', 'no_safety_vest']
    
    compliance = check_ppe_compliance(dummy_boxes, dummy_labels, dummy_scores, class_names)
    report = generate_compliance_report(compliance)
    print(report)
    
    print("Utility functions test completed!")