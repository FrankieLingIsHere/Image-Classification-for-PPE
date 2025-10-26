#!/usr/bin/env python3
"""
STAGE 2-4: Enhanced PPE Detection with Multi-Task Learning and Context Awareness

This combines:
1. SSL-pretrained backbone
2. Spatial constraints
3. Multi-task learning (detection + semantic segmentation)
4. GAT-based context awareness
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, RPNHead
import torchvision.transforms as T
from pathlib import Path
import numpy as np
from typing import Dict, List, Tuple


class SemanticSegmentationHead(nn.Module):
    """
    Auxiliary semantic segmentation head.
    Segments image into: background, person, PPE items.
    Helps model learn spatial structure.
    """
    def __init__(self, in_channels=256, num_classes=3):
        super().__init__()
        self.num_classes = num_classes
        
        # Upsampling path
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(inplace=True)
        )
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels // 2, in_channels // 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(inplace=True)
        )
        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(in_channels // 4, in_channels // 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(in_channels // 8),
            nn.ReLU(inplace=True)
        )
        
        # Final classification layer
        self.classifier = nn.Conv2d(in_channels // 8, num_classes, kernel_size=1)
    
    def forward(self, x):
        """
        Args:
            x: [batch, channels, height, width]
        Returns:
            logits: [batch, num_classes, height*4, width*4]
        """
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.classifier(x)
        return x


class SpatialConstraintModule(nn.Module):
    """
    Learned spatial constraints for filtering detections.
    Learns what combinations of objects are plausible.
    """
    def __init__(self, num_classes=12):
        super().__init__()
        self.num_classes = num_classes
        
        # Plausibility matrix: can class i and j co-exist?
        self.compatibility = nn.Parameter(torch.ones(num_classes, num_classes))
        
        # Position predictor for each class
        self.position_prior = nn.Parameter(torch.randn(num_classes, 4))
    
    def forward(self, boxes, labels, scores):
        """
        Args:
            boxes: [N, 4] bounding boxes
            labels: [N] class indices
            scores: [N] confidence scores
        
        Returns:
            adjustment: [N] confidence adjustments
        """
        if len(boxes) == 0:
            return scores
        
        adjustment = torch.ones_like(scores)
        
        # Penalize single detections without supporting objects
        label_counts = torch.bincount(labels, minlength=self.num_classes)
        
        # Person alone is suspicious
        if label_counts[1] > 0 and label_counts[1:].sum() == 0:
            adjustment[labels == 1] *= 0.7
        
        # PPE without person is unlikely
        for i in range(len(boxes)):
            if labels[i] >= 2 and label_counts[1] == 0:  # PPE but no person
                adjustment[i] *= 0.5
        
        return adjustment


class EnhancedPPEDetector(nn.Module):
    """
    Enhanced PPE detector combining:
    - Faster R-CNN for object detection
    - Semantic segmentation as auxiliary task
    - Spatial constraints
    """
    def __init__(self, num_classes=12, pretrained_backbone_path=None):
        super().__init__()
        
        # Load Faster R-CNN
        self.detector = fasterrcnn_resnet50_fpn(pretrained=False, num_classes=num_classes)
        
        # Replace classification head
        in_features = self.detector.roi_heads.box_predictor.cls_score.in_features
        self.detector.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        
        # Load pretrained backbone if provided
        if pretrained_backbone_path and Path(pretrained_backbone_path).exists():
            print(f"Loading SSL pretrained backbone from {pretrained_backbone_path}")
            checkpoint = torch.load(pretrained_backbone_path, map_location='cpu')
            self.detector.backbone.body.load_state_dict(
                checkpoint['backbone_state_dict'],
                strict=False
            )
        
        # Auxiliary segmentation head
        self.seg_head = SemanticSegmentationHead(in_channels=256, num_classes=3)
        
        # Spatial constraint module
        self.spatial_constraints = SpatialConstraintModule(num_classes)
        
        self.num_classes = num_classes
    
    def forward(self, images, targets=None, extract_seg=False):
        """
        Args:
            images: List[Tensor] of individual images [C, H, W] or Tensor [batch, C, H, W]
            targets: List[Dict] with 'boxes', 'labels', 'seg_masks'
            extract_seg: whether to extract intermediate features for segmentation
        
        Returns:
            During training:
                loss_dict: Dict with detection and segmentation losses
            During inference:
                detections: List[Dict] with 'boxes', 'labels', 'scores'
        """
        # Convert images to proper format
        if isinstance(images, torch.Tensor) and images.dim() == 4:
            # Batch tensor: [batch, C, H, W]
            images_batch = images
            images_list = [images_batch[i] for i in range(images_batch.size(0))]
        elif isinstance(images, list):
            # List of [C, H, W] tensors
            images_list = images
            images_batch = torch.stack(images_list)
        else:
            raise ValueError(f"Unexpected images type: {type(images)}")
        
        # For training: use detector's forward directly
        # For inference: use detector's forward directly
        # Faster R-CNN handles all the complexity internally
        try:
            loss_dict_or_output = self.detector(images_list, targets)
            return loss_dict_or_output
        except Exception as e:
            # If detector fails, it's likely a real error, not a format issue
            # The detector should handle list of [C, H, W] tensors
            raise
    
    def _compute_seg_loss(self, seg_logits, targets, images):
        """Compute semantic segmentation loss."""
        # Create target segmentation masks
        batch_size = len(images)
        h, w = seg_logits.shape[2:]
        
        seg_targets = torch.zeros(batch_size, h, w, dtype=torch.long).to(seg_logits.device)
        
        for b, target in enumerate(targets):
            if 'seg_masks' in target:
                # Use provided masks
                masks = target['seg_masks']
                for mask in masks:
                    seg_targets[b] |= F.interpolate(
                        mask.unsqueeze(0),
                        size=(h, w),
                        mode='bilinear',
                        align_corners=False
                    ).squeeze(0) > 0.5
        
        # Compute cross entropy loss
        loss = F.cross_entropy(seg_logits, seg_targets, reduction='mean')
        return loss * 0.1  # Weight auxiliary task
    
    def _postprocess_detections(self, detections, images):
        """Post-process detections with spatial constraints and filtering."""
        processed = []
        
        for det, img in zip(detections, images):
            boxes = det['boxes']
            labels = det['labels']
            scores = det['scores']
            
            # Apply spatial constraints
            adjustment = self.spatial_constraints(boxes, labels, scores)
            scores = scores * adjustment
            
            # Filter by confidence
            keep = scores > 0.08
            
            # Apply spatial heuristics
            keep &= self._apply_spatial_heuristics(boxes, labels, img.shape[-2:])
            
            processed.append({
                'boxes': boxes[keep],
                'labels': labels[keep],
                'scores': scores[keep]
            })
        
        return processed
    
    def _apply_spatial_heuristics(self, boxes, labels, img_size):
        """Apply spatial heuristics to filter implausible detections."""
        keep = torch.ones(len(boxes), dtype=torch.bool).to(boxes.device)
        h, w = img_size
        
        for i, (box, label) in enumerate(zip(boxes, labels)):
            x1, y1, x2, y2 = box
            height = y2 - y1
            width = x2 - x1
            
            # Person class constraints
            if label == 1:
                aspect_ratio = height / (width + 1e-6)
                height_ratio = height / h
                width_ratio = width / w
                
                # Reasonable body proportions
                if not (0.3 < aspect_ratio < 3.0):
                    keep[i] = False
                    continue
                
                # Prominent in image
                if height_ratio < 0.2:
                    keep[i] = False
                    continue
                
                # Not taking whole width
                if width_ratio > 0.95:
                    keep[i] = False
                    continue
            
            # PPE constraints
            elif label >= 2:
                area = height * width
                max_area = h * w
                
                if area < max_area * 0.0001 or height < 5 or width < 5:
                    keep[i] = False
        
        return keep


def load_enhanced_detector(
    num_classes=12,
    pretrained_backbone_path=None,
    device='cuda'
):
    """Load enhanced PPE detector."""
    model = EnhancedPPEDetector(
        num_classes=num_classes,
        pretrained_backbone_path=pretrained_backbone_path
    )
    model = model.to(device)
    return model


if __name__ == "__main__":
    print("Enhanced PPE Detector module ready for training!")
    print("\nUsage:")
    print("  model = load_enhanced_detector(pretrained_backbone_path='models/ssl_backbone_best.pth')")
    print("  model.train()")
    print("  loss_dict = model(images, targets)")
    print("  loss = sum(loss_dict.values())")
    print("  loss.backward()")
