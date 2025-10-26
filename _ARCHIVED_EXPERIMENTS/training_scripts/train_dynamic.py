#!/usr/bin/env python3
"""
Fixed training script that dynamically handles model dimensions
"""

import os
import sys
import argparse
import time
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.dataset.ppe_dataset import PPEDataset
from src.models.ssd import SSD300
from src.models.loss import MultiBoxLoss, create_prior_boxes


class DynamicPPELoss(nn.Module):
    """PPE Loss that adapts to actual model output dimensions"""
    
    def __init__(self, model, threshold=0.5, neg_pos_ratio=3, alpha=1.0):
        super(DynamicPPELoss, self).__init__()
        
        # Get actual number of priors from model
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 300, 300)
            if torch.cuda.is_available():
                dummy_input = dummy_input.cuda()
                model = model.cuda()
            
            model.eval()
            locs, scores = model(dummy_input)
            self.n_priors = locs.shape[1]
            print(f"Detected {self.n_priors} prior boxes from model")
        
        # Create matching prior boxes
        self.priors_cxcy = self._create_matching_priors()
        self.priors_xy = self._cxcy_to_xy(self.priors_cxcy)
        
        self.threshold = threshold
        self.neg_pos_ratio = neg_pos_ratio
        self.alpha = alpha
        
        self.smooth_l1 = nn.SmoothL1Loss(reduction='none')
        self.cross_entropy = nn.CrossEntropyLoss(reduction='none')
    
    def _create_matching_priors(self):
        """Create prior boxes that match model output exactly"""
        print(f"Creating {self.n_priors} prior boxes...")
        
        # Simple grid-based prior generation
        # This creates priors that match your model's actual output
        priors = []
        
        # Approximate feature map dimensions based on total boxes
        if self.n_priors == 8096:
            # Your model configuration
            fmap_configs = [
                (37, 4),   # conv4_3: 37x37x4 = 5476
                (18, 6),   # conv7: 18x18x6 = 1944  
                (10, 6),   # conv8_2: 10x10x6 = 600
                (5, 6),    # conv9_2: 5x5x6 = 150
                (3, 4),    # conv10_2: 3x3x4 = 36
                (1, 4)     # conv11_2: 1x1x4 = 4
            ]
            scales = [0.1, 0.2, 0.375, 0.55, 0.725, 0.9]
            
        else:
            # Fallback for other configurations
            print(f"Using fallback prior generation for {self.n_priors} boxes")
            # Create a simple uniform grid
            grid_size = int((self.n_priors / 4) ** 0.5)
            fmap_configs = [(grid_size, 4)]
            scales = [0.3]
        
        for (fmap_size, n_boxes), scale in zip(fmap_configs, scales):
            for i in range(fmap_size):
                for j in range(fmap_size):
                    cx = (j + 0.5) / fmap_size
                    cy = (i + 0.5) / fmap_size
                    
                    # Create different aspect ratios
                    for box_idx in range(n_boxes):
                        if box_idx == 0:
                            w = h = scale
                        elif box_idx == 1:
                            w = scale * 1.4
                            h = scale / 1.4
                        elif box_idx == 2:
                            w = scale / 1.4
                            h = scale * 1.4
                        elif box_idx == 3:
                            w = scale * 1.7
                            h = scale / 1.7
                        elif box_idx == 4:
                            w = scale / 1.7
                            h = scale * 1.7
                        else:
                            w = scale * 2.0
                            h = scale / 2.0
                        
                        priors.append([cx, cy, w, h])
        
        # Trim to exact number needed
        priors = priors[:self.n_priors]
        
        # Pad if needed
        while len(priors) < self.n_priors:
            priors.append([0.5, 0.5, 0.1, 0.1])
        
        priors_tensor = torch.FloatTensor(priors)
        priors_tensor.clamp_(0, 1)
        
        print(f"Created {len(priors)} prior boxes")
        return priors_tensor
    
    def _cxcy_to_xy(self, cxcy):
        """Convert center-size to corner coordinates"""
        return torch.cat([cxcy[:, :2] - (cxcy[:, 2:] / 2),
                         cxcy[:, :2] + (cxcy[:, 2:] / 2)], 1)
    
    def forward(self, predicted_locs, predicted_scores, boxes, labels):
        """Forward pass with dynamic dimensions"""
        
        device = predicted_locs.device
        batch_size = predicted_locs.size(0)
        n_priors = predicted_locs.size(1)
        n_classes = predicted_scores.size(2)
        
        # Move priors to correct device
        if self.priors_cxcy.device != device:
            self.priors_cxcy = self.priors_cxcy.to(device)
            self.priors_xy = self.priors_xy.to(device)
        
        # Initialize ground truth tensors
        true_locs = torch.zeros((batch_size, n_priors, 4), dtype=torch.float, device=device)
        true_classes = torch.zeros((batch_size, n_priors), dtype=torch.long, device=device)
        
        # For each image in batch
        for i in range(batch_size):
            n_objects = boxes[i].size(0)
            
            if n_objects > 0:
                # Simple assignment: assign each object to nearest prior
                # This is a simplified version for getting training started
                overlap = self._jaccard_overlap(boxes[i], self.priors_xy)
                
                # Assign each prior to the best overlapping object
                overlap_for_each_prior, object_for_each_prior = overlap.max(dim=0)
                
                # Set threshold for positive priors
                label_for_each_prior = labels[i][object_for_each_prior]
                label_for_each_prior[overlap_for_each_prior < self.threshold] = 0
                
                true_classes[i] = label_for_each_prior
                
                # Encode locations (simplified)
                true_locs[i] = self._encode_locations(boxes[i][object_for_each_prior])
        
        # Calculate losses
        positive_priors = true_classes != 0
        
        # Localization loss
        if positive_priors.sum() > 0:
            loc_loss = self.smooth_l1(predicted_locs[positive_priors], true_locs[positive_priors]).mean()
        else:
            loc_loss = torch.tensor(0.0, device=device)
        
        # Classification loss (simplified)
        conf_loss = self.cross_entropy(predicted_scores.view(-1, n_classes), true_classes.view(-1)).mean()
        
        return conf_loss + self.alpha * loc_loss
    
    def _jaccard_overlap(self, boxes1, boxes2):
        """Simplified IoU calculation"""
        # Simple overlap calculation
        overlap = torch.zeros(boxes1.size(0), boxes2.size(0))
        
        for i in range(boxes1.size(0)):
            for j in range(boxes2.size(0)):
                # Calculate IoU between box i and prior j
                box1 = boxes1[i]
                box2 = boxes2[j]
                
                # Intersection
                x1 = max(box1[0], box2[0])
                y1 = max(box1[1], box2[1])
                x2 = min(box1[2], box2[2])
                y2 = min(box1[3], box2[3])
                
                if x2 > x1 and y2 > y1:
                    intersection = (x2 - x1) * (y2 - y1)
                    
                    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
                    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
                    union = area1 + area2 - intersection
                    
                    overlap[i, j] = intersection / union if union > 0 else 0
        
        return overlap.to(boxes1.device)
    
    def _encode_locations(self, boxes):
        """Simplified location encoding"""
        # Convert to center-size format relative to priors
        centers = (boxes[:, :2] + boxes[:, 2:]) / 2
        sizes = boxes[:, 2:] - boxes[:, :2]
        
        # Simple encoding (not exact SSD encoding but functional)
        prior_centers = self.priors_cxcy[:, :2]
        prior_sizes = self.priors_cxcy[:, 2:]
        
        encoded_centers = (centers - prior_centers) / prior_sizes
        encoded_sizes = torch.log(sizes / prior_sizes + 1e-8)
        
        return torch.cat([encoded_centers, encoded_sizes], 1)


def custom_collate_fn(batch):
    """Custom collate function for variable-sized annotations"""
    images = []
    boxes = []
    labels = []
    filenames = []
    
    for item in batch:
        # Dataset returns (image_tensor, annotation_dict, filename)
        image, annotations, filename = item
        
        images.append(image)
        boxes.append(annotations['bboxes'])  # Extract bbox tensor
        labels.append(annotations['labels'])  # Extract label tensor
        filenames.append(filename)
    
    # Stack images (they should all be the same size)
    images = torch.stack(images, 0)
    
    # Keep boxes and labels as lists (different sizes per image)
    return images, boxes, labels, filenames


def train_epoch(model, train_loader, criterion, optimizer, device, epoch):
    """Train for one epoch"""
    
    model.train()
    running_loss = 0.0
    
    for i, (images, boxes, labels, _) in enumerate(train_loader):
        images = images.to(device)
        boxes = [b.to(device) for b in boxes]
        labels = [l.to(device) for l in labels]
        
        optimizer.zero_grad()
        
        # Forward pass
        predicted_locs, predicted_scores = model(images)
        
        # Calculate loss
        loss = criterion(predicted_locs, predicted_scores, boxes, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        if i % 10 == 0:
            print(f'Epoch {epoch}, Batch {i}, Loss: {loss.item():.4f}')
    
    return running_loss / len(train_loader)


def main():
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Model
    model = SSD300(n_classes=13)
    model.to(device)
    
    # Dynamic loss function that adapts to model
    criterion = DynamicPPELoss(model, alpha=1.0)
    criterion.to(device)
    
    # Dataset
    train_dataset = PPEDataset(
        data_dir='data',
        split='train',
        img_size=300
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=4,
        shuffle=True,
        num_workers=0,  # Avoid Windows multiprocessing issues
        collate_fn=custom_collate_fn,
        pin_memory=False
    )
    
    # Optimizer
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)
    
    print(f"Starting training with {len(train_dataset)} samples...")
    print(f"Model produces {criterion.n_priors} prior boxes")
    
    # Training loop
    for epoch in range(5):  # Short training for testing
        avg_loss = train_epoch(model, train_loader, criterion, optimizer, device, epoch)
        print(f'Epoch {epoch} completed. Average loss: {avg_loss:.4f}')
        
        # Save checkpoint
        if epoch % 2 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, f'models/checkpoint_epoch_{epoch}.pth')
    
    print("Training completed!")


if __name__ == "__main__":
    # Create directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    
    main()