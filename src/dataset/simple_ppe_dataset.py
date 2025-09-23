#!/usr/bin/env python3
"""
Windows-compatible PPE dataset for training
Simplified version to avoid multiprocessing issues
"""

import os
import xml.etree.ElementTree as ET
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import numpy as np
from pathlib import Path

class SimplePPEDataset(Dataset):
    """Simplified PPE Dataset for Windows training"""
    
    def __init__(self, data_dir, split='train', img_size=300):
        self.data_dir = data_dir
        self.split = split
        self.img_size = img_size
        
        # PPE classes
        self.ppe_classes = [
            'background', 'person', 'hard_hat', 'safety_vest', 
            'safety_gloves', 'safety_boots', 'eye_protection',
            'no_hard_hat', 'no_safety_vest', 'no_safety_gloves',
            'no_safety_boots', 'no_eye_protection'
        ]
        
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.ppe_classes)}
        
        # Load image paths and annotations
        self.samples = self._load_samples()
        
        # Simple transforms
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        print(f"Loaded {len(self.samples)} samples for {split}")
    
    def _load_samples(self):
        """Load image paths and corresponding annotation files"""
        samples = []
        
        # Read split file
        split_file = Path(self.data_dir) / "splits" / f"{self.split}.txt"
        if not split_file.exists():
            print(f"Split file not found: {split_file}")
            return samples
        
        with open(split_file, 'r') as f:
            image_names = [line.strip() for line in f.readlines()]
        
        images_dir = Path(self.data_dir) / "images"
        annotations_dir = Path(self.data_dir) / "annotations"
        
        for image_name in image_names:
            image_path = images_dir / image_name
            annotation_path = annotations_dir / f"{Path(image_name).stem}.xml"
            
            if image_path.exists() and annotation_path.exists():
                samples.append({
                    'image_path': str(image_path),
                    'annotation_path': str(annotation_path),
                    'image_name': image_name
                })
            else:
                print(f"Missing files for {image_name}")
        
        return samples
    
    def _parse_annotation(self, annotation_path):
        """Parse XML annotation file"""
        tree = ET.parse(annotation_path)
        root = tree.getroot()
        
        # Get image dimensions
        size = root.find('size')
        width = int(size.find('width').text)
        height = int(size.find('height').text)
        
        boxes = []
        labels = []
        
        # Parse objects
        for obj in root.findall('object'):
            class_name = obj.find('name').text
            if class_name in self.class_to_idx:
                label = self.class_to_idx[class_name]
                
                bbox = obj.find('bndbox')
                xmin = float(bbox.find('xmin').text)
                ymin = float(bbox.find('ymin').text)
                xmax = float(bbox.find('xmax').text)
                ymax = float(bbox.find('ymax').text)
                
                # Normalize coordinates to [0, 1]
                xmin /= width
                ymin /= height
                xmax /= width
                ymax /= height
                
                boxes.append([xmin, ymin, xmax, ymax])
                labels.append(label)
        
        return torch.tensor(boxes, dtype=torch.float32), torch.tensor(labels, dtype=torch.long)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load image
        image = Image.open(sample['image_path']).convert('RGB')
        image = self.transform(image)
        
        # Load annotations
        boxes, labels = self._parse_annotation(sample['annotation_path'])
        
        # Create target dictionary
        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': torch.tensor([idx])
        }
        
        return image, target, sample['image_name']
    
    def __len__(self):
        return len(self.samples)

def simple_collate_fn(batch):
    """Simple collate function for batching"""
    images = []
    targets = []
    filenames = []
    
    for image, target, filename in batch:
        images.append(image)
        targets.append(target)
        filenames.append(filename)
    
    images = torch.stack(images, 0)
    
    return images, targets, filenames