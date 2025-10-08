import os
import torch
import torchvision.transforms.v2 as transforms
from torch.utils.data.dataset import Dataset
import xml.etree.ElementTree as ET
from torchvision import tv_tensors
from torchvision.io import read_image
import json


def load_ppe_images_and_annotations(data_dir, label2idx, split):
    """
    Load PPE dataset images and annotations
    
    Args:
        data_dir: Root directory containing images and annotations
        label2idx: Dictionary mapping class names to indices
        split: 'train', 'val', or 'test'
    
    Returns:
        List of image information dictionaries
    """
    im_infos = []
    
    # Define paths
    split_file = os.path.join(data_dir, 'splits', f'{split}.txt')
    images_dir = os.path.join(data_dir, 'images')
    annotations_dir = os.path.join(data_dir, 'annotations')
    
    # If split file doesn't exist, create a basic one
    if not os.path.exists(split_file):
        print(f"Split file {split_file} not found. Creating basic structure.")
        os.makedirs(os.path.dirname(split_file), exist_ok=True)
        # Create empty split file for now
        with open(split_file, 'w') as f:
            f.write("")
        return im_infos
    
    # Read image names from split file
    if os.path.getsize(split_file) == 0:
        print(f"Split file {split_file} is empty. No images to load.")
        return im_infos
        
    with open(split_file, 'r') as f:
        image_names = [line.strip() for line in f.readlines() if line.strip()]
    
    for img_name in image_names:
        # Support both .jpg and .xml annotation formats
        base_name = os.path.splitext(img_name)[0]
        
        # Try different image extensions
        img_path = None
        for ext in ['.jpg', '.png', '.jpeg']:
            potential_path = os.path.join(images_dir, f"{base_name}{ext}")
            if os.path.exists(potential_path):
                img_path = potential_path
                break
        
        # If we have the full filename with extension in the split file, try that too
        if img_path is None:
            potential_path = os.path.join(images_dir, img_name)
            if os.path.exists(potential_path):
                img_path = potential_path
        
        if img_path is None:
            print(f"Warning: Image not found for {img_name} (tried {base_name}.jpg, {base_name}.png, {base_name}.jpeg)")
            continue
        
        # Try XML annotation first, then JSON
        xml_path = os.path.join(annotations_dir, f"{base_name}.xml")
        json_path = os.path.join(annotations_dir, f"{base_name}.json")
            
        im_info = {
            'img_id': base_name,
            'filename': img_path,
            'detections': []
        }
        
        # Parse XML annotations (VOC format)
        if os.path.exists(xml_path):
            im_info.update(parse_xml_annotation(xml_path, label2idx))
        # Parse JSON annotations (COCO-like format)
        elif os.path.exists(json_path):
            im_info.update(parse_json_annotation(json_path, label2idx))
        else:
            # Create default image info if no annotation exists
            try:
                from PIL import Image
                with Image.open(img_path) as img:
                    im_info['width'] = img.width
                    im_info['height'] = img.height
            except:
                im_info['width'] = 640
                im_info['height'] = 480
                
        im_infos.append(im_info)
    
    print(f'Total {len(im_infos)} images found for {split} split')
    return im_infos


def parse_xml_annotation(xml_path, label2idx):
    """Parse VOC-style XML annotation"""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    # Get image dimensions
    size = root.find('size')
    width = int(size.find('width').text)
    height = int(size.find('height').text)
    
    detections = []
    for obj in root.findall('object'):
        class_name = obj.find('name').text
        if class_name not in label2idx:
            continue
            
        difficult = int(obj.find('difficult').text) if obj.find('difficult') is not None else 0
        bbox = obj.find('bndbox')
        
        # Convert to [xmin, ymin, xmax, ymax] format (0-indexed)
        xmin = int(bbox.find('xmin').text) - 1
        ymin = int(bbox.find('ymin').text) - 1
        xmax = int(bbox.find('xmax').text) - 1
        ymax = int(bbox.find('ymax').text) - 1
        
        detection = {
            'label': label2idx[class_name],
            'bbox': [xmin, ymin, xmax, ymax],
            'difficult': difficult
        }
        detections.append(detection)
    
    return {
        'width': width,
        'height': height,
        'detections': detections
    }


def parse_json_annotation(json_path, label2idx):
    """Parse JSON annotation (custom format)"""
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    width = data.get('width', 640)
    height = data.get('height', 480)
    
    detections = []
    for ann in data.get('annotations', []):
        class_name = ann.get('class', ann.get('category'))
        if class_name not in label2idx:
            continue
            
        bbox = ann.get('bbox', ann.get('bounding_box'))
        if bbox:
            # Assume bbox is in [x, y, width, height] format, convert to [xmin, ymin, xmax, ymax]
            if len(bbox) == 4:
                x, y, w, h = bbox
                detection = {
                    'label': label2idx[class_name],
                    'bbox': [x, y, x + w, y + h],
                    'difficult': ann.get('difficult', 0)
                }
                detections.append(detection)
    
    return {
        'width': width,
        'height': height,
        'detections': detections
    }


class PPEDataset(Dataset):
    """
    PPE Detection Dataset for construction safety compliance
    
    This dataset is designed to detect Personal Protective Equipment (PPE)
    violations in construction environments according to OSHA regulations.
    """
    
    def __init__(self, data_dir, split='train', img_size=300, transforms=None):
        """
        Initialize PPE Dataset
        
        Args:
            data_dir: Root directory containing the dataset
            split: 'train', 'val', or 'test'
            img_size: Target image size for training
            transforms: Custom transforms (if None, default transforms are used)
        """
        self.data_dir = data_dir
        self.split = split
        self.img_size = img_size
        
        # PPE classes according to OSHA requirements
        self.ppe_classes = [
            'background',          # 0: background
            'person',              # 1: worker/person
            'hard_hat',            # 2: hard hat/helmet
            'safety_vest',         # 3: high-visibility safety vest
            'safety_gloves',       # 4: protective gloves
            'safety_boots',        # 5: safety footwear
            'eye_protection',      # 6: safety glasses/goggles
            'no_hard_hat',         # 7: person without hard hat (violation)
            'no_safety_vest',      # 8: person without safety vest (violation)
            'no_safety_gloves',    # 9: person without safety gloves (violation)
            'no_safety_boots',     # 10: person without safety boots (violation)
            'no_eye_protection',   # 11: person without eye protection (violation)
        ]
        
        # Create label mappings
        self.label2idx = {cls: idx for idx, cls in enumerate(self.ppe_classes)}
        self.idx2label = {idx: cls for idx, cls in enumerate(self.ppe_classes)}
        
        print(f"PPE Classes: {self.idx2label}")
        
        # Default image normalization values
        self.imagenet_mean = [0.485, 0.456, 0.406]
        self.imagenet_std = [0.229, 0.224, 0.225]
        self.img_mean = [123.0, 117.0, 104.0]  # BGR format for SSD
        
        # Set up transforms
        if transforms is None:
            self.transforms = self._get_default_transforms()
        else:
            self.transforms = transforms
            
        # Load dataset
        self.images_info = load_ppe_images_and_annotations(
            self.data_dir, self.label2idx, self.split
        )
    
    def _get_default_transforms(self):
        """Get default transforms for training and testing"""
        # Simple transforms without lambda functions for Windows compatibility
        train_transforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.imagenet_mean, std=self.imagenet_std)
        ])
        
        test_transforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.imagenet_mean, std=self.imagenet_std)
        ])
        
        return {
            'train': train_transforms,
            'val': test_transforms,
            'test': test_transforms
        }
    
    def __len__(self):
        return len(self.images_info)
    
    def __getitem__(self, index):
        """
        Get a sample from the dataset
        
        Returns:
            image: Transformed image tensor
            targets: Dictionary containing bboxes, labels, and difficult flags
            filename: Original filename for reference
        """
        im_info = self.images_info[index]
        
        # Load image
        try:
            image = read_image(im_info['filename'])
            # Ensure image has exactly 3 channels (RGB)
            if image.shape[0] == 4:  # RGBA
                image = image[:3]  # Remove alpha channel
            elif image.shape[0] == 1:  # Grayscale
                image = image.repeat(3, 1, 1)  # Convert to RGB
        except Exception as e:
            print(f"Error loading image {im_info['filename']}: {e}")
            # Try loading with PIL as fallback for problematic formats
            try:
                from PIL import Image
                pil_img = Image.open(im_info['filename']).convert('RGB')
                import numpy as np
                image = torch.from_numpy(np.array(pil_img)).permute(2, 0, 1)
                print(f"Successfully loaded {im_info['filename']} using PIL fallback")
            except Exception as e2:
                print(f"PIL fallback also failed for {im_info['filename']}: {e2}")
                # Return a dummy image
                image = torch.zeros(3, 480, 640, dtype=torch.uint8)
        
        # Prepare targets
        targets = {}
        if im_info['detections']:
            targets['bboxes'] = tv_tensors.BoundingBoxes(
                [det['bbox'] for det in im_info['detections']],
                format='XYXY',
                canvas_size=image.shape[-2:]
            )
            targets['labels'] = torch.tensor(
                [det['label'] for det in im_info['detections']], 
                dtype=torch.long
            )
            targets['difficult'] = torch.tensor(
                [det['difficult'] for det in im_info['detections']], 
                dtype=torch.long
            )
        else:
            # No annotations - create empty targets
            targets['bboxes'] = tv_tensors.BoundingBoxes(
                torch.empty(0, 4), 
                format='XYXY',
                canvas_size=image.shape[-2:]
            )
            targets['labels'] = torch.empty(0, dtype=torch.long)
            targets['difficult'] = torch.empty(0, dtype=torch.long)
        
        # Apply transforms
        if self.split in self.transforms:
            image, targets = self.transforms[self.split](image, targets)
        
        # Normalize bounding boxes to [0, 1]
        h, w = image.shape[-2:]
        if len(targets['bboxes']) > 0:
            wh_tensor = torch.tensor([[w, h, w, h]], dtype=torch.float32)
            wh_tensor = wh_tensor.expand_as(targets['bboxes'])
            targets['bboxes'] = targets['bboxes'] / wh_tensor
        
        return image, targets, im_info['filename']
    
    def get_class_names(self):
        """Return list of class names"""
        return self.ppe_classes
    
    def get_num_classes(self):
        """Return number of classes including background"""
        return len(self.ppe_classes)


def create_sample_data_structure(data_dir):
    """
    Create sample data structure for PPE dataset
    This function helps users understand the expected directory structure
    """
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(os.path.join(data_dir, 'images'), exist_ok=True)
    os.makedirs(os.path.join(data_dir, 'annotations'), exist_ok=True)
    os.makedirs(os.path.join(data_dir, 'splits'), exist_ok=True)
    
    # Create sample split files
    splits = ['train', 'val', 'test']
    for split in splits:
        split_file = os.path.join(data_dir, 'splits', f'{split}.txt')
        with open(split_file, 'w') as f:
            f.write("# Add your image filenames here (one per line)\n")
            f.write("# Example:\n")
            f.write("# construction_site_001.jpg\n")
            f.write("# worker_with_ppe_002.jpg\n")
    
    # Create README for data structure
    readme_content = """
# PPE Dataset Structure

This directory should contain your PPE detection dataset in the following structure:

```
data/
├── images/              # All images (.jpg, .png)
├── annotations/         # Annotation files (.xml or .json)
└── splits/              # Train/val/test split files
    ├── train.txt
    ├── val.txt
    └── test.txt
```

## Annotation Formats Supported:

### 1. VOC XML Format (.xml)
```xml
<annotation>
    <size>
        <width>640</width>
        <height>480</height>
    </size>
    <object>
        <name>hard_hat</name>
        <difficult>0</difficult>
        <bndbox>
            <xmin>100</xmin>
            <ymin>100</ymin>
            <xmax>200</xmax>
            <ymax>200</ymax>
        </bndbox>
    </object>
</annotation>
```

### 2. JSON Format (.json)
```json
{
    "width": 640,
    "height": 480,
    "annotations": [
        {
            "class": "hard_hat",
            "bbox": [100, 100, 100, 100],
            "difficult": 0
        }
    ]
}
```

## PPE Classes:
- person: Worker/person in the image
- hard_hat: Hard hat/helmet
- safety_vest: High-visibility safety vest
- safety_gloves: Protective gloves
- safety_boots: Safety footwear
- eye_protection: Safety glasses/goggles
- no_hard_hat: Person without hard hat (violation)
- no_safety_vest: Person without safety vest (violation)
"""
    
    with open(os.path.join(data_dir, 'README.md'), 'w') as f:
        f.write(readme_content)
    
    print(f"Sample data structure created in {data_dir}")
    print("Please add your images and annotations following the structure described in the README.md")


if __name__ == "__main__":
    # Example usage
    data_dir = "data"
    
    # Create sample data structure
    create_sample_data_structure(data_dir)
    
    # Test dataset loading
    try:
        dataset = PPEDataset(data_dir, split='train')
        print(f"Dataset created successfully with {len(dataset)} images")
        print(f"Classes: {dataset.get_class_names()}")
    except Exception as e:
        print(f"Error creating dataset: {e}")
        print("Make sure to add your data following the structure in data/README.md")