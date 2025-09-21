# Image Classification for PPE

A comprehensive SSD (Single Shot MultiBox Detector) implementation for Personal Protective Equipment (PPE) detection in construction environments to ensure OSHA compliance.

## Overview

This project implements an advanced computer vision system that can:

- **Detect PPE Equipment**: Identify hard hats, safety vests, gloves, boots, and eye protection
- **Monitor OSHA Compliance**: Check for PPE violations and generate compliance reports
- **Real-time Processing**: Fast inference suitable for surveillance systems
- **Detailed Reporting**: Generate comprehensive safety violation reports

## Features

### PPE Detection Classes
- `person`: Workers in the scene
- `hard_hat`: Hard hats/helmets (OSHA critical)
- `safety_vest`: High-visibility safety vests (OSHA critical)
- `safety_gloves`: Protective gloves
- `safety_boots`: Safety footwear
- `eye_protection`: Safety glasses/goggles
- `no_hard_hat`: Workers without hard hats (violation)
- `no_safety_vest`: Workers without safety vests (violation)

### OSHA Compliance Features
- **Critical Violation Detection**: Immediate alerts for missing hard hats or safety vests
- **Weighted Loss Function**: Prioritizes critical safety equipment in training
- **Compliance Scoring**: Automatic assessment of safety compliance levels
- **Detailed Reports**: Text and JSON reports with recommendations

## Installation

### Prerequisites
- Python 3.8+
- PyTorch 2.0+
- CUDA (optional, for GPU acceleration)

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Quick Setup
```bash
git clone <repository-url>
cd Image-Classification-for-PPE
pip install -r requirements.txt
python -c "from src.dataset.ppe_dataset import create_sample_data_structure; create_sample_data_structure('data')"
```

## Dataset Preparation

### Directory Structure
```
data/
├── images/              # All images (.jpg, .png)
├── annotations/         # Annotation files (.xml or .json)
└── splits/              # Train/val/test split files
    ├── train.txt
    ├── val.txt
    └── test.txt
```

### Annotation Formats

#### VOC XML Format
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

#### JSON Format
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

### Split Files
Create text files with image filenames (one per line):
```
# train.txt
construction_site_001.jpg
worker_with_ppe_002.jpg
safety_inspection_003.jpg
```

## Usage

### Training

#### Basic Training
```bash
python scripts/train.py --data_dir data --batch_size 8 --epochs 100
```

#### Advanced Training
```bash
python scripts/train.py \
    --data_dir data \
    --batch_size 16 \
    --epochs 200 \
    --lr 0.001 \
    --img_size 300 \
    --save_dir models/experiment_1 \
    --log_dir logs/experiment_1
```

#### Training Parameters
- `--data_dir`: Path to dataset directory
- `--batch_size`: Training batch size (default: 8)
- `--epochs`: Number of training epochs (default: 100)
- `--lr`: Learning rate (default: 0.001)
- `--img_size`: Input image size (default: 300)
- `--num_classes`: Number of classes including background (default: 9)
- `--save_dir`: Directory to save model checkpoints
- `--resume`: Path to checkpoint to resume training

### Inference

#### Single Image
```bash
python scripts/inference.py \
    --model_path models/best_model.pth \
    --input path/to/image.jpg \
    --output_dir results \
    --save_images --save_reports --save_json
```

#### Batch Processing
```bash
python scripts/inference.py \
    --model_path models/best_model.pth \
    --input path/to/images/ \
    --output_dir results \
    --conf_threshold 0.5 \
    --save_images --save_reports
```

#### Inference Parameters
- `--model_path`: Path to trained model checkpoint
- `--input`: Path to image file or directory
- `--output_dir`: Directory to save results
- `--conf_threshold`: Confidence threshold (default: 0.5)
- `--nms_threshold`: NMS threshold (default: 0.45)
- `--save_images`: Save visualization images
- `--save_reports`: Save compliance reports
- `--save_json`: Save detection results as JSON

### Configuration

Modify `configs/ppe_config.yaml` to adjust:
- Model parameters
- Training settings
- Loss function weights
- OSHA compliance rules

## Model Architecture

### SSD300 Architecture
- **Backbone**: VGG16 with additional convolutional layers
- **Feature Maps**: Multi-scale detection (38×38, 19×19, 10×10, 5×5, 3×3, 1×1)
- **Anchor Boxes**: 8,732 default boxes with various aspect ratios
- **Output**: Bounding box coordinates and class probabilities

### Loss Function
- **Classification Loss**: Weighted cross-entropy for PPE classes
- **Localization Loss**: Smooth L1 loss for bounding box regression
- **Hard Negative Mining**: 3:1 negative to positive ratio
- **PPE Weighting**: Higher penalties for critical safety violations

## Results and Evaluation

### Performance Metrics
- **mAP (mean Average Precision)**: Overall detection accuracy
- **Compliance Rate**: Percentage of OSHA-compliant scenes
- **Critical Violation Detection**: Precision/recall for safety violations

### Sample Output
```
PPE COMPLIANCE REPORT
====================
Status: ❌ NON-COMPLIANT
Severity: HIGH
Workers Detected: 3

VIOLATIONS DETECTED:
1. Worker without hard hat (Score: 0.89)
2. Worker without safety vest (Score: 0.76)

RECOMMENDATIONS:
• IMMEDIATE ACTION REQUIRED
• Stop work until all workers have proper PPE
• Conduct safety briefing
```

## API Reference

### Dataset Class
```python
from src.dataset.ppe_dataset import PPEDataset

dataset = PPEDataset(
    data_dir='data',
    split='train',
    img_size=300
)
```

### Model Creation
```python
from src.models.ssd import build_ssd_model

model = build_ssd_model(num_classes=9)
```

### Loss Function
```python
from src.models.loss import PPELoss

criterion = PPELoss(
    priors_cxcy=priors,
    alpha=1.0,
    neg_pos_ratio=3
)
```

### Compliance Checking
```python
from src.utils.utils import check_ppe_compliance, generate_compliance_report

compliance = check_ppe_compliance(boxes, labels, scores, class_names)
report = generate_compliance_report(compliance)
```

## Project Structure

```
Image-Classification-for-PPE/
├── src/
│   ├── dataset/
│   │   ├── __init__.py
│   │   └── ppe_dataset.py      # Dataset implementation
│   ├── models/
│   │   ├── __init__.py
│   │   ├── ssd.py              # SSD model architecture
│   │   └── loss.py             # Loss functions
│   └── utils/
│       ├── __init__.py
│       └── utils.py            # Utility functions
├── scripts/
│   ├── train.py                # Training script
│   └── inference.py            # Inference script
├── configs/
│   └── ppe_config.yaml         # Configuration file
├── data/                       # Dataset directory
├── models/                     # Saved models
├── requirements.txt            # Dependencies
└── README.md                   # This file
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## OSHA Compliance Guidelines

This system is designed to help enforce OSHA construction safety standards:

### 29 CFR 1926.95 - Personal Protective Equipment
- **Head Protection**: Hard hats required in areas with potential head injuries
- **Eye Protection**: Safety glasses required when exposed to eye hazards
- **High-Visibility Clothing**: Required in areas with vehicular traffic

### Critical Safety Equipment
- **Hard Hats**: Essential for protection against falling objects
- **Safety Vests**: Critical for worker visibility
- **Safety Footwear**: Protection against foot injuries

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Based on the SSD architecture and implementation patterns from [explainingai-code/SSD-PyTorch](https://github.com/explainingai-code/SSD-PyTorch)
- OSHA safety guidelines and regulations
- PyTorch and torchvision communities

## Support

For questions, issues, or contributions, please open an issue on the GitHub repository.

---

**⚠️ Important Safety Note**: This system is designed to assist with safety monitoring but should not be the sole method for ensuring OSHA compliance. Always follow proper safety protocols and use qualified safety personnel for comprehensive safety assessments.