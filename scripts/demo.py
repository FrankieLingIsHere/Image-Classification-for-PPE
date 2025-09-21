#!/usr/bin/env python3
"""
Example script demonstrating PPE detection system usage

This script shows how to:
1. Create and test the model
2. Set up the dataset structure
3. Run basic inference
4. Generate compliance reports
"""

import os
import sys
import torch
import numpy as np

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from dataset.ppe_dataset import PPEDataset, create_sample_data_structure
from models.ssd import build_ssd_model
from models.loss import PPELoss, create_prior_boxes
from utils.utils import check_ppe_compliance, generate_compliance_report


def demo_model_creation():
    """Demonstrate model creation and basic forward pass"""
    print("=" * 60)
    print("DEMO 1: Model Creation and Testing")
    print("=" * 60)
    
    # Create SSD model
    print("Creating SSD300 model for PPE detection...")
    model = build_ssd_model(num_classes=9)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Model created successfully!")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Test forward pass
    print("\nTesting forward pass...")
    dummy_input = torch.randn(1, 3, 300, 300)
    
    model.eval()
    with torch.no_grad():
        locs, scores = model(dummy_input)
    
    print(f"Forward pass successful!")
    print(f"Locations output shape: {locs.shape}")
    print(f"Scores output shape: {scores.shape}")
    print(f"Number of anchor boxes: {locs.shape[1]}")
    
    return model


def demo_loss_function():
    """Demonstrate loss function computation"""
    print("\n" + "=" * 60)
    print("DEMO 2: Loss Function Testing")
    print("=" * 60)
    
    # Create loss function
    print("Creating PPE loss function...")
    priors = create_prior_boxes()
    criterion = PPELoss(priors)
    
    print(f"Prior boxes created: {priors.shape}")
    print("Loss function initialized with PPE-specific weights")
    
    # Test loss computation
    print("\nTesting loss computation...")
    batch_size = 2
    
    # Dummy predictions
    pred_locs = torch.randn(batch_size, 8732, 4)
    pred_scores = torch.randn(batch_size, 8732, 9)
    
    # Dummy ground truth
    boxes = [
        torch.tensor([[0.1, 0.1, 0.3, 0.3], [0.5, 0.5, 0.8, 0.8]]),  # Image 1: 2 objects
        torch.tensor([[0.2, 0.2, 0.4, 0.4]])  # Image 2: 1 object
    ]
    labels = [
        torch.tensor([2, 3]),  # hard_hat, safety_vest
        torch.tensor([7])      # no_hard_hat (violation)
    ]
    
    # Compute loss
    loss = criterion(pred_locs, pred_scores, boxes, labels)
    print(f"Loss computation successful!")
    print(f"Loss value: {loss.item():.4f}")
    
    return criterion


def demo_dataset_setup():
    """Demonstrate dataset setup and structure"""
    print("\n" + "=" * 60)
    print("DEMO 3: Dataset Setup")
    print("=" * 60)
    
    data_dir = "data"
    
    # Create sample data structure
    print(f"Creating sample data structure in '{data_dir}'...")
    create_sample_data_structure(data_dir)
    
    # Test dataset creation
    print("Testing dataset creation...")
    try:
        dataset = PPEDataset(data_dir, split='train')
        print(f"Dataset created successfully!")
        print(f"Number of images: {len(dataset)}")
        print(f"Number of classes: {dataset.get_num_classes()}")
        print(f"Class names: {dataset.get_class_names()}")
        
        # Show expected directory structure
        print("\nExpected directory structure:")
        print("data/")
        print("├── images/              # Place your .jpg/.png images here")
        print("├── annotations/         # Place your .xml/.json annotations here")
        print("└── splits/              # Created automatically")
        print("    ├── train.txt        # List training image names")
        print("    ├── val.txt          # List validation image names")
        print("    └── test.txt         # List test image names")
        
    except Exception as e:
        print(f"Dataset creation failed: {e}")
        print("This is expected if no data is provided yet.")
    
    return data_dir


def demo_compliance_checking():
    """Demonstrate PPE compliance checking"""
    print("\n" + "=" * 60)
    print("DEMO 4: PPE Compliance Checking")
    print("=" * 60)
    
    # Class names
    class_names = [
        'background', 'person', 'hard_hat', 'safety_vest', 'safety_gloves',
        'safety_boots', 'eye_protection', 'no_hard_hat', 'no_safety_vest'
    ]
    
    # Scenario 1: Compliant scene
    print("Scenario 1: COMPLIANT scene")
    print("-" * 30)
    boxes1 = np.array([[0.1, 0.1, 0.3, 0.3], [0.2, 0.2, 0.4, 0.4], [0.3, 0.3, 0.5, 0.5]])
    labels1 = np.array([1, 2, 3])  # person, hard_hat, safety_vest
    scores1 = np.array([0.95, 0.89, 0.87])
    
    compliance1 = check_ppe_compliance(boxes1, labels1, scores1, class_names)
    report1 = generate_compliance_report(compliance1)
    print(report1)
    
    # Scenario 2: Non-compliant scene with violations
    print("\nScenario 2: NON-COMPLIANT scene")
    print("-" * 35)
    boxes2 = np.array([[0.1, 0.1, 0.3, 0.3], [0.5, 0.5, 0.7, 0.7]])
    labels2 = np.array([1, 7])  # person, no_hard_hat
    scores2 = np.array([0.92, 0.84])
    
    compliance2 = check_ppe_compliance(boxes2, labels2, scores2, class_names)
    report2 = generate_compliance_report(compliance2)
    print(report2)


def demo_training_command():
    """Show example training commands"""
    print("\n" + "=" * 60)
    print("DEMO 5: Training Commands")
    print("=" * 60)
    
    print("To train the model (after adding your data), use:")
    print()
    
    print("Basic training:")
    print("python scripts/train.py --data_dir data --batch_size 8 --epochs 100")
    print()
    
    print("Advanced training with custom settings:")
    print("python scripts/train.py \\")
    print("    --data_dir data \\")
    print("    --batch_size 16 \\")
    print("    --epochs 200 \\")
    print("    --lr 0.001 \\")
    print("    --save_dir models/experiment_1 \\")
    print("    --log_dir logs/experiment_1")
    print()
    
    print("Resume training from checkpoint:")
    print("python scripts/train.py --resume models/checkpoint_epoch_50.pth")
    print()


def demo_inference_command():
    """Show example inference commands"""
    print("\n" + "=" * 60)
    print("DEMO 6: Inference Commands")
    print("=" * 60)
    
    print("To run inference (after training), use:")
    print()
    
    print("Single image inference:")
    print("python scripts/inference.py \\")
    print("    --model_path models/best_model.pth \\")
    print("    --input path/to/image.jpg \\")
    print("    --output_dir results \\")
    print("    --save_images --save_reports --save_json")
    print()
    
    print("Batch processing:")
    print("python scripts/inference.py \\")
    print("    --model_path models/best_model.pth \\")
    print("    --input path/to/images/ \\")
    print("    --output_dir results \\")
    print("    --conf_threshold 0.5 \\")
    print("    --save_images --save_reports")
    print()


def main():
    """Run all demonstrations"""
    print("PPE Detection System Demo")
    print("This script demonstrates the key components of the system")
    print()
    
    # Run demonstrations
    model = demo_model_creation()
    criterion = demo_loss_function()
    data_dir = demo_dataset_setup()
    demo_compliance_checking()
    demo_training_command()
    demo_inference_command()
    
    print("\n" + "=" * 60)
    print("NEXT STEPS")
    print("=" * 60)
    print("1. Add your images to data/images/")
    print("2. Add your annotations to data/annotations/")
    print("3. Update the split files in data/splits/")
    print("4. Run training: python scripts/train.py --data_dir data")
    print("5. Run inference: python scripts/inference.py --model_path models/best_model.pth")
    print()
    print("For detailed instructions, see README.md")
    print()
    print("Demo completed successfully! 🎉")


if __name__ == "__main__":
    main()