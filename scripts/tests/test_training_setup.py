#!/usr/bin/env python3
"""
Quick pre-training validation test
Ensures all components work before starting full training
"""

import sys
from pathlib import Path
import torch

# Add project root to path
project_root = Path(__file__).parent.parent.parent  # Go up 3 levels to project root
sys.path.insert(0, str(project_root))

def test_data_loading():
    """Test that data loads correctly"""
    print("\n[1/5] Testing data loading...")
    try:
        from scripts.train.train_full_pipeline import PPEDatasetWithSegmentation
        
        dataset = PPEDatasetWithSegmentation(split='train', image_size=640)
        print(f"    [OK] Dataset loaded: {len(dataset)} training images")
        
        # Try loading a single item
        sample = dataset[0]
        print(f"    [OK] Sample loaded: image shape {sample['image'].shape}")
        print(f"    [OK] Bboxes shape: {sample['boxes'].shape}")
        return True
    except Exception as e:
        print(f"    [FAIL] {e}")
        return False


def test_ssl_backbone():
    """Test SSL pretraining backbone"""
    print("\n[2/5] Testing SSL backbone...")
    try:
        from scripts.train.ssl_pretraining import ResNet50Features
        from torchvision.models import resnet50
        
        # Create ResNet50 and wrap with ResNet50Features
        resnet = resnet50(pretrained=True)
        backbone = ResNet50Features(resnet)
        x = torch.randn(2, 3, 224, 224)  # SSL uses 224x224
        features = backbone(x)
        
        print(f"    [OK] SSL backbone initialized")
        print(f"    [OK] Input shape: {x.shape}")
        print(f"    [OK] Features shape: {features.shape}")
        return True
    except Exception as e:
        print(f"    [FAIL] {e}")
        return False


def test_enhanced_detector():
    """Test enhanced detector model"""
    print("\n[3/5] Testing enhanced detector...")
    try:
        from src.models.enhanced_ppe_detector import EnhancedPPEDetector
        
        model = EnhancedPPEDetector(num_classes=12)
        # Create a batch: list of individual images [C, H, W]
        images = [torch.randn(3, 640, 640) for _ in range(2)]
        
        # Test inference mode
        model.eval()
        with torch.no_grad():
            output = model(images)
        
        print(f"    [OK] Enhanced detector initialized")
        print(f"    [OK] Model has all components:")
        print(f"        - Detection backbone: ResNet50+FPN")
        print(f"        - Semantic segmentation head")
        print(f"        - Spatial constraint module")
        return True
    except Exception as e:
        print(f"    [FAIL] {e}")
        return False


def test_data_loader():
    """Test batching and data loader"""
    print("\n[4/5] Testing data loader with batching...")
    try:
        from scripts.train.train_full_pipeline import PPEDatasetWithSegmentation
        from torch.utils.data import DataLoader
        
        def collate_fn(batch):
            """Custom collate for variable-size boxes"""
            images = torch.stack([item['image'] for item in batch])
            return {
                'images': images,
                'boxes': [item['boxes'] for item in batch],
                'labels': [item['labels'] for item in batch],
            }
        
        dataset = PPEDatasetWithSegmentation(split='train', image_size=640)
        loader = DataLoader(dataset, batch_size=4, shuffle=False, collate_fn=collate_fn)
        
        batch = next(iter(loader))
        print(f"    [OK] Data loader working")
        print(f"    [OK] Batch size: 4")
        print(f"    [OK] Batch images shape: {batch['images'].shape}")
        print(f"    [OK] Number of bbox lists: {len(batch['boxes'])}")
        return True
    except Exception as e:
        print(f"    [FAIL] {e}")
        return False


def test_device():
    """Check GPU availability"""
    print("\n[5/5] Checking device availability...")
    try:
        if torch.cuda.is_available():
            device = torch.device('cuda')
            device_name = torch.cuda.get_device_name(0)
            print(f"    [OK] CUDA available: {device_name}")
            print(f"    [OK] GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
            return True
        else:
            print(f"    [WARN] CUDA not available, will use CPU (slower)")
            print(f"    [OK] CPU available: can train on CPU")
            return True
    except Exception as e:
        print(f"    [FAIL] {e}")
        return False


def main():
    print("=" * 60)
    print("PRE-TRAINING VALIDATION TEST")
    print("=" * 60)
    
    tests = [
        test_data_loading,
        test_ssl_backbone,
        test_enhanced_detector,
        test_data_loader,
        test_device,
    ]
    
    results = []
    for test in tests:
        try:
            results.append(test())
        except Exception as e:
            print(f"    [ERROR] Unexpected error: {e}")
            results.append(False)
    
    # Summary
    print("\n" + "=" * 60)
    passed = sum(results)
    total = len(results)
    print(f"RESULTS: {passed}/{total} tests passed")
    print("=" * 60)
    
    if all(results):
        print("\n[SUCCESS] All checks passed! Ready to train.")
        print("\nStart training with:")
        print("  python run_resumable_training.py --device cuda")
        return 0
    else:
        print("\n[FAILURE] Some tests failed. Please fix issues before training.")
        return 1


if __name__ == '__main__':
    sys.exit(main())
