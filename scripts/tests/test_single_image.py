#!/usr/bin/env python3
"""
Simple script to test a single image with different detection thresholds
"""

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from scripts.evaluate_detection_performance import PPEDetectionEvaluator
from pathlib import Path

def test_single_image():
    """Test detection on a single image with different thresholds"""
    
    # Find the latest model
    models_dir = Path("models")
    model_files = list(models_dir.glob("*.pth"))
    if not model_files:
        print("❌ No model files found in models directory")
        return
    
    latest_model = max(model_files, key=lambda x: x.stat().st_ctime)
    
    # Find a test image
    test_images_dir = Path("data/images")
    test_images = list(test_images_dir.glob("*.jpg")) + list(test_images_dir.glob("*.png"))
    if not test_images:
        print("❌ No test images found")
        return
    
    test_image = test_images[0]  # Use first available image
    print(f"🖼️  Testing with image: {test_image.name}")
    
    # Test different thresholds
    thresholds = [0.5, 0.3, 0.1, 0.05, 0.02]
    
    for threshold in thresholds:
        print(f"\n🧪 Testing threshold: {threshold}")
        
        try:
            # Create evaluator
            evaluator = PPEDetectionEvaluator(
                model_path=str(latest_model),
                data_dir="data", 
                config_path="configs/ppe_config.yaml",
                output_dir="outputs/single_test"
            )
            
            # Set threshold
            evaluator.conf_threshold = threshold
            
            # Run detection on single image
            detections = evaluator._detect_image(test_image)
            
            print(f"   🎯 Found {len(detections)} detections:")
            
            # Count by class
            class_counts = {}
            for det in detections:
                class_name = det.get('class') or det.get('class_name')
                class_counts[class_name] = class_counts.get(class_name, 0) + 1
            
            for class_name, count in class_counts.items():
                print(f"      {class_name}: {count}")
            
            if not detections:
                print("      (No detections)")
                
        except Exception as e:
            print(f"   ❌ Error: {e}")

if __name__ == "__main__":
    print("🔍 Single Image Detection Test")
    print("Testing different confidence thresholds...")
    test_single_image()
