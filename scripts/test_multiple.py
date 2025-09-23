#!/usr/bin/env python3
"""
Simple Multi-Image Test Script
"""

import os
import sys
import glob

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from test_model_fixed import PPEInferenceEngine

def test_multiple_images():
    """Test model on first 5 images"""
    
    print("🔍 Multi-Image PPE Testing")
    print("=" * 40)
    
    # Initialize inference engine
    model_path = 'models/checkpoint_epoch_4.pth'
    inference_engine = PPEInferenceEngine(model_path)
    
    # Get first 5 images
    image_pattern = 'data/images/image*.png'
    image_files = sorted(glob.glob(image_pattern))[:5]  # First 5 images
    
    print(f"Testing {len(image_files)} images...\n")
    
    for i, image_path in enumerate(image_files, 1):
        image_name = os.path.basename(image_path)
        print(f"[{i}/{len(image_files)}] Testing: {image_name}")
        
        try:
            # Run inference
            detections, _ = inference_engine.predict(image_path, confidence_threshold=0.15)
            
            # Count violations
            violations = [d for d in detections if d['class_name'].startswith('no_')]
            
            # Print summary
            print(f"  Detections: {len(detections)}")
            print(f"  Violations: {len(violations)}")
            
            # Show top 3 detections
            if detections:
                print("  Top detections:")
                for j, det in enumerate(detections[:3]):
                    print(f"    {j+1}. {det['class_name']}: {det['confidence']*100:.1f}%")
            else:
                print("  No strong detections found")
            
            print()  # Empty line
            
        except Exception as e:
            print(f"  ❌ Error: {e}")
            print()
    
    print("✅ Multi-image testing completed!")

if __name__ == "__main__":
    test_multiple_images()