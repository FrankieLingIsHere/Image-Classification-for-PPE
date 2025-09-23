#!/usr/bin/env python3
"""
Batch Test PPE Detection Model on Multiple Images
"""

import os
import sys
import argparse
import json
from pathlib import Path
import torch

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from test_model import PPEInferenceEngine

def batch_test(model_path, test_dir, output_dir, confidence_threshold=0.3):
    """Test model on all images in directory"""
    
    print("🔍 PPE Detection Batch Testing")
    print("=" * 50)
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Initialize inference engine
    inference_engine = PPEInferenceEngine(model_path)
    
    # Find all test images
    test_dir = Path(test_dir)
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    image_files = [f for f in test_dir.iterdir() 
                   if f.suffix.lower() in image_extensions]
    
    if not image_files:
        print(f"❌ No images found in {test_dir}")
        return
    
    print(f"Found {len(image_files)} images to test")
    
    # Test each image
    all_results = []
    total_detections = 0
    total_violations = 0
    
    for i, image_path in enumerate(image_files, 1):
        print(f"\n[{i}/{len(image_files)}] Testing: {image_path.name}")
        
        try:
            # Run inference
            detections, original_image = inference_engine.predict(
                str(image_path), 
                confidence_threshold=confidence_threshold
            )
            
            # Count violations
            violations = [d for d in detections if d['class_name'].startswith('no_')]
            
            # Prepare result
            result = {
                'image_name': image_path.name,
                'image_path': str(image_path),
                'num_detections': len(detections),
                'num_violations': len(violations),
                'detections': detections,
                'violations': [v['class_name'] for v in violations]
            }
            
            all_results.append(result)
            total_detections += len(detections)
            total_violations += len(violations)
            
            # Print summary for this image
            print(f"  Objects: {len(detections)}, Violations: {len(violations)}")
            if violations:
                print(f"  ⚠️ Violations: {', '.join([v.replace('no_', '') for v in result['violations']])}")
            
            # Save visualization
            output_file = output_dir / f"result_{image_path.stem}.jpg"
            inference_engine.visualize_predictions(
                original_image, detections, str(output_file)
            )
            
        except Exception as e:
            print(f"  ❌ Error processing {image_path.name}: {e}")
            continue
    
    # Generate summary report
    generate_batch_report(all_results, output_dir, total_detections, total_violations)
    
    print(f"\n✅ Batch testing completed!")
    print(f"Results saved to: {output_dir}")

def generate_batch_report(results, output_dir, total_detections, total_violations):
    """Generate comprehensive batch testing report"""
    
    # Save detailed JSON results
    json_file = output_dir / "batch_results.json"
    with open(json_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Generate text summary
    summary_file = output_dir / "batch_summary.txt"
    
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("PPE Detection Batch Test Summary\n")
        f.write("=" * 50 + "\n\n")
        
        # Overall statistics
        total_images = len(results)
        images_with_detections = sum(1 for r in results if r['num_detections'] > 0)
        images_with_violations = sum(1 for r in results if r['num_violations'] > 0)
        
        f.write("Overall Statistics:\n")
        f.write("  Total Images Tested: {}\n".format(total_images))
        f.write("  Images with Detections: {} ({:.1f}%)\n".format(images_with_detections, images_with_detections/total_images*100))
        f.write("  Images with Violations: {} ({:.1f}%)\n".format(images_with_violations, images_with_violations/total_images*100))
        f.write("  Total Objects Detected: {}\n".format(total_detections))
        f.write("  Total Safety Violations: {}\n\n".format(total_violations))
        
        # Safety compliance rate
        compliance_rate = ((total_images - images_with_violations) / total_images * 100) if total_images > 0 else 0
        f.write("Safety Compliance Rate: {:.1f}%\n\n".format(compliance_rate))
        
        # Class distribution
        class_counts = {}
        for result in results:
            for detection in result['detections']:
                class_name = detection['class_name']
                class_counts[class_name] = class_counts.get(class_name, 0) + 1
        
        f.write("Detection Distribution:\n")
        for class_name, count in sorted(class_counts.items()):
            f.write("  {}: {}\n".format(class_name, count))
        
        # Violation analysis
        violation_counts = {}
        for result in results:
            for violation in result['violations']:
                violation_counts[violation] = violation_counts.get(violation, 0) + 1
        
        if violation_counts:
            f.write("\nViolation Breakdown:\n")
            for violation, count in sorted(violation_counts.items()):
                f.write("  {}: {}\n".format(violation.replace('no_', 'Missing '), count))
        
        # Individual image results
        f.write("\nIndividual Results:\n")
        f.write("-" * 60 + "\n")
        
        for result in results:
            f.write("{}: {} objects".format(result['image_name'], result['num_detections']))
            if result['num_violations'] > 0:
                f.write(", {} violations".format(result['num_violations']))
            f.write("\n")
    
    print("Summary report saved to: {}".format(summary_file))

def main():
    parser = argparse.ArgumentParser(description='Batch Test PPE Detection Model')
    parser.add_argument('--model', type=str, default='models/checkpoint_epoch_4.pth',
                       help='Path to trained model checkpoint')
    parser.add_argument('--test_dir', type=str, default='data/images',
                       help='Directory containing test images')
    parser.add_argument('--output_dir', type=str, default='test_results',
                       help='Directory to save results')
    parser.add_argument('--confidence', type=float, default=0.3,
                       help='Confidence threshold')
    
    args = parser.parse_args()
    
    # Check if model exists
    if not os.path.exists(args.model):
        print(f"❌ Model not found: {args.model}")
        return
    
    # Check if test directory exists
    if not os.path.exists(args.test_dir):
        print(f"❌ Test directory not found: {args.test_dir}")
        return
    
    # Run batch testing
    batch_test(args.model, args.test_dir, args.output_dir, args.confidence)

if __name__ == "__main__":
    main()