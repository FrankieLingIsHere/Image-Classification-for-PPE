#!/usr/bin/env python3
"""
Demo script for Hybrid PPE Description Model
Tests the combined PPE detection + general scene description
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.models.hybrid_ppe_model import HybridPPEDescriptionModel
from PIL import Image
import json
import argparse
from pathlib import Path

def test_with_real_image(image_path: str, model: HybridPPEDescriptionModel):
    """Test the model with a real construction site image"""
    
    print(f"🖼️ Loading image: {image_path}")
    
    try:
        image = Image.open(image_path).convert('RGB')
        print(f"   Image size: {image.size}")
        
        # Generate comprehensive description
        print("\n🔍 Analyzing construction site image...")
        results = model.generate_hybrid_description(
            image,
            include_general_caption=True,
            custom_prompt="Describe this construction site image, focusing on worker safety, PPE compliance, and work activities."
        )
        
        # Display results
        print("\n" + "="*80)
        print("HYBRID PPE ANALYSIS RESULTS")
        print("="*80)
        
        print(f"\n🎯 GENERAL SCENE DESCRIPTION:")
        print(f"   {results['general_caption']}")
        
        print(f"\n🦺 SAFETY SUMMARY:")
        print(f"   {results['ppe_descriptions']['safety_summary']}")
        
        print(f"\n⚖️ COMPLIANCE STATUS:")
        print(f"   {results['ppe_descriptions']['compliance_status']}")
        
        print(f"\n📋 DETAILED TECHNICAL ANALYSIS:")
        print(f"   {results['ppe_descriptions']['detailed_analysis']}")
        
        print(f"\n🔗 COMPLETE HYBRID DESCRIPTION:")
        print("-" * 60)
        print(results['hybrid_description'])
        print("-" * 60)
        
        # Save results
        output_file = f"ppe_analysis_{Path(image_path).stem}.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n💾 Results saved to '{output_file}'")
        return results
        
    except Exception as e:
        print(f"❌ Error processing image: {e}")
        return None

def batch_process_images(image_dir: str, model: HybridPPEDescriptionModel):
    """Process all images in a directory"""
    
    image_path = Path(image_dir)
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    
    image_files = [
        f for f in image_path.iterdir() 
        if f.suffix.lower() in image_extensions
    ]
    
    if not image_files:
        print(f"❌ No images found in {image_dir}")
        return
    
    print(f"📁 Found {len(image_files)} images to process")
    
    all_results = []
    
    for i, image_file in enumerate(image_files, 1):
        print(f"\n{'='*20} Processing {i}/{len(image_files)} {'='*20}")
        
        results = test_with_real_image(str(image_file), model)
        if results:
            results['image_file'] = str(image_file)
            all_results.append(results)
    
    # Save batch results
    batch_output = "batch_ppe_analysis_results.json"
    with open(batch_output, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n🎉 Batch processing complete! Results saved to '{batch_output}'")
    
    # Generate summary report
    generate_summary_report(all_results)

def generate_summary_report(results_list):
    """Generate a summary report from batch analysis"""
    
    print("\n" + "="*80)
    print("BATCH ANALYSIS SUMMARY REPORT")
    print("="*80)
    
    total_images = len(results_list)
    compliant_images = 0
    critical_violations = 0
    minor_violations = 0
    
    for result in results_list:
        compliance = result['ppe_descriptions']['compliance_status']
        if "COMPLIANT" in compliance:
            compliant_images += 1
        elif "CRITICAL" in compliance:
            critical_violations += 1
        elif "MINOR" in compliance:
            minor_violations += 1
    
    print(f"\n📊 OVERALL STATISTICS:")
    print(f"   Total Images Analyzed: {total_images}")
    print(f"   Compliant Sites: {compliant_images} ({compliant_images/total_images*100:.1f}%)")
    print(f"   Critical Violations: {critical_violations} ({critical_violations/total_images*100:.1f}%)")
    print(f"   Minor Violations: {minor_violations} ({minor_violations/total_images*100:.1f}%)")
    
    print(f"\n🚨 SAFETY SCORE: {compliant_images/total_images*100:.1f}%")
    
    if critical_violations > 0:
        print(f"⚠️ URGENT ACTION REQUIRED: {critical_violations} sites with critical safety violations")

def main():
    parser = argparse.ArgumentParser(description="Demo Hybrid PPE Description Model")
    parser.add_argument("--image", type=str, help="Single image to analyze")
    parser.add_argument("--batch", type=str, help="Directory of images to analyze")
    parser.add_argument("--model", type=str, choices=["blip2", "llava"], default="blip2",
                       help="Vision-language model to use")
    parser.add_argument("--ppe-model", type=str, help="Path to trained PPE detection model")
    
    args = parser.parse_args()
    
    if not args.image and not args.batch:
        # Default demo with data directory
        args.batch = "data/images" if os.path.exists("data/images") else None
        if not args.batch:
            print("Please provide --image or --batch argument")
            return
    
    print("🚀 Initializing Hybrid PPE Description Model...")
    print(f"   Vision Model: {args.model}")
    print(f"   PPE Model: {args.ppe_model or 'Mock Detection (for demo)'}")
    
    try:
        # Initialize the hybrid model
        model = HybridPPEDescriptionModel(
            ppe_model_path=args.ppe_model,
            vision_model=args.model,
            device="auto"
        )
        
        print("✅ Model initialized successfully!")
        
        if args.image:
            # Single image analysis
            test_with_real_image(args.image, model)
            
        elif args.batch:
            # Batch processing
            batch_process_images(args.batch, model)
            
    except Exception as e:
        print(f"❌ Error initializing model: {e}")
        print("\n💡 Make sure you have installed the requirements:")
        print("   pip install transformers accelerate torch-audio")

if __name__ == "__main__":
    main()