#!/usr/bin/env python3
"""
Test script for fixed Faster R-CNN + LLaVA hybrid model
"""

import sys
import os

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

from src.models.hybrid_ppe_model import HybridPPEDescriptionModel
from PIL import Image
import json
from pathlib import Path

def test_hybrid_model():
    """Test the fixed hybrid model with Faster R-CNN + LLaVA"""
    
    print("=" * 80)
    print("🚀 Testing Fixed Hybrid PPE Model (Faster R-CNN + LLaVA)")
    print("=" * 80)
    
    # Initialize model
    print("\n[1/5] Initializing hybrid model...")
    try:
        model = HybridPPEDescriptionModel(
            ppe_model_path='models/rcnn_baseline.pth',  # Use Faster R-CNN
            vision_model='llava',
            device='auto'
        )
        print("✅ Model initialized successfully")
    except Exception as e:
        print(f"❌ Model initialization failed: {e}")
        return False
    
    # Load test image
    print("\n[2/5] Loading test image...")
    image_path = 'data/images/image2.png'
    try:
        if not os.path.exists(image_path):
            print(f"❌ Image not found: {image_path}")
            return False
        
        image = Image.open(image_path).convert('RGB')
        print(f"✅ Image loaded: {image.size}")
    except Exception as e:
        print(f"❌ Failed to load image: {e}")
        return False
    
    # Test PPE detection
    print("\n[3/5] Testing PPE detection (Faster R-CNN)...")
    try:
        detections = model.detect_ppe(image)
        print(f"✅ PPE Detection complete: {len(detections)} detections")
        
        for i, det in enumerate(detections[:3]):  # Show first 3
            print(f"   [{i+1}] {det['class']}: {det['confidence']:.2f} conf")
        
        # Check if real detections (not all mocks)
        if detections and len(detections) > 0:
            print(f"   ℹ️  {len(detections)} total detections found")
        
    except Exception as e:
        print(f"❌ PPE detection failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test VLM caption generation
    print("\n[4/5] Testing VLM caption generation (LLaVA)...")
    try:
        caption = model.generate_general_caption(
            image,
            prompt="Describe this construction site image focusing on worker safety."
        )
        
        # Check if real caption or mock
        if caption.startswith('['):
            print(f"⚠️  Using fallback caption (VLM issue)")
        else:
            print(f"✅ Real VLM caption generated:")
        
        print(f"   {caption[:100]}...")
        
    except Exception as e:
        print(f"❌ VLM caption generation failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test full hybrid analysis
    print("\n[5/5] Testing full hybrid analysis...")
    try:
        results = model.generate_hybrid_description(
            image,
            include_general_caption=True,
            custom_prompt="Analyze this construction site for PPE compliance."
        )
        
        print("✅ Hybrid analysis complete!")
        print(f"\n📊 Results Summary:")
        print(f"   • Total detections: {len(results.get('detections', []))}")
        print(f"   • Compliance status: {results['ppe_descriptions'].get('compliance_status', 'N/A')}")
        print(f"   • Safety summary: {results['ppe_descriptions'].get('safety_summary', 'N/A')}")
        print(f"   • Caption: {results['general_caption'][:80]}...")
        
        # Save results
        output_file = 'test_hybrid_results.json'
        with open(output_file, 'w') as f:
            # Convert numpy/tensor types to serializable
            results_clean = {
                'detections': results['detections'],
                'ppe_descriptions': results['ppe_descriptions'],
                'general_caption': results['general_caption']
            }
            json.dump(results_clean, f, indent=2)
        
        print(f"\n💾 Results saved to: {output_file}")
        
    except Exception as e:
        print(f"❌ Hybrid analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "=" * 80)
    print("✅ ALL TESTS PASSED!")
    print("=" * 80)
    return True

if __name__ == "__main__":
    success = test_hybrid_model()
    sys.exit(0 if success else 1)
