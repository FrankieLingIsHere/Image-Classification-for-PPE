#!/usr/bin/env python3
"""
Quick test script to see the enhanced BLIP-2 descriptions
"""

from src.models.hybrid_ppe_model import HybridPPEModel
import cv2
import json
from datetime import datetime

def test_enhanced_descriptions():
    """Test the enhanced description generation"""
    
    print("🚀 Testing Enhanced PPE Description Model...")
    
    # Initialize model
    model = HybridPPEModel(
        ppe_model_path='models/best_model_regularized.pth',
        device='cpu'
    )
    
    # Load test image
    image_path = 'data/images/image2.png'
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"❌ Could not load image: {image_path}")
        return
    
    print(f"🖼️ Analyzing: {image_path}")
    
    # Analyze image
    results = model.analyze_image(image)
    
    print("\n" + "="*60)
    print("🎯 ENHANCED VISION-LANGUAGE DESCRIPTIONS")
    print("="*60)
    
    # Extract and display components
    visual_desc = results.get('visual_description', 'No description')
    print(f"\n📸 Visual Description:\n{visual_desc}")
    
    ppe_analysis = results.get('ppe_analysis', {})
    
    safety_summary = ppe_analysis.get('safety_summary', 'No summary')
    print(f"\n🛡️ Safety Summary:\n{safety_summary}")
    
    detailed_analysis = ppe_analysis.get('detailed_analysis', 'No analysis')
    print(f"\n🔍 Detailed Analysis:\n{detailed_analysis}")
    
    safety_analysis = results.get('safety_analysis', 'No safety analysis')
    print(f"\n⚠️ Safety Analysis:\n{safety_analysis}")
    
    # Show detection counts
    detections = ppe_analysis.get('detections', [])
    print(f"\n📊 Detection Summary:")
    print(f"   Total detections: {len(detections)}")
    for detection in detections[:5]:  # Show first 5
        print(f"   - {detection['class']}: {detection['confidence']:.3f}")
    
    # Save detailed results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"enhanced_description_test_{timestamp}.json"
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Full results saved to: {output_file}")
    print("\n✅ Enhanced description test completed!")

if __name__ == "__main__":
    test_enhanced_descriptions()