#!/usr/bin/env python3
"""
Test the hybrid model's PPE classification logic
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from PIL import Image
from src.models.hybrid_ppe_model import HybridPPEDescriptionModel
import argparse

def analyze_classification_logic(image_path, ppe_model_path):
    """Analyze how the model classifies good vs bad PPE compliance"""
    
    print("🔍 Testing PPE Classification Logic")
    print("=" * 50)
    
    # Initialize model
    model = HybridPPEDescriptionModel(
        ppe_model_path=ppe_model_path,
        vision_model="blip2",
        device="auto"
    )
    
    # Load and process image
    image = Image.open(image_path)
    print(f"📸 Analyzing: {os.path.basename(image_path)}")
    print(f"   Image size: {image.size}")
    
    # Get raw detections
    detections = model.detect_ppe(image)
    
    print(f"\n🔍 Raw Detections Found: {len(detections)}")
    print("-" * 30)
    
    # Analyze classification categories
    positive_ppe = []  # Good PPE items
    violations = []    # Bad/missing PPE
    people = []        # Person detections
    
    for detection in detections:
        class_name = detection['class']
        confidence = detection['confidence']
        
        if class_name == 'person':
            people.append(detection)
        elif class_name.startswith('no_'):
            violations.append(detection)
        elif class_name in ['hard_hat', 'safety_vest', 'safety_gloves', 'safety_boots', 'eye_protection']:
            positive_ppe.append(detection)
        
        print(f"  {class_name}: confidence {confidence:.3f}")
    
    print(f"\n📊 Classification Analysis:")
    print("-" * 30)
    print(f"👥 People detected: {len(people)}")
    print(f"✅ Positive PPE items: {len(positive_ppe)}")
    print(f"❌ PPE violations: {len(violations)}")
    
    if positive_ppe:
        print(f"\n✅ Good PPE Detected:")
        for item in positive_ppe:
            print(f"   • {item['class']}: {item['confidence']:.3f}")
    
    if violations:
        print(f"\n❌ PPE Violations Detected:")
        for violation in violations:
            print(f"   • {violation['class']}: {violation['confidence']:.3f}")
    
    # Test the description logic
    print(f"\n📋 Description Generation Test:")
    print("-" * 30)
    
    descriptions = model.generate_ppe_focused_description(detections)
    
    print(f"Safety Summary: {descriptions['safety_summary']}")
    print(f"Compliance Status: {descriptions['compliance_status']}")
    
    # Analyze classification logic
    print(f"\n🧠 Classification Logic Analysis:")
    print("-" * 30)
    
    # Check for contradictions
    ppe_types_positive = set([item['class'] for item in positive_ppe])
    violation_types = set([v['class'].replace('no_', '') for v in violations])
    
    contradictions = ppe_types_positive.intersection(violation_types)
    if contradictions:
        print(f"⚠️ CONTRADICTIONS FOUND: {contradictions}")
        print(f"   Model detected both presence AND absence of: {', '.join(contradictions)}")
    else:
        print(f"✅ No logical contradictions found")
    
    # Overall assessment logic
    if len(people) == 0:
        overall = "No people to assess"
    elif len(violations) == 0:
        overall = "COMPLIANT - No violations detected"
    else:
        critical_violations = [v for v in violations if v['class'] in ['no_hard_hat', 'no_safety_vest']]
        if critical_violations:
            overall = "CRITICAL NON-COMPLIANCE"
        else:
            overall = "MINOR NON-COMPLIANCE"
    
    print(f"🎯 Overall Assessment: {overall}")
    
    return {
        'people_count': len(people),
        'positive_ppe_count': len(positive_ppe),
        'violations_count': len(violations),
        'contradictions': list(contradictions),
        'overall_assessment': overall
    }

def main():
    parser = argparse.ArgumentParser(description="Test PPE classification logic")
    parser.add_argument('--image', required=True, help='Path to test image')
    parser.add_argument('--ppe-model', required=True, help='Path to PPE model')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.image):
        print(f"Error: Image file not found: {args.image}")
        return
    
    if not os.path.exists(args.ppe_model):
        print(f"Error: PPE model file not found: {args.ppe_model}")
        return
    
    result = analyze_classification_logic(args.image, args.ppe_model)
    
    print(f"\n📈 Summary Statistics:")
    print(f"   People: {result['people_count']}")
    print(f"   Good PPE: {result['positive_ppe_count']}")
    print(f"   Violations: {result['violations_count']}")
    print(f"   Contradictions: {len(result['contradictions'])}")
    print(f"   Assessment: {result['overall_assessment']}")

if __name__ == "__main__":
    main()