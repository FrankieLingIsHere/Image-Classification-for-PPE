#!/usr/bin/env python3
"""
Debug script to check SSD model output dimensions
"""

import torch
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.models.ssd import SSD300

def debug_model_dimensions():
    """Check model output dimensions"""
    
    print("🔍 Debugging SSD Model Dimensions")
    print("=" * 50)
    
    # Create model
    model = SSD300(n_classes=13)
    model.eval()
    
    # Create dummy input
    batch_size = 2
    dummy_input = torch.randn(batch_size, 3, 300, 300)
    
    print(f"Input shape: {dummy_input.shape}")
    
    # Forward pass
    with torch.no_grad():
        try:
            locs, class_scores = model(dummy_input)
            
            print(f"\nModel outputs:")
            print(f"  Locations: {locs.shape}")
            print(f"  Class scores: {class_scores.shape}")
            
            expected_boxes = 8732
            actual_boxes = locs.shape[1]
            
            print(f"\nBox count comparison:")
            print(f"  Expected: {expected_boxes}")
            print(f"  Actual: {actual_boxes}")
            print(f"  Difference: {actual_boxes - expected_boxes}")
            
            if actual_boxes != expected_boxes:
                print(f"\n❌ MISMATCH DETECTED!")
                print(f"   This explains the training error.")
                
                # Let's debug the individual feature maps
                print(f"\n🔍 Debugging individual feature maps...")
                debug_feature_maps(model, dummy_input)
            else:
                print(f"\n✅ Dimensions match!")
                
        except Exception as e:
            print(f"❌ Model forward pass failed: {e}")

def debug_feature_maps(model, dummy_input):
    """Debug individual feature map sizes"""
    
    # Get VGG backbone
    vgg = model.base
    
    # Get auxiliary convolutions  
    aux = model.aux_convs
    
    # Get prediction convolutions
    pred = model.pred_convs
    
    print("\nFeature map analysis:")
    
    with torch.no_grad():
        # Forward through VGG to get conv4_3 and conv7
        x = dummy_input
        
        # Forward through VGG layers to get conv4_3
        for i in range(23):  # Up to conv4_3
            x = vgg[i](x)
        conv4_3_feats = x
        print(f"  conv4_3: {conv4_3_feats.shape}")
        
        # Continue to conv7
        for i in range(23, len(vgg)):
            x = vgg[i](x)
        conv7_feats = x
        print(f"  conv7: {conv7_feats.shape}")
        
        # Get auxiliary feature maps
        aux_feats = aux(conv7_feats)
        for i, feat in enumerate(aux_feats):
            print(f"  aux_{i+8}: {feat.shape}")
        
        # Calculate expected vs actual boxes per feature map
        feature_map_info = [
            ("conv4_3", conv4_3_feats, 4),
            ("conv7", conv7_feats, 6),
            ("conv8_2", aux_feats[0], 6),
            ("conv9_2", aux_feats[1], 6),
            ("conv10_2", aux_feats[2], 4),
            ("conv11_2", aux_feats[3], 4),
        ]
        
        print(f"\nBox count per feature map:")
        total_actual = 0
        total_expected = 8732
        
        for name, feat_map, n_boxes in feature_map_info:
            h, w = feat_map.shape[2], feat_map.shape[3]
            actual_boxes = h * w * n_boxes
            total_actual += actual_boxes
            print(f"  {name}: {h}x{w}x{n_boxes} = {actual_boxes} boxes")
        
        print(f"\nTotals:")
        print(f"  Expected: {total_expected}")
        print(f"  Actual: {total_actual}")
        print(f"  Difference: {total_actual - total_expected}")

def check_prior_boxes():
    """Check if prior boxes generation matches"""
    
    print(f"\n🔍 Checking prior boxes generation...")
    
    try:
        # This should match the SSD model output
        from src.models.loss import create_prior_boxes
        
        priors = create_prior_boxes()
        print(f"  Prior boxes shape: {priors.shape}")
        print(f"  Number of priors: {priors.shape[0]}")
        
    except Exception as e:
        print(f"  ❌ Could not load prior boxes: {e}")

def main():
    debug_model_dimensions()
    check_prior_boxes()
    
    print(f"\n💡 Next steps:")
    print(f"   If dimensions don't match, we need to:")
    print(f"   1. Fix the model architecture")
    print(f"   2. Update the prior box generation")
    print(f"   3. Ensure loss function compatibility")

if __name__ == "__main__":
    main()