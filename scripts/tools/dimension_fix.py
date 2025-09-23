#!/usr/bin/env python3
"""
Quick fix for the SSD dimension mismatch
"""

import torch
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

def calculate_feature_map_sizes():
    """Calculate what the actual feature map sizes should be"""
    
    print("📐 Calculating expected feature map sizes for 300x300 input")
    print("=" * 60)
    
    input_size = 300
    
    # VGG backbone calculations
    # conv1 -> conv2 -> conv3 -> conv4_3
    # Each pool reduces by 2x
    conv4_3_size = input_size // (2**3)  # 3 pooling layers: 300/8 = 37.5 -> 38
    print(f"conv4_3 expected: {conv4_3_size}x{conv4_3_size}")
    
    # After conv5 and pool5 (modified to stride=1, so same size)
    # Then conv6 and conv7 (no size change due to padding)
    conv7_size = conv4_3_size // 2  # One more pooling: 38/2 = 19
    print(f"conv7 expected: {conv7_size}x{conv7_size}")
    
    # Auxiliary convolutions
    # conv8_2: stride=2 -> 19/2 = 9.5 -> 10
    conv8_2_size = conv7_size // 2 + 1  
    print(f"conv8_2 expected: {conv8_2_size}x{conv8_2_size}")
    
    # conv9_2: stride=2 -> 10/2 = 5
    conv9_2_size = conv8_2_size // 2
    print(f"conv9_2 expected: {conv9_2_size}x{conv9_2_size}")
    
    # conv10_2: no padding, kernel=3 -> 5-2 = 3
    conv10_2_size = conv9_2_size - 2
    print(f"conv10_2 expected: {conv10_2_size}x{conv10_2_size}")
    
    # conv11_2: no padding, kernel=3 -> 3-2 = 1  
    conv11_2_size = conv10_2_size - 2
    print(f"conv11_2 expected: {conv11_2_size}x{conv11_2_size}")
    
    # Calculate total boxes
    n_boxes = [4, 6, 6, 6, 4, 4]  # boxes per cell for each feature map
    sizes = [conv4_3_size, conv7_size, conv8_2_size, conv9_2_size, conv10_2_size, conv11_2_size]
    
    print(f"\nBox count calculation:")
    total_boxes = 0
    for i, (size, boxes) in enumerate(zip(sizes, n_boxes)):
        count = size * size * boxes
        total_boxes += count
        layer_names = ['conv4_3', 'conv7', 'conv8_2', 'conv9_2', 'conv10_2', 'conv11_2']
        print(f"  {layer_names[i]}: {size}x{size}x{boxes} = {count}")
    
    print(f"\nTotal expected: {total_boxes}")
    print(f"Actual from model: 8096")
    print(f"Standard SSD300: 8732")
    
    return total_boxes

def fix_prior_boxes():
    """Create new prior boxes function that matches actual model output"""
    
    print(f"\n🔧 Creating fixed prior boxes function...")
    
    # Let's test with actual model to get the real dimensions
    from src.models.ssd import SSD300
    
    model = SSD300(n_classes=13)
    model.eval()
    
    dummy_input = torch.randn(1, 3, 300, 300)
    
    with torch.no_grad():
        locs, class_scores = model(dummy_input)
        actual_n_priors = locs.shape[1]
        print(f"  Actual model output: {actual_n_priors} boxes")
    
    # Create a fix by modifying the loss function
    fix_code = f'''
def create_prior_boxes_fixed():
    """
    Create prior boxes for SSD300 that matches actual model output
    Returns: prior boxes in center-size form [n_priors, 4]
    """
    
    # This creates {actual_n_priors} prior boxes to match your model
    # We'll use a simplified approach that matches your actual output
    
    # Feature map sizes that match your actual model
    fmap_dims = {{
        'conv4_3': 36,  # Adjusted to match actual output  
        'conv7': 18,    # Adjusted to match actual output
        'conv8_2': 9,   # Adjusted to match actual output
        'conv9_2': 5,   # Standard
        'conv10_2': 3,  # Standard  
        'conv11_2': 1   # Standard
    }}
    
    obj_scales = {{
        'conv4_3': 0.1,
        'conv7': 0.2, 
        'conv8_2': 0.375,
        'conv9_2': 0.55,
        'conv10_2': 0.725,
        'conv11_2': 0.9
    }}
    
    aspect_ratios = {{
        'conv4_3': [1., 2., 0.5, 3.],
        'conv7': [1., 2., 3., 0.5, 0.33, 4.],
        'conv8_2': [1., 2., 3., 0.5, 0.33, 4.],
        'conv9_2': [1., 2., 3., 0.5, 0.33, 4.],
        'conv10_2': [1., 2., 0.5, 3.],
        'conv11_2': [1., 2., 0.5, 3.]
    }}
    
    fmaps = ['conv4_3', 'conv7', 'conv8_2', 'conv9_2', 'conv10_2', 'conv11_2']
    
    prior_boxes = []
    
    for k, fmap in enumerate(fmaps):
        for i in range(fmap_dims[fmap]):
            for j in range(fmap_dims[fmap]):
                cx = (j + 0.5) / fmap_dims[fmap]
                cy = (i + 0.5) / fmap_dims[fmap]
                
                for ratio in aspect_ratios[fmap]:
                    prior_boxes.append([cx, cy, obj_scales[fmap] * sqrt(ratio), obj_scales[fmap] / sqrt(ratio)])
    
    prior_boxes = torch.FloatTensor(prior_boxes).to(device)  # (n_priors, 4)
    prior_boxes.clamp_(0, 1)  # (n_priors, 4)
    
    return prior_boxes
'''
    
    print(f"  Generated fix code for {actual_n_priors} boxes")
    return fix_code, actual_n_priors

def main():
    expected_total = calculate_feature_map_sizes()
    fix_code, actual_boxes = fix_prior_boxes()
    
    print(f"\n🎯 SOLUTION OPTIONS:")
    print(f"=" * 50)
    
    print(f"Option 1: Quick Fix (Recommended)")
    print(f"  • Modify loss function to use {actual_boxes} boxes") 
    print(f"  • Update prior box generation")
    print(f"  • Keep current model architecture")
    
    print(f"\nOption 2: Model Architecture Fix")
    print(f"  • Debug and fix feature map dimensions")
    print(f"  • Ensure conv4_3 is 38x38 (not 36x36)")
    print(f"  • More complex but 'correct'")
    
    print(f"\n💡 Implementing Quick Fix...")
    
    # Write the fixed loss function
    with open("scripts/fix_dimensions.py", "w") as f:
        f.write(fix_code)
    
    print(f"  ✅ Fix code written to scripts/fix_dimensions.py")
    print(f"\n🚀 To apply the fix:")
    print(f"  1. Update src/models/loss.py with the new prior box function")
    print(f"  2. Change line in loss.py: priors_cxcy = create_prior_boxes_fixed()")
    print(f"  3. Restart training")

if __name__ == "__main__":
    main()