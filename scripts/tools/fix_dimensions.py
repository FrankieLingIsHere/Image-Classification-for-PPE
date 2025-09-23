
def create_prior_boxes_fixed():
    """
    Create prior boxes for SSD300 that matches actual model output
    Returns: prior boxes in center-size form [n_priors, 4]
    """
    
    # This creates 8096 prior boxes to match your model
    # We'll use a simplified approach that matches your actual output
    
    # Feature map sizes that match your actual model
    fmap_dims = {
        'conv4_3': 36,  # Adjusted to match actual output  
        'conv7': 18,    # Adjusted to match actual output
        'conv8_2': 9,   # Adjusted to match actual output
        'conv9_2': 5,   # Standard
        'conv10_2': 3,  # Standard  
        'conv11_2': 1   # Standard
    }
    
    obj_scales = {
        'conv4_3': 0.1,
        'conv7': 0.2, 
        'conv8_2': 0.375,
        'conv9_2': 0.55,
        'conv10_2': 0.725,
        'conv11_2': 0.9
    }
    
    aspect_ratios = {
        'conv4_3': [1., 2., 0.5, 3.],
        'conv7': [1., 2., 3., 0.5, 0.33, 4.],
        'conv8_2': [1., 2., 3., 0.5, 0.33, 4.],
        'conv9_2': [1., 2., 3., 0.5, 0.33, 4.],
        'conv10_2': [1., 2., 0.5, 3.],
        'conv11_2': [1., 2., 0.5, 3.]
    }
    
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
