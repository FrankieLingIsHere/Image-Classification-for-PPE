import torch

# Load checkpoint
checkpoint = torch.load('models/checkpoint_epoch_4.pth', map_location='cpu')

# Find prediction layer keys
pred_keys = [k for k in checkpoint['model_state_dict'].keys() if 'pred_convs' in k and 'cl_' in k]
print("Classification layer keys:")
for key in pred_keys[:3]:
    shape = checkpoint['model_state_dict'][key].shape
    print(f"{key}: {shape}")
    if len(shape) == 1:  # bias
        num_classes = shape[0] // 4  # Each class has 4 values per anchor
        print(f"  -> Num classes inferred: {num_classes}")
        break
    elif len(shape) == 4 and 'weight' in key:  # conv weight
        num_classes = shape[0] // 4  # Output channels divided by 4 anchors
        print(f"  -> Num classes inferred: {num_classes}")
        break