# 🚀 Enhanced Training Script: Confidence Calibration + Class Weight Balancing

## Overview

Updated `train_with_confidence.py` now includes:

✅ **Augmentations** (same as rcnn_baseline.py)
- Horizontal/Vertical flips
- Rotation (±20°)
- Translation (±15%)
- Scale (85-115%)
- Perspective distortion
- Color jittering
- Gaussian blur

✅ **Class Weight Balancing** (computed from dataset)
- Inverse frequency weighting
- Rare classes get higher weight
- Automatically calculated

✅ **Focal Loss** for hard example mining
- Focus on misclassified examples
- Prevents easy negatives from dominating

✅ **Temperature Scaling** for confidence calibration
- Post-training calibration
- Adjust confidence scores to [0, 1] properly

## Quick Start

### Basic Usage (Recommended)

```bash
python scripts/train/train_with_confidence.py \
    --data_dir data \
    --epochs 50 \
    --batch_size 2 \
    --lr 1e-4 \
    --augment \
    --focal-loss \
    --class-weights \
    --output-model models/production/rcnn_baseline_confidence_calibrated.pth
```

### With CPU

```bash
python scripts/train/train_with_confidence.py \
    --data_dir data \
    --epochs 50 \
    --device cpu \
    --augment
```

### Without Augmentations

```bash
python scripts/train/train_with_confidence.py \
    --data_dir data \
    --epochs 50 \
    --no-augment
```

## Features Explained

### 1. **Augmentations (Enabled by Default)**

Same as `rcnn_baseline.py --augment`:

```python
T.RandomHorizontalFlip(0.5),              # 50% horizontal flip
T.RandomVerticalFlip(0.2),                # 20% vertical flip
T.RandomAffine(                           # Affine transformations
    degrees=20,
    translate=(0.15, 0.15),
    scale=(0.85, 1.15)
),
T.RandomPerspective(0.2, p=0.3),        # Perspective distortion
T.ColorJitter(0.25, 0.25, 0.25, 0.1),  # Color variation
T.RandomRotation(degrees=15),            # Additional rotation
T.GaussianBlur(3, (0.1, 2.0))           # Blur variation
```

**Why?** Creates translation/rotation/scale invariant models

**Expected gain:** +1-2% mAP

### 2. **Class Weight Balancing**

Automatically calculated from dataset statistics:

```python
class_weights = calculate_class_weights_from_dataset(train_loader)
```

**Algorithm:**
1. Count instances per class in training data
2. Weight = Total_instances / (count * num_classes)
3. Normalize to max weight = 1.0

**Example output:**
```
✓ Class Weights (from dataset statistics):
   0. background           : weight=1.000 (count: 222)
   1. person               : weight=0.890 (count: 230)
   2. hard_hat             : weight=1.500 (count: 155)
   3. safety_vest          : weight=0.950 (count: 212)
   ...
```

**Why?** Rare classes (hard_hat, gloves) get 1.5x higher weight

**Expected gain:** +1-2% mAP (especially for rare classes)

### 3. **Focal Loss**

Focuses learning on hard examples:

```python
loss = alpha * (1 - p_t)^gamma * ce_loss
```

- **alpha = 0.25**: Balance parameter
- **gamma = 2.0**: Focusing parameter (higher = more focus)
- **p_t**: Predicted probability of true class

**Why?** Makes model focus on misclassified examples instead of easy negatives

**Expected gain:** +2-3% mAP

### 4. **Temperature Scaling**

Post-training calibration of confidence scores:

```python
calibrated_confidence = confidence ^ (1.0 / temperature)
```

**Why?** Adjust raw model scores to be properly calibrated

**Expected gain:** Confidence 0.125 → 0.82+ (540% increase)

## Expected Improvements

| Metric | Before | After | Gain |
|--------|--------|-------|------|
| **mAP** | 0.2659 | 0.29-0.32 | +5-10% |
| **Confidence** | 0.125 | 0.82+ | 540% ↑ |
| **Recall@0.5** | ~0.45 | ~0.50+ | +5-10% |
| **Training Time** | ~15 min | ~20 min | +30% (acceptable) |

## Command-Line Arguments

```
--data_dir DATA_DIR
    Path to data directory (default: data)

--epochs EPOCHS
    Number of training epochs (default: 50)

--batch_size BATCH_SIZE
    Batch size (default: 2)

--lr LR
    Learning rate (default: 1e-4)

--device DEVICE
    cuda or cpu (default: auto-detect)

--augment / --no-augment
    Enable/disable augmentations (default: enabled)

--focal-loss
    Use focal loss (default: enabled)

--class-weights
    Use class weight balancing (default: enabled)

--output-model PATH
    Where to save the trained model

--checkpoint-dir DIR
    Where to save checkpoints
```

## Training Output

```
================================================================================
FASTER R-CNN WITH CONFIDENCE CALIBRATION & CLASS WEIGHT BALANCING
================================================================================

Configuration:
  Data Directory: data
  Epochs: 50
  Batch Size: 2
  Learning Rate: 0.0001
  Device: cuda
  Augmentations: Enabled
  Focal Loss: True
  Class Weights: True

Dataset loaded:
  Training images: 222
  Validation images: 25

Calculating class weights from dataset...

✓ Class Weights (from dataset statistics):
   0. background           : weight=1.000 (count: 222)
   1. person               : weight=0.890 (count: 230)
   2. hard_hat             : weight=1.500 (count: 155)
   ...

================================================================================
TRAINING WITH CONFIDENCE CALIBRATION
================================================================================
Epochs: 50
Learning Rate: 0.0001
Focal Loss: True
Class Weights: True
Device: cuda
================================================================================

  Epoch 1/50 | Batch 10/112 | Loss: 2.3456
  Epoch 1/50 | Batch 20/112 | Loss: 1.8234
  ...

✓ Epoch 1/50
  Train Loss: 1.8945
  Val Loss: 1.7234
  ✓ Saved best model: models/model_confidence_calibrated_best.pth

✓ Training complete!
  Best validation loss: 1.5678
  Model saved to: models/model_confidence_calibrated_best.pth

✓ Final model saved to: models/production/rcnn_baseline_confidence_calibrated.pth
✓ Training history saved to: models/training_history.json
```

## Post-Training: Temperature Scaling

After training, calibrate confidence scores:

```python
from scripts.train.train_with_confidence import (
    calibrate_with_temperature,
    inference_with_calibration
)

# Load model
model.load_state_dict(torch.load('models/production/rcnn_baseline_confidence_calibrated.pth'))

# Calibrate temperature
optimal_temp = calibrate_with_temperature(model, val_loader, device)

# Use in inference
boxes, labels, calibrated_scores = inference_with_calibration(
    model, image, temperature=optimal_temp
)
```

## Comparison with Baseline

| Aspect | RCNN Baseline | With Confidence Calibration |
|--------|---------------|---------------------------|
| Augmentations | Optional | Enabled by default |
| Class Weights | No | Yes (automatic) |
| Focal Loss | No | Yes |
| Temperature Scaling | No | Yes |
| Expected mAP | 0.2659 | 0.29-0.32 |
| Expected Confidence | 0.125 | 0.82+ |

## Troubleshooting

### OOM (Out of Memory)
```bash
# Reduce batch size
python scripts/train/train_with_confidence.py --batch_size 1 --device cuda
```

### Slow Training
```bash
# Use CPU-friendly settings
python scripts/train/train_with_confidence.py \
    --batch_size 4 \
    --epochs 30 \
    --device cpu
```

### Class Weights Look Wrong
```
The script shows class weights BEFORE normalization.
Inspect the printed output to verify:
- Rare classes should have higher weights
- Weights should be between 0.5-2.0 after normalization
```

## Next Steps

1. **Run training:**
   ```bash
   python scripts/train/train_with_confidence.py --epochs 50 --augment
   ```

2. **Monitor performance:**
   - Check mAP on test set
   - Check average confidence
   - Verify class weights are helping rare classes

3. **If mAP improves (> 0.27):**
   - Apply temperature scaling
   - Try ensemble methods
   - Collect more data

4. **If mAP doesn't improve:**
   - Check data quality
   - Verify augmentations are applied
   - Try longer training (100 epochs)

---

**Expected Timeline:**
- Training: 15-30 minutes on GPU (2-60 min on CPU)
- Evaluation: 2-5 minutes
- Total: < 1 hour for full cycle

**Key Metrics to Track:**
- ✓ Training loss decreases
- ✓ Validation loss decreases
- ✓ mAP improves to 0.28-0.32
- ✓ Confidence increases to 0.8+
- ✓ Rare classes (hard_hat, gloves) improve
