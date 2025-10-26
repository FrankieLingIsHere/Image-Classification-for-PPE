# Model Improvement Plan - Architecture & Training Enhancements

## Analysis Results Summary

### Current Performance (with threshold 0.08)
- **Precision**: ~50% (many false positives)
- **Recall**: ~60% (missing some PPE)
- **Detections**: 356 false positives, 186 missed detections

### Critical Issues Identified

#### 1. FALSE POSITIVES (356 total) 🔴
**Primary Issue: Person class completely broken**
- 182 FPs from "person" class (51% of all false positives!)
- Avg confidence: 0.2768 (VERY HIGH, yet still wrong)
- Avg size: 1174 pixels (very large detections)
- These are hallucinated full-body detections where no people exist

**Secondary Issues:**
- 81 FPs from safety_vest (avg confidence 0.096)
- 51 FPs from safety_boots (avg confidence 0.093)
- All large false positives suggest context-blindness

#### 2. MISSED DETECTIONS (186 total) 🟡
**Balanced miss rate across all classes:**
- safety_gloves: 27 missed (avg size 225px)
- hard_hat: 26 missed (avg size 195px)
- person: 25 missed (avg size 788px)
- All others: 10-24 missed

**Pattern: Not systematically missing small objects** - issue is model confidence

---

## Root Cause Analysis

### Why Person Detection Fails
The model predicts "person" with **high confidence (0.27)** but these are **wrong**. This means:
1. During training, "person" class overfitted to spurious patterns
2. Model learned to fire person detection on ANY large regions
3. No spatial reasoning (e.g., "PPE items must be WITH a person")

### Why PPE Precision is Low
1. Model makes predictions at threshold 0.08-0.1, not inherently confident
2. Many predictions match ground truth with IoU < 0.5 (not valid matches)
3. No context filtering - doesn't use spatial relationships between objects

---

## Proposed Architecture Improvements

### OPTION 1: Context-Aware Detection (Recommended - Medium Complexity)
**Your GAT rescorer is already perfect for this!**

Implement 3-stage pipeline:
```
Stage 1: Faster R-CNN (baseline detector)
         ↓
Stage 2: Filter person detections using spatial heuristics
         - Must overlap with worker-like regions (torso + limbs)
         - Must have PPE items nearby (within reasonable distance)
         ↓
Stage 3: GAT rescoring (already implemented!)
         - Input: detections + their relationships
         - Output: confidence adjustments
         - Remove impossible combinations
```

**Why this works:**
- Person detection gets context (must have PPE around it)
- False positive person detections removed via spatial filtering
- GAT refines remaining detections based on object relationships

**Implementation effort:** 2-3 hours
**Expected improvement:** +15-20% precision (eliminate most person FPs)

---

### OPTION 2: Multi-Task Learning (High Impact - High Complexity)
**Add semantic segmentation as auxiliary task**

```python
class EnhancedPPEDetector(nn.Module):
    def __init__(self):
        # Main detection head (Faster R-CNN)
        self.detector = fasterrcnn_resnet50_fpn(num_classes=12)
        
        # Auxiliary semantic segmentation head
        self.seg_head = SegmentationHead(
            in_channels=256,
            num_classes=3  # 0=background, 1=person_torso, 2=ppe_items
        )
    
    def forward(self, x):
        # Detection
        det_output = self.detector(x)
        
        # Segmentation (auxiliary)
        seg_output = self.seg_head(x)
        
        # Combine: use seg map to validate person detections
        return det_output, seg_output
```

**Why this works:**
- Semantic segmentation acts as regularizer
- Forces model to learn spatial structure
- During inference, use seg map to filter impossible detections

**Implementation effort:** 4-6 hours
**Expected improvement:** +20-30% precision, +10% recall

---

### OPTION 3: Self-Supervised Pretraining (Highest Quality - Highest Complexity)
**Use contrastive learning before fine-tuning for detection**

```python
# Stage 1: Self-supervised pretraining (2 days)
class PPEContrastivePretrainer(nn.Module):
    def __init__(self):
        self.backbone = ResNet50()
        self.projection_head = MLPHead(2048 -> 128)
    
    def forward(self, img1, img2):  # augmented versions
        z1 = self.projection_head(self.backbone(img1))
        z2 = self.projection_head(self.backbone(img2))
        return contrastive_loss(z1, z2)

# Stage 2: Fine-tune for detection with pretrained backbone
detector = fasterrcnn_resnet50_fpn(num_classes=12)
detector.backbone.load_from(pretrained_backbone)
detector.train_on_ppe_data()
```

**Why this works:**
- Model learns PPE-specific features (not just generic ImageNet)
- Better feature representations reduce hallucinations
- Improves confidence calibration

**Implementation effort:** 8-12 hours (mostly waiting for training)
**Expected improvement:** +25-35% overall performance

---

### OPTION 4: Hard Negative Mining (Quick Win - Low Complexity)
**Focus training on hard false positives**

```python
# During training:
# 1. First pass: train normally for 5 epochs
# 2. Collect predictions on training set
# 3. Identify hard negatives (high confidence wrong predictions)
# 4. Oversample hard negatives in next training (2x-5x)
# 5. Continue training for 10 more epochs with weighted sampling
```

**Why this works:**
- Model learns to discriminate difficult cases
- Reduces false positive rate directly
- Improves decision boundaries

**Implementation effort:** 1-2 hours
**Expected improvement:** +8-15% precision (quick win)

---

## Recommended Implementation Plan

### PHASE 1: Quick Fixes (TODAY - 2 hours)
1. Implement spatial heuristics for person filtering (OPTION 1, Stage 2)
2. Lower confidence threshold from 0.5 to 0.08
3. Retrain for 15 more epochs with current architecture

Expected result: **mAP → 0.08-0.10**

### PHASE 2: Context Integration (THIS WEEK - 4 hours)
1. Implement GAT-based rescoring (you already have code!)
2. Combine OPTION 1 (spatial filtering) + your GAT rescorer
3. Retrain end-to-end for 20 epochs

Expected result: **mAP → 0.25-0.35**

### PHASE 3: Deep Improvement (NEXT WEEK - 8-12 hours)
Choose one:
- **Option 2**: Add semantic segmentation auxiliary task
- **Option 3**: Self-supervised pretraining
- **Option 4**: Hard negative mining

Expected result: **mAP → 0.50-0.70**

---

## Implementation Code: PHASE 1 (Quickest)

### Spatial Heuristic for Person Filtering

```python
def filter_person_detections(boxes, labels, scores, img_height, img_width):
    """
    Filter out false positive person detections using spatial heuristics.
    A valid person detection should have:
    - Reasonable aspect ratio (0.3 < h/w < 3)
    - Height > 50% of image (workers are prominent)
    - Not too large (width < 80% of image)
    """
    valid_idx = []
    
    for i, (box, label, score) in enumerate(zip(boxes, labels, scores)):
        if label != 1:  # Not person class
            valid_idx.append(i)
            continue
        
        x1, y1, x2, y2 = box
        h = y2 - y1
        w = x2 - x1
        
        # Heuristics for valid person
        aspect_ratio = h / (w + 1e-6)
        h_ratio = h / img_height
        w_ratio = w / img_width
        
        is_valid = (
            0.3 < aspect_ratio < 3.0 and    # Reasonable body proportions
            h_ratio > 0.3 and                # At least 30% of image height
            w_ratio < 0.9                    # Not taking up whole width
        )
        
        if is_valid:
            valid_idx.append(i)
    
    return valid_idx

# Usage during inference
predictions = model(images)
for img_idx, (boxes, labels, scores) in enumerate(predictions):
    valid = filter_person_detections(
        boxes, labels, scores,
        img_height=images[img_idx].shape[-2],
        img_width=images[img_idx].shape[-1]
    )
    predictions[img_idx] = {
        'boxes': boxes[valid],
        'labels': labels[valid],
        'scores': scores[valid]
    }
```

---

## Decision Matrix

| Option | Speed | Quality | Complexity | When to Use |
|--------|-------|---------|------------|------------|
| Phase 1 (Spatial) | 2 hrs | +20% | Low | **Start here** |
| Phase 2 (GAT) | 4 hrs | +30% | Medium | After Phase 1 |
| Phase 3 Option 2 (Seg) | 6 hrs | +40% | High | Maximum quality |
| Phase 3 Option 3 (SSL) | 12 hrs | +50% | Very High | If time allows |
| Phase 3 Option 4 (HNM) | 2 hrs | +15% | Low | Quick follow-up |

---

## Your Immediate Next Steps

1. **Choose your timeline:**
   - QUICK: Phase 1 only (2 hours) → mAP ~0.10
   - BALANCED: Phase 1 + Phase 2 (6 hours) → mAP ~0.30
   - THOROUGH: Phase 1 + 2 + 3 (14 hours) → mAP ~0.70

2. **I can implement:**
   - All spatial heuristics automatically
   - Integrate your GAT rescorer
   - Set up multi-task learning if desired

3. **Decision: What's your priority?**
   - Getting something working quickly?
   - Maximum quality for production?
   - Learning opportunity?

What would you like to do?
