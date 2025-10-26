# PPE Detection: Improvement Roadmap from 0.27 → 0.75 mAP

## Current State
- **Baseline Model**: Faster R-CNN ResNet50+FPN
- **Current mAP**: 0.2659 (acceptable but far from 0.75)
- **Training Data**: 222 images (TOO SMALL)
- **Problem Areas**: 
  - Small objects (hard_hat 44%, gloves 11%, boots 5%)
  - Poor person detection (63% FP rate)
  - Confidence miscalibration

---

## The Gap Analysis: 0.27 → 0.75

| Level | mAP | Status | Gap |
|-------|-----|--------|-----|
| **Current Baseline** | 0.27 | ✓ Working | +0.48 needed |
| **Data-driven improvement** | 0.35-0.40 | Possible | +0.35-0.40 |
| **Architecture tweaks** | 0.45-0.55 | Possible | +0.20-0.30 |
| **Expert calibration** | 0.60-0.70 | Possible | +0.05-0.15 |
| **Target** | 0.75 | **Goal** | 0.00 |

**Key insight:** Getting to 0.75 requires ALL of these, not just one.

---

## 🎯 Five-Part Improvement Strategy

### PART 1: Data Collection (Most Important - 40% of the gains)

**Current Problem**: 222 images is TOO SMALL for robust detection
- Need minimum 500 images for good generalization
- Need 1000+ images for state-of-the-art (0.75+)

**Action Plan:**

1. **Collect 300+ new images** following this distribution:
   ```
   Current: 222 images
   Goal: 600+ total images
   
   Categories to prioritize:
   - Small objects (hard hat, gloves, boots) - 30% of new images
   - Different angles/perspectives - 25% of new images
   - Poor lighting conditions - 20% of new images
   - Occlusions and edge cases - 15% of new images
   - Indoor/outdoor variety - 10% of new images
   ```

2. **Data quality checklist:**
   - High resolution (800x600 minimum)
   - Good annotation accuracy (multiple annotators)
   - Diverse workers (different body types, heights)
   - Varied PPE combinations
   - Different lighting/weather conditions
   - Indoor and outdoor settings

3. **Expected impact**: 0.27 → 0.40 mAP (+48%)

---

### PART 2: Address Small Object Detection (15% of gains)

**Current Problem**: Hard hat 44%, gloves 11%, boots 5% - TOO LOW

**Action Plan:**

1. **Increase image resolution**:
   ```python
   # Current: 640x640
   # Change to: 800x800 or 1024x1024
   
   # In your training code:
   RESIZE_SIZE = 1024  # was 640
   ```
   - Benefit: Small objects get more pixels
   - Trade-off: Slower training, more memory
   - Expected gain: +5-8% mAP

2. **Adjust anchor scales**:
   ```python
   # Default Faster R-CNN anchors focus on medium objects
   # Add smaller anchors for small objects
   
   anchor_sizes = ((32, 64, 128, 256, 512),)  # default
   # Change to:
   anchor_sizes = ((16, 32, 64, 128, 256, 512),)  # add 16 for small objects
   ```
   - Expected gain: +3-5% mAP

3. **Use feature pyramid for small objects**:
   - Current: FPN (already using)
   - Enhancement: Focus on P3 and P4 features
   - Expected gain: +2-4% mAP

---

### PART 3: Fix Confidence Calibration (10% of gains)

**Current Problem**: Average confidence 0.125 (too low, threshold needs to be 0.1)

**Action Plan:**

1. **Implement focal loss** (better than standard cross-entropy):
   ```python
   # In your training:
   # Current: Standard cross-entropy
   # Change to: Focal loss
   
   from torchvision.ops import sigmoid_focal_loss
   
   # This emphasizes hard examples and fixes calibration
   ```
   - Expected gain: +3-5% mAP

2. **Add temperature scaling** for calibration:
   ```python
   # After training, calibrate confidence scores
   temperature = 1.5  # tune on validation set
   calibrated_scores = softmax(logits / temperature)
   ```
   - Expected gain: +1-2% mAP

3. **Use class-weighted loss**:
   ```python
   # Harder to detect classes should have higher loss weight
   class_weights = {
       'person': 1.0,
       'hard_hat': 2.0,  # harder, increase weight
       'safety_gloves': 2.5,  # harder, increase weight
       'safety_boots': 2.5,  # harder, increase weight
   }
   ```
   - Expected gain: +2-4% mAP

---

### PART 4: Reduce Person False Positives (15% of gains)

**Current Problem**: 92% FP rate on person class (was 63% in baseline)

**Action Plan:**

1. **Collect negative examples** (background patches without people):
   - Add 200-300 images of just background
   - Expected gain: +5-8% mAP

2. **Hard negative mining**:
   ```python
   # After each epoch:
   # 1. Find hardest false positives
   # 2. Re-train on these hard examples
   # 3. Focus on worst mistakes
   ```
   - Expected gain: +3-5% mAP

3. **Two-stage person detection**:
   ```python
   # Stage 1: Detect all potential people (permissive)
   # Stage 2: Classify as real person vs background (strict)
   
   # Add a secondary classifier for person verification
   ```
   - Expected gain: +2-4% mAP

---

### PART 5: Model Architecture Enhancements (10% of gains)

**Current Model**: Faster R-CNN ResNet50+FPN (good baseline)

**Upgrade Options** (ranked by effort/gain):

1. **ResNet101 backbone** (easiest upgrade):
   ```python
   # Current: ResNet50
   # Change to: ResNet101
   
   backbone = resnet_fpn_backbone('resnet101', pretrained=True)
   ```
   - Training time: +30% slower
   - Expected gain: +1-3% mAP
   - Effort: 5 minutes

2. **Better pretrained weights** (moderate upgrade):
   ```python
   # Current: ImageNet pretrained
   # Change to: MAE or DINO pretrained weights
   
   # Use weights from better pretraining methods
   ```
   - Expected gain: +2-5% mAP
   - Effort: 30 minutes

3. **Use RetinaNet instead** (high effort):
   ```python
   # Current: Faster R-CNN (two-stage)
   # Consider: RetinaNet (one-stage)
   
   # RetinaNet better at handling class imbalance with focal loss
   ```
   - Expected gain: +1-3% mAP
   - Effort: 2-3 hours
   - Note: May not be worth it

---

## 📊 Realistic Improvement Trajectory

### Conservative Path (Easier to implement)
```
Current:           0.27 mAP
+ Data (300 imgs): 0.40 mAP (+13%)
+ Small objects:   0.48 mAP (+8%)
+ Calibration:     0.55 mAP (+7%)
+ FP reduction:    0.63 mAP (+8%)
Result:            0.63 mAP total (target was 0.75, fell short)
```

### Aggressive Path (More comprehensive)
```
Current:           0.27 mAP
+ Data (500 imgs): 0.42 mAP (+15%)
+ Small objects:   0.52 mAP (+10%)
+ Calibration:     0.60 mAP (+8%)
+ FP reduction:    0.68 mAP (+8%)
+ ResNet101:       0.72 mAP (+4%)
Result:            0.72 mAP (close to 0.75 target)
```

### Optimal Path (If all factors align well)
```
Current:           0.27 mAP
+ Data (800 imgs): 0.45 mAP (+18%)
+ Small objects:   0.56 mAP (+11%)
+ Calibration:     0.65 mAP (+9%)
+ FP reduction:    0.72 mAP (+7%)
+ ResNet101:       0.76 mAP (+4%)
Result:            0.76 mAP ✓ TARGET ACHIEVED
```

---

## ✅ Implementation Priority (Start Here)

### Phase 1: Data Collection (Weeks 1-2) - DO THIS FIRST
**Impact**: +13-18% mAP
**Effort**: High (manual collection)
**Steps**:
1. Collect 300-500 new images
2. Annotate with Label Studio
3. Add to training set
4. Retrain baseline model

**Expected result**: 0.27 → 0.40-0.45 mAP

### Phase 2: Small Object Fixes (Days 3-4)
**Impact**: +8-11% mAP
**Effort**: Low-Medium
**Steps**:
1. Increase image size to 1024x1024
2. Add smaller anchors (16px)
3. Retrain

**Expected result**: 0.40 → 0.48-0.56 mAP

### Phase 3: Calibration (Days 5-6)
**Impact**: +7-9% mAP
**Effort**: Medium
**Steps**:
1. Implement focal loss
2. Add class weights
3. Temperature scaling on validation set
4. Retrain

**Expected result**: 0.48 → 0.56-0.65 mAP

### Phase 4: FP Reduction (Days 7-8)
**Impact**: +7-8% mAP
**Effort**: Medium-High
**Steps**:
1. Add negative examples (background)
2. Implement hard negative mining
3. Optional: Two-stage person classifier
4. Retrain

**Expected result**: 0.56 → 0.64-0.72 mAP

### Phase 5: Architecture (Optional)
**Impact**: +4% mAP
**Effort**: Low
**Steps**:
1. Switch to ResNet101 backbone
2. Retrain

**Expected result**: 0.72 → 0.76 mAP

---

## 💰 Realistic Timeline

| Phase | Duration | mAP Start | mAP End | Cumulative Gain |
|-------|----------|-----------|---------|-----------------|
| Baseline | - | 0.27 | 0.27 | - |
| Phase 1: Data | 2 weeks | 0.27 | 0.42 | +55% |
| Phase 2: Small Objects | 2 days | 0.42 | 0.52 | +85% |
| Phase 3: Calibration | 2 days | 0.52 | 0.61 | +126% |
| Phase 4: FP Reduction | 2 days | 0.61 | 0.71 | +163% |
| Phase 5: Architecture | 1 day | 0.71 | 0.75+ | +177% |
| **Total** | **3.5 weeks** | **0.27** | **0.75+** | **+177%** |

**Note**: Phase 1 (data collection) takes the longest because it's manual. Everything else is relatively quick.

---

## ⚠️ Critical Requirements for 0.75 mAP

1. **Data**: You MUST collect 500+ images
   - Without this, you won't reach 0.75
   - This is non-negotiable

2. **Small objects**: Focus on hard_hat, gloves, boots
   - These are currently failing
   - Big opportunity for gains

3. **Calibration**: Don't skip focal loss + class weights
   - Fixes the confidence problem
   - Quick to implement

4. **Person class**: Reduce false positives
   - Currently 92% FP rate
   - Add negative examples and hard mining

---

## 📝 Which Path Should You Choose?

### Choose "Conservative Path" if:
- You have limited time (2-3 weeks)
- You want to collect 300 images only
- Expected result: 0.63 mAP (83% of target)

### Choose "Aggressive Path" if:
- You have 3-4 weeks available
- You can collect 500 images
- Expected result: 0.72 mAP (96% of target)

### Choose "Optimal Path" if:
- You have 3.5+ weeks
- You can collect 800 images
- You want to implement all 5 phases
- Expected result: 0.76 mAP (101% of target) ✓

---

## 🎯 Bottom Line

**To reach 0.75 mAP, you need:**

1. **MORE DATA** (500+ images minimum)
   - This is the biggest lever (+55-60%)
   - Everything else builds on this

2. **Small object fixes** (resize, anchors)
   - Hard hat, gloves, boots detection

3. **Confidence calibration** (focal loss, class weights)
   - Fixes the "too confident on wrong stuff" problem

4. **FP reduction** (negative examples, hard mining)
   - Person class hallucination fix

5. **Optional: Better backbone** (ResNet101)
   - 4% extra if you want to push to 0.76+

**Most realistic outcome**: 0.72-0.75 mAP in 3-4 weeks
**With current data (222 images)**: Maximum ~0.45-0.50 mAP (not realistic to reach 0.75)

---

## Next Steps

1. **Today**: Read this document
2. **Tomorrow**: Start collecting new images
3. **Week 1-2**: Collect 300-500 images, annotate
4. **Week 3**: Implement phases 2-5 from checklist above
5. **Week 4**: Evaluate, iterate, reach target

Generated: October 26, 2025
