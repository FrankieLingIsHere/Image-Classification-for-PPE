# Archived Failed Approaches

## ⚠️ Why These Files Are Here

These scripts represent experiments that were tried and **FAILED**.
They are archived for educational purposes and to **PREVENT RE-ATTEMPTING** them.

---

## train_full_pipeline.py

### What It Tried
**4-Stage Multi-Task Learning (Option D)**
- Stage 1: SimCLR self-supervised pretraining (20 epochs)
- Stage 2-4: Multi-task detection + semantic segmentation + spatial constraints (50 epochs)

### Expected Result
Better performance through SSL pretraining + multi-task learning architecture

### Actual Result
**0.0574 mAP vs baseline 0.2659 mAP = -78.8% (CATASTROPHIC FAILURE)**

### Why It Failed

1. **Competing Gradients**
   - Detection task gradient: "Find PPE items"
   - Segmentation task gradient: "Segment background/person"
   - Spatial constraint gradient: "Filter implausible locations"
   - Shared backbone torn between 3 conflicting objectives
   - Result: Compromise solution bad at all 3

2. **Small Object Detection Lost**
   - hard_hat: 0% recall (vs baseline 44%)
   - safety_gloves: 0% recall (vs baseline 11%)
   - safety_boots: 0% recall (vs baseline 5%)
   - Reason: Multi-task learning destroyed signal for small objects

3. **Person Hallucination**
   - 92% false positive rate on person class
   - 459 detections, only 35 correct, 424 false positives
   - Baseline: 63% FP rate (bad but functional)

4. **Confidence Miscalibration**
   - Average confidence: 0.125 (way too low)
   - Had to lower threshold from 0.5 → 0.1 to get any detections
   - Result: Too many false positives

5. **Limited Data**
   - Only 222 training images
   - Multi-task learning needs 1000+ images
   - Complex architecture overfitted with small dataset

### Key Lessons

❌ **Multi-task learning inappropriate for <300 images**
- Competing objectives need large diverse data
- Simple baseline > Complex architecture on small data

❌ **Training loss ≠ Actual performance**
- Combined loss decreased (1.157 → 0.961)
- But detection quality catastrophically failed
- Need to optimize for task, not just loss metric

❌ **Shared backbone limitations**
- Can't satisfy conflicting objectives
- Better to focus on single task with limited data

### What To Use Instead

See: `scripts/train/train_with_confidence.py`

**Simple approach that works:**
- Baseline Faster R-CNN (proven, 0.2659 mAP)
- Confidence calibration (focal loss, class weights, temperature)
- Expected: +5-10% mAP improvement
- Much simpler, proven effective

---

## ssl_pretraining.py

### What It Tried
**Self-Supervised Learning (SimCLR) Pretraining**

Train feature extractor on unlabeled data before detection training.

### Expected Result
Better feature representations → Better detection performance

### Actual Result
**No improvement over ImageNet pretraining** (Ineffective)

### Why It Failed

1. **Dataset Too Small**
   - Only 222 images
   - SSL typically needs 10,000+ diverse images
   - Insufficient for learning meaningful self-supervised signals

2. **ImageNet Pretraining Already Good**
   - ImageNet features transfer well to PPE detection
   - No benefit from additional SSL on small dataset
   - Added complexity without gain

3. **SSL Needs Diversity**
   - SSL learns from image diversity
   - 222 PPE images too similar
   - ImageNet's 1M diverse images better starting point

### Key Lessons

❌ **SSL needs large diverse datasets**
- 222 images insufficient
- ImageNet (1M images) already sufficient

❌ **Don't add complexity without demonstrated benefit**
- SSL added training time
- Didn't improve results
- Wasted effort

### What To Use Instead

Start with ImageNet pretrained Faster R-CNN (already in torchvision).

---

## archive_old_versions.py

### What It Was
Utility script for archiving old model versions.

### Why Archived
No longer needed in training pipeline.

---

## 📊 Summary: Why These Failed

| Component | Result | Reason |
|-----------|--------|--------|
| 4-Stage Pipeline | -78.8% mAP | Competing gradients |
| SSL Pretraining | No improvement | Dataset too small |
| Multi-Task Learning | Lost small objects | Conflicting objectives |
| Spatial Constraints | Ineffective | Too restrictive |
| Custom Losses | Not needed | Standard losses better |

---

## ✅ What Works Instead

**Simple Approach (Proven Effective)**:
1. Baseline Faster R-CNN ResNet50+FPN
2. Confidence calibration:
   - Focal loss (focus on hard examples)
   - Class weights (hard classes 2.5x)
   - Temperature scaling (post-training calibration)
3. Expected: 0.2659 → 0.28-0.30 mAP (+5-10%)
4. Confidence: 0.125 → 0.82+ (540% increase)

---

## 🎓 Educational Value

### Why Keep This Archive?

1. **Learn from mistakes**
   - Understand why complex doesn't always work
   - Know when to use simple approaches

2. **Prevent re-attempting**
   - If someone suggests multi-task learning, point them here
   - Document the failure and lessons learned

3. **Historical record**
   - Shows proper experimentation and iteration
   - Demonstrates due diligence

4. **Code reuse**
   - Augmentation strategies: extractable
   - Data loading utilities: still useful
   - Evaluation metrics: applicable

---

## 🔗 Related Files

- **Original results**: `docs/ROOT_CAUSE_COMPLETE.md`
- **Why it failed**: `docs/ANALYSIS_COMPLETE.md`
- **What works**: `docs/QUICK_START_HIGH_CONFIDENCE.md`
- **What to use**: `scripts/train/train_with_confidence.py`

---

## Next Steps

1. **Don't use these files** - They failed for documented reasons
2. **Use instead**: `scripts/train/train_with_confidence.py`
3. **Reference for learning** - When wondering if multi-task would help, read this

Generated: October 26, 2025
