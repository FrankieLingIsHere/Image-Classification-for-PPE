# Archived Model Components

## ⚠️ Why These Files Are Here

These model files are archived because they represent **FAILED EXPERIMENTAL APPROACHES**.
Kept for educational reference and code reuse, but **NOT RECOMMENDED FOR USE**.

---

## enhanced_ppe_detector.py

### What It Was
**Multi-task learning detector combining:**
- Primary task: Object detection (Faster R-CNN)
- Auxiliary task: Semantic segmentation (3-class)
- Additional: Spatial constraint module (filtering)

### Why It Failed

**Result**: 0.0574 mAP vs baseline 0.2659 (-78.8% worse)

**Failures Documented In**:
1. Small object detection lost completely
   - hard_hat: 0% recall
   - safety_gloves: 0% recall
   - safety_boots: 0% recall

2. Person hallucination
   - 92% false positive rate
   - 459 detections, only 35 correct

3. Confidence miscalibration
   - Average confidence 0.125 (way too low)

**Root Cause**: Competing gradients between detection and segmentation tasks on limited training data (222 images)

### Code Reuse (If Needed)

Some components might be useful:
- `EnhancedPPEDetector.__init__()` - Architecture pattern (NOT recommended)
- Data loading utilities - Could be extracted
- Augmentation strategies - Potentially useful

### Don't Use For

❌ Production deployment
❌ New training
❌ Architecture reference
❌ Loss function reference

---

## relational_rescorer.py

### What It Was
**Spatial Constraint Module**

Learned "plausibility matrix" to filter implausible detections:
- Person must be detected before PPE can be detected
- PPE items must be in spatial proximity to person
- Attempted to enforce domain knowledge via learned constraints

### Why It Failed

**Result**: Ineffective (no measurable improvement)

**Problems**:
- Spatial constraints too restrictive
- Didn't improve detection quality
- Added complexity without benefit

### Lessons Learned

❌ **Hard-coded constraints not effective**
- Domain knowledge not easily captured in learned constraints
- Better to collect more diverse data
- Better to improve core detection capability

### Don't Use For

❌ New spatial filtering modules
❌ Constraint-based detection
❌ Similar applications

---

## loss.py

### What It Was
**Custom Loss Functions**

Implemented:
- Weighted losses for class imbalance
- Focal loss variants
- Custom multi-task loss combinations

### Why Archived

**Result**: Not needed (standard losses work fine)

**Why**:
- PyTorch focal loss (`torch.nn.functional.cross_entropy` + manual focal) works better
- Standard Faster R-CNN losses sufficient
- Added unnecessary complexity

### Don't Use For

❌ New training pipelines
❌ Custom loss functions
❌ Architecture design

### What To Use Instead

Standard PyTorch + torchvision losses:
- Cross-entropy for classification
- Smooth L1 for regression
- Focal loss (from new `confidence_calibration.py`)

---

## 📊 Summary: Component Failures

| Component | Status | Reason |
|-----------|--------|--------|
| enhanced_ppe_detector.py | FAILED | Competing gradients, 0.0574 mAP |
| relational_rescorer.py | FAILED | Ineffective, no improvement |
| loss.py | INEFFECTIVE | Standard losses work better |

---

## 🔄 What To Use Instead

### For Detection
✅ **Use**: `torchvision.models.detection.fasterrcnn_resnet50_fpn`
- Standard, proven, well-maintained
- 0.2659 mAP baseline
- Simple and reliable

### For Calibration
✅ **Use**: `scripts/train/confidence_calibration.py`
- Focal loss module
- Temperature scaling
- Class weighting

### For Training
✅ **Use**: `scripts/train/train_with_confidence.py`
- Clean training script
- Confidence-focused optimization
- Expected +5-10% mAP

---

## 🎓 Educational Value

### Lessons Learned

1. **Multi-task learning complexity**
   - Competing objectives break performance
   - Shared backbone can't satisfy conflicting goals
   - Need 1000+ images for multi-task to work

2. **Constraints are hard to learn**
   - Domain knowledge difficult to encode
   - Better to improve core capability
   - More data > Hard constraints

3. **Spatial relationships**
   - Can't be reliably learned with 222 images
   - Need 500+ annotated examples
   - Simple hierarchical approach insufficient

4. **Loss design**
   - Standard losses often better than custom
   - Focal loss effective, but use proven implementations
   - Class weighting simple and effective

---

## 🔗 Related Documentation

- **Why these failed**: `ARCHIVE_README.md` (in parent directory)
- **Complete analysis**: `docs/ROOT_CAUSE_COMPLETE.md`
- **What to use**: `docs/QUICK_START_HIGH_CONFIDENCE.md`

---

## Next Steps

1. **Don't use these components** - They failed for documented reasons
2. **Use standard Faster R-CNN** - Proven, simple, effective
3. **Add confidence calibration** - Use `confidence_calibration.py`
4. **Reference for learning** - When designing new approaches, avoid similar patterns

Generated: October 26, 2025
