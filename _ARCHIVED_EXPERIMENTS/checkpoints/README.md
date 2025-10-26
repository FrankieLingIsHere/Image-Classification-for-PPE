# Archived Failed Checkpoints

## ⚠️ Important

These checkpoints represent **FAILED EXPERIMENTS**.
They are kept for reference, but should **NOT BE USED FOR PRODUCTION**.

---

## ppe_enhanced_best.pth

### Model Type
**Enhanced Multi-Task PPE Detector**
- Architecture: Enhanced Faster R-CNN with multi-task learning
- Training: 4-stage pipeline (SSL pretraining + multi-task detection)
- File size: 168 MB

### Performance

| Metric | Value | Comparison |
|--------|-------|-----------|
| **mAP** | 0.0574 | Baseline: 0.2659 (-78.8%) |
| Avg Confidence | 0.125 | Should be: 0.8+ |
| hard_hat Recall | 0% | Baseline: 44% |
| safety_gloves Recall | 0% | Baseline: 11% |
| safety_boots Recall | 0% | Baseline: 5% |
| person FP Rate | 92% | Baseline: 63% |

### Why It Failed

**Root Cause**: Multi-task learning with competing gradients on limited data

1. **Detection & Segmentation Conflict**
   - Detection gradient: "Find all objects"
   - Segmentation gradient: "Segment background vs person"
   - Spatial constraint gradient: "Filter implausible"
   - Shared backbone can't satisfy all 3

2. **Small Object Detection Destroyed**
   - Model focused on person class
   - Lost ability to detect small PPE items
   - hard_hat, gloves, boots: 0% recall

3. **Person Class Hallucination**
   - 459 detections on person class
   - Only 35 correct, 424 false positives
   - 92% false positive rate

4. **Confidence Miscalibration**
   - Average confidence: 0.125
   - Had to use threshold 0.1 (vs normal 0.5)
   - Result: Too many false positives

### Training History

```
Epoch 1:   Loss: 1.157
Epoch 50:  Loss: 0.961  ← Loss decreased!
           mAP:  0.0574 ← But quality catastrophically failed

Key Insight: Training loss ≠ Actual performance
             Competing objectives: lower combined loss
             But destroy individual task quality
```

### Documentation

For complete failure analysis, see:
- `docs/ANALYSIS_COMPLETE.md`
- `docs/ROOT_CAUSE_COMPLETE.md`
- `docs/DEBUG_ROOT_CAUSE.md`

---

## What To Use Instead

### ✅ Baseline Faster R-CNN
- **Performance**: 0.2659 mAP
- **Status**: Working, proven baseline
- **Where to find**: Standard torchvision model
- **How to train**: `scripts/train/train_with_confidence.py`

### ✅ Baseline with Confidence Calibration
- **Expected**: 0.28-0.30 mAP (+5-10%)
- **Confidence**: 0.125 → 0.82+
- **Much simpler**: No multi-task complexity
- **How to train**: `scripts/train/train_with_confidence.py`

---

## 🎓 Key Lessons

1. **Simple > Complex on small datasets**
   - 222 images insufficient for multi-task learning
   - Baseline Faster R-CNN is better choice

2. **Competing objectives problematic**
   - Each task needs its own focused optimization
   - Shared backbone architecture problematic
   - Better to focus on single objective

3. **Training loss misleading**
   - Combined loss decreased (good)
   - But actual performance failed (bad)
   - Always validate on actual metrics

4. **Data scale matters**
   - Multi-task learning needs 1000+ images minimum
   - With 222 images, keep it simple
   - Collect more data before adding complexity

---

## 🔗 Related Files

- **Training script that failed**: `scripts/train/archived_failed_approaches/train_full_pipeline.py`
- **Model code that failed**: `src/models/archived/enhanced_ppe_detector.py`
- **Why it failed (detailed)**: `docs/ROOT_CAUSE_COMPLETE.md`
- **What works**: `scripts/train/train_with_confidence.py`

---

## Next Steps

1. **Don't use this checkpoint** - It's worse than baseline
2. **Use baseline instead** - `best_model_regularized.pth` or similar
3. **Add confidence calibration** - `train_with_confidence.py`
4. **Expect**: 0.28-0.30 mAP (+5-10% improvement)

Generated: October 26, 2025
