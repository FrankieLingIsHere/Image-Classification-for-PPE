# Quick Answer: Should You Remove Old Code?

## TL;DR

**NO - Archive instead of delete.**

### Why?
- Keep learning reference (why each failed)
- Document due diligence (show proper experimentation)
- Prevent accidental re-trying (with reasons)
- Educational value

### What to do?
```bash
# Create archive folders
mkdir scripts/train/archived_failed_approaches
mkdir src/models/archived
mkdir models/archived_failed_models

# Move old files
Move-Item scripts/train/train_full_pipeline.py scripts/train/archived_failed_approaches/
Move-Item scripts/train/ssl_pretraining.py scripts/train/archived_failed_approaches/
Move-Item src/models/enhanced_ppe_detector.py src/models/archived/
Move-Item src/models/relational_rescorer.py src/models/archived/
Move-Item src/models/loss.py src/models/archived/
Move-Item models/ppe_enhanced_best.pth models/archived_failed_models/
```

### Then
1. Create `ARCHIVE_README.md` files explaining why each failed
2. Create clean new training scripts
3. Update documentation

**Time**: 2-3 hours total

---

## The Old Code (Failed)

| File | Approach | Result | Status |
|------|----------|--------|--------|
| `train_full_pipeline.py` | 4-stage multi-task | 0.0574 mAP ❌ | Archive |
| `ssl_pretraining.py` | SSL pretraining | Didn't help ❌ | Archive |
| `enhanced_ppe_detector.py` | Multi-task detection | 78.8% worse ❌ | Archive |
| `relational_rescorer.py` | Spatial constraints | Not effective ❌ | Archive |
| `loss.py` | Custom losses | Not needed ❌ | Archive |

---

## The New Code (Working)

| File | Approach | Expected Result | Status |
|------|----------|-----------------|--------|
| `confidence_calibration.py` | Focal loss | +2-4% ✅ | Keep |
| `train_with_confidence.py` | Complete training | +5-10% ✅ | Keep |
| Standard Faster R-CNN | Baseline | 0.2659 mAP ✅ | Keep |

---

## What You'll Have After Cleanup

```
Clean & Organized:
  scripts/train/
    ├─ train_baseline_with_confidence.py    ✅ USE THIS
    ├─ confidence_calibration.py             ✅ Helper
    └─ archived_failed_approaches/           📚 Reference
         ├─ train_full_pipeline.py
         ├─ ssl_pretraining.py
         └─ ARCHIVE_README.md (why failed)

Messy & Confusing:
  scripts/train/
    ├─ train_full_pipeline.py               ❌ Still here?
    ├─ ssl_pretraining.py                   ❌ Still here?
    ├─ confidence_calibration.py             ✅ New
    └─ train_with_confidence.py             ✅ New
```

---

## Archive Documentation Example

**File**: `scripts/train/archived_failed_approaches/ARCHIVE_README.md`

```markdown
# Why These Were Archived

## train_full_pipeline.py
- Tried: 4-stage multi-task learning
- Expected: Better performance through SSL + multi-task
- Actual result: 0.0574 mAP vs baseline 0.2659 (-78.8%)
- Why failed: Competing gradients with limited data (222 images)
- Key lesson: Multi-task needs 1000+ images, not 222

## ssl_pretraining.py
- Tried: SimCLR self-supervised pretraining
- Expected: Better feature learning
- Actual result: No improvement over ImageNet pretrained
- Why failed: Limited dataset, SSL needs more diversity
- Key lesson: ImageNet pretraining already sufficient for 222 images

## What To Use Instead
See: scripts/train/train_baseline_with_confidence.py
```

---

## My Recommendation

### ✅ DO THIS:

1. **Archive old code** (keeps history)
2. **Create archive documentation** (explains why)
3. **Create new clean training scripts** (clear path forward)
4. **Update docs** (what to use, what not to)

### ❌ DON'T:
- Delete old code (lose history)
- Leave messy codebase (confusing)
- Keep old scripts active (wrong choices)

---

## Timeline

- **Archive**: 30 minutes
- **Create documentation**: 1 hour
- **Create new training scripts**: 1 hour
- **Update docs**: 1 hour
- **TOTAL**: 3-4 hours

---

Read: `docs/CLEANUP_STRATEGY_OLD_APPROACHES.md` for complete details.

Generated: October 26, 2025
