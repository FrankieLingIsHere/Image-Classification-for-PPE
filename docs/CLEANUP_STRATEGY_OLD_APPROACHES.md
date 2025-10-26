# Codebase Cleanup Strategy: Old vs New Approaches

## What Old Approaches Exist?

### ❌ The 4-Stage Option D Pipeline (FAILED)
**Files**:
- `scripts/train/train_full_pipeline.py` - Orchestrator for 4-stage training
- `scripts/train/ssl_pretraining.py` - SimCLR self-supervised pretraining
- `src/models/enhanced_ppe_detector.py` - Multi-task detector with segmentation + spatial constraints
- `src/models/relational_rescorer.py` - Spatial constraint module
- `src/models/loss.py` - Custom losses for multi-task learning
- `scripts/eval/evaluate_detection_performance.py` - Has enhanced model support

**Why Failed**:
1. Confidence miscalibration (avg 0.125, too low)
2. Small object detection lost (hard_hat 0%, gloves 0%)
3. Person hallucination (92% false positive rate)
4. Root cause: Multi-task learning competing gradients on limited data (222 images)
5. **Result: Enhanced 0.0574 mAP vs Baseline 0.2659 mAP (78.8% WORSE)**

**Checkpoint**: `models/ppe_enhanced_best.pth` (168MB, not recommended for production)

---

### ✅ The Baseline (WORKING)
**Files**:
- `torchvision.models.detection.fasterrcnn_resnet50_fpn` - Standard Faster R-CNN
- `scripts/eval/evaluate_detection_performance.py` - Evaluation script

**Why Works**:
- Simple, proven architecture
- 0.2659 mAP (acceptable baseline)
- Can be improved with confidence calibration
- Needs more data for higher mAP

**Checkpoint**: `models/best_model_regularized.pth` or similar

---

### 🆕 New Approaches (READY NOW)
**Files**:
- `scripts/train/confidence_calibration.py` - Focal loss + temperature
- `scripts/train/train_with_confidence.py` - Confidence-focused training
- `scripts/eval/visualize_confidence_improvement.py` - Show improvements

**Why Works**:
- Tested approach (focal loss, class weights, temperature scaling)
- Expected: 0.2659 → 0.28-0.30 mAP (+5-10%)
- More importantly: confidence 0.125 → 0.82+ (540% increase)

---

## 🤔 Should You Remove Old Code?

### Option 1: REMOVE Completely ❌
**Pros**:
- Cleaner codebase
- No confusion about what to use
- Easier to maintain

**Cons**:
- Lose the failed experiment for reference
- Hard to remember why it failed in 6 months
- Can't show the learning journey

**When to choose**: If focused on production only

---

### Option 2: ARCHIVE for Learning ✅ RECOMMENDED
**Pros**:
- Keep historical record of what failed
- Educational value (learn why they failed)
- Can reference when other ideas arise
- Shows due diligence

**Cons**:
- Slightly larger codebase
- Potential for accidental use

**When to choose**: If want to learn and document progress

**My recommendation**: Archive old approaches

---

### Option 3: HYBRID (Best Practice) ✅ BEST
**Keep**:
- Baseline model code (fast, works, reference)
- New confidence calibration (upgrade path)

**Archive**:
- Old failed 4-stage pipeline
- SSL pretraining (not effective)
- Multi-task learning (competing gradients)

**Document**:
- Why each failed
- What to do instead

---

## 📋 Cleanup Action Plan (RECOMMENDED)

### Step 1: Keep These (Core Pipeline)
```
✅ KEEP:
   scripts/train/train_baseline_faster_rcnn.py (rename/create)
   scripts/train/confidence_calibration.py (NEW)
   scripts/train/train_with_confidence.py (NEW)
   scripts/eval/evaluate_detection_performance.py
   src/models/ - Keep basic RCNN loading, archive others
```

### Step 2: Archive These (Failed Experiments)
```
📦 ARCHIVE to: scripts/train/archived_failed_approaches/
   ├─ train_full_pipeline.py (4-stage pipeline that failed)
   ├─ ssl_pretraining.py (SSL pretraining that didn't help)
   └─ ARCHIVE_README.md (explain why they failed)

📦 ARCHIVE to: src/models/archived/
   ├─ enhanced_ppe_detector.py (multi-task learning failed)
   ├─ relational_rescorer.py (spatial constraints didn't work)
   ├─ loss.py (custom losses not needed)
   └─ ARCHIVE_README.md
```

### Step 3: Archive Checkpoints
```
📦 ARCHIVE to: models/archived_failed_models/
   ├─ ppe_enhanced_best.pth (0.0574 mAP, too low)
   ├─ README.md (why it failed)
```

### Step 4: Create Clean README
```
📄 Create: docs/CODEBASE_STRUCTURE.md
   - What to use (baseline + confidence calibration)
   - What not to use (archived approaches)
   - Why each archived
   - How to upgrade baseline
```

---

## 🎯 My Recommendation: DO THIS

### Phase 1: Archive (Today - 30 minutes)
```bash
# 1. Create archive folders
mkdir -p scripts/train/archived_failed_approaches
mkdir -p src/models/archived
mkdir -p models/archived_failed_models

# 2. Move old files
Move-Item scripts/train/train_full_pipeline.py scripts/train/archived_failed_approaches/
Move-Item scripts/train/ssl_pretraining.py scripts/train/archived_failed_approaches/
Move-Item src/models/enhanced_ppe_detector.py src/models/archived/
Move-Item src/models/relational_rescorer.py src/models/archived/
Move-Item src/models/loss.py src/models/archived/
Move-Item models/ppe_enhanced_best.pth models/archived_failed_models/

# 3. Create documentation
# Create archived_failed_approaches/ARCHIVE_README.md
# Create src/models/archived/ARCHIVE_README.md
# Create models/archived_failed_models/README.md
```

### Phase 2: Create Clean Training Script (Tomorrow - 1 hour)
```
Create: scripts/train/train_baseline_faster_rcnn.py
   - Clean, simple baseline training
   - Uses torchvision standard Faster R-CNN
   - No fancy multi-task stuff
   - Save to: models/baseline_faster_rcnn_best.pth
```

### Phase 3: Create Upgrade Script (Tomorrow - 1 hour)
```
Create: scripts/train/train_baseline_with_confidence.py
   - Same baseline but with confidence calibration
   - Uses confidence_calibration.py module
   - Save to: models/baseline_with_confidence_best.pth
```

### Phase 4: Update Documentation (Tomorrow - 1 hour)
```
Create: docs/TRAINING_GUIDE_CLEAN.md
   1. What to use: Baseline with confidence
   2. What NOT to use: Multi-task learning (archived)
   3. Why: Failed experiments explained
   4. How: Step-by-step training
```

---

## 📊 What You'll End Up With

### Current Codebase (Messy)
```
scripts/train/
  ├─ train_full_pipeline.py       ❌ Failed 4-stage
  ├─ ssl_pretraining.py            ❌ Failed SSL
  ├─ confidence_calibration.py      ✅ New focal loss
  ├─ train_with_confidence.py       ✅ New training
  └─ archive_old_versions.py        ❓ Unclear purpose

src/models/
  ├─ enhanced_ppe_detector.py       ❌ Failed multi-task
  ├─ relational_rescorer.py         ❌ Failed spatial
  ├─ loss.py                        ❌ Not needed
  └─ hybrid_ppe_model.py            ❓ Not used
```

### Clean Codebase (Organized)
```
scripts/train/
  ├─ train_baseline_faster_rcnn.py        ✅ Simple baseline
  ├─ train_baseline_with_confidence.py    ✅ Recommended
  ├─ confidence_calibration.py            ✅ Helper module
  └─ archived_failed_approaches/          📚 Learning reference
       ├─ train_full_pipeline.py          (4-stage failed)
       ├─ ssl_pretraining.py              (SSL failed)
       └─ ARCHIVE_README.md               (Why archived)

src/models/
  ├─ __init__.py                   ✅ Standard imports
  ├─ archived/                     📚 Learning reference
  │    ├─ enhanced_ppe_detector.py (Multi-task failed)
  │    ├─ relational_rescorer.py   (Spatial failed)
  │    └─ ARCHIVE_README.md
  └─ (standard torchvision models) ✅ What to use

models/
  ├─ baseline_faster_rcnn_best.pth ✅ Production baseline
  ├─ baseline_with_confidence_best.pth ✅ Recommended
  └─ archived_failed_models/       📚 Reference
       ├─ ppe_enhanced_best.pth    (0.0574 mAP, failed)
       └─ README.md
```

---

## 🧠 Why Archive Instead of Delete?

### Benefits of Archiving
1. **Learning reference**: See why each approach failed
2. **Future ideas**: If someone proposes multi-task learning again, you can reference this
3. **Due diligence**: Shows you did proper experimentation
4. **Code snippets**: Reuse pieces (e.g., augmentation from enhanced model)
5. **Git history**: Code not lost, just organized

### Example: Using Archived Code
```python
# 6 months later, someone might say: "Let's try multi-task learning"
# You can point them to: scripts/train/archived_failed_approaches/ARCHIVE_README.md
# Which explains: "We tried this with 222 images, lost 78.8% mAP
#                 Reason: competing gradients. Need 1000+ images for this to work."
```

---

## ✅ My Final Recommendation

### DO THIS NOW (30 min):

1. **Archive old approaches**
   ```bash
   mkdir scripts/train/archived_failed_approaches
   mkdir src/models/archived
   mkdir models/archived_failed_models
   
   # Move files
   Move-Item scripts/train/train_full_pipeline.py scripts/train/archived_failed_approaches/
   Move-Item scripts/train/ssl_pretraining.py scripts/train/archived_failed_approaches/
   Move-Item src/models/enhanced_ppe_detector.py src/models/archived/
   Move-Item src/models/relational_rescorer.py src/models/archived/
   Move-Item src/models/loss.py src/models/archived/
   Move-Item models/ppe_enhanced_best.pth models/archived_failed_models/
   ```

2. **Create archive documentation**
   - `scripts/train/archived_failed_approaches/ARCHIVE_README.md`
   - `src/models/archived/ARCHIVE_README.md`
   - `models/archived_failed_models/README.md`

3. **Rename/clean main training scripts**
   - Keep: `confidence_calibration.py`, `train_with_confidence.py`
   - Create simple baseline training

4. **Update docs**
   - Show what to use
   - Show what NOT to use and why

---

## Files to Create for Archive Documentation

### scripts/train/archived_failed_approaches/ARCHIVE_README.md
```markdown
# Archived Failed Approaches

## Why These Are Here

These files represent experiments that failed:

### train_full_pipeline.py (4-Stage Option D)
- Purpose: Multi-task learning with SSL pretraining
- Result: 0.0574 mAP vs baseline 0.2659 (78.8% WORSE)
- Failure reason: 
  - Competing gradients (detection vs segmentation)
  - Small object detection lost (0% recall)
  - Person hallucination (92% false positives)
- Key lesson: Multi-task learning needs 1000+ images, not 222

### ssl_pretraining.py (Self-Supervised Pretraining)
- Purpose: Improve feature learning with SimCLR
- Result: Pretraining completed but didn't help detection
- Failure reason:
  - Limited dataset (222 images) insufficient for SSL
  - ImageNet pretraining already good
- Key lesson: SSL needs larger, more diverse datasets

## What To Use Instead

See: scripts/train/train_baseline_with_confidence.py
This uses simple Faster R-CNN with confidence calibration (focal loss).
```

---

## Summary

| Action | What | Why | Time |
|--------|------|-----|------|
| **Archive** | Old 4-stage pipeline | Failed experiments, keep for reference | 30 min |
| **Archive** | SSL pretraining | Ineffective, document why | 5 min |
| **Keep** | Confidence calibration | New, working approach | - |
| **Create** | Clean baseline training | Simple, documented path | 1 hour |
| **Document** | Archive README | Why each failed | 30 min |
| **Update** | Training docs | What to use now | 30 min |
| **TOTAL** | - | Organize and document | **2-3 hours** |

---

## Answer to Your Question

**Should you remove old code?**

**Answer**: No, archive it instead.

**Why**:
1. Keeps historical record of failed experiments
2. Educational value (why each failed)
3. Prevents accidental re-trying
4. Shows proper experimentation

**What to do**:
1. Move old files to `archived_failed_approaches/` folder
2. Create README explaining why each failed
3. Create clean new training scripts
4. Update documentation

**Time**: 2-3 hours total
**Outcome**: Clean, organized, documented codebase

---

## Next Question (When Ready)

Once archived, ask me: "How should I organize the new training pipeline?"

I'll help you create:
1. Simple baseline training script
2. Baseline + confidence calibration training
3. Clear documentation on what to use
4. Easy path to reproduce results

Generated: October 26, 2025
