# 📦 Centralized Archive - Archived Experiments & Old Code

This folder centralizes all archived code from failed experiments and old approaches. Everything here is for **reference only** - do not use these files in production.

## 📂 Structure

```
_ARCHIVED_EXPERIMENTS/
├── training_scripts/          # Old training scripts and experimental training approaches
├── model_files/               # Old model component implementations
├── checkpoints/               # Failed model checkpoints
└── experimental_scripts/      # Miscellaneous experimental code
```

## 📋 Inventory

### 🔴 Training Scripts (`training_scripts/`)

#### ❌ Failed Multi-Task Learning
- **train_full_pipeline.py** - 4-stage multi-task detection (detected objects, segmentation masks, spatial constraints, pseudo-labels)
  - **Status**: Failed - 78.8% worse than baseline (0.0574 mAP vs 0.2659)
  - **Why**: Competing gradients between 4 different tasks, too complex for 222 training images
  - **Lesson**: Keep models simple when data is limited

- **train_full_pipeline_resumable.py** - Same as above with resume capability
  - **Status**: Failed - same issues as train_full_pipeline.py
  - **Why**: Resume feature didn't fix underlying architectural issues

#### ❌ SSL Pretraining (Ineffective)
- **ssl_pretraining.py** - Self-supervised pretraining on ImageNet
  - **Status**: Ineffective - no improvement over standard ImageNet
  - **Why**: Dataset too small (222 images), standard ImageNet already sufficient
  - **Lesson**: SSL only helps with limited labeled data + large unlabeled pool

#### ❌ Data Augmentation Approaches
- **train_with_augmentation.py** - Heavy data augmentation
  - **Status**: Not effective on small dataset
  - **Why**: Augmentation can't replace real data

#### ✅ Previous Working Versions (Now Archived)
- **train.py**, **train_simple.py**, **simple_train.py** - Earlier baseline approaches
  - **Status**: Superseded by modern scripts
  - **Location**: Use `scripts/train/train_with_confidence.py` instead

- **train_regularized.py** - Baseline with L2 regularization
  - **Status**: Superseded by confidence calibration approach
  - **Use**: Modern implementation has better confidence calibration

#### 🧪 Experimental/Diagnostic Scripts
- **train_dynamic.py** - Dynamic learning rate scheduling
- **continue_training.py** - Resume training from checkpoint
- **split_and_train.py** - Train/val/test split utilities
- **plot_training.py** - Visualization utilities
- **test_model.py**, **simple_test.py** - Inference testing
- **archive_old_versions.py** - Archive management utility

### 🔴 Model Files (`model_files/`)

#### ❌ Multi-Task Detection Model
- **enhanced_ppe_detector.py** - 4-stage multi-task detector
  - **Status**: Failed - competing gradients, -78.8% mAP loss
  - **Why**: 4 task heads fighting for gradient updates
  - **Lesson**: Use simple baseline + post-processing instead of end-to-end multi-task

#### ❌ Spatial Constraints
- **relational_rescorer.py** - Graph-based spatial relationship modeling
  - **Status**: Ineffective - too restrictive, no improvement
  - **Why**: Hard constraints removed valid detections
  - **Lesson**: Let model learn soft relationships through data

#### ❌ Custom Loss Functions
- **loss.py** - Custom loss implementations
  - **Status**: Not needed - standard losses work better
  - **Why**: Standard PyTorch losses already well-tuned for detection
  - **Lesson**: Don't reinvent the wheel; use battle-tested implementations

### 🔴 Checkpoints (`checkpoints/`)

#### ❌ Enhanced Model Checkpoints (DO NOT USE)
- **ppe_enhanced_best.pth** - Best checkpoint from 4-stage multi-task training
  - **mAP**: 0.0574 (vs baseline 0.2659)
  - **Confidence**: 0.125 (needs 0.8+)
  - **Performance**: 78.8% worse than baseline
  - **Lesson**: Complex models fail on small datasets

- **ppe_enhanced_final.pth** - Final checkpoint from multi-task training
  - **Status**: Same poor performance as best checkpoint
  - **Why**: Model never converged to good solution

- **ssl_backbone_best.pth**, **ssl_backbone_final.pth** - SSL pretraining checkpoints
  - **Status**: No improvement over standard ImageNet
  - **Lesson**: SSL pretraining ineffective without large unlabeled pool

#### ✅ Production Checkpoints (In Main `models/` Folder)
- Use baseline checkpoints: `rcnn_baseline.pth`, `rcnn_baseline_adamw.pth`
- Location: `models/` folder (not archived)

### 🧪 Experimental Scripts (`experimental_scripts/`)

Reserved for future failed experiments and prototypes.

## 🎯 Current Production Approach

**Use these files in `scripts/train/`** (not archived):
- ✅ `train_with_confidence.py` - Baseline + confidence calibration
- ✅ `confidence_calibration.py` - Focal loss + temperature scaling
- ✅ Standard torchvision Faster R-CNN

**Expected Performance**:
- mAP: 0.2659 → 0.28-0.30 (+5-10%)
- Confidence: 0.125 → 0.82+ (540% increase)

## 📚 Why Each Approach Failed

### 4-Stage Multi-Task Learning (-78.8% mAP)
**Architecture**:
1. Shared backbone (ResNet50)
2. Detection head (class + bbox)
3. Segmentation head (mask)
4. Spatial constraint head (relationships)

**The Problem**:
- 4 different gradient signals competing on same backbone
- With only 222 training images, model couldn't balance all tasks
- Gradients from segmentation/spatial tasks overwhelmed detection
- Result: Detection quality collapsed completely

**Why Not Just Use It as Regularization?**
- If it were just regularizing, would see small improvement
- Instead saw 78.8% degradation
- Suggests tasks are fundamentally conflicting, not complementary

**Key Lesson**: On small data (< 1K images), keep models simple. Stick with single-task learning + post-processing.

### Self-Supervised Pretraining (Ineffective)
**Approach**: Train self-supervised encoder on unlabeled images, fine-tune on labeled data

**Why It Failed**:
- Requires large unlabeled pool (10K-100K images minimum)
- Our dataset: 222 labeled images, no large unlabeled pool
- ImageNet pretraining already provides good initialization
- Additional SSL pretraining didn't improve over direct fine-tuning

**Key Lesson**: SSL is for limited labeled data + large unlabeled pool. When you have neither (small labeled + no unlabeled), stick with standard pretraining.

### Spatial Constraints (Too Restrictive)
**Approach**: Hard rules about which object pairs can coexist

**Examples**:
- "Helmet only valid if Hard Hat area > 50 pixels"
- "Goggles must be within 30px of face"

**Why It Failed**:
- Removed valid detections at different scales
- Removed detections at image edges
- Database biases worse than model errors

**Key Lesson**: Let the model learn soft relationships through data. Hard constraints are brittle and domain-specific.

## 🚀 What to Do Instead

### For 0.27-0.30 mAP (5-10% improvement)
✅ **Use confidence calibration** (current approach)
- Focal loss for hard examples
- Class-weighted sampling
- Temperature scaling for calibration
- Expected: 0.2659 → 0.28-0.30 mAP

### For 0.40+ mAP (50% improvement)
1. Collect 300-500 more images (biggest lever)
2. Fix small object detection:
   - Increase input resolution to 1024x1024
   - Add small object anchors
3. Hard negative mining (focus on worst FPs)

### For 0.75+ mAP (long-term goal)
1. Collect 1000+ images
2. Better backbone (ResNet101, EfficientNet)
3. Multi-scale training and testing
4. Test-time augmentation (TTA)
5. Ensemble methods

## ⚠️ DO NOT USE

- ❌ `train_full_pipeline.py` - Use baseline + confidence calibration instead
- ❌ `ssl_pretraining.py` - Standard ImageNet pretraining is sufficient
- ❌ `enhanced_ppe_detector.py` - Keep detection simple
- ❌ `relational_rescorer.py` - Use post-processing filters instead
- ❌ `ppe_enhanced_best.pth` - Use baseline checkpoints instead

## 📖 Archive Guidelines

**When to Archive**:
1. Experiment shows < 1% improvement and adds complexity
2. Approach incompatible with data size (e.g., SSL on 222 images)
3. Better simpler baseline exists
4. Implementation introduces maintenance burden

**When NOT to Archive**:
1. Shows > 1-2% improvement
2. Improves speed significantly
3. Reduces memory usage significantly
4. Is actively used in production

**Archive Format**:
1. Keep original code for reference
2. Add README explaining why it failed
3. Document expected vs actual results
4. List key lessons learned
5. Recommend alternative approaches

## 📝 Git Recommendations

When committing archive changes:
```bash
# Archive old experiments, consolidate to _ARCHIVED_EXPERIMENTS/
git rm -r scripts/archived
git rm -r scripts/train/archived_*
git rm -r src/models/archived
git rm -r models/archived_failed_models
git add _ARCHIVED_EXPERIMENTS/
git commit -m "chore: consolidate archives to _ARCHIVED_EXPERIMENTS/ folder"
```

This keeps git history clean while centralizing archives.

## 📞 Questions?

- Why did multi-task fail? → See ANALYSIS_MULTITASK_FAILURE.md
- Why didn't SSL help? → See ANALYSIS_SSL_INEFFECTIVE.md
- What should I use? → Use `scripts/train/train_with_confidence.py`
- How do I improve further? → See IMPROVEMENT_RECOMMENDATIONS.md in docs/

---

**Status**: All archived code consolidated and documented.
**Next Step**: Use `train_with_confidence.py` to train baseline with confidence calibration.
**Expected Results**: 0.2659 → 0.28-0.30 mAP, confidence 0.125 → 0.82+
