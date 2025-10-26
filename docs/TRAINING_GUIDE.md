# Training Scripts Organization

## Active Training Scripts (Currently Used)

### 1. **train_full_pipeline.py** ⭐ MAIN SCRIPT
- **Purpose**: Full Option D training pipeline with all stages
- **Features**: SSL pretraining + Multi-task detection + Spatial constraints
- **Usage**:
  ```bash
  python scripts/train/train_full_pipeline.py \
    --ssl_epochs 20 \
    --detection_epochs 50 \
    --batch_size 4 \
    --device cuda
  ```
- **Output**: `models/ppe_enhanced_best.pth`

### 2. **ssl_pretraining.py** ⭐ UTILITY SCRIPT
- **Purpose**: Self-supervised pretraining component (Stage 1)
- **Features**: SimCLR contrastive learning with 7 augmentations
- **Used by**: `train_full_pipeline.py` (imported automatically)
- **Can also run standalone**:
  ```bash
  python scripts/train/ssl_pretraining.py \
    --epochs 20 \
    --batch_size 32 \
    --device cuda
  ```
- **Output**: `models/ssl_backbone_best.pth`

## Launcher Scripts

### **run_full_training.py** (in project root) 🚀 USER ENTRY POINT
- Interactive launcher for full training
- Handles all setup and error checking
- Recommended entry point for users

### **run_resumable_training.py** (in project root) 🔄 RESUMABLE TRAINING
- Resumable training with checkpoints
- Can pause and resume across devices (CPU → GPU)
- Recommended for long training sessions
- Supports `--resume` flag for continuing training

## Archived Legacy Scripts

All older versions have been moved to `archived_old_versions/`:
- `train.py` - Old basic training
- `train_with_augmentation.py` - Augmentation experiments (superseded)
- `train_enhanced.py` - Early enhancement attempts (superseded)
- `train_regularized.py` - Regularization experiments (superseded)
- `rcnn_baseline.py` - Baseline without enhancements (reference only)
- `continue_training.py` - Old resume mechanism (superseded by run_resumable_training.py)
- `split_and_train.py` - Old preprocessing (superseded)
- `train_simple.py` - Simplified version (reference only)
- `train_full_pipeline_resumable.py` - Intermediate version (superseded)

These are kept for reference but should NOT be used for new training.

## Quick Start Guide

### For New Training on GPU:
```bash
# Full training (recommended)
python run_full_training.py

# Or use launcher with options
python run_resumable_training.py --device cuda --ssl-epochs 20 --detection-epochs 50
```

### For Training on CPU (Small Dataset Testing):
```bash
python run_resumable_training.py --device cpu --ssl-epochs 1 --detection-epochs 2
```

### For Resuming Training After Interruption:
```bash
python run_resumable_training.py --resume --device cuda
```

### For Explicit Training Script (Advanced):
```bash
python scripts/train/train_full_pipeline.py \
  --ssl_epochs 20 \
  --detection_epochs 50 \
  --batch_size 4 \
  --lr 5e-5 \
  --device cuda
```

## Architecture Overview

```
Entry Point (run_full_training.py or run_resumable_training.py)
    ↓
    └─→ train_full_pipeline.py
            ├─→ Stage 1: ssl_pretraining.py
            │   └─→ SimCLR contrastive learning (20 epochs)
            │
            ├─→ Stage 2-4: Enhanced Detection Training
            │   ├─→ Multi-task learning (detection + segmentation)
            │   ├─→ Spatial constraints module
            │   └─→ Context-aware inference (50 epochs)
            │
            └─→ Output: models/ppe_enhanced_best.pth
```

## File Structure

```
scripts/train/
├── train_full_pipeline.py          ⭐ ACTIVE - Main training script
├── ssl_pretraining.py              ⭐ ACTIVE - SSL component
├── train_full_pipeline_resumable.py (deprecated, replaced by run_resumable_training.py)
├── archived_old_versions/          (legacy training scripts)
│   ├── train.py
│   ├── train_with_augmentation.py
│   ├── train_enhanced.py
│   ├── train_regularized.py
│   ├── rcnn_baseline.py
│   ├── continue_training.py
│   ├── split_and_train.py
│   └── train_simple.py
│
├── (other directories: analysis/, eval/, visualization/, etc.)
└── README.md (this file)

project_root/
├── run_full_training.py            🚀 User entry point
├── run_resumable_training.py        🔄 Resumable training entry point
└── (other project files)
```

## Training Configuration

### Recommended Settings

**For RTX 5090 (High-end):**
```bash
python run_resumable_training.py \
  --device cuda \
  --ssl-epochs 20 \
  --detection-epochs 50 \
  --batch-size 8
```
Expected time: 30-60 minutes

**For RTX 4090 (High-end):**
```bash
python run_resumable_training.py \
  --device cuda \
  --ssl-epochs 20 \
  --detection-epochs 50 \
  --batch-size 4
```
Expected time: 1-1.5 hours

**For RTX 3090 (Mid-range):**
```bash
python run_resumable_training.py \
  --device cuda \
  --ssl-epochs 20 \
  --detection-epochs 50 \
  --batch-size 2
```
Expected time: 2-3 hours

**For CPU (Testing Only):**
```bash
python run_resumable_training.py \
  --device cpu \
  --ssl-epochs 1 \
  --detection-epochs 2
```
Expected time: ~2 hours (for verification only)

## Checkpoints & Resuming

Checkpoints are automatically saved to `models/`:
- `ssl_checkpoint_best.pth` - Best SSL model
- `detection_checkpoint_best.pth` - Best detection model
- `detection_checkpoint_latest.pth` - Latest checkpoint (for resuming)
- `detection_checkpoint_epoch_*.pth` - Per-epoch checkpoints

To resume from a checkpoint:
```bash
python run_resumable_training.py --resume --device cuda
```

## Expected Results

After full training:
```
Before (baseline):          After (Option D):
mAP: 0.028 (2.8%)          mAP: 0.50-0.60 (50-60%)
Precision: 50%              Precision: 80%+
Recall: 60%                 Recall: 85%+
FP Count: 356               FP Count: ~50 (-86%)
Missed: 186                 Missed: ~25 (-87%)

Improvement: 1700-2000% better! 🎉
```

## Troubleshooting

### "CUDA out of memory"
```bash
# Reduce batch size
python run_resumable_training.py --resume --batch-size 2 --device cuda
```

### "Training seems stuck"
Monitor GPU usage:
```bash
watch -n 1 nvidia-smi
```

### "Need to switch from CPU to GPU"
Just run with `--resume`:
```bash
# Previously trained on CPU
# Now resume on GPU
python run_resumable_training.py --resume --device cuda
```

### "Lost progress / Lost checkpoint"
Checkpoints are saved every epoch. Check:
```bash
python run_resumable_training.py --list-checkpoints
```

## For Developers

When adding new training features:
1. Update `train_full_pipeline.py` (main script)
2. Update `ssl_pretraining.py` if SSL-related
3. Do NOT create new training scripts without discussion
4. Archive old versions to `archived_old_versions/`
5. Update this README

## Notes

- Only use `train_full_pipeline.py` for new training
- Use launchers (`run_*.py`) for user-friendly interface
- Archived scripts are kept for reference/history only
- All augmentation happens in dataset classes
- Multi-task learning combines detection + segmentation losses
- Spatial constraints learned during training
