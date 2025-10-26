# START TRAINING - Quick Guide

## Overview
The training pipeline will run in **4 stages**:
1. **Stage 1:** Self-Supervised Pretraining (20 epochs) - Learn general visual features
2. **Stage 2-4:** Multi-Task Detection Training (50 epochs) - Learn PPE detection with spatial reasoning

**Expected Improvement:** mAP 0.028 → 0.50-0.60 (1700-2000% improvement!)

---

## Quick Start

### Option 1: Start on Current Device (Recommended)
```bash
python run_resumable_training.py --device cuda
```

If CUDA not available, automatically falls back to CPU.

### Option 2: Specify Device Explicitly
```bash
# On GPU
python run_resumable_training.py --device cuda --ssl-epochs 20 --detection-epochs 50

# On CPU (slower)
python run_resumable_training.py --device cpu --ssl-epochs 20 --detection-epochs 50
```

### Option 3: Quick Test (5 epochs each)
```bash
python run_resumable_training.py --device cuda --ssl-epochs 5 --detection-epochs 5
```

---

## Training Parameters

All parameters can be overridden:

```bash
python run_resumable_training.py \
  --device cuda \
  --ssl-epochs 20 \
  --detection-epochs 50 \
  --batch-size 8 \
  --learning-rate 0.001 \
  --resume  # Resume from latest checkpoint
```

---

## What Gets Trained

### Stage 1: SSL Pretraining (20 epochs)
- **Method:** SimCLR (Contrastive Learning)
- **Output:** `models/ssl_backbone_pretrained.pth`
- **Time:** ~5-10 min (GPU), ~30 min (CPU)
- **Purpose:** Learn general visual features

### Stages 2-4: Multi-Task Detection (50 epochs)
- **Detection:** Faster R-CNN ResNet50+FPN (12 PPE classes)
- **Segmentation:** 3-class semantic segmentation head
- **Spatial Constraints:** Learned plausibility matrix
- **Output:** `models/ppe_enhanced_best.pth`
- **Time:** ~30-60 min (GPU), ~2-3 hours (CPU)
- **Features:**
  - 7 augmentations (flip, rotate, color jitter, etc.)
  - Automatic checkpoint saving
  - Best model selection (by mAP)
  - Full evaluation at end

---

## Output Files

After training completes:

```
models/
├── ssl_checkpoint_latest.pth       # Latest SSL checkpoint
├── ssl_backbone_pretrained.pth     # Final SSL backbone
├── detection_checkpoint_latest.pth # Latest detection checkpoint
├── ppe_enhanced_best.pth           # Best detection model (USE THIS!)
└── training_results.json           # Metrics and stats

outputs/
└── evaluation_results/
    ├── metrics.json               # Final evaluation metrics
    └── visualizations/            # Detection visualizations
```

---

## Resume Training

If training gets interrupted:

```bash
# Resume from latest checkpoint automatically
python run_resumable_training.py --resume --device cuda

# Or specify explicit checkpoint
python run_resumable_training.py --resume-checkpoint models/detection_checkpoint_latest.pth --device cuda
```

---

## Monitor Training

While training runs, watch for:
- **SSL Phase:** Loss should decrease steadily
- **Detection Phase:** mAP should increase, confidence scores improve
- **Checkpoints:** Auto-saved every epoch
- **Best Model:** Selected when validation mAP improves

---

## Performance Expectations

| Metric | Baseline | After Training | Improvement |
|--------|----------|----------------|------------|
| mAP | 0.028 | 0.50-0.60 | 1700-2000% |
| Precision | ~40% | 80%+ | 2x |
| Recall | ~60% | 85%+ | 1.4x |
| FP Count | 356 | ~50 | 7x less |
| FN Count | 186 | ~30 | 6x less |

---

## Troubleshooting

### Out of Memory (CUDA)
```bash
# Reduce batch size
python run_resumable_training.py --device cuda --batch-size 4
```

### Slow Training
- Ensure GPU is being used: `nvidia-smi`
- Check CPU isn't maxed out (should be <50%)
- Can continue on faster GPU later: `--resume --device cuda`

### Training Stuck
- Check if data loads: `python scripts/train/train_full_pipeline.py --test-data`
- Verify models: `python verify_architecture.py`

---

## Next Steps After Training

1. **Evaluate Model:**
   ```bash
   python scripts/eval/evaluate_detection_performance.py \
     --model_path models/ppe_enhanced_best.pth \
     --data_dir data \
     --output_dir outputs/final_evaluation \
     --split test
   ```

2. **Deploy to Streamlit:**
   ```bash
   python streamlit_app.py
   ```
   The app will automatically detect the best trained model!

3. **Generate Visualizations:**
   ```bash
   python scripts/visualize/visualize_detections.py \
     --model_path models/ppe_enhanced_best.pth \
     --data_dir data \
     --output_dir outputs/visualizations
   ```

---

## Questions?

- Check `docs/TRAINING_GUIDE.md` for detailed documentation
- Review `docs/QUICK_REFERENCE.md` for common commands
- See `docs/TRAINING_CLEANUP_SUMMARY.md` for file organization

**Ready to train? Run:** `python run_resumable_training.py --device cuda`
