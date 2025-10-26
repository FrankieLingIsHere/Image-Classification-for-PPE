# Quick Start Guide - Clone & Train

This guide lets you clone the repo and start training immediately.

## 1. Clone Repository
```bash
git clone https://github.com/FrankieLingIsHere/Image-Classification-for-PPE.git
cd Image-Classification-for-PPE
```

## 2. Install Dependencies

### Step 2a: Check CUDA Version
```bash
nvidia-smi
```
Look for CUDA version in output.

### Step 2b: Install PyTorch with Correct CUDA Version

**For CUDA 12.8 (RTX 5090 with SM 120 - RECOMMENDED):**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

**For CUDA 12.1 (RTX 4090, etc.):**
```bash
pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

**For CUDA 11.8 (older GPUs):**
```bash
pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

**For CPU-only:**
```bash
pip install torch==2.0.1 torchvision==0.15.2
pip install -r requirements.txt
```

**Verify Installation:**
```bash
python -c "import torch; print(f'PyTorch {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"
```

## 3. Verify Data Structure
Make sure you have:
```
data/
├── images/          # All training images
├── annotations/     # Annotation files (XML or JSON)
└── splits/
    ├── train.txt    # Image filenames for training (222 images)
    ├── val.txt      # Image filenames for validation (19 images)
    └── test.txt     # Image filenames for testing (25 images)
```

Check with:
```bash
ls data/images/ | wc -l      # Should show ~266 images
head data/splits/train.txt    # Should show image filenames
```

## 4. Check Label Distribution (Optional)
```bash
python scripts/tools/check_label_distribution.py
```

This shows you:
- Class balance (e.g., person: 403 instances, no_safety_boots: 16 instances)
- Data split distribution

## 5. Train Two-Stage Pipeline (RECOMMENDED)

### On GPU (RTX 5090: ~15-25 minutes)
```bash
python scripts/train/train_two_stage.py \
    --data_dir data \
    --epochs 50 \
    --batch_size 2 \
    --augment \
    --device cuda
```

### On CPU (2-4 hours)
```bash
python scripts/train/train_two_stage.py \
    --data_dir data \
    --epochs 50 \
    --batch_size 2 \
    --augment \
    --device cpu
```

## 6. Monitor Training
Open another terminal:
```bash
# Check GPU usage (if using CUDA)
nvidia-smi -l 1
```

## 7. Check Results
After training completes (look for `[OK] Training complete!`):

```bash
# View trained models
ls -lh models/stage1_human_best.pth
ls -lh models/stage2_ppe_best.pth

# View loss curves
cat models/training_history_two_stage.json | python -m json.tool
```

## Alternative: All-in-One Model
If two-stage is too complex, use:
```bash
python scripts/train/train_with_confidence.py \
    --data_dir data \
    --epochs 50 \
    --batch_size 2 \
    --augment \
    --class-weights \
    --focal-loss \
    --device cuda
```

## Expected Results
- **Stage 1 (Human detection):** mAP ~0.7-0.8
- **Stage 2 (PPE detection):** mAP ~0.28-0.32
- **Training loss:** Should decrease smoothly over epochs
- **Validation loss:** Should follow or track slightly above training loss

## Troubleshooting

### "No module named 'src'"
This is handled automatically. Make sure you run from project root:
```bash
cd Image-Classification-for-PPE
python scripts/train/train_two_stage.py ...
```

### "CUDA out of memory"
Reduce batch size:
```bash
--batch_size 1
```

### "No annotations found"
Check that annotation files exist:
```bash
ls data/annotations/ | head
```

### Training is too slow
Use GPU instead:
```bash
--device cuda  # instead of cpu
```

## Next Steps After Training
1. Evaluate on test set (create evaluation script)
2. Deploy trained models to inference pipeline
3. Create REST API for model serving

## Need Help?
Check the documentation:
- `docs/TRAINING_SCRIPT_UPDATES.md` - Detailed training script info
- `_ARCHIVED_EXPERIMENTS/README.md` - Why old approaches failed
- `docs/` - Other guides
