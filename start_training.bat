@echo off
cd /d "c:\Users\102782407\Documents\PPE\Image-Classification-for-PPE"
set PYTHONPATH=c:\Users\102782407\Documents\PPE\Image-Classification-for-PPE
python scripts/train/rcnn_baseline.py --epochs 15 --batch_size 2 --lr 0.0001 --augment --optimizer adamw --step_lr --device cpu
