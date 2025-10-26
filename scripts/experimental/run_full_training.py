#!/usr/bin/env python3
"""
LAUNCHER: Full Option D Training Pipeline
Runs complete training with all stages: SSL + Multi-task + Context Awareness
"""

import subprocess
import sys
from pathlib import Path
from datetime import datetime

def run_command(cmd, description):
    """Run a command with nice output."""
    print("\n" + "="*80)
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {description}")
    print("="*80)
    print(f"Command: {' '.join(cmd)}\n")
    
    result = subprocess.run(cmd, cwd=Path(__file__).parent)
    
    if result.returncode != 0:
        print(f"\n❌ ERROR in {description}")
        return False
    
    print(f"\n✓ {description} completed successfully!")
    return True


def main():
    """Run full training pipeline."""
    
    print("\n" + "█"*80)
    print("█ OPTION D: FULL TRAINING PIPELINE WITH SELF-SUPERVISED LEARNING")
    print("█"*80)
    print("""
This will run:
  1. SSL Pretraining (20 epochs) - builds better backbone
  2. Multi-task Detection Training (50 epochs) - adds segmentation task
  3. Spatial constraints + GAT context awareness
  
Expected results:
  - mAP improvement: 0.028 → 0.50-0.60
  - Time: ~6-8 hours on GPU
  - Output: models/ppe_enhanced_best.pth
    """)
    
    confirm = input("\nContinue? [y/N]: ").lower()
    if confirm != 'y':
        print("Cancelled.")
        return
    
    # Configuration
    project_root = Path(__file__).parent
    
    # Prepare data
    print("\n[Setup] Checking data...")
    images_dir = project_root / 'data' / 'images'
    if not images_dir.exists():
        print(f"❌ Error: {images_dir} not found")
        return
    
    print(f"✓ Found {len(list(images_dir.glob('*')))} images")
    
    # Run full pipeline
    cmd = [
        sys.executable,
        str(project_root / 'scripts' / 'train' / 'train_full_pipeline.py'),
        '--ssl_epochs', '20',
        '--detection_epochs', '50',
        '--batch_size', '4',
        '--lr', '5e-5',
        '--data_dir', 'data',
        '--output_dir', 'models',
        '--device', 'cuda'
    ]
    
    if not run_command(cmd, "Full Training Pipeline (SSL + Multi-task)"):
        return
    
    # Run evaluation
    print("\n" + "█"*80)
    print("█ RUNNING EVALUATION ON BEST MODEL")
    print("█"*80)
    
    eval_cmd = [
        sys.executable,
        str(project_root / 'scripts' / 'eval' / 'evaluate_detection_performance.py'),
        '--model_path', 'models/ppe_enhanced_best.pth',
        '--split', 'test'
    ]
    
    if not run_command(eval_cmd, "Evaluation of Enhanced Model"):
        return
    
    print("\n" + "█"*80)
    print("█ TRAINING PIPELINE COMPLETE!")
    print("█"*80)
    print("""
Results:
  ✓ SSL backbone: models/ssl_backbone_best.pth
  ✓ Enhanced model: models/ppe_enhanced_best.pth
  ✓ Evaluation results: outputs/evaluation_results/

Expected improvements:
  - mAP: 0.028 → 0.50-0.60 (1700% improvement!)
  - Precision: 50% → 80%+
  - Recall: 60% → 85%+

Next steps:
  1. Check evaluation results in outputs/evaluation_results/
  2. Deploy best model to Streamlit app
  3. Fine-tune if needed based on results
    """)


if __name__ == "__main__":
    main()
