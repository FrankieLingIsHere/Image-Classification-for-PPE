#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RESUMABLE TRAINING LAUNCHER
Allows training on CPU now, then switch to GPU/better device later and continue

Quick Usage:
  # Start training on CPU
  python run_resumable_training.py --device cpu --detection-epochs 5

  # Later, switch to GPU and continue
  python run_resumable_training.py --resume --device cuda

  # Explicitly specify checkpoint
  python run_resumable_training.py --resume-checkpoint models/detection_checkpoint_latest.pth --device cuda
"""

import subprocess
import sys
from pathlib import Path
from datetime import datetime
import json

def find_latest_checkpoint(output_dir='models'):
    """Find the latest checkpoint to resume from."""
    output_dir = Path(output_dir)
    
    # Look for latest checkpoints in order of preference
    for pattern in [
        'detection_checkpoint_latest.pth',
        'ssl_checkpoint_latest.pth',
    ]:
        checkpoint = output_dir / pattern
        if checkpoint.exists():
            return checkpoint
    
    return None


def list_checkpoints(output_dir='models'):
    """List all available checkpoints."""
    output_dir = Path(output_dir)
    
    checkpoints = sorted(
        output_dir.glob('*_checkpoint_*.pth'),
        key=lambda p: p.stat().st_mtime,
        reverse=True
    )
    
    if not checkpoints:
        return None
    
    print("\n[CHECKPOINTS] Available checkpoints:")
    for i, ckpt in enumerate(checkpoints[:10], 1):
        size_mb = ckpt.stat().st_size / (1024 * 1024)
        mtime = datetime.fromtimestamp(ckpt.stat().st_mtime)
        metadata = f"({size_mb:.1f}MB, {mtime.strftime('%Y-%m-%d %H:%M')})"
        
        # Try to extract epoch info
        if 'epoch' in ckpt.name:
            parts = ckpt.name.split('_')
            epoch_str = [p for p in parts if p.isdigit()]
            if epoch_str:
                metadata += f" [Epoch {epoch_str[-1]}]"
        
        print(f"  [{i}] {ckpt.name} {metadata}")
    
    return checkpoints


def main():
    """Main launcher."""
    
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Resumable Training Pipeline Launcher',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Start training on CPU
  python run_resumable_training.py --device cpu --detection-epochs 10

  # Continue on GPU using latest checkpoint
  python run_resumable_training.py --resume --device cuda

  # Continue with specific checkpoint
  python run_resumable_training.py --resume-checkpoint models/detection_checkpoint_epoch_005.pth --device cuda

  # Check available checkpoints
  python run_resumable_training.py --list-checkpoints
        """
    )
    
    # Training control
    parser.add_argument('--resume', action='store_true',
                        help='Resume from latest checkpoint')
    parser.add_argument('--resume-checkpoint', type=str, default=None,
                        help='Specific checkpoint to resume from')
    parser.add_argument('--list-checkpoints', action='store_true',
                        help='List available checkpoints and exit')
    
    # Training parameters
    parser.add_argument('--ssl-epochs', type=int, default=20,
                        help='SSL pretraining epochs (default: 20)')
    parser.add_argument('--detection-epochs', type=int, default=50,
                        help='Detection training epochs (default: 50)')
    parser.add_argument('--batch-size', type=int, default=4,
                        help='Batch size (default: 4)')
    parser.add_argument('--lr', type=float, default=5e-5,
                        help='Learning rate (default: 5e-5)')
    
    # Paths
    parser.add_argument('--data-dir', type=str, default='data',
                        help='Data directory (default: data)')
    parser.add_argument('--output-dir', type=str, default='models',
                        help='Output directory (default: models)')
    
    # Device
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device: cuda or cpu (default: cuda)')
    
    args = parser.parse_args()
    
    # Just list checkpoints and exit
    if args.list_checkpoints:
        print("\n" + "="*80)
        print("= AVAILABLE CHECKPOINTS")
        print("="*80)
        
        checkpoints = list_checkpoints(args.output_dir)
        
        if checkpoints:
            print(f"\n✓ Found {len(checkpoints)} checkpoint(s)")
            print(f"\n💡 To resume from any checkpoint, run:")
            print(f"   python run_resumable_training.py --resume")
        else:
            print("❌ No checkpoints found. Start new training with:")
            print(f"   python run_resumable_training.py --device {args.device}")
        
        return
    
    # Check for resume
    resume_checkpoint = args.resume_checkpoint
    
    if args.resume:
        resume_checkpoint = find_latest_checkpoint(args.output_dir)
        
        if resume_checkpoint:
            print("\n✅ Found latest checkpoint:")
            print(f"   {resume_checkpoint}")
        else:
            print("\n❌ No checkpoint found to resume from!")
            print("   Starting fresh training instead...")
            args.resume = False
    
    # Display info
    print("\n" + "="*80)
    print("= RESUMABLE TRAINING PIPELINE")
    print("="*80)
    
    if resume_checkpoint:
        print(f"""
📂 Resuming from: {resume_checkpoint}

Training Configuration:
  Device:              {args.device}
  Detection Epochs:    {args.detection_epochs}
  Batch Size:          {args.batch_size}
  Learning Rate:       {args.lr}

💡 To interrupt:
   Press Ctrl+C anytime

💡 To resume later:
   python run_resumable_training.py --resume --device {args.device}

💡 To see all checkpoints:
   python run_resumable_training.py --list-checkpoints
        """)
    else:
        print(f"""
🚀 Starting fresh training

Training Configuration:
  SSL Epochs:          {args.ssl_epochs}
  Detection Epochs:    {args.detection_epochs}
  Device:              {args.device}
  Batch Size:          {args.batch_size}
  Learning Rate:       {args.lr}

💡 You can interrupt anytime with Ctrl+C

💡 To resume later on a different device:
   python run_resumable_training.py --resume --device cuda

💡 To see all checkpoints:
   python run_resumable_training.py --list-checkpoints
        """)
    
    confirm = input("Continue? [y/N]: ").lower()
    if confirm != 'y':
        print("Cancelled.")
        return
    
    # Build command
    project_root = Path(__file__).parent
    
    cmd = [
        sys.executable,
        str(project_root / 'scripts' / 'train' / 'train_full_pipeline.py'),
        '--ssl_epochs', str(args.ssl_epochs),
        '--detection_epochs', str(args.detection_epochs),
        '--batch_size', str(args.batch_size),
        '--lr', str(args.lr),
        '--data_dir', args.data_dir,
        '--output_dir', args.output_dir,
        '--device', args.device,
    ]
    
    if resume_checkpoint:
        cmd.extend(['--resume-checkpoint', str(resume_checkpoint)])
    
    # Run training
    print("\n" + "="*80)
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Starting training...")
    print("="*80 + "\n")
    
    try:
        result = subprocess.run(cmd, cwd=project_root)
        
        if result.returncode == 0:
            print("\n" + "✅ "*40)
            print("TRAINING COMPLETED SUCCESSFULLY")
            print("✅ "*40)
            
            checkpoints = list_checkpoints(args.output_dir)
            if checkpoints:
                print(f"\n💾 Best model: {args.output_dir}/detection_checkpoint_best.pth")
                
                print(f"\n🎯 Next steps:")
                print(f"   1. Evaluate model: python scripts/eval/evaluate_detection_performance.py")
                print(f"   2. Deploy to Streamlit: python streamlit_app.py")
                print(f"   3. Continue training: python run_resumable_training.py --resume")
        else:
            print(f"\n❌ Training failed with exit code {result.returncode}")
    
    except KeyboardInterrupt:
        print("\n\n⏸️  Training interrupted by user!")
        
        checkpoints = list_checkpoints(args.output_dir)
        if checkpoints:
            latest = checkpoints[0]
            print(f"\n💾 Latest checkpoint saved: {latest.name}")
            print(f"\n🔄 To resume training, run:")
            print(f"   python run_resumable_training.py --resume")
            print(f"\n   Or to continue on GPU:")
            print(f"   python run_resumable_training.py --resume --device cuda")


if __name__ == "__main__":
    main()
