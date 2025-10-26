#!/usr/bin/env python3
"""
Archive old training scripts to clean up the training directory.
This consolidates all legacy versions into archived_old_versions/ folder.
"""

import shutil
from pathlib import Path

# Files to archive (legacy versions not currently used)
LEGACY_FILES = [
    'train.py',
    'train_with_augmentation.py',
    'train_enhanced.py',
    'train_regularized.py',
    'rcnn_baseline.py',
    'continue_training.py',
    'split_and_train.py',
    'train_simple.py',
    'train_full_pipeline_resumable.py',  # Replaced by run_resumable_training.py
]

# Files to keep (currently active)
ACTIVE_FILES = [
    'train_full_pipeline.py',    # Main training script
    'ssl_pretraining.py',        # SSL component
]

def archive_old_files():
    """Move legacy training scripts to archived folder."""
    
    train_dir = Path(__file__).parent
    archive_dir = train_dir / 'archived_old_versions'
    
    # Create archive directory if it doesn't exist
    archive_dir.mkdir(exist_ok=True)
    
    print("\n" + "="*80)
    print("ARCHIVING OLD TRAINING SCRIPTS")
    print("="*80)
    
    archived_count = 0
    
    for filename in LEGACY_FILES:
        file_path = train_dir / filename
        
        if file_path.exists():
            dest_path = archive_dir / filename
            
            # Avoid overwriting if already archived
            if not dest_path.exists():
                print(f"\n📦 Archiving: {filename}")
                shutil.move(str(file_path), str(dest_path))
                archived_count += 1
                print(f"   → Moved to: archived_old_versions/{filename}")
            else:
                print(f"\n⏭️  Already archived: {filename}")
    
    print("\n" + "="*80)
    print(f"✓ Archived {archived_count} legacy scripts")
    print("="*80)
    
    print("\n📁 ACTIVE TRAINING SCRIPTS (in scripts/train/):")
    for filename in ACTIVE_FILES:
        file_path = train_dir / filename
        if file_path.exists():
            size_kb = file_path.stat().st_size / 1024
            print(f"  ✓ {filename:40s} ({size_kb:.1f} KB)")
    
    print("\n📦 ARCHIVED SCRIPTS (in scripts/train/archived_old_versions/):")
    for filename in LEGACY_FILES:
        file_path = archive_dir / filename
        if file_path.exists():
            size_kb = file_path.stat().st_size / 1024
            print(f"  📦 {filename:40s} ({size_kb:.1f} KB)")
    
    print("\n" + "="*80)
    print("✓ Training scripts cleanup complete!")
    print("\nTo use training:")
    print("  python run_resumable_training.py --device cuda")
    print("="*80)


if __name__ == "__main__":
    archive_old_files()
