#!/usr/bin/env python3
"""
Split new images into train/test and optionally start training.

This script finds image files in data/images that are not listed in
data/splits/{train,test,val}.txt, assigns them to train or test so the two
splits are balanced, writes the updated split files, and then invokes the
existing training script (scripts/train.py) in a background process by
default.

Usage:
  python scripts/split_and_train.py        # update splits and start training
  python scripts/split_and_train.py --no-train  # update splits only
  python scripts/split_and_train.py --epochs 50  # pass through to train.py
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path


IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}


def read_split(path: Path):
    if not path.exists():
        return []
    with open(path, 'r', encoding='utf-8') as f:
        lines = [ln.strip() for ln in f.readlines() if ln.strip()]
    return lines


def write_split(path: Path, items):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        for it in items:
            f.write(f"{it}\n")


def find_images(images_dir: Path):
    if not images_dir.exists():
        return []
    files = [p.name for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    return sorted(files)


def main():
    parser = argparse.ArgumentParser(description='Add new images to train/test splits and optionally start training')
    parser.add_argument('--data_dir', type=str, default='data', help='Data directory (default: data)')
    parser.add_argument('--no-train', action='store_true', help='Do not start training after updating splits')
    parser.add_argument('--train-script', type=str, default='scripts/train.py', help='Path to training script')
    parser.add_argument('--train-args', type=str, default='', help='Extra args to pass to train.py (quoted)')
    parser.add_argument('--epochs', type=int, default=None, help='If set, adds --epochs to the training command')
    parser.add_argument('--train-ratio', type=float, default=None, help='If set, recompute train/test split ratio (e.g. 0.9 for 90%% train)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed used when recomputing splits')
    args = parser.parse_args()

    workspace_root = Path(__file__).resolve().parents[2]
    data_dir = workspace_root / args.data_dir
    images_dir = data_dir / 'images'
    splits_dir = data_dir / 'splits'

    train_path = splits_dir / 'train.txt'
    test_path = splits_dir / 'test.txt'
    val_path = splits_dir / 'val.txt'

    print(f"Workspace root: {workspace_root}")
    print(f"Looking for images in: {images_dir}")

    all_images = find_images(images_dir)
    if not all_images:
        print("No images found in data/images. Nothing to do.")
        return

    train_list = read_split(train_path)
    test_list = read_split(test_path)
    val_list = read_split(val_path)

    existing = set(train_list) | set(test_list) | set(val_list)
    new_images = [im for im in all_images if im not in existing]

    print(f"Found {len(all_images)} images total; {len(existing)} already in splits; {len(new_images)} new images to assign.")

    # If user requested a specific train/test ratio, recompute the two splits
    if args.train_ratio is not None:
        from random import Random

        images_for_split = [im for im in all_images if im not in val_list]
        rng = Random(args.seed)
        rng.shuffle(images_for_split)

        n_total = len(images_for_split)
        n_train = int(round(n_total * float(args.train_ratio)))
        # guard bounds
        n_train = max(0, min(n_total, n_train))

        train_list = images_for_split[:n_train]
        test_list = images_for_split[n_train:]
        print(f"Recomputed splits using train_ratio={args.train_ratio} (seed={args.seed}): train={len(train_list)}, test={len(test_list)}, val={len(val_list)}")
    else:
        # Assign new images to whichever split is currently smaller to keep balance
        for im in new_images:
            if len(train_list) <= len(test_list):
                train_list.append(im)
            else:
                test_list.append(im)

        # Optionally, you could shuffle before assignment; we keep deterministic ordering

    # Write updated splits back
    write_split(train_path, train_list)
    write_split(test_path, test_list)

    print(f"Updated splits:")
    print(f"  train: {len(train_list)} entries (saved to {train_path})")
    print(f"  test:  {len(test_list)} entries (saved to {test_path})")
    if val_list:
        print(f"  val:   {len(val_list)} entries (left unchanged)")

    if args.no_train:
        print("Done; not starting training (--no-train specified).")
        return

    # Build training command
    train_script = workspace_root / args.train_script
    if not train_script.exists():
        print(f"Training script not found at {train_script}. Aborting training start.")
        return

    cmd = [sys.executable, str(train_script), '--data_dir', str(data_dir)]
    if args.epochs is not None:
        cmd += ['--epochs', str(args.epochs)]
    if args.train_args:
        # naive split of extra args; users can pass quoted args
        cmd += args.train_args.split()

    print(f"Starting training with command: {' '.join(cmd)}")

    # Start training in a background process so the script returns control
    # On Windows PowerShell, subprocess.Popen is fine. We'll detach so user can close the terminal.
    try:
        # Use creationflags to detach on Windows
        if os.name == 'nt':
            DETACHED = 0x00000008
            proc = subprocess.Popen(cmd, creationflags=DETACHED)
        else:
            proc = subprocess.Popen(cmd)

        print(f"Training started (PID: {proc.pid}). Check the logs directory for tensorboard logs and models/ for checkpoints.")
    except Exception as e:
        print(f"Failed to start training: {e}")


if __name__ == '__main__':
    main()
