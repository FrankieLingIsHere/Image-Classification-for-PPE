#!/usr/bin/env python3
"""
Copy images to a simpler path for Label Studio
"""

import shutil
import os
from pathlib import Path

def create_simple_path():
    """Copy images to C:/data/ppe_images for easier Label Studio access"""
    
    source_dir = Path("data/images")
    dest_dir = Path("C:/data/ppe_images")
    
    # Create destination directory
    dest_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy all images
    copied = 0
    for img_file in source_dir.rglob("image*"):
        if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png']:
            dest_file = dest_dir / img_file.name
            shutil.copy2(img_file, dest_file)
            copied += 1
    
    print(f"✅ Copied {copied} images to {dest_dir}")
    print(f"📁 Use this path in Label Studio: C:/data/ppe_images")
    print(f"🔗 URL path prefix: /data/local-files/")

if __name__ == "__main__":
    create_simple_path()