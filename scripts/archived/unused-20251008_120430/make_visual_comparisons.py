"""Run inference on a small set of images with a chosen config and save side-by-side raw vs filtered images.

Usage: python scripts/make_visual_comparisons.py
"""
import os, subprocess, sys
from pathlib import Path
ROOT = Path(__file__).parent.parent
OUT = ROOT / 'outputs' / 'visual_comparisons'
OUT.mkdir(parents=True, exist_ok=True)

images = ['image82.png','image75.jpg','image61.jpg','image59.png','image118.jpg']
config = ROOT / 'outputs' / 'balanced_config.yaml'
model = ROOT / 'models' / 'best_model_regularized.pth'

for img in images:
    img_path = ROOT / 'data' / 'images' / img
    out_dir = OUT / Path(img).stem
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, 'scripts/inference.py',
        '--model_path', str(model),
        '--input', str(img_path),
        '--output_dir', str(out_dir),
        '--save_images', '--save_json',
        '--config_path', str(config)
    ]
    print('Running inference for', img)
    subprocess.run(cmd, check=True)

    # copy saved raw and filtered images to a single comparison filename if present
    raw = out_dir / f"{Path(img).stem}_raw_detection.jpg"
    filt = out_dir / f"{Path(img).stem}_detection.jpg"
    cmp_path = OUT / f"{Path(img).stem}_comparison.jpg"
    # prefer raw + filtered; if not present, try other naming
    if raw.exists() and filt.exists():
        from PIL import Image
        a = Image.open(str(raw)).convert('RGB')
        b = Image.open(str(filt)).convert('RGB')
        # make them same height
        h = min(a.height, b.height)
        a = a.resize((int(a.width * h / a.height), h))
        b = b.resize((int(b.width * h / b.height), h))
        new = Image.new('RGB', (a.width + b.width, h))
        new.paste(a, (0,0))
        new.paste(b, (a.width,0))
        new.save(str(cmp_path))
        print('Saved comparison', cmp_path)
    else:
        print('Missing raw or filtered images for', img)

print('Done')
