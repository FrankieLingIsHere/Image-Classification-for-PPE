import os
import sys
from pathlib import Path
from PIL import Image
from pprint import pprint

# ensure repo root is on sys.path so `from src...` imports work
sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.models.hybrid_ppe_model import HybridPPEDescriptionModel

IMG = 'data/images/image7.jpg'
if not os.path.exists(IMG):
    raise SystemExit(f'Image not found: {IMG}')

print('Loading model (will lazy-load components if needed)...')
model = HybridPPEDescriptionModel(vision_model='blip', device='cpu')

print('Coercing and loading image...')
img = Image.open(IMG).convert('RGB')

print('\nRunning PPE detection (may use mock detections if no checkpoint found)...')
detections = model.detect_ppe(img)
print('Detections:')
pprint(detections)

# --- Post-process detections: confidence thresholding + class-wise NMS to reduce noise
conf_threshold = 0.35
print(f"\nFiltering detections with confidence >= {conf_threshold}...")
filtered = [d for d in detections if float(d.get('confidence', 0.0)) >= conf_threshold]
print(f"Detections before filter: {len(detections)}, after confidence filter: {len(filtered)}")

# Apply NMS to reduce duplicates (model._apply_nms groups by class)
nmsed = model._apply_nms(filtered, iou_threshold=0.3)
print(f"Detections after NMS: {len(nmsed)}")
detections = nmsed
print('Filtered detections:')
pprint(detections)

print('\nGenerating PPE-focused descriptions from detections...')
ppe_desc = model.generate_ppe_focused_description(detections)
print('PPE descriptions:')
pprint(ppe_desc)

print('\nGenerating general caption (deterministic beam search)...')
general = model.generate_general_caption(img, deterministic=True)
print('General caption:')
print(general)

print('\nComposing CSV-style caption from detections + general caption...')
csv_paragraph = HybridPPEDescriptionModel.generate_csv_style_caption(ppe_desc, detections, general)
print('\nCSV-style paragraph:\n')
print(csv_paragraph)

print('\nDone.')
