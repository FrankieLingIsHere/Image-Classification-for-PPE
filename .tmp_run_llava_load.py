import os
os.environ['LLAVA_ALLOW_CPU_DOWNLOAD'] = '1'
# Ensure local project modules are importable
import sys
sys.path.insert(0, os.getcwd())

from src.models.hybrid_ppe_model import HybridPPEDescriptionModel
from PIL import Image

print('LLAVA_ALLOW_CPU_DOWNLOAD=', os.environ.get('LLAVA_ALLOW_CPU_DOWNLOAD'))
model = HybridPPEDescriptionModel(vision_model='llava')
print('Created Hybrid model; calling _ensure_vision_model_loaded()...')
ok = model._ensure_vision_model_loaded()
print('Loaded OK:', ok)
print('processor:', type(model.processor))
print('vlm_model:', type(model.vlm_model))
print('has_generate:', hasattr(model.vlm_model, 'generate'))

# Try to generate a caption for a sample image if available
sample = os.path.join('data','images','image39.jpg')
if os.path.exists(sample):
    img = Image.open(sample)
    print('\nGenerating caption (may be BLIP if LLaVA unusable):')
    cap = model.generate_general_caption(img)
    print('caption:', cap)
else:
    print('Sample image not found at', sample)
