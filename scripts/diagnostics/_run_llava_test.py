from PIL import Image
from src.models.hybrid_ppe_model import HybridPPEDescriptionModel
import os

model = HybridPPEDescriptionModel(vision_model='llava')
model._ensure_vision_model_loaded()

candidates = [
    os.path.join('data','images','image39.jpg'),
    os.path.join('data','images','image64.jpg'),
    os.path.join('data','images','image92.png')
]
for path in candidates:
    if not os.path.exists(path):
        print('SKIP - not found:', path)
        continue
    img = Image.open(path)
    print('\n---', os.path.basename(path))
    res = model.generate_hybrid_description(img, include_general_caption=True)
    print('general_caption:', res.get('general_caption'))
    print('hybrid_description:', (res.get('hybrid_description') or '')[:300])
