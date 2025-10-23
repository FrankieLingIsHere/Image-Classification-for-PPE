"""Small diagnostic to exercise the LLaVA processor + generate path and print full exceptions.

Usage: run with the project venv: python scripts/diagnostics/debug_llava_generate.py

This script will:
- instantiate HybridPPEDescriptionModel(vision_model='llava')
- call _ensure_vision_model_loaded()
- load a sample image from data/ (or user-specified)
- call processor(text=[prompt], images=pil_image, return_tensors='pt')
- move tensors to model device and call vlm_model.generate(...)
- print any exception and traceback so we can debug why LLaVA generation fails
"""

import traceback
import os
import sys
from PIL import Image

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
# adjust path so imports resolve when script run from project root
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.models.hybrid_ppe_model import HybridPPEDescriptionModel


def main():
    img_path = os.path.join(REPO_ROOT, 'data', 'images')
    # find a sample image
    sample = None
    if os.path.isdir(img_path):
        for fn in os.listdir(img_path):
            if fn.lower().endswith(('.jpg', '.jpeg', '.png')):
                sample = os.path.join(img_path, fn)
                break
    if sample is None:
        print('No sample image found under data/images; please provide one.')
        return

    print('Using sample image:', sample)

    model = HybridPPEDescriptionModel(vision_model='llava')
    try:
        ok = model._ensure_vision_model_loaded()
        print('_ensure_vision_model_loaded ->', ok)
    except Exception as e:
        print('Exception during _ensure_vision_model_loaded:')
        traceback.print_exc()
        return

    # Print diagnostics recorded by the loader (if any)
    print('processor type:', type(model.processor))
    print('vlm_model type:', type(model.vlm_model), 'has_generate=', hasattr(model.vlm_model, 'generate'))
    print('_last_vlm_error:', getattr(model, '_last_vlm_error', None))
    print('vision_fallback_msg:', getattr(model, 'vision_fallback_msg', None))
    try:
        print('processor repr:', repr(model.processor))
    except Exception:
        pass
    try:
        print('vlm_model repr:', repr(model.vlm_model))
    except Exception:
        pass

    try:
        pil = Image.open(sample).convert('RGB')
    except Exception as e:
        print('Failed to open sample image:', e)
        traceback.print_exc()
        return

    prompt = 'Describe this image focusing on the workers and PPE in one concise sentence.'
    try:
        text_prompts = [prompt]
        print('Calling processor...')
        inputs = None
        try:
            inputs = model.processor(text=text_prompts, images=pil, return_tensors='pt')
        except Exception as e:
            print('processor(text=... , images=...) raised; trying alternate call signature...')
            traceback.print_exc()
            try:
                inputs = model.processor(images=pil, return_tensors='pt')
            except Exception:
                print('processor(images=...) also failed; rethrowing')
                traceback.print_exc()
                raise

        print('processor returned keys:', list(inputs.keys()))

        # Move to device
        try:
            inputs = {k: v.to(model.device) for k, v in inputs.items() if hasattr(v, 'to')}
            print('Inputs moved to device')
        except Exception:
            print('Failed to move inputs to device; continuing with original tensors')

        gen_kwargs = {'max_new_tokens': 48, 'do_sample': True, 'temperature': 0.7, 'top_p': 0.9, 'num_return_sequences': 1}
        print('Calling vlm_model.generate...')
        try:
            with model.vlm_model.device:
                pass
        except Exception:
            pass

        try:
            out = model.vlm_model.generate(**{k: v for k, v in inputs.items() if hasattr(v, 'dtype')}, **gen_kwargs)
            print('generate returned type:', type(out))
            print('generate output (preview):', str(out)[:500])
        except Exception as ge:
            print('Exception during vlm_model.generate:')
            traceback.print_exc()
    except Exception:
        print('Unexpected failure in diagnostic run:')
        traceback.print_exc()


if __name__ == '__main__':
    main()
