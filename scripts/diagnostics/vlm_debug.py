import sys, os, traceback
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))

from PIL import Image

try:
    from src.models.hybrid_ppe_model import HybridPPEDescriptionModel
except Exception as e:
    print('Failed to import HybridPPEDescriptionModel:', e)
    traceback.print_exc()
    raise

# Try candidate image paths
candidates = [
    'data/images/image7.jpg',
    'data/images/image7.jpeg',
    'data/images/image7.png',
    'image7.jpg',
    str(Path.home() / 'Downloads' / 'image7.jpg')
]
img_path = None
for p in candidates:
    if os.path.exists(p):
        img_path = p
        break

if img_path is None:
    print('No candidate image found. Tried:', candidates)
    raise SystemExit(1)

print('Using image:', img_path)
img = Image.open(img_path).convert('RGB')

# instantiate model (lazy-loads heavy components when needed)
model = HybridPPEDescriptionModel(vision_model='blip2', device='cpu')
# Ensure vision model components are loaded (may download weights)
print('Ensuring vision-language model is loaded (this may take a moment)...')
loaded_vlm = False
try:
    loaded_vlm = model._ensure_vision_model_loaded()
    print('VLM loaded:', loaded_vlm)
except Exception as e:
    print('VLM lazy-load failed:', e)

# coerce
pil = model._coerce_to_pil(img)
print('Image size:', pil.size)

# Build prompt and inputs similar to generate_general_caption
prompt_simple = "Describe what is shown in this image in one natural sentence."
try:
    inputs = model.processor(pil, text=prompt_simple, return_tensors='pt')
except Exception:
    inputs = model.processor(pil, return_tensors='pt')

# move to device if tensors
try:
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
except Exception:
    pass

gen_inputs = {k: v for k, v in inputs.items() if hasattr(v, 'dtype')}
print('gen_inputs keys:', list(gen_inputs.keys()))

# One generation attempt with verbose inspection
try:
    gen_kwargs = dict(model.blip_gen_kwargs)
    print('Generation kwargs:', gen_kwargs)
    with __import__('torch').no_grad():
        out = model.vlm_model.generate(**gen_inputs, **gen_kwargs)
    print('Type(out):', type(out))
    try:
        first = out[0]
        print('Type(out[0]):', type(first))
        # if tensor
        import torch
        if isinstance(first, torch.Tensor):
            print('out[0] tensor shape:', first.shape)
            # show first 20 token ids
            ids = first.cpu().numpy().tolist()
            print('first token ids (first 50):', ids[:50])
        else:
            print('out[0] repr (truncated):', repr(first)[:400])
    except Exception as e:
        print('Error inspecting out[0]:', e)

    # Try decode via processor
    raw_caption = None
    try:
        raw_caption = model.processor.decode(out[0], skip_special_tokens=True)
        print('\nprocessor.decode ->', repr(raw_caption))
    except Exception as e:
        print('processor.decode failed:', e)

    # Try tokenizer fallback
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(model.processor.tokenizer.name_or_path)
        candidate = out[0]
        if hasattr(candidate, 'tolist'):
            ids = candidate.tolist()
        else:
            ids = candidate
        tok_decoded = tok.decode(ids, skip_special_tokens=True)
        print('\ntokenizer.decode ->', repr(tok_decoded))
    except Exception as e:
        print('tokenizer fallback failed:', e)

    # Postprocess first sentence
    caption = (raw_caption or tok_decoded or str(out))
    import re
    cap = re.sub(r'(?i)^(description:|caption:|answer:|response:)\s*', '', caption).strip()
    m = re.search(r'([A-Z][^.!?]*[.!?])', cap)
    if m:
        sentence = m.group(1).strip()
    else:
        sentence = cap.splitlines()[0].strip()
    print('\nExtracted sentence ->', repr(sentence))
    print('\nCleaned via _clean_description ->', repr(model._clean_description(sentence, prompt_simple)))

except Exception as e:
    print('Generation attempt failed:', e)
    traceback.print_exc()
    raise

print('\nNow calling generate_general_caption() to observe its behavior:')
try:
    gc = model.generate_general_caption(img)
    print('generate_general_caption ->', repr(gc))
except Exception as e:
    print('generate_general_caption failed:', e)
    traceback.print_exc()

print('\nDone')
