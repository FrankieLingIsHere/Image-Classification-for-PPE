import sys, os, traceback
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))
from PIL import Image
from src.models.hybrid_ppe_model import HybridPPEDescriptionModel

# candidate image
img_path = 'data/images/image7.jpg'
if not os.path.exists(img_path):
    raise SystemExit(f'Image not found: {img_path}')
img = Image.open(img_path).convert('RGB')

model = HybridPPEDescriptionModel(vision_model='blip2', device='cpu')
print('Ensuring VLM loaded...')
model._ensure_vision_model_loaded()
print('VLM ready')

pil = model._coerce_to_pil(img)
print('Image size:', pil.size)

# prepare inputs (image-only)
try:
    inputs = model.processor(images=pil, return_tensors='pt')
except Exception:
    inputs = model.processor(pil, return_tensors='pt')

try:
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
except Exception:
    pass

gen_inputs = {k: v for k, v in inputs.items() if hasattr(v, 'dtype')}
print('gen_inputs keys:', list(gen_inputs.keys()))

# deterministic gen kwargs
gen_kwargs = {
    'max_new_tokens': 80,
    'do_sample': False,
    'num_beams': 2,
    'num_return_sequences': 1,
    'use_cache': True
}
print('Generation kwargs:', gen_kwargs)

import torch, re
with torch.no_grad():
    out = model.vlm_model.generate(**gen_inputs, **gen_kwargs)

print('Type(out):', type(out))
first = out[0]
print('Type(out[0]):', type(first))
if hasattr(first, 'shape'):
    print('out[0] shape:', first.shape)

# try to slice off input ids if present
raw_caption = None
try:
    if hasattr(first, 'cpu'):
        seq = first.cpu()
        input_len = 0
        if 'input_ids' in gen_inputs:
            try:
                input_len = int(gen_inputs['input_ids'].shape[-1])
            except Exception:
                input_len = 0
        gen_seq = seq[input_len:] if seq.shape[-1] > input_len else seq
        try:
            raw_caption = model.processor.decode(gen_seq, skip_special_tokens=True)
        except Exception:
            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained(model.processor.tokenizer.name_or_path)
            ids = gen_seq.tolist() if hasattr(gen_seq, 'tolist') else gen_seq
            raw_caption = tok.decode(ids, skip_special_tokens=True)
    else:
        raw_caption = str(first)
except Exception as e:
    raw_caption = str(out)

print('\nRaw (decoded generated tail):', repr(raw_caption))
# extract first sentence
cap = (raw_caption or '').strip()
m = re.search(r'([A-Z][^.!?]*[.!?])', cap)
if m:
    sentence = m.group(1).strip()
else:
    sentence = cap.splitlines()[0].strip() if cap else ''

print('Extracted sentence:', repr(sentence))
print('Cleaned:', repr(model._clean_description(sentence, 'Describe what is shown in this image in one natural sentence.')))

# Also call the public pipeline to show final value
print('\nPublic API generate_general_caption ->', repr(model.generate_general_caption(img, deterministic=True)))
