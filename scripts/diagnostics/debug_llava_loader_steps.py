"""Step-by-step diagnostic for LLaVA loader.

This script attempts each loading step the `_ensure_vision_model_loaded()` function would do,
but prints detailed exceptions for each stage so we can see why `processor` / `vlm_model`
end up as None.
"""
import os
import traceback
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

ckpt = os.environ.get('LLAVA_MODEL_CHECKPOINT', 'xtuner/llava-phi-3-mini-hf')
print('Using checkpoint:', ckpt)

# 1) Try AutoProcessor.from_pretrained(local_files_only=True)
try:
    from transformers import AutoProcessor
    print('\n1) Trying AutoProcessor.from_pretrained(local_files_only=True)')
    try:
        proc = AutoProcessor.from_pretrained(ckpt, local_files_only=True)
        print('  -> AutoProcessor loaded (local_files_only=True):', type(proc))
    except Exception as e:
        print('  -> AutoProcessor local failed:')
        traceback.print_exc()
        proc = None
except Exception:
    print('  -> transformers AutoProcessor import failed:')
    traceback.print_exc()
    proc = None

# 2) Try AutoProcessor.from_pretrained() (network)
if proc is None:
    try:
        print('\n2) Trying AutoProcessor.from_pretrained() [network]')
        proc = AutoProcessor.from_pretrained(ckpt)
        print('  -> AutoProcessor loaded (network):', type(proc))
    except Exception:
        print('  -> AutoProcessor network load failed:')
        traceback.print_exc()
        proc = None

# 3) Try AutoFeatureExtractor + AutoTokenizer local-only
try:
    from transformers import AutoFeatureExtractor, AutoTokenizer
    print('\n3) Trying AutoFeatureExtractor.from_pretrained(local_files_only=True) and AutoTokenizer')
    feat = None
    tok = None
    try:
        feat = AutoFeatureExtractor.from_pretrained(ckpt, local_files_only=True)
        print('  -> AutoFeatureExtractor loaded (local):', type(feat))
    except Exception:
        print('  -> AutoFeatureExtractor local failed:')
        traceback.print_exc()
    try:
        tok = AutoTokenizer.from_pretrained(ckpt, local_files_only=True)
        print('  -> AutoTokenizer loaded (local):', type(tok))
    except Exception:
        print('  -> AutoTokenizer local failed:')
        traceback.print_exc()
except Exception:
    print('  -> AutoFeatureExtractor/AutoTokenizer import failed:')
    traceback.print_exc()
    feat = None
    tok = None

# 4) Try AutoFeatureExtractor + AutoTokenizer (network)
if feat is None:
    try:
        print('\n4) Trying AutoFeatureExtractor.from_pretrained() (network)')
        feat = AutoFeatureExtractor.from_pretrained(ckpt)
        print('  -> AutoFeatureExtractor loaded (network):', type(feat))
    except Exception:
        print('  -> AutoFeatureExtractor network load failed:')
        traceback.print_exc()
        feat = None

if tok is None:
    try:
        print('\n5) Trying AutoTokenizer.from_pretrained() (network)')
        tok = AutoTokenizer.from_pretrained(ckpt)
        print('  -> AutoTokenizer loaded (network):', type(tok))
    except Exception:
        print('  -> AutoTokenizer network load failed:')
        traceback.print_exc()
        tok = None

# 6) If feature extractor exists, try calling it on a small dummy image
if feat is not None:
    try:
        from PIL import Image
        import numpy as _np
        print('\n6) Trying to call feature extractor on dummy image')
        dummy = Image.new('RGB', (64, 64), color=(128, 128, 128))
        out = feat(images=dummy, return_tensors='pt')
        print('  -> Feature extractor returned keys:', list(out.keys()))
    except Exception:
        print('  -> Feature extractor call failed:')
        traceback.print_exc()

# 7) Try to load LlavaForConditionalGeneration with CPU-safe args
try:
    print('\n7) Trying LlavaForConditionalGeneration.from_pretrained(low_cpu_mem_usage=True, torch_dtype=float32, device_map=None)')
    from transformers import LlavaForConditionalGeneration
    try:
        m_cpu = LlavaForConditionalGeneration.from_pretrained(ckpt, low_cpu_mem_usage=True, torch_dtype='float32', device_map=None)
        print('  -> Llava model loaded (cpu args):', type(m_cpu), 'has_generate=', hasattr(m_cpu, 'generate'))
    except Exception:
        print('  -> Llava cpu-path load failed:')
        traceback.print_exc()
        m_cpu = None
except Exception:
    print('  -> LlavaForConditionalGeneration import failed:')
    traceback.print_exc()
    m_cpu = None

# 8) Try to load LlavaForConditionalGeneration with device_map='auto' (if possible)
try:
    print('\n8) Trying LlavaForConditionalGeneration.from_pretrained(device_map="auto", torch_dtype=float16, low_cpu_mem_usage=True)')
    try:
        m_auto = LlavaForConditionalGeneration.from_pretrained(ckpt, device_map='auto', torch_dtype='float16', low_cpu_mem_usage=True)
        print('  -> Llava model loaded (auto):', type(m_auto), 'has_generate=', hasattr(m_auto, 'generate'))
    except Exception:
        print('  -> Llava auto-path load failed:')
        traceback.print_exc()
        m_auto = None
except Exception:
    print('  -> LlavaForConditionalGeneration import failed earlier; skipping auto load')
    m_auto = None

print('\nSummary:')
print('  AutoProcessor (proc):', type(proc) if proc is not None else None)
print('  FeatureExtractor (feat):', type(feat) if feat is not None else None)
print('  Tokenizer (tok):', type(tok) if tok is not None else None)
print('  Llava cpu model (m_cpu):', type(m_cpu) if m_cpu is not None else None)
print('  Llava auto model (m_auto):', type(m_auto) if m_auto is not None else None)

print('\nDone.')
