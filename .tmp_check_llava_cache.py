import os, traceback
ckpt='xtuner/llava-phi-3-mini-hf'
print('CKPT=',ckpt)
print('ENV HF_HOME=', os.environ.get('HF_HOME'))
print('ENV TRANSFORMERS_CACHE=', os.environ.get('TRANSFORMERS_CACHE'))
print('ENV HUGGINGFACE_HUB_CACHE=', os.environ.get('HUGGINGFACE_HUB_CACHE'))

try:
    from transformers import AutoProcessor
    try:
        p = AutoProcessor.from_pretrained(ckpt, local_files_only=True)
        print('AutoProcessor local OK:', type(p))
    except Exception as e:
        print('AutoProcessor local FAILED ->', repr(e))
        traceback.print_exc()
except Exception as e:
    print('Cannot import AutoProcessor:', repr(e))
    traceback.print_exc()

try:
    from transformers import LlavaForConditionalGeneration
    try:
        m = LlavaForConditionalGeneration.from_pretrained(ckpt, local_files_only=True)
        print('LlavaForConditionalGeneration local OK:', type(m))
    except Exception as e:
        print('LlavaForConditionalGeneration local FAILED ->', repr(e))
        traceback.print_exc()
except Exception as e:
    print('Cannot import LlavaForConditionalGeneration:', repr(e))
    traceback.print_exc()

try:
    from huggingface_hub import snapshot_download
    try:
        d = snapshot_download(repo_id=ckpt, local_files_only=True)
        print('snapshot_download local dir:', d)
        try:
            print('files in snapshot (sample):', os.listdir(d)[:50])
        except Exception as e:
            print('listing snapshot dir failed:', e)
    except Exception as e:
        print('snapshot_download local FAILED ->', repr(e))
        traceback.print_exc()
except Exception as e:
    print('Cannot import huggingface_hub.snapshot_download:', repr(e))
    traceback.print_exc()
