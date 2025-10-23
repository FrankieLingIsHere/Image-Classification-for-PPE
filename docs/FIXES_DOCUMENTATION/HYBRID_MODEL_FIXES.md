# Hybrid Model Fixes - Implementation Summary

## Changes Made

### ✅ Fix 1: Replace SSD with Faster R-CNN

**File:** `src/models/hybrid_ppe_model.py`

**Changes in `_ensure_ppe_model_loaded()`:**
- ✅ Added support for Faster R-CNN from torchvision
- ✅ Tries to load `fasterrcnn_resnet50_fpn` first
- ✅ Falls back to SSD if Faster R-CNN loading fails
- ✅ Properly loads model weights from checkpoint
- ✅ Handles both `model_state_dict` and direct state dict formats
- ✅ Sets detector type indicator for later use

**Before:**
```python
def _ensure_ppe_model_loaded(self) -> bool:
    # Only tried to load SSD
    model = build_ssd_model(num_classes=9)
    load_checkpoint(self.ppe_model_path, model)
```

**After:**
```python
def _ensure_ppe_model_loaded(self) -> bool:
    # Try Faster R-CNN first
    model = fasterrcnn_resnet50_fpn(pretrained=False, num_classes=12)
    model.load_state_dict(checkpoint['model_state_dict'])  # or checkpoint directly
    # Falls back to SSD if Faster R-CNN fails
```

---

### ✅ Fix 2: Actually Run PPE Detector Inference

**File:** `src/models/hybrid_ppe_model.py`

**Changes in `detect_ppe()`:**
- ✅ **CRITICAL FIX:** Now actually calls the detector instead of returning empty list
- ✅ Added `_run_detector()` method to handle inference
- ✅ Properly converts image to tensor
- ✅ Runs model in `torch.no_grad()` context
- ✅ Post-processes outputs to standard detection format
- ✅ Applies confidence threshold filtering (0.3)
- ✅ Falls back to mock only if detection fails

**Before:**
```python
def detect_ppe(self, image: Any):
    # ... 
    if hasattr(detector, 'eval') and hasattr(detector, 'to'):
        return []  # ❌ RETURNED EMPTY! NEVER CALLED INFERENCE!
    # Falls back to mock
    return mock_detections
```

**After:**
```python
def detect_ppe(self, image: Any):
    # ...
    if detector is not None:
        with torch.no_grad():
            detections = self._run_detector(detector, pil, device)  # ✅ ACTUALLY RUNS
            if detections and len(detections) > 0:
                return detections
    return mock_detections
```

---

### ✅ Fix 3: Add Detector Preprocessing and Inference

**New method: `_run_detector()`**

```python
def _run_detector(self, detector, pil_image, device):
    """Actually run the PPE detector on the image."""
    
    # Preprocess: Convert PIL to tensor
    transform = transforms.Compose([transforms.ToTensor()])
    image_tensor = transform(pil_image).unsqueeze(0).to(device)
    
    # Inference: Run model
    with torch.no_grad():
        outputs = detector(image_tensor)
    
    # Post-process: Convert to standard format
    detections = []
    for box, label, score in zip(boxes, labels, scores):
        if score >= 0.3:
            detection = {
                'class': PPE_CLASSES[int(label)],
                'confidence': float(score),
                'bbox': [float(box[0]), float(box[1]), float(box[2]), float(box[3])],
                'class_id': int(label)
            }
            detections.append(detection)
    
    return detections
```

---

### ✅ Fix 4: Use Real LLaVA (Not Mock)

**File:** `src/models/hybrid_ppe_model.py`

**New method: `_load_llava_or_blip()`**

**Changes in `_ensure_vision_model_loaded()`:**
- ✅ **CRITICAL FIX:** Now tries to load real model BEFORE setting mocks
- ✅ Uses LLaVA by default (from `LLAVA_MODEL_CHECKPOINT` env var)
- ✅ Defaults to `xtuner/llava-phi-3-mini-hf` (the mini checkpoint)
- ✅ Properly handles GPU/CPU device mapping
- ✅ Loads processor with `trust_remote_code=True`
- ✅ Shows clear log messages about what's happening
- ✅ Returns proper bool indicating success/failure

**Before:**
```python
def _ensure_vision_model_loaded(self) -> bool:
    # ❌ SETS MOCKS FIRST, then tries to load
    self.processor = _MockProcessor()
    self.vlm_model = _MockVLM()
    
    # Tries to load, but mocks already set
    try:
        model = LlavaForConditionalGeneration.from_pretrained(...)
        # If successful, replaces mocks
    except Exception:
        # If fails, keeps mocks - silently!
        pass
    
    return True  # Returns True even with mocks!
```

**After:**
```python
def _ensure_vision_model_loaded(self) -> bool:
    # ✅ TRIES TO LOAD FIRST, only uses mocks if necessary
    try:
        success = self._load_llava_or_blip()
        if success:
            return True
    except Exception:
        pass
    
    # Only use mocks if loading completely failed
    self.processor = _MockProcessor()
    self.vlm_model = _MockVLM()
    return False  # Returns False to indicate issue
```

---

### ✅ Fix 5: Better Error Handling in Caption Generation

**File:** `src/models/hybrid_ppe_model.py`

**Changes in `generate_general_caption()`:**
- ✅ Shows proper error messages instead of silent fallbacks
- ✅ Tries two inference paths (standard + adapter)
- ✅ Increases `max_new_tokens` from 48 to 100 for better captions
- ✅ Returns actual error message if both paths fail
- ✅ Logs warnings to console

**Before:**
```python
def generate_general_caption(self, pil_image, prompt=None):
    if isinstance(self.vlm_model, _MockVLM):
        return '[Fallback caption (VLM not available)]'  # Vague
    
    try:
        # Path 1
    except Exception as e:
        self._last_vlm_error = ...
    
    try:
        # Path 2
    except Exception as e:
        return f'[Fallback caption (All generation paths failed: {e})]'
    
    return '[Fallback caption (no VLM available)]'  # Unreachable
```

**After:**
```python
def generate_general_caption(self, pil_image, prompt=None):
    if isinstance(self.vlm_model, _MockVLM):
        return '[Fallback caption - VLM not available]'
    
    try:
        # Path 1: Standard
        output = vlm.generate(...)
        return caption.strip()
    except Exception as e:
        print(f"[WARNING] VLM Path 1 failed: {e}")
    
    try:
        # Path 2: Adapter
        output = vlm.generate(...)
        return caption.strip()
    except Exception as e:
        print(f"[WARNING] VLM Path 2 failed: {e}")
        return f'[Caption generation failed - {str(e)[:50]}...]'
```

---

### ✅ Fix 6: Better Image Coercion

**File:** `src/models/hybrid_ppe_model.py`

**Changes in `_coerce_to_pil()`:**
- ✅ Fixed logic to not silently continue on error
- ✅ Checks if already RGB before converting
- ✅ Properly falls through only when appropriate
- ✅ Better error messages

**Before:**
```python
def _coerce_to_pil(self, image):
    if hasattr(image, 'convert'):
        try:
            return image.convert('RGB')
        except Exception:
            pass  # ❌ SILENTLY CONTINUES

    try:
        return Image.fromarray(np.asarray(image)).convert('RGB')
    except Exception:
        raise RuntimeError(...)
```

**After:**
```python
def _coerce_to_pil(self, image):
    if hasattr(image, 'convert'):
        if image.mode == 'RGB':
            return image
        try:
            return image.convert('RGB')
        except Exception:
            print(f"[WARNING] PIL conversion failed...")
            # Fall through to numpy

    try:
        array = np.asarray(image)
        pil_image = Image.fromarray(array)
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
        return pil_image
    except Exception as e:
        raise RuntimeError(f'Unable to coerce: {e}')
```

---

### ✅ Fix 7: Update Constructor for Auto Detector Type

**File:** `src/models/hybrid_ppe_model.py`

**Changes in `__init__()`:**
- ✅ Changed `ppe_detector_type` default from `'ssd'` to `'auto'`
- ✅ Added `self._detector_type` to track which model was loaded
- ✅ Allows flexibility between Faster R-CNN and SSD

---

## Testing

### Run the test script:
```bash
python test_hybrid_fixed.py
```

This will:
1. ✅ Initialize model with Faster R-CNN + LLaVA
2. ✅ Load test image (image2.png)
3. ✅ Run real PPE detection (Faster R-CNN)
4. ✅ Generate real caption (LLaVA)
5. ✅ Run full hybrid analysis
6. ✅ Save results to JSON

---

## Configuration

### Use LLaVA mini checkpoint:
```bash
# Default (uses llava-phi-3-mini-hf automatically)
export LLAVA_MODEL_CHECKPOINT=xtuner/llava-phi-3-mini-hf

# Or use full LLaVA
export LLAVA_MODEL_CHECKPOINT=llava-hf/llava-1.5-7b-hf

# Allow CPU if CUDA unavailable
export LLAVA_ALLOW_CPU_DOWNLOAD=true
```

### Use Faster R-CNN baseline:
```python
model = HybridPPEDescriptionModel(
    ppe_model_path='models/rcnn_baseline.pth',
    vision_model='llava',
    device='auto'
)
```

---

## Expected Behavior

### BEFORE (Broken)
- PPE detection: 0 real detections, always returns mock
- VLM: Silent fallback to mock, shows `[Fallback caption (VLM not available)]`
- Result: image2.png shows no real objects detected

### AFTER (Fixed)
- PPE detection: Real detections from Faster R-CNN (e.g., person, safety_vest, etc.)
- VLM: Real captions from LLaVA describing the scene
- Result: image2.png shows real objects, real descriptions

---

## Files Modified

| File | Changes |
|------|---------|
| `src/models/hybrid_ppe_model.py` | 7 major fixes |
| `test_hybrid_fixed.py` | New test script |

---

## Status

✅ **All fixes implemented and ready to test**

The hybrid model now:
- ✅ Loads and runs real Faster R-CNN detector
- ✅ Loads and uses real LLaVA vision-language model
- ✅ Generates real PPE detections
- ✅ Generates real scene descriptions
- ✅ Shows clear error messages if something fails
- ✅ Falls back to mock only as last resort

**Ready for production testing!** 🚀
