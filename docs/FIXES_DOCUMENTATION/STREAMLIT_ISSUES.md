# Streamlit App Issues Report

## Critical Issues Found

### 1. **Missing Required Dependencies** ❌
The `requirements.txt` is missing several packages that are imported in `streamlit_app.py`:

- **`streamlit`** - Main framework (imported at line 1)
- **`plotly`** - For interactive charts (imported at lines 35-36)
- **`pandas`** - For DataFrame display (imported at line 37)

**Current requirements.txt** only includes:
- torch, torchvision, numpy, opencv-python, Pillow, matplotlib, tqdm, PyYAML, tensorboard, scikit-learn, transformers, accelerate, torch-audio, sentencepiece, protobuf

**Missing packages to add:**
```
streamlit>=1.28.0
plotly>=5.14.0
pandas>=1.5.0
opencv-python>=4.8.0  # Also ensure cv2 support
```

### 2. **Import Statement Issues** ⚠️

**Line 42-46:** Problematic import pattern for postprocessing functions:
```python
try:
    from scripts.inference import _apply_per_class_nms, _postprocess_persons
except Exception:
    # fallback: import by module path
    try:
        from scripts.inference import _apply_per_class_nms, _postprocess_persons
    except Exception:
        _apply_per_class_nms = None
        _postprocess_persons = None
```

**Issues:**
- The fallback tries the exact same import twice
- Should add the scripts path to `sys.path` BEFORE attempting import
- Current `sys.path.append()` at line 39 happens AFTER the import attempts

**Fix:** Move `sys.path.append()` before the try/except block

### 3. **Potential KeyError at Line 760** 🔴

```python
compliance_status = results['ppe_descriptions']['compliance_status']
```

**Issues:**
- No null/error checking before accessing nested dictionary keys
- If `results.get('ppe_descriptions')` is missing or doesn't have `'compliance_status'`, app will crash
- Should use `.get()` method with fallback values

**Current risk:** The variable `results` may not have the expected structure if the model fails gracefully

### 4. **Inconsistent Normalization** ⚠️

**Lines 713-718:** Detections are normalized twice:
```python
# First normalization (line 713)
raw_dets = normalize_detections(raw_dets, image)

# ... processing ...

# Second normalization (line 745)
filtered_detections = normalize_detections(filtered_detections, image)
```

**Issues:**
- Redundant processing
- Could cause issues if normalization changes data structure
- No validation that normalized data has required fields

### 5. **Missing Error Handling for Model Methods** 🔴

**Lines 755-758:**
```python
try:
    recomputed_desc = model.generate_ppe_focused_description(filtered_detections)
    results['ppe_descriptions'] = recomputed_desc
except Exception as e:
    st.warning(f"Could not recompute PPE descriptions from filtered detections: {e}")
```

**Issues:**
- If this exception occurs, `results['ppe_descriptions']` may be undefined
- Later code (line 760) will crash when trying to access it
- Need to ensure fallback structure

### 6. **Unsafe DataFrame Construction** ⚠️

**Lines 791-797:** 
```python
detection_data = []
for i, det in enumerate(filtered_detections):
    detection_data.append({
        'Item': det['class'],
        'Confidence': f"{det['confidence']:.3f}",
        'Category': 'Person' if det['class'] == 'person' else ('Violation' if det['class'].startswith('no_') else 'PPE'),
        'Bounding Box': f"[{det['bbox'][0]:.0f}, {det['bbox'][1]:.0f}, {det['bbox'][2]:.0f}, {det['bbox'][3]:.0f}]"
    })
```

**Issues:**
- No validation that `det['bbox']` has 4 elements
- Assumes all `det` dictionaries have `'class'`, `'confidence'`, `'bbox'` keys
- No error handling for unexpected detection formats

### 7. **Sidebar Model Path Resolution** ⚠️

**Lines 553-567:**
```python
default_model_path = None
try:
    model_dir = os.path.join(os.path.dirname(__file__), 'models')
    rcnn_candidate = os.path.join(model_dir, 'rcnn_baseline.pth')
    # ...
except Exception:
    default_model_path = 'models/best_model_regularized.pth'
```

**Issues:**
- In Streamlit, `__file__` may not be reliably available depending on deployment
- Silently falls back to hardcoded path without warning user
- No verification that fallback path actually exists

### 8. **Unvalidated User Input** ⚠️

**Line 543:**
```python
thresholds_file = st.sidebar.text_input("Thresholds JSON path", value=thresholds_file_default)
```

**Issues:**
- User can input arbitrary file paths
- No path validation or sanitization
- Could lead to unexpected file access
- Error handling exists but could be more robust

### 9. **Chart Creation Without Data Validation** ⚠️

**Lines 505-527 (`create_detection_chart` function):**
```python
def create_detection_chart(detections):
    if not detections:
        return None
    
    detection_counts = {}
    for det in detections:
        class_name = det['class']  # KeyError if 'class' missing
```

**Issues:**
- Doesn't validate that `det['class']` exists
- Could crash if detection structure is malformed
- No type checking on `detections` parameter

### 10. **Missing `cv2` Import Dependency** ❌

**Line 241** in `draw_bounding_boxes()`:
```python
def draw_bounding_boxes(image, detections):
    """Draw bounding boxes on image"""
    import cv2
    import numpy as np
```

**Issues:**
- `cv2` (OpenCV) is imported inside function but package name in requirements is `opencv-python`
- While the package exists, there could be version compatibility issues
- Should be listed in requirements for clarity

---

## Summary of Required Fixes

### High Priority (Blocks Execution) 🔴
1. Add missing packages to requirements.txt: **streamlit**, **plotly**, **pandas**
2. Fix import order for scripts.inference postprocessing functions
3. Add null/error checking for nested dictionary access

### Medium Priority (Runtime Stability) ⚠️
4. Validate detection data structures before accessing nested fields
5. Add error handling for model method calls
6. Ensure fallback structures exist before dependent code accesses them

### Low Priority (Code Quality) 💡
7. Remove duplicate detection normalization
8. Add path validation for user inputs
9. Improve error messages in sidebar model path resolution
10. Add type hints and docstrings to helper functions

---

## Recommended Changes

### Step 1: Update requirements.txt
Add these lines:
```
streamlit>=1.28.0
plotly>=5.14.0
pandas>=1.5.0
```

### Step 2: Fix imports section (lines 42-50)
```python
# Move sys.path.append BEFORE import attempts
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'scripts'))

# Try to import postprocessing helpers
try:
    from scripts.inference import _apply_per_class_nms, _postprocess_persons
except ImportError:
    st.warning("⚠️ Could not load postprocessing helpers. Advanced filtering disabled.")
    _apply_per_class_nms = None
    _postprocess_persons = None
```

### Step 3: Add safeguards for nested dictionary access
```python
# Instead of:
compliance_status = results['ppe_descriptions']['compliance_status']

# Use:
compliance_status = results.get('ppe_descriptions', {}).get('compliance_status', 'UNKNOWN STATUS')
```

### Step 4: Validate detection structure
Add validation before accessing detection fields:
```python
def safe_get_detection_class(detection, default='unknown'):
    return detection.get('class', detection.get('class_name', default))

def safe_get_detection_confidence(detection, default=0.0):
    try:
        return float(detection.get('confidence', default))
    except (ValueError, TypeError):
        return default

def safe_get_detection_bbox(detection, default=None):
    bbox = detection.get('bbox', default)
    if bbox and len(bbox) == 4:
        return bbox
    return default
```
