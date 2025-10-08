import streamlit as st
import torch
from PIL import Image
import json
import numpy as np
import io
import os
import warnings
import logging

# Suppress deprecation and future warnings (keep UI clean). We still allow
# critical errors to surface but hide noisy deprecation/future notices.
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Suppress specific known noisy messages (e.g. torchvision ToTensor deprecation)
warnings.filterwarnings("ignore", message=r".*ToTensor\(\) is deprecated.*")

# Lower logging verbosity for heavy libraries used by the app so Streamlit UI
# doesn't get flooded with library logs. Errors will still appear.
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("torch").setLevel(logging.ERROR)
logging.getLogger("torchvision").setLevel(logging.ERROR)
logging.getLogger("diffusers").setLevel(logging.ERROR)

# Also set transformers env var and runtime verbosity where available
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
try:
    from transformers import logging as _tf_logging
    _tf_logging.set_verbosity_error()
except Exception:
    pass
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from typing import Optional

# Import your hybrid model
from src.models.hybrid_ppe_model import HybridPPEDescriptionModel
# optional imports for postprocessing
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'scripts'))
try:
    from scripts.inference import _apply_per_class_nms, _postprocess_persons
except Exception:
    # fallback: import by module path
    try:
        from scripts.inference import _apply_per_class_nms, _postprocess_persons
    except Exception:
        _apply_per_class_nms = None
        _postprocess_persons = None

# Page configuration
st.set_page_config(
    page_title="PPE Safety Detection System",
    page_icon="🦺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #FF6B6B;
        text-align: center;
        margin-bottom: 2rem;
    }
    .detection-box {
        border: 2px solid #4ECDC4;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
        background-color: #F8F9FA;
    }
    .compliance-good {
        background-color: #D4EDDA;
        border-color: #C3E6CB;
        color: #155724;
    }
    .compliance-bad {
        background-color: #F8D7DA;
        border-color: #F5C6CB;
        color: #721C24;
    }
    .metric-card {
        background-color: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model(model_path: Optional[str] = None, detector_type: str = 'auto', vision_model: str = 'blip', vision_checkpoint: Optional[str] = None):
    """Load the hybrid PPE model (cached for performance).

    vision_model: one of 'blip2', 'llava', or a short alias. If a
    vision_checkpoint is provided, the appropriate environment variable
    (BLIP2_MODEL_CHECKPOINT or LLAVA_MODEL_CHECKPOINT) will be set so the
    HybridPPEDescriptionModel lazy-loader can find the weights.
    """
    print(f"DEBUG: load_model called with model_path='{model_path}', detector_type='{detector_type}', vision_model='{vision_model}', vision_checkpoint='{vision_checkpoint}'")
    try:
        with st.spinner("🚀 Loading AI models... This may take a moment."):
            # If no model_path supplied, try to auto-detect the latest checkpoint
            if model_path is None or not os.path.exists(model_path):
                # look for .pth / .pt files in models/ sorted by modified time
                def find_latest_model(models_dir='models'):
                    if not os.path.exists(models_dir):
                        return None
                    candidates = []
                    for fn in os.listdir(models_dir):
                        if fn.lower().endswith(('.pth', '.pt')):
                            candidates.append(os.path.join(models_dir, fn))
                    if not candidates:
                        return None
                    # sort by mtime desc
                    candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
                    return candidates[0]

                detected = find_latest_model('models')
                if detected:
                    model_path = detected
                else:
                    model_path = 'models/best_model_regularized.pth'

            # If a vision checkpoint string was supplied, prefer it by setting the
            # corresponding environment variable so the hybrid model's lazy loader
            # will pick it up.
            try:
                if vision_checkpoint and isinstance(vision_checkpoint, str) and vision_checkpoint.strip():
                    vm = (vision_model or '').lower()
                    if 'llava' in vm:
                        os.environ['LLAVA_MODEL_CHECKPOINT'] = vision_checkpoint
                    elif 'blip' in vm or 'blip2' in vm:
                        os.environ['BLIP2_MODEL_CHECKPOINT'] = vision_checkpoint
                    else:
                        # If user provided a checkpoint but didn't choose a model type,
                        # place it in BLIP2 by default (backwards compatible)
                        os.environ['BLIP2_MODEL_CHECKPOINT'] = vision_checkpoint
            except Exception:
                pass

            model = HybridPPEDescriptionModel(
                ppe_model_path=model_path,
                vision_model=(vision_model or "blip"),
                device="auto",
                ppe_detector_type=detector_type
            )
        return model, None
    except Exception as e:
        return None, str(e)


def draw_bounding_boxes(image, detections):
    """Draw bounding boxes on image"""
    import cv2
    import numpy as np
    # Ensure PIL Image is RGB and convert to numpy BGR for OpenCV
    if hasattr(image, 'convert'):
        image = image.convert('RGB')
    img_array = np.array(image)
    # Convert RGB (PIL) to BGR (cv2)
    img_cv2 = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

    # Image dimensions
    img_height, img_width = img_cv2.shape[:2]

    # Determine coordinate space of bboxes and set scaling
    # bbox expected format: [x_min, y_min, x_max, y_max]
    bbox_max = 0
    for det in detections:
        try:
            bbox_max = max(bbox_max, max(det.get('bbox', [0, 0, 0, 0])))
        except Exception:
            continue

    # Heuristics:
    # - If bbox_max <= 1.0 -> normalized coords in [0,1]
    # - If bbox_max <= 300*1.5 -> SSD-style coords in [0,300]
    # - Else -> absolute pixel coords already in image space
    if bbox_max <= 1.001 and bbox_max > 0:
        coord_space = 'normalized'
        scale_x = img_width
        scale_y = img_height
    elif bbox_max <= 450:  # 300 * 1.5 margin
        coord_space = 'ssd_300'
        scale_x = img_width / 300.0
        scale_y = img_height / 300.0
    else:
        coord_space = 'pixels'
        scale_x = 1.0
        scale_y = 1.0
    
    colors = {
        'person': (255, 255, 0),  # Yellow
        'hard_hat': (0, 255, 0),  # Green
        'safety_vest': (0, 255, 0),  # Green
        'safety_gloves': (0, 255, 0),  # Green
        'safety_boots': (0, 255, 0),  # Green
        'eye_protection': (0, 255, 0),  # Green
        'no_hard_hat': (0, 0, 255),  # Red
        'no_safety_vest': (0, 0, 255),  # Red
        'no_safety_gloves': (0, 0, 255),  # Red
        'no_safety_boots': (0, 0, 255),  # Red
        'no_eye_protection': (0, 0, 255),  # Red
    }
    
    for det in detections:
        bbox = det.get('bbox', [0, 0, 0, 0])
        class_name = det.get('class', 'unknown')
        confidence = det.get('confidence', 0.0)

        # Scale coordinates according to detected coordinate space
        try:
            if coord_space == 'normalized':
                x1 = int(bbox[0] * scale_x)
                y1 = int(bbox[1] * scale_y)
                x2 = int(bbox[2] * scale_x)
                y2 = int(bbox[3] * scale_y)
            elif coord_space == 'ssd_300':
                x1 = int(bbox[0] * scale_x)
                y1 = int(bbox[1] * scale_y)
                x2 = int(bbox[2] * scale_x)
                y2 = int(bbox[3] * scale_y)
            else:  # pixels
                x1 = int(bbox[0])
                y1 = int(bbox[1])
                x2 = int(bbox[2])
                y2 = int(bbox[3])
        except Exception:
            # Fallback to zero box if unexpected format
            x1 = y1 = x2 = y2 = 0

        # Clip coordinates to image bounds
        x1 = max(0, min(img_width - 1, x1))
        x2 = max(0, min(img_width - 1, x2))
        y1 = max(0, min(img_height - 1, y1))
        y2 = max(0, min(img_height - 1, y2))

        color = colors.get(class_name, (128, 128, 128))

        # Draw rectangle
        cv2.rectangle(img_cv2, (x1, y1), (x2, y2), color, 2)

        # Draw label (keep inside image)
        label = f"{class_name}: {confidence:.2f}"
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        lx1 = x1
        ly1 = max(0, y1 - 20)
        lx2 = min(img_width - 1, x1 + label_size[0] + 4)
        ly2 = y1
        cv2.rectangle(img_cv2, (lx1, ly1), (lx2, ly2), color, -1)
        cv2.putText(img_cv2, label, (lx1 + 2, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
    
    # Convert back to PIL
    img_rgb = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img_rgb)


def normalize_detections(dets, pil_image):
    """Normalize detection dicts so downstream helpers expect consistent keys.

    Ensures each detection contains 'class' and 'class_name' (strings), a float
    'confidence', and a bbox list (attempts to normalize to fractional coords).
    """
    normed = []
    if dets is None:
        return []
    w, h = pil_image.size
    for d in dets:
        # copy to avoid mutating original caller data structures
        nd = dict(d)
        cls = nd.get('class') or nd.get('class_name') or nd.get('class_id')
        if isinstance(cls, (int, float)):
            cls = str(cls)
        nd['class'] = cls
        nd['class_name'] = cls

        try:
            nd['confidence'] = float(nd.get('confidence', 0.0))
        except Exception:
            nd['confidence'] = 0.0

        bbox = nd.get('bbox', [])
        if bbox and len(bbox) == 4:
            try:
                bbox = [float(x) for x in bbox]
                maxv = max(bbox)
                if maxv > 1.001 and maxv <= 450:
                    # SSD-style 0..300 -> convert to fractional
                    bbox = [bbox[0] / 300.0, bbox[1] / 300.0, bbox[2] / 300.0, bbox[3] / 300.0]
                elif maxv > 450:
                    # pixel coords -> convert to fractional by image size
                    bbox = [bbox[0] / w, bbox[1] / h, bbox[2] / w, bbox[3] / h]
                # else assume fractional already
                nd['bbox'] = bbox
            except Exception:
                nd['bbox'] = bbox
        else:
            nd['bbox'] = bbox

        normed.append(nd)
    return normed


def _bbox_iou(boxA, boxB):
    """Compute IoU for two boxes in fractional format [x1,y1,x2,y2]."""
    # intersection
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interW = max(0.0, xB - xA)
    interH = max(0.0, yB - yA)
    interArea = interW * interH
    # union
    boxAArea = max(0.0, (boxA[2] - boxA[0])) * max(0.0, (boxA[3] - boxA[1]))
    boxBArea = max(0.0, (boxB[2] - boxB[0])) * max(0.0, (boxB[3] - boxB[1]))
    union = boxAArea + boxBArea - interArea
    if union <= 0:
        return 0.0
    return interArea / union


def infer_missing_ppe(detections, required_ppe=None, iou_thresh=0.05):
    """Infer missing PPE (no_*) per detected person by checking overlap with PPE items.

    detections: list of normalized detection dicts (bbox in fractional coords)
    required_ppe: list of PPE class names to check
    Returns a new list with inferred violation detections appended (if enabled).
    """
    if required_ppe is None:
        required_ppe = ['hard_hat', 'safety_vest', 'safety_gloves', 'safety_boots', 'eye_protection']

    people = [d for d in detections if d.get('class') == 'person']
    ppe_items = [d for d in detections if d.get('class') in required_ppe]
    augmented = list(detections)

    for person in people:
        pbox = person.get('bbox', [0,0,1,1])
        for ppe in required_ppe:
            present = False
            for item in ppe_items:
                iou = _bbox_iou(pbox, item.get('bbox', [0,0,0,0]))
                # treat even small overlap as presence (low threshold)
                if iou > iou_thresh:
                    present = True
                    break
            if not present:
                # create a synthetic violation detection attached to the person bbox
                aug = {
                    'class': f'no_{ppe}',
                    'class_name': f'no_{ppe}',
                    'confidence': 0.5,
                    'bbox': pbox,
                    'class_id': None,
                    'inferred': True
                }
                augmented.append(aug)

    return augmented


def resolve_conflicts_per_person(detections, required_ppe=None, iou_thresh=0.05):
    """For each detected person, if both a PPE item and the corresponding
    'no_*' violation exist for that person, keep only the detection with higher
    confidence. Returns a new list of detections.
    """
    if required_ppe is None:
        required_ppe = ['hard_hat', 'safety_vest', 'safety_gloves', 'safety_boots', 'eye_protection']

    # make a shallow copy
    dets = list(detections)

    # collect persons and their indices
    persons = [(i, d) for i, d in enumerate(dets) if d.get('class') == 'person']
    if not persons:
        return dets

    # assign each non-person detection to the most likely person using a robust strategy:
    # 1) if the detection center is inside a person bbox -> assign to that person
    # 2) else use highest IoU above threshold
    person_boxes = [p[1].get('bbox', [0,0,1,1]) for p in persons]
    person_indices = [p[0] for p in persons]

    assigned = {pi: [] for pi in person_indices}
    unassigned = []

    for idx, d in enumerate(dets):
        if d.get('class') == 'person':
            continue

        assigned_to = None
        # center-based assignment
        try:
            bx = d.get('bbox', [0,0,0,0])
            cx = (float(bx[0]) + float(bx[2])) / 2.0
            cy = (float(bx[1]) + float(bx[3])) / 2.0
            for pidx, pbox in zip(person_indices, person_boxes):
                if cx >= pbox[0] and cx <= pbox[2] and cy >= pbox[1] and cy <= pbox[3]:
                    assigned_to = pidx
                    break
        except Exception:
            assigned_to = None

        # fallback to IoU-based assignment
        if assigned_to is None:
            best_iou = 0.0
            best_person_idx = None
            for pidx, pbox in zip(person_indices, person_boxes):
                iou = _bbox_iou(pbox, d.get('bbox', [0,0,0,0]))
                if iou > best_iou:
                    best_iou = iou
                    best_person_idx = pidx
            if best_iou >= iou_thresh and best_person_idx is not None:
                assigned_to = best_person_idx

        if assigned_to is not None:
            assigned[assigned_to].append(idx)
        else:
            unassigned.append(idx)

    to_remove = set()

    # For each person, resolve conflicts per required PPE type using explicit positive<->negative mapping
    for pidx in person_indices:
        assigned_idxs = assigned.get(pidx, [])
        for ppe in required_ppe:
            pos_class = ppe
            neg_class = f'no_{ppe}'
            positive_idxs = [i for i in assigned_idxs if dets[i].get('class') == pos_class]
            negative_idxs = [i for i in assigned_idxs if dets[i].get('class') == neg_class]

            if positive_idxs and negative_idxs:
                # compare top confidences
                best_pos = max(positive_idxs, key=lambda i: float(dets[i].get('confidence', 0.0)))
                best_neg = max(negative_idxs, key=lambda i: float(dets[i].get('confidence', 0.0)))
                if float(dets[best_pos].get('confidence', 0.0)) >= float(dets[best_neg].get('confidence', 0.0)):
                    # remove negatives
                    for i in negative_idxs:
                        to_remove.add(i)
                else:
                    for i in positive_idxs:
                        to_remove.add(i)

    # Build final list excluding removed indices
    final = [d for idx, d in enumerate(dets) if idx not in to_remove]
    return final

def create_detection_chart(detections):
    """Create a chart showing detection results"""
    if not detections:
        return None
    
    # Count detections by class
    detection_counts = {}
    for det in detections:
        class_name = det['class']
        if class_name not in detection_counts:
            detection_counts[class_name] = []
        detection_counts[class_name].append(det['confidence'])
    
    # Create data for plotting
    classes = []
    avg_confidences = []
    counts = []
    colors = []
    
    for class_name, confidences in detection_counts.items():
        classes.append(class_name)
        avg_confidences.append(np.mean(confidences))
        counts.append(len(confidences))
        
        # Color coding
        if class_name.startswith('no_'):
            colors.append('#FF6B6B')  # Red for violations
        elif class_name == 'person':
            colors.append('#4ECDC4')  # Teal for persons
        else:
            colors.append('#45B7D1')  # Blue for good PPE
    
    # Create bar chart
    fig = go.Figure(data=[
        go.Bar(
            x=classes,
            y=counts,
            text=[f'{count} ({conf:.2f})' for count, conf in zip(counts, avg_confidences)],
            textposition='auto',
            marker_color=colors,
            hovertemplate='<b>%{x}</b><br>Count: %{y}<br>Avg Confidence: %{text}<extra></extra>'
        )
    ])
    
    fig.update_layout(
        title="PPE Detection Results",
        xaxis_title="PPE Category",
        yaxis_title="Number of Detections",
        template="plotly_white",
        height=400
    )
    
    return fig

def main():
    # Header
    st.markdown('<h1 class="main-header">🦺 PPE Safety Detection System</h1>', unsafe_allow_html=True)
    st.markdown("### Upload an image to analyze Personal Protective Equipment compliance")
    
    # Sidebar
    st.sidebar.markdown("## ⚙️ Settings")
    
    confidence_threshold = st.sidebar.slider(
        "Confidence Threshold",
        min_value=0.1,
        max_value=0.9,
        value=0.3,
        step=0.1,
        help="Minimum confidence for detections"
    )
    
    show_bboxes = st.sidebar.checkbox(
        "Show Bounding Boxes",
        value=True,
        help="Draw bounding boxes on detected objects"
    )
    
    show_detailed_analysis = st.sidebar.checkbox(
        "Show Detailed Analysis",
        value=True,
        help="Display detailed technical analysis"
    )

    # Toggle to use tuned thresholds (from previous tuning runs)
    use_tuned = st.sidebar.checkbox("Use tuned per-class thresholds", value=True)
    thresholds_file_default = 'outputs/safe_maxap_cap_calibrated/safe_maxap_cap_thresholds.json'
    if use_tuned:
        thresholds_file = st.sidebar.text_input("Thresholds JSON path", value=thresholds_file_default)
        try:
            if os.path.exists(thresholds_file):
                with open(thresholds_file, 'r', encoding='utf-8') as tf:
                    tuned_thresholds = json.load(tf)
            else:
                tuned_thresholds = {}
        except Exception:
            tuned_thresholds = {}
    else:
        tuned_thresholds = {}

    # postprocessing controls
    person_merge_iou = st.sidebar.slider('Person merge IoU', min_value=0.1, max_value=0.9, value=0.45, step=0.05)
    per_class_iou = st.sidebar.slider('Per-class NMS IoU', min_value=0.1, max_value=0.9, value=0.6, step=0.05)
    final_conf_min = st.sidebar.slider('Final confidence min (keep-high-confidence)', min_value=0.0, max_value=1.0, value=0.3, step=0.05)
    infer_missing_ppe_opt = st.sidebar.checkbox('Infer missing PPE (person-centric)', value=False, help='If enabled, infer violations (no_*) when PPE items are absent inside detected person boxes')
    
    # Load model
    # Model selection UI: detect latest model and allow override
    default_model_path = None
    # attempt to detect latest model quickly; prefer rcnn_baseline.pth if present
    try:
        model_dir = os.path.join(os.path.dirname(__file__), 'models')
        rcnn_candidate = os.path.join(model_dir, 'rcnn_baseline.pth')
        if os.path.exists(rcnn_candidate):
            default_model_path = rcnn_candidate
        else:
            if os.path.exists(model_dir):
                candidates = [os.path.join(model_dir, f) for f in os.listdir(model_dir) if f.lower().endswith(('.pth', '.pt'))]
                if candidates:
                    candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
                    default_model_path = candidates[0]
    except Exception:
        default_model_path = 'models/best_model_regularized.pth'

    model_path_input = st.sidebar.text_input('PPE model path', value=default_model_path or 'models/best_model_regularized.pth')
    detector_choice = st.sidebar.selectbox('Detector type', options=['auto', 'rcnn', 'ssd'], index=0, help='Choose which detector architecture to use')

    # Vision-language model selection for scene captioning
    vision_model_choice = st.sidebar.selectbox('Vision-language model', options=['blip2', 'llava', 'none'], index=0, help='Choose a vision-language model for scene descriptions')
    vision_checkpoint_input = st.sidebar.text_input('Vision model checkpoint / hub-id (optional)', value='')
    if st.sidebar.button('Reload model'):
        # clear cache and reload on demand
        try:
            load_model.clear()
        except Exception:
            pass

    model, error = load_model(model_path=model_path_input, detector_type=detector_choice, vision_model=vision_model_choice, vision_checkpoint=(vision_checkpoint_input or None))
    if model and getattr(model, 'vision_fallback_msg', None):
        st.info(model.vision_fallback_msg)
    
    if error:
        st.error(f"❌ Failed to load model: {error}")
        st.stop()
    
    if model is None:
        st.error("❌ Model failed to load properly")
        st.stop()
    
    st.success("✅ AI Models loaded successfully!")

    # Show explicit status badges so the user can tell if the real PPE model
    # and/or vision-language model were actually loaded (vs. falling back to mocks).
    try:
        # Attempt to eagerly ensure the PPE detector is loaded so the UI can
        # show whether real detections will be used. This is safe and quick
        # when a local checkpoint exists; otherwise it will silently fail.
        ppe_loaded = False
        try:
            ppe_loaded = bool(model._ensure_ppe_model_loaded())
        except Exception:
            ppe_loaded = False

        if ppe_loaded and getattr(model, 'ppe_model', None) is not None:
            st.info(f"PPE detector ready: {detector_choice.upper()} (model: {os.path.basename(model_path_input) if model_path_input else 'unknown'})")
        else:
            st.warning("PPE detector not loaded; the app will use lightweight mock detections until a valid PPE model path is provided.")
    except Exception:
        st.warning("Could not determine PPE detector status.")

    try:
        vlm_loaded = False
        # Only check vision model if user has explicitly requested one
        if (vision_model_choice or '').lower() != 'none':
            try:
                vlm_loaded = bool(model._ensure_vision_model_loaded())
            except Exception:
                vlm_loaded = False

        if (vision_model_choice or '').lower() == 'none':
            st.info("Vision-language model: disabled (using deterministic scene descriptions)")
        elif vlm_loaded:
            st.info(f"Vision-language model loaded: {vision_model_choice}")
        else:
            st.warning("Vision-language model not available; captions will be deterministic fallbacks.")
    except Exception:
        st.warning("Could not determine vision-language model status.")
    
    # File upload
    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=['png', 'jpg', 'jpeg'],
        help="Upload a construction site or workplace image for PPE analysis"
    )
    
    if uploaded_file is not None:
        # Load and display image
        image = Image.open(uploaded_file)
        
        # Create columns for layout
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("### 📷 Original Image")
            st.image(image, caption=f"Uploaded: {uploaded_file.name}", use_column_width=True)
        
        # Run analysis
        with st.spinner("🔍 Analyzing PPE compliance..."):
            # ensure variable exists even if analysis fails
            filtered_detections = []
            try:
                # Run hybrid analysis
                results = model.generate_hybrid_description(
                    image,
                    include_general_caption=True
                )
                
                # Apply tuned thresholds if requested, otherwise simple confidence filter
                raw_dets = results.get('ppe_detections', [])
                # Normalize raw detections early so any postprocessing helpers
                # that expect 'class_name' can operate safely.
                raw_dets = normalize_detections(raw_dets, image)
                if use_tuned and tuned_thresholds:
                    # apply per-class thresholding
                    filtered = []
                    for d in raw_dets:
                        cname = d.get('class')
                        thr = tuned_thresholds.get(cname, confidence_threshold)
                        if float(d.get('confidence', 0.0)) >= float(thr):
                            filtered.append(d)
                    # apply per-class NMS and person postprocessing if helpers are available
                    if _apply_per_class_nms:
                        filtered = _apply_per_class_nms(filtered, iou_thresh=per_class_iou, per_class_iou=None, max_per_class=None, final_conf_min=final_conf_min)
                    if _postprocess_persons:
                        from argparse import Namespace
                        darr = Namespace()
                        darr.person_merge_iou = person_merge_iou
                        darr.person_conf_min = None
                        darr.person_area_min_frac = 0.0
                        darr.person_area_max_frac = 1.0
                        darr.person_final_max = None
                        results_temp = {'image_path': '', 'detections': filtered}
                        results_temp = _postprocess_persons(results_temp, None, darr, [d.get('class') for d in raw_dets])
                        filtered = results_temp.get('detections', [])
                    results['ppe_detections'] = filtered
                else:
                    # simple confidence filter
                    filtered_detections = [d for d in raw_dets if d['confidence'] >= confidence_threshold]
                    results['ppe_detections'] = filtered_detections

                # ensure filtered_detections variable exists for later UI blocks
                filtered_detections = results.get('ppe_detections', [])

                # Normalize detections to a consistent schema for downstream code
                filtered_detections = normalize_detections(filtered_detections, image)
                # Optionally infer missing PPE (person-centric) to surface violations
                if infer_missing_ppe_opt:
                    try:
                        filtered_detections = infer_missing_ppe(filtered_detections)
                    except Exception as e:
                        st.warning(f"Could not infer missing PPE: {e}")
                # Resolve conflicts between PPE and inferred violations per person
                try:
                    filtered_detections = resolve_conflicts_per_person(filtered_detections)
                except Exception:
                    # non-fatal
                    pass
                
            except Exception as e:
                st.error(f"❌ Analysis failed: {str(e)}")
                st.stop()
        
        with col2:
            st.markdown("### 🎯 Analysis Results")
            
            # Show image with bounding boxes if enabled
            if show_bboxes and filtered_detections:
                try:
                    annotated_image = draw_bounding_boxes(image, filtered_detections)
                    st.image(annotated_image, caption="Detected PPE Items", use_column_width=True)
                except Exception as e:
                    st.warning(f"Could not draw bounding boxes: {e}")
                    st.image(image, use_column_width=True)
            else:
                st.image(image, use_column_width=True)
        
        # Results section
        st.markdown("---")
        st.markdown("## 📊 Safety Analysis Results")
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        # Count different types of detections
        people_count = len([d for d in filtered_detections if d['class'] == 'person'])
        good_ppe_count = len([d for d in filtered_detections if not d['class'].startswith('no_') and d['class'] != 'person'])
        violation_count = len([d for d in filtered_detections if d['class'].startswith('no_')])
        total_detections = len(filtered_detections)
        
        with col1:
            st.metric("👥 People Detected", people_count)
        
        with col2:
            st.metric("✅ PPE Items Found", good_ppe_count)
        
        with col3:
            st.metric("❌ Violations Found", violation_count)
        
        with col4:
            st.metric("🔍 Total Detections", total_detections)
        
        # Recompute PPE descriptions from the filtered detections so the
        # textual compliance/status metrics match the visualized detections.
        try:
            recomputed_desc = model.generate_ppe_focused_description(filtered_detections)
            # replace the descriptions in results so downstream UI uses the filtered view
            results['ppe_descriptions'] = recomputed_desc
        except Exception as e:
            # If recompute fails, keep original descriptions but warn the user
            st.warning(f"Could not recompute PPE descriptions from filtered detections: {e}")

        # Compliance status
        st.markdown("### 🚨 Compliance Status")
        compliance_status = results['ppe_descriptions']['compliance_status']
        
        if "COMPLIANT" in compliance_status and "NON-COMPLIANCE" not in compliance_status:
            st.markdown(f'<div class="detection-box compliance-good"><h4>✅ {compliance_status}</h4></div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="detection-box compliance-bad"><h4>❌ {compliance_status}</h4></div>', unsafe_allow_html=True)
        
        # Safety summary
        st.markdown("### 🦺 Safety Summary")
        st.info(results['ppe_descriptions']['safety_summary'])
        
        # Scene description
        if results.get('general_caption'):
            st.markdown("### 🎯 Scene Description")
            st.write(f"**AI Description:** {results['general_caption']}")
        
        # Detection visualization
        if filtered_detections:
            st.markdown("### 📈 Detection Breakdown")
            
            # Create and display chart
            fig = create_detection_chart(filtered_detections)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            
            # Detailed detection list
            if show_detailed_analysis:
                st.markdown("### 🔍 Detailed Detections")
                
                # Create DataFrame for better display
                detection_data = []
                for i, det in enumerate(filtered_detections):
                    detection_data.append({
                        'Item': det['class'],
                        'Confidence': f"{det['confidence']:.3f}",
                        'Category': 'Person' if det['class'] == 'person' else ('Violation' if det['class'].startswith('no_') else 'PPE'),
                        'Bounding Box': f"[{det['bbox'][0]:.0f}, {det['bbox'][1]:.0f}, {det['bbox'][2]:.0f}, {det['bbox'][3]:.0f}]"
                    })
                
                df = pd.DataFrame(detection_data)
                st.dataframe(df, use_container_width=True)
        
        # Technical details
        if show_detailed_analysis:
            st.markdown("### 🔧 Technical Analysis")
            st.text(results['ppe_descriptions']['detailed_analysis'])
        
        # Download results
        st.markdown("### 💾 Export Results")
        
        # Prepare data for download
        export_data = {
            'timestamp': datetime.now().isoformat(),
            'image_name': uploaded_file.name,
            'confidence_threshold': confidence_threshold,
            'total_detections': total_detections,
            'people_count': people_count,
            'good_ppe_count': good_ppe_count,
            'violation_count': violation_count,
            'compliance_status': compliance_status,
            'safety_summary': results['ppe_descriptions']['safety_summary'],
            'scene_description': results.get('general_caption', ''),
            'detailed_analysis': results['ppe_descriptions']['detailed_analysis'],
            'detections': filtered_detections
        }
        
        # Convert to JSON
        json_string = json.dumps(export_data, indent=2)
        
        st.download_button(
            label="📋 Download Analysis Report (JSON)",
            data=json_string,
            file_name=f"ppe_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )
    
    else:
        # Welcome message when no image is uploaded
        st.markdown("""
        ### 🎯 How to use this system:
        
        1. **Upload an image** using the file uploader above
        2. **Adjust settings** in the sidebar if needed
        3. **View results** including:
           - PPE compliance status
           - Detected violations and safety equipment
           - Visual analysis with bounding boxes
           - Detailed technical breakdown
        4. **Export results** for reporting and documentation
        
        ### 🦺 What this system detects:
        
        **✅ Safety Equipment:**
        - Hard hats
        - Safety vests
        - Safety gloves
        - Safety boots
        - Eye protection
        
        **❌ Violations:**
        - Missing hard hats
        - Missing safety vests
        - Missing safety gloves
        - Missing safety boots
        - Missing eye protection
        
        ### 📈 Perfect for:
        - Construction site safety audits
        - Workplace compliance monitoring
        - Safety training and education
        - Documentation and reporting
        """)

if __name__ == "__main__":
    main()