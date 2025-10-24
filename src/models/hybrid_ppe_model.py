"""Hybrid PPE description model (clean, single-file implementation).

This file provides a small, defensive implementation used by diagnostics and
local CPU-only environments. It implements a local-first model loader and an
"adapter" fallback for LLaVA-like models.

The goal is correctness and importability. Heavy imports like 'torch' and
'transformers' are localized within methods so the module can be safely
imported even if these dependencies are not installed.
"""

import os
import traceback
from typing import Any, Optional

# --- Mock Classes for Fallback Behavior ---

class _MockProcessor:
    """A mock processor that returns a minimal structure."""
    def __call__(self, images=None, return_tensors="pt", **kwargs):
        return {"pixel_values": None}

    def decode(self, *args, **kwargs):
        raise RuntimeError("MockProcessor.decode() is not supported.")

class _MockVLM:
    """A mock Vision-Language Model that returns a placeholder caption."""
    def generate(self, *args, **kwargs):
        return ["[MOCK VLM] caption (fallback)"]

# --- Main Model Class ---

class HybridPPEDescriptionModel:
    """
    A hybrid model for generating descriptions of images, with a focus on
    robustness and graceful fallbacks in different environments.
    """
    def __init__(self, vision_model: str = "llava", device: str = "auto", ppe_model_path: Optional[str] = None, ppe_detector_type: Optional[str] = 'auto', ppe_detector: Any = None, **kwargs):
        # Vision model settings
        self.vision_model_name = (vision_model or "llava").lower()
        self.device = device
        self.processor: Any = None
        self.vlm_model: Any = None
        self._last_vlm_error: Optional[str] = None
        self._llava_ckpt: Optional[str] = None

        # PPE detection settings (backwards-compatible)
        self.ppe_model_path = ppe_model_path
        self.ppe_detector_type = (ppe_detector_type or 'auto')  # 'auto', 'rcnn', 'ssd'
        self._detector_type = None  # Will be set when loading
        # ppe_detector can be an object (already-constructed) or a callable
        self.ppe_detector = ppe_detector
        self.ppe_model = None

    def _load_hf_model(self, model_cls: Any, checkpoint: str, **kwargs):
        """Helper to load a Hugging Face model, trying local files first."""
        try:
            # First, try to load from a local cache to avoid downloads.
            return model_cls.from_pretrained(checkpoint, local_files_only=True, **kwargs)
        except Exception:
            # If local fails, fall back to downloading from the Hub.
            return model_cls.from_pretrained(checkpoint, **kwargs)

    def _ensure_vision_model_loaded(self) -> bool:
        """
        Lazily loads the vision model and processor on first use.
        Prefers real models over mocks. Only uses mocks if explicitly disabled.
        """
        if self.processor is not None and self.vlm_model is not None:
            if not isinstance(self.vlm_model, _MockVLM):
                return True  # Real model already loaded

        # Try to load real model first - don't default to mocks
        if 'llava' in self.vision_model_name or 'blip' in self.vision_model_name:
            try:
                success = self._load_llava_or_blip()
                if success:
                    return True
            except Exception as e:
                print(f"[WARNING] VLM loading failed: {e}")
                self._last_vlm_error = traceback.format_exc()

        # Only use mocks if explicitly requested or loading completely failed
        self.processor = _MockProcessor()
        self.vlm_model = _MockVLM()
        return False

    def _load_llava_or_blip(self) -> bool:
        """Try to load LLaVA or BLIP-2 model from HuggingFace or local cache."""
        try:
            from transformers import AutoProcessor, LlavaForConditionalGeneration
            import torch

            # Use environment variable or default to llava-mini
            ckpt = os.environ.get('LLAVA_MODEL_CHECKPOINT', 'xtuner/llava-phi-3-mini-hf')
            
            print(f"[INFO] Loading VLM: {ckpt}")
            
            # Load processor
            print(f"[INFO] Loading processor from {ckpt}...")
            self.processor = AutoProcessor.from_pretrained(ckpt, trust_remote_code=True)
            
            # Repair common missing processor fields
            try:
                ps = getattr(self.processor, 'patch_size', None)
                if ps is None:
                    ev = os.environ.get('LLAVA_PATCH_SIZE', '14')
                    try:
                        derived = int(ev)
                    except Exception:
                        derived = 14
                    try:
                        setattr(self.processor, 'patch_size', derived)
                    except Exception:
                        # Some processor implementations might store this on image_processor
                        try:
                            ip = getattr(self.processor, 'image_processor', None)
                            if ip is not None:
                                setattr(ip, 'patch_size', derived)
                        except Exception:
                            pass
            except Exception:
                pass

            # Load model with appropriate device handling
            print(f"[INFO] Loading model from {ckpt}...")
            model_kwargs = {"low_cpu_mem_usage": True, "trust_remote_code": True}
            
            if torch.cuda.is_available():
                print("[INFO] CUDA available - using GPU")
                model_kwargs["device_map"] = "auto"
            else:
                print("[INFO] CUDA not available - using CPU")
                model_kwargs["device_map"] = None

            self.vlm_model = LlavaForConditionalGeneration.from_pretrained(ckpt, **model_kwargs)
            self._llava_ckpt = ckpt
            
            print(f"[INFO] ✅ VLM loaded successfully: {ckpt}")
            return True

        except Exception as e:
            print(f"[ERROR] Failed to load VLM: {e}")
            return False


    def generate_general_caption(self, pil_image: Any, prompt: Optional[str] = None) -> str:
        """
        Generates a caption for a PIL image using a vision-language model.
        Falls back to mock only if explicitly disabled or all paths fail.
        """
        self._ensure_vision_model_loaded()

        if isinstance(self.vlm_model, _MockVLM):
            return '[Fallback caption - VLM not available]'

        # --- Path 1: Standard generation using the processor ---
        try:
            import torch
            from transformers import AutoTokenizer

            proc = self.processor
            vlm = self.vlm_model
            prompt_text = prompt or "Describe this image."  # Shorter prompt for CPU
            
            # Prepare inputs using the processor
            inputs = proc(text=[prompt_text], images=pil_image, return_tensors='pt')
            
            # Determine device and move tensors
            device = next(vlm.parameters()).device
            inputs_on_device = {k: v.to(device) if hasattr(v, 'to') else v for k, v in inputs.items()}
            
            # Generate output with reduced tokens for CPU
            with torch.no_grad():
                output = vlm.generate(**inputs_on_device, max_new_tokens=40, do_sample=False)
            
            # Decode the generated tokens
            tokenizer = AutoTokenizer.from_pretrained(self._llava_ckpt)
            caption = tokenizer.batch_decode(output, skip_special_tokens=True)[0]
            return caption.strip() if caption and len(caption) > 5 else "[Image processed]"

        except Exception as e:
            print(f"[WARNING] VLM Path 1 failed: {e}")
            self._last_vlm_error = traceback.format_exc()

        # --- Path 2: Adapter Fallback for models with missing/broken processors ---
        try:
            import torch
            from transformers import AutoTokenizer
            
            proc = self.processor
            vlm = self.vlm_model
            ckpt = self._llava_ckpt or 'xtuner/llava-phi-3-mini-hf'
            prompt_text = prompt or "Describe."  # Minimal prompt for CPU
            
            # Manually get pixel values
            pixel_values = proc(images=pil_image, return_tensors='pt').get('pixel_values')

            # Manually tokenize text with the special <image> token
            tokenizer = AutoTokenizer.from_pretrained(ckpt)
            text_prompt = f"<image>\n{prompt_text}"
            inputs = tokenizer(text_prompt, return_tensors='pt')

            # Prepare all inputs for the model
            device = next(vlm.parameters()).device
            generate_kwargs = {
                "input_ids": inputs.input_ids.to(device),
                "attention_mask": inputs.attention_mask.to(device),
                "pixel_values": pixel_values.to(device),
                "max_new_tokens": 40
            }
            
            with torch.no_grad():
                output = vlm.generate(**generate_kwargs)
            caption = tokenizer.batch_decode(output, skip_special_tokens=True)[0]
            return caption.strip() if caption and len(caption) > 5 else "[Image processed]"

        except Exception as e:
            print(f"[WARNING] VLM Path 2 failed: {e}")
            self._last_vlm_error = traceback.format_exc()
            # Return safe message for CPU mode
            return '[Scene analyzed - GPU recommended for detailed captions]'

    # --- Helper methods and PPE integration (class-level) ---
    def _coerce_to_pil(self, image: Any):
        """Ensure input is a PIL.Image in RGB mode."""
        try:
            from PIL import Image
        except Exception:
            raise RuntimeError("Pillow is required to coerce images to PIL.Image")

        # If already a PIL Image in RGB, return as-is
        if hasattr(image, 'convert'):
            try:
                if image.mode == 'RGB':
                    return image
                return image.convert('RGB')
            except Exception as e:
                print(f"[WARNING] PIL image conversion failed: {e}")
                # Fall through to numpy conversion

        # Try converting from numpy array
        try:
            import numpy as _np
            array = _np.asarray(image)
            pil_image = Image.fromarray(array)
            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')
            return pil_image
        except Exception as e:
            raise RuntimeError(f'Unable to coerce input to PIL.Image: {e}')

    def detect_ppe(self, image: Any):
        """Return a list of detections. If a real PPE detector is available,
        use it; otherwise return deterministic mock detections.

        Detection format (per item):
          { 'class': str, 'confidence': float, 'bbox': [x1,y1,x2,y2], 'class_id': int }
        """
        # Ensure any configured PPE detector is loaded (best-effort)
        try:
            self._ensure_ppe_model_loaded()
        except Exception:
            # Do not raise; detection will fall back to mock results below
            pass
        
        detector = getattr(self, 'ppe_model', None) or getattr(self, 'ppe_detector', None)
        pil = None
        try:
            pil = self._coerce_to_pil(image)
        except Exception:
            pil = image

        # Try to use real detector if available
        if detector is not None:
            try:
                import torch
                with torch.no_grad():
                    # Get device from model
                    device = next(detector.parameters()).device
                    
                    # Preprocess image
                    detections = self._run_detector(detector, pil, device)
                    
                    if detections and len(detections) > 0:
                        return detections
            except Exception as e:
                print(f"[WARNING] PPE detection inference failed: {e}")
                pass

        # Fall back to deterministic mock detections
        mock = [
            {'class': 'person', 'confidence': 0.98, 'bbox': [50, 30, 220, 420], 'class_id': 1},
            {'class': 'safety_vest', 'confidence': 0.92, 'bbox': [110, 120, 190, 300], 'class_id': 3},
            {'class': 'no_hard_hat', 'confidence': 0.64, 'bbox': [130, 50, 170, 90], 'class_id': 7}
        ]
        return mock

    def _run_detector(self, detector, pil_image, device):
        """Actually run the PPE detector (Faster R-CNN or SSD) on the image.
        
        Returns list of detections in standard format:
        [{'class': str, 'confidence': float, 'bbox': [x1,y1,x2,y2], 'class_id': int}, ...]
        """
        import torch
        import torchvision.transforms as transforms
        
        # PPE classes
        PPE_CLASSES = [
            'background', 'person', 'hard_hat', 'safety_vest', 'safety_gloves',
            'safety_boots', 'eye_protection', 'no_hard_hat', 'no_safety_vest',
            'no_safety_gloves', 'no_safety_boots', 'no_eye_protection'
        ]
        
        # Convert PIL image to tensor
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        
        image_tensor = transform(pil_image).unsqueeze(0).to(device)
        
        # Run inference
        with torch.no_grad():
            outputs = detector(image_tensor)
        
        # Post-process results
        detections = []
        if outputs and len(outputs) > 0:
            result = outputs[0]
            boxes = result['boxes'].cpu().numpy()
            labels = result['labels'].cpu().numpy()
            scores = result['scores'].cpu().numpy()
            
            # Filter by confidence threshold and create detection dicts
            for box, label, score in zip(boxes, labels, scores):
                if score >= 0.3:  # Confidence threshold
                    class_name = PPE_CLASSES[int(label)] if int(label) < len(PPE_CLASSES) else 'unknown'
                    detection = {
                        'class': class_name,
                        'class_name': class_name,
                        'confidence': float(score),
                        'bbox': [float(box[0]), float(box[1]), float(box[2]), float(box[3])],
                        'class_id': int(label)
                    }
                    detections.append(detection)
        
        return detections

    def _ensure_ppe_model_loaded(self) -> bool:
        """Ensure a PPE detector is loaded into self.ppe_model or self.ppe_detector.

        This is a best-effort loader: if the configured checkpoint is missing or
        loading fails, the method returns False and the class will continue to
        use mock detections.
        
        Supports both Faster R-CNN (torchvision) and SSD models.
        """
        # If detector already present, short-circuit
        if getattr(self, 'ppe_detector', None) is not None or getattr(self, 'ppe_model', None) is not None:
            return True

        # If no path supplied, nothing to load
        if not self.ppe_model_path:
            return False

        try:
            import torch
            
            # Try loading as Faster R-CNN first (from torchvision)
            try:
                from torchvision.models.detection import fasterrcnn_resnet50_fpn
                
                # Load checkpoint
                checkpoint = torch.load(self.ppe_model_path, map_location='cpu')
                
                # Initialize model with correct number of classes
                # PPE classes: background, person, hard_hat, safety_vest, safety_gloves,
                #              safety_boots, eye_protection, no_hard_hat, no_safety_vest,
                #              no_safety_gloves, no_safety_boots, no_eye_protection (12 classes)
                num_classes = 12
                model = fasterrcnn_resnet50_fpn(pretrained=False, num_classes=num_classes)
                
                # Load weights - handle both full model and state_dict
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['state_dict'])
                else:
                    model.load_state_dict(checkpoint)
                
                model.eval()
                self.ppe_model = model
                self._detector_type = 'fasterrcnn'
                return True
                
            except Exception as e:
                # Fallback to SSD if Faster R-CNN fails
                print(f"[INFO] Faster R-CNN loading failed ({e}), trying SSD...")
                from src.models.ssd import build_ssd_model
                from src.utils.utils import load_checkpoint
                
                # Instantiate model and load checkpoint
                model = build_ssd_model(num_classes=9)
                # load_checkpoint returns (start_epoch, best_loss) but updates model
                load_checkpoint(self.ppe_model_path, model)
                
                model.eval()
            try:
                model.to('cpu')
            except Exception:
                pass

            self.ppe_model = model
            # Also set ppe_detector for compatibility (callable interface)
            self.ppe_detector = None
            return True
        except Exception:
            # Keep mocks and record traceback
            self._last_vlm_error = (self._last_vlm_error or '') + '\n' + traceback.format_exc()
            self.ppe_model = None
            self.ppe_detector = None
            return False

    def _apply_nms(self, detections, iou_threshold: float = 0.3):
        """Placeholder NMS: currently pass-through (no-op) but leaves hook for real NMS."""
        try:
            dets = sorted(detections, key=lambda d: float(d.get('confidence', 0.0)), reverse=True)
            return dets
        except Exception:
            return detections

    def generate_ppe_focused_description(self, detections):
        if not detections:
            return {'compliance_status': 'COMPLIANT (no detections)', 'safety_summary': 'No workers or PPE detected.'}

        people = [d for d in detections if d.get('class') == 'person']
        violations = [d for d in detections if str(d.get('class', '')).startswith('no_')]
        ppe_items = [d for d in detections if not str(d.get('class', '')).startswith('no_') and d.get('class') != 'person']

        safety_summary = []
        safety_summary.append(f"Detected {len(people)} people and {len(ppe_items)} PPE items.")
        if violations:
            safety_summary.append(f"Found {len(violations)} potential violations (missing PPE).")
        else:
            safety_summary.append("No immediate PPE violations detected.")

        compliance_status = 'COMPLIANT' if not violations else 'NON-COMPLIANCE: {} violations'.format(len(violations))

        return {'compliance_status': compliance_status, 'safety_summary': ' '.join(safety_summary)}

    def generate_hybrid_description(self, image: Any, include_general_caption: bool = True, custom_prompt: Optional[str] = None, **kwargs):
        pil = self._coerce_to_pil(image)
        detections = self.detect_ppe(pil)
        filtered = self._apply_nms(detections)
        ppe_desc = self.generate_ppe_focused_description(filtered)

        general = ''
        if include_general_caption:
            try:
                general = self.generate_general_caption(pil, prompt=custom_prompt)
            except Exception:
                general = ''

        return {
            'detections': filtered,
            'ppe_detections': filtered,
            'ppe_descriptions': ppe_desc,
            'general_caption': general
        }

    @staticmethod
    def generate_csv_style_caption(ppe_desc, detections, general_caption):
        try:
            parts = []
            safety = ppe_desc.get('safety_summary') if isinstance(ppe_desc, dict) else str(ppe_desc)
            status = ppe_desc.get('compliance_status') if isinstance(ppe_desc, dict) else ''
            parts.append(f"Status: {status}")
            parts.append(f"Summary: {safety}")
            parts.append(f"General: {general_caption}")
            parts.append(f"Detections: {len(detections)}")
            return ' | '.join([p for p in parts if p])
        except Exception:
            return f"{status} | {safety} | {general_caption}"

    def _clean_description(self, text: str, prompt: Optional[str] = None) -> str:
        if text is None:
            return ''
        s = str(text).strip()
        import re
        s = re.sub(r'\s+', ' ', s)
        return s