# 🏗️ Model Implementations

This folder contains model code, implementations, and utilities for building PPE detection models.

## 📂 Structure

```
src/models/
├── hybrid_ppe_model.py      # ✅ Hybrid model (currently used)
├── relational_rescorer.py.disabled  # ❌ Disabled - ineffective
├── ssd.py                   # ⚠️ SSD implementation (experimental)
└── __init__.py              # Module initialization
```

## ✅ Active Models

### `hybrid_ppe_model.py`
- **Purpose**: Current production model implementation
- **Architecture**: Hybrid approach combining detection with PPE-specific logic
- **Status**: ✅ Active - in use
- **Usage**: 
  ```python
  from src.models.hybrid_ppe_model import HybridPPEModel
  model = HybridPPEModel(num_classes=12)
  ```

## ❌ Archived/Disabled Models

### `relational_rescorer.py.disabled`
- **Purpose**: Spatial relationship modeling using graph neural networks
- **Why Disabled**: Ineffective - too restrictive, no measurable improvement
- **Performance**: No improvement over baseline
- **Status**: ❌ Disabled - do NOT use
- **Reference**: See `_ARCHIVED_EXPERIMENTS/model_files/` for archived version

### `ssd.py`
- **Purpose**: Single Shot MultiBox Detector implementation
- **Status**: ⚠️ Experimental - not recommended for production
- **Performance**: Not validated against baseline
- **Note**: Kept for reference only

## 🔧 Implementation Notes

### Building Custom Models
When implementing new models:

1. **Follow torchvision API** - Use standard PyTorch detection APIs
2. **Test against baseline** - Must show > 1% improvement
3. **Keep it simple** - Simpler models generalize better on small data
4. **Document performance** - Include expected metrics in docstring

### Adding New Models
```python
# New model should follow this structure:
class MyModel(torch.nn.Module):
    def __init__(self, num_classes=13):
        super().__init__()
        # initialization
    
    def forward(self, images, targets=None):
        # return predictions or losses
        
    @staticmethod
    def load_pretrained(num_classes=13):
        # load with pretrained weights
```

## 📊 Current Model Hierarchy

```
torchvision Faster R-CNN
    ↓
Base detection backbone (ResNet50+FPN)
    ↓
PPE-specific adaptations
    ↓
Confidence calibration (focal loss, temperature scaling)
    ↓
Post-processing filters
```

## 🚀 Production Pipeline

1. **Base Model**: `torchvision.models.detection.fasterrcnn_resnet50_fpn`
2. **Training**: `scripts/train/train_with_confidence.py` 
3. **Checkpoint**: `models/production/rcnn_baseline_adamw.pth`
4. **Inference**: `scripts/inference.py`

## ⚙️ Model Configuration

**Current Best Config**:
```yaml
model: Faster R-CNN ResNet50+FPN
backbone: ResNet50 pretrained on ImageNet
fpn_layers: [256, 256, 256, 256]
num_classes: 12 PPE + 1 background = 13
anchor_scales: [32, 64, 128, 256, 512]
loss_type: Focal Loss + Class Weights
optimizer: AdamW
learning_rate: 0.001
regularization: L2 (weight decay)
```

**Expected Performance**:
- mAP: 0.2659 (baseline) → 0.28-0.30 (with confidence calibration)
- Confidence: 0.125 → 0.82+

## 🔄 Migration Guide

### From Old Multi-Task Model
```python
# ❌ OLD (don't use)
from src.models.enhanced_ppe_detector import EnhancedPPEDetector
model = EnhancedPPEDetector()  # File removed, archived

# ✅ NEW (use this)
from torchvision.models.detection import fasterrcnn_resnet50_fpn
model = fasterrcnn_resnet50_fpn(num_classes=13)
```

### Loading Checkpoints
```python
# ✅ Load production checkpoint
model = fasterrcnn_resnet50_fpn(num_classes=13)
checkpoint = torch.load('models/production/rcnn_baseline_adamw.pth')
model.load_state_dict(checkpoint)
```

## 📚 Related Documentation

- **Training**: See `scripts/train/README.md`
- **Inference**: See `scripts/inference.py`
- **Failed Approaches**: See `_ARCHIVED_EXPERIMENTS/README.md`
- **Improvement Tips**: See `docs/IMPROVEMENT_RECOMMENDATIONS.md`

## 🎯 Performance Benchmarks

| Model | mAP | Confidence | Speed | Status |
|-------|-----|------------|-------|--------|
| Baseline RCNN | 0.2659 | 0.125 | ✅ Fast | ✅ Production |
| Enhanced (4-task) | 0.0574 | 0.105 | ❌ Slow | ❌ Archived |
| RCNN + Rescorer | 0.2650 | 0.128 | ⚠️ Slow | ❌ Archived |
| RCNN + Confidence Cal. | 0.28-0.30 | 0.82+ | ✅ Fast | 🚀 Next |

## ✅ Checklist for Adding New Models

- [ ] Inherits from `torch.nn.Module`
- [ ] Implements `forward()` method
- [ ] Supports both training and inference modes
- [ ] Tested against baseline (must improve)
- [ ] Documented with docstrings
- [ ] Added to this README
- [ ] Performance metrics recorded

---

**Current Active Model**: Faster R-CNN ResNet50+FPN
**Location**: `torchvision.models.detection`
**Checkpoint**: `models/production/rcnn_baseline_adamw.pth`
**Next**: Confidence calibration training
