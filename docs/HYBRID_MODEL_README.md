# Hybrid PPE Detection + Description System

## Overview
This system combines **specialized PPE detection** with **general scene understanding** to provide comprehensive analysis of construction site images.

### Architecture: Option 3 - Separate Caption Model (Hybrid)
- **PPE Detection**: Your trained SSD300 model for precise PPE/violation detection
- **Scene Description**: Pre-trained vision-language model (BLIP-2/LLaVA) for general understanding  
- **Fusion Layer**: Intelligent combination of both outputs

## Features

### 🔍 PPE Detection
- Detects 12 PPE classes including violations
- Bounding box coordinates and confidence scores
- OSHA compliance assessment

### 📝 Scene Description  
- General construction site description
- Work activity identification
- Environmental context

### 🔗 Hybrid Analysis
- Safety summary with compliance status
- Technical detection details
- Combined narrative description

## Quick Start

### 1. Install Dependencies
```bash
# Install hybrid model dependencies
python scripts/setup_hybrid_model.py
```

### 2. Test with Demo
```bash
# Single image analysis
python scripts/demo_hybrid_ppe.py --image path/to/construction_image.jpg

# Batch processing
python scripts/demo_hybrid_ppe.py --batch data/images

# Advanced model
python scripts/demo_hybrid_ppe.py --model llava --image image.jpg
```

### 3. Integrate with Your Trained Model
```bash
# Use your trained PPE model + descriptions
python scripts/integrated_ppe_model.py
```

## Usage Examples

### Single Image Analysis
```python
from src.models.hybrid_ppe_model import HybridPPEDescriptionModel
from PIL import Image

# Initialize model
model = HybridPPEDescriptionModel(
    ppe_model_path="path/to/your/trained_model.pth",
    vision_model="blip2",
    device="auto"
)

# Analyze image
image = Image.open("construction_site.jpg")
results = model.generate_hybrid_description(
    image,
    custom_prompt="Analyze PPE compliance and safety."
)

print(results['hybrid_description'])
```

### Batch Processing
```python
from scripts.integrated_ppe_model import IntegratedPPEModel

# Initialize integrated model
integrated = IntegratedPPEModel("configs/ppe_config.yaml")
integrated.initialize_description_model("blip2")

# Process directory
integrated.batch_analyze("data/images", "results.json")
```

## Output Formats

### PPE Detection Results
```json
{
  "class": "safety_vest",
  "confidence": 0.87,
  "bbox": [120, 100, 180, 250],
  "class_id": 3
}
```

### Description Results
```json
{
  "safety_summary": "Construction site with 2 workers. PPE detected: safety vest, hard hat. All workers properly equipped.",
  "compliance_status": "✅ COMPLIANT - All 2 worker(s) properly equipped.",
  "general_caption": "Two construction workers wearing safety gear at a building site.",
  "hybrid_description": "Scene Analysis: Two construction workers wearing safety gear..."
}
```

## Model Options

### Vision-Language Models
- **BLIP-2**: Faster, good general descriptions (~500MB)
- **LLaVA**: More detailed, better reasoning (~13GB)

### PPE Model Integration
- Use your trained SSD300 model for real detection
- Mock detection available for testing
- Easy integration with existing pipeline

## Configuration

Edit `hybrid_model_config.yaml`:
```yaml
# Model Settings
vision_model: "blip2"  # or "llava"
device: "auto"  # "cuda", "cpu", or "auto"

# PPE Classes (matches your trained model)
ppe_classes:
  - background
  - person
  - hard_hat
  - safety_vest
  # ... etc

# Description Settings
include_general_caption: true
custom_prompt: "Describe this construction site focusing on worker safety."
```

## Performance

### Speed (per image)
- **BLIP-2**: ~2-3 seconds (GPU), ~8-10 seconds (CPU)
- **LLaVA**: ~5-8 seconds (GPU), ~30-45 seconds (CPU)
- **PPE Detection**: ~0.1-0.5 seconds (your model)

### Memory Requirements
- **BLIP-2**: ~2GB GPU memory
- **LLaVA**: ~8GB GPU memory  
- **PPE Model**: ~500MB GPU memory

## Integration Steps

### 1. Replace Mock Detection
In `hybrid_ppe_model.py`, update the `detect_ppe()` method:
```python
def detect_ppe(self, image):
    # Your actual detection pipeline:
    # 1. Preprocess image
    # 2. Run through your SSD model
    # 3. Apply NMS and filtering
    # 4. Convert to detection format
    return detections
```

### 2. Update Class Mappings
Ensure class names match your trained model:
```python
self.ppe_classes = [
    'background', 'person', 'hard_hat', 'safety_vest',
    # ... your exact class names
]
```

### 3. Load Your Trained Weights
```python
model = SSD300(num_classes=13)
checkpoint = torch.load("your_trained_model.pth")
model.load_state_dict(checkpoint['model_state_dict'])
```

## Troubleshooting

### Common Issues
1. **CUDA out of memory**: Use smaller batch size or CPU
2. **Model download fails**: Check internet connection
3. **Import errors**: Run `setup_hybrid_model.py` first

### Performance Tips
- Use GPU for faster inference
- Batch multiple images when possible
- Cache model initialization for multiple runs

## Next Steps

1. **Complete PPE model training** using your annotated dataset
2. **Test hybrid system** with real construction site images  
3. **Fine-tune descriptions** with custom prompts
4. **Deploy for production** safety monitoring

## Files Created
- `src/models/hybrid_ppe_model.py` - Main hybrid model
- `scripts/demo_hybrid_ppe.py` - Demo script
- `scripts/setup_hybrid_model.py` - Installation script
- `scripts/integrated_ppe_model.py` - Integration with your model
- `hybrid_model_config.yaml` - Configuration file

Ready to enhance your PPE detection with rich descriptions! 🚀