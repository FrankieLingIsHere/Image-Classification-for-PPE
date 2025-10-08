# PPE Detection Performance Evaluation

Comprehensive evaluation tools to assess model performance against ground truth annotations.

## Quick Start

### Option 1: Quick Evaluation (Recommended)
```bash
# Auto-finds latest model and runs evaluation
python scripts/quick_evaluate.py
```

### Option 2: Detailed Evaluation
```bash
# Specify model and parameters
python scripts/evaluate_detection_performance.py \
    --model_path models/best_model.pth \
    --data_dir data \
    --split test \
    --output_dir outputs/evaluation_results
```

## What Gets Evaluated

### Core Metrics
- **mAP (mean Average Precision)**: Overall detection quality
- **Per-class AP**: Individual class performance
- **Precision/Recall/F1**: Detection accuracy metrics
- **True/False Positives**: Detailed error analysis

### Problem Detection
1. **Missed Workers**: Images where person count is underestimated
2. **Environmental False Positives**: PPE detected in background/environment
3. **Missed Violations**: Safety violations not properly identified

## Output Structure

All results saved to: `outputs/evaluation_results/`

```
evaluation_results/
├── evaluation_results_YYYYMMDD_HHMMSS.json    # Complete results
├── class_metrics_YYYYMMDD_HHMMSS.csv          # Per-class summary
├── problem_analysis_YYYYMMDD_HHMMSS.txt       # Problem cases
├── class_performance.png                       # Performance charts
└── problem_summary.png                         # Problem visualization
```

### Key Files Explained

**📄 evaluation_results_*.json**
- Complete evaluation data including all detections vs ground truth
- Problem case details with specific image names and metrics
- Model configuration and evaluation parameters

**📊 class_metrics_*.csv** 
- Spreadsheet with AP, precision, recall, F1 for each class
- Ground truth vs detection counts
- Ready for further analysis or reporting

**🔍 problem_analysis_*.txt**
- Human-readable summary of the 3 main problem categories
- Lists specific problematic images for manual review
- Suggests areas needing improvement

## Interpreting Results

### Good Performance Indicators
- **Overall mAP > 0.6**: Generally acceptable detection quality
- **Person AP > 0.7**: Good worker detection (most critical)
- **Violation AP > 0.5**: Adequate safety violation detection
- **< 5% missed workers**: Most people properly detected

### Warning Signs
- **mAP < 0.4**: Poor overall performance - needs model improvement
- **High environmental FPs**: Consider post-processing filters
- **Low violation recall**: May need lower confidence thresholds
- **Inconsistent person detection**: NMS tuning required

### Common Issues & Solutions

| Problem | Symptom | Suggested Fix |
|---------|---------|---------------|
| Missed workers | Person AP < 0.5 | Lower NMS threshold, add multi-person training data |
| Environmental FPs | High FP count for PPE classes | Add contextual filtering, more negative examples |
| Small object misses | Low AP for hard_hat/gloves | Adjust anchor sizes, lower confidence thresholds |
| Violation blindness | Low no_hard_hat/no_safety_vest AP | Class-specific thresholds, focal loss training |

## Requirements

- Ground truth annotations in `data/annotations/` (XML or JSON format)
- Test split defined in `data/splits/test.txt`
- Trained model checkpoint in `models/` directory
- Standard dependencies: torch, matplotlib, pandas, seaborn

## Dataset Format Expected

### Annotation Files (XML)
```xml
<annotation>
    <object>
        <name>hard_hat</name>
        <bndbox>
            <xmin>100</xmin><ymin>100</ymin>
            <xmax>200</xmax><ymax>200</ymax>
        </bndbox>
    </object>
</annotation>
```

### Split Files
```
# data/splits/test.txt
construction_site_001.jpg
worker_safety_002.jpg
ppe_violation_003.jpg
```

## Troubleshooting

**"No model files found"**
- Ensure you have a `.pth` file in the `models/` directory
- Or specify model path explicitly with `--model_path`

**"Split file not found"**
- Create `data/splits/test.txt` with your test image filenames
- Or use `--split train` or `--split val` if you have those

**"No annotation found for image"**
- Check that annotation files exist in `data/annotations/`
- Verify filename matching (e.g., `image.jpg` → `image.xml`)
- Supported formats: VOC XML, JSON

**Low performance results**
- Verify annotation quality and consistency
- Check class name matching between model and annotations
- Consider if model training completed successfully

## Integration with Training

This evaluation script is designed to work with the existing PPE detection pipeline:

```bash
# Typical workflow
python scripts/train.py --epochs 100          # Train model
python scripts/quick_evaluate.py              # Evaluate performance
python scripts/train.py --epochs 50 --resume  # Continue training if needed
```

The evaluation results can guide training improvements and hyperparameter tuning.