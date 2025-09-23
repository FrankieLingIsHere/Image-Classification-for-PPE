# Scripts Directory Structure

This directory contains all the scripts for the PPE detection project, organized into logical categories.

## Core Scripts (Main Directory)

### Training Scripts
- **`train.py`** - Original comprehensive training script with full features
- **`train_enhanced.py`** - Enhanced training with validation monitoring and early stopping (RECOMMENDED)
- **`continue_training.py`** - Continue training from existing checkpoints

### Inference & Testing Scripts  
- **`demo.py`** - Main demonstration script for PPE detection on images
- **`inference.py`** - Core inference functionality
- **`batch_test.py`** - Batch processing for multiple images
- **`test_model_fixed.py`** - Current working test script with comprehensive analysis
- **`test_multiple.py`** - Test multiple images with summary

### Analysis & Visualization Scripts
- **`plot_losses.py`** - Generate training and validation loss curves

## Organized Subdirectories

### 📁 `tools/` - Utility Scripts
Development and maintenance utilities:
- `check_classes.py` - Verify class definitions and counts
- `check_dataset.py` - Dataset validation and analysis
- `convert_osha_dataset.py` - Convert OSHA dataset format
- `copy_images_simple.py` - Simple image copying utility
- `debug_inference.py` - Debug inference issues
- `debug_model.py` - Model debugging utilities
- `dimension_fix.py` - Fix image dimension issues
- `fix_dimensions.py` - Another dimension fixing utility
- `fix_label_studio.py` - Label Studio data fixes
- `fix_label_studio_export.py` - Fix Label Studio export format
- `setup_label_studio.py` - Set up Label Studio environment
- `training_guide.py` - Training guidance and tips
- `training_workflow.py` - Automated training workflows

### 📁 `experimental/` - Research & Development
Experimental features and hybrid models:
- `demo_hybrid_ppe.py` - Hybrid PPE detection demo
- `integrated_ppe_model.py` - Integrated PPE model approach
- `setup_hybrid_model.py` - Set up hybrid model architecture

### 📁 `archived/` - Deprecated Scripts
Old versions and deprecated functionality:
- `plot_training.py` - Old training visualization (replaced by plot_losses.py)
- `simple_test.py` - Simple test script (superseded)
- `simple_train.py` - Basic training script (superseded)
- `test_model.py` - Old test model script (superseded)
- `train_dynamic.py` - Dynamic training approach (archived)
- `train_fixed.py` - Fixed training script (archived)
- `train_simple.py` - Simple training script (archived)

## Quick Start Guide

### For Training:
```bash
# New training with validation monitoring (recommended)
python scripts/train_enhanced.py --epochs 20 --batch_size 8 --lr 1e-4

# Continue from checkpoint
python scripts/continue_training.py
```

### For Testing:
```bash
# Test single image
python scripts/demo.py path/to/image.jpg

# Batch test multiple images
python scripts/batch_test.py

# Comprehensive model testing
python scripts/test_model_fixed.py
```

### For Analysis:
```bash
# Generate training curves
python scripts/plot_losses.py

# Check dataset
python scripts/tools/check_dataset.py
```

## File Cleanup Summary

### Moved to Better Locations:
- Documentation → `docs/` folder
- Test results and outputs → `outputs/` folder
- Utility scripts → `scripts/tools/`
- Experimental code → `scripts/experimental/`
- Deprecated scripts → `scripts/archived/`

### Current Active Scripts:
- **Training**: `train_enhanced.py` (primary), `continue_training.py`
- **Testing**: `test_model_fixed.py`, `batch_test.py`, `demo.py`
- **Analysis**: `plot_losses.py`
- **Core**: `train.py`, `inference.py`

This organization makes the project more maintainable and easier to navigate for both development and production use.