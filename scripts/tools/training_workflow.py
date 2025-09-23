#!/usr/bin/env python3
"""
Complete PPE Model Training Workflow
Step-by-step guide to train your PPE detection model
"""

import os
import sys
import subprocess
from pathlib import Path
import json
import yaml

def check_prerequisites():
    """Check if all prerequisites for training are met"""
    
    print("🔍 Checking training prerequisites...")
    
    checks = {
        'annotations': False,
        'images': False,
        'splits': False,
        'config': False,
        'python_env': False
    }
    
    # Check annotations
    annotations_dir = Path("data/annotations")
    if annotations_dir.exists():
        xml_files = list(annotations_dir.glob("*.xml"))
        if len(xml_files) > 0:
            checks['annotations'] = True
            print(f"   ✅ Annotations: {len(xml_files)} XML files found")
        else:
            print("   ❌ Annotations: No XML files found")
    else:
        print("   ❌ Annotations: Directory not found")
    
    # Check images
    images_dir = Path("data/images")
    if images_dir.exists():
        image_files = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))
        if len(image_files) > 0:
            checks['images'] = True
            print(f"   ✅ Images: {len(image_files)} image files found")
        else:
            print("   ❌ Images: No image files found")
    else:
        print("   ❌ Images: Directory not found")
    
    # Check splits
    splits_dir = Path("data/splits")
    required_splits = ['train.txt', 'val.txt', 'test.txt']
    if splits_dir.exists():
        existing_splits = [f for f in required_splits if (splits_dir / f).exists()]
        if len(existing_splits) == 3:
            checks['splits'] = True
            print(f"   ✅ Splits: All split files found")
        else:
            print(f"   ❌ Splits: Missing {set(required_splits) - set(existing_splits)}")
    else:
        print("   ❌ Splits: Directory not found")
    
    # Check config
    config_file = Path("configs/ppe_config.yaml")
    if config_file.exists():
        checks['config'] = True
        print("   ✅ Config: ppe_config.yaml found")
    else:
        print("   ❌ Config: ppe_config.yaml not found")
    
    # Check Python environment
    try:
        import torch
        import torchvision
        checks['python_env'] = True
        print(f"   ✅ Python Environment: PyTorch {torch.__version__} ready")
    except ImportError as e:
        print(f"   ❌ Python Environment: Missing dependencies - {e}")
    
    return checks

def validate_annotations():
    """Validate annotation files and check for issues"""
    
    print("\n🔍 Validating annotations...")
    
    annotations_dir = Path("data/annotations")
    images_dir = Path("data/images")
    
    xml_files = list(annotations_dir.glob("*.xml"))
    issues = []
    
    # Check annotation-image pairs
    for xml_file in xml_files[:5]:  # Check first 5 for demo
        try:
            import xml.etree.ElementTree as ET
            tree = ET.parse(xml_file)
            root = tree.getroot()
            
            # Get image filename from XML
            filename_elem = root.find('filename')
            if filename_elem is not None:
                image_name = filename_elem.text
                image_path = images_dir / image_name
                
                if not image_path.exists():
                    issues.append(f"Missing image for {xml_file.name}: {image_name}")
            
            # Check if XML has objects
            objects = root.findall('object')
            if len(objects) == 0:
                issues.append(f"No objects in {xml_file.name}")
                
        except Exception as e:
            issues.append(f"Error reading {xml_file.name}: {e}")
    
    if issues:
        print("   ⚠️ Issues found:")
        for issue in issues[:3]:  # Show first 3 issues
            print(f"      • {issue}")
        if len(issues) > 3:
            print(f"      • ... and {len(issues) - 3} more issues")
    else:
        print("   ✅ Sample annotations look good")
    
    return len(issues) == 0

def setup_training_environment():
    """Setup directories and environment for training"""
    
    print("\n📁 Setting up training environment...")
    
    # Create necessary directories
    directories = [
        "models",
        "logs", 
        "results",
        "checkpoints"
    ]
    
    for dir_name in directories:
        Path(dir_name).mkdir(exist_ok=True)
        print(f"   ✅ Created {dir_name}/ directory")
    
    # Create training script if needed
    training_script = Path("scripts/train.py")
    if training_script.exists():
        print("   ✅ Training script exists")
    else:
        print("   ❌ Training script missing")
    
    return True

def create_training_command():
    """Generate the training command with optimal parameters"""
    
    print("\n🎯 Generating training command...")
    
    # Load config to get parameters
    config_file = Path("configs/ppe_config.yaml")
    if config_file.exists():
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
    else:
        print("   ❌ Config file not found")
        return None
    
    # Build command
    cmd_parts = [
        "python scripts/train.py",
        "--data_dir data",
        f"--num_classes {config['dataset']['num_classes']}",
        f"--batch_size {config['training']['batch_size']}",
        f"--epochs {config['training']['epochs']}",
        f"--lr {config['training']['learning_rate']}",
        "--save_dir models",
        "--log_dir logs",
        "--device auto"
    ]
    
    command = " ".join(cmd_parts)
    
    print("   📋 Training command:")
    print(f"   {command}")
    
    return command

def run_training_dry_run():
    """Test training setup without actually training"""
    
    print("\n🧪 Running training dry run...")
    
    try:
        # Test imports
        sys.path.append('.')
        from src.dataset.ppe_dataset import PPEDataset
        from src.models.ssd import SSD300
        
        print("   ✅ Model imports successful")
        
        # Test dataset loading
        try:
            dataset = PPEDataset(
                data_dir="data",
                split="train",
                img_size=300,
                transform=None
            )
            print(f"   ✅ Dataset loading successful ({len(dataset)} samples)")
            
            # Test single sample
            if len(dataset) > 0:
                sample = dataset[0]
                print(f"   ✅ Sample loading successful (image shape: {sample[0].shape})")
            
        except Exception as e:
            print(f"   ❌ Dataset loading failed: {e}")
            return False
        
        # Test model creation
        try:
            model = SSD300(num_classes=13)
            print("   ✅ Model creation successful")
        except Exception as e:
            print(f"   ❌ Model creation failed: {e}")
            return False
        
        print("   ✅ Dry run completed successfully!")
        return True
        
    except Exception as e:
        print(f"   ❌ Dry run failed: {e}")
        return False

def start_training():
    """Actually start the training process"""
    
    print("\n🚀 Starting training...")
    
    command = create_training_command()
    if not command:
        return False
    
    print("   Starting training process...")
    print("   (This will take several hours)")
    print("   Monitor progress in logs/ directory")
    print("   Press Ctrl+C to stop training")
    
    try:
        # Run training command
        process = subprocess.Popen(
            command.split(),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            cwd='.'
        )
        
        # Stream output
        for line in process.stdout:
            print(f"   {line.strip()}")
        
        process.wait()
        
        if process.returncode == 0:
            print("   ✅ Training completed successfully!")
            return True
        else:
            print(f"   ❌ Training failed with exit code {process.returncode}")
            return False
            
    except KeyboardInterrupt:
        print("   ⏹️ Training interrupted by user")
        process.terminate()
        return False
    except Exception as e:
        print(f"   ❌ Training failed: {e}")
        return False

def main():
    """Main training workflow"""
    
    print("🚀 PPE Model Training Workflow")
    print("=" * 50)
    
    # Step 1: Check prerequisites
    checks = check_prerequisites()
    
    if not all(checks.values()):
        print("\n❌ Prerequisites not met. Please fix the issues above.")
        print("\n💡 Next steps:")
        
        if not checks['annotations']:
            print("   1. Complete annotation using Label Studio")
            print("      Run: start_label_studio.bat")
        
        if not checks['images']:
            print("   2. Ensure images are in data/images/")
        
        if not checks['splits']:
            print("   3. Run: python scripts/convert_osha_dataset.py")
        
        if not checks['python_env']:
            print("   4. Install dependencies: pip install -r requirements.txt")
        
        return
    
    print("\n✅ All prerequisites met!")
    
    # Step 2: Validate annotations
    if not validate_annotations():
        print("\n⚠️ Annotation issues detected. Consider reviewing your annotations.")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            return
    
    # Step 3: Setup environment
    setup_training_environment()
    
    # Step 4: Dry run
    if not run_training_dry_run():
        print("\n❌ Dry run failed. Please fix the issues above.")
        return
    
    # Step 5: Confirm training
    print("\n" + "=" * 50)
    print("🎯 READY TO START TRAINING")
    print("=" * 50)
    
    command = create_training_command()
    print(f"\nTraining will run with:")
    print(f"   • Dataset: {Path('data').absolute()}")
    print(f"   • Classes: 13 PPE classes")
    print(f"   • Annotations: ~130 images")
    print(f"   • Estimated time: 2-4 hours (GPU) / 8-12 hours (CPU)")
    print(f"   • Output: models/ and logs/ directories")
    
    response = input("\nStart training now? (y/n): ")
    
    if response.lower() == 'y':
        success = start_training()
        
        if success:
            print("\n🎉 Training completed!")
            print("\n📋 Next steps:")
            print("   1. Check model performance in logs/")
            print("   2. Test model: python scripts/inference.py")
            print("   3. Try hybrid descriptions: python scripts/demo_hybrid_ppe.py")
        
    else:
        print("\n💡 To start training later, run:")
        print(f"   {command}")

if __name__ == "__main__":
    main()