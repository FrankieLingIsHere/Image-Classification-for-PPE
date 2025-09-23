#!/usr/bin/env python3
"""
Simple PPE Model Training Guide
Complete step-by-step training process
"""

import os
import sys
from pathlib import Path
import subprocess

def check_training_readiness():
    """Check if ready to start training"""
    
    print("🔍 Checking training readiness...")
    
    # Check key components
    checks = []
    
    # 1. Annotations
    annotations_dir = Path("data/annotations")
    xml_count = len(list(annotations_dir.glob("*.xml"))) if annotations_dir.exists() else 0
    checks.append(("Annotations", xml_count > 100, f"{xml_count} XML files"))
    
    # 2. Images  
    images_dir = Path("data/images")
    img_count = len(list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))) if images_dir.exists() else 0
    checks.append(("Images", img_count > 100, f"{img_count} image files"))
    
    # 3. Splits
    splits_dir = Path("data/splits")
    split_files = ['train.txt', 'val.txt', 'test.txt']
    splits_exist = all((splits_dir / f).exists() for f in split_files) if splits_dir.exists() else False
    checks.append(("Data splits", splits_exist, "train/val/test splits"))
    
    # 4. Config
    config_exists = Path("configs/ppe_config.yaml").exists()
    checks.append(("Configuration", config_exists, "ppe_config.yaml"))
    
    # 5. Training script
    train_script_exists = Path("scripts/train.py").exists()
    checks.append(("Training script", train_script_exists, "train.py"))
    
    # Print results
    print()
    all_ready = True
    for name, status, detail in checks:
        status_icon = "✅" if status else "❌"
        print(f"   {status_icon} {name}: {detail}")
        if not status:
            all_ready = False
    
    return all_ready, checks

def show_training_steps():
    """Show the complete training process"""
    
    print("\n📋 COMPLETE TRAINING PROCESS")
    print("=" * 50)
    
    steps = [
        ("1️⃣ Annotation Completion", [
            "Open Label Studio: start_label_studio.bat",
            "Complete annotation of all 129 images", 
            "Export annotations as Pascal VOC XML",
            "Run export fixer: python scripts/fix_label_studio_export.py"
        ]),
        
        ("2️⃣ Environment Setup", [
            "Activate virtual environment: .venv\\Scripts\\activate",
            "Install dependencies: pip install -r requirements.txt",
            "Configure Python environment: python scripts/configure_python_environment.py"
        ]),
        
        ("3️⃣ Data Preparation", [
            "Verify data structure: data/images/, data/annotations/, data/splits/",
            "Check annotation quality: python scripts/validate_annotations.py",
            "Review class distribution and balance"
        ]),
        
        ("4️⃣ Training Execution", [
            "Start training: python scripts/train.py --config configs/ppe_config.yaml",
            "Monitor progress: tensorboard --logdir logs/",
            "Wait for completion: ~2-4 hours (GPU) or 8-12 hours (CPU)"
        ]),
        
        ("5️⃣ Model Evaluation", [
            "Test trained model: python scripts/inference.py",
            "Evaluate performance: python scripts/evaluate.py", 
            "Try hybrid descriptions: python scripts/demo_hybrid_ppe.py"
        ])
    ]
    
    for step_name, substeps in steps:
        print(f"\n{step_name}")
        for substep in substeps:
            print(f"   • {substep}")

def create_training_commands():
    """Generate ready-to-use training commands"""
    
    print("\n🚀 TRAINING COMMANDS")
    print("=" * 50)
    
    commands = {
        "Basic Training": "python scripts/train.py --data_dir data --num_classes 13 --epochs 100 --batch_size 8",
        
        "GPU Training (Recommended)": "python scripts/train.py --data_dir data --num_classes 13 --epochs 100 --batch_size 16 --device cuda",
        
        "Fast Training (Test)": "python scripts/train.py --data_dir data --num_classes 13 --epochs 10 --batch_size 4",
        
        "Resume Training": "python scripts/train.py --resume models/checkpoint_latest.pth",
        
        "With Tensorboard": "python scripts/train.py --data_dir data --log_dir logs && tensorboard --logdir logs"
    }
    
    for name, command in commands.items():
        print(f"\n📋 {name}:")
        print(f"   {command}")

def check_current_status():
    """Check what's been completed so far"""
    
    print("\n📊 CURRENT STATUS")
    print("=" * 50)
    
    # Check annotations
    annotations_count = len(list(Path("data/annotations").glob("*.xml"))) if Path("data/annotations").exists() else 0
    print(f"   Annotations: {annotations_count}/129 completed")
    
    # Check if Label Studio was used
    if Path("start_label_studio.bat").exists():
        print("   ✅ Label Studio setup completed")
    
    # Check training history  
    models_dir = Path("models")
    if models_dir.exists():
        checkpoints = list(models_dir.glob("*.pth"))
        if checkpoints:
            print(f"   Previous training: {len(checkpoints)} checkpoints found")
        else:
            print("   No previous training found")
    
    # Check logs
    logs_dir = Path("logs")
    if logs_dir.exists() and list(logs_dir.iterdir()):
        print("   ✅ Training logs directory exists")
    else:
        print("   No training logs found")

def main():
    """Main function"""
    
    print("🚀 PPE Model Training Guide")
    print("=" * 50)
    
    # Check current status
    check_current_status()
    
    # Check readiness
    ready, checks = check_training_readiness()
    
    if ready:
        print("\n🎉 READY TO TRAIN!")
        print("\n🚀 Quick Start:")
        print("   1. Activate environment: .venv\\Scripts\\activate")
        print("   2. Start training: python scripts/train.py --data_dir data --num_classes 13")
        print("   3. Monitor: tensorboard --logdir logs")
        
        create_training_commands()
        
    else:
        print("\n⚠️ NOT READY FOR TRAINING")
        print("\nMissing components:")
        for name, status, detail in checks:
            if not status:
                print(f"   ❌ {name}")
        
        show_training_steps()
        
        print("\n💡 IMMEDIATE NEXT STEPS:")
        
        # Check what's most urgently needed
        annotations_count = len(list(Path("data/annotations").glob("*.xml"))) if Path("data/annotations").exists() else 0
        
        if annotations_count < 50:
            print("   🎯 PRIORITY: Complete annotations in Label Studio")
            print("   📋 Commands:")
            print("      start_label_studio.bat")
            print("      # Complete annotations, then export and run fixer")
            
        elif annotations_count < 129:
            print("   🎯 PRIORITY: Finish remaining annotations")
            print(f"      Progress: {annotations_count}/129 ({annotations_count/129*100:.1f}%)")
            
        else:
            print("   🎯 PRIORITY: Verify data and start training")
            print("   📋 Commands:")
            print("      .venv\\Scripts\\activate")
            print("      python scripts/train.py --data_dir data --num_classes 13")

if __name__ == "__main__":
    main()