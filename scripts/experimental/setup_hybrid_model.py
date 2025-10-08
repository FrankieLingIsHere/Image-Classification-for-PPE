#!/usr/bin/env python3
"""
Setup script for Hybrid PPE Description Model
Installs dependencies and tests the model
"""

import subprocess
import sys
import os
from pathlib import Path

def install_dependencies():
    """Install required packages for the hybrid model"""
    
    print("📦 Installing hybrid model dependencies...")
    
    packages = [
        "transformers>=4.30.0",
        "accelerate>=0.20.0", 
        "torchaudio>=2.0.0",
        "sentencepiece>=0.1.99",
        "protobuf>=3.20.0"
    ]
    
    for package in packages:
        print(f"   Installing {package}...")
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", package
            ])
        except subprocess.CalledProcessError as e:
            print(f"   ❌ Failed to install {package}: {e}")
            return False
    
    print("✅ All dependencies installed successfully!")
    return True

def test_model_imports():
    """Test if the hybrid model can be imported"""
    
    print("🧪 Testing model imports...")
    
    try:
        from transformers import BlipProcessor, BlipForConditionalGeneration
        print("   ✅ BLIP model imports working")
        
        from PIL import Image
        print("   ✅ PIL imports working")
        
        import torch
        print(f"   ✅ PyTorch working (version: {torch.__version__})")
        
        # Test CUDA availability
        if torch.cuda.is_available():
            print(f"   🚀 CUDA available ({torch.cuda.get_device_name(0)})")
        else:
            print("   💻 Running on CPU")
        
        return True
        
    except ImportError as e:
        print(f"   ❌ Import error: {e}")
        return False

def create_sample_test():
    """Create a simple test to verify the model works"""
    
    print("🧪 Running quick model test...")
    
    try:
        # Import the hybrid model
        sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
        from src.models.hybrid_ppe_model import HybridPPEDescriptionModel
        
        # Create a simple test image
        from PIL import Image
        test_image = Image.new('RGB', (640, 480), color='lightblue')
        
        # Initialize model with lightweight settings
        print("   Initializing model...")
        model = HybridPPEDescriptionModel(
            ppe_model_path=None,  # Use mock detection
            vision_model="blip2",  # Smaller model
            device="auto"
        )
        
        print("   Generating test description...")
        results = model.generate_hybrid_description(
            test_image,
            include_general_caption=True
        )
        
        print("   ✅ Model test successful!")
        print(f"   📝 Sample output: {results['general_caption'][:100]}...")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Model test failed: {e}")
        return False

def setup_demo_environment():
    """Set up directory structure for demos"""
    
    print("📁 Setting up demo environment...")
    
    # Create demo directories
    demo_dirs = [
        "demo_images",
        "results",
        "exports"
    ]
    
    for dir_name in demo_dirs:
        Path(dir_name).mkdir(exist_ok=True)
        print(f"   ✅ Created {dir_name}/ directory")
    
    # Create a sample config file
    sample_config = """# Hybrid PPE Model Configuration

# Model Settings
vision_model: "blip2"  # or "llava" for more advanced descriptions
device: "auto"  # "cuda", "cpu", or "auto"

# PPE Classes
ppe_classes:
  - background
  - person
  - hard_hat
  - safety_vest
  - safety_gloves
  - safety_boots
  - eye_protection
  - no_hard_hat
  - no_safety_vest
  - no_safety_gloves
  - no_safety_boots
  - no_eye_protection

# Description Settings
include_general_caption: true
custom_prompt: "Describe this construction site focusing on worker safety and PPE compliance."

# Output Settings
save_results: true
output_format: "json"  # "json" or "text"
"""
    
    with open("hybrid_model_config.yaml", "w") as f:
        f.write(sample_config)
    
    print("   ✅ Created hybrid_model_config.yaml")

def main():
    """Main setup function"""
    
    print("🚀 Setting up Hybrid PPE Description Model")
    print("=" * 50)
    
    # Step 1: Install dependencies
    if not install_dependencies():
        print("❌ Setup failed during dependency installation")
        return
    
    # Step 2: Test imports
    if not test_model_imports():
        print("❌ Setup failed during import testing")
        return
    
    # Step 3: Test model functionality  
    if not create_sample_test():
        print("❌ Setup failed during model testing")
        return
    
    # Step 4: Setup demo environment
    setup_demo_environment()
    
    print("\n" + "=" * 50)
    print("🎉 SETUP COMPLETE!")
    print("=" * 50)
    
    print("\n📖 Next Steps:")
    print("1. Run the demo script:")
    print("   python scripts/demo_hybrid_ppe.py --image path/to/construction_image.jpg")
    print("\n2. Process multiple images:")
    print("   python scripts/demo_hybrid_ppe.py --batch data/images")
    print("\n3. Use advanced model:")
    print("   python scripts/demo_hybrid_ppe.py --model llava --image image.jpg")
    print("\n4. With your trained PPE model:")
    print("   python scripts/demo_hybrid_ppe.py --ppe-model path/to/trained_model.pth --image image.jpg")
    
    print("\n💡 Tips:")
    print("- First run will download BLIP-2 model (~500MB)")
    print("- For faster inference, use CUDA if available")
    print("- Check hybrid_model_config.yaml for customization options")
    print("- Results are saved as JSON files for further analysis")

if __name__ == "__main__":
    main()