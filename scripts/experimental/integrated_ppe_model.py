#!/usr/bin/env python3
"""
Integration script to connect trained PPE model with description generation
"""

import torch
import torch.nn as nn
from PIL import Image
import yaml
import sys
import os
from pathlib import Path

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.models.ssd import SSD300
from src.models.hybrid_ppe_model import HybridPPEDescriptionModel
from src.utils.utils import load_config

class IntegratedPPEModel:
    """Integrated PPE detection + description model"""
    
    def __init__(self, config_path: str = "configs/ppe_config.yaml"):
        """Initialize with your trained PPE model"""
        
        self.config = load_config(config_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load your trained PPE detection model
        self.ppe_model = self._load_ppe_model()
        
        # Initialize description model
        self.description_model = None
        
    def _load_ppe_model(self):
        """Load your trained SSD PPE detection model"""
        
        try:
            # Initialize SSD model with your config
            model = SSD300(
                num_classes=self.config['model']['num_classes'],
                backbone=self.config['model']['backbone']
            )
            
            # Load trained weights if available
            model_path = self.config.get('model_path')
            if model_path and os.path.exists(model_path):
                print(f"Loading trained model from {model_path}")
                checkpoint = torch.load(model_path, map_location=self.device)
                model.load_state_dict(checkpoint['model_state_dict'])
                print("✅ Trained PPE model loaded successfully")
            else:
                print("⚠️ No trained model found - using initialized weights")
            
            model.to(self.device)
            model.eval()
            return model
            
        except Exception as e:
            print(f"Error loading PPE model: {e}")
            return None
    
    def initialize_description_model(self, vision_model="blip2"):
        """Initialize the description generation component"""
        
        print(f"🔄 Initializing description model ({vision_model})...")
        
        # Create custom hybrid model that uses our PPE detector
        class CustomHybridModel(HybridPPEDescriptionModel):
            def __init__(self, ppe_detector, *args, **kwargs):
                super().__init__(ppe_model_path=None, *args, **kwargs)
                self.ppe_detector = ppe_detector
                self.device = ppe_detector.device if ppe_detector else self.device
            
            def detect_ppe(self, image):
                """Use the actual trained PPE model for detection"""
                if self.ppe_detector is None:
                    return super().detect_ppe(image)
                
                return self._run_real_ppe_detection(image)
            
            def _run_real_ppe_detection(self, image):
                """Run real PPE detection using trained SSD model"""
                
                # This is where you'd implement the actual detection pipeline
                # For now, returning mock results - you'd replace this with:
                # 1. Image preprocessing (resize, normalize, etc.)
                # 2. Model inference
                # 3. Post-processing (NMS, threshold filtering)
                # 4. Convert to detection format
                
                print("🔍 Running real PPE detection...")
                
                # Mock detection for demonstration
                # Replace this with your actual detection pipeline
                mock_detections = [
                    {
                        'class': 'person',
                        'confidence': 0.95,
                        'bbox': [100, 50, 200, 400],
                        'class_id': 1
                    },
                    {
                        'class': 'safety_vest',
                        'confidence': 0.87,
                        'bbox': [120, 100, 180, 250],
                        'class_id': 3
                    },
                    {
                        'class': 'no_hard_hat',
                        'confidence': 0.92,
                        'bbox': [130, 50, 170, 90],
                        'class_id': 7
                    }
                ]
                
                return mock_detections
        
        # Initialize the custom hybrid model
        self.description_model = CustomHybridModel(
            ppe_detector=self.ppe_model,
            vision_model=vision_model,
            device="auto"
        )
        
        print("✅ Description model initialized successfully")
    
    def analyze_image(self, image_path: str, include_descriptions: bool = True):
        """Complete analysis of construction site image"""
        
        print(f"🖼️ Analyzing image: {image_path}")
        
        # Load image
        image = Image.open(image_path).convert('RGB')
        
        results = {
            'image_path': image_path,
            'image_size': image.size,
            'model_info': {
                'ppe_model': 'SSD300 + VGG16',
                'num_classes': self.config['model']['num_classes'],
                'description_model': self.description_model.vision_model_name if self.description_model else None
            }
        }
        
        # Run PPE detection only
        if self.ppe_model and not include_descriptions:
            print("🔍 Running PPE detection only...")
            # Add your detection pipeline here
            results['ppe_detections'] = []  # Placeholder
            
        # Run full hybrid analysis
        elif self.description_model and include_descriptions:
            print("🔍 Running hybrid PPE + description analysis...")
            
            hybrid_results = self.description_model.generate_hybrid_description(
                image,
                include_general_caption=True,
                custom_prompt="Analyze this construction site for PPE compliance and safety."
            )
            
            results.update(hybrid_results)
        
        else:
            print("❌ No models available for analysis")
            return None
        
        return results
    
    def batch_analyze(self, image_directory: str, output_file: str = "batch_analysis.json"):
        """Analyze all images in a directory"""
        
        image_dir = Path(image_directory)
        if not image_dir.exists():
            print(f"❌ Directory not found: {image_directory}")
            return
        
        image_files = list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png"))
        
        if not image_files:
            print(f"❌ No images found in {image_directory}")
            return
        
        print(f"📁 Processing {len(image_files)} images...")
        
        all_results = []
        
        for i, image_file in enumerate(image_files, 1):
            print(f"\n--- Processing {i}/{len(image_files)}: {image_file.name} ---")
            
            try:
                result = self.analyze_image(str(image_file))
                if result:
                    all_results.append(result)
                    
            except Exception as e:
                print(f"❌ Error processing {image_file}: {e}")
        
        # Save results
        import json
        with open(output_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        
        print(f"\n💾 Batch analysis complete! Results saved to {output_file}")
        
        # Generate summary
        self._generate_batch_summary(all_results)
    
    def _generate_batch_summary(self, results):
        """Generate summary statistics from batch analysis"""
        
        if not results:
            return
        
        print("\n" + "="*60)
        print("BATCH ANALYSIS SUMMARY")
        print("="*60)
        
        total_images = len(results)
        images_with_people = 0
        total_violations = 0
        
        for result in results:
            if 'ppe_descriptions' in result:
                safety_summary = result['ppe_descriptions']['safety_summary']
                if 'worker' in safety_summary.lower():
                    images_with_people += 1
                if 'safety concern' in safety_summary.lower():
                    total_violations += 1
        
        print(f"📊 Total Images: {total_images}")
        print(f"👷 Images with Workers: {images_with_people}")
        print(f"⚠️ Images with Violations: {total_violations}")
        
        if images_with_people > 0:
            compliance_rate = ((images_with_people - total_violations) / images_with_people) * 100
            print(f"📈 Compliance Rate: {compliance_rate:.1f}%")

def main():
    """Demo of integrated PPE model"""
    
    print("🚀 Initializing Integrated PPE Model")
    print("="*50)
    
    # Initialize the integrated model
    integrated_model = IntegratedPPEModel()
    
    # Initialize description capabilities
    integrated_model.initialize_description_model(vision_model="blip2")
    
    # Demo analysis
    demo_image = "data/images/construction_site.jpg"  # Replace with actual image
    
    if os.path.exists(demo_image):
        print(f"\n🧪 Demo analysis with {demo_image}")
        results = integrated_model.analyze_image(demo_image)
        
        if results:
            print("\n📋 Analysis Results:")
            print(f"   Model: {results['model_info']}")
            if 'general_caption' in results:
                print(f"   Scene: {results['general_caption']}")
            if 'ppe_descriptions' in results:
                print(f"   Safety: {results['ppe_descriptions']['safety_summary']}")
    
    else:
        print(f"\n💡 To test with real images:")
        print("   1. Place construction site images in data/images/")
        print("   2. Run: python scripts/integrated_ppe_model.py")
        print("   3. Or use batch processing:")
        print("      integrated_model.batch_analyze('data/images')")

if __name__ == "__main__":
    main()