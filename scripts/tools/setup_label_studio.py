#!/usr/bin/env python3
"""
Setup Label Studio for PPE Detection Project
Creates Label Studio project configuration and imports the dataset
"""

import os
import json
import subprocess
import sys
from pathlib import Path

def install_label_studio():
    """Install Label Studio via pip"""
    print("Installing Label Studio...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "label-studio"])
        print("✅ Label Studio installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing Label Studio: {e}")
        return False

def create_label_studio_config():
    """Create Label Studio configuration for PPE detection"""
    
    # Label Studio XML template for object detection
    label_config = """
<View>
  <Image name="image" value="$image" zoom="true" zoomControl="true" rotateControl="true"/>
  <RectangleLabels name="label" toName="image" strokeWidth="2" smart="true">
    <Label value="person" background="red"/>
    <Label value="hard_hat" background="blue"/>
    <Label value="safety_vest" background="yellow"/>
    <Label value="safety_gloves" background="green"/>
    <Label value="safety_boots" background="purple"/>
    <Label value="eye_protection" background="orange"/>
    <Label value="no_hard_hat" background="darkred"/>
    <Label value="no_safety_vest" background="darkgoldenrod"/>
  </RectangleLabels>
</View>
"""
    
    return label_config.strip()

def create_label_studio_tasks():
    """Create tasks JSON file for Label Studio import"""
    
    images_dir = Path("data/images")
    annotation_guides_dir = Path("data/annotation_guides")
    
    tasks = []
    
    # Get all images
    image_files = list(images_dir.glob("image*.png")) + list(images_dir.glob("image*.jpg")) + list(images_dir.glob("image*.jpeg"))
    for img_file in image_files:
        image_id = img_file.stem
        
        # Read annotation guide if exists
        guide_file = annotation_guides_dir / f"{image_id}_guide.txt"
        guide_text = ""
        if guide_file.exists():
            with open(guide_file, 'r', encoding='utf-8') as f:
                guide_text = f.read()
        
        # Create task
        task = {
            "data": {
                "image": f"/data/local-files/?d=images/{img_file.name}",
                "image_id": image_id,
                "annotation_guide": guide_text
            }
        }
        tasks.append(task)
    
    # Save tasks to JSON file
    with open("label_studio_tasks.json", "w") as f:
        json.dump(tasks, f, indent=2)
    
    print(f"✅ Created {len(tasks)} tasks for Label Studio")
    return len(tasks)

def create_setup_script():
    """Create a setup script for Label Studio"""
    
    setup_script = """@echo off
echo Setting up Label Studio for PPE Detection...
echo.

REM Start Label Studio
echo Starting Label Studio...
echo Open your browser to: http://localhost:8080
echo.
echo Instructions:
echo 1. Create a new project called "PPE Detection"
echo 2. Import the label_studio_tasks.json file
echo 3. Use the provided label configuration
echo 4. Start annotating!
echo.

label-studio start
"""
    
    with open("start_label_studio.bat", "w") as f:
        f.write(setup_script)
    
    print("✅ Created start_label_studio.bat")

def create_instructions():
    """Create detailed setup instructions"""
    
    instructions = """
# Label Studio Setup Instructions for PPE Detection

## 1. Start Label Studio
Run the setup script:
```bash
# Windows
start_label_studio.bat

# Or manually:
label-studio start
```

## 2. Create Project
1. Open browser to http://localhost:8080
2. Sign up/Login (use any email)
3. Click "Create Project"
4. Name: "PPE Detection"
5. Description: "OSHA PPE Compliance Detection"

## 3. Configure Labels
In Project Settings > Labeling Interface, paste this configuration:

```xml
<View>
  <Image name="image" value="$image" zoom="true" zoomControl="true" rotateControl="true"/>
  <RectangleLabels name="label" toName="image" strokeWidth="2" smart="true">
    <Label value="person" background="red"/>
    <Label value="hard_hat" background="blue"/>
    <Label value="safety_vest" background="yellow"/>
    <Label value="safety_gloves" background="green"/>
    <Label value="safety_boots" background="purple"/>
    <Label value="eye_protection" background="orange"/>
    <Label value="no_hard_hat" background="darkred"/>
    <Label value="no_safety_vest" background="darkgoldenrod"/>
  </RectangleLabels>
</View>
```

## 4. Import Data
1. Go to Project > Import
2. Upload `label_studio_tasks.json`
3. Click "Import"

## 5. Setup Storage
1. Go to Project Settings > Cloud Storage
2. Add Local Storage:
   - Storage Type: Local files
   - Absolute local path: `{absolute_path_to_data}`
   - URL path prefix: `/data/local-files/`

## 6. Start Annotating!
1. Click on any task to start
2. Use the annotation guide shown in task details
3. Draw bounding boxes around PPE items
4. Use correct class labels
5. Save and move to next

## 7. Export Annotations
When done:
1. Go to Project > Export
2. Choose "Pascal VOC XML" format
3. Download and replace the placeholder annotations

## Annotation Tips:
- Use the annotation guide for each image
- Draw tight bounding boxes
- Pay attention to violation classes (no_hard_hat, no_safety_vest)
- Use zoom for precision
- Keyboard shortcuts: Space (next), Ctrl+Z (undo)

## Progress Tracking:
- Dashboard shows completion percentage
- Statistics show annotation quality
- Can filter completed/pending tasks
"""
    
    with open("LABEL_STUDIO_SETUP.md", "w") as f:
        f.write(instructions.format(
            absolute_path_to_data=os.path.abspath("data")
        ))
    
    print("✅ Created LABEL_STUDIO_SETUP.md")

def main():
    """Main setup function"""
    print("🚀 Setting up Label Studio for PPE Detection...")
    print()
    
    # Install Label Studio
    if not install_label_studio():
        return
    
    # Create configuration
    print("📝 Creating Label Studio configuration...")
    config = create_label_studio_config()
    
    # Create tasks
    print("📋 Creating annotation tasks...")
    num_tasks = create_label_studio_tasks()
    
    # Create setup script
    print("🔧 Creating setup scripts...")
    create_setup_script()
    create_instructions()
    
    print()
    print("🎉 Setup Complete!")
    print(f"📊 Ready to annotate {num_tasks} images")
    print()
    print("Next steps:")
    print("1. Run: start_label_studio.bat")
    print("2. Follow instructions in LABEL_STUDIO_SETUP.md")
    print("3. Start annotating your PPE dataset!")
    print()
    print("💡 Label Studio will be much faster than manual XML editing!")

if __name__ == "__main__":
    main()